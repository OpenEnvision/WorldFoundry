"""LTX-2 Gemma-3 :class:`~...contracts.ConditionEncoder`.

Runs Gemma-3, then the official embeddings connector / processor so
video and audio caption tokens match the DiT cross-attention width.
``encode(DiffusionRequest, ...)`` returns
:class:`~...contracts.Conditioning` with video/audio context tensors.
19B graphs project captions inside the transformer; 22B projects here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import lru_cache, partial

import torch
from transformers import AutoModel, Gemma3Config, Gemma3ForConditionalGeneration

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import (
    CheckpointSpec,
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
    safetensors_json_metadata,
)
from ....optimizations import OffloadMode, OffloadPolicy, QuantizationPolicy, RuntimePolicy
from .embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from .embeddings_processor import EmbeddingsProcessor
from .feature_extractor import FeatureExtractorV1, FeatureExtractorV2
from .tokenizer import LTXVGemmaTokenizer


class LTXGemmaModule(torch.nn.Module):
    """Gemma model constructed from its local Hugging Face configuration."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.model = Gemma3ForConditionalGeneration(Gemma3Config.from_dict(dict(checkpoint_config)))


class LTXEmbeddingProcessorModule(torch.nn.Module):
    """Checkpoint-compatible LTX feature projection and connector stack."""

    def __init__(
        self,
        checkpoint_config: Mapping[str, object],
        *,
        gemma_config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        config = dict(checkpoint_config)
        transformer = dict(config.get("transformer", {}))
        hidden_size = int(transformer.get("caption_channels", 3840))
        hidden_layers = int(transformer.get("caption_hidden_layers", 48)) + 1
        if gemma_config is not None:
            text_config = gemma_config["text_config"]
            hidden_size = int(text_config["hidden_size"])
            hidden_layers = int(text_config["num_hidden_layers"]) + 1
        flat_dim = hidden_size * hidden_layers
        if transformer.get("caption_proj_before_connector", False):
            video_dim = int(transformer["num_attention_heads"]) * int(transformer["attention_head_dim"])
            audio_dim = int(transformer["audio_num_attention_heads"]) * int(transformer["audio_attention_head_dim"])
            feature_extractor = FeatureExtractorV2(
                video_aggregate_embed=torch.nn.Linear(flat_dim, video_dim, bias=True),
                audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_dim, bias=True),
                embedding_dim=hidden_size,
            )
        else:
            feature_extractor = FeatureExtractorV1(
                aggregate_embed=torch.nn.Linear(flat_dim, hidden_size, bias=False),
                is_av=True,
            )
        self.processor = EmbeddingsProcessor(
            feature_extractor=feature_extractor,
            video_connector=(
                Embeddings1DConnectorConfigurator.from_config(config)
                if gemma_config is None
                else Embeddings1DConnectorConfigurator.from_metadata({"config": config})
            ),
            audio_connector=(
                AudioEmbeddings1DConnectorConfigurator.from_config(config)
                if gemma_config is None
                else AudioEmbeddings1DConnectorConfigurator.from_metadata({"config": config})
            ),
        )


class LTXVideoEmbeddingProcessorModule(torch.nn.Module):
    """Video-only feature projection and connector used by Alaya/LTX-Video."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        config = dict(checkpoint_config)
        transformer = dict(config.get("transformer", {}))
        hidden_size = int(transformer.get("caption_channels", 3840))
        hidden_layers = int(transformer.get("caption_hidden_layers", 48)) + 1
        flat_dim = hidden_size * hidden_layers
        if transformer.get("caption_proj_before_connector", False):
            video_dim = int(transformer["num_attention_heads"]) * int(transformer["attention_head_dim"])
            feature_extractor = FeatureExtractorV2(
                video_aggregate_embed=torch.nn.Linear(flat_dim, video_dim, bias=True),
                audio_aggregate_embed=None,
                embedding_dim=hidden_size,
            )
        else:
            feature_extractor = FeatureExtractorV1(
                aggregate_embed=torch.nn.Linear(flat_dim, hidden_size, bias=False),
                is_av=False,
            )
        self.processor = EmbeddingsProcessor(
            feature_extractor=feature_extractor,
            video_connector=Embeddings1DConnectorConfigurator.from_config(config),
            audio_connector=None,
        )


def _gemma_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    path = checkpoint.root / "config.json"
    if not path.is_file():
        path = checkpoint.root / "text_encoder" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"LTX Gemma config does not exist: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("LTX Gemma config.json must contain an object")
    return {"checkpoint_config": value}


@lru_cache(maxsize=1)
def _gemma_vision_has_wrapper() -> bool:
    """Inspect the installed Transformers layout without allocating weights."""

    config = Gemma3Config().vision_config
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.image_size = 8
    config.patch_size = 4
    with torch.device("meta"):
        vision = AutoModel.from_config(config)
    return hasattr(vision, "vision_model")


def convert_ltx_gemma_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    """Map official Gemma weights to the installed Transformers module layout."""
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        keys = convert_ltx_gemma_state_dict({key: key for key in state_dict})
        return TransformedStateDict(state_dict, keys)

    converted: dict[str, object] = {}
    for key in state_dict:
        if key.startswith("language_model.model."):
            value = state_dict[key]
            destination = f"model.model.language_model.{key.removeprefix('language_model.model.')}"
            converted[destination] = value
            if key == "language_model.model.embed_tokens.weight":
                converted["model.lm_head.weight"] = value
        elif key.startswith("vision_tower."):
            # Transformers versions differ in whether SigLIP retains this
            # wrapper. Normalize the checkpoint to the actual module layout.
            suffix = key.removeprefix("vision_tower.").removeprefix("vision_model.")
            if _gemma_vision_has_wrapper():
                suffix = f"vision_model.{suffix}"
            converted[f"model.model.vision_tower.{suffix}"] = state_dict[key]
        elif key.startswith("multi_modal_projector."):
            converted[f"model.model.{key}"] = state_dict[key]
    return converted


def convert_ltx_embedding_processor_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        keys = convert_ltx_embedding_processor_state_dict({key: key for key in state_dict})
        return TransformedStateDict(state_dict, keys)
    prefixes = {
        "text_embedding_projection.aggregate_embed.": "processor.feature_extractor.aggregate_embed.",
        "text_embedding_projection.video_aggregate_embed.": ("processor.feature_extractor.video_aggregate_embed."),
        "text_embedding_projection.audio_aggregate_embed.": ("processor.feature_extractor.audio_aggregate_embed."),
        "model.diffusion_model.video_embeddings_connector.": "processor.video_connector.",
        "model.diffusion_model.embeddings_connector.": "processor.video_connector.",
        "model.diffusion_model.audio_embeddings_connector.": "processor.audio_connector.",
    }
    converted: dict[str, object] = {}
    for key in state_dict:
        for source, destination in prefixes.items():
            if key.startswith(source):
                converted[f"{destination}{key.removeprefix(source)}"] = state_dict[key]
                break
    return converted


def convert_ltx_video_embedding_processor_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map only video-side text projections and connectors."""

    converted = convert_ltx_embedding_processor_state_dict(state_dict)
    return {
        key: value
        for key, value in converted.items()
        if not key.startswith("processor.audio_") and not key.startswith("processor.feature_extractor.audio_")
    }


def _ltx_text_module_map():
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )

    return {
        Embeddings1DConnector: AutoWrappedNonRecurseModule,
        torch.nn.Linear: AutoWrappedLinear,
        torch.nn.Embedding: AutoWrappedModule,
        torch.nn.LayerNorm: AutoWrappedModule,
        torch.nn.RMSNorm: AutoWrappedModule,
        Gemma3RMSNorm: AutoWrappedModule,
    }


def load_ltx_gemma_module(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> LTXGemmaModule:
    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXGemmaModule,
            config_resolver=_gemma_config,
            state_dict_converter=convert_ltx_gemma_state_dict,
            vram_module_map=_ltx_text_module_map(),
            layer_container="model.model.language_model.layers",
        ),
        checkpoint,
        policy,
    )
    if not isinstance(module, LTXGemmaModule):
        raise TypeError("LTX Gemma construction returned an unexpected module")
    return module


def load_ltx_embedding_processor_module(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
    *,
    video_only: bool = False,
) -> LTXEmbeddingProcessorModule | LTXVideoEmbeddingProcessorModule:
    module_class = LTXVideoEmbeddingProcessorModule if video_only else LTXEmbeddingProcessorModule
    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config_resolver=lambda materialized: {"checkpoint_config": safetensors_json_metadata(materialized)},
            state_dict_converter=(
                convert_ltx_video_embedding_processor_state_dict
                if video_only
                else convert_ltx_embedding_processor_state_dict
            ),
            vram_module_map=_ltx_text_module_map(),
            layer_container="processor.video_connector.transformer_1d_blocks",
        ),
        checkpoint,
        policy,
    )
    if not isinstance(module, (LTXEmbeddingProcessorModule, LTXVideoEmbeddingProcessorModule)):
        raise TypeError("LTX embedding processor construction returned an unexpected module")
    return module


class LTXPromptConditioner:
    """Encode prompts into the distinct LTX video and audio contexts."""

    def __init__(
        self,
        gemma: LTXGemmaModule,
        processor: LTXEmbeddingProcessorModule,
        tokenizer: LTXVGemmaTokenizer,
        *,
        execution_device: str | torch.device,
    ) -> None:
        self.gemma = gemma
        self.processor = processor
        self.tokenizer = tokenizer
        self.execution_device = torch.device(execution_device)

    def _execution_device(self) -> torch.device:
        for tensor in self.gemma.parameters():
            if tensor.device.type != "meta":
                return tensor.device
        # Disk-backed VRAM wrappers intentionally keep every inactive
        # parameter on the meta device.  Inputs must still be created on the
        # policy's computation device so each wrapper can materialize its
        # weight immediately before use.
        return self.execution_device

    @torch.no_grad()
    def _encode_one(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_pairs = self.tokenizer.tokenize_with_weights(prompt)["gemma"]
        device = self._execution_device()
        input_ids = torch.tensor([[int(pair[0]) for pair in token_pairs]], device=device)
        attention_mask = torch.tensor([[int(pair[1]) for pair in token_pairs]], device=device)
        outputs = self.gemma.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        encoded = self.processor.processor.process_hidden_states(
            outputs.hidden_states,
            attention_mask,
            padding_side="left",
        )
        if encoded.audio_encoding is None:
            raise RuntimeError("LTX AV prompt processor returned no audio context")
        return encoded.video_encoding, encoded.audio_encoding, encoded.attention_mask

    def _encode_batch(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = [self._encode_one(prompt) for prompt in prompts]
        return tuple(torch.cat([item[index] for item in encoded], dim=0) for index in range(3))  # type: ignore[return-value]

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return Gemma-processed ``video_context`` / ``audio_context`` tensors."""
        video, audio, mask = self._encode_batch(request.prompts)
        return Conditioning(
            positive={
                "video_context": video.to(device=device, dtype=dtype),
                "audio_context": audio.to(device=device, dtype=dtype),
                "context_mask": mask.to(device=device),
            }
        )


def build_ltx_prompt_conditioner(context: ComponentBuildContext) -> LTXPromptConditioner:
    policy = replace(context.policy, quantization=QuantizationPolicy())
    gemma = load_ltx_gemma_module(
        context.require_checkpoint("gemma"),
        policy,
    )
    processor = load_ltx_embedding_processor_module(
        context.require_checkpoint("weights"),
        policy,
    )
    tokenizer_checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer_root = tokenizer_checkpoint.root / "tokenizer"
    if not tokenizer_root.is_dir():
        tokenizer_root = tokenizer_checkpoint.directory()
    tokenizer = LTXVGemmaTokenizer(
        str(tokenizer_root),
        max_length=int(context.component_options.get("max_length", 1024)),
    )
    if not isinstance(processor, LTXEmbeddingProcessorModule):
        raise TypeError("LTX AV recipe constructed a video-only embedding processor")
    return LTXPromptConditioner(
        gemma,
        processor,
        tokenizer,
        execution_device=context.policy.device,
    )


__all__ = [
    "LTXEmbeddingProcessorModule",
    "LTXGemmaModule",
    "LTXPromptConditioner",
    "LTXVideoEmbeddingProcessorModule",
    "build_ltx_prompt_conditioner",
    "convert_ltx_embedding_processor_state_dict",
    "convert_ltx_gemma_state_dict",
    "convert_ltx_video_embedding_processor_state_dict",
    "load_ltx_embedding_processor_module",
    "load_ltx_gemma_module",
]


class LTXSplitGemmaModule(torch.nn.Module):
    """Construct the canonical Gemma encoder with local assets and derived buffers."""

    def __init__(self, assets: object, execution_device: torch.device) -> None:
        super().__init__()
        from transformers import AutoModelForImageTextToText

        from .gemma import (
            LTXGemmaTextEncoder,
            build_gemma_hf_config,
            build_gemma_processor,
            build_gemma_tokenizer,
            populate_gemma3_buffers,
            populate_gemma4_dense_buffers,
            populate_gemma4_unified_buffers,
        )

        config = build_gemma_hf_config(assets)
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
        tokenizer = build_gemma_tokenizer(assets)
        self.encoder = LTXGemmaTextEncoder(
            model=model,
            tokenizer=tokenizer,
            processor=build_gemma_processor(assets, tokenizer.tokenizer),
            execution_device=execution_device,
        )
        populate = {
            "gemma3": populate_gemma3_buffers,
            "gemma4": populate_gemma4_dense_buffers,
            "gemma4_unified": populate_gemma4_unified_buffers,
        }[config.model_type]
        populate(self.encoder)


def _convert_split_gemma_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    model_type: str,
) -> Mapping[str, torch.Tensor]:
    from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

    from .gemma import convert_gemma4_state_dict

    converted = (
        convert_ltx_gemma_state_dict(state_dict, lazy=True)
        if model_type == "gemma3"
        else convert_gemma4_state_dict(state_dict, model_type=model_type)
    )
    return TransformedStateDict(converted, {f"encoder.{key}": key for key in converted})


def _load_split_gemma(checkpoint: CheckpointSpec, policy: RuntimePolicy) -> tuple[torch.nn.Module, object]:
    from worldfoundry.core.vram import AutoWrappedModule, AutoWrappedNonRecurseModule

    from .gemma import GemmaAssets

    materialized = NativeCheckpointResolver().materialize(checkpoint)
    source = materialized.paths[0] if len(materialized.paths) == 1 else materialized.root
    assets = GemmaAssets.load(source)
    module_map = _ltx_text_module_map()
    config_type = str(assets.config_dict["model_type"])
    if config_type == "gemma4_unified":
        try:
            from transformers.models.gemma4_unified.modeling_gemma4_unified import (
                Gemma4UnifiedRMSNorm,
                Gemma4UnifiedVisionEmbedder,
            )
        except ModuleNotFoundError:
            from .gemma4_unified import load_native_gemma4_unified_encoder

            return load_native_gemma4_unified_encoder(
                source,
                device=policy.device,
                dtype=policy.dtype,
                offload_to_cpu=policy.offload.mode is not OffloadMode.NONE,
            )

        module_map[Gemma4UnifiedRMSNorm] = AutoWrappedModule
        module_map[Gemma4UnifiedVisionEmbedder] = AutoWrappedNonRecurseModule
    elif config_type == "gemma4":
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioAttention,
            Gemma4RMSNorm,
            Gemma4VisionPatchEmbedder,
        )

        module_map[Gemma4RMSNorm] = AutoWrappedModule
        module_map[Gemma4VisionPatchEmbedder] = AutoWrappedNonRecurseModule
        module_map[Gemma4AudioAttention] = AutoWrappedNonRecurseModule
    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXSplitGemmaModule,
            config={"assets": assets, "execution_device": policy.device},
            state_dict_converter=partial(_convert_split_gemma_state_dict, model_type=config_type),
            vram_module_map=module_map,
            layer_container="encoder.model.model.language_model.layers",
            lazy_state_dict=True,
            buffer_aware_offload=True,
        ),
        CheckpointSpec(source=assets.weight_paths),
        policy,
    )
    if not isinstance(module, LTXSplitGemmaModule):
        raise TypeError("LTX split Gemma construction returned an unexpected module")
    return module, assets


class LTXGemmaPromptConditioner:
    """Gemma-4 split encoder, projection/connectors and optional duration prediction."""

    def __init__(
        self,
        gemma: torch.nn.Module,
        processor: LTXEmbeddingProcessorModule,
        *,
        duration_head: torch.nn.Module | None = None,
        enhancer: torch.nn.Module | None = None,
        supports_generated_keyframes: bool = False,
    ) -> None:
        self.gemma = gemma
        self.processor = processor
        self.duration_head = duration_head
        self.enhancer = enhancer
        self.supports_generated_keyframes = supports_generated_keyframes

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        if request.inputs.get("num_generated_keyframes", 0) and not self.supports_generated_keyframes:
            raise ValueError("LTX generated keyframes require a transformer trained with keyframe position embeddings")
        if request.inputs.get("auto_duration") is not None and self.duration_head is None:
            raise ValueError("LTX auto_duration requires the duration_head checkpoint")
        prompts = list(request.prompts)
        if request.inputs.get("enhance_prompt", False):
            if self.enhancer is None:
                raise ValueError("LTX prompt enhancement requires a local instruct Gemma checkpoint")
            from ...initializers.ltx import enhance_ltx_prompt

            prompts[0] = enhance_ltx_prompt(self.enhancer.encoder, prompts[0], request.inputs.get("images", ()))
        encoded = [
            self.processor.processor.process_hidden_states(hidden, mask)
            for hidden, mask in self.gemma.encoder.encode(prompts)
        ]
        return Conditioning(
            positive={
                "video_context": torch.cat([value.video_encoding for value in encoded]).to(device=device, dtype=dtype),
                "audio_context": torch.cat([value.audio_encoding for value in encoded]).to(device=device, dtype=dtype),
            }
        )

    def resolve_request(self, request: DiffusionRequest, conditioning: Conditioning) -> DiffusionRequest:
        bounds = request.inputs.get("auto_duration")
        if bounds is None:
            return request
        fps = float(request.inputs.get("frame_rate", 24.0))
        seconds = self.duration_head(conditioning.positive["video_context"], conditioning.positive["audio_context"])
        if seconds.shape != (1,):
            raise ValueError("LTX auto_duration supports a single prompt")
        minimum, maximum = (round(float(value) * fps) for value in bounds)
        frames = max(minimum, min(round(seconds.item() * fps), maximum))
        frames = ((frames - 1) // 8) * 8 + 1
        if frames < minimum:
            frames = min(-(-(minimum - 1) // 8) * 8 + 1, maximum)
        return replace(request, num_frames=frames)


class LTXDurationHeadModule(torch.nn.Module):
    """Native-loader wrapper for the connector-conditioned duration head."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        from ...networks.ltx.duration_head import DurationHeadConfigurator

        self.head = DurationHeadConfigurator.from_config(dict(checkpoint_config))

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(video, audio)


def convert_ltx_duration_head_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return {
        f"head.{key.removeprefix('duration_head.')}": state_dict[key]
        for key in state_dict
        if key.startswith("duration_head.")
    }


def build_ltx_gemma_prompt_conditioner(context: ComponentBuildContext) -> LTXGemmaPromptConditioner:
    """Load split text/projector weights, transformer connectors and duration head."""
    # Quantization is a DiT policy; text and codec components retain checkpoint precision.
    policy = replace(context.policy, quantization=QuantizationPolicy())
    text_checkpoint = context.require_checkpoint("text_encoder")
    gemma, assets = _load_split_gemma(text_checkpoint, policy)
    if assets.config_dict["model_type"] not in {"gemma3", "gemma4_unified"}:
        raise ValueError("LTX encoding requires Gemma-3 or Gemma-4 unified")
    transformer = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
    config = safetensors_json_metadata(transformer)
    processor = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXEmbeddingProcessorModule,
            config={"checkpoint_config": config, "gemma_config": assets.config_dict},
            state_dict_converter=partial(convert_ltx_embedding_processor_state_dict, lazy=True),
            vram_module_map=_ltx_text_module_map(),
            layer_container="processor.video_connector.transformer_1d_blocks",
            lazy_state_dict=True,
            buffer_aware_offload=True,
        ),
        CheckpointSpec(source=tuple(str(path) for path in (*transformer.paths, *assets.weight_paths))),
        policy,
    )
    duration = (
        context.checkpoint("duration_head") if context.component_options.get("load_duration_head", True) else None
    )
    head = (
        None
        if duration is None
        else NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXDurationHeadModule,
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=convert_ltx_duration_head_state_dict,
            ),
            duration,
            replace(policy, offload=OffloadPolicy()),
        )
    )
    enhancer_checkpoint = context.component_options.get("prompt_enhancer_checkpoint")
    if enhancer_checkpoint is not None and not isinstance(enhancer_checkpoint, CheckpointSpec):
        raise TypeError("prompt_enhancer_checkpoint must be an explicit CheckpointSpec")
    enhancer = None if enhancer_checkpoint is None else _load_split_gemma(enhancer_checkpoint, policy)[0]
    return LTXGemmaPromptConditioner(
        gemma,
        processor,
        duration_head=head,
        enhancer=enhancer,
        supports_generated_keyframes=bool(config.get("transformer", {}).get("use_keyframes_abs_pos_embedding", False)),
    )
