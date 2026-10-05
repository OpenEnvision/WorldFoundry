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
from functools import lru_cache

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
from ....optimizations import RuntimePolicy
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

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        config = dict(checkpoint_config)
        transformer = dict(config.get("transformer", {}))
        hidden_size = int(transformer.get("caption_channels", 3840))
        hidden_layers = int(transformer.get("caption_hidden_layers", 48)) + 1
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
            video_connector=Embeddings1DConnectorConfigurator.from_config(config),
            audio_connector=AudioEmbeddings1DConnectorConfigurator.from_config(config),
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


def convert_ltx_gemma_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map official Gemma weights to the installed Transformers module layout."""

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        if key.startswith("language_model.model."):
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
            converted[f"model.model.vision_tower.{suffix}"] = value
        elif key.startswith("multi_modal_projector."):
            converted[f"model.model.{key}"] = value
    return converted


def convert_ltx_embedding_processor_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    prefixes = {
        "text_embedding_projection.aggregate_embed.": "processor.feature_extractor.aggregate_embed.",
        "text_embedding_projection.video_aggregate_embed.": ("processor.feature_extractor.video_aggregate_embed."),
        "text_embedding_projection.audio_aggregate_embed.": ("processor.feature_extractor.audio_aggregate_embed."),
        "model.diffusion_model.video_embeddings_connector.": "processor.video_connector.",
        "model.diffusion_model.embeddings_connector.": "processor.video_connector.",
        "model.diffusion_model.audio_embeddings_connector.": "processor.audio_connector.",
    }
    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        for source, destination in prefixes.items():
            if key.startswith(source):
                converted[f"{destination}{key.removeprefix(source)}"] = value
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
    gemma = load_ltx_gemma_module(
        context.require_checkpoint("gemma"),
        context.policy,
    )
    processor = load_ltx_embedding_processor_module(
        context.require_checkpoint("weights"),
        context.policy,
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
