"""Cosmos Predict 2.5 Reason1 (Qwen2.5-VL) :class:`~...contracts.ConditionEncoder`.

Tokenizes prompts with an optional English system prefix and encodes
them with the Qwen2.5-VL text tower (layer-concat strategy).  Returns
``Conditioning.positive[\"context\"]`` for the Predict 2.5 DiT.  Does
not rewrite user prompt strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLTextModel,
)

try:
    # transformers <= 4.56 kept a Qwen2.5-VL-specific export.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm
except ImportError:
    # transformers 4.57 reuses and exports the base Qwen2 implementation.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2RMSNorm as Qwen2_5_VLRMSNorm,
    )

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from worldfoundry.core.model_loading.text_embeddings import EmbeddingConcatStrategy

from ....loaders import (
    CheckpointSpec,
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
)
from ....optimizations import RuntimePolicy

_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."


class Cosmos25TextBackbone(nn.Module):
    """Language-only slice of the Reason1 Qwen2.5-VL checkpoint."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2_5_VLTextModel(config)

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)


def _reason1_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    config = AutoConfig.from_pretrained(checkpoint.root, local_files_only=True, trust_remote_code=False)
    return {"config": config.text_config}


def convert_reason1_text_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Discard the unused vision tower and LM head from Reason1."""

    return {key: value for key, value in state_dict.items() if key.startswith("model.")}


class Cosmos25PromptConditioner:
    def __init__(
        self,
        model: Cosmos25TextBackbone,
        tokenizer,
        *,
        sequence_length: int = 512,
        embedding_concat_strategy: str = "full_concat",
        n_layers_per_group: int = 5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.embedding_concat_strategy = str(embedding_concat_strategy)
        self.n_layers_per_group = int(n_layers_per_group)
        valid = {str(item) for item in EmbeddingConcatStrategy}
        if self.embedding_concat_strategy not in valid:
            raise ValueError(f"invalid embedding_concat_strategy: {self.embedding_concat_strategy}")
        if self.n_layers_per_group <= 0:
            raise ValueError("n_layers_per_group must be positive")

    @torch.inference_mode()
    def _encode(self, prompts: Sequence[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=self.sequence_length,
                truncation=True,
                padding="max_length",
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            rows.append(torch.tensor(ids, dtype=torch.long))
        input_ids = torch.stack(rows).to(device=device)
        hidden_states = self.model(input_ids).hidden_states
        normalized = [
            (value - value.mean(dim=-1, keepdim=True)) / (value.std(dim=-1, keepdim=True) + 1e-8)
            for value in hidden_states[1:]
        ]
        if self.embedding_concat_strategy == str(EmbeddingConcatStrategy.FULL_CONCAT):
            output = torch.cat(normalized, dim=-1)
        elif self.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING):
            output = torch.stack(normalized).mean(dim=0)
        else:
            groups = [
                torch.stack(normalized[index : index + self.n_layers_per_group]).mean(dim=0)
                for index in range(0, len(normalized), self.n_layers_per_group)
            ]
            output = torch.cat(groups, dim=-1)
        return output.to(dtype=dtype)

    def compute_text_embeddings_online(
        self,
        data_batch: Mapping[str, object],
        input_caption_key: str,
    ) -> torch.Tensor:
        """Compatibility surface for model-internal multi-view conditioners."""

        raw_prompts = data_batch[input_caption_key]
        if isinstance(raw_prompts, str) or not isinstance(raw_prompts, Sequence):
            raw_prompts = [raw_prompts]
        prompts: list[str] = []
        for value in raw_prompts:
            if isinstance(value, str):
                prompts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                prompts.append(" ".join(str(item) for item in value))
            else:
                prompts.append(str(value))
        parameter = next(self.model.parameters())
        return self._encode(prompts, device=parameter.device, dtype=parameter.dtype)

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        positive = {"context": self._encode(request.prompts, device=device, dtype=dtype)}
        negative: dict[str, torch.Tensor] = {}
        if request.sampling.guidance_scale != 1.0:
            prompts = request.negative_prompts or (("",) * request.batch_size)
            negative["context"] = self._encode(prompts, device=device, dtype=dtype)
        return Conditioning(
            positive=positive,
            negative=negative,
            shared={"fps": float(request.inputs.get("fps", request.inputs.get("frame_rate", 16.0)))},
        )


def _load_reason1_backbone(
    checkpoint_spec: CheckpointSpec,
    policy: RuntimePolicy,
) -> Cosmos25TextBackbone:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos25TextBackbone,
            config_resolver=_reason1_config,
            state_dict_converter=convert_reason1_text_state_dict,
            vram_module_map={
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.RMSNorm: AutoWrappedModule,
                Qwen2_5_VLRMSNorm: AutoWrappedModule,
                Qwen2_5_VLRotaryEmbedding: AutoWrappedModule,
            },
            layer_container="model.layers",
        ),
        checkpoint_spec,
        policy,
    )
    if not isinstance(model, Cosmos25TextBackbone):
        raise TypeError(f"expected Cosmos25TextBackbone, got {type(model).__name__}")
    return model


def _load_reason1_tokenizer(checkpoint_spec: CheckpointSpec):
    checkpoint = NativeCheckpointResolver().materialize(checkpoint_spec)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    return tokenizer


def build_cosmos25_prompt_conditioner(context: ComponentBuildContext) -> Cosmos25PromptConditioner:
    model = _load_reason1_backbone(context.require_checkpoint("weights"), context.policy)
    tokenizer = _load_reason1_tokenizer(context.require_checkpoint("tokenizer"))
    return Cosmos25PromptConditioner(
        model,
        tokenizer,
        sequence_length=int(context.component_options.get("sequence_length", 512)),
        embedding_concat_strategy=str(
            context.component_options.get("embedding_concat_strategy", "full_concat")
        ),
        n_layers_per_group=int(context.component_options.get("n_layers_per_group", 5)),
    )


def load_cosmos_reason1_prompt_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sequence_length: int = 512,
    embedding_concat_strategy: str = "full_concat",
    n_layers_per_group: int = 5,
) -> Cosmos25PromptConditioner:
    """Load one shared language-only Cosmos Reason1 prompt encoder."""

    source = str(checkpoint_path)
    if source.startswith("hf://"):
        from worldfoundry.core.io.assets.easy_io import resolve_checkpoint_path

        source = resolve_checkpoint_path(source)
    checkpoint = CheckpointSpec(source=source)
    policy = RuntimePolicy(device=device, dtype=dtype)
    return Cosmos25PromptConditioner(
        _load_reason1_backbone(checkpoint, policy),
        _load_reason1_tokenizer(checkpoint),
        sequence_length=sequence_length,
        embedding_concat_strategy=embedding_concat_strategy,
        n_layers_per_group=n_layers_per_group,
    )


@dataclass(slots=True)
class CosmosReason1TextEncoderConfig:
    compute_online: bool = False
    embedding_concat_strategy: str = "full_concat"
    n_layers_per_group: int = 5
    ckpt_path: str = "hf://nvidia/Cosmos-Reason1-7B"
    sequence_length: int = 512


class CosmosReason1TextEncoder:
    """Thin legacy-facing view of the shared native prompt encoder."""

    def __init__(
        self,
        config: CosmosReason1TextEncoderConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = config
        self.encoder = load_cosmos_reason1_prompt_encoder(
            config.ckpt_path,
            device=device,
            embedding_concat_strategy=str(config.embedding_concat_strategy),
            n_layers_per_group=int(config.n_layers_per_group),
            sequence_length=int(getattr(config, "sequence_length", 512)),
        )
        self.model = self.encoder.model

    def compute_text_embeddings_online(
        self,
        data_batch: Mapping[str, object],
        input_caption_key: str,
    ) -> torch.Tensor:
        return self.encoder.compute_text_embeddings_online(data_batch, input_caption_key)


__all__ = [
    "CosmosReason1TextEncoder",
    "CosmosReason1TextEncoderConfig",
    "Cosmos25PromptConditioner",
    "Cosmos25TextBackbone",
    "build_cosmos25_prompt_conditioner",
    "convert_reason1_text_state_dict",
    "load_cosmos_reason1_prompt_encoder",
]
