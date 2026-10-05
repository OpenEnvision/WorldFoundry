"""Shared T5 :class:`~...contracts.ConditionEncoder`.

Tokenizes ``DiffusionRequest.prompts`` / ``negative_prompts`` and runs a
Hugging Face T5 encoder loaded through
:class:`~...loaders.module.NativeModuleLoader`.  Returns
:class:`~...contracts.Conditioning` with ``positive[\"context\"]`` and,
when CFG ≠ 1, ``negative[\"context\"]``.  Used by recipes that do not
own a family-specific text tower (and as a fallback for Wan-compatible
T5 variants).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch
from transformers import AutoTokenizer, T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5LayerNorm

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import MaterializedCheckpoint, ModuleLoadSpec, NativeCheckpointResolver, NativeModuleLoader


class T5EncoderModule(torch.nn.Module):
    """T5 encoder architecture materialized by the shared module loader."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.model = T5EncoderModel(T5Config.from_dict(dict(checkpoint_config)))


def _t5_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    candidates = (checkpoint.root / "config.json", checkpoint.root / "text_encoder" / "config.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"T5 config does not exist below {checkpoint.root}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("T5 config.json must contain an object")
    return {"checkpoint_config": value}


def convert_t5_encoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Select encoder-only Hugging Face tensors below the stable wrapper."""

    converted = {
        f"model.{key}": value
        for key, value in state_dict.items()
        if key == "shared.weight" or key.startswith("encoder.")
    }
    if "shared.weight" in state_dict:
        converted["model.encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    return converted


class T5EncoderConditioner:
    """Encode prompt branches with configurable framework conditioning keys."""

    def __init__(
        self,
        model: T5EncoderModule,
        tokenizer,
        *,
        max_length: int,
        context_key: str,
        mask_key: str,
        mask_mode: str,
        negative_fallback: str,
        zero_padding: bool,
    ) -> None:
        if negative_fallback not in {"prompt", "omit"}:
            raise ValueError("T5 negative_fallback must be 'prompt' or 'omit'")
        if mask_mode not in {"tokenizer", "ones"}:
            raise ValueError("T5 mask_mode must be 'tokenizer' or 'ones'")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.context_key = str(context_key)
        self.mask_key = str(mask_key)
        self.mask_mode = mask_mode
        self.negative_fallback = negative_fallback
        self.zero_padding = bool(zero_padding)

    @torch.no_grad()
    def _encode(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        attention_mask = tokens.attention_mask.to(device)
        hidden_states = self.model.model(
            input_ids=tokens.input_ids.to(device),
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        if self.zero_padding:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        output_mask = torch.ones_like(attention_mask) if self.mask_mode == "ones" else attention_mask
        return hidden_states, output_mask

    def _values(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        context, mask = encoded
        return {
            self.context_key: context.to(device=device, dtype=dtype),
            self.mask_key: mask.to(device=device),
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Tokenize prompts and return T5 ``context`` / mask tensors."""
        positive = self._values(
            self._encode(request.prompts, device=device),
            device=device,
            dtype=dtype,
        )
        negative_prompts = request.negative_prompts
        if negative_prompts is None and self.negative_fallback == "prompt":
            negative_prompts = request.prompts
        negative = (
            self._values(
                self._encode(negative_prompts, device=device),
                device=device,
                dtype=dtype,
            )
            if negative_prompts is not None
            else {}
        )
        return Conditioning(positive=positive, negative=negative)


def build_t5_encoder_conditioner(context: ComponentBuildContext) -> T5EncoderConditioner:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=T5EncoderModule,
            config_resolver=_t5_config,
            state_dict_converter=convert_t5_encoder_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            layer_container="model.encoder.block",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, T5EncoderModule):
        raise TypeError(f"expected T5EncoderModule, got {type(model).__name__}")
    tokenizer_checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer_root = tokenizer_checkpoint.root / "tokenizer"
    if not tokenizer_root.is_dir():
        tokenizer_root = tokenizer_checkpoint.directory()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    options = context.component_options
    return T5EncoderConditioner(
        model,
        tokenizer,
        max_length=int(options.get("max_length", 512)),
        context_key=str(options.get("context_key", "context")),
        mask_key=str(options.get("mask_key", "context_mask")),
        mask_mode=str(options.get("mask_mode", "tokenizer")),
        negative_fallback=str(options.get("negative_fallback", "prompt")),
        zero_padding=bool(options.get("zero_padding", True)),
    )


__all__ = [
    "T5EncoderConditioner",
    "T5EncoderModule",
    "build_t5_encoder_conditioner",
    "convert_t5_encoder_state_dict",
]
