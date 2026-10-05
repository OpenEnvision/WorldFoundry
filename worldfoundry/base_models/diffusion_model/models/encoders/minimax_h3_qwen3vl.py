# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen3-VL layer-50 text/visual encoder.

Ported from SGLang (Apache-2.0):
``python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py``.
The SGLang reference wrapped an in-tree, tensor-parallel Qwen3-VL model whose
API mirrors HuggingFace ``transformers`` ``Qwen3VLModel``. This single-GPU port
drops the SGLang distributed / TP-folding machinery and wraps the HF class
directly, preserving the layer-50 trimming, the final-norm -> Identity swap, the
bf16 ``[num_tokens, 5120]`` output contract, and the unused-weight skip logic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Constants preserved from the SGLang reference. H3 consumes the *unnormalized*
# hidden state immediately after language-model layer index 49 ("layer 50").
MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120

# Checkpoint weight names carry a leading ``model.`` prefix (from the full
# Qwen3-VL-for-conditional-generation checkpoint); the inner HF ``Qwen3VLModel``
# parameter names do not. This matches ``model.language_model.layers.<idx>.``.
_LAYER_WEIGHT_RE = re.compile(r"^(?:model\.)?language_model\.layers\.(\d+)\.")
_CHECKPOINT_PREFIX = "model."


def _is_unconsumed_checkpoint_weight(name: str) -> bool:
    """Weights intentionally absent from the layer-50 feature extractor.

    The final ``language_model.norm`` (replaced by ``nn.Identity``), the
    ``lm_head``, and every language-model layer at index >= 50 are dropped.
    """
    if name == "lm_head.weight" or name.startswith(
        (f"{_CHECKPOINT_PREFIX}language_model.norm.", "language_model.norm.")
    ):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)


@dataclass
class MiniMaxH3Qwen3VLConfig:
    """Architecture configuration for the MiniMax H3 Qwen3-VL encoder.

    The checkpoint is Qwen3-VL-32B, consumed at ``hidden_states[50]``. Defaults
    reflect the SGLang reference; a full HF ``Qwen3VLConfig`` (used to build the
    backbone) is carried on ``hf_config`` and is normally populated by
    :meth:`MiniMaxH3Qwen3VLEncoder.from_pretrained`.
    """

    hidden_size: int = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
    intermediate_size: int = 25600
    num_hidden_layers: int = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    text_len: int = 262144
    # Qwen3-VL special token ids (HF defaults; overridden from hf_config).
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    architectures: list[str] = field(default_factory=lambda: ["MiniMaxH3Qwen3VLEncoder"])
    # Full HuggingFace Qwen3VLConfig instance for backbone construction; typed
    # Any to avoid importing transformers at module import time.
    hf_config: Any = None


class MiniMaxH3Qwen3VLEncoder(nn.Module):
    """Qwen3-VL multimodal backbone ending at ``hidden_states[50]``.

    Single-GPU PyTorch port: the backbone is a HuggingFace ``Qwen3VLModel``
    trimmed to 50 language-model layers, with its final norm replaced by
    ``nn.Identity`` so the encoder returns the unnormalized hidden state right
    after layer index 49.
    """

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return "rotary_emb.inv_freq" not in name and not _is_unconsumed_checkpoint_weight(name)

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        # Lazy import: importing this module must not require transformers'
        # heavy modeling stack (which currently trips a torchao/torch mismatch
        # in some envs). Heavy construction happens here, at __init__ time.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        hf_config = config.hf_config
        if hf_config is None:
            hf_config = self._build_hf_config(config)

        selected_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        # Trim to 50 language-model layers before construction.
        hf_config.text_config.num_hidden_layers = selected_layer
        hf_config.text_config.use_cache = False
        hf_config.text_config.output_hidden_states = False
        if int(hf_config.text_config.num_hidden_layers) != selected_layer:
            raise ValueError(
                f"MiniMax H3 Qwen3-VL config must be trimmed to {selected_layer} language layers before construction"
            )

        self.config = config
        self.model = Qwen3VLModel(hf_config)
        # H3 consumes the unnormalized output immediately after layer 49.
        self.model.language_model.norm = nn.Identity()

        self.image_token_id = int(hf_config.image_token_id)
        self.video_token_id = int(hf_config.video_token_id)
        self.selected_lm_layer = selected_layer
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM

    @staticmethod
    def _build_hf_config(config: MiniMaxH3Qwen3VLConfig) -> Any:
        """Construct a HF ``Qwen3VLConfig`` from the dataclass defaults."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
        )

        hf_config = Qwen3VLConfig()
        text_config = hf_config.text_config
        text_config.hidden_size = config.hidden_size
        text_config.intermediate_size = config.intermediate_size
        text_config.num_attention_heads = config.num_attention_heads
        text_config.num_key_value_heads = config.num_key_value_heads
        text_config.head_dim = config.head_dim
        text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        return hf_config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        strict: bool = False,
        preserve_rotary_precision: bool = False,
        low_cpu_mem_usage: bool = False,
        **hf_config_kwargs: Any,
    ) -> "MiniMaxH3Qwen3VLEncoder":
        """Build the encoder from an HF checkpoint directory.

        Loads the HF ``Qwen3VLConfig`` from ``model_path`` (so the real token
        ids / vision config are used), constructs the trimmed backbone, then
        streams the checkpoint weights, skipping the unused ones.
        """
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, **hf_config_kwargs)
        hf_config.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER

        config = MiniMaxH3Qwen3VLConfig(
            hidden_size=int(hf_config.text_config.hidden_size),
            intermediate_size=int(hf_config.text_config.intermediate_size),
            num_attention_heads=int(hf_config.text_config.num_attention_heads),
            num_key_value_heads=int(hf_config.text_config.num_key_value_heads),
            head_dim=int(hf_config.text_config.head_dim),
            image_token_id=int(hf_config.image_token_id),
            video_token_id=int(hf_config.video_token_id),
            vision_start_token_id=int(hf_config.vision_start_token_id),
            hf_config=hf_config,
        )
        if low_cpu_mem_usage:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                encoder = cls(config)
        else:
            encoder = cls(config)
        rotary = {
            name: value.clone()
            for name, value in encoder.named_buffers()
            if preserve_rotary_precision and name.endswith("inv_freq")
        }
        if low_cpu_mem_usage:
            encoder.to(torch_dtype)
        loaded = encoder.load_weights(_iter_checkpoint_weights(model_path))
        names = {name.removeprefix(_CHECKPOINT_PREFIX) for name in loaded}
        missing = sorted(set(dict(encoder.model.named_parameters())) - names)
        encoder.loading_report = {"missing": missing, "loaded_tensors": len(loaded)}
        if strict and missing:
            raise ValueError(f"Incomplete MiniMax H3 text encoder checkpoint: {missing[:10]}")
        encoder.to(torch_dtype)
        for name, value in rotary.items():
            parent, _, field = name.rpartition(".")
            setattr(encoder.get_submodule(parent), field, value)
        return encoder

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the last hidden state (unnormalized layer-50 output)."""
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a 1-D ``input_ids`` sequence to ``[num_tokens, 5120]`` bf16.

        Optional ``pixel_values`` / ``image_grid_thw`` and
        ``pixel_values_videos`` / ``video_grid_thw`` feed the vision tower;
        mRoPE positions are computed via ``get_rope_index`` when vision inputs
        are present.
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be given together")

        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        host_image_grid_thw = image_grid_thw.to(device="cpu", dtype=torch.long) if image_grid_thw is not None else None
        host_video_grid_thw = video_grid_thw.to(device="cpu", dtype=torch.long) if video_grid_thw is not None else None
        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )

        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "use_cache": False,
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid_thw.to(self.device)
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(self.device, torch.bfloat16)
            call_kwargs["video_grid_thw"] = host_video_grid_thw.to(self.device)

        hidden = self.model(**call_kwargs).last_hidden_state[0].to(torch.bfloat16)
        expected_shape = [int(ids.shape[1]), self.hidden_dim]
        if list(hidden.shape) != expected_shape:
            raise ValueError(f"unexpected hidden shape {list(hidden.shape)}, expected {expected_shape}")
        return hidden

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load checkpoint weights, skipping the unused ones (lm_head, final
        norm, layers >= 50). Checkpoint names may carry a leading ``model.``
        prefix that the inner HF ``Qwen3VLModel`` params lack; it is stripped.
        """
        params = dict(self.model.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if not self.should_materialize_checkpoint_weight(name):
                continue
            param_name = name
            if param_name.startswith(_CHECKPOINT_PREFIX):
                param_name = param_name[len(_CHECKPOINT_PREFIX) :]
            param = params.get(param_name)
            if param is None:
                raise KeyError(f"Unexpected MiniMax H3 Qwen3-VL checkpoint weight: {name}")
            try:
                with torch.no_grad():
                    if param.shape != loaded_weight.shape:
                        raise ValueError("checkpoint tensor shape mismatch")
                    if param.is_meta:
                        from accelerate.utils import set_module_tensor_to_device

                        set_module_tensor_to_device(
                            self.model, param_name, "cpu", value=loaded_weight, dtype=param.dtype
                        )
                    else:
                        param.copy_(loaded_weight.to(param.dtype))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load MiniMax H3 Qwen3-VL weight "
                    f"{name!r}: checkpoint={tuple(loaded_weight.shape)}, "
                    f"parameter={tuple(param.shape)}"
                ) from exc
            loaded.add(name)
        return loaded

    @torch.no_grad()
    def encode_presentation(self, prompt, *, tokenizer, processor, images=()):
        """Encode H3's verbatim prompt and numbered reference images (no chat template).

        Vision rows carry modality tag 0, all other rows tag 1. Support both
        Transformers' original mRoPE interface and its explicit token-type interface.
        """
        import inspect

        token_ids, tags, vision = [], [], {}
        if images:
            vision = processor.image_processor(images=list(images), return_tensors="pt")
            merge = processor.image_processor.merge_size**2
            for index, grid in enumerate(vision["image_grid_thw"]):
                label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                image_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids.extend(label + image_ids)
                tags.extend([1] * len(label) + [0] * len(image_ids))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids.extend(prompt_ids)
        tags.extend([1] * len(prompt_ids))
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        arguments = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
        if "mm_token_type_ids" in inspect.signature(self.model.forward).parameters:
            arguments["mm_token_type_ids"] = torch.tensor(
                processor.create_mm_token_type_ids([token_ids]), device=self.device, dtype=torch.long
            )
        if images:
            arguments.update(
                pixel_values=vision["pixel_values"].to(self.device, torch.bfloat16),
                image_grid_thw=vision["image_grid_thw"].to(self.device),
            )
        hidden = self.model(**arguments).last_hidden_state
        return hidden.to(torch.bfloat16), torch.tensor(tags, device=self.device, dtype=torch.long)


def _iter_checkpoint_weights(
    model_path: str,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Stream (name, tensor) pairs from a HF safetensors checkpoint dir."""
    import glob
    import os

    from safetensors import safe_open

    shard_paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No *.safetensors found under {model_path!r}")
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                # The H3 feature extractor stops after layer 49. Avoid reading
                # unused checkpoint tensors (layers 50+, final norm, lm_head)
                # from the multi-GB shards merely to discard them downstream.
                if not MiniMaxH3Qwen3VLEncoder.should_materialize_checkpoint_weight(name):
                    continue
                yield name, f.get_tensor(name)


EntryClass = MiniMaxH3Qwen3VLEncoder

__all__ = [
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "MiniMaxH3Qwen3VLConfig",
    "MiniMaxH3Qwen3VLEncoder",
]
