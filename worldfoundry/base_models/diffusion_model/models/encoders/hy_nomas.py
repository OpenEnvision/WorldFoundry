# Copyright 2026 The Hy-Nomas Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hy-Nomas prompt formatting around the canonical Qwen2.5-VL text backbone."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest
from ...loaders import ModuleLoadSpec, NativeCheckpointResolver, NativeModuleLoader
from ...optimizations import OffloadPolicy, QuantizationPolicy
from .cosmos2p5.component import Cosmos25TextBackbone, _qwen_class, _reason1_config

SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "        1. The main content and theme of the video. "
    "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "        4. background environment, light, style and atmosphere. "
    "        5. camera angles, movements, and transitions used in the video."
)


def segment_prompts(request: DiffusionRequest) -> list[str]:
    """A prompt list describes successive segments, never independent batch items."""
    values = request.inputs.get("segments")
    if values is None:
        values = list(request.prompts)
    if not isinstance(values, (tuple, list)) or not values or not all(isinstance(p, str) and p for p in values):
        raise ValueError("Hy-Nomas segments must be a nonempty list of prompt strings")
    return list(values)


def action_variants(prompt: str) -> list[str]:
    """Prefixes locate structured action boundaries in the actual chat tokenization."""
    try:
        obj = json.loads(prompt)
    except ValueError:
        return []
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return []
    actions = segments[0].get("contained_actions")
    if not isinstance(actions, (dict, list)) or not actions:
        return []
    variants = []
    for count in range(len(actions)):
        segment = dict(segments[0])
        if not count:
            segment.pop("contained_actions")
        else:
            segment["contained_actions"] = (
                dict(list(actions.items())[:count]) if isinstance(actions, dict) else actions[:count]
            )
        variants.append(
            json.dumps({k: [segment] if k == "segments" else v for k, v in obj.items()}, ensure_ascii=False)
        )
    return variants


def _timestamp(value: Any) -> float:
    parts = str(value).split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid action timestamp: {value}")
    return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))


def action_windows(prompt: str, frames: int, fps: float) -> list[tuple[int, int, int]] | None:
    """Convert the released JSON action timestamps into non-overlapping latent windows."""
    try:
        obj = json.loads(prompt)
    except ValueError:
        return None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return None
    actions = segments[0].get("contained_actions", [])
    if isinstance(actions, dict):
        actions = list(actions.values())
    if not isinstance(actions, list) or not actions:
        return None
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return None
        start, end = [
            max(0, min(frames, int(max(_timestamp(action[k]), 0) * fps) // 4)) for k in ("start_time", "end_time")
        ]
        if end > start:
            spans.append((start, end, index))
    if not spans:
        return None
    spans.sort()
    cursor, windows = 0, []
    for start, end, index in spans:
        if start < cursor:
            raise ValueError("action time windows must not overlap")
        if start > cursor:
            windows.append((cursor, start, -1))
        windows.append((start, end, index))
        cursor = end
    if cursor < frames:
        windows.append((cursor, frames, -1))
    return None if len(windows) == len(actions) == 1 else windows


class HyNomasConditioner:
    """Use the released chat template, prefix crop, and third-last hidden state."""

    def __init__(
        self, model: Cosmos25TextBackbone, tokenizer: Any, *, length: int = 1280, low_memory: bool = False
    ) -> None:
        self.model, self.tokenizer, self.length = model, tokenizer, length
        self.low_memory = low_memory

    def _tokenize(self, prompt: str, length: int | None = None) -> Any:
        options = {} if length is None else {"max_length": length, "truncation": True, "padding": "max_length"}
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **options,
        )

    @torch.inference_mode()
    def encode_text(self, prompt: str, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        marker = self.tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        empty = self._tokenize("")["input_ids"][0].tolist()
        crop = next((i + len(marker) for i in range(len(empty)) if empty[i : i + len(marker)] == marker), None)
        if crop is None:
            raise ValueError("Hy-Nomas requires the Qwen user marker in its chat template")
        tokens = self._tokenize(prompt, self.length + crop)
        model_device = next(self.model.parameters()).device
        encoded = self.model.model(
            **{k: v.to(model_device) for k, v in tokens.items()},
            output_hidden_states=True,
            use_cache=False,
        )
        ids = tokens["input_ids"][0].tolist()
        boundaries = []
        for variant in action_variants(prompt):
            other = self._tokenize(variant, self.length + crop)["input_ids"][0].tolist()
            common = next(
                (i for i, (a, b) in enumerate(zip(ids, other, strict=False)) if a != b), min(len(ids), len(other))
            )
            boundaries.append(max(boundaries[-1] if boundaries else 0, min(self.length - 1, common - crop)))
        return {
            "encoder_hidden_states": encoded.hidden_states[-3][:, crop:].to(device=device, dtype=dtype),
            "encoder_attention_mask": tokens["attention_mask"][:, crop:].to(device=device),
            "action_text_offsets": [boundaries + [min(self.length, int(tokens["attention_mask"].sum()))]]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
            if boundaries
            else None,
            "cache_action_start": boundaries[0] if boundaries else None,
        }

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        prompts = segment_prompts(request)
        positive = [self.encode_text(prompt, device=device, dtype=dtype) for prompt in prompts]
        negative: dict[str, Any] = {}
        if request.sampling.guidance_scale != 1:
            negative = self.encode_text(
                request.negative_prompts[0] if request.negative_prompts else "", device=device, dtype=dtype
            )
        return Conditioning(positive={"segments": positive}, negative=negative)


def convert_text_checkpoint(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the language tower of the published Qwen2.5-VL checkpoint."""
    return {
        key.replace("model.language_model.", "model.", 1): value
        for key, value in state.items()
        if key.startswith("model.") and not key.startswith("model.visual.")
    }


def _restore_text_rotary(model: Cosmos25TextBackbone) -> None:
    """Keep checkpoint-derived Qwen rotary frequencies at their FP32 inference precision."""
    rotary = model.model.rotary_emb
    rotary.inv_freq, rotary.attention_scaling = rotary.rope_init_fn(rotary.config, rotary.inv_freq.device)
    rotary.original_inv_freq = rotary.inv_freq


def build_hy_nomas_conditioner(context: ComponentBuildContext) -> HyNomasConditioner:
    from transformers import AutoTokenizer

    checkpoint = context.require_checkpoint()
    resolver = NativeCheckpointResolver()
    materialized = resolver.materialize(checkpoint)
    architecture = json.loads((materialized.root / "config.json").read_text(encoding="utf-8"))
    if architecture.get("model_type") != "qwen2_5_vl":
        raise ValueError("Hy-Nomas requires a Qwen2.5-VL checkpoint, not a different PE architecture")
    config = _reason1_config(materialized)
    if int(cast(Any, config["config"]).hidden_size) != 3584:
        raise ValueError("Hy-Nomas requires the Qwen2.5-VL-7B text tower (hidden_size=3584)")
    low_memory = bool(context.component_options.get("low_memory", False))
    policy = replace(
        context.policy,
        device="cpu" if low_memory else context.policy.device,
        compile=False,
        offload=OffloadPolicy(),
        quantization=QuantizationPolicy(),
    )
    # Resolve optional model imports before the shared loader's meta construction context.
    _qwen_class("Qwen2_5_VLTextModel")
    model = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=Cosmos25TextBackbone, config=config, state_dict_converter=convert_text_checkpoint),
        checkpoint,
        policy,
    )
    _restore_text_rotary(cast(Cosmos25TextBackbone, model))
    tokenizer = AutoTokenizer.from_pretrained(materialized.root, local_files_only=True, padding_side="right")
    return HyNomasConditioner(
        cast(Cosmos25TextBackbone, model),
        tokenizer,
        length=int(cast(int, context.component_options.get("max_sequence_length", 1280))),
        low_memory=low_memory,
    )
