# SPDX-License-Identifier: Apache-2.0
# Adapter contract adapted from Pika-Labs/Pika-H3-Flex, revision
# 0ddf55ad7067ca1e76812253bd7b4600df87f759.
"""Pika H3 Flex adapter loading on the canonical MiniMax-H3 DiT.

Pika publishes PEFT LoRA adapters against Diffusers' unfused H3 names.  The
canonical WorldFoundry model uses the already-converted fused QKV and SwiGLU
projections, so this loader converts and merges the adapter deltas without
copying the transformer or introducing a second H3 implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ._lora import read_lora_groups, validate_lora_pair, validate_lora_scale


def _target_spec(source: str) -> tuple[str, str | None, bool] | None:
    """Map one Diffusers linear name to a canonical target and optional QKV slice."""

    source = source.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    source = source.replace("transformer_blocks.", "blocks.")
    qkv_targets = {
        ".attn.to_q": "q",
        ".attn.to_k": "k",
        ".attn.to_v": "v",
    }
    for suffix, logical in qkv_targets.items():
        if source.endswith(suffix):
            return source[: -len(suffix)] + ".attn.qkv_proj", logical, False
    if source.endswith(".attn.to_out.0"):
        return source[: -len(".attn.to_out.0")] + ".attn.out_proj", None, False
    if source.endswith(".ff.net.0.proj"):
        return source[: -len(".ff.net.0.proj")] + ".mlp.fc1", None, True
    if source.endswith(".ff.net.2"):
        return source[: -len(".ff.net.2")] + ".mlp.fc2", None, False
    return None


def load_pika_h3_flex_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    rank: int = 128,
    alpha: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge one released Pika H3 Flex LoRA adapter into a canonical DiT.

    Args:
        model: Loaded :class:`MiniMaxH3DiTModel` instance.
        path: ``adapter.safetensors`` from the ``fl2va`` or ``ref2va`` release.
        rank: Published LoRA rank.
        alpha: Published LoRA alpha; defaults to ``rank``.
        strict: Reject unsupported targets and incomplete LoRA pairs.

    Returns:
        A small loading report containing the merged target count and names.
    """

    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Pika H3 Flex adapter not found: {adapter_path}")
    rank, alpha = validate_lora_scale(rank=rank, alpha=alpha, label="Pika H3 Flex")

    merged: list[str] = []
    for source, pair in read_lora_groups(adapter_path, label="Pika H3 Flex").items():
        target_spec = _target_spec(source)
        if target_spec is None:
            if strict:
                raise ValueError(f"Unsupported Pika H3 Flex LoRA target: {source}")
            continue
        matrix_a, matrix_b = validate_lora_pair(source, pair, rank=rank, label="Pika H3 Flex")
        target, logical, swap_halves = target_spec
        module = model.get_submodule(target)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Pika H3 Flex LoRA target is not linear: {target}")
        delta = (matrix_b.float() @ matrix_a.float()) * (float(alpha) / float(rank))
        if swap_halves:
            if delta.shape[0] % 2:
                raise ValueError(f"Fused SwiGLU adapter output is not divisible by two: {source}")
            delta = torch.cat(delta.chunk(2, dim=0)[::-1], dim=0)

        if logical is None:
            expected = tuple(module.weight.shape)
            if tuple(delta.shape) != expected:
                raise ValueError(f"Pika H3 Flex delta shape {tuple(delta.shape)} != {expected} for {target}")
            with torch.no_grad():
                module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        else:
            inner = int(module.weight.shape[0]) // 3
            if tuple(delta.shape) != (inner, int(module.weight.shape[1])):
                raise ValueError(f"Pika H3 Flex QKV delta shape {tuple(delta.shape)} is incompatible with {target}")
            offsets = {"q": 0, "k": inner, "v": 2 * inner}
            start = offsets[logical]
            with torch.no_grad():
                module.weight[start : start + inner].add_(
                    delta.to(device=module.weight.device, dtype=module.weight.dtype)
                )
        merged.append(source)

    if not merged:
        raise ValueError(f"Pika H3 Flex adapter contains no supported LoRA targets: {adapter_path}")
    return {"path": str(adapter_path), "rank": rank, "alpha": alpha, "merged_targets": tuple(merged)}


__all__ = ["load_pika_h3_flex_adapter"]
