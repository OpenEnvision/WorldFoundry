# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared LoRA safetensors loading utilities for MiniMax-H3 adapter variants."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

LORA_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
)

_ADAPTER_PREFIXES: tuple[str, ...] = ("base_model.model.", "pipe.dit.", "model.", "transformer.")


def strip_adapter_prefix(key: str) -> str:
    """Normalize common PEFT and framework wrapper prefixes on LoRA keys."""

    for prefix in _ADAPTER_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def read_lora_groups(path: Path, *, label: str) -> dict[str, dict[str, torch.Tensor]]:
    """Read LoRA A/B safetensor pairs grouped by stripped target module name."""

    from safetensors.torch import load_file

    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for raw_key, value in load_file(str(path), device="cpu").items():
        key = strip_adapter_prefix(raw_key)
        for suffix, side in LORA_SUFFIXES:
            if key.endswith(suffix):
                groups[key[: -len(suffix)]][side] = value
                break
        else:
            raise ValueError(f"Unsupported {label} adapter tensor: {raw_key}")
    return dict(groups)


def validate_lora_pair(
    source: str,
    pair: dict[str, torch.Tensor],
    *,
    rank: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate completeness, 2-D matrix shapes, and rank for one LoRA pair."""

    if set(pair) != {"A", "B"}:
        raise ValueError(f"Incomplete {label} LoRA pair for {source}: {sorted(pair)}")
    matrix_a, matrix_b = pair["A"], pair["B"]
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError(f"{label} LoRA tensors must be matrices: {source}")
    if int(matrix_a.shape[0]) != rank or int(matrix_b.shape[1]) != rank:
        raise ValueError(
            f"{label} LoRA rank mismatch for {source}: A={tuple(matrix_a.shape)} B={tuple(matrix_b.shape)} rank={rank}"
        )
    return matrix_a, matrix_b


def validate_lora_scale(*, rank: int, alpha: int | None, label: str) -> tuple[int, int]:
    """Validate LoRA rank and alpha values, defaulting alpha to rank."""

    if isinstance(rank, bool) or int(rank) <= 0:
        raise ValueError(f"{label} LoRA rank must be positive")
    resolved_rank = int(rank)
    resolved_alpha = resolved_rank if alpha is None else int(alpha)
    if resolved_alpha <= 0:
        raise ValueError(f"{label} LoRA alpha must be positive")
    return resolved_rank, resolved_alpha


__all__ = [
    "LORA_SUFFIXES",
    "read_lora_groups",
    "strip_adapter_prefix",
    "validate_lora_pair",
    "validate_lora_scale",
]
