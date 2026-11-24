"""Torch helpers required by MolmoBot inference."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def move_to_device(value: T, device: torch.device) -> T:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}  # type: ignore[return-value]
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)  # type: ignore[return-value]
    return value


def ensure_finite_(
    tensor: torch.Tensor,
    check_neg_inf: bool = True,
    check_pos_inf: bool = False,
) -> None:
    if check_neg_inf:
        tensor.masked_fill_(tensor == float("-inf"), torch.finfo(tensor.dtype).min)
    if check_pos_inf:
        tensor.masked_fill_(tensor == float("inf"), torch.finfo(tensor.dtype).max)


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """Non-persistent device tensor cache for masks and RoPE values."""


__all__ = [
    "BufferCache",
    "barrier",
    "ensure_finite_",
    "get_global_rank",
    "get_local_rank",
    "is_distributed",
    "move_to_device",
]
