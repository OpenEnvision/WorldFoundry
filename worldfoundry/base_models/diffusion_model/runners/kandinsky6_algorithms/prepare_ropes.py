"""Materialize RoPE tensors for DiT.forward (pipeline / export).

DiT.forward takes precomputed rotary matrices — no position→RoPE work inside the
exported graph. Visual grids may be cached at a max canvas and sliced.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def compute_rope1d(rope: nn.Module, length: int, device: torch.device | None = None) -> Tensor:
    """Build 1-D RoPE for positions ``0..length-1``."""
    if length < 1:
        raise ValueError(f"rope length must be positive, got {length}")
    dev = device if device is not None else next(rope.buffers()).device
    pos = torch.arange(length, device=dev)
    return rope(pos)


def compute_visual_rope(
    rope: nn.Module,
    shape: tuple[int, int, int],
    scale_factor: tuple[float, float, float],
    device: torch.device | None = None,
) -> Tensor:
    """Build 3-D RoPE grid for ``shape=(T,H,W)`` with prefix arange positions."""
    T, H, W = (int(shape[0]), int(shape[1]), int(shape[2]))
    if T < 1 or H < 1 or W < 1:
        raise ValueError(f"visual rope shape must be positive, got {shape}")
    dev = device if device is not None else next(rope.buffers()).device
    pos = [
        torch.arange(T, device=dev),
        torch.arange(H, device=dev),
        torch.arange(W, device=dev),
    ]
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    return rope((T, H, W), pos, scale)


class VisualRopeCache:
    """Eager max-canvas table: grow per axis, slice for smaller generations.

    Not used inside ``DiT.forward`` / ``torch.export`` — pipeline only.
    """

    def __init__(self) -> None:
        self._cache: Tensor | None = None
        self._shape: tuple[int, int, int] | None = None
        self._scale: tuple[float, float, float] | None = None

    def clear(self) -> None:
        self._cache = None
        self._shape = None
        self._scale = None

    @torch.no_grad()
    def get(
        self,
        rope: nn.Module,
        shape: tuple[int, int, int],
        scale_factor: tuple[float, float, float],
    ) -> Tensor:
        scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
        T, H, W = (int(shape[0]), int(shape[1]), int(shape[2]))
        device = next(rope.buffers()).device
        if (
            self._cache is not None
            and self._scale == scale
            and self._cache.device == device
            and self._shape is not None
            and self._shape[0] >= T
            and self._shape[1] >= H
            and self._shape[2] >= W
        ):
            return self._cache[:T, :H, :W]

        build_T, build_H, build_W = T, H, W
        if self._cache is not None and self._scale == scale and self._shape is not None:
            build_T = max(T, self._shape[0])
            build_H = max(H, self._shape[1])
            build_W = max(W, self._shape[2])

        self._cache = compute_visual_rope(rope, (build_T, build_H, build_W), scale, device=device)
        self._shape = (build_T, build_H, build_W)
        self._scale = scale
        return self._cache[:T, :H, :W]
