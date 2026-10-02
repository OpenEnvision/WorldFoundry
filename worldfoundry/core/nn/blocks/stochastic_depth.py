"""Stochastic-depth helpers shared by ViT-style transformer blocks.

Subset-batch residual add used when DropPath would otherwise keep a
full-batch mask. Training only; inference is a no-op add.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────
# Subset-batch residual — cheaper than a full-batch DropPath mask at high p
# ──────────────────────────────────────────────────────────────────────────


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[..., Tensor],
    sample_drop_ratio: float = 0.0,
    pos: Optional[Tensor] = None,
    **residual_kwargs: Any,
) -> Tensor:
    """Apply a residual branch on a random batch subset and scale back to full batch size."""

    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    if pos is not None:
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos, **residual_kwargs)
    elif residual_kwargs:
        residual = residual_func(x_subset, **residual_kwargs)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    # Scale the subset residual so the full-batch expectation matches an un-dropped add.
    residual_scale_factor = b / sample_subset_size

    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


# ──────────────────────────────────────────────────────────────────────────
# Shared index / scale helpers — LayerScale uses xformers scaled_index_add
# ──────────────────────────────────────────────────────────────────────────


def get_branges_scales(x: Tensor, sample_drop_ratio: float = 0.0) -> tuple[Tensor, float]:
    """Return the batch indices and scale factor for stochastic depth."""

    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(
    x: Tensor,
    brange: Tensor,
    residual: Tensor,
    residual_scale_factor: float,
    scaling_vector: Optional[Tensor] = None,
) -> Tensor:
    """Scatter-add a residual onto selected batch rows, optionally with layer-scale."""

    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        return torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)

    from xformers.ops import scaled_index_add

    return scaled_index_add(
        x,
        brange,
        residual.to(dtype=x.dtype),
        scaling=scaling_vector,
        alpha=residual_scale_factor,
    )


__all__ = [
    "add_residual",
    "drop_add_residual_stochastic_depth",
    "get_branges_scales",
]
