# SPDX-FileCopyrightText: 2025 The Helios Team and The HuggingFace Team
# SPDX-FileCopyrightText: 2026 The Evoke Team
# SPDX-License-Identifier: Apache-2.0

"""Patch visibility masks for Helios-derived history conditioning."""

import torch

from worldfoundry.core.nn.blocks.volume import pad_for_3d_conv


def pool_history_visible_mask(mask: torch.Tensor | None, patch_size: tuple[int, int, int]) -> torch.Tensor | None:
    """Pool pixel-domain visibility mask to patch granularity via avg_pool3d."""
    if mask is None:
        return None
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"history visible mask must be 4D/5D, got {tuple(mask.shape)}")
    mask = pad_for_3d_conv(mask.float(), patch_size)
    return torch.nn.functional.avg_pool3d(mask, kernel_size=patch_size, stride=patch_size)


def resolve_history_keep_mask(keep_mask: torch.Tensor | None, threshold: float = 0.5) -> torch.Tensor | None:
    """Threshold patch occupancy mask to a bool keep mask; requires identical masks across batch."""
    if keep_mask is None:
        return None
    if keep_mask.ndim == 5:
        if keep_mask.shape[1] != 1:
            raise ValueError(f"history visible mask channel dimension must be 1, got {tuple(keep_mask.shape)}")
        keep_mask = keep_mask[:, 0]
    if keep_mask.ndim != 4:
        raise ValueError(f"history visible mask must reduce to [B,T,H,W], got {tuple(keep_mask.shape)}")
    keep_flat = keep_mask.flatten(1)
    if keep_flat.dtype != torch.bool:
        keep_flat = keep_flat >= float(threshold)
    if keep_flat.shape[0] == 1:
        return keep_flat[0]
    if not torch.equal(keep_flat, keep_flat[0:1].expand_as(keep_flat)):
        raise ValueError("history visible masking currently requires identical masks across the batch.")
    return keep_flat[0]
