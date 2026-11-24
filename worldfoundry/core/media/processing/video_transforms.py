# SPDX-FileCopyrightText: 2026 OpenEnvision
# Copyright 2024 Vchitect/Latte
# SPDX-License-Identifier: Apache-2.0
# Adapted from Latte/Open-Sora video transforms.
"""Four-dimensional video tensor conversion, spatial crops and antialiased resize."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def is_tensor_video_clip(clip: torch.Tensor) -> bool:
    """Require a four-dimensional tensor without changing its layout."""
    if not torch.is_tensor(clip):
        raise TypeError(f"clip should be Tensor. Got {type(clip)}")
    if clip.ndimension() != 4:
        raise ValueError(f"clip should be 4D. Got {clip.dim()}D")
    return True


def to_tensor(clip: torch.Tensor) -> torch.Tensor:
    """Convert uint8 video to FP32 [0, 1], preserving its existing axis order."""
    is_tensor_video_clip(clip)
    if clip.dtype != torch.uint8:
        raise TypeError(f"clip tensor should have data type uint8. Got {clip.dtype}")
    return clip.float() / 255.0


def normalize(clip: torch.Tensor, mean: Sequence[float], std: Sequence[float], inplace: bool = False) -> torch.Tensor:
    """Normalize a CTHW video along its first axis, optionally in place."""
    is_tensor_video_clip(clip)
    if not inplace:
        clip = clip.clone()
    mean_tensor = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std_tensor = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean_tensor[:, None, None, None]).div_(std_tensor[:, None, None, None])
    return clip


def crop(clip: torch.Tensor, i: int, j: int, h: int, w: int) -> torch.Tensor:
    """Slice the last two spatial axes of a four-dimensional video tensor."""
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]


def random_shift_crop(clip: torch.Tensor) -> torch.Tensor:
    """Crop a short-edge square, consuming one Torch RNG draw for each spatial axis."""
    is_tensor_video_clip(clip)
    h, w = clip.size(-2), clip.size(-1)
    short_edge = h if h <= w else w
    th, tw = short_edge, short_edge
    i = int(torch.randint(0, h - th + 1, size=(1,)).item())
    j = int(torch.randint(0, w - tw + 1, size=(1,)).item())
    return crop(clip, i, j, th, tw)


def resize(clip: torch.Tensor, target_size: Sequence[int], interpolation_mode: str) -> torch.Tensor:
    """Resize spatial axes with aligned corners and antialiasing."""
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return F.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)


def resize_scale(clip: torch.Tensor, target_size: Sequence[int], interpolation_mode: str) -> torch.Tensor:
    """Scale both spatial axes using the requested short edge and aligned corners."""
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    height, width = clip.size(-2), clip.size(-1)
    scale = target_size[0] / min(height, width)
    return F.interpolate(clip, scale_factor=scale, mode=interpolation_mode, align_corners=True, antialias=True)
