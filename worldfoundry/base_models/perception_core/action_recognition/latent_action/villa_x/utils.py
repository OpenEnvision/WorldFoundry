"""Preprocessing helpers required by the Villa-X encoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from einops import rearrange
from torchvision.transforms import Resize


def flatten_internal(function: Callable, flatten_ndim: int = 3) -> Callable:
    def wrapper(values: torch.Tensor, *args, **kwargs):
        leading_shape = values.shape[:-flatten_ndim]
        values = values.reshape(-1, *values.shape[-flatten_ndim:])
        values = function(values, *args, **kwargs)
        return values.reshape(*leading_shape, *values.shape[-values.ndim + 1 :])

    return wrapper


@flatten_internal
def resize(images: torch.Tensor, size: tuple[int, int] | int) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    return Resize(size)(images)


def hwc2chw(images: torch.Tensor) -> torch.Tensor:
    return rearrange(images, "... h w c -> ... c h w")


def _chw2hwc(images: torch.Tensor) -> torch.Tensor:
    return rearrange(images, "... c h w -> ... h w c")


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    images = _chw2hwc(images).to(torch.float32) / 255
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)
    return hwc2chw((images - mean) / std)
