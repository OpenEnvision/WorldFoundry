"""MiDaS preprocessing for WonderJourney's tensor-based depth calls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.depth.midas.transforms import Resize


def _dpt_transform(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("MiDaS input must have shape [batch, 3, height, width]")

    height, width = image.shape[-2:]
    resize = Resize(
        size,
        size,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal",
    )
    target_width, target_height = resize.get_size(width, height)
    if (target_height, target_width) != (height, width):
        image = F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
    # DPT BeIT and DPT large use MiDaS mean/std 0.5 for RGB in [0, 1].
    return image.mul(2).sub(1)


def dpt_transform(image: torch.Tensor) -> torch.Tensor:
    return _dpt_transform(image, 384)


def dpt_512_transform(image: torch.Tensor) -> torch.Tensor:
    return _dpt_transform(image, 512)
