# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Training-aligned PIL preprocessing for Wan low-poly control video."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def resize_control_frame(image: Image.Image, *, height: int, width: int) -> Image.Image:
    """Bilinearly resize to cover the target and apply a rounded center crop."""
    from PIL import Image

    image = image.convert("RGB")
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    size = (round(source_width * scale), round(source_height * scale))
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    left, top = round((size[0] - width) / 2), round((size[1] - height) / 2)
    return image.crop((left, top, left + width, top + height))
