"""Image input normalization shared by image-based metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Load a path, PIL image, or array as a three-channel PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


__all__ = ["ImageInput", "load_rgb_image"]
