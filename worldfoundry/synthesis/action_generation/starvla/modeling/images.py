"""Image coercion shared by the StarVLA inference variants."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def to_pil_preserve(images: Any, scale_float: bool = True) -> Any:
    """Convert image leaves to RGB PIL while preserving list/tuple structure."""

    if isinstance(images, list):
        return [to_pil_preserve(item, scale_float=scale_float) for item in images]
    if isinstance(images, tuple):
        return tuple(to_pil_preserve(item, scale_float=scale_float) for item in images)
    if isinstance(images, Image.Image):
        return images.convert("RGB")

    array = np.asarray(images)
    if array.ndim != 3:
        raise ValueError(f"StarVLA image must be rank-3 HWC/CHW, got {array.shape}.")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"StarVLA image must have 1, 3, or 4 channels, got {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        if not scale_float:
            raise TypeError("Float StarVLA image supplied with scale_float=False.")
        if not np.isfinite(array).all():
            raise ValueError("StarVLA image contains NaN or infinite values.")
        finite_min = float(array.min()) if array.size else 0.0
        finite_max = float(array.max()) if array.size else 0.0
        if -1.0 <= finite_min < 0.0 and finite_max <= 1.0:
            array = (array + 1.0) * 127.5
        elif finite_min >= 0.0 and finite_max <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


__all__ = ["to_pil_preserve"]
