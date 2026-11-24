"""Input packing and checkpoint-paired normalization for LDA-1B inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LDA-1B image path does not exist: {path}")
        with Image.open(path) as source:
            return source.convert("RGB")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"LDA-1B image must be HWC or CHW, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(finite.max()) <= 1.0:
            finite = finite * 255.0
        array = finite
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"LDA-1B image must have 3 RGB channels, got shape {array.shape}")
    return Image.fromarray(array, mode="RGB")


def temporal_images(value: Any) -> list[Image.Image]:
    if isinstance(value, Mapping):
        for key in ("rgb", "color", "image", "images", "frames"):
            if key in value:
                return temporal_images(value[key])
        raise ValueError("LDA-1B image mapping has no rgb/color/image/images/frames field")
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            return [as_rgb_image(frame) for frame in value]
        return [as_rgb_image(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output: list[Image.Image] = []
        for item in value:
            output.extend(temporal_images(item))
        return output
    return [as_rgb_image(value)]


def select_history(frames: Sequence[Image.Image], steps: int) -> list[Image.Image]:
    if not frames:
        raise ValueError("LDA-1B requires at least one observation image")
    if steps <= 0:
        raise ValueError("LDA-1B observation history length must be positive")
    selected = list(frames[-steps:])
    while len(selected) < steps:
        selected.insert(0, selected[0])
    return selected


def pack_parts(value: Any, parts: Sequence[Mapping[str, Any]], *, label: str) -> np.ndarray:
    expected = sum(int(part["dim"]) for part in parts)
    if isinstance(value, Mapping):
        arrays = []
        leading: int | None = None
        for part in parts:
            key, dim = str(part["key"]), int(part["dim"])
            if key not in value:
                raise ValueError(f"LDA-1B {label} mapping is missing {key!r}")
            array = np.asarray(value[key], dtype=np.float32)
            if array.shape[-1] != dim:
                raise ValueError(f"LDA-1B {label} field {key!r} expects {dim} values, got {array.shape}")
            array = array.reshape(-1, dim)
            leading = array.shape[0] if leading is None else leading
            if array.shape[0] != leading:
                raise ValueError(f"LDA-1B {label} fields have inconsistent history lengths")
            arrays.append(array)
        return np.concatenate(arrays, axis=-1)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[-1] != expected:
        raise ValueError(f"LDA-1B {label} expects shape (history, {expected}) or ({expected},), got {array.shape}")
    return array


def state_sincos(
    raw_state: np.ndarray,
    *,
    steps: int,
    expected_dim: int,
    part_widths: Sequence[int] | None = None,
) -> np.ndarray:
    selected = raw_state[-steps:]
    while selected.shape[0] < steps:
        selected = np.concatenate((selected[:1], selected), axis=0)
    if part_widths is not None:
        widths = [int(width) for width in part_widths]
        if sum(widths) != selected.shape[-1]:
            raise ValueError(
                f"LDA-1B state part widths sum {sum(widths)} does not match raw state width {selected.shape[-1]}"
            )
        offset = 0
        chunks: list[np.ndarray] = []
        for width in widths:
            part = selected[..., offset : offset + width]
            chunks.append(np.sin(part))
            chunks.append(np.cos(part))
            offset += width
        encoded = np.concatenate(chunks, axis=-1).astype(np.float32)
    else:
        encoded = np.concatenate((np.sin(selected), np.cos(selected)), axis=-1).astype(np.float32)
    if encoded.shape[-1] != expected_dim:
        raise ValueError(f"LDA-1B sin/cos state width {encoded.shape[-1]} does not match model width {expected_dim}")
    return encoded


def denormalize_minmax(
    normalized: np.ndarray,
    stats: Mapping[str, Any],
    *,
    expected_dim: int,
) -> np.ndarray:
    low = np.asarray(stats.get("min"), dtype=np.float32).reshape(-1)
    high = np.asarray(stats.get("max"), dtype=np.float32).reshape(-1)
    mask = np.asarray(stats.get("mask", np.ones(expected_dim)), dtype=bool).reshape(-1)
    if low.size != expected_dim or high.size != expected_dim or mask.size != expected_dim:
        raise ValueError(
            "LDA-1B action statistics do not match the released raw action layout: "
            f"expected {expected_dim}, got min={low.size}, max={high.size}, mask={mask.size}"
        )
    clipped = np.clip(np.asarray(normalized, dtype=np.float32), -1.0, 1.0)
    raw = (clipped + 1.0) * 0.5 * (high - low) + low
    return np.where(mask, raw, clipped).astype(np.float32)


__all__ = [
    "as_rgb_image",
    "denormalize_minmax",
    "pack_parts",
    "select_history",
    "state_sincos",
    "temporal_images",
]
