"""Tensor, PIL, and video helpers for memory artifact normalization.

Memory records accept paths, PIL images, numpy arrays, or torch
tensors. This module converts those shapes into RGB PIL frames (or
frame lists) so :mod:`worldfoundry.core.memory.store` never has to
know about CHW vs HWC, ``[-1, 1]`` vs ``[0, 1]``, or video layouts.

Public helpers: :func:`to_pil_image`, :func:`extract_video_frames`,
:func:`extract_last_frame`, :func:`infer_content_type`, and
:func:`release_accelerator_cache`. Optional deps (numpy, imageio,
torch) are imported at use time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# Suffix tables — store never needs to know CHW vs HWC or video layouts
# ──────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


# ──────────────────────────────────────────────────────────────────────────
# Image / array coercion — optional numpy and torch imported at use time
# ──────────────────────────────────────────────────────────────────────────


def is_tensor_like(value: Any) -> bool:
    """Return True when *value* looks like a torch tensor."""
    return hasattr(value, "detach") or hasattr(value, "cpu") or hasattr(value, "permute")


def to_pil_image(value: Any) -> Image.Image:
    """Convert a path, array, or tensor to an RGB PIL image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return Image.open(path).convert("RGB")
        raise ValueError(f"{value!s} is not an image path.")

    array = to_numpy(value)
    if array.ndim == 2:
        pass
    elif array.ndim == 3:
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = move_channel_first_to_last(array)
    else:
        raise ValueError(f"Cannot convert array with shape {array.shape} to an image.")

    array = normalize_uint8_array(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return Image.fromarray(array).convert("RGB")


def to_numpy(value: Any):
    """Detach / CPU / ``.numpy()`` a tensor-like value, then ``np.asarray``."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise ValueError("Array conversion requires numpy.") from exc

    array_like = value
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        array_like = array_like.numpy()
    return np.asarray(array_like)


def normalize_uint8_array(array: Any):
    """Map float arrays in ``[-1, 1]`` or ``[0, 1]`` to ``uint8``; pass uint8 through."""
    import numpy as np

    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array.astype(np.uint8, copy=False)

    array = array.astype("float32")
    if array.size:
        min_value = float(np.nanmin(array))
        max_value = float(np.nanmax(array))
        if min_value >= -1.0 and max_value <= 1.0:
            array = (array + 1.0) * 127.5 if min_value < 0.0 else array * 255.0
    return np.clip(array, 0.0, 255.0).astype("uint8")


def move_channel_first_to_last(array: Any):
    """Move axis 0 to the end (CHW → HWC)."""
    import numpy as np

    return np.moveaxis(np.asarray(array), 0, -1)


# ──────────────────────────────────────────────────────────────────────────
# Video frames — path / sequence / [T,H,W,C]; last-frame is best-effort
# ──────────────────────────────────────────────────────────────────────────


def extract_video_frames(video_data: Any, *, copy: bool = True) -> list[Any]:
    """Return a list of frames from a path, sequence, or ``[T,H,W,C]`` array."""
    if isinstance(video_data, (str, Path)):
        path = Path(video_data).expanduser()
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            raise ValueError(f"{video_data!s} is not a video path.")
        try:
            import imageio
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise ValueError("Video path frame extraction requires imageio.") from exc
        reader = imageio.get_reader(str(path))
        try:
            return [frame.copy() if copy and hasattr(frame, "copy") else frame for frame in reader]
        finally:
            reader.close()

    if isinstance(video_data, (list, tuple)):
        return [frame.copy() if copy and hasattr(frame, "copy") else frame for frame in video_data]

    array = normalize_video_array(to_numpy(video_data))
    if array.ndim < 4:
        raise ValueError(f"Cannot identify a video frame sequence from shape {array.shape}.")
    return [frame.copy() if copy and hasattr(frame, "copy") else frame for frame in array]


def normalize_video_array(array: Any):
    """Squeeze a leading batch dim and move channels last when they are first."""
    import numpy as np

    array = np.asarray(array)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        return array
    if array.shape[-1] in (1, 3, 4):
        return array
    if array.shape[1] in (1, 3, 4):
        return np.moveaxis(array, 1, -1)
    if array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1)
    return array


def extract_last_frame(video_data: Any) -> Image.Image | None:
    """Best-effort last-frame PIL image; returns ``None`` on any failure."""
    try:
        frames = extract_video_frames(video_data)
    except Exception:  # noqa: BLE001 - selection should degrade gracefully.
        return None
    if not frames:
        return None
    return to_pil_image(frames[-1])


# ──────────────────────────────────────────────────────────────────────────
# Type inference and accelerator cache — degrade to "other" / no-op on ImportError
# ──────────────────────────────────────────────────────────────────────────


def infer_content_type(data: Any) -> str:
    """Infer a coarse content type (``image``, ``video``, ``text``, or ``other``)."""
    if isinstance(data, Image.Image):
        return "image"
    if isinstance(data, (str, Path)):
        suffix = Path(data).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return "text" if isinstance(data, str) else "other"
    try:
        array = to_numpy(data)
    except Exception:  # noqa: BLE001
        return "other"
    if array.ndim >= 4:
        return "video"
    if array.ndim in (2, 3):
        return "image"
    return "other"


def release_accelerator_cache() -> None:
    """Run GC and empty the CUDA cache when torch is available."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "extract_last_frame",
    "extract_video_frames",
    "infer_content_type",
    "is_tensor_like",
    "move_channel_first_to_last",
    "normalize_uint8_array",
    "normalize_video_array",
    "release_accelerator_cache",
    "to_numpy",
    "to_pil_image",
]
