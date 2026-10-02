"""Preprocessing helpers for Wan / MultiWorld VGGT environment encoders.

VGGT (and the Wan env-obs adapters that wrap it) expect a batched
``float32`` RGB tensor at a ViT-friendly size (default 224, sides
rounded to multiples of 14). This module owns that resize/crop/pad
contract so model families do not each reimplement letterboxing:

- :func:`preprocess_pil_for_vggt` — one RGB PIL image.
- :func:`load_and_preprocess_images` — paths, optional stereo
  ``left`` / ``right`` crop, then stack + pad to a common H×W.
- :func:`load_and_preprocess_videos` — OpenCV frame sampling with
  stride / start / target-frame count, same view + resize rules.

``mode="crop"`` center-crops a too-tall resize; ``mode="pad"`` pads
with white. Video loading requires optional ``opencv-python``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ViewMode = Literal["full", "left", "right"]
ResizeMode = Literal["crop", "pad"]

# ──────────────────────────────────────────────────────────────────────────
# VGGT letterbox — long side = target; short side rounded to multiples of 14
# ──────────────────────────────────────────────────────────────────────────


def _normalize_view(view: str) -> ViewMode:
    """Reject unknown stereo views so a typo cannot silently return a full frame."""

    if view not in {"full", "left", "right"}:
        raise ValueError("return_view must be one of 'full', 'left', or 'right'")
    return view  # type: ignore[return-value]


def _resize_for_vggt(img: Image.Image, *, target_size: int, mode: ResizeMode, source_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Resize then crop/pad; *source_size* keeps stereo FOV math on the uncropped frame."""

    width, height = source_size or img.size
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be either 'crop' or 'pad'")

    if width >= height:
        new_width = target_size
        # ViT patch grid is 14×14; rounding after scale keeps aspect close.
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        tensor = tensor[:, start_y : start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - tensor.shape[1]
        w_padding = target_size - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

    return tensor


def preprocess_pil_for_vggt(
    pil_image: Image.Image,
    *,
    target_size: int = 224,
    mode: ResizeMode = "pad",
) -> torch.Tensor:
    """Preprocess one PIL image for the MultiWorld VGGT environment encoder.

    Converts to RGB, resizes so the long side is *target_size* (short
    side rounded to a multiple of 14), then crops or pads per *mode*.
    Returns ``[3, H, W]`` in ``[0, 1]``.
    """

    img = pil_image.convert("RGB")
    return _resize_for_vggt(img, target_size=target_size, mode=mode)


def _open_rgb(path: str | Path) -> Image.Image:
    """Load RGB; composite RGBA onto white so alpha does not become black."""

    img = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _select_view(img: Image.Image, view: ViewMode) -> tuple[Image.Image, tuple[int, int]]:
    """Crop left/right stereo halves; return the *original* size for letterbox FOV."""

    width, height = img.size
    if view == "left":
        return img.crop((0, 0, width // 2, height)), (width, height)
    if view == "right":
        return img.crop((width // 2, 0, width, height)), (width, height)
    return img, (width, height)


def _pad_to_common_shape(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """White-pad mixed H×W tensors so they can stack into one batch."""

    shapes = {(tensor.shape[1], tensor.shape[2]) for tensor in tensors}
    if len(shapes) <= 1:
        return tensors
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded = []
    for tensor in tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
        padded.append(tensor)
    return padded


def load_and_preprocess_images(
    image_path_list: Sequence[str | Path],
    mode: ResizeMode = "crop",
    return_view: str = "full",
    target_size: int = 224,
) -> torch.Tensor:
    """Load image paths and produce a batched VGGT env-observation tensor.

    Args:
        image_path_list: One or more image files. RGBA is composited on
            white.
        mode: ``"crop"`` or ``"pad"`` after the 14-aligned resize.
        return_view: ``"full"``, or ``"left"`` / ``"right"`` half of a
            side-by-side stereo frame (crop uses the *original* size so
            the 14-grid stays consistent).
        target_size: Long-side target in pixels.

    Returns:
        ``[N, 3, H, W]`` float32 tensor, padded to a common spatial size.
    """

    if not image_path_list:
        raise ValueError("At least 1 image is required")

    view = _normalize_view(return_view)
    tensors = []
    for image_path in image_path_list:
        img = _open_rgb(image_path)
        img, original_size = _select_view(img, view)
        tensors.append(_resize_for_vggt(img, target_size=target_size, mode=mode, source_size=original_size))
    return torch.stack(_pad_to_common_shape(tensors))


def _read_video_frames(path: str | Path, *, start_point: int, frame_stride: int, target_frames: int | None) -> list[np.ndarray]:
    """Sample RGB frames via OpenCV; missing cv2 or an empty file raises, never returns []."""

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Video environment observations require the optional opencv-python package"
        ) from error

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {path}")
        frame_index = max(0, int(start_point))
        while frame_index < total_frames and (target_frames is None or len(frames) < target_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_index += max(1, int(frame_stride))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames loaded from video: {path}")
    return frames


def _select_frame_view(frame: np.ndarray, view: ViewMode) -> Image.Image:
    """Stereo crop for a video frame; unlike images, FOV uses the cropped size."""

    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size
    if view == "left":
        return img.crop((0, 0, width // 2, height))
    if view == "right":
        return img.crop((width // 2, 0, width, height))
    return img


def load_and_preprocess_videos(
    video_path_list: Sequence[str | Path],
    mode: ResizeMode = "crop",
    target_frames: int | None = None,
    frame_stride: int = 1,
    return_view: str = "full",
    start_point: int = 0,
    target_size: int = 224,
) -> torch.Tensor:
    """Load videos and produce a batched VGGT env-observation tensor.

    Samples frames with OpenCV (``start_point``, ``frame_stride``,
    optional ``target_frames``), applies the same view + resize rules as
    :func:`load_and_preprocess_images`, and stacks to
    ``[N, T, 3, H, W]``. Videos of different spatial sizes are padded
    to the max H×W in the batch.

    Raises:
        ModuleNotFoundError: ``opencv-python`` is not installed.
        ValueError: A path is empty or yields no frames.
    """

    if not video_path_list:
        raise ValueError("At least 1 video is required")

    view = _normalize_view(return_view)
    videos = []
    for video_path in video_path_list:
        frames = _read_video_frames(
            video_path,
            start_point=start_point,
            frame_stride=frame_stride,
            target_frames=target_frames,
        )
        tensors = [
            preprocess_pil_for_vggt(_select_frame_view(frame, view), target_size=target_size, mode=mode)
            for frame in frames
        ]
        videos.append(torch.stack(_pad_to_common_shape(tensors)))

    shapes = {(video.shape[2], video.shape[3]) for video in videos}
    if len(shapes) <= 1:
        return torch.stack(videos)

    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded_videos = []
    for video in videos:
        h_padding = max_height - video.shape[2]
        w_padding = max_width - video.shape[3]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
        padded_videos.append(video)
    return torch.stack(padded_videos)
