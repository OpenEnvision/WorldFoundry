# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Operator implementation for the Pi4 (pi^4) 4D geometry and motion model.

Preprocessing mirrors upstream ``yyfz/Pi4`` revision
``32dfbea9ce6268ea17446208a804710df9475c32`` (``pi4/io.py``): shared aspect-ratio
Lanczos resize to positive multiples of ``PATCH_SIZE = 16`` bounded by ``max_size``,
with frame sampling via ``stride`` and ``max_frames``, plus camera navigation tokens
mirroring :class:`~worldfoundry.operators.pi3_operator.Pi3Operator`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps

from .base_operator import BaseOperator

NAVIGATION_TEMPLATE: list[str] = [
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
    "camera_up",
    "camera_down",
    "camera_l",
    "camera_r",
    "camera_ul",
    "camera_ur",
    "camera_dl",
    "camera_dr",
    "camera_zoom_in",
    "camera_zoom_out",
]

NAVIGATION_DELTAS: dict[str, list[float]] = {
    "forward": [0.0, 0.0, -0.5, 0.0, 0.0],
    "backward": [0.0, 0.0, 0.5, 0.0, 0.0],
    "left": [-0.5, 0.0, 0.0, 0.0, 0.0],
    "right": [0.5, 0.0, 0.0, 0.0, 0.0],
    "forward_left": [-0.35, 0.0, -0.35, 0.0, 0.0],
    "forward_right": [0.35, 0.0, -0.35, 0.0, 0.0],
    "backward_left": [-0.35, 0.0, 0.35, 0.0, 0.0],
    "backward_right": [0.35, 0.0, 0.35, 0.0, 0.0],
    "camera_up": [0.0, 0.0, 0.0, -0.15, 0.0],
    "camera_down": [0.0, 0.0, 0.0, 0.15, 0.0],
    "camera_l": [0.0, 0.0, 0.0, 0.0, -0.15],
    "camera_r": [0.0, 0.0, 0.0, 0.0, 0.15],
    "camera_ul": [0.0, 0.0, 0.0, -0.1, -0.1],
    "camera_ur": [0.0, 0.0, 0.0, -0.1, 0.1],
    "camera_dl": [0.0, 0.0, 0.0, 0.1, -0.1],
    "camera_dr": [0.0, 0.0, 0.0, 0.1, 0.1],
    "camera_zoom_in": [0.0, 0.0, -0.3, 0.0, 0.0],
    "camera_zoom_out": [0.0, 0.0, 0.3, 0.0, 0.0],
}

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})
_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"})


class Pi4Operator(BaseOperator):
    """Perception and navigation operator for Pi4 4D geometry and tracking."""

    PATCH_SIZE = 16
    DEFAULT_MAX_SIZE = 512

    def __init__(
        self,
        operation_types: list[str] | None = None,
        interaction_template: list[str] | None = None,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        """Initialize the Pi4 operator with visual/action contracts and default max size."""
        if operation_types is None:
            operation_types = ["visual_instruction", "action_instruction"]
        if interaction_template is None:
            interaction_template = list(NAVIGATION_TEMPLATE)
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.max_size = int(max_size)
        self.interaction_template_init()

    def collect_paths(self, path: str | Path) -> list[str]:
        """Collect sorted image file paths from an image file, directory, or ``.txt`` list."""
        path_str = str(path)
        if os.path.isfile(path_str):
            if path_str.lower().endswith(".txt"):
                with open(path_str, encoding="utf-8") as handle:
                    return [line.strip() for line in handle if line.strip()]
            return [path_str]
        if os.path.isdir(path_str):
            return [
                os.path.join(path_str, name)
                for name in sorted(os.listdir(path_str))
                if not name.startswith(".") and os.path.splitext(name)[1].lower() in _IMAGE_EXTS
            ]
        raise FileNotFoundError(f"Input path does not exist: {path_str}")

    @staticmethod
    def _is_video(path: str | Path) -> bool:
        return os.path.splitext(str(path))[1].lower() in _VIDEO_EXTS

    @staticmethod
    def _is_image_file(path: str | Path) -> bool:
        return os.path.splitext(str(path))[1].lower() in _IMAGE_EXTS

    @staticmethod
    def _load_single_image(image_path: str | Path) -> np.ndarray:
        path_obj = Path(image_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with Image.open(path_obj) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))

    @staticmethod
    def _load_video_frames(
        video_path: str | Path,
        *,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> list[np.ndarray]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames: list[np.ndarray] = []
        frame_idx = 0
        try:
            while True:
                available, frame = capture.read()
                if not available:
                    break
                if frame_idx % stride == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                frame_idx += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        return frames

    @staticmethod
    def _compute_target_size(
        width: int,
        height: int,
        max_size: int = DEFAULT_MAX_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> tuple[int, int]:
        """Compute patch-16-aligned ``(target_width, target_height)`` preserving aspect ratio."""
        if width < 1 or height < 1:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
        if max_size < patch_size:
            raise ValueError(f"max_size must be at least {patch_size}, got {max_size}")
        scale = max_size / max(height, width)
        target_w = max(patch_size, round(width * scale / patch_size) * patch_size)
        target_h = max(patch_size, round(height * scale / patch_size) * patch_size)
        return int(target_w), int(target_h)

    @staticmethod
    def _to_uint8_rgb(img: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            img_uint8 = arr
        else:
            scale = 255.0 if arr.size == 0 or float(np.nanmax(arr)) <= 1.0 else 1.0
            img_uint8 = np.round(arr.astype(np.float64) * scale).clip(0, 255).astype(np.uint8)
        if img_uint8.ndim == 2:
            img_uint8 = np.stack([img_uint8] * 3, axis=-1)
        elif img_uint8.ndim == 3 and img_uint8.shape[-1] == 4:
            img_uint8 = img_uint8[..., :3]
        elif img_uint8.ndim != 3 or img_uint8.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image array, got shape {img_uint8.shape}")
        return img_uint8

    def images_to_tensor(
        self,
        images: Sequence[np.ndarray | Image.Image],
        device: str = "cuda",
        max_size: int | None = None,
    ) -> torch.Tensor:
        """Resize frames with Lanczos to a shared multiple-of-16 size and stack as ``[1, T, 3, H, W]``.

        Target dimensions come from the first frame using ``max_size`` (default ``self.max_size``)
        and ``PATCH_SIZE = 16``. Returned float32 tensor values lie in ``[0, 1]``.
        """
        if len(images) == 0:
            raise ValueError("No images provided")

        resolved_max_size = self.max_size if max_size is None else int(max_size)
        first = self._to_uint8_rgb(images[0])
        height, width = first.shape[:2]
        target_w, target_h = self._compute_target_size(
            width=width,
            height=height,
            max_size=resolved_max_size,
            patch_size=self.PATCH_SIZE,
        )

        resized_frames: list[np.ndarray] = []
        for idx, img in enumerate(images):
            frame_u8 = first if idx == 0 else self._to_uint8_rgb(img)
            pil_img = Image.fromarray(frame_u8, mode="RGB")
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized_frames.append(np.asarray(pil_img, dtype=np.uint8))

        stacked = np.stack(resized_frames, axis=0)  # [T, H, W, 3]
        tensor = torch.from_numpy(stacked.copy()).permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)
        return tensor.to(device)

    @staticmethod
    def _subsample_sequence(items: list[Any], *, stride: int, max_frames: int | None) -> list[Any]:
        sampled = items[::stride]
        if max_frames is not None:
            sampled = sampled[:max_frames]
        return sampled

    def process_perception(
        self,
        input_signal: (
            str | Path | np.ndarray | torch.Tensor | Image.Image | Sequence[str | Path | np.ndarray | Image.Image]
        ),
        *,
        stride: int = 1,
        max_frames: int | None = None,
        interval: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Load and subsample images or video frames into a list of RGB ``uint8`` NumPy arrays."""
        del kwargs
        resolved_stride = int(interval) if (interval is not None and interval > 0 and stride == 1) else int(stride)
        if resolved_stride < 1:
            raise ValueError(f"stride must be >= 1, got {resolved_stride}")
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        resolved_max = int(max_frames) if max_frames is not None else None

        if isinstance(input_signal, (str, Path)):
            path_obj = Path(input_signal)
            if not path_obj.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_signal}")
            if path_obj.is_dir() or str(path_obj).lower().endswith(".txt"):
                image_paths = self.collect_paths(path_obj)
                if not image_paths:
                    raise ValueError(f"No images found in: {input_signal}")
                selected_paths = self._subsample_sequence(image_paths, stride=resolved_stride, max_frames=resolved_max)
                return [self._load_single_image(p) for p in selected_paths]
            if self._is_image_file(path_obj):
                return [self._load_single_image(path_obj)]
            return self._load_video_frames(path_obj, stride=resolved_stride, max_frames=resolved_max)

        if isinstance(input_signal, Image.Image):
            return [self._to_uint8_rgb(input_signal)]

        if isinstance(input_signal, torch.Tensor):
            if input_signal.ndim == 3:
                imgs = input_signal.unsqueeze(0)
            elif input_signal.ndim == 4:
                imgs = input_signal
            elif input_signal.ndim == 5:
                imgs = input_signal[0]
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(input_signal.shape)}")
            frames = [self._to_uint8_rgb(imgs[i].permute(1, 2, 0).detach().cpu().numpy()) for i in range(imgs.shape[0])]
            return self._subsample_sequence(frames, stride=resolved_stride, max_frames=resolved_max)

        if isinstance(input_signal, np.ndarray):
            if input_signal.ndim == 3:
                frames = [self._to_uint8_rgb(input_signal)]
            elif input_signal.ndim == 4:
                frames = [self._to_uint8_rgb(input_signal[i]) for i in range(input_signal.shape[0])]
            else:
                raise ValueError(f"Unsupported array shape: {input_signal.shape}")
            return self._subsample_sequence(frames, stride=resolved_stride, max_frames=resolved_max)

        if isinstance(input_signal, Sequence) and not isinstance(input_signal, (bytes, bytearray)):
            items = list(input_signal)
            if not items:
                raise ValueError("Empty input list")
            first = items[0]
            if isinstance(first, (str, Path)):
                if len(items) == 1 and self._is_video(first):
                    return self._load_video_frames(first, stride=resolved_stride, max_frames=resolved_max)
                selected = self._subsample_sequence(items, stride=resolved_stride, max_frames=resolved_max)
                return [self._load_single_image(cast(str | Path, p)) for p in selected]
            if isinstance(first, (np.ndarray, Image.Image)):
                selected = self._subsample_sequence(items, stride=resolved_stride, max_frames=resolved_max)
                return [self._to_uint8_rgb(cast(np.ndarray | Image.Image, item)) for item in selected]
            raise ValueError(f"Unsupported sequence element type: {type(first)}")

        raise ValueError(f"Unsupported input type: {type(input_signal)}")

    def check_interaction(self, interaction: str) -> bool:
        """Validate that ``interaction`` belongs to ``interaction_template``."""
        if interaction not in self.interaction_template:
            raise ValueError(
                f"Interaction '{interaction}' not in interaction_template. "
                f"Available interactions: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction: str | list[str]) -> None:
        """Validate and queue one or more camera navigation interactions."""
        if isinstance(interaction, list):
            for item in interaction:
                self.check_interaction(item)
                self.current_interaction.append(item)
        else:
            self.check_interaction(interaction)
            self.current_interaction.append(interaction)

    def process_interaction(self, num_frames: int | None = None) -> list[float]:
        """Accumulate queued navigation interactions into ``[dx, dy, dz, theta_x, theta_z]``."""
        del num_frames
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        delta = [0.0, 0.0, 0.0, 0.0, 0.0]
        for interaction in self.current_interaction:
            self.interaction_history.append(interaction)
            step = NAVIGATION_DELTAS.get(interaction, [0.0, 0.0, 0.0, 0.0, 0.0])
            for axis in range(5):
                delta[axis] += step[axis]

        self.current_interaction = []
        return delta

    def process_interaction_single(self, interaction: str) -> list[float]:
        """Return camera delta for a single navigation token without mutating queued interactions."""
        self.check_interaction(interaction)
        self.interaction_history.append(interaction)
        return list(NAVIGATION_DELTAS.get(interaction, [0.0, 0.0, 0.0, 0.0, 0.0]))


__all__ = ["NAVIGATION_DELTAS", "NAVIGATION_TEMPLATE", "Pi4Operator"]
