# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Low-poly RGB preprocessing for the released Matrix-Game 4 scene models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .runtime_video_operator import RuntimeVideoOperator


class MatrixGame4Operator(RuntimeVideoOperator):
    """Prepare video files, THWC uint8 arrays or signed BCTHW RGB tensors."""

    def __init__(self) -> None:
        super().__init__(generation_type="v2v")

    def prepare_video(self, video: Any) -> tuple[Any, float | None]:
        """Return signed BCTHW pixels and a file's source frame rate."""
        import numpy as np
        import torch
        from PIL import Image

        fps = None
        if isinstance(video, (str, Path)):
            from worldfoundry.core.media.codecs.video import read_video

            video, metadata = read_video(video)
            fps = float(metadata["fps"])
        if isinstance(video, torch.Tensor):
            pixels = video
            if pixels.ndim != 5 or tuple(pixels.shape[:2]) != (1, 3) or tuple(pixels.shape[-2:]) != (480, 864):
                raise ValueError("Matrix-Game 4 tensor input must be signed BCTHW [1,3,T,480,864]")
            if pixels.shape[2] < 1:
                raise ValueError("Matrix-Game 4 requires a nonempty video")
            if (
                not pixels.is_floating_point()
                or not torch.isfinite(pixels).all()
                or pixels.min() < -1
                or pixels.max() > 1
            ):
                raise ValueError("Matrix-Game 4 tensor RGB must be finite floating point in [-1,1]")
            return pixels, fps
        if isinstance(video, np.ndarray):
            if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
                raise ValueError("Matrix-Game 4 array input must be THWC uint8 RGB")
            frames = [Image.fromarray(frame) for frame in video]
        elif isinstance(video, Sequence) and not isinstance(video, (str, bytes)):
            frames = list(video)
            if not all(isinstance(frame, Image.Image) for frame in frames):
                raise TypeError("Matrix-Game 4 frame sequences must contain PIL RGB images")
        else:
            raise TypeError("Matrix-Game 4 requires a low-poly video path, RGB array or tensor")
        if not frames:
            raise ValueError("Matrix-Game 4 requires a nonempty video")
        from worldfoundry.base_models.diffusion_model.models.encoders.wan.control_video import resize_control_frame

        array = np.stack(
            [np.asarray(resize_control_frame(frame, height=480, width=864), dtype=np.uint8) for frame in frames]
        )
        pixels = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).float()
        return pixels * (2.0 / 255.0) - 1.0, fps
