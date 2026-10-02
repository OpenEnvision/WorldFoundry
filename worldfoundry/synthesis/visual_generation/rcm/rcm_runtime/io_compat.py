# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Artifact writer matching the upstream ``imaginaire.utils.io`` contract.

The vendored rCM entrypoints call ``save_image_or_video`` with a CTHW (or BCTHW)
tensor. WorldFoundry does not vendor ``imaginaire``, so this module reproduces
that behaviour on top of the shared WorldFoundry video backend, keeping the
upstream normalization rules (``[-1, 1]`` auto-detected, uint8 accepted) intact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from worldfoundry.core.media.codecs.video import write_video


def _to_uint8_thwc(tensor: Tensor) -> np.ndarray:
    """Normalize a CTHW tensor to a uint8 THWC array."""

    if torch.is_floating_point(tensor):
        # Upstream heuristic: a visibly negative minimum means [-1, 1] input.
        if tensor.min() < -0.5:
            tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
    else:
        if tensor.dtype != torch.uint8:
            raise TypeError(f"Only float or uint8 tensors are supported, got {tensor.dtype}")
        tensor = tensor.float().div(255)

    array = rearrange(tensor.cpu().float().numpy() * 255, "c t h w -> t h w c")
    return (array + 0.5).astype(np.uint8)


def save_image_or_video(
    tensor: Tensor,
    save_path: str | os.PathLike[str],
    fps: int = 24,
    quality: Any = None,
    ffmpeg_params: Any = None,
) -> None:
    """Write a CTHW/BCTHW tensor as an image (T == 1) or a video.

    Args:
        tensor: Frames shaped ``(C, T, H, W)`` or ``(B, C, T, H, W)`` in ``[-1, 1]``,
            ``[0, 1]``, or uint8. A batch dimension keeps only the first item.
        save_path: Destination path. A missing extension becomes ``.jpg`` for a
            single frame and ``.mp4`` otherwise.
        fps: Frames per second for video output.
        quality: Optional encoder quality forwarded to the video backend.
        ffmpeg_params: Optional ffmpeg arguments forwarded to the video backend.
    """

    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape (C, T, H, W) or (B, C, T, H, W)")

    frames = _to_uint8_thwc(tensor)
    target = Path(os.fspath(save_path)).expanduser()

    if frames.shape[0] == 1:
        from PIL import Image

        if not target.suffix:
            target = target.with_suffix(".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[0], mode="RGB").save(target, format="JPEG", quality=85)
        return

    if not target.suffix:
        target = target.with_suffix(".mp4")
    extra: dict[str, Any] = {}
    if ffmpeg_params is not None:
        extra["ffmpeg_params"] = ffmpeg_params
    write_video(frames, target, fps=fps, quality=quality, **extra)


__all__ = ["save_image_or_video"]
