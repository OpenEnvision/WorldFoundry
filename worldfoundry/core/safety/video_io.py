# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video I/O shared by content-safety filters and postprocessors.

Safety classifiers and face-blur postprocessors need a tiny, imageio-
backed read/write path that does not pull Studio or the full
:mod:`worldfoundry.core.media.codecs.video` stack. This module:

- :func:`get_video_filepaths` — recursive ``mp4`` / ``avi`` / ``mov``
  discovery.
- :func:`read_video` — decode every frame into a
  :class:`VideoData` (``frames``, ``fps``, ``duration``).
- :func:`save_video` — write an ``[T,H,W,C]`` array with
  ``macro_block_size=1`` so odd resolutions are not silently padded.

Errors are raised as :class:`ValueError` with the path in the message
so a guardrail runner can attribute a failure to a specific file.
"""

from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np

from worldfoundry.core.distributed.runtime.logging import log

# ──────────────────────────────────────────────────────────────────────────
# Decode / encode — imageio ffmpeg only; odd resolutions stay unpadded
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoData:
    """Decoded video payload used by safety classifiers.

    Attributes:
        frames: ``[T, H, W, C]`` uint8 (or float) array from imageio.
        fps: Container frame rate; ``0.0`` when metadata is missing.
        duration: Container duration in seconds.
    """

    frames: np.ndarray
    fps: float
    duration: float


def get_video_filepaths(input_dir: str | Path) -> list[str]:
    """Return sorted ``.mp4`` / ``.avi`` / ``.mov`` paths under *input_dir*."""
    root = Path(input_dir)
    paths = sorted(str(path) for suffix in ("*.mp4", "*.avi", "*.mov") for path in root.rglob(suffix))
    log.debug("Found {} videos", len(paths))
    return paths


def read_video(filepath: str | Path) -> VideoData:
    """Decode *filepath* with the imageio ffmpeg plugin into :class:`VideoData`.

    Raises:
        ValueError: The file cannot be opened or contains no frames.
    """
    path = str(filepath)
    try:
        reader = imageio.get_reader(path, "ffmpeg")
    except Exception as error:
        raise ValueError(f"Failed to read video file: {path}") from error

    try:
        metadata = reader.get_meta_data()
        try:
            frame_count = reader.count_frames()
        except (AttributeError, NotImplementedError, RuntimeError):
            frame_count = round(float(metadata.get("duration", 0.0)) * float(metadata.get("fps", 0.0)))
        if frame_count <= 0:
            raise ValueError("video contains no decodable frames")
        frames = np.stack([reader.get_data(index) for index in range(frame_count)])
        return VideoData(
            frames=frames,
            fps=float(metadata.get("fps", 0.0)),
            duration=float(metadata.get("duration", 0.0)),
        )
    except Exception as error:
        raise ValueError(f"Failed to decode video file: {path}") from error
    finally:
        reader.close()


def save_video(filepath: str | Path, frames: np.ndarray, fps: float) -> None:
    """Write *frames* (``[T,H,W,C]``) to *filepath* at *fps*.

    Raises:
        ValueError: imageio failed to create or write the file.
    """
    path = str(filepath)
    writer = None
    try:
        writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
    except Exception as error:
        raise ValueError(f"Failed to save video file to {path}") from error
    finally:
        if writer is not None:
            writer.close()
