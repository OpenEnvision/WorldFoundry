# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Original-frame-index video probing and RGB decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import VideoDecodeError


@dataclass(frozen=True)
class VideoInfo:
    total_frames: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class DecodedFrame:
    frame_idx: int
    rgb: Any


def fixed_stride_indices(total_frames: int, stride: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    return list(range(0, total_frames, stride))


class VideoReader:
    def probe(self, video_path: str) -> VideoInfo:
        import cv2

        path = str(Path(video_path).expanduser().resolve())
        capture = cv2.VideoCapture(path)
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or total_frames <= 0 or width <= 0 or height <= 0:
                raise VideoDecodeError(f"cannot probe video: {path}")
            return VideoInfo(
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
            )
        finally:
            capture.release()

    def read_indices(
        self,
        video_path: str,
        indices: list[int] | tuple[int, ...],
    ) -> list[DecodedFrame]:
        import cv2

        path = str(Path(video_path).expanduser().resolve())
        info = self.probe(path)
        for index in indices:
            if index < 0 or index >= info.total_frames:
                raise VideoDecodeError(f"requested frame {index} outside [0, {info.total_frames - 1}] for {path}")
        capture = cv2.VideoCapture(path)
        frames: list[DecodedFrame] = []
        try:
            previous_index: int | None = None
            for index in indices:
                # Consecutive all-frame consumers (notably Human Entity) should
                # decode sequentially. Seeking before every frame is orders of
                # magnitude slower for inter-frame codecs.
                if previous_index is None or index != previous_index + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise VideoDecodeError(f"cannot decode frame {index} from {path}")
                frames.append(
                    DecodedFrame(
                        frame_idx=int(index),
                        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    )
                )
                previous_index = int(index)
        finally:
            capture.release()
        return frames
