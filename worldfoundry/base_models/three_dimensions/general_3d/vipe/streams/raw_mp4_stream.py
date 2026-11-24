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

from pathlib import Path

import cv2
import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.base import (
    ProcessedVideoStream,
    StreamList,
    VideoFrame,
    VideoStream,
)


class RawMp4Stream(VideoStream):
    """
    A video stream from a raw mp4 file, using opencv.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None) -> None:
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.stem

        # Read metadata
        vcap = cv2.VideoCapture(str(self.path))
        self._width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = vcap.get(cv2.CAP_PROP_FPS)
        _n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vcap.release()

        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step
        self._fps = _fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.vcap = cv2.VideoCapture(self.path)
        self.current_frame_idx = -1
        return self

    def __next__(self) -> VideoFrame:
        while True:
            ret, frame = self.vcap.read()
            self.current_frame_idx += 1

            if not ret:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx >= self.end:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx < self.start:
                continue

            if (self.current_frame_idx - self.start) % self.step == 0:
                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        return VideoFrame(raw_frame_idx=self.current_frame_idx, rgb=frame_rgb)


class RawMP4StreamList(StreamList):
    """Raw stream list implementation."""

    def __init__(self, base_path: str, frame_start: int, frame_end: int, frame_skip: int, cached: bool = False) -> None:
        super().__init__()
        if Path(base_path).is_file():
            self.mp4_sequences = [Path(base_path)]
            assert self.mp4_sequences[0].suffix == ".mp4", "Only mp4 files are accepted."
        else:
            self.mp4_sequences = sorted(Path(base_path).glob("*.mp4"))
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached

    def __len__(self) -> int:
        return len(self.mp4_sequences)

    def __getitem__(self, index: int) -> VideoStream:
        stream: VideoStream = RawMp4Stream(self.mp4_sequences[index], seek_range=self.frame_range)
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading video", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return self.mp4_sequences[index].stem
