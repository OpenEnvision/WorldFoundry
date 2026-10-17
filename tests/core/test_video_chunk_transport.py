# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Declared video semantics across NumPy/Torch and Studio transport."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest
import torch

from worldfoundry.core.media.codecs.video import save_image_or_video_tensor, video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk, VideoTensorLayout
from worldfoundry.studio.inference.execution import _normalize_frame_list, _safe_json
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result

pytestmark = pytest.mark.unit


def _layout(frames: np.ndarray, layout: str) -> Any:
    if layout == "frame-list":
        return list(frames)
    if layout in {"tchw", "btchw", "bvtchw"}:
        frames = frames.transpose(0, 3, 1, 2)
    elif layout == "bcthw":
        frames = frames.transpose(3, 0, 1, 2)
    if layout.startswith("b"):
        frames = frames[None]
    if layout.startswith("bv"):
        frames = frames[None]
    return frames


@pytest.mark.parametrize("layout", ["thwc", "tchw", "bthwc", "btchw", "bcthw", "bvthwc", "bvtchw"])
@pytest.mark.parametrize("frame_count", [1, 3, 4, 5])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("value_range, pixel", [("-1,1", 191), ("0,1", 127), ("0,255", 0)])
def test_explicit_layout_and_range_override_content_heuristics(
    layout: str,
    frame_count: int,
    backend: str,
    value_range: str,
    pixel: int,
) -> None:
    source = np.full((frame_count, 3, 4, 3), 0.5, dtype=np.float32)
    frames = _layout(source, layout)
    if backend == "torch":
        frames = torch.from_numpy(frames.copy())
    chunk = VideoChunk(frames=frames, layout=cast(VideoTensorLayout, layout), value_range=value_range)
    result = realtime_frames_from_result({"video_chunk": chunk, "video": frames})
    assert chunk.frame_count == len(result) == frame_count
    assert np.stack(result).shape == source.shape
    assert np.all(np.stack(result) == pixel)
    materialized = _normalize_frame_list(chunk)
    assert materialized is not None
    np.testing.assert_array_equal(np.stack(materialized), np.stack(result))
    if backend == "torch":
        assert torch.all(frames == 0.5)
    else:
        assert np.all(frames == 0.5)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_byte_frames_are_not_rescaled_and_axes_preserve_pixel_order(backend: str) -> None:
    source = np.arange(5 * 3 * 4 * 3, dtype=np.uint8).reshape(5, 3, 4, 3)
    frames = _layout(source, "bcthw")
    if backend == "torch":
        frames = torch.from_numpy(frames.copy())
    chunk = VideoChunk(frames=frames, layout="bcthw", value_range="0,255")
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), source)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_signed_frames_use_canonical_conversion(dtype: torch.dtype) -> None:
    chunk = VideoChunk(frames=torch.full((1, 3, 5, 8, 8), 0.5, dtype=dtype), layout="bcthw", value_range="-1,1")
    assert int(video_chunk_to_uint8_frames(chunk)[0, 0, 0, 0]) == 191


def test_typed_iterator_preserves_every_chunk_and_closes_producer() -> None:
    closed = []

    def chunks() -> Iterator[dict[str, Any]]:
        try:
            for pixel in (0.0, 0.5):
                yield {"video_chunk": VideoChunk(frames=[np.full((8, 8, 3), pixel)], value_range="-1,1")}
        finally:
            closed.append(True)

    frames = realtime_frames_from_result(chunks())
    assert [int(frame[0, 0, 0]) for frame in frames] == [127, 191]
    assert closed == [True]


def test_empty_typed_chunk_represents_buffering() -> None:
    chunk = VideoChunk(frames=[])
    assert realtime_frames_from_result(chunk) == []
    assert video_chunk_to_uint8_frames(chunk).shape[0] == 0


def test_typed_frame_list_preserves_prefetch_and_declared_range() -> None:
    events: list[tuple[str, int]] = []

    class Frame:
        def __init__(self, index: int) -> None:
            self.index = index

        def prefetch_to_numpy(self) -> None:
            events.append(("prefetch", self.index))

        def to_numpy(self) -> np.ndarray:
            events.append(("read", self.index))
            return np.full((8, 8, 3), 0.5)

    frames = realtime_frames_from_result(VideoChunk(frames=[Frame(0), Frame(1)], value_range="-1,1"))
    assert events == [("prefetch", 0), ("prefetch", 1), ("read", 0), ("read", 1)]
    assert np.all(np.stack(frames) == 191)


@pytest.mark.parametrize(
    "shape, layout, message",
    [
        ((2, 3, 5, 8, 8), "bcthw", "batch size"),
        ((1, 2, 5, 3, 8, 8), "bvtchw", "view count"),
        ((3, 5, 8, 8), "bcthw", "dimensions"),
    ],
)
def test_invalid_declared_layout_does_not_fall_back_to_guessing(
    shape: tuple[int, ...], layout: str, message: str
) -> None:
    chunk = VideoChunk(frames=np.zeros(shape, dtype=np.uint8), layout=cast(VideoTensorLayout, layout))
    with pytest.raises(ValueError, match=message):
        realtime_frames_from_result({"video_chunk": chunk, "video": np.zeros((5, 8, 8, 3), dtype=np.uint8)})


def test_artifact_conversion_uses_the_same_declared_pixels() -> None:
    from io import BytesIO

    from PIL import Image

    tensor = torch.full((1, 3, 1, 3, 4), 0.5)
    target = BytesIO()
    save_image_or_video_tensor(tensor, target, layout="bcthw", value_range="-1,1", image_format="PNG")
    target.seek(0)
    with Image.open(target) as image:
        actual = np.asarray(image)
    expected = video_chunk_to_uint8_frames(VideoChunk(frames=tensor, layout="bcthw", value_range="-1,1"))[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3, 4, 3) and np.all(actual == 191)


def test_manifest_records_video_semantics_without_serializing_pixels() -> None:
    chunk = VideoChunk(frames=torch.zeros(1, 3, 5, 8, 8), layout="bcthw", value_range="-1,1")
    assert _safe_json(chunk) == {
        "frames": {"shape": [1, 3, 5, 8, 8], "dtype": "torch.float32"},
        "layout": "bcthw",
        "value_range": "-1,1",
        "frame_count": 5,
        "metadata": {},
    }
