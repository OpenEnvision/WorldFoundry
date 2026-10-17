# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Declared frame memory retains raw output while selecting canonical RGB pixels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.synthesis.visual_generation.memory.stream import SceneStateMemory, VisualFrameMemory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("value_range,pixel", [("-1,1", 191), ("0,1", 127), ("0,255", 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frame_memory_uses_declared_layout_range_and_preserves_raw_aliases(
    count: int, value_range: str, pixel: int, dtype: torch.dtype
) -> None:
    raw = torch.full((1, 3, count, 3, 4), 0.5, dtype=dtype)
    original = raw.clone()
    memory = VisualFrameMemory(model_id="declared")
    result: dict[str, Any] = {
        "video": raw,
        "video_chunk": VideoChunk(frames=raw, layout="bcthw", value_range=value_range),
    }
    memory.record(result)
    assert memory.latest_result is result and memory.current_video is raw
    assert memory.select(mode="dynamic") is raw
    assert len(memory.all_frames) == count and memory.storage[-1]["content"] is raw
    np.testing.assert_array_equal(np.stack(memory.all_frames), np.full((count, 3, 4, 3), pixel, np.uint8))
    np.testing.assert_array_equal(np.asarray(memory.select()), np.full((3, 4, 3), pixel, np.uint8))
    torch.testing.assert_close(raw, original, rtol=0, atol=0)


def test_scene_state_and_camera_history_remain_separate_from_declared_pixels() -> None:
    memory = SceneStateMemory(model_id="scene")
    scene = {"scene_name": "scene", "intrinsics": np.eye(3)}
    memory.record({"scene_context": scene})
    poses = [np.eye(4), np.eye(4)]
    frames = [Image.new("RGB", (4, 3), "red"), Image.new("RGB", (4, 3), "blue")]
    result = {
        "frames": frames,
        "scene_context": scene,
        "pose_history": poses,
        "video_chunk": VideoChunk(frames=frames, layout="frame-list", value_range="0,255"),
    }
    memory.record(result)
    assert memory.select(context_query="state") is scene and memory.current_scene_context is scene
    assert memory.current_video is frames and memory.pose_history == poses
    assert memory.latest_result is result and memory.current_image is not None
    assert memory.current_image.getpixel((0, 0)) == (0, 0, 255)


def test_empty_declared_chunk_preserves_the_conditioning_frame_without_recording_frames() -> None:
    memory = VisualFrameMemory()
    memory.record(Image.new("RGB", (4, 3), "blue"))
    last_image = memory.current_image
    empty = np.empty((0, 3, 4, 3), np.uint8)
    result = {"video": empty, "video_chunk": VideoChunk(frames=empty, layout="thwc", value_range="0,255")}
    memory.record(result)
    assert memory.current_image is last_image and memory.all_frames == []
    assert memory.latest_result is result


def test_declared_tail_selection_can_omit_full_frame_history_and_reset() -> None:
    memory = VisualFrameMemory()
    frames = np.full((5, 3, 4, 3), 128, np.uint8)
    memory.record(
        {"video": frames, "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255")},
        record_frames=False,
    )
    assert memory.all_frames == [] and memory.current_video is frames
    assert memory.current_image is not None and memory.current_image.getpixel((0, 0)) == (128, 128, 128)
    memory.manage(action="reset")
    assert memory.current_video is memory.latest_result is memory.current_image is None
    assert memory.storage == []
