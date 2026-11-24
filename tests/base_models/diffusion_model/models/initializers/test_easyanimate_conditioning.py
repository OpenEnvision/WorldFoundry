# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""EasyAnimate conditioning keeps frame placement and branch-specific mask dtypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.initializers.easyanimate_conditioning import (
    get_image_to_video_latent,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("start_kind", ["file", "list"])
@pytest.mark.parametrize("end_kind", [None, "file", "list"])
def test_conditioning_frame_placement_and_mask_dtype(tmp_path: Path, start_kind: str, end_kind: str | None) -> None:
    start_image = Image.fromarray(np.full((5, 7, 3), 100, dtype=np.uint8))
    end_image = Image.fromarray(np.full((5, 7, 3), 200, dtype=np.uint8))
    start_path, end_path = tmp_path / "start.png", tmp_path / "end.png"
    start_image.save(start_path)
    end_image.save(end_path)
    start = str(start_path) if start_kind == "file" else [start_image, start_image]
    end = None if end_kind is None else str(end_path) if end_kind == "file" else [end_image]

    video, mask, clip_image = get_image_to_video_latent(start, end, 5, (4, 6))

    assert video.shape == (1, 3, 5, 4, 6)
    assert video.dtype == torch.float32
    assert mask.shape == (1, 1, 5, 4, 6)
    assert mask.dtype == (torch.float32 if end is None else torch.uint8)
    start_frames = 1 if start_kind == "file" else 2
    assert torch.count_nonzero(mask[:, :, :start_frames]) == 0
    expected_mask = torch.full_like(mask, 255)
    expected_mask[:, :, :start_frames] = 0
    expected_video = torch.full_like(video, 100 / 255)
    if end is not None:
        expected_mask[:, :, -1] = 0
        expected_video[:, :, -1] = 200 / 255
    torch.testing.assert_close(mask, expected_mask, atol=0, rtol=0)
    torch.testing.assert_close(video, expected_video, atol=0, rtol=0)
    assert clip_image is not None
    assert clip_image.size == (6, 4)
    assert np.all(np.array(clip_image) == 100)


def test_multiple_end_frames_keep_the_reference_assignment_error() -> None:
    image = Image.fromarray(np.full((5, 7, 3), 100, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="expanded size of the tensor"):
        get_image_to_video_latent([image], [image, image], 5, (4, 6))


def test_unconditional_video_respects_default_float_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        video, mask, _ = get_image_to_video_latent(None, None, 5, (4, 6))
        assert video.dtype == mask.dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_unconditional_video_has_float_mask_and_no_clip_image() -> None:
    video, mask, clip_image = get_image_to_video_latent(None, None, 5, (4, 6))
    assert video.shape == (1, 3, 5, 4, 6)
    assert torch.count_nonzero(video) == 0
    assert torch.all(mask == 255)
    assert mask.dtype == torch.float32
    assert clip_image is None
