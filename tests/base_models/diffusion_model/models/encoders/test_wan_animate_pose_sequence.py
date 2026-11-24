# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Frame-to-frame body keypoint fallback in Wan Animate preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.animate import pose2d

pytestmark = pytest.mark.unit


def _load_metas(frames: np.ndarray, detection_lists: bool) -> list[dict[str, Any]] | None:
    if detection_lists:
        return pose2d.load_pose_metas_from_kp2ds_seq_list([[frame] for frame in frames], 320, 240)
    return pose2d.load_pose_metas_from_kp2ds_seq(frames, 320, 240)


@pytest.mark.parametrize("detection_lists", [False, True], ids=["sequence", "detection-lists"])
def test_invalid_body_frames_reuse_the_last_valid_body(detection_lists: bool) -> None:
    valid = np.arange(133 * 3, dtype=np.float32).reshape(133, 3) + 1
    missing = np.full_like(valid, -1)
    frames = np.stack([valid, missing, missing * 2, valid * 2, missing * 3])
    original = frames.copy()

    metas = _load_metas(frames, detection_lists)

    assert metas is not None
    assert len(metas) == len(frames)
    last_body = None
    for frame, meta in zip(frames, metas):
        normalized = frame.copy()
        normalized[:, 0] /= 320
        normalized[:, 1] /= 240
        body, left, right, face = pose2d.split_kp2ds_for_aa(normalized, ret_face=True)
        if body[:, :2].min(axis=1).max() >= 0:
            last_body = body
        assert last_body is not None
        np.testing.assert_array_equal(meta["keypoints_body"], last_body)
        np.testing.assert_array_equal(meta["keypoints_left_hand"], left)
        np.testing.assert_array_equal(meta["keypoints_right_hand"], right)
        np.testing.assert_array_equal(meta["keypoints_face"], face)
        assert (meta["width"], meta["height"]) == (320, 240)
    np.testing.assert_array_equal(frames, original)


@pytest.mark.parametrize("detection_lists", [False, True], ids=["sequence", "detection-lists"])
def test_initial_invalid_body_has_no_previous_frame(detection_lists: bool) -> None:
    frames = np.full((1, 133, 3), -1, dtype=np.float32)

    with pytest.raises(UnboundLocalError, match="last_kp2ds_body"):
        _load_metas(frames, detection_lists)


@pytest.mark.parametrize("detection_lists", [False, True], ids=["sequence", "detection-lists"])
def test_empty_pose_sequence_returns_no_metadata(detection_lists: bool) -> None:
    assert _load_metas(np.empty((0, 133, 3), dtype=np.float32), detection_lists) == []
