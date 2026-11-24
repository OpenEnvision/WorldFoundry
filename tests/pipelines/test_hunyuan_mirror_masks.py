# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-frame filter mask assembly for HunyuanWorld-Mirror."""

from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_mirror import HunyuanMirrorPipeline

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("frame_count", [0, 1, 3])
@pytest.mark.parametrize("confidence_mask", [False, True])
@pytest.mark.parametrize("sky_mask", [False, True])
def test_filter_masks_keep_frame_order_and_require_a_frame(
    frame_count: int, confidence_mask: bool, sky_mask: bool
) -> None:
    confidences = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 2, 3)
    depths = np.zeros((*confidences.shape, 1), dtype=np.float32)
    normals = np.zeros((*confidences.shape, 3), dtype=np.float32)
    non_sky = np.indices(confidences.shape).sum(axis=0) % 2 == 0
    pipeline = HunyuanMirrorPipeline(device="cpu")

    def build_mask() -> np.ndarray:
        return pipeline.create_filter_mask(
            confidences,
            depths,
            normals,
            non_sky,
            confidence_percentile=50.0,
            apply_confidence_mask=confidence_mask,
            apply_sky_mask=sky_mask,
            apply_edge_mask=False,
        )

    if frame_count == 0:
        with pytest.raises(IndexError):
            build_mask()
        return

    result = build_mask()

    expected = np.ones(confidences.shape, dtype=bool)
    if confidence_mask:
        expected &= confidences >= np.quantile(confidences, 0.5, axis=(1, 2), keepdims=True)
    if sky_mask:
        expected &= non_sky
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)
