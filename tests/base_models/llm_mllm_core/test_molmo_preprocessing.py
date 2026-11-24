# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Molmo patch ordering and centered pooling padding are shared across vision preprocessors."""

import numpy as np
import pytest

from worldfoundry.base_models.llm_mllm_core.mllm.molmo.preprocessing import (
    arange_for_pooling,
    batch_pixels_to_patches,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("channels", [None, 3])
def test_batch_pixels_to_patches_preserves_patch_and_pixel_order(channels: int | None) -> None:
    shape = (2, 4, 6) if channels is None else (2, 4, 6, channels)
    pixels = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.stack(
        [pixels[:, h : h + 2, w : w + 2].reshape(2, -1) for h in range(0, 4, 2) for w in range(0, 6, 2)],
        axis=1,
    )
    np.testing.assert_array_equal(batch_pixels_to_patches(pixels, 2), expected)


def test_arange_for_pooling_centers_odd_padding_with_minus_one() -> None:
    indices = np.arange(12).reshape(3, 4)
    expected = np.array(
        [[[-1, 0, 1, -1, 4, 5], [2, 3, -1, 6, 7, -1]], [[-1, 8, 9, -1, -1, -1], [10, 11, -1, -1, -1, -1]]]
    )
    np.testing.assert_array_equal(arange_for_pooling(indices, 2, 3), expected)
