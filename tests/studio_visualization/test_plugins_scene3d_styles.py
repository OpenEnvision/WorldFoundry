"""Depth and normal colour maps used by the perception rendering CLI."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from worldfoundry.studio.visualization.plugins.styles.colormaps import colorize_depth_affine, colorize_normal


class TestColorizeDepthAffine:
    """Tests for colorize_depth_affine function."""

    def test_basic_returns_uint8_hwc(self):
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask(self):
        depth = np.random.rand(48, 48).astype(np.float32) + 0.1
        mask = depth > 0.5
        result = colorize_depth_affine(depth, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8

    def test_mask_none_default(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_different_cmap(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth, cmap="turbo")
        assert result.dtype == np.uint8

    def test_output_values_in_uint8_range(self):
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        result = colorize_depth_affine(depth)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_contiguous_array(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth)
        assert result.flags["C_CONTIGUOUS"]

class TestColorizeNormal:
    """Tests for colorize_normal function.

    NOTE: When mask is provided, masked pixels are set to 0 in the normal
    array first, then the color transform `normal * [0.5, -0.5, -0.5] + 0.5`
    is applied to ALL pixels including masked ones. So masked pixels become
    [0.5, 0.5, 0.5] -> uint8 [127, 127, 127], NOT [0, 0, 0].
    """

    def test_basic_returns_uint8_hwc(self):
        normal = np.random.rand(64, 64, 3).astype(np.float32) * 2 - 1
        result = colorize_normal(normal)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask_masked_pixels_are_neutral_gray(self):
        """Masked pixels become [127, 127, 127] not [0, 0, 0],
        because 0 * [0.5, -0.5, -0.5] + 0.5 = 0.5 -> 127."""
        normal = np.random.rand(48, 48, 3).astype(np.float32) * 2 - 1
        mask = np.ones((48, 48), dtype=bool)
        mask[:24, :] = False
        result = colorize_normal(normal, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8
        # Masked pixels: 0 * transform + 0.5 = [0.5, 0.5, 0.5] -> ~127
        masked_region = result[:24, :]
        assert np.all(masked_region[:, :, 0] == 127)
        assert np.all(masked_region[:, :, 1] == 127)
        assert np.all(masked_region[:, :, 2] == 127)

    def test_mask_none(self):
        normal = np.random.rand(32, 32, 3).astype(np.float32) * 2 - 1
        result = colorize_normal(normal, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_normal_color_transform(self):
        # normal * [0.5, -0.5, -0.5] + 0.5
        normal = np.array([[1.0, -1.0, -1.0]], dtype=np.float32).reshape(1, 1, 3)
        result = colorize_normal(normal)
        # Expected: [1*0.5+0.5, -1*-0.5+0.5, -1*-0.5+0.5] = [1.0, 1.0, 1.0] * 255
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255

    def test_zero_normal(self):
        normal = np.zeros((16, 16, 3), dtype=np.float32)
        result = colorize_normal(normal)
        expected_val = int(0.5 * 255)
        assert result[0, 0, 0] == expected_val

    def test_all_ones_normal(self):
        """normal=[1,1,1] -> 1*[0.5,-0.5,-0.5]+0.5 = [1.0, 0.0, 0.0] * 255."""
        normal = np.ones((4, 4, 3), dtype=np.float32)
        result = colorize_normal(normal)
        assert result[0, 0, 0] == 255  # 1.0 * 255
        assert result[0, 0, 1] == 0    # 0.0 * 255
        assert result[0, 0, 2] == 0    # 0.0 * 255
