"""CPU regressions for shared depth and video artifact helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


class TestArtifactsHelpers:
    """Test artifact visualization helpers from core.io.artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = importlib.import_module("worldfoundry.core.media.artifacts")

    def test_depth_helpers_available(self):
        for name in [
            "COLORMAP_INFERNO",
            "COLORMAP_VIRIDIS",
            "build_depth_visualizations",
            "create_depth_visualization",
            "depth_to_colormap_pil",
            "depth_to_colormap_rgb",
            "depth_to_uint8",
            "depths_to_pil_images",
            "prepare_depth_visualization",
            "render_point_cloud",
            "save_depth_colormap",
            "squeeze_depth_to_2d",
        ]:
            assert hasattr(self.mod, name), f"{name} not found in artifacts module"

    def test_colormap_constants_are_strings(self):
        assert isinstance(self.mod.COLORMAP_INFERNO, str)
        assert isinstance(self.mod.COLORMAP_VIRIDIS, str)
        assert self.mod.COLORMAP_INFERNO == "inferno"
        assert self.mod.COLORMAP_VIRIDIS == "viridis"

    def test_depth_to_uint8_basic(self):
        depth = np.random.rand(1, 10, 10).astype(np.float32)
        result = self.mod.depth_to_uint8(depth)
        assert result.dtype == np.uint8
        assert result.shape == (10, 10)

    def test_squeeze_depth_to_2d(self):
        depth_3d = np.random.rand(1, 1, 10, 10).astype(np.float32)
        result = self.mod.squeeze_depth_to_2d(depth_3d)
        assert result.ndim == 2
        assert result.shape == (10, 10)

    def test_depth_to_colormap_rgb_inferno(self):
        depth_2d = np.random.rand(10, 10).astype(np.float32)
        result = self.mod.depth_to_colormap_rgb(depth_2d, self.mod.COLORMAP_INFERNO)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_depth_to_colormap_rgb_viridis(self):
        depth_2d = np.random.rand(10, 10).astype(np.float32)
        result = self.mod.depth_to_colormap_rgb(depth_2d, self.mod.COLORMAP_VIRIDIS)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

class TestTensorVideoHelpers:
    """Test tensor video helpers from core.io.artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = importlib.import_module("worldfoundry.core.media.artifacts")

    def test_tensor_video_helpers_callable(self):
        for name in [
            "save_batch_img",
            "show_batch_img",
            "visualize_latent_tensor_bcthw",
            "visualize_tensor_bcthw",
        ]:
            assert hasattr(self.mod, name), f"{name} not found in artifacts module"
            assert callable(getattr(self.mod, name))
