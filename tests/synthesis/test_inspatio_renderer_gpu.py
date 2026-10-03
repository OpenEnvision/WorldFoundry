"""Check strict rendering with repeated pixels and occluded point splats."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strict_renderer_matches_cpu_and_repeats(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", types.ModuleType("open3d"))
    trajectory_name = "worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.trajectory"
    trajectory = types.ModuleType(trajectory_name)
    trajectory.generate_traj_txt = None
    monkeypatch.setitem(sys.modules, trajectory_name, trajectory)
    from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.scripts.render_point_cloud import (
        render_batch,
    )

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
        points = torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 2.], [0., 0., 1.00005]])
        colors = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
        pose = torch.eye(4)
        intrinsic = torch.tensor([[2., 0., 4.], [0., 2., 4.], [0., 0., 1.]])
        expected_rgb, expected_mask = render_batch(points, colors, pose, intrinsic, 8, 8,
                                                    point_size=1, ss_ratio=1.)
        np.testing.assert_array_equal(expected_rgb[4, 4], [0, 255, 255])
        assert expected_mask[4, 4] == 255
        gpu_inputs = [value.cuda() for value in (points, colors, pose, intrinsic)]
        for _ in range(8):
            rgb, mask = render_batch(*gpu_inputs, 8, 8, point_size=1, ss_ratio=1.)
            np.testing.assert_array_equal(rgb, expected_rgb)
            np.testing.assert_array_equal(mask, expected_mask)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
