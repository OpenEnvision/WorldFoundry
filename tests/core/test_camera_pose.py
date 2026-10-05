from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from worldfoundry.core.geometry.pose import Camera, RealEstate10KPoseProcessor


@pytest.mark.parametrize("views", [1, 3, 4])
@pytest.mark.parametrize("flipped", [False, True])
def test_pose_processor_preserves_pixel_centers_ray_layout_and_flip(views, flipped, monkeypatch):
    height, width = 2, 4
    processor = RealEstate10KPoseProcessor(
        sample_stride=1, sample_n_frames=views, sample_size=(height, width), use_flip=flipped,
    )
    # A 90-degree Z rotation with a nonzero origin exercises both ray halves.
    c2w = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    w2c = np.linalg.inv(c2w)[:3].reshape(-1).tolist()
    cameras = [Camera([i, 1, 1, 0.5, 0.5, 0, 0, *w2c]) for i in range(views)]
    flags = torch.zeros(views, dtype=torch.bool)
    flags[-1] = flipped
    if flipped:
        monkeypatch.setattr(processor.pixel_transforms[1], "get_flip_flag", lambda count: flags)

    actual = processor._plucker_from_camera_params(cameras)

    expected = torch.empty(1, views, height, width, 6)
    for view in range(views):
        for row in range(height):
            for column in range(width):
                u = width - 1 - column if flags[view] else column
                x, y = (u + 0.5) / width - 0.5, (row + 0.5) / height - 0.5
                norm = math.sqrt(x * x + y * y + 1)
                dx, dy, dz = -y / norm, x / norm, 1 / norm
                expected[0, view, row, column] = torch.tensor(
                    [2 * dz - 3 * dy, 3 * dx - dz, dy - 2 * dx, dx, dy, dz],
                )
    torch.testing.assert_close(actual, expected)
