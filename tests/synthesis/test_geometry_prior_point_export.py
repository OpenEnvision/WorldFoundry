from types import SimpleNamespace

import numpy as np
import torch

from worldfoundry.synthesis.visual_generation.geometry_priors import (
    _result_points,
    _write_camera_points_ply,
)


def test_result_points_normalizes_channel_first_tensor():
    points = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3)

    result = _result_points(SimpleNamespace(points=points))

    assert result.shape == (2, 3, 3)
    np.testing.assert_array_equal(result[..., 0], points[0, 0].numpy())


def test_camera_point_export_flips_only_opencv_y(tmp_path):
    points = np.array([[[1.0, 2.0, 3.0], [-1.0, -2.0, 4.0]]], dtype=np.float32)
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    path = tmp_path / "points.ply"

    _write_camera_points_ply(path, rgb=rgb, points=points, max_points=10)

    rows = path.read_text().split("end_header\n", 1)[1].splitlines()
    assert rows == ["1.000000 -2.000000 3.000000 255 0 0", "-1.000000 2.000000 4.000000 0 255 0"]
