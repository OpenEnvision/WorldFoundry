from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "diffusers" dependency at import time; skip when it is unavailable.
pytest.importorskip("diffusers")

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime.utils.export import (
    _covariance_to_quaternion_and_scale,
    _inverse_sigmoid,
    save_as_ply,
)


def test_covariance_export_round_trips_with_wxyz_quaternion() -> None:
    rotation = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
    scales = np.array([0.5, 1.25, 2.0])
    covariance = rotation @ np.diag(scales**2) @ rotation.T

    quaternion_wxyz, exported_scales = _covariance_to_quaternion_and_scale(
        torch.from_numpy(covariance[None]).double()
    )

    exported_rotation = Rotation.from_quat(quaternion_wxyz[:, [1, 2, 3, 0]]).as_matrix()[0]
    reconstructed = exported_rotation @ np.diag(exported_scales[0] ** 2) @ exported_rotation.T
    np.testing.assert_allclose(reconstructed, covariance, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(exported_rotation), 1.0, atol=1e-10)


def test_exported_opacity_is_standard_3dgs_logit() -> None:
    probabilities = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    logits = _inverse_sigmoid(probabilities)

    torch.testing.assert_close(torch.sigmoid(logits), probabilities)


def test_ply_preserves_predicted_scale_and_reorders_xyzw(tmp_path) -> None:
    means = torch.zeros((1, 1, 1, 3))
    scales = torch.tensor([[[[1.0, 2.0, 4.0]]]])
    rotation_xyzw = torch.tensor([[[[0.0, 0.0, 2**-0.5, 2**-0.5]]]])
    sh = torch.zeros((1, 1, 1, 3, 1))
    opacity = torch.tensor([[[[0.2]]]])
    covariance = torch.eye(3).reshape(1, 1, 1, 3, 3)
    pred1 = {
        "means": means,
        "scales": scales,
        "rotations": rotation_xyzw,
        "covariances": covariance,
        "sh": sh,
        "opacities": opacity,
    }
    pred2 = {
        "means_in_other_view": means,
        "scales": scales,
        "rotations": rotation_xyzw,
        "covariances": covariance,
        "sh": sh,
        "opacities": opacity,
    }
    path = tmp_path / "splatt3r.ply"

    save_as_ply(pred1, pred2, str(path))

    vertex = PlyData.read(path)["vertex"][0]
    np.testing.assert_allclose(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]],
        np.log([1.0, 2.0, 4.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        [2**-0.5, 0.0, 0.0, 2**-0.5],
        atol=1e-6,
    )
    np.testing.assert_allclose(vertex["opacity"], np.log(0.2 / 0.8), atol=1e-6)
