"""Known camera coordinates and quaternion identities guard 3D convention changes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldfoundry.core.geometry.transforms import (
    depth_to_world_points,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pose_rows", [3, 4])
def test_depth_unprojection_uses_opencv_axes_and_camera_to_world_pose(dtype, pose_rows):
    depth = np.array([[2, 4, 6], [1, 3, 5]], dtype=dtype)
    intrinsics = np.array([[2, 0, 1], [0, 4, 0.5], [0, 0, 1]], dtype=dtype)
    pose = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], dtype=dtype)
    original = [array.copy() for array in (depth, intrinsics, pose)]
    actual = depth_to_world_points(depth, intrinsics, pose[:pose_rows])
    expected = np.array(
        [
            [[10.25, 19, 32], [10.5, 20, 34], [10.75, 23, 36]],
            [[9.875, 19.5, 31], [9.625, 20, 33], [9.375, 22.5, 35]],
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    for array, before in zip((depth, intrinsics, pose), original):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("depth", np.ones((2, 3, 1))),
        ("intrinsics", np.eye(4)),
        ("pose", np.eye(3)),
    ],
)
def test_wrong_geometry_shapes_fail_instead_of_broadcasting(bad_field, bad_value):
    options = {"depth": np.ones((2, 3)), "intrinsics": np.eye(3), "pose": np.eye(4)}
    options[bad_field] = bad_value
    with pytest.raises(ValueError, match="expected"):
        depth_to_world_points(**options)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_batched_half_turns_sign_and_scale_preserve_rotations(dtype):
    quaternions = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0, 0, 0, 1],
            [0, 0, 2**-0.5, 2**-0.5],
        ],
        dtype=dtype,
    ).reshape(2, 3, 4)
    rotations = quaternion_xyzw_to_rotation_matrix(quaternions)
    torch.testing.assert_close(rotations[0, 0], torch.diag(torch.tensor([1, -1, -1], dtype=dtype)), atol=0, rtol=0)
    torch.testing.assert_close(rotations[0, 1], torch.diag(torch.tensor([-1, 1, -1], dtype=dtype)), atol=0, rtol=0)
    torch.testing.assert_close(rotations[0, 2], torch.diag(torch.tensor([-1, -1, 1], dtype=dtype)), atol=0, rtol=0)
    torch.testing.assert_close(quaternion_xyzw_to_rotation_matrix(-quaternions), rotations, atol=0, rtol=0)
    torch.testing.assert_close(quaternion_xyzw_to_rotation_matrix(quaternions * 4), rotations, atol=0, rtol=0)
    restored = rotation_matrix_to_quaternion_xyzw(rotations)
    assert restored.shape == quaternions.shape
    assert (restored[..., 3] >= 0).all()
    torch.testing.assert_close(quaternion_xyzw_to_rotation_matrix(restored), rotations, atol=1e-6, rtol=1e-6)
    identity = torch.eye(3, dtype=dtype).expand(2, 3, 3, 3)
    torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity, atol=1e-6, rtol=1e-6)
