# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pixel, matrix and depth contracts for the shared geometry primitives."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from worldfoundry.core.geometry.projection import (
    invert_matrix,
    pixel_coordinate_grid,
    transform_points,
    unproject_depth_with_focal,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_transform_points_affine_projection_and_dtype(backend: str) -> None:
    matrix = np.array([[2, 0, 5], [0, 3, -2], [0, 0, 1]], dtype=np.float32)
    points = np.array([[1, 2], [0, -1]], dtype=np.float64)
    expected = np.array([[3.5, 2], [2.5, -2.5]])
    if backend == "numpy":
        result = transform_points(matrix, points, norm=0.5)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64
    else:
        result = transform_points(torch.from_numpy(matrix), torch.from_numpy(points), norm=0.5)
        torch.testing.assert_close(result, torch.from_numpy(expected).float(), rtol=0, atol=0)
        assert result.dtype == torch.float32


@pytest.mark.parametrize("spatial", [False, True])
def test_transform_points_batched_affine_and_normalization(spatial: bool) -> None:
    matrix = torch.eye(4).repeat(2, 1, 1)
    matrix[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]])
    points = torch.arange(1, 25, dtype=torch.float32).reshape(2, 4, 3)
    expected = points + matrix[:, None, :3, 3]
    if spatial:
        points = points.reshape(2, 2, 2, 3)
        expected = expected.reshape(2, 2, 2, 3)
        # Spatial batches apply the leading affine block, then normalize by Z.
        expected = expected / expected[..., -1:]
    result = transform_points(matrix, points, ncol=2, norm=True)
    torch.testing.assert_close(result, expected[..., :2], rtol=0, atol=0)


def test_transform_points_flattens_multiple_batch_axes() -> None:
    matrix = torch.eye(4, dtype=torch.float64).repeat(2, 3, 1, 1)
    matrix[..., :3, 3] = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    points = torch.arange(360, dtype=torch.float64).reshape(2, 3, 4, 5, 3)
    expected = points + matrix[..., None, None, :3, 3]
    torch.testing.assert_close(transform_points(matrix, points), expected, rtol=0, atol=0)
    np.testing.assert_array_equal(transform_points(matrix.numpy(), points.numpy()), expected.numpy())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_invert_matrix_uses_native_backend_precision(dtype: torch.dtype) -> None:
    matrix = torch.tensor([[1.3, 0.2, 0.1], [0.1, 1.7, 0.3], [0.0, 0.0, 1.0]], dtype=dtype)
    torch.testing.assert_close(invert_matrix(matrix), torch.linalg.inv(matrix), rtol=0, atol=0)
    np.testing.assert_array_equal(invert_matrix(matrix.numpy()), np.linalg.inv(matrix.numpy()))
    with pytest.raises(ValueError, match="bad matrix type"):
        invert_matrix([[1, 0], [0, 1]])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("cat_dim", [None, 0, -1])
def test_pixel_coordinate_grid_axis_order_and_origin(device: str | None, cat_dim: int | None) -> None:
    result = pixel_coordinate_grid(3, 2, device=device, origin=(0.5, -0.5), cat_dim=cat_dim)
    expected = np.array([[[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]], [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]])
    if cat_dim is None:
        assert len(result) == 2
        actual = np.stack([np.asarray(axis) for axis in result])
    else:
        actual = np.asarray(result)
        if cat_dim == -1:
            expected = expected.transpose(1, 2, 0)
    np.testing.assert_array_equal(actual, expected)


def test_pixel_coordinate_grid_homogeneous_and_batch_axis_contract() -> None:
    result = pixel_coordinate_grid(3, 2, device="cpu", homogeneous=True, dtype=torch.int64)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 3)
    assert result.dtype == torch.get_default_dtype()
    torch.testing.assert_close(result[..., 2], torch.ones(2, 3), rtol=0, atol=0)
    batched = pixel_coordinate_grid(3, 2, device="cpu", homogeneous=True, unsqueeze=0)
    assert isinstance(batched, torch.Tensor)
    assert batched.shape == (1, 2, 3, 2)
    assert batched.dtype == torch.int64
    torch.testing.assert_close(batched[0], result[..., :2].long(), rtol=0, atol=0)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("explicit_center", [False, True])
def test_unproject_depth_with_focal_preserves_plane_arithmetic_and_output_dtype(
    default_dtype: torch.dtype, input_dtype: torch.dtype, explicit_center: bool
) -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        torch.manual_seed(31)
        depth = torch.rand(2, 3, 4, dtype=input_dtype)
        focal = torch.rand(2, 2, 3, 4, dtype=input_dtype) + 0.1
        pp = torch.tensor([[0.7, 1.2], [1.3, 0.8]], dtype=input_dtype) if explicit_center else None
        x = torch.arange(4)[None, None, :].expand(1, 3, 4)
        y = torch.arange(3)[None, :, None].expand(1, 3, 4)
        x = x - (1.5 if pp is None else pp[:, 0, None, None])
        y = y - (1.0 if pp is None else pp[:, 1, None, None])
        single = unproject_depth_with_focal(depth, focal, pp=pp)
        multi = unproject_depth_with_focal(depth[..., None], focal, pp=pp)
        expected_single = torch.stack((depth * x / focal[:, 0], depth * y / focal[:, 1], depth), dim=-1)
        expected_multi = torch.stack((depth * (x / focal[:, 0]), depth * (y / focal[:, 1]), depth), dim=-1)
        assert single.dtype == multi.dtype == default_dtype
        torch.testing.assert_close(single, expected_single.to(default_dtype), rtol=0, atol=0)
        torch.testing.assert_close(multi[..., 0], expected_multi.to(default_dtype), rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous)


def test_shared_geometry_imports_do_not_load_model_runtimes_or_hydra() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import worldfoundry.core.geometry.projection; "
            "import worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.geometry; "
            "assert 'hydra' not in sys.modules; "
            "assert not any('.cut3r' in name or '.mapanything' in name or '.pixelsplat' in name "
            "or name.endswith('depth_anything_v3.api') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
