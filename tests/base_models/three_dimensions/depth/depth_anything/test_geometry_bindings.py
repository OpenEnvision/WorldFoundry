# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""World-ray composition keeps native and FP32-inverse camera policies separate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import worldfoundry
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.geometry import (
    get_world_rays,
    homogenize_points,
)

pytestmark = pytest.mark.unit


def _native_unproject(coordinates: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", intrinsics.inverse(), homogenize_points(coordinates)) * z[..., None]


def _native_transform(coordinates: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", extrinsics, coordinates)


def test_get_world_rays_retains_fp32_inverse_and_native_precision_bindings() -> None:
    coordinates = torch.tensor([[[0.17, 0.83], [0.37, 0.61]]], dtype=torch.float64)
    intrinsics = torch.tensor([[[1.713, 0.123, 0.417], [0, 1.331, 0.533], [0, 0, 1]]], dtype=torch.float64)
    extrinsics = torch.eye(4, dtype=torch.float64)[None]
    extrinsics[:, :3, 3] = torch.tensor([1, 2, 3], dtype=torch.float64)
    origins, aligned = get_world_rays(coordinates, extrinsics, intrinsics)
    native_origins, native = get_world_rays(
        coordinates,
        extrinsics,
        intrinsics,
        unproject_fn=_native_unproject,
        transform_cam2world_fn=_native_transform,
    )
    expected = torch.einsum("...ij,...j->...i", intrinsics.float().inverse().double(), homogenize_points(coordinates))
    expected = expected / expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(aligned, expected, rtol=0, atol=0)
    assert not torch.equal(aligned, native)
    torch.testing.assert_close(origins, native_origins, rtol=0, atol=0)
    torch.testing.assert_close(origins, extrinsics[:, None, :3, 3].expand(1, 2, 3), rtol=0, atol=0)
    assert origins.stride(-2) == 0


def test_get_world_rays_aligns_mixed_camera_dtypes_only_for_depth_anything() -> None:
    coordinates = torch.tensor([[[0.17, 0.83], [0.37, 0.61]]], dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32)[None]
    extrinsics = torch.eye(4, dtype=torch.float64)[None]
    origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
    assert origins.dtype == directions.dtype == torch.float64
    with pytest.raises(RuntimeError):
        get_world_rays(
            coordinates,
            extrinsics,
            intrinsics,
            unproject_fn=_native_unproject,
            transform_cam2world_fn=_native_transform,
        )


def test_pixelsplat_binds_its_native_helpers_to_the_shared_world_ray_function() -> None:
    path = (
        Path(worldfoundry.__file__).parent
        / "base_models/three_dimensions/point_clouds/pixelsplat/src/geometry/projection.py"
    )
    tree = ast.parse(path.read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == get_world_rays.__module__
        and any(alias.name == "get_world_rays" and alias.asname == "_get_world_rays" for alias in node.names)
        for node in tree.body
    )
    binding = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "get_world_rays" for target in node.targets)
    )
    assert isinstance(binding, ast.Call)
    assert ast.unparse(binding.func) == "partial"
    assert [ast.unparse(argument) for argument in binding.args] == ["_get_world_rays"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in binding.keywords} == {
        "unproject_fn": "unproject",
        "transform_cam2world_fn": "transform_cam2world",
    }
