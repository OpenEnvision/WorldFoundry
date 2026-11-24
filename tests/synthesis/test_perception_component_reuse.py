# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Canonical geometry and GEGLU contracts for the perception runtimes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.lvdm.attention import GEGLU
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.prope import _lift_K
from worldfoundry.base_models.three_dimensions.general_3d.uni3c.utils import points_padding
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.geometry import (
    depth_to_world_coords_points,
)
from worldfoundry.synthesis.visual_generation.vmem.vmem_runtime.modeling.modules.transformer import FeedForward

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2] / "worldfoundry/synthesis/visual_generation"


def test_vmem_feedforward_loads_canonical_geglu_keys() -> None:
    torch.manual_seed(3)
    module = FeedForward(8, mult=2).eval()
    assert isinstance(module.net[0], GEGLU)
    restored = FeedForward(8, mult=2).eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    assert set(module.state_dict()) == {"net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"}
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), module(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("relative", "owner", "names"),
    [
        (
            "neoverse/native/rasterization.py",
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.geometry",
            {"depth_to_world_coords_points"},
        ),
        (
            "neoverse/native/depth_anything_reconstructor.py",
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.geometry",
            {"depth_to_world_coords_points"},
        ),
        (
            "uniworld_view/uniworld_view_runtime/demo.py",
            "worldfoundry.base_models.three_dimensions.general_3d.uni3c.utils",
            {"points_padding"},
        ),
        (
            "worldfm/worldfm_runtime/diffusion/model/nets/prope.py",
            "worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.prope",
            {"_apply_tiled_projmat", "_invert_K", "_invert_SE3", "_rope_precompute_coeffs", "_apply_block_diagonal"},
        ),
    ],
)
def test_perception_load_sites_import_canonical_geometry(relative: str, owner: str, names: set[str]) -> None:
    tree = ast.parse((ROOT / relative).read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == owner
        for alias in node.names
    }
    assert names <= imported
    assert not names & {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_neoverse_geometry_and_uniworld_padding_keep_depth_mask() -> None:
    depth = torch.tensor([[[2.0, 0.0], [-1.0, 4.0]]])
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 3.0
    world, camera, valid = depth_to_world_coords_points(depth, pose, torch.eye(3)[None])
    torch.testing.assert_close(world[..., 0], camera[..., 0] + 3, rtol=0, atol=0)
    assert torch.equal(valid, depth > 1e-8)
    torch.testing.assert_close(points_padding(camera)[..., :3], camera, rtol=0, atol=0)
    assert torch.equal(points_padding(camera)[..., 3], torch.ones_like(depth))


def test_worldfm_intrinsics_lift_preserves_bfloat16() -> None:
    intrinsics = torch.eye(3, dtype=torch.bfloat16)[None]
    lifted = _lift_K(intrinsics)
    assert lifted.dtype == intrinsics.dtype
    torch.testing.assert_close(lifted[:, :3, :3], intrinsics, rtol=0, atol=0)
    assert lifted[0, 3, 3] == 1
