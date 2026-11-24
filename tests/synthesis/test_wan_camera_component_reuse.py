# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Camera-conditioning contracts shared by Matrix-Game, LingBot and its operator."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan import camera_geometry as camera
from worldfoundry.operators import lingbot_world_operator as operator
from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime.utils import cam_utils as lingbot
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.utils import cam_utils as matrix

pytestmark = pytest.mark.unit


def test_matrix_and_operator_import_canonical_camera_geometry() -> None:
    assert matrix.interpolate_camera_poses is camera.interpolate_camera_poses
    assert operator.compute_relative_poses is camera.compute_relative_poses
    assert operator.get_plucker_embeddings is camera.get_plucker_embeddings
    assert operator.get_Ks_transformed is camera.get_Ks_transformed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_wan_camera_embeddings_keep_origin_direction_order(dtype: torch.dtype) -> None:
    poses = torch.eye(4, dtype=dtype)[None]
    poses[:, :3, 3] = torch.tensor([1, 2, 3], dtype=dtype)
    intrinsics = torch.tensor([[1, 1, 0.5, 0.5]], dtype=dtype)
    expected = torch.tensor([[[[1, 2, 3, 0, 0, 1]]]], dtype=dtype)
    common = camera.get_plucker_embeddings(poses, intrinsics, 1, 1)
    torch.testing.assert_close(common, expected, rtol=0, atol=0)
    torch.testing.assert_close(lingbot.get_plucker_embeddings(poses, intrinsics, 1, 1), common, rtol=0, atol=0)


def test_lingbot_direction_only_branch_does_not_build_six_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_six_channel_path(*args: object) -> torch.Tensor:
        raise AssertionError("The direction-only branch must not construct six-channel embeddings")

    monkeypatch.setattr(lingbot, "_get_plucker_embeddings", reject_six_channel_path)
    poses = torch.eye(4)[None].repeat(2, 1, 1)
    intrinsics = torch.tensor([[4.0, 4.0, 2.0, 2.0]]).repeat(2, 1)
    directions = lingbot.get_plucker_embeddings(poses, intrinsics, 4, 4, only_rays_d=True)
    expected = camera.world_ray_directions(poses, intrinsics, 4, 4).view(2, 4, 4, 3)
    torch.testing.assert_close(directions, expected, rtol=0, atol=0)
    assert directions.is_contiguous()


def test_camera_relative_pose_retains_first_frame_and_translation_scale() -> None:
    poses = torch.eye(4)[None].repeat(2, 1, 1)
    poses[:, 0, 3] = torch.tensor([3.0, 5.0])
    relative = camera.compute_relative_poses(poses)
    torch.testing.assert_close(relative[0], torch.eye(4), rtol=0, atol=0)
    assert relative[1, 0, 3] == 1
    stationary = camera.compute_relative_poses(poses[:1].repeat(2, 1, 1), framewise=True)
    assert torch.equal(stationary, torch.eye(4)[None].repeat(2, 1, 1))


def test_matrix_refiner_imports_hunyuan_stack() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry/synthesis/visual_generation/matrix_game/dit/token_refiner.py"
    )
    tree = ast.parse(path.read_text())
    owner = "worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.layers.token_refiner"
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == owner
        and any(alias.name == "IndividualTokenRefiner" for alias in node.names)
        for node in tree.body
    )
    assert {node.name for node in tree.body if isinstance(node, ast.ClassDef)} == {"SingleTokenRefiner"}
