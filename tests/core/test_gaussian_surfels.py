# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Flat-surface geometry and standard splat artifact contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from worldfoundry.core.geometry.surfels import GaussianSurfels, reconstruction_loss
from worldfoundry.core.geometry.transforms import quaternion_xyzw_to_rotation_matrix

pytestmark = pytest.mark.unit


def plane() -> tuple[GaussianSurfels, torch.Tensor, torch.Tensor]:
    image = torch.linspace(0, 1, 48).reshape(1, 3, 4, 4)
    depth = torch.ones(1, 1, 4, 4) * 2
    normals = torch.zeros_like(image)
    normals[:, 2] = -1
    intrinsic = torch.tensor([[4.0, 0, 2], [0, 4, 2], [0, 0, 1]])
    camera = torch.eye(4)
    surfels = GaussianSurfels.from_rgbd(
        image, depth, normals, intrinsic=intrinsic, world_to_camera=camera, mask=torch.ones_like(depth).bool(), layer=1
    )
    return surfels, camera, intrinsic


def test_rgbd_unprojection_flat_scale_and_normal_orientation() -> None:
    surfels, _, _ = plane()
    assert surfels.means.shape == (16, 3)
    torch.testing.assert_close(surfels.means[0], torch.tensor([-0.75, -0.75, 2.0]))
    torch.testing.assert_close(surfels.scales[:, :2], torch.full((16, 2), 0.5))
    assert (surfels.scales[:, 2] == 0).all()
    rotation = quaternion_xyzw_to_rotation_matrix(surfels.quaternions[:, (1, 2, 3, 0)])
    torch.testing.assert_close(rotation[:, :, 2], torch.tensor([[0.0, 0, -1]]).expand(16, 3))


def test_camera_pose_moves_surface_not_camera_normal_frame() -> None:
    surfels, camera, intrinsic = plane()
    camera[0, 3] = -1
    moved = GaussianSurfels.from_rgbd(
        torch.ones(1, 3, 4, 4),
        torch.full((1, 1, 4, 4), 2.0),
        torch.tensor([0.0, 0, -1])[None, :, None, None].expand(1, 3, 4, 4),
        intrinsic=intrinsic,
        world_to_camera=camera,
        mask=torch.ones(1, 1, 4, 4).bool(),
        layer=2,
    )
    torch.testing.assert_close(moved.means, surfels.means + torch.tensor([1.0, 0, 0]))
    joined = surfels.append(moved)
    assert len(joined.means) == 32 and len(surfels.means) == 16
    assert joined.layers.tolist() == [1] * 16 + [2] * 16


@pytest.mark.parametrize(
    "normal",
    [(0.0, 0.0, -1.0), (0.8, 0.0, -0.6), (0.0, -0.95, -math.sqrt(1 - 0.95**2)), (0.3, 0.4, -math.sqrt(0.75))],
)
@pytest.mark.parametrize("roll", [0.0, 37.0, 90.0])
def test_tangent_covariance_projects_to_unit_pixels(normal: tuple[float, float, float], roll: float) -> None:
    angle = math.radians(roll)
    camera = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0, 0.3],
            [math.sin(angle), math.cos(angle), 0, -0.2],
            [0, 0, 1, 0.1],
            [0, 0, 0, 1],
        ]
    )
    intrinsic = torch.tensor([[12.0, 0, 1], [0, 18.0, 1], [0, 0, 1]])
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
    rays = torch.stack(((xx + 0.5 - 1) / 12, (yy + 0.5 - 1) / 18, torch.ones_like(xx)), dim=-1)
    world_normal = torch.tensor(normal)
    camera_normal = camera[:3, :3] @ world_normal
    center = camera[:3, :3] @ torch.tensor([0.0, 0, 2]) + camera[:3, 3]
    depth = ((camera_normal @ center) / (rays @ camera_normal))[None, None]
    scene = GaussianSurfels.from_rgbd(
        torch.ones(1, 3, 3, 3),
        depth,
        camera_normal[None, :, None, None].expand(1, 3, 3, 3),
        intrinsic=intrinsic,
        world_to_camera=camera,
        mask=torch.ones_like(depth).bool(),
        layer=1,
    )
    rotation = quaternion_xyzw_to_rotation_matrix(scene.quaternions[:, (1, 2, 3, 0)])
    torch.testing.assert_close(rotation[:, :, 2], world_normal.expand(9, 3), rtol=1e-5, atol=1e-6)
    covariance = rotation @ torch.diag_embed(scene.scales.square()) @ rotation.transpose(-1, -2)
    camera_points = scene.means @ camera[:3, :3].T + camera[:3, 3]
    z = camera_points[:, 2]
    jacobian = torch.zeros(9, 2, 3)
    jacobian[:, 0, 0] = 12 / z
    jacobian[:, 1, 1] = 18 / z
    jacobian[:, 0, 2] = -12 * camera_points[:, 0] / z.square()
    jacobian[:, 1, 2] = -18 * camera_points[:, 1] / z.square()
    projection = jacobian @ camera[:3, :3]
    pixel_covariance = projection @ covariance @ projection.transpose(-1, -2)
    torch.testing.assert_close(pixel_covariance, torch.eye(2).expand(9, 2, 2), rtol=2e-5, atol=2e-5)
    assert (scene.scales[:, 2] == 0).all()


def test_optical_axis_roll_preserves_world_covariance() -> None:
    normal = torch.tensor([0.8, 0.0, -0.6])
    covariances = []
    for rotation in (torch.eye(3), torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])):
        camera = torch.eye(4)
        camera[:3, :3] = rotation
        depth = torch.full((1, 1, 1, 1), 2.0)
        scene = GaussianSurfels.from_rgbd(
            torch.ones(1, 3, 1, 1),
            depth,
            (rotation @ normal)[None, :, None, None],
            intrinsic=torch.tensor([[960.0, 0, 0.5], [0, 960, 0.5], [0, 0, 1]]),
            world_to_camera=camera,
            mask=torch.ones_like(depth).bool(),
            layer=1,
        )
        frame = quaternion_xyzw_to_rotation_matrix(scene.quaternions[:, (1, 2, 3, 0)])
        covariances.append(frame @ torch.diag_embed(scene.scales.square()) @ frame.transpose(-1, -2))
    torch.testing.assert_close(covariances[0], covariances[1], rtol=1e-5, atol=1e-12)


def test_splat_record_layout(tmp_path: Path) -> None:
    surfels, _, _ = plane()
    path = surfels.save_splat(tmp_path / "scene.splat")
    assert path.stat().st_size == 16 * 32
    records = np.fromfile(
        path, dtype=[("position", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rotation", "u1", (4,))]
    )
    np.testing.assert_array_equal(records["position"], surfels.means.numpy())
    np.testing.assert_array_equal(records["scale"], surfels.scales.numpy())
    assert (records["rgba"][:, 3] == 252).all()
    assert (np.ptp(records["rotation"].astype(np.float32), axis=1) > 0).all()


def test_pixel_center_order_and_configured_initialization() -> None:
    image = torch.arange(3 * 3 * 4).float().reshape(1, 3, 3, 4) / 36
    scene = GaussianSurfels.from_rgbd(
        image,
        torch.full((1, 1, 3, 4), 2.0),
        torch.tensor([0.0, 0, -1])[None, :, None, None].expand_as(image),
        intrinsic=torch.tensor([[4.0, 0, 2], [0, 4, 1.5], [0, 0, 1]]),
        world_to_camera=torch.eye(4),
        mask=torch.ones(1, 1, 3, 4).bool(),
        layer=1,
        footprint=0.7,
        opacity=0.15,
    )
    torch.testing.assert_close(scene.means[:3], torch.tensor([[-0.75, -0.5, 2], [-0.75, 0, 2], [-0.75, 0.5, 2]]))
    torch.testing.assert_close(scene.colors, image[0].permute(2, 1, 0).reshape(-1, 3))
    torch.testing.assert_close(scene.scales[:, :2], torch.full((12, 2), 0.35))
    torch.testing.assert_close(scene.opacities, torch.full((12,), 0.15))


def test_nonzero_normal_thickness_uses_the_depth_sampling_scale() -> None:
    image = torch.ones(1, 3, 2, 3)
    depth = torch.full((1, 1, 2, 3), 2.0)
    scene = GaussianSurfels.from_rgbd(
        image,
        depth,
        torch.tensor([0.0, 0, -1])[None, :, None, None].expand_as(image),
        intrinsic=torch.tensor([[4.0, 0, 1.5], [0, 4, 1], [0, 0, 1]]),
        world_to_camera=torch.eye(4),
        mask=torch.ones_like(depth).bool(),
        layer=1,
        footprint=0.7,
        thickness=0.7,
        opacity=0.15,
    )
    torch.testing.assert_close(scene.scales, torch.full((6, 3), 0.35), rtol=0, atol=0)


def test_reconstruction_loss_retains_structural_gradients() -> None:
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    torch.testing.assert_close(reconstruction_loss(image, image), torch.tensor(0.0), rtol=0, atol=0)
    changed = image.flip(-1).clone().requires_grad_(True)
    loss = reconstruction_loss(changed, image)
    loss.backward()
    assert float(loss) > 0 and changed.grad is not None and torch.isfinite(changed.grad).all()
    assert changed.grad.abs().sum() > 0


def test_refinement_does_not_mutate_the_input_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    scene, camera, intrinsic = plane()
    original = {field: getattr(scene, field).clone() for field in scene.__dataclass_fields__}

    def rasterize(gaussians: GaussianSurfels, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, None]:
        return gaussians.quaternions[:, 0].sum().expand(1, 3, 1, 1), torch.zeros(1, 1, 1, 1), None

    monkeypatch.setattr(GaussianSurfels, "_rasterize", rasterize)
    monkeypatch.setattr(
        "worldfoundry.core.geometry.surfels.reconstruction_loss", lambda prediction, target: prediction.sum()
    )
    fitted = scene.refine(torch.ones(1, 3, 1, 1), camera, intrinsic, steps=2)
    for field, expected in original.items():
        torch.testing.assert_close(getattr(scene, field), expected, rtol=0, atol=0)
    assert not torch.equal(fitted.quaternions, original["quaternions"])


@pytest.mark.parametrize("all_pruned", [False, True])
def test_refinement_prunes_with_aligned_adam_state_and_stops_when_empty(
    monkeypatch: pytest.MonkeyPatch, all_pruned: bool
) -> None:
    scene, camera, intrinsic = plane()
    scene = replace(scene, scales=torch.full_like(scene.scales, 0.02), opacities=torch.full_like(scene.opacities, 0.15))
    if all_pruned:
        scene.opacities.fill_(0.01)
    else:
        scene.opacities[:1] = 0.01
        scene.scales[:1] = 0.1
        scene.opacities[2] = 0.01
        scene.scales[3] = 0.07
    calls = []
    real_step = torch.optim.Adam.step

    def rasterize(gaussians: GaussianSurfels, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, None]:
        value = gaussians.opacities.sum() + gaussians.scales.sum() + gaussians.quaternions.sum()
        return value.expand(1, 3, 1, 1), torch.zeros(1, 1, 1, 1), None

    def optimizer_step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            state = optimizer.state.get(parameter, {})
            if state:
                assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
                if len(calls) == 80:
                    assert float(state["step"]) == 79
        calls.append(True)
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(GaussianSurfels, "_rasterize", rasterize)
    monkeypatch.setattr(
        "worldfoundry.core.geometry.surfels.reconstruction_loss", lambda prediction, target: prediction.sum() * 0
    )
    monkeypatch.setattr(torch.optim.Adam, "step", optimizer_step)
    fitted = scene.refine(torch.ones(1, 3, 1, 1), camera, intrinsic, steps=82, start=0 if all_pruned else 1)
    if all_pruned:
        assert len(fitted.means) == 0 and len(calls) == 79
    else:
        keep = torch.ones(16, dtype=torch.bool)
        keep[2:4] = False
        torch.testing.assert_close(fitted.means, scene.means[keep], rtol=0, atol=0)
        torch.testing.assert_close(fitted.opacities[:1], scene.opacities[:1], rtol=0, atol=0)
        assert len(calls) == 81
    assert all(not getattr(fitted, field).requires_grad for field in fitted.__dataclass_fields__)


@pytest.mark.parametrize("all_pruned", [False, True])
def test_refinement_range_prunes_only_trainable_particles_and_freezes_both_sides(
    monkeypatch: pytest.MonkeyPatch, all_pruned: bool
) -> None:
    scene, camera, intrinsic = plane()
    scene = replace(scene, scales=torch.full_like(scene.scales, 0.02), opacities=torch.full_like(scene.opacities, 0.15))
    scene.scales[:2] = scene.scales[12:] = 0.1
    scene.opacities[:2] = scene.opacities[12:] = 0.01
    scene.opacities[3] = 0.01
    scene.scales[6] = 0.07
    if all_pruned:
        scene.opacities[2:12] = 0.01
    original = {name: getattr(scene, name).clone() for name in scene.__dataclass_fields__}

    def rasterize(gaussians: GaussianSurfels, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, None]:
        value = gaussians.opacities.sum() + gaussians.scales.sum() + gaussians.quaternions.sum()
        return value.expand(1, 3, 1, 1), torch.zeros(1, 1, 1, 1), None

    monkeypatch.setattr(GaussianSurfels, "_rasterize", rasterize)
    monkeypatch.setattr(
        "worldfoundry.core.geometry.surfels.reconstruction_loss", lambda prediction, target: prediction.sum() * 0
    )
    fitted = scene.refine(torch.ones(1, 3, 1, 1), camera, intrinsic, steps=82, start=2, stop=12)
    keep = torch.ones(16, dtype=torch.bool)
    if all_pruned:
        keep[2:12] = False
    else:
        keep[[3, 6]] = False
    torch.testing.assert_close(fitted.means, scene.means[keep], rtol=0, atol=0)
    for name, value in original.items():
        torch.testing.assert_close(getattr(scene, name), value, rtol=0, atol=0)
        torch.testing.assert_close(getattr(fitted, name)[:2], value[:2], rtol=0, atol=0)
        torch.testing.assert_close(getattr(fitted, name)[-4:], value[-4:], rtol=0, atol=0)


def test_empty_refinement_prefix_keeps_the_frozen_scene() -> None:
    scene, camera, intrinsic = plane()
    assert scene.refine(torch.ones(1, 3, 1, 1), camera, intrinsic, steps=100, stop=0) is scene


def test_three_neighbor_floater_filter_keeps_aligned_parameters() -> None:
    scene, _, _ = plane()
    scene = GaussianSurfels(
        torch.cat((torch.zeros(15, 3), torch.full((1, 3), 10.0))),
        scene.scales,
        scene.quaternions,
        scene.colors,
        scene.opacities,
        scene.layers,
    )
    filtered = scene.remove_floaters(mean_squared_distance=0.0002)
    assert len(filtered.means) == 15
    torch.testing.assert_close(filtered.colors, scene.colors[:15], rtol=0, atol=0)
