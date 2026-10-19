# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""CUDA rasterization and scene-reconstruction contracts without weights."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest
import torch

from worldfoundry.core.geometry.surfels import GaussianSurfels
from worldfoundry.core.geometry.transforms import quaternion_xyzw_to_rotation_matrix

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")]


def test_flat_gaussians_render_and_refine_without_moving_frozen_geometry() -> None:
    device = "cuda"
    image = torch.full((1, 3, 16, 16), 0.6, device=device)
    depth = torch.full((1, 1, 16, 16), 2.0, device=device)
    normals = torch.zeros_like(image)
    normals[:, 2] = -1
    intrinsic = torch.tensor([[16.0, 0, 8], [0, 16, 8], [0, 0, 1]], device=device)
    camera = torch.eye(4, device=device)
    scene = GaussianSurfels.from_rgbd(
        image, depth, normals, intrinsic=intrinsic, world_to_camera=camera, mask=torch.ones_like(depth).bool(), layer=1
    )
    rgb, alpha, rendered_depth = scene.render(camera, intrinsic, height=16, width=16)
    assert rgb.shape == image.shape and alpha.shape == depth.shape
    assert torch.isfinite(rgb).all() and torch.isfinite(rendered_depth).all()
    assert float(alpha.mean()) > 0.5
    torch.testing.assert_close(
        rendered_depth[alpha > 0.5], torch.full_like(rendered_depth[alpha > 0.5], 2), rtol=1e-5, atol=1e-5
    )
    fitted = scene.refine(image, camera, intrinsic, steps=3, start=128)
    torch.testing.assert_close(fitted.means, scene.means, rtol=0, atol=0)
    torch.testing.assert_close(fitted.opacities[:128], scene.opacities[:128], rtol=0, atol=0)
    assert all(torch.isfinite(getattr(fitted, field)).all() for field in fitted.__dataclass_fields__)
    assert not fitted.opacities.requires_grad


@pytest.mark.parametrize("roll", [0.0, 90.0])
def test_tilted_plane_pixel_covariance_and_rendering(roll: float) -> None:
    angle = math.radians(roll)
    camera = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0, 0],
            [math.sin(angle), math.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device="cuda",
    )
    intrinsic = torch.tensor([[96.0, 0, 16], [0, 96.0, 16], [0, 0, 1]], device="cuda")
    yy, xx = torch.meshgrid(torch.arange(32, device="cuda"), torch.arange(32, device="cuda"), indexing="ij")
    rays = torch.stack(((xx + 0.5 - 16) / 96, (yy + 0.5 - 16) / 96, torch.ones_like(xx)), dim=-1)
    normal = camera[:3, :3] @ torch.tensor([0.0, -0.95, -math.sqrt(1 - 0.95**2)], device="cuda")
    depth = (2 * normal[2] / (rays @ normal))[None, None]
    scene = GaussianSurfels.from_rgbd(
        torch.full((1, 3, 32, 32), 0.6, device="cuda"),
        depth,
        normal[None, :, None, None].expand(1, 3, 32, 32),
        intrinsic=intrinsic,
        world_to_camera=camera,
        mask=torch.ones_like(depth).bool(),
        layer=1,
    )
    rotations = quaternion_xyzw_to_rotation_matrix(scene.quaternions[:, (1, 2, 3, 0)])
    covariance = rotations @ torch.diag_embed(scene.scales.square()) @ rotations.transpose(-1, -2)
    points = scene.means @ camera[:3, :3].T
    z = points[:, 2]
    jacobian = torch.zeros(32 * 32, 2, 3, device="cuda")
    jacobian[:, 0, 0] = jacobian[:, 1, 1] = 96 / z
    jacobian[:, 0, 2] = -96 * points[:, 0] / z.square()
    jacobian[:, 1, 2] = -96 * points[:, 1] / z.square()
    projection = jacobian @ camera[:3, :3]
    pixel_covariance = projection @ covariance @ projection.transpose(-1, -2)
    torch.testing.assert_close(
        pixel_covariance, torch.eye(2, device="cuda").expand(32 * 32, 2, 2), rtol=2e-5, atol=2e-5
    )
    rgb, alpha, rendered_depth = scene.render(camera, intrinsic, height=32, width=32)
    assert torch.isfinite(rgb).all() and torch.isfinite(rendered_depth).all()
    assert (alpha[:, :, 2:-2, 2:-2] > 0.5).all()
    assert (rendered_depth[alpha > 0.5] > 0).all()


def test_median_depth_selects_the_occlusion_layer_and_marks_transparency() -> None:
    image = torch.full((1, 3, 16, 16), 0.6, device="cuda")
    normal = torch.tensor([0.0, 0, -1], device="cuda")[None, :, None, None].expand_as(image)
    camera = torch.eye(4, device="cuda")
    intrinsic = torch.tensor([[64.0, 0, 8], [0, 64, 8], [0, 0, 1]], device="cuda")
    near_depth = torch.full((1, 1, 16, 16), 2.0, device="cuda")
    near = GaussianSurfels.from_rgbd(
        image,
        near_depth,
        normal,
        intrinsic=intrinsic,
        world_to_camera=camera,
        mask=torch.ones_like(near_depth).bool(),
        layer=2,
        footprint=0.7,
        opacity=0.15,
    )
    _, alpha, depth = near.render(camera, intrinsic, height=16, width=16)
    assert (alpha < 0.5).all() and (depth == -1).all()
    far = GaussianSurfels.from_rgbd(
        image,
        near_depth * 2,
        normal,
        intrinsic=intrinsic,
        world_to_camera=camera,
        mask=torch.ones_like(near_depth).bool(),
        layer=1,
        footprint=0.7,
    )
    _, alpha, depth = near.append(far).render(camera, intrinsic, height=16, width=16)
    assert (alpha[..., 2:-2, 2:-2] > 0.5).all()
    torch.testing.assert_close(depth[..., 2:-2, 2:-2], torch.full_like(depth[..., 2:-2, 2:-2], 4), rtol=0, atol=0)


def test_median_depth_continues_after_an_exact_half_transmission() -> None:
    scene = GaussianSurfels(
        torch.tensor([[0.0, 0, 2], [0.0, 0, 4]], device="cuda"),
        torch.full((2, 3), 0.02, device="cuda"),
        torch.tensor([[1.0, 0, 0, 0]], device="cuda").expand(2, 4).contiguous(),
        torch.full((2, 3), 0.6, device="cuda"),
        torch.full((2,), 0.5, device="cuda"),
        torch.ones(2, device="cuda", dtype=torch.long),
    )
    rgb, alpha, median = scene.render(
        torch.eye(4, device="cuda"),
        torch.tensor([[64.0, 0, 0.5], [0, 64, 0.5], [0, 0, 1]], device="cuda"),
        height=1,
        width=1,
    )
    torch.testing.assert_close(alpha, torch.full_like(alpha, 0.75), rtol=0, atol=0)
    torch.testing.assert_close(median, torch.full_like(median, 4), rtol=0, atol=0)
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.625), rtol=0, atol=1e-7)


@pytest.mark.parametrize("name", ["_raster_forward", "_raster_backward"])
def test_changing_intersection_counts_do_not_specialize_compositing(name: str) -> None:
    from worldfoundry.core.kernels import triton_gaussian_raster

    kernel = getattr(triton_gaussian_raster, name)
    assert kernel.arg_names.index("INTERSECTIONS") not in kernel.constexprs


@pytest.mark.parametrize("include_alpha", [False, True])
def test_projected_compositing_gradients_match_torch_reference(include_alpha: bool) -> None:
    from worldfoundry.core.kernels.triton_gaussian_raster import gaussian_raster_rgb

    parameters = [
        torch.tensor([[[0.4, 0.7], [1.5, 0.5]]], device="cuda"),
        torch.tensor([[[0.7, 0.1, 0.9], [0.8, -0.2, 1.1]]], device="cuda"),
        torch.tensor([0.6, 1.0], device="cuda"),
        torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]], device="cuda"),
    ]
    actual_parameters = [value.clone().requires_grad_() for value in parameters]
    reference_parameters = [value.clone().requires_grad_() for value in parameters]
    actual_rgb, actual_alpha = gaussian_raster_rgb(
        means2d=actual_parameters[0],
        conics=actual_parameters[1],
        opacities=actual_parameters[2],
        colors=actual_parameters[3],
        isect_offsets=torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda"),
        flatten_ids=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        height=2,
        width=3,
    )
    means, conics, opacities, colors = reference_parameters
    yy, xx = torch.meshgrid(torch.arange(2, device="cuda"), torch.arange(3, device="cuda"), indexing="ij")
    transmission = torch.ones((2, 3), device="cuda")
    rgb = torch.zeros((3, 2, 3), device="cuda")
    for index in range(2):
        dx, dy = means[0, index, 0] - (xx + 0.5), means[0, index, 1] - (yy + 0.5)
        a, b, c = conics[0, index].unbind()
        exponent = 0.5 * (a * dx.square() + c * dy.square()) + b * dx * dy
        alpha = (opacities[index] * torch.exp(-exponent)).clamp_max(0.99)
        rgb = rgb + (transmission * alpha)[None] * colors[index, :, None, None]
        transmission = transmission * (1 - alpha)
    reference_rgb = (rgb + transmission[None] * 0.7)[None]
    reference_alpha = (1 - transmission)[None, None]
    torch.testing.assert_close(actual_rgb, reference_rgb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_alpha, reference_alpha, rtol=1e-5, atol=1e-6)
    weights = torch.linspace(0.1, 1.8, 18, device="cuda").reshape_as(actual_rgb)
    actual_loss = (actual_rgb * weights).sum()
    reference_loss = (reference_rgb * weights).sum()
    if include_alpha:
        alpha_weights = torch.linspace(0.2, 1.2, 6, device="cuda").reshape_as(actual_alpha)
        actual_loss = actual_loss + (actual_alpha * alpha_weights).sum()
        reference_loss = reference_loss + (reference_alpha * alpha_weights).sum()
    actual_loss.backward()
    reference_loss.backward()
    for actual, reference in zip(actual_parameters, reference_parameters):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-4, atol=2e-5)


def test_opaque_tail_is_composited_before_early_termination() -> None:
    scene = GaussianSurfels(
        torch.tensor([[0.0, 0, 2], [0.0, 0, 4]], device="cuda"),
        torch.full((2, 3), 0.02, device="cuda"),
        torch.tensor([[1.0, 0, 0, 0]], device="cuda").expand(2, 4).contiguous(),
        torch.tensor([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([0.91, 1.0], device="cuda"),
        torch.ones(2, device="cuda", dtype=torch.long),
    )
    rgb, alpha, _ = scene.render(
        torch.eye(4, device="cuda"),
        torch.tensor([[64.0, 0, 0.5], [0, 64, 0.5], [0, 0, 1]], device="cuda"),
        height=1,
        width=1,
    )
    torch.testing.assert_close(alpha, torch.full_like(alpha, 0.9991), rtol=0, atol=1e-7)
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.18263), rtol=0, atol=1e-7)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shift", [0, 1])
def test_subpixel_gaussian_support_crosses_tile_boundaries(axis: int, shift: int) -> None:
    coordinates = [16.5, 16.5]
    coordinates[axis] = 14.4 + shift
    scales = [1 / 32, 0.1 / 32, 0]
    if axis:
        scales[:2] = scales[1::-1]
    scene = GaussianSurfels(
        torch.tensor([[(coordinates[0] - 16) * 2 / 64, (coordinates[1] - 16) * 2 / 64, 2]], device="cuda"),
        torch.tensor([scales], device="cuda"),
        torch.tensor([[1.0, 0, 0, 0]], device="cuda"),
        torch.full((1, 3), 0.2, device="cuda"),
        torch.tensor([0.9], device="cuda"),
        torch.ones(1, dtype=torch.int64, device="cuda"),
    )
    rgb, alpha, _ = scene.render(
        torch.eye(4, device="cuda"),
        torch.tensor([[64.0, 0, 16], [0, 64, 16], [0, 0, 1]], device="cuda"),
        height=32,
        width=32,
    )
    pixel = [16, 16]
    pixel[axis] += shift
    x, y = pixel
    expected_alpha = 0.9 * math.exp(-0.5 * 2.1**2)
    torch.testing.assert_close(alpha[0, 0, y, x], alpha.new_tensor(expected_alpha), rtol=0, atol=1e-6)
    torch.testing.assert_close(
        rgb[0, :, y, x], rgb.new_full((3,), 0.2 * expected_alpha + 0.7 * (1 - expected_alpha)), rtol=0, atol=1e-6
    )


def test_near_plane_matches_the_released_rasterizer_culling_distance() -> None:
    scene_image = torch.full((1, 3, 16, 16), 0.6, device="cuda")
    depth = torch.full((1, 1, 16, 16), 2.0, device="cuda")
    scene = GaussianSurfels.from_rgbd(
        scene_image,
        depth,
        torch.tensor([0.0, 0, -1], device="cuda")[None, :, None, None].expand_as(scene_image),
        intrinsic=torch.tensor([[16.0, 0, 8], [0, 16, 8], [0, 0, 1]], device="cuda"),
        world_to_camera=torch.eye(4, device="cuda"),
        mask=torch.ones_like(depth).bool(),
        layer=1,
    )
    camera = torch.eye(4, device="cuda")
    camera[2, 3] = -1.85
    rgb, alpha, median = scene.render(
        camera, torch.tensor([[16.0, 0, 8], [0, 16, 8], [0, 0, 1]], device="cuda"), height=16, width=16
    )
    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.7), rtol=0, atol=0)
    assert (alpha == 0).all() and (median == -1).all()


@pytest.mark.parametrize("parameter_name", ["log_scale", "quaternion"])
def test_unblurred_rasterizer_gradients_match_central_finite_differences(parameter_name: str) -> None:
    camera = torch.eye(4, device="cuda")
    intrinsic = torch.tensor([[64.0, 0, 8], [0, 64, 8], [0, 0, 1]], device="cuda")
    log_scale = torch.tensor([[0.018, 0.028, 0.010]], device="cuda").log()
    quaternion = torch.nn.functional.normalize(torch.tensor([[0.96, 0.12, 0.08, 0.19]], device="cuda"), dim=-1)
    parameter = (log_scale if parameter_name == "log_scale" else quaternion).detach().clone().requires_grad_()

    def objective(value: torch.Tensor) -> torch.Tensor:
        scene = GaussianSurfels(
            torch.tensor([[0.012, 0.007, 2.0]], device="cuda"),
            (value if parameter_name == "log_scale" else log_scale).exp(),
            torch.nn.functional.normalize(value if parameter_name == "quaternion" else quaternion, dim=-1),
            torch.tensor([[0.05, 0.15, 0.25]], device="cuda"),
            torch.tensor([0.4], device="cuda"),
            torch.ones(1, device="cuda", dtype=torch.long),
        )
        rgb = scene._rasterize(camera, intrinsic, height=16, width=16)[0]
        return (rgb[:, :, 7:9, 7:9] - 0.7).sum()

    analytic = torch.autograd.grad(objective(parameter), parameter)[0]
    finite = torch.empty_like(parameter)
    with torch.no_grad():
        for index in range(parameter.numel()):
            plus, minus = parameter.clone(), parameter.clone()
            plus.flatten()[index] += 0.001
            minus.flatten()[index] -= 0.001
            finite.flatten()[index] = (objective(plus) - objective(minus)) / 0.002
    torch.testing.assert_close(analytic, finite, rtol=0.01, atol=0.0003)


def test_new_equal_depth_surface_fits_in_front_of_the_frozen_scene() -> None:
    from worldfoundry.core.geometry.surfels import reconstruction_loss

    camera = torch.eye(4, device="cuda")
    intrinsic = torch.tensor([[64.0, 0, 8], [0, 64, 8], [0, 0, 1]], device="cuda")
    old = GaussianSurfels(
        torch.tensor([[0.012, 0.007, 2.0]], device="cuda"),
        torch.full((1, 3), 0.04, device="cuda"),
        torch.tensor([[1.0, 0, 0, 0]], device="cuda"),
        torch.full((1, 3), 0.8, device="cuda"),
        torch.tensor([0.9], device="cuda"),
        torch.ones(1, device="cuda", dtype=torch.long),
    )
    new = replace(
        old, colors=torch.tensor([[0.1, 0.2, 0.3]], device="cuda"), opacities=old.opacities.new_full((1,), 0.15)
    )
    scene = new.append(old)
    target_scene = replace(new, opacities=old.opacities.new_full((1,), 0.95)).append(old)
    target = target_scene._rasterize(camera, intrinsic, height=16, width=16)[0].detach()
    initial = scene._rasterize(camera, intrinsic, height=16, width=16)[0]
    reverse = old.append(new)._rasterize(camera, intrinsic, height=16, width=16)[0]
    assert reconstruction_loss(initial, target) < reconstruction_loss(reverse, target)
    fitted = scene.refine(target, camera, intrinsic, steps=40, stop=1)
    result = fitted._rasterize(camera, intrinsic, height=16, width=16)[0]
    assert reconstruction_loss(result, target) < reconstruction_loss(initial, target) * 0.1
    for name in scene.__dataclass_fields__:
        torch.testing.assert_close(getattr(fitted, name)[1:], getattr(old, name), rtol=0, atol=0)
    torch.testing.assert_close(scene.opacities, torch.tensor([0.15, 0.9], device="cuda"), rtol=0, atol=0)


def test_deterministic_mixed_normal_frames_cover_y_aligned_surfaces() -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        normals = torch.tensor([[0.0, 0, -1], [0.0, 1, 0], [0.0, -1, 0]], device="cuda")
        depth = torch.full((1, 1, 1, 3), 2.0, device="cuda")
        scene = GaussianSurfels.from_rgbd(
            torch.ones((1, 3, 1, 3), device="cuda"),
            depth,
            normals.T[None, :, None],
            intrinsic=torch.eye(3, device="cuda"),
            world_to_camera=torch.eye(4, device="cuda"),
            mask=torch.ones_like(depth).bool(),
            layer=1,
        )
        rotations = quaternion_xyzw_to_rotation_matrix(scene.quaternions[:, (1, 2, 3, 0)])
        torch.testing.assert_close(rotations[:, :, 2], normals, rtol=0, atol=1e-6)
        assert torch.isfinite(scene.scales).all() and (scene.scales[:, 2] == 0).all()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
