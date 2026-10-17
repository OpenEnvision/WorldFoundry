# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Normal-oriented Gaussians, native compositing and standard 32-byte splat IO.

Camera matrices use OpenCV world-to-camera (+X right, +Y down, +Z forward).
Quaternions are WXYZ. Positions and scales share the caller's scene units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .transforms import rotation_matrix_to_quaternion_xyzw
from .warp import unproject_depth


@dataclass(frozen=True)
class GaussianSurfels:
    """Activated Gaussian parameters; layer labels are caller-owned integers."""

    means: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    layers: torch.Tensor

    @classmethod
    def from_rgbd(
        cls,
        image: torch.Tensor,
        depth: torch.Tensor,
        normals: torch.Tensor,
        *,
        intrinsic: torch.Tensor,
        world_to_camera: torch.Tensor,
        mask: torch.Tensor,
        layer: int,
        footprint: float = 1.0,
        opacity: float = 0.99,
        thickness: float = 0.0,
    ) -> GaussianSurfels:
        """Unproject BCHW RGB/depth/normals into normal-oriented Gaussians.

        Normals are OpenCV camera-space and samples lie at pixel centers.
        Tangent standard deviations project to ``footprint`` pixels, capped
        at ten footprints on grazing surfaces. ``thickness`` sets the
        normal-axis scale in depth/focal-length units; zero gives a disk.
        """
        if image.shape[0] != 1 or depth.shape != image[:, :1].shape or normals.shape != image.shape:
            raise ValueError("surfel initialization requires aligned single-view BCHW RGB, depth and normals")
        distance = depth[0, 0].T.reshape(-1)
        selected = mask[0, 0].T.reshape(-1).bool() & torch.isfinite(distance) & (distance > 0)
        sample_intrinsic = intrinsic.clone()
        sample_intrinsic[:2, 2] -= 0.5
        points = unproject_depth(depth, world_to_camera=world_to_camera[None], intrinsic=sample_intrinsic[None])
        points = points[0].transpose(0, 1).reshape(-1, 3)[selected]
        normal = functional.normalize(normals[0].permute(2, 1, 0).reshape(-1, 3)[selected], dim=-1)
        reference = torch.where(
            (normal[:, 1].abs() > 0.9)[:, None],
            normal.new_tensor([1, 0, 0]),
            normal.new_tensor([0, 1, 0]),
        )
        tangent = functional.normalize(torch.linalg.cross(reference, normal), dim=-1)
        bitangent = torch.linalg.cross(normal, tangent)
        basis = torch.stack((tangent, bitangent), dim=-1)
        distance = distance[selected]
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        rays = camera_points / distance[:, None]
        # Restrict the perspective Jacobian to the camera-space tangent plane.
        projection = torch.stack(
            (
                (basis[:, 0] - rays[:, 0, None] * basis[:, 2]) * (intrinsic[0, 0] / distance)[:, None],
                (basis[:, 1] - rays[:, 1, None] * basis[:, 2]) * (intrinsic[1, 1] / distance)[:, None],
            ),
            dim=1,
        )
        a = projection[:, :, 0].square().sum(dim=1)
        b = (projection[:, :, 0] * projection[:, :, 1]).sum(dim=1)
        c = projection[:, :, 1].square().sum(dim=1)
        angle = 0.5 * torch.atan2(2 * b, a - c)
        cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
        principal = tangent * cosine + bitangent * sine
        second = torch.linalg.cross(normal, principal)
        responses = torch.stack(
            (
                (projection[:, :, 0] * cosine + projection[:, :, 1] * sine).norm(dim=1),
                (-projection[:, :, 0] * sine + projection[:, :, 1] * cosine).norm(dim=1),
            ),
            dim=-1,
        )
        minimum_response = 0.1 * torch.minimum(intrinsic[0, 0], intrinsic[1, 1]) / distance
        tangent_scales = responses.clamp_min(minimum_response[:, None]).reciprocal() * footprint
        normal_scale = thickness * distance / torch.maximum(intrinsic[0, 0], intrinsic[1, 1])
        scales = torch.cat((tangent_scales, normal_scale[:, None]), dim=-1)
        camera_rotation = world_to_camera[:3, :3]
        frame = torch.stack((principal @ camera_rotation, second @ camera_rotation, normal @ camera_rotation), dim=-1)
        xyzw = rotation_matrix_to_quaternion_xyzw(frame)
        quaternions = xyzw[:, (3, 0, 1, 2)]
        colors = image[0].permute(2, 1, 0).reshape(-1, 3)[selected]
        return cls(
            points.float(),
            scales.float(),
            quaternions.float(),
            colors.float(),
            torch.full_like(distance, opacity).float(),
            torch.full((len(points),), layer, device=points.device, dtype=torch.int64),
        )

    def append(self, other: GaussianSurfels) -> GaussianSurfels:
        """Return one scene without modifying either source scene."""
        return GaussianSurfels(
            *(torch.cat((getattr(self, name), getattr(other, name))) for name in self.__dataclass_fields__)
        )

    def render(
        self,
        world_to_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        *,
        height: int,
        width: int,
        exclude_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render RGB over gray, accumulated alpha and median depth as BCHW.

        The Gaussian covariance has no additional image-space blur. Median
        depth is -1 where front-to-back opacity does not cross 0.5.
        """
        from worldfoundry.core.kernels.triton_gaussian_median import gaussian_median_depth

        rgb, alpha, projection = self._rasterize(
            world_to_camera, intrinsic, height=height, width=width, exclude_layer=exclude_layer
        )
        if projection is None:
            return rgb, alpha, self.colors.new_full((1, 1, height, width), -1)
        depth = gaussian_median_depth(
            means2d=projection["means2d"],
            conics=projection["conics"],
            opacities=projection["opacities"],
            depths=projection["depths"],
            isect_offsets=projection["isect_offsets"],
            flatten_ids=projection["flatten_ids"],
            height=height,
            width=width,
            tile_size=16,
        )
        return rgb, alpha, depth

    def _rasterize(
        self,
        world_to_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        *,
        height: int,
        width: int,
        exclude_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Share gsplat projections and native RGB/median compositing."""
        from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

        from worldfoundry.core.kernels.triton_gaussian_raster import gaussian_raster_rgb

        keep = torch.ones_like(self.layers, dtype=torch.bool) if exclude_layer is None else self.layers != exclude_layer
        if not keep.any():
            return (
                self.colors.new_full((1, 3, height, width), 0.7),
                self.colors.new_zeros((1, 1, height, width)),
                None,
            )
        opacity = self.opacities[keep].contiguous()
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=self.means[keep].contiguous(),
            covars=None,
            quats=self.quaternions[keep].contiguous(),
            scales=self.scales[keep].contiguous(),
            opacities=opacity,
            viewmats=world_to_camera[None].contiguous(),
            Ks=intrinsic[None].contiguous(),
            width=width,
            height=height,
            near_plane=0.2,
            far_plane=1000.0,
            eps2d=0.0,
            packed=False,
        )
        tile_width, tile_height = math.ceil(width / 16), math.ceil(height / 16)
        _, intersections, indices = isect_tiles(means2d, radii, depths, 16, tile_width, tile_height)
        offsets = isect_offset_encode(intersections, 1, tile_width, tile_height)
        projection = {
            "means2d": means2d,
            "conics": conics,
            "depths": depths,
            "opacities": opacity[None],
            "radii": radii,
            "isect_offsets": offsets,
            "flatten_ids": indices,
        }
        rendered, alpha = gaussian_raster_rgb(
            means2d=means2d,
            conics=conics,
            opacities=opacity,
            colors=self.colors[keep],
            isect_offsets=offsets,
            flatten_ids=indices,
            height=height,
            width=width,
        )
        return rendered, alpha, projection

    def refine(
        self,
        image: torch.Tensor,
        world_to_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        *,
        steps: int,
        start: int = 0,
        stop: int | None = None,
    ) -> GaussianSurfels:
        """Fit Gaussian opacity, scales and orientation in ``[start, stop)``.

        All other Gaussians and all positions/colors stay frozen; particle
        order is retained. ``stop=None`` selects the remaining suffix.
        Iterations include the final evaluation; opacity/extent pruning runs
        every 80 iterations before 15000. This does not train model weights.
        """
        stop = len(self.means) if stop is None else stop
        if not 0 <= start <= stop <= len(self.means):
            raise ValueError("Gaussian refinement requires 0 <= start <= stop <= particle count")
        if not steps or start == stop:
            return self
        with torch.enable_grad():
            opacity = (
                torch.atanh(((self.opacities[start:stop] - 0.5) / 0.51).clamp(-0.999, 0.999))
                .detach()
                .requires_grad_(True)
            )
            scale = self.scales[start:stop].log().detach().requires_grad_(True)
            rotation = self.quaternions[start:stop].detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam(
                [
                    {"params": [opacity], "lr": 0.08},
                    {"params": [scale], "lr": 0.005},
                    {"params": [rotation], "lr": 0.001},
                ],
                eps=1e-15,
                betas=(0.0, 0.99),
            )
            scene = self
            for iteration in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                stop = start + len(opacity)
                fitted = scene._with_parameters(start, opacity, scale, rotation, stop=stop)
                prediction, _, _ = fitted._rasterize(
                    world_to_camera, intrinsic, height=image.shape[-2], width=image.shape[-1]
                )
                loss = reconstruction_loss(prediction, image)
                loss.backward()
                if iteration < 15000 and iteration % 80 == 0:
                    keep = (fitted.opacities[start:stop] >= 0.05) & (fitted.scales[start:stop].amax(dim=1) <= 0.06)
                    all_keep = torch.cat(
                        (
                            torch.ones(start, device=keep.device, dtype=torch.bool),
                            keep,
                            torch.ones(len(scene.means) - stop, device=keep.device, dtype=torch.bool),
                        )
                    )
                    scene = scene.select(all_keep)
                    parameters = []
                    for group in optimizer.param_groups:
                        previous = group["params"][0]
                        replacement = previous[keep].detach().requires_grad_(True)
                        state = optimizer.state.pop(previous, {})
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state:
                                state[key] = state[key][keep]
                        group["params"][0] = replacement
                        optimizer.state[replacement] = state
                        parameters.append(replacement)
                    opacity, scale, rotation = parameters
                    if not len(opacity):
                        break
                if iteration < steps:
                    optimizer.step()
            result = scene._with_parameters(start, opacity, scale, rotation, stop=start + len(opacity))
        return GaussianSurfels(*(getattr(result, name).detach() for name in self.__dataclass_fields__))

    def select(self, mask: torch.Tensor) -> GaussianSurfels:
        """Select particles consistently across every activated parameter."""
        return GaussianSurfels(*(getattr(self, name)[mask] for name in self.__dataclass_fields__))

    def remove_floaters(self, *, mean_squared_distance: float) -> GaussianSurfels:
        """Discard points whose three closest neighbors are too distant."""
        from scipy.spatial import KDTree

        if len(self.means) < 4:
            return self.select(torch.zeros(len(self.means), device=self.means.device, dtype=torch.bool))
        points = self.means.detach().cpu().numpy()
        distance, _ = KDTree(points).query(points, k=4, workers=2)
        keep = np.square(distance[:, 1:].astype(np.float32)).mean(axis=1) <= mean_squared_distance
        return self.select(torch.from_numpy(keep).to(self.means.device))

    def _with_parameters(
        self,
        start: int,
        opacity: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        *,
        stop: int | None = None,
    ) -> GaussianSurfels:
        stop = len(self.means) if stop is None else stop
        scales = scale.exp()
        return GaussianSurfels(
            self.means,
            torch.cat((self.scales[:start], scales, self.scales[stop:])),
            torch.cat((self.quaternions[:start], functional.normalize(rotation, dim=-1), self.quaternions[stop:])),
            self.colors,
            torch.cat((self.opacities[:start], (opacity.tanh() * 0.51).clamp(-0.5, 0.5) + 0.5, self.opacities[stop:])),
            self.layers,
        )

    def save_splat(self, path: str | Path) -> Path:
        """Write standard little-endian position/scale/RGBA/WXYZ records."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = [
            getattr(self, name).detach().cpu().numpy()
            for name in ("means", "scales", "colors", "opacities", "quaternions")
        ]
        means, scales, colors, opacity, rotation = arrays
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("cannot export nonfinite Gaussian parameters")
        records = np.empty(
            len(means),
            dtype=[("position", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rotation", "u1", (4,))],
        )
        records["position"] = means
        records["scale"] = scales
        records["rgba"] = (np.concatenate((colors, opacity[:, None]), axis=1) * 255).clip(0, 255).astype(np.uint8)
        records["rotation"] = (rotation * 128 + 128).clip(0, 255).astype(np.uint8)
        # Preserve initialized particle order; viewers sort by projected depth.
        records.tofile(destination)
        return destination


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return 0.8 L1 + 0.2 DSSIM with an 11-pixel, sigma-1.5 window.

    Local statistics use zero padding and a unit RGB data range. The loss
    preserves gradients for per-scene Gaussian fitting.
    """
    prediction, target = prediction.contiguous(), target.contiguous()
    weights = torch.tensor([math.exp(-((index - 5) ** 2) / 4.5) for index in range(11)], dtype=torch.float32)
    weights = weights / weights.sum()
    channels = prediction.shape[1]
    window = (weights[:, None] * weights[None]).expand(channels, 1, 11, 11).contiguous().to(prediction)
    first = functional.conv2d(prediction, window, padding=5, groups=channels)
    second = functional.conv2d(target, window, padding=5, groups=channels)
    first_square, second_square, product = first.square(), second.square(), first * second
    first_variance = functional.conv2d(prediction * prediction, window, padding=5, groups=channels) - first_square
    second_variance = functional.conv2d(target * target, window, padding=5, groups=channels) - second_square
    covariance = functional.conv2d(prediction * target, window, padding=5, groups=channels) - product
    numerator = (2 * product + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (first_square + second_square + 0.01**2) * (first_variance + second_variance + 0.03**2)
    similarity = (numerator / denominator).mean()
    return 0.8 * (prediction - target).abs().mean() + 0.2 * (1 - similarity)
