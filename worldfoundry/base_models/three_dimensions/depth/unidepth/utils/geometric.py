"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple

import torch
from torch.nn import functional as F


@torch.jit.script
def generate_rays(camera_intrinsics: torch.Tensor, image_shape: Tuple[int, int], noisy: bool = False):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat([pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2)  # (H, W, 3)
    ray_directions = torch.matmul(intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1))  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    # pitch = torch.asin(ray_directions[..., 1])
    # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


@torch.jit.script
def spherical_zbuffer_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def downsample(data: torch.Tensor, downsample_factor: int = 2):
    N, _, H, W = data.shape
    data = data.view(
        N,
        H // downsample_factor,
        downsample_factor,
        W // downsample_factor,
        downsample_factor,
        1,
    )
    data = data.permute(0, 1, 3, 5, 2, 4).contiguous()
    data = data.view(-1, downsample_factor * downsample_factor)
    data_tmp = torch.where(data == 0.0, 1e5 * torch.ones_like(data), data)
    data = torch.min(data_tmp, dim=-1).values
    data = data.view(N, 1, H // downsample_factor, W // downsample_factor)
    data = torch.where(data > 1000, torch.zeros_like(data), data)
    return data


@torch.jit.script
def flat_interpolate(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(0, 3, 1, 2)  # b c h w
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(flat_tensor.shape[0], -1, new[0] * new[1]).permute(0, 2, 1)  # b (h w) c
    return flat_tensor_interp.contiguous()
