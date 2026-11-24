"""Camera-pose preprocessing used by VideoX-Fun inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import rearrange


class _Camera:
    def __init__(self, values: list[float]) -> None:
        self.fx, self.fy, self.cx, self.cy = values[1:5]
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3] = np.asarray(values[7:], dtype=np.float32).reshape(3, 4)
        self.world_to_camera = world_to_camera
        self.camera_to_world = np.linalg.inv(world_to_camera)


def _relative_poses(cameras: list[_Camera]) -> np.ndarray:
    target = np.eye(4, dtype=np.float32)
    absolute_to_relative = target @ cameras[0].world_to_camera
    poses = [target]
    poses.extend(absolute_to_relative @ camera.camera_to_world for camera in cameras[1:])
    return np.asarray(poses, dtype=np.float32)


def _plucker_rays(
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    device: str | torch.device,
) -> torch.Tensor:
    batch = intrinsics.shape[0]
    y, x = torch.meshgrid(
        torch.linspace(
            0,
            height - 1,
            height,
            device=device,
            dtype=camera_to_world.dtype,
        ),
        torch.linspace(
            0,
            width - 1,
            width,
            device=device,
            dtype=camera_to_world.dtype,
        ),
        indexing="ij",
    )
    x = x.reshape(1, 1, height * width).expand(batch, 1, -1) + 0.5
    y = y.reshape(1, 1, height * width).expand(batch, 1, -1) + 0.5
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    z = torch.ones_like(x)
    x_direction = (x - cx) / fx * z
    y_direction = (y - cy) / fy * z
    z = z.expand_as(y_direction)
    directions = torch.stack((x_direction, y_direction, z), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    ray_directions = directions @ camera_to_world[..., :3, :3].transpose(-1, -2)
    ray_origins = camera_to_world[..., :3, 3].unsqueeze(-2).expand_as(ray_directions)
    rays = torch.cat(
        (torch.cross(ray_origins, ray_directions, dim=-1), ray_directions),
        dim=-1,
    )
    return rays.reshape(
        batch,
        camera_to_world.shape[1],
        height,
        width,
        6,
    )


def process_pose_file(
    pose_file_path: str | Path,
    width: int = 672,
    height: int = 384,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a CameraCtrl pose file to a VideoX-Fun Plücker control tensor."""
    with Path(pose_file_path).open(encoding="utf-8") as handle:
        rows = [line.strip().split() for line in handle.readlines()[1:]]
    if not rows:
        raise ValueError(f"Camera pose file is empty: {pose_file_path}")
    cameras = [_Camera([float(value) for value in row]) for row in rows]

    sample_ratio = width / height
    source_ratio = original_pose_width / original_pose_height
    if source_ratio > sample_ratio:
        resized_width = height * source_ratio
        for camera in cameras:
            camera.fx *= resized_width / width
    else:
        resized_height = width / source_ratio
        for camera in cameras:
            camera.fy *= resized_height / height

    intrinsics = torch.as_tensor(
        [
            [
                camera.fx * width,
                camera.fy * height,
                camera.cx * width,
                camera.cy * height,
            ]
            for camera in cameras
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    camera_to_world = torch.as_tensor(
        _relative_poses(cameras),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    rays = (
        _plucker_rays(
            intrinsics,
            camera_to_world,
            height,
            width,
            device,
        )[0]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return rearrange(rays.unsqueeze(0), "b f c h w -> b f h w c")[0]


__all__ = ["process_pose_file"]
