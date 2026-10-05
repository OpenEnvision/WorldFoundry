"""Camera-pose to Plücker embedding conversion used by MagicWorld inference."""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange


class Camera:
    def __init__(self, entry: list[float]) -> None:
        if len(entry) == 19:
            self.fx, self.fy, self.cx, self.cy = entry[1:5]
            w2c = np.eye(4)
            w2c[:3, :] = np.asarray(entry[7:]).reshape(3, 4)
        elif len(entry) == 25:
            intrinsics = np.asarray(entry[:9]).reshape(3, 3)
            self.fx = intrinsics[0, 0]
            self.fy = intrinsics[1, 1]
            self.cx = intrinsics[0, 2]
            self.cy = intrinsics[1, 2]
            w2c = np.asarray(entry[9:]).reshape(4, 4)
        else:
            raise ValueError(f"Unsupported camera row length {len(entry)}; expected 19 or 25")
        self.w2c_mat = w2c
        self.c2w_mat = np.linalg.inv(w2c)


def _relative_poses(cameras: list[Camera]) -> np.ndarray:
    target = np.eye(4)
    target[1, 3] = 0
    absolute_to_relative = target @ cameras[0].w2c_mat
    poses = [target, *(absolute_to_relative @ camera.c2w_mat for camera in cameras[1:])]
    return np.asarray(poses, dtype=np.float32)


def _ray_condition(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    batch = intrinsics.shape[0]
    rows, columns = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=device, dtype=c2w.dtype),
        torch.linspace(0, width - 1, width, device=device, dtype=c2w.dtype),
        indexing="ij",
    )
    columns = columns.reshape(1, 1, height * width).expand(batch, 1, -1) + 0.5
    rows = rows.reshape(1, 1, height * width).expand(batch, 1, -1) + 0.5
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    # Broadcasting the per-frame intrinsics expands x/y from [B, 1, HW] to
    # [B, F, HW].  Build z from that expanded shape as well; using
    # ``ones_like(columns)`` leaves z at [B, 1, HW] and makes torch.stack fail
    # for multi-frame camera trajectories.
    z = torch.ones_like((columns - cx) / fx)
    directions = torch.stack(((columns - cx) / fx, (rows - cy) / fy, z), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3][:, :, None].expand_as(rays_d)
    plucker = torch.cat([torch.linalg.cross(rays_o, rays_d), rays_d], dim=-1)
    return plucker.reshape(batch, c2w.shape[1], height, width, 6)


def process_pose_file(
    pose_file_path: str,
    width: int = 672,
    height: int = 384,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
    device: str = "cpu",
    return_poses: bool = False,
):
    with open(pose_file_path, encoding="utf-8") as file:
        rows = [line.strip().split() for line in file.readlines()[1:] if line.strip()]
    raw_cameras = [[float(value) for value in row] for row in rows]
    if return_poses:
        return raw_cameras

    cameras = [Camera(row) for row in raw_cameras]
    sample_ratio = width / height
    original_ratio = original_pose_width / original_pose_height
    if original_ratio > sample_ratio:
        resized_width = height * original_ratio
        for camera in cameras:
            camera.fx = resized_width * camera.fx / width
    else:
        resized_height = width / original_ratio
        for camera in cameras:
            camera.fy = resized_height * camera.fy / height

    intrinsics = np.asarray(
        [
            [camera.fx * width, camera.fy * height, camera.cx * width, camera.cy * height]
            for camera in cameras
        ],
        dtype=np.float32,
    )
    intrinsic_tensor = torch.as_tensor(intrinsics, device=device)[None]
    c2w_tensor = torch.as_tensor(_relative_poses(cameras), device=device)[None]
    plucker = _ray_condition(intrinsic_tensor, c2w_tensor, height, width, device)[0]
    return rearrange(plucker.permute(0, 3, 1, 2)[None], "b f c h w -> b f h w c")[0].contiguous()


__all__ = ["process_pose_file"]
