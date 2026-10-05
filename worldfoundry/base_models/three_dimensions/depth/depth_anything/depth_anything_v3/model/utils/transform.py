# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Camera pose encoding transformations for Depth Anything v3."""

import torch

from worldfoundry.core.geometry.transforms import (
    quaternion_xyzw_to_rotation_matrix as quat_to_mat,
    rotation_matrix_to_quaternion_xyzw as mat_to_quat,
    standardize_quaternion_xyzw as standardize_quaternion,
)


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None):
    """Convert camera extrinsics and intrinsics to a compact pose encoding."""

    rotation = extrinsics[:, :, :3, :3]
    translation = extrinsics[:, :, :3, 3]
    quaternion = mat_to_quat(rotation)
    height, width = image_size_hw
    fov_h = 2 * torch.atan((height / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((width / 2) / intrinsics[..., 0, 0])
    return torch.cat([translation, quaternion, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw=None):
    """Convert a compact pose encoding back to extrinsics and intrinsics."""

    translation = pose_encoding[..., :3]
    quaternion = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    rotation = quat_to_mat(quaternion)
    extrinsics = torch.cat([rotation, translation[..., None]], dim=-1)

    height, width = image_size_hw
    fy = (height / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (width / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w):
    """Transform camera-local quaternion rotations into world space."""

    batch_size, num_views = cam_quat_xyzw.shape[:2]
    cam_quat_wxyz = torch.cat(
        [
            cam_quat_xyzw[..., 3:4],
            cam_quat_xyzw[..., 0:1],
            cam_quat_xyzw[..., 1:2],
            cam_quat_xyzw[..., 2:3],
        ],
        dim=-1,
    )
    rotmat_cam = quat_to_mat(cam_quat_wxyz.reshape(-1, 4)).reshape(batch_size, num_views, 3, 3)
    rotmat_world = torch.matmul(c2w[..., :3, :3], rotmat_cam)
    return mat_to_quat(rotmat_world.reshape(-1, 3, 3)).reshape(batch_size, num_views, 4)


__all__ = [
    "cam_quat_xyzw_to_world_quat_wxyz",
    "extri_intri_to_pose_encoding",
    "mat_to_quat",
    "pose_encoding_to_extri_intri",
    "quat_to_mat",
    "standardize_quaternion",
]
