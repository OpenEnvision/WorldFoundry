"""Inference-only state encoding and action decoding for Hy-VLA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from worldfoundry.core.geometry.transforms import (
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
)


def _xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(quaternion, dtype=torch.float64)
    return quaternion_xyzw_to_rotation_matrix(tensor).cpu().numpy()


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(matrix, dtype=torch.float64)
    return rotation_matrix_to_quaternion_xyzw(tensor).cpu().numpy()


def pose16_wxyz_to_posrot20(pose: Any) -> np.ndarray:
    """Encode dual-arm ``[xyz, quat_wxyz, grip] * 2`` as Hy-VLA's 20-D state."""

    value = np.asarray(pose, dtype=np.float32)
    if value.shape != (16,):
        raise ValueError(f"Expected a 16-D dual-arm pose, got {value.shape}")
    encoded: list[np.ndarray] = []
    for offset in (0, 8):
        quaternion_wxyz = value[offset + 3 : offset + 7]
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        rotation = _xyzw_to_matrix(quaternion_xyzw)
        encoded.append(
            np.concatenate(
                [
                    value[offset : offset + 3],
                    rotation[0],
                    rotation[1],
                    value[offset + 7 : offset + 8],
                ]
            )
        )
    return np.concatenate(encoded).astype(np.float32, copy=False)


def pose16_wxyz_to_xyzw(pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float32).copy()
    if value.shape != (16,):
        raise ValueError(f"Expected a 16-D dual-arm pose, got {value.shape}")
    value[3:7] = value[[4, 5, 6, 3]]
    value[11:15] = value[[12, 13, 14, 11]]
    return value


def pose16_xyzw_to_wxyz(poses: Any) -> np.ndarray:
    value = np.asarray(poses, dtype=np.float32).copy()
    if value.ndim != 2 or value.shape[-1] != 16:
        raise ValueError(f"Expected a (T, 16) dual-arm pose chunk, got {value.shape}")
    value[:, 3:7] = value[:, [6, 3, 4, 5]]
    value[:, 11:15] = value[:, [14, 11, 12, 13]]
    return value


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = rotation_6d[:, :3]
    second = rotation_6d[:, 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=1)


def _relative_arm_to_pose(relative: np.ndarray, start_xyzw: np.ndarray) -> np.ndarray:
    count = relative.shape[0]
    start = np.eye(4, dtype=np.float64)
    start[:3, :3] = _xyzw_to_matrix(start_xyzw[3:7])
    start[:3, 3] = start_xyzw[:3]
    delta = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    delta[:, :3, :3] = _rotation_6d_to_matrix(relative[:, 3:9])
    delta[:, :3, 3] = relative[:, :3]
    result = start @ delta
    return np.concatenate(
        [result[:, :3, 3], _matrix_to_xyzw(result[:, :3, :3])],
        axis=1,
    )


def relative20_to_pose16_xyzw(relative: Any, start_pose_xyzw: Any) -> np.ndarray:
    """Decode official 20-D RT-relative chunks into absolute dual-arm poses."""

    chunk = np.asarray(relative, dtype=np.float32)
    start = np.asarray(start_pose_xyzw, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[-1] != 20 or start.shape != (16,):
        raise ValueError(f"Expected relative=(T,20), start=(16,), got {chunk.shape}, {start.shape}")
    left = _relative_arm_to_pose(chunk[:, :9], start[:7])
    right = _relative_arm_to_pose(chunk[:, 10:19], start[8:15])
    return np.concatenate(
        [left, chunk[:, 9:10], right, chunk[:, 19:20]],
        axis=1,
    ).astype(np.float32, copy=False)


def absolute20_to_pose16_xyzw(absolute: Any) -> np.ndarray:
    """Decode absolute ``[xyz, rot6d, grip] * 2`` chunks into 16-D poses."""

    chunk = np.asarray(absolute, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[-1] != 20:
        raise ValueError(f"Expected an absolute (T, 20) chunk, got {chunk.shape}")
    arms: list[np.ndarray] = []
    for offset in (0, 10):
        rotation = _rotation_6d_to_matrix(chunk[:, offset + 3 : offset + 9])
        arms.append(
            np.concatenate(
                [
                    chunk[:, offset : offset + 3],
                    _matrix_to_xyzw(rotation),
                    chunk[:, offset + 9 : offset + 10],
                ],
                axis=1,
            )
        )
    return np.concatenate(arms, axis=1).astype(np.float32, copy=False)


def _slerp_half(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    second = second.copy()
    dots = np.sum(first * second, axis=1, keepdims=True)
    second[dots[:, 0] < 0] *= -1
    dots = np.abs(np.clip(dots, -1.0, 1.0))
    near = dots[:, 0] > 0.9995
    result = np.empty_like(first)
    if near.any():
        linear = first[near] + second[near]
        result[near] = linear / np.maximum(np.linalg.norm(linear, axis=1, keepdims=True), 1e-12)
    if (~near).any():
        theta = np.arccos(dots[~near])
        denominator = np.sin(theta)
        weight = np.sin(theta * 0.5) / denominator
        result[~near] = weight * first[~near] + weight * second[~near]
    return result


def blend_pose16_xyzw(relative: Any, absolute: Any) -> np.ndarray:
    """Blend relative and absolute pose chunks exactly at a 1:1 ratio."""

    first = np.asarray(relative, dtype=np.float32)
    second = np.asarray(absolute, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2 or first.shape[-1] != 16:
        raise ValueError(f"Expected matching (T, 16) pose chunks, got {first.shape}, {second.shape}")
    output = (first + second) * 0.5
    output[:, 3:7] = _slerp_half(first[:, 3:7], second[:, 3:7])
    output[:, 11:15] = _slerp_half(first[:, 11:15], second[:, 11:15])
    return output.astype(np.float32, copy=False)


__all__ = [
    "absolute20_to_pose16_xyzw",
    "blend_pose16_xyzw",
    "pose16_wxyz_to_posrot20",
    "pose16_wxyz_to_xyzw",
    "pose16_xyzw_to_wxyz",
    "relative20_to_pose16_xyzw",
]
