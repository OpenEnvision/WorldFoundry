"""Interactive camera trajectories for DreamX-World inference."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from worldfoundry.core.geometry.transforms import euler_angles_to_rotation_matrix_zyx

_ACTIONS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "j": "left_rot",
    "l": "right_rot",
    "i": "up_rot",
    "k": "down_rot",
}


def _translation_step(
    action: str,
    rotation: np.ndarray,
    amount: float,
    duration: int,
) -> np.ndarray:
    pitch, yaw = np.radians(rotation[:2])
    if action in {"forward", "backward"}:
        direction = np.asarray(
            [-math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)]
        )
        return direction * amount * (1 if action == "forward" else -1) / duration
    if action in {"left", "right"}:
        direction = np.asarray([math.cos(yaw), 0.0, math.sin(yaw)])
        return direction * amount * (-1 if action == "left" else 1) / duration
    return np.zeros(3)


def _rotation_step(action: str, amount: float, duration: int) -> np.ndarray:
    output = np.zeros(3)
    if action == "left_rot":
        output[1] = amount
    elif action == "right_rot":
        output[1] = -amount
    elif action == "up_rot":
        output[0] = -amount
    elif action == "down_rot":
        output[0] = amount
    return output / duration


def _interpolate_w2c(w2cs: np.ndarray, count: int) -> np.ndarray:
    source = np.arange(len(w2cs), dtype=np.float64)
    target = np.linspace(0.0, len(w2cs) - 1, count)
    translations = interp1d(source, w2cs[:, :3, 3], axis=0)(target)
    quaternions = Rotation.from_matrix(w2cs[:, :3, :3]).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0:
            quaternions[index] *= -1
    rotations = Slerp(source, Rotation.from_quat(quaternions))(target).as_matrix()
    output = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    output[:, :3, :3] = rotations
    output[:, :3, 3] = translations
    return output


def camera_condition_from_actions(
    action_ids: Sequence[str],
    speeds: Sequence[float],
    *,
    duration: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert per-frame keyboard actions into latent-frame PRoPE cameras."""

    if len(action_ids) != len(speeds):
        raise ValueError("DreamX camera actions and speeds must have equal lengths.")
    position = np.zeros(3)
    rotation = np.zeros(3)
    w2cs = [np.eye(4)]
    for action_id, speed in zip(action_ids, speeds):
        actions = [_ACTIONS[key] for key in action_id]
        translation = sum(
            (_translation_step(action, rotation, speed, duration) for action in actions),
            start=np.zeros(3),
        )
        rotation_delta = sum(
            (_rotation_step(action, speed * 10.0, duration) for action in actions),
            start=np.zeros(3),
        )
        for step in range(1, duration + 1):
            step_position = position + translation * step
            step_rotation = rotation + rotation_delta * step
            matrix = np.eye(4)
            matrix[:3, :3] = euler_angles_to_rotation_matrix_zyx(
                np.radians(step_rotation)
            )
            matrix[:3, 3] = -matrix[:3, :3] @ step_position
            w2cs.append(matrix)
        position += translation * duration
        rotation += rotation_delta * duration
    if len(w2cs) != target_length:
        raise ValueError(
            f"DreamX camera trajectory has {len(w2cs)} frames; expected {target_length}."
        )

    latent_count = 1 + (target_length - 1) // 4
    w2cs = _interpolate_w2c(np.asarray(w2cs), latent_count)
    relative_c2w = np.stack([w2cs[0] @ np.linalg.inv(matrix) for matrix in w2cs])
    viewmats = torch.as_tensor(
        np.linalg.inv(relative_c2w), dtype=dtype, device=device
    )
    intrinsics = torch.zeros((latent_count, 3, 3), dtype=dtype, device=device)
    intrinsics[:, 0, 0] = 969.6969696969696 / 1920.0
    intrinsics[:, 1, 1] = 969.6969696969696 / 1080.0
    intrinsics[:, 2, 2] = 1.0
    return {"viewmats": viewmats, "K": intrinsics}
