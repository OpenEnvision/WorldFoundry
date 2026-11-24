# Copyright (C) 2026 Xiaomi Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Observation and action transforms for Xiaomi-Robotics-1 inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.action_generation._native_policy_runtime import first_present, to_numpy_image

_EEF_REFRAME = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def normalize_tensor(value: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    """Apply dense Gaussian or quantile normalization."""

    mode = str(statistics.get("mode") or "none").lower()
    if mode == "none":
        return value
    if mode == "gaussian":
        mean = torch.as_tensor(statistics["mean"], device=value.device, dtype=value.dtype)
        std = torch.as_tensor(statistics["std"], device=value.device, dtype=value.dtype)
        valid = std.abs() > 1e-5
        return torch.where(valid, (value - mean) / torch.where(valid, std, torch.ones_like(std)), value)
    if mode == "quantile":
        low = torch.as_tensor(statistics["q01"], device=value.device, dtype=value.dtype)
        high = torch.as_tensor(statistics["q99"], device=value.device, dtype=value.dtype)
        span = high - low
        valid = span.abs() > 1e-5
        normalized = 2 * (value - low) / torch.where(valid, span, torch.ones_like(span)) - 1
        return torch.where(valid, normalized, value)
    raise ValueError(f"unsupported normalization mode: {mode!r}")


def denormalize_tensor(value: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    """Invert dense Gaussian or quantile normalization."""

    mode = str(statistics.get("mode") or "none").lower()
    if mode == "none":
        return value
    if mode == "gaussian":
        mean = torch.as_tensor(statistics["mean"], device=value.device, dtype=value.dtype)
        std = torch.as_tensor(statistics["std"], device=value.device, dtype=value.dtype)
        valid = std.abs() > 1e-5
        return torch.where(valid, value * std + mean, value)
    if mode == "quantile":
        low = torch.as_tensor(statistics["q01"], device=value.device, dtype=value.dtype)
        high = torch.as_tensor(statistics["q99"], device=value.device, dtype=value.dtype)
        span = high - low
        valid = span.abs() > 1e-5
        restored = (value + 1) * 0.5 * span + low
        return torch.where(valid, restored, value)
    raise ValueError(f"unsupported normalization mode: {mode!r}")


def center_crop_image(image: Any, *, crop_ratio: float, output_size: Sequence[int]) -> Image.Image:
    """Apply the policy's resize/crop/resize image transform."""

    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1]")
    width, height = (int(output_size[0]), int(output_size[1]))
    array = to_numpy_image(image)
    if array.ndim != 3:
        raise ValueError(f"expected an RGB image, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"expected three image channels, got {array.shape}")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    pil = Image.fromarray(array).resize((width, height), resampling)
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return pil.crop((left, top, left + crop_width, top + crop_height)).resize(
        (width, height),
        resampling,
    )


def resolve_camera_images(
    observation: Mapping[str, Any],
    image: Any,
    camera_keys: Sequence[Sequence[str]],
) -> list[Any]:
    """Resolve the exact three camera slots without silently duplicating views."""

    containers: list[Mapping[str, Any]] = [observation]
    for name in ("images", "vision", "observations"):
        value = observation.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
    if isinstance(image, Mapping):
        containers.append(image)
    positional = list(image) if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)) else []
    resolved: list[Any] = []
    for index, aliases in enumerate(camera_keys):
        selected = None
        for container in containers:
            selected = first_present(container, *aliases)
            if isinstance(selected, Mapping):
                selected = first_present(selected, "color", "colors", "rgb", "image")
            if selected is not None:
                break
        if selected is None and index < len(positional):
            selected = positional[index]
        if selected is None:
            raise KeyError(f"missing camera slot {index}; accepted aliases={tuple(aliases)}")
        resolved.append(selected)
    return resolved


def _state_mapping(observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state = observation.get("state")
    return state if isinstance(state, Mapping) else None


def pack_robot_state(observation: Mapping[str, Any], *, state_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack a bimanual observation into the sparse checkpoint layout."""

    state = _state_mapping(observation)
    if state is None:
        raw = first_present(observation, "state", "proprio", "robot_state")
        if raw is None:
            raise ValueError("Xiaomi-Robotics-1 requires a robot state")
        raw_array = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw_array.size > state_dim:
            raise ValueError(f"state has {raw_array.size} values but the model accepts {state_dim}")
        packed = np.zeros(state_dim, dtype=np.float32)
        packed[: raw_array.size] = raw_array
        if raw_array.size < 16:
            raise ValueError("flat bimanual state must contain at least 16 sparse slots")
        current = {
            "left_arm_joint": packed[0:6].copy(),
            "right_arm_joint": packed[8:14].copy(),
        }
        return packed, current

    required = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError(f"robot state is missing fields: {missing}")
    left_arm = np.asarray(state["left_arm_joint_state"], dtype=np.float32).reshape(-1)
    right_arm = np.asarray(state["right_arm_joint_state"], dtype=np.float32).reshape(-1)
    if left_arm.size < 6 or right_arm.size < 6:
        raise ValueError("each arm must provide at least six joint values")
    packed = np.zeros(state_dim, dtype=np.float32)
    packed[0:6] = left_arm[:6]
    packed[7] = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
    packed[8:14] = right_arm[:6]
    packed[15] = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
    return packed, {
        "left_arm_joint": left_arm[:6].copy(),
        "right_arm_joint": right_arm[:6].copy(),
        "state_mapping": state,
    }


def _quaternion_to_matrix(quaternion_wxyz: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("end-effector quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math_sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math_sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math_sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math_sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))


def _rotvec_to_matrix(rotation_vector: Any) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        skew = np.asarray([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])
        return np.eye(3, dtype=np.float64) + skew
    axis = vector / angle
    skew = np.asarray([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def add_end_effector_state(current: dict[str, Any]) -> None:
    """Add MiBot-frame end-effector poses required for delta restoration."""

    state = current.get("state_mapping")
    if not isinstance(state, Mapping):
        raise ValueError("end-effector action mode requires structured left/right ee poses")
    for side in ("left", "right"):
        key = f"{side}_ee_pose"
        if key not in state:
            raise KeyError(f"end-effector action mode requires {key}")
        pose = np.asarray(state[key], dtype=np.float64).reshape(7)
        current[f"{side}_ee_position"] = pose[:3]
        current[f"{side}_ee_rotation"] = _quaternion_to_matrix(pose[3:]) @ _EEF_REFRAME


def decode_action_chunk(
    actions: Any,
    *,
    current_state: Mapping[str, Any],
    action_type: str,
) -> list[dict[str, list[float]]]:
    """Convert sparse relative policy actions to absolute robot commands."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 16:
        raise ValueError(f"expected an action chunk shaped [T, >=16], got {values.shape}")
    decoded: list[dict[str, list[float]]] = []
    for action in values:
        if action_type == "joint":
            decoded.append(
                {
                    "left_arm_joint_state": (np.asarray(current_state["left_arm_joint"]) + action[0:6])
                    .astype(np.float32)
                    .tolist(),
                    "left_ee_joint_state": action[7:8].tolist(),
                    "right_arm_joint_state": (np.asarray(current_state["right_arm_joint"]) + action[8:14])
                    .astype(np.float32)
                    .tolist(),
                    "right_ee_joint_state": action[15:16].tolist(),
                }
            )
            continue
        command: dict[str, list[float]] = {}
        for side, position_slice, rotation_slice, gripper_slice in (
            ("left", slice(0, 3), slice(3, 6), slice(7, 8)),
            ("right", slice(8, 11), slice(11, 14), slice(15, 16)),
        ):
            current_position = np.asarray(current_state[f"{side}_ee_position"], dtype=np.float64)
            current_rotation = np.asarray(current_state[f"{side}_ee_rotation"], dtype=np.float64)
            position = current_position + current_rotation @ action[position_slice].astype(np.float64)
            rotation = current_rotation @ _rotvec_to_matrix(action[rotation_slice])
            simulation_rotation = rotation @ _EEF_REFRAME.T
            quaternion = _matrix_to_quaternion(simulation_rotation)
            command[f"{side}_ee_pose"] = np.concatenate((position.astype(np.float32), quaternion)).tolist()
            command[f"{side}_ee_joint_state"] = action[gripper_slice].tolist()
        decoded.append(command)
    return decoded


__all__ = [
    "add_end_effector_state",
    "center_crop_image",
    "decode_action_chunk",
    "denormalize_tensor",
    "normalize_tensor",
    "pack_robot_state",
    "resolve_camera_images",
]
