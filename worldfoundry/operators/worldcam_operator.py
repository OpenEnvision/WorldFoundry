"""Module for the WorldCam operator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base_operator import BaseOperator

_RESAMPLE_BICUBIC = Image.Resampling.BICUBIC

_ACTION_ALIASES = {
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "camera_l": "camera_l",
    "camera_left": "camera_l",
    "camera_r": "camera_r",
    "camera_right": "camera_r",
    "camera_up": "camera_up",
    "camera_down": "camera_down",
    "forward_left": "forward_left",
    "forward_right": "forward_right",
    "backward_left": "backward_left",
    "backward_right": "backward_right",
    "no-op": "no-op",
    "noop": "no-op",
    "none": "no-op",
}

_INTRINSICS_1920X1080 = np.array([722.91626, 722.6145, 960.0, 540.0], dtype=np.float32)


def _normalize_action(action: str) -> str:
    key = action.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ACTION_ALIASES:
        raise ValueError(f"Unsupported WorldCam interaction: {action}")
    return _ACTION_ALIASES[key]


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.dtype in (np.float16, np.float32, np.float64):
        if data.min() >= -1.0 and data.max() <= 1.0:
            data = (data + 1.0) * 127.5 if data.min() < 0.0 else data * 255.0
        data = np.clip(data, 0.0, 255.0).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return data


def _to_pil_image(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, str):
        return Image.open(data).convert("RGB")

    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported WorldCam perception input: {type(data)}")

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported WorldCam perception shape: {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    array = _to_uint8_image(array)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


class WorldCamOperator(BaseOperator):
    """Convert WorldBench navigation actions into WorldCam camera trajectories."""

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
        conditioning_frames: int = 65,
        conditioning_latent_frames: int = 16,
        frames_per_latent: int = 4,
        latent_steps_per_action: int = 3,
        translation_step: float = 0.10,
        turn_angle_deg: float = 10.0,
        max_pitch_deg: float = 30.0,
        height: int = 480,
        width: int = 832,
    ):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types or ["textual_instruction", "action_instruction", "visual_instruction"]
        )
        self.interaction_template = interaction_template or [
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
            "camera_l",
            "camera_r",
            "camera_up",
            "camera_down",
            "no-op",
        ]
        self.interaction_template_init()
        self.conditioning_frames = int(conditioning_frames)
        self.conditioning_latent_frames = int(conditioning_latent_frames)
        self.frames_per_latent = int(frames_per_latent)
        self.latent_steps_per_action = int(latent_steps_per_action)
        self.translation_step = float(translation_step)
        self.turn_angle_rad = float(np.deg2rad(turn_angle_deg))
        self.max_pitch_rad = float(np.deg2rad(max_pitch_deg))
        self.height = int(height)
        self.width = int(width)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        _normalize_action(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, str):
            interaction = [interaction]
        if not isinstance(interaction, Sequence):
            raise TypeError("interaction must be a string or a sequence of strings.")
        normalized = [_normalize_action(item) for item in interaction]
        self.current_interaction.append(normalized)

    def process_perception(self, images, height: int | None = None, width: int | None = None):
        """Process perception inputs like images, videos, and reference frames."""
        resize_h = int(height or self.height)
        resize_w = int(width or self.width)
        image = _to_pil_image(images)
        image = image.resize((resize_w, resize_h), resample=_RESAMPLE_BICUBIC)
        condition_video = [image.copy() for _ in range(self.conditioning_frames)]
        return {
            "input_image": image,
            "condition_video": condition_video,
            "height": resize_h,
            "width": resize_w,
            "conditioning_frames": self.conditioning_frames,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction registered. Use get_interaction() first.")

        actions = list(self.current_interaction[-1])
        self.interaction_history.extend(actions)
        expanded_actions: list[str] = []
        for action in actions:
            expanded_actions.extend([action] * self.latent_steps_per_action)

        poses = self._build_pose_sequence(expanded_actions)
        intrinsics = np.repeat(_INTRINSICS_1920X1080[None, :], len(poses), axis=0)

        return {
            "actions": actions,
            "expanded_actions": expanded_actions,
            "num_ar_steps": len(expanded_actions),
            "num_output_frames": len(expanded_actions) * self.frames_per_latent,
            "intrinsics": torch.from_numpy(intrinsics).unsqueeze(0),
            "extrinsics": torch.from_numpy(np.stack(poses, axis=0)).unsqueeze(0),
        }

    def _build_pose_sequence(self, expanded_actions: Sequence[str]) -> list[np.ndarray]:
        position = np.zeros(3, dtype=np.float32)
        yaw = 0.0
        pitch = 0.0

        poses = [self._compose_c2w(position, yaw, pitch)]
        warmup_frames = self.conditioning_latent_frames * self.frames_per_latent
        poses.extend([poses[0].copy() for _ in range(warmup_frames)])

        for action in expanded_actions:
            start_position = position.copy()
            start_yaw = float(yaw)
            start_pitch = float(pitch)

            position, yaw, pitch = self._apply_action_end_state(start_position, start_yaw, start_pitch, action)

            for frame_idx in range(1, self.frames_per_latent + 1):
                alpha = frame_idx / self.frames_per_latent
                interp_position = start_position + (position - start_position) * alpha
                interp_yaw = start_yaw + (yaw - start_yaw) * alpha
                interp_pitch = start_pitch + (pitch - start_pitch) * alpha
                poses.append(self._compose_c2w(interp_position, interp_yaw, interp_pitch))

        return poses

    def _apply_action_end_state(
        self,
        position: np.ndarray,
        yaw: float,
        pitch: float,
        action: str,
    ) -> tuple[np.ndarray, float, float]:
        end_position = position.copy()
        end_yaw = float(yaw)
        end_pitch = float(pitch)

        forward, right, _ = self._camera_axes(yaw, pitch)
        if action == "forward":
            end_position = position + forward * self.translation_step
        elif action == "backward":
            end_position = position - forward * self.translation_step
        elif action == "left":
            end_position = position - right * self.translation_step
        elif action == "right":
            end_position = position + right * self.translation_step
        elif action == "forward_left":
            direction = forward - right
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end_position = position + direction * self.translation_step
        elif action == "forward_right":
            direction = forward + right
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end_position = position + direction * self.translation_step
        elif action == "backward_left":
            direction = -forward - right
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end_position = position + direction * self.translation_step
        elif action == "backward_right":
            direction = -forward + right
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end_position = position + direction * self.translation_step
        elif action == "camera_l":
            end_yaw = yaw - self.turn_angle_rad
        elif action == "camera_r":
            end_yaw = yaw + self.turn_angle_rad
        elif action == "camera_up":
            end_pitch = min(pitch + self.turn_angle_rad, self.max_pitch_rad)
        elif action == "camera_down":
            end_pitch = max(pitch - self.turn_angle_rad, -self.max_pitch_rad)
        elif action == "no-op":
            pass
        else:
            raise ValueError(f"Unsupported WorldCam action: {action}")

        return end_position.astype(np.float32), float(end_yaw), float(end_pitch)

    @staticmethod
    def _compose_c2w(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        forward, right, up = WorldCamOperator._camera_axes(yaw, pitch)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([right, up, forward], axis=1)
        c2w[:3, 3] = position.astype(np.float32)
        return c2w

    @staticmethod
    def _camera_axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)
        return forward, right, up
