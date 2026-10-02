"""Module for the Lyra operator implementation."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np
import torch

from worldfoundry.core.geometry.path import sample_camera_path
from worldfoundry.core.io.image import load_pil_image

from .base_operator import BaseOperator


class LyraOperator(BaseOperator):
    """Convert WorldFoundry interaction signals into a Lyra-2 camera trajectory."""

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
        chunk_stride: int = 80,
        translation_step: float = 0.12,
        turn_angle_deg: float = 15.0,
        zoom_step: float = 0.35,
        max_pitch_deg: float = 35.0,
    ):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types or [
                "textual_instruction",
                "action_instruction",
                "visual_instruction",
            ]
        )
        self.interaction_template = interaction_template or [
            "forward",
            "backward",
            "left",
            "right",
            "camera_l",
            "camera_r",
            "camera_up",
            "camera_down",
            "camera_zoom_in",
            "camera_zoom_out",
        ]
        self.interaction_template_init()
        self.chunk_stride = int(chunk_stride)
        self.translation_step = float(translation_step)
        self.turn_angle_rad = float(np.deg2rad(turn_angle_deg))
        self.max_pitch_rad = float(np.deg2rad(max_pitch_deg))
        self.zoom_step = float(zoom_step)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        action = interaction
        if isinstance(interaction, dict):
            action = (
                interaction.get("action")
                or interaction.get("signal")
                or interaction.get("interaction")
            )
        if action not in self.interaction_template:
            raise ValueError(f"{action} not in template. Available: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, (list, tuple)):
            normalized = [self._normalize_interaction(item) for item in interaction]
        else:
            normalized = [self._normalize_interaction(interaction)]
        self.current_interaction.append(normalized)

    def process_interaction(
        self,
        prompt: str = "",
    ) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)

        actions = [item["action"] for item in latest_interaction]
        prompts = [item["caption"] or prompt or "" for item in latest_interaction]
        camera_w2c, zoom_factors = self._actions_to_camera_trajectory(actions)
        chunk_captions = self._build_chunk_captions(prompts)

        return {
            "actions": actions,
            "camera_w2c": camera_w2c,
            "zoom_factors": zoom_factors,
            "chunk_captions": chunk_captions,
            "num_frames": int(camera_w2c.shape[0]),
        }

    def process_perception(self, images):
        """Process perception inputs like images, videos, and reference frames."""
        return load_pil_image(images)

    def process_camera_path(
        self,
        camera_path: Dict[str, Any],
        *,
        prompt: str = "",
        region_hint: str = "",
    ) -> Dict[str, Any]:
        """Convert the shared camera-path schema into the Lyra trajectory."""

        sampled = sample_camera_path(camera_path, frame_stride=self.chunk_stride)
        chunk_captions = dict(sampled["chunk_captions"])
        fallback_caption = (region_hint or prompt or "").strip()
        if fallback_caption and not chunk_captions:
            chunk_captions["0"] = fallback_caption
        return {
            "actions": ["camera_path"] * (len(sampled["camera_path"]["keyframes"]) - 1),
            "camera_path": sampled["camera_path"],
            "camera_w2c": torch.from_numpy(sampled["camera_w2c"]),
            "zoom_factors": torch.from_numpy(sampled["zoom_factors"]),
            "chunk_captions": chunk_captions,
            "num_frames": int(sampled["frame_count"]),
        }

    def _normalize_interaction(self, interaction: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """Normalize interaction implementation."""
        if isinstance(interaction, str):
            self.check_interaction(interaction)
            return {"action": interaction, "caption": ""}
        if isinstance(interaction, dict):
            self.check_interaction(interaction)
            return {
                "action": interaction.get("action")
                or interaction.get("signal")
                or interaction.get("interaction"),
                "caption": interaction.get("caption")
                or interaction.get("prompt")
                or interaction.get("text_prompt")
                or "",
            }
        raise TypeError(f"Unsupported interaction type: {type(interaction)}")

    def _build_chunk_captions(self, prompts: Sequence[str]) -> Dict[str, str]:
        """Build chunk captions implementation."""
        captions: Dict[str, str] = {}
        for index, prompt in enumerate(prompts):
            frame_index = 0 if index == 0 else 1 + index * self.chunk_stride
            captions[str(frame_index)] = prompt
        return captions

    def _actions_to_camera_trajectory(self, actions: Sequence[str]):
        """Actions to camera trajectory implementation."""
        current_position = np.zeros(3, dtype=np.float32)
        current_yaw = 0.0
        current_pitch = 0.0
        current_zoom = 1.0

        poses = [self._compose_w2c(current_position, current_yaw, current_pitch)]
        zoom_factors = [current_zoom]

        for action in actions:
            start_position = current_position.copy()
            start_yaw = float(current_yaw)
            start_pitch = float(current_pitch)
            start_zoom = float(current_zoom)

            end_position = start_position.copy()
            end_yaw = start_yaw
            end_pitch = start_pitch
            end_zoom = start_zoom

            forward, right, _ = self._camera_axes(start_yaw, start_pitch)
            if action == "forward":
                end_position = start_position + forward * self.translation_step
            elif action == "backward":
                end_position = start_position - forward * self.translation_step
            elif action == "left":
                end_position = start_position - right * self.translation_step
            elif action == "right":
                end_position = start_position + right * self.translation_step
            elif action == "camera_l":
                end_yaw = start_yaw - self.turn_angle_rad
            elif action == "camera_r":
                end_yaw = start_yaw + self.turn_angle_rad
            elif action == "camera_up":
                end_pitch = min(start_pitch + self.turn_angle_rad, self.max_pitch_rad)
            elif action == "camera_down":
                end_pitch = max(start_pitch - self.turn_angle_rad, -self.max_pitch_rad)
            elif action == "camera_zoom_in":
                end_zoom = start_zoom * (1.0 + self.zoom_step)
            elif action == "camera_zoom_out":
                end_zoom = max(start_zoom / (1.0 + self.zoom_step), 0.6)
            else:
                raise ValueError(f"Unsupported action: {action}")

            for frame_idx in range(1, self.chunk_stride + 1):
                alpha = frame_idx / self.chunk_stride
                position = start_position + (end_position - start_position) * alpha
                yaw = start_yaw + (end_yaw - start_yaw) * alpha
                pitch = start_pitch + (end_pitch - start_pitch) * alpha
                zoom = start_zoom + (end_zoom - start_zoom) * alpha
                poses.append(self._compose_w2c(position, yaw, pitch))
                zoom_factors.append(zoom)

            current_position = end_position
            current_yaw = end_yaw
            current_pitch = end_pitch
            current_zoom = end_zoom

        pose_tensor = torch.from_numpy(np.stack(poses, axis=0))
        zoom_tensor = torch.tensor(zoom_factors, dtype=torch.float32)
        return pose_tensor, zoom_tensor

    def _compose_w2c(self, position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Compose w2c implementation."""
        forward, right, up = self._camera_axes(yaw, pitch)
        rotation = np.stack([right, up, forward], axis=0)
        translation = -rotation @ position.astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation
        return w2c

    @staticmethod
    def _camera_axes(yaw: float, pitch: float):
        """Camera axes implementation."""
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
