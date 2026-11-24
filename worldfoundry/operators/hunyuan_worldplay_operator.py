"""Module for the HunyuanWorldPlay operator implementation."""

import os
from typing import Any

import numpy as np

from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.pose_utils import (
    parse_pose_string,
    pose_to_input,
    pose_to_latent_num,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.generate_custom_trajectory import (
    generate_camera_trajectory_local,
)

from .base_operator import BaseOperator


class HunyuanWorldPlayOperator(BaseOperator):
    """Operator class for the HunyuanWorldPlay model integration."""

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
        *,
        forward_speed: float = 0.08,
        yaw_speed_deg: float = 3.0,
        pitch_speed_deg: float = 3.0,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["action_instruction"]
        super().__init__(operation_types=operation_types)
        self.forward_speed = forward_speed
        self.yaw_speed_deg = yaw_speed_deg
        self.pitch_speed_deg = pitch_speed_deg
        self.interaction_template = interaction_template or [
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
            "camera_up",
            "camera_down",
            "camera_l",
            "camera_r",
            "camera_ul",
            "camera_ur",
            "camera_dl",
            "camera_dr",
            "camera_zoom_in",
            "camera_zoom_out",
        ]
        self.interaction_template_init()

    def process_perception(self, input_):
        """Process perception inputs like images, videos, and reference frames."""
        if input_ is None:
            raise ValueError("reference_image must be provided")
        return input_

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if self._is_action_sequence(interaction):
            return True
        if isinstance(interaction, str):
            pose_str = interaction.strip()
            if pose_str == "":
                raise ValueError("interaction cannot be empty")
            if pose_str.endswith(".json"):
                if not os.path.exists(pose_str):
                    raise ValueError(f"Pose json not found: {pose_str}")
                return True
            parse_pose_string(pose_str)
            return True
        if isinstance(interaction, dict):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            sample = next(iter(interaction.values()))
            if not isinstance(sample, dict) or "extrinsic" not in sample or "K" not in sample:
                raise ValueError("pose dict must contain {'extrinsic','K'} per frame")
            return True
        raise TypeError(f"interaction must be str or dict, got {type(interaction)}")

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self._is_action_sequence(interaction):
            self.current_interaction.append(interaction)
            return
        if not isinstance(interaction, list):
            interaction = [interaction]
        if len(interaction) == 0:
            raise ValueError("interaction cannot be empty")
        for item in interaction:
            self.check_interaction(item)
        self.current_interaction.append(interaction)

    def process_interaction(self, latent_frames: int) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        if self._is_action_sequence(now_interaction):
            pose_data = self._actions_to_pose_json(now_interaction)
        else:
            pose_data = now_interaction[-1]
        viewmats, Ks, action = pose_to_input(pose_data, latent_frames)
        return {
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
        }

    def infer_video_length(self, interaction) -> int:
        """Infer or estimate the video length based on parameters."""
        if isinstance(interaction, list):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            if self._is_action_sequence(interaction):
                interaction = self._actions_to_pose_json(interaction)
            else:
                interaction = interaction[-1]
        latent_num = pose_to_latent_num(interaction)
        if latent_num <= 0:
            raise ValueError("pose must not be empty")
        return latent_num * 4 - 3

    def _is_action_sequence(self, interaction) -> bool:
        if not isinstance(interaction, list) or len(interaction) == 0:
            return False
        if not all(isinstance(item, str) for item in interaction):
            return False
        return all(item in self.interaction_template for item in interaction)

    def _actions_to_pose_json(self, actions: list[str]) -> dict:
        forward_speed = self.forward_speed
        yaw_speed = np.deg2rad(self.yaw_speed_deg)
        pitch_speed = np.deg2rad(self.pitch_speed_deg)
        motions: list[dict] = []
        for action in actions:
            move = {}
            if action == "forward":
                move["forward"] = forward_speed
            elif action == "backward":
                move["forward"] = -forward_speed
            elif action == "left":
                move["right"] = -forward_speed
            elif action == "right":
                move["right"] = forward_speed
            elif action == "forward_left":
                move["forward"] = forward_speed
                move["right"] = -forward_speed
            elif action == "forward_right":
                move["forward"] = forward_speed
                move["right"] = forward_speed
            elif action == "backward_left":
                move["forward"] = -forward_speed
                move["right"] = -forward_speed
            elif action == "backward_right":
                move["forward"] = -forward_speed
                move["right"] = forward_speed
            elif action == "camera_up":
                move["pitch"] = pitch_speed
            elif action == "camera_down":
                move["pitch"] = -pitch_speed
            elif action == "camera_l":
                move["yaw"] = -yaw_speed
            elif action == "camera_r":
                move["yaw"] = yaw_speed
            elif action == "camera_ul":
                move["yaw"] = -yaw_speed
                move["pitch"] = pitch_speed
            elif action == "camera_ur":
                move["yaw"] = yaw_speed
                move["pitch"] = pitch_speed
            elif action == "camera_dl":
                move["yaw"] = -yaw_speed
                move["pitch"] = -pitch_speed
            elif action == "camera_dr":
                move["yaw"] = yaw_speed
                move["pitch"] = -pitch_speed
            elif action == "camera_zoom_in":
                move["forward"] = forward_speed
            elif action == "camera_zoom_out":
                move["forward"] = -forward_speed
            else:
                raise ValueError(f"Unknown action: {action}")
            motions.append(move)
        poses = generate_camera_trajectory_local(motions)
        intrinsic = [
            [969.6969696969696, 0.0, 960.0],
            [0.0, 969.6969696969696, 540.0],
            [0.0, 0.0, 1.0],
        ]
        pose_json: dict = {}
        for i, p in enumerate(poses):
            pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}
        return pose_json
