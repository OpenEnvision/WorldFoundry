"""
HunyuanMirror Operator

This operator handles user interactions for the HunyuanMirror pipeline.
It processes user commands like camera control, object manipulation, etc.
"""

import torch

from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.camera_utils import (
    vector_to_camera_matrices,
)

from .base_operator import BaseOperator


class HunyuanMirrorOperator(BaseOperator):
    """Operator class for the HunyuanMirror model integration."""

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["action_instruction", "textual_instruction"]
        if interaction_template is None:
            interaction_template = [
                "camera_look_up",
                "camera_look_down",
                "camera_look_left",
                "camera_look_right",
                "camera_zoom_in",
                "camera_zoom_out",
                "object_move_forward",
                "object_move_backward",
                "object_move_left",
                "object_move_right",
                "reset_camera",
            ]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.current_interaction = []  # interactions in the order the user issued them

    def check_interaction(self, interaction):
        """Check that the interaction is one of the interaction template entries."""
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in interaction template: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        """Validate an interaction and append it to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames, image_hw=(518, 518)):
        """
        Convert the latest interaction into camera parameters for the representation and synthesis models.

        Args:
            num_frames: Number of frames to generate.
            image_hw: Image (height, width).

        Returns:
            dict: The interaction, ``num_frames``, ``image_hw``, ``camera_poses`` (num_frames, 3, 4)
            and ``camera_intrinsics`` (num_frames, 3, 3).
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")

        latest_interaction = self.current_interaction[-1]

        self.interaction_history.append(latest_interaction)

        h, w = image_hw

        # Camera vector: position, scalar-last quaternion, vertical and horizontal FOV.
        default_cam_vec = torch.tensor(
            [
                0.0,
                0.0,
                0.0,  # position (x, y, z)
                0.0,
                0.0,
                0.0,
                1.0,  # rotation (quaternion: x, y, z, w; scalar-last)
                torch.pi / 3,  # vertical FOV
                torch.pi / 3,  # horizontal FOV
            ]
        )

        def with_axis_angle(vec, axis, angle):
            """Return ``vec`` with a normalized scalar-last axis-angle quaternion."""
            axis = torch.as_tensor(axis, dtype=vec.dtype, device=vec.device)
            axis = axis / axis.norm().clamp_min(torch.finfo(vec.dtype).eps)
            half_angle = torch.as_tensor(angle * 0.5, dtype=vec.dtype, device=vec.device)
            quaternion = torch.cat((axis * torch.sin(half_angle), torch.cos(half_angle)[None]))
            adjusted = vec.clone()
            adjusted[3:7] = quaternion
            return adjusted

        # Each adjustment is applied to the default camera, not to the previous one.
        cam_adjustments = {
            "camera_look_up": lambda vec: with_axis_angle(vec, (1.0, 0.0, 0.0), 0.2),
            "camera_look_down": lambda vec: with_axis_angle(vec, (1.0, 0.0, 0.0), -0.2),
            "camera_look_left": lambda vec: with_axis_angle(vec, (0.0, 0.0, 1.0), 0.2),
            "camera_look_right": lambda vec: with_axis_angle(vec, (0.0, 0.0, 1.0), -0.2),
            # Zooming moves the camera along z, toward (-0.1) or away from (+0.1) the object.
            "camera_zoom_in": lambda vec: torch.cat([vec[:2], vec[2:3] - 0.1, vec[3:7], vec[7:]]),
            "camera_zoom_out": lambda vec: torch.cat([vec[:2], vec[2:3] + 0.1, vec[3:7], vec[7:]]),
            "reset_camera": lambda vec: default_cam_vec.clone(),
            # Object moves leave the camera unchanged.
            "object_move_forward": lambda vec: vec,
            "object_move_backward": lambda vec: vec,
            "object_move_left": lambda vec: vec,
            "object_move_right": lambda vec: vec,
        }

        if latest_interaction in cam_adjustments:
            adjusted_cam_vec = cam_adjustments[latest_interaction](default_cam_vec)
        else:
            adjusted_cam_vec = default_cam_vec.clone()

        ext, intr = vector_to_camera_matrices(adjusted_cam_vec, image_hw=image_hw)

        # The same camera is used for every frame.
        ext = ext.unsqueeze(0).repeat(num_frames, 1, 1)
        intr = intr.unsqueeze(0).repeat(num_frames, 1, 1)

        return {
            "interaction": latest_interaction,
            "num_frames": num_frames,
            "camera_poses": ext,  # extrinsics (num_frames, 3, 4)
            "camera_intrinsics": intr,  # intrinsics (num_frames, 3, 3)
            "image_hw": image_hw,
        }
