"""Module for the TrajectoryGenerator operator implementation."""

import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.networks.wan.camera_geometry import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)

from .base_operator import BaseOperator

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """
    Tool class for generating camera trajectories from text commands.
    OpenCV coordinate system:
        X: Right
        Y: Down
        Z: Forward
    """

    def __init__(self):
        """Initialize the operator with specific configurations."""
        self.trans_speed = 0.1
        self.rot_speed = np.radians(0.2)
        self.zoom_speed = 0.01

        self.COMMAND_MAPPING = {
            # Translation
            "forward": np.array([0, 0, 1, 0, 0, 0, 0]) * self.trans_speed,
            "backward": np.array([0, 0, -1, 0, 0, 0, 0]) * self.trans_speed,
            "left": np.array([-1, 0, 0, 0, 0, 0, 0]) * self.trans_speed,
            "right": np.array([1, 0, 0, 0, 0, 0, 0]) * self.trans_speed,
            # Diagonal Translation
            "forward_left": np.array([-1, 0, 1, 0, 0, 0, 0]) * self.trans_speed,
            "forward_right": np.array([1, 0, 1, 0, 0, 0, 0]) * self.trans_speed,
            "backward_left": np.array([-1, 0, -1, 0, 0, 0, 0]) * self.trans_speed,
            "backward_right": np.array([1, 0, -1, 0, 0, 0, 0]) * self.trans_speed,
            # Camera Pan/Tilt
            "camera_up": np.array([0, 0, 0, -1, 0, 0, 0]) * self.rot_speed,
            "camera_down": np.array([0, 0, 0, 1, 0, 0, 0]) * self.rot_speed,
            "camera_l": np.array([0, 0, 0, 0, -1, 0, 0]) * self.rot_speed,
            "camera_r": np.array([0, 0, 0, 0, 1, 0, 0]) * self.rot_speed,
            # Diagonal Rotation
            "camera_ul": np.array([0, 0, 0, -1, -1, 0, 0]) * self.rot_speed,
            "camera_ur": np.array([0, 0, 0, -1, 1, 0, 0]) * self.rot_speed,
            "camera_dl": np.array([0, 0, 0, 1, -1, 0, 0]) * self.rot_speed,
            "camera_dr": np.array([0, 0, 0, 1, 1, 0, 0]) * self.rot_speed,
            # Zooming
            "camera_zoom_in": np.array([0, 0, 0, 0, 0, 0, 1]) * self.zoom_speed,
            "camera_zoom_out": np.array([0, 0, 0, 0, 0, 0, -1]) * self.zoom_speed,
        }

    def generate(self, action_list: list, num_frames: int, height: int, width: int):
        """Generate c2ws and Ks matrices"""
        delta_vec = np.zeros(7)  # [tx, ty, tz, rx, ry, rz, zoom]
        valid_cmds = 0
        for action in action_list:
            if action in self.COMMAND_MAPPING:
                delta_vec += self.COMMAND_MAPPING[action]
                valid_cmds += 1
            else:
                logger.warning("Unknown action command: %s", action)

        if valid_cmds == 0:
            pass

        c2ws = []
        Ks = []
        current_pose = np.eye(4)

        d_trans = delta_vec[:3]
        d_rot_euler = delta_vec[3:6]
        d_zoom = delta_vec[6]

        base_fx = max(width, height)
        base_fy = base_fx
        cx = width / 2.0
        cy = height / 2.0

        delta_mat = np.eye(4)
        if np.linalg.norm(d_rot_euler) > 1e-8:
            from scipy.spatial.transform import Rotation

            r = Rotation.from_euler("xyz", d_rot_euler, degrees=False)
            delta_mat[:3, :3] = r.as_matrix()
        delta_mat[:3, 3] = d_trans

        for i in range(num_frames):
            c2ws.append(current_pose.copy())

            current_zoom_factor = 1.0 + (d_zoom * i)
            current_zoom_factor = max(0.1, current_zoom_factor)

            K = np.array([base_fx * current_zoom_factor, base_fy * current_zoom_factor, cx, cy], dtype=np.float32)
            Ks.append(K)

            if valid_cmds > 0 and i < num_frames - 1:
                current_pose = current_pose @ delta_mat

        c2ws = np.array(c2ws, dtype=np.float32)
        Ks = np.array(Ks, dtype=np.float32)  # [num_frames, 4]

        return c2ws, Ks


class LingBotOperator(BaseOperator):
    """Operator class for the LingBot model integration."""

    def __init__(self, operation_types=None):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["textual_instruction", "action_instruction"]
        super().__init__(operation_types=operation_types)
        # Supported interaction templates: prompt, action_path (file), action_list (commands)
        self.interaction_template = ["prompt", "action_path", "action_list"]
        self.interaction_template_init()
        self.current_interaction: dict[str, Any] = {}

        # Initialize trajectory generator
        self.traj_generator = TrajectoryGenerator()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, dict):
            raise ValueError("Interaction must be a dictionary")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction = interaction

    def process_perception(self, input_image, resize_H=480, resize_W=832, device="cuda"):
        """Process perception inputs like images, videos, and reference frames."""
        import torchvision.transforms.functional as TF

        if not isinstance(input_image, Image.Image):
            raise ValueError("Input image must be a PIL Image")
        img = input_image.resize((resize_W, resize_H), Image.Resampling.BICUBIC)
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
        return {"image_tensor": img_tensor, "pil_image": img}

    def process_interaction(self, resize_H=480, resize_W=832, num_frames=81, device="cuda"):
        """Process the recorded interactions and return the generated actions."""
        prompt = self.current_interaction.get("prompt", "")
        action_path = self.current_interaction.get("action_path", None)
        action_list = self.current_interaction.get("action_list", None)

        camera_data = None

        c2ws = None
        Ks = None

        # Branch 1: Generate from command list
        if action_list is not None and isinstance(action_list, list):
            logger.info("Generating trajectory from commands: %s", action_list)
            c2ws, Ks_np = self.traj_generator.generate(action_list, num_frames, resize_H, resize_W)
            Ks = torch.from_numpy(Ks_np).float()

        # Branch 2: Load from file path (legacy mode)
        elif action_path is not None:
            # Load original data
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            Ks_orig = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

            # Align frame count
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            real_frame_num = min(num_frames, len_c2ws)
            c2ws = c2ws[:real_frame_num]

            # Transform intrinsics (adapt to resize)
            Ks = get_Ks_transformed(
                Ks_orig,
                height_org=480,
                width_org=832,
                height_resize=resize_H,
                width_resize=resize_W,
                height_final=resize_H,
                width_final=resize_W,
            )
            Ks = Ks[0].repeat(len(c2ws), 1)  # [N, 4]

        # Common processing logic (Generate Embedding)
        if c2ws is not None and Ks is not None:
            len_c2ws = len(c2ws)

            # 1. Interpolate/smooth trajectory
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )

            # 2. Compute relative poses
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=False)

            # 3. Adjust Ks length to match downsampled c2ws
            Ks_downsampled = Ks[: len(c2ws_infer)]

            c2ws_infer = c2ws_infer.to(device)
            Ks_downsampled = Ks_downsampled.to(device)

            # 4. Compute Plucker Embedding
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks_downsampled, resize_H, resize_W)

            camera_data = {"c2ws_plucker_emb": c2ws_plucker_emb}

        return {"prompt": prompt, "camera_data": camera_data}
