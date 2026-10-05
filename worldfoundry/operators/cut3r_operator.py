"""Module for the CUT3R operator implementation."""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from PIL import Image

from .base_operator import BaseOperator


class CUT3ROperator(BaseOperator):
    """Operator for CUT3R pipeline utilities."""
    
    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
    ):
        """
        Initialize CUT3R operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - Unified 3D camera controls:
                  forward/backward/left/right, forward_left/forward_right,
                  backward_left/backward_right,
                  camera_up/camera_down, camera_l/camera_r,
                  camera_ul/camera_ur/camera_dl/camera_dr,
                  camera_zoom_in/camera_zoom_out
        """
        if operation_types is None:
            operation_types = ["visual_instruction"]
        if interaction_template is None:
            interaction_template = [
                "forward", "backward", "left", "right",
                "forward_left", "forward_right", "backward_left", "backward_right",
                "camera_up", "camera_down",
                "camera_l", "camera_r",
                "camera_ul", "camera_ur", "camera_dl", "camera_dr",
                "camera_zoom_in", "camera_zoom_out",
            ]
        super(CUT3ROperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """
        Collect file paths from a file, directory, or txt list.
        
        Args:
            path: File path, directory path, or txt file containing paths
            
        Returns:
            List of file paths
        """
        path = str(path)
        if os.path.isfile(path):
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as handle:
                    files = [line.strip() for line in handle.readlines() if line.strip()]
            else:
                files = [path]
        else:
            files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if not name.startswith(".") and name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            files.sort()
        return files
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, Image.Image, List[str], List[np.ndarray], List[Image.Image]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Process visual signal (image/video) for real-time interactive updates.
        This function handles loading and preprocessing of images from various input types.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - List of image file paths (List[str])
                - Numpy array (H, W, 3) in RGB format
                - List of numpy arrays
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                
        Returns:
            Preprocessed RGB image array(s) (normalized to [0, 1]) with shape (H, W, 3)
            or list of such arrays
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        # Handle list inputs (paths, numpy arrays, tensors, PIL Images)
        if isinstance(input_signal, list):
            return [self.process_perception(item) for item in input_signal]
        
        # Handle single input
        if isinstance(input_signal, Image.Image):
            image_rgb = np.array(input_signal)
            # PIL images decode to integer arrays; remember before the float cast.
            is_integer_input = np.issubdtype(image_rgb.dtype, np.integer)
            if image_rgb.dtype != np.float32:
                image_rgb = image_rgb.astype(np.float32)
            if is_integer_input or image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
        elif isinstance(input_signal, torch.Tensor):
            # Assume tensor is in CHW format, convert to numpy
            if input_signal.dim() == 3:
                image_rgb = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_rgb = input_signal[0].permute(1, 2, 0).cpu().numpy()
            # Integer dtypes are always 0-255; the value-range check is only a
            # fallback for float inputs whose scale is ambiguous.
            if np.issubdtype(image_rgb.dtype, np.integer) or image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
        elif isinstance(input_signal, np.ndarray):
            # Array inputs are expected in RGB channel order.
            if np.issubdtype(input_signal.dtype, np.integer) or input_signal.max() > 1.0:
                image_rgb = input_signal / 255.0
            else:
                image_rgb = input_signal
        else:
            # String path: support single image, directory, or txt list.
            if isinstance(input_signal, (str, Path)):
                input_path = str(input_signal)
                if os.path.isdir(input_path) or (os.path.isfile(input_path) and input_path.lower().endswith(".txt")):
                    file_list = self.collect_paths(input_path)
                    if len(file_list) == 0:
                        raise ValueError(f"No valid image files found in {input_path}")
                    return [self.process_perception(p) for p in file_list]

                raw_image = cv2.imread(input_path)
                if raw_image is None:
                    raise ValueError(f"Could not read image from {input_signal}")
                image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            else:
                raise ValueError(f"Unsupported input type for process_perception: {type(input_signal)}")
        
        return image_rgb
    
    def check_interaction(self, interaction):
        """
        Check if interaction is in the interaction template.
        
        Args:
            interaction: Interaction string to check
            
        Returns:
            True if interaction is valid
            
        Raises:
            ValueError: If interaction is not in template
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction '{interaction}' not in interaction_template. "
                           f"Available interactions: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        Add interaction to current_interaction list after validation.
        
        Args:
            interaction: Interaction string to add
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        
        Args:
            num_frames: Number of frames (for video processing, optional)
            
        Returns:
            Dictionary containing processed interaction features:
                - data_type: "image" or "video"
                - output_type: "point_cloud", "depth_map", "camera_pose", or "all"
                - camera_control: Dict with camera movement parameters (if applicable)
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        # Process interaction based on type
        result = {
            "data_type": "image",
            "output_type": "all",  # point_cloud, depth_map, camera_pose, or all
            "camera_control": None
        }
        
        # Camera control interactions (unified 3D schema)
        if latest_interaction in [
            "forward", "backward", "left", "right",
            "forward_left", "forward_right", "backward_left", "backward_right",
            "camera_up", "camera_down",
            "camera_l", "camera_r",
            "camera_ul", "camera_ur", "camera_dl", "camera_dr",
            "camera_zoom_in", "camera_zoom_out",
        ]:
            result["camera_control"] = {"interaction": latest_interaction}
        
        # Add num_frames if provided (for video processing)
        if num_frames is not None:
            result["num_frames"] = num_frames
        
        return result
    
    @staticmethod
    def normalize_interaction_sequence(
        interaction: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Normalize interaction input to a flat list of strings.
        Supports None, single string, or list of strings.
        """
        if interaction is None:
            return []
        if isinstance(interaction, str):
            return [interaction]
        return [str(sig) for sig in interaction if str(sig).strip()]

    @staticmethod
    def apply_interaction_to_camera(
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 30.0,
        pitch_step: float = 20.0,
        zoom_factor: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Update a simple (radius, yaw, pitch) camera configuration according to a
        high-level interaction signal, clamped by camera_range.
        Only supports the unified 3D interaction schema
        (forward/backward/left/right, forward_left, camera_l, camera_zoom_in, ...).
        """
        yaw = float(camera_cfg.get("yaw", 0.0))
        pitch = float(camera_cfg.get("pitch", 0.0))
        radius = float(camera_cfg.get("radius", 4.0))
        sig = interaction.strip().lower()

        # Yaw (left/right)
        if sig in ["left", "camera_l"]:
            yaw -= yaw_step
        elif sig in ["right", "camera_r"]:
            yaw += yaw_step
        elif sig == "camera_ul":
            yaw -= yaw_step
            pitch += pitch_step
        elif sig == "camera_ur":
            yaw += yaw_step
            pitch += pitch_step
        elif sig == "camera_dl":
            yaw -= yaw_step
            pitch -= pitch_step
        elif sig == "camera_dr":
            yaw += yaw_step
            pitch -= pitch_step
        # Pitch (up/down)
        elif sig == "camera_up":
            pitch += pitch_step
        elif sig == "camera_down":
            pitch -= pitch_step
        # Radius (forward/backward, zoom)
        elif sig in ["forward", "camera_zoom_in"]:
            radius *= zoom_factor
        elif sig in ["backward", "camera_zoom_out"]:
            radius /= zoom_factor
        elif sig == "forward_left":
            yaw -= yaw_step
            radius *= zoom_factor
        elif sig == "forward_right":
            yaw += yaw_step
            radius *= zoom_factor
        elif sig == "backward_left":
            yaw -= yaw_step
            radius /= zoom_factor
        elif sig == "backward_right":
            yaw += yaw_step
            radius /= zoom_factor

        yaw = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
        pitch = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))
        radius = max(camera_range["radius_min"], min(camera_range["radius_max"], radius))

        camera_cfg["yaw"] = yaw
        camera_cfg["pitch"] = pitch
        camera_cfg["radius"] = radius

        return camera_cfg
