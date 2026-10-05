"""Module for the VGGT operator implementation."""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


class VGGTOperator(BaseOperator):
    """Operator for VGGT pipeline utilities."""
    
    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["visual_instruction", "action_instruction"]
        if interaction_template is None:
            interaction_template = [
                "forward", "backward", "left", "right",
                "forward_left", "forward_right", "backward_left", "backward_right",
                "camera_up", "camera_down",
                "camera_l", "camera_r",
                "camera_ul", "camera_ur", "camera_dl", "camera_dr",
                "camera_zoom_in", "camera_zoom_out",
            ]
        super(VGGTOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """Collect file paths from a file, directory, or txt list."""
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
                if not name.startswith(".")
            ]
            files.sort()
        return files
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Process visual signal (image/images) for real-time interactive updates."""
        if isinstance(input_signal, (str, Path)):
            if os.path.isdir(input_signal) or (isinstance(input_signal, str) and input_signal.lower().endswith(".txt")):
                image_paths = self.collect_paths(input_signal)
                if len(image_paths) == 0:
                    raise ValueError(f"No images found in directory: {input_signal}")
                # Process all images
                images = []
                for img_path in image_paths:
                    images.append(self._load_single_image(img_path))
                return images
            else:
                # Single image path
                return self._load_single_image(input_signal)
        elif isinstance(input_signal, list):
            # List of paths
            images = []
            for img_path in input_signal:
                images.append(self._load_single_image(img_path))
            return images
        elif isinstance(input_signal, torch.Tensor):
            if input_signal.dim() == 3:
                image_rgb = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_rgb = input_signal[0].permute(1, 2, 0).cpu().numpy()
            # Integer dtypes are always 0-255; the value-range check is only a
            # fallback for float inputs whose scale is ambiguous.
            if np.issubdtype(image_rgb.dtype, np.integer) or image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
            return image_rgb
        elif isinstance(input_signal, np.ndarray):
            # Array inputs are expected in RGB channel order.
            if np.issubdtype(input_signal.dtype, np.integer) or input_signal.max() > 1.0:
                image_rgb = input_signal / 255.0
            else:
                image_rgb = input_signal
            return image_rgb
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")
    
    def _load_single_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        return image_rgb
    
    def check_interaction(self, interaction):
        """Check if interaction is in the interaction template."""
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction '{interaction}' not in interaction_template. "
                           f"Available interactions: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """Add interaction to current_interaction list after validation."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """Process current interactions and convert to features for representation/synthesis."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        result = {
            "predict_cameras": True,
            "predict_depth": True,
            "predict_points": True,
            "predict_tracks": False,
            "camera_controls": {
                "move_left": False,
                "move_right": False,
                "move_up": False,
                "move_down": False,
                "zoom_in": False,
                "zoom_out": False,
                "rotate_left": False,
                "rotate_right": False,
            }
        }
        
        if latest_interaction == "single_view_reconstruction":
            result["predict_cameras"] = True
            result["predict_depth"] = True
            result["predict_points"] = True
            result["predict_tracks"] = False
        elif latest_interaction == "multi_view_reconstruction":
            result["predict_cameras"] = True
            result["predict_depth"] = True
            result["predict_points"] = True
            result["predict_tracks"] = False
        elif latest_interaction == "camera_pose_estimation":
            result["predict_cameras"] = True
            result["predict_depth"] = False
            result["predict_points"] = False
            result["predict_tracks"] = False
        elif latest_interaction == "depth_estimation":
            result["predict_cameras"] = False
            result["predict_depth"] = True
            result["predict_points"] = False
            result["predict_tracks"] = False
        elif latest_interaction == "point_cloud_generation":
            result["predict_cameras"] = True
            result["predict_depth"] = True
            result["predict_points"] = True
            result["predict_tracks"] = False
        elif latest_interaction == "point_tracking":
            result["predict_cameras"] = True
            result["predict_depth"] = True
            result["predict_points"] = True
            result["predict_tracks"] = True
        elif latest_interaction == "move_left":
            result["camera_controls"]["move_left"] = True
        elif latest_interaction == "move_right":
            result["camera_controls"]["move_right"] = True
        elif latest_interaction == "move_up":
            result["camera_controls"]["move_up"] = True
        elif latest_interaction == "move_down":
            result["camera_controls"]["move_down"] = True
        elif latest_interaction == "zoom_in":
            result["camera_controls"]["zoom_in"] = True
        elif latest_interaction == "zoom_out":
            result["camera_controls"]["zoom_out"] = True
        elif latest_interaction == "rotate_left":
            result["camera_controls"]["rotate_left"] = True
        elif latest_interaction == "rotate_right":
            result["camera_controls"]["rotate_right"] = True
        
        if num_frames is not None:
            result["num_frames"] = num_frames
        
        return result
