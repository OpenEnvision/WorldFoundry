"""Module for the DepthAnything3 operator implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .base_operator import BaseOperator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


class DepthAnything3Operator(BaseOperator):
    """Operator utilities for Depth Anything 3 pipelines.

    This operator orchestrates multimodal and multidimensional depth estimation flows,
    acting as a bridge between high-level task specifications (e.g. single-view, multi-view,
    point cloud generation, or Gaussian Splatting estimation) and the underlying
    Depth Anything 3 model implementations.
    """

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
    ):
        """Initializes the Operator with default interaction templates covering supported depth modalities."""
        if operation_types is None:
            operation_types = ["visual_instruction"]
        if interaction_template is None:
            interaction_template = [
                "single_view_depth",
                "multi_view_depth",
                "video_depth",
                "pose_conditioned_depth",
                "point_cloud_generation",
                "gaussian_estimation",
            ]
        super(DepthAnything3Operator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """Crawls a directory, file list, or text manifest to yield absolute image asset paths."""
        path = Path(path).expanduser()
        if path.is_file():
            if path.suffix.lower() == ".txt":
                with path.open("r", encoding="utf-8") as handle:
                    return [line.strip() for line in handle if line.strip()]
            return [str(path.resolve())]

        if not path.is_dir():
            raise FileNotFoundError(f"DepthAnything3 input path not found: {path}")

        files = [
            str(item.resolve())
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return files

    def check_interaction(self, interaction):
        """Validates that a requested inference modality is supported by this DA3 pipeline."""
        if interaction not in self.interaction_template:
            raise ValueError(
                f"Interaction '{interaction}' not in interaction_template. "
                f"Available interactions: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction):
        """Registers a verified interaction intent to the operator's internal state."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """Translates the registered interaction state into physical model pipeline flags.

        Converts high-level prompts (e.g., 'gaussian_estimation') into boolean switches
        (`infer_gs=True`) passed to the underlying runner architectures.
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)

        result = {
            "data_type": "image_sequence",
            "infer_gs": False,
            "expects_camera_inputs": False,
            "return_point_cloud": True,
        }

        if latest_interaction == "single_view_depth":
            result["data_type"] = "image_sequence"
        elif latest_interaction == "multi_view_depth":
            result["data_type"] = "image_sequence"
        elif latest_interaction == "video_depth":
            result["data_type"] = "video"
        elif latest_interaction == "pose_conditioned_depth":
            result["expects_camera_inputs"] = True
        elif latest_interaction == "point_cloud_generation":
            result["return_point_cloud"] = True
        elif latest_interaction == "gaussian_estimation":
            result["infer_gs"] = True

        if num_frames is not None:
            result["num_frames"] = num_frames
        return result

    @staticmethod
    def normalize_interaction_sequence(
        interactions: Optional[Union[str, Sequence[str]]]
    ) -> List[str]:
        """Normalize interaction coordinates and sequences."""
        if interactions is None:
            return []
        if isinstance(interactions, str):
            return [interactions]
        return [str(item) for item in interactions if str(item).strip()]

    @staticmethod
    def is_video_path(path: Union[str, Path]) -> bool:
        """Verify if a given path points to a valid video format."""
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _to_uint8_rgb(image: Any) -> np.ndarray:
        """To uint8 rgb implementation."""
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = np.asarray(image)

        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Unsupported image shape for DepthAnything3: {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        if np.issubdtype(array.dtype, np.floating):
            if array.min() >= -1.0 and array.max() <= 1.0:
                if array.min() < 0.0:
                    array = (array + 1.0) * 127.5
                else:
                    array = array * 255.0
            array = np.clip(array, 0.0, 255.0)
        else:
            array = np.clip(array, 0, 255)

        array = array.astype(np.uint8)
        # Array inputs are expected in RGB channel order; no BGR guessing.
        return array

    def _coerce_single_image(self, input_signal: Any) -> Union[str, np.ndarray]:
        """Coerce single image implementation."""
        if isinstance(input_signal, (str, Path)):
            input_path = Path(input_signal).expanduser()
            if input_path.is_dir():
                raise ValueError(
                    f"Expected a single image but got directory: {input_path}. "
                    "Use collect_paths()/process_perception() for multi-view input."
                )
            if self.is_video_path(input_path):
                raise ValueError(
                    f"Expected image input but got video path: {input_path}. "
                    "Use load_video_frames() for video input."
                )
            return str(input_path.resolve())

        return self._to_uint8_rgb(input_signal)

    def process_perception(
        self,
        input_signal: Union[
            str,
            Path,
            np.ndarray,
            torch.Tensor,
            Image.Image,
            Sequence[str],
            Sequence[np.ndarray],
            Sequence[Image.Image],
        ],
    ) -> List[Union[str, np.ndarray]]:
        """Process perception inputs like images, videos, and reference frames."""
        if isinstance(input_signal, (str, Path)):
            input_path = Path(input_signal).expanduser()
            if input_path.is_dir() or input_path.suffix.lower() == ".txt":
                return self.collect_paths(input_path)
            return [self._coerce_single_image(input_path)]

        if isinstance(input_signal, Sequence) and not isinstance(
            input_signal, (np.ndarray, torch.Tensor, Image.Image)
        ):
            return [self._coerce_single_image(item) for item in input_signal]

        return [self._coerce_single_image(input_signal)]

    def load_video_frames(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
    ) -> Dict[str, Any]:
        """Load video frames from a video path."""
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"DepthAnything3 video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames: List[np.ndarray] = []
        frame_index = 0
        kept_frames = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % max(int(frame_stride), 1) == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                kept_frames += 1
                if max_frames is not None and kept_frames >= max_frames:
                    break
            frame_index += 1

        capture.release()
        if not frames:
            raise ValueError(f"No frames decoded from video: {video_path}")

        return {
            "frames": frames,
            "fps": fps,
            "frame_width": width or frames[0].shape[1],
            "frame_height": height or frames[0].shape[0],
            "video_path": str(video_path),
        }
