"""Depth Anything V2 visual generation pipeline module."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v2 import (
    DepthAnything2Representation,
)
from worldfoundry.core.io import read_video

from ...operators.depth_anything_operator import DepthAnythingOperator
from ..pipeline_utils import PipelineABC
from .pipeline_depth_anything_v1 import DepthResult

logger = logging.getLogger(__name__)


class DepthAnything2Pipeline(PipelineABC):
    """Pipeline wrapper for Depth Anything V2 relative depth estimation."""

    def __init__(
        self,
        representation: Optional[DepthAnything2Representation] = None,
        operator: Optional[DepthAnythingOperator] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
        default_input_size: int = 518,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        if data_type not in {"image", "video"}:
            raise ValueError("data_type must be either 'image' or 'video'")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.data_type = data_type
        self.default_input_size = int(default_input_size)
        self.representation = representation
        self.operator = operator or DepthAnythingOperator()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, Any]] = None,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
        default_input_size: int = 518,
        **kwargs,
    ) -> "DepthAnything2Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop("model_path", None)
        pretrained_model_path = (
            pretrained_model_path
            or model_path
            or component_options.pop("pretrained_model_path", None)
        )
        encoder = component_options.pop("encoder", encoder)
        data_type = component_options.pop("data_type", data_type)
        default_input_size = component_options.pop("default_input_size", default_input_size)
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})

        representation = DepthAnything2Representation.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            encoder=encoder,
            device=device,
            default_input_size=default_input_size,
            **kwargs,
        )
        return cls(
            representation=representation,
            operator=DepthAnythingOperator(),
            encoder=representation.encoder,
            device=device,
            data_type=data_type,
            default_input_size=default_input_size,
        )

    def _resolve_raw_bgr(
        self,
        input_image: Union[str, Path, np.ndarray, torch.Tensor],
        color_order: str = "rgb",
    ) -> np.ndarray:
        """Resolve raw bgr for DepthAnything2Pipeline."""
        if isinstance(input_image, (str, Path)):
            raw_bgr = cv2.imread(str(input_image))
            if raw_bgr is None:
                raise ValueError(f"Could not read image from {input_image}")
            return raw_bgr
        return DepthAnything2Representation._coerce_raw_bgr(
            input_image,
            color_order=color_order,
        )

    def process(
        self,
        input_image: Union[str, Path, np.ndarray, torch.Tensor],
        return_visualization: bool = False,
        grayscale: bool = False,
        input_size: Optional[int] = None,
        color_order: str = "rgb",
    ) -> Union[torch.Tensor, np.ndarray]:
        """Process and normalize input arguments and conditions for inference."""
        if self.representation is None:
            raise RuntimeError("Representation not loaded. Use from_pretrained() first.")

        raw_bgr = self._resolve_raw_bgr(input_image, color_order=color_order)
        result = self.representation.get_representation(
            {
                "raw_bgr": raw_bgr,
                "return_visualization": return_visualization,
                "grayscale": grayscale,
                "input_size": input_size or self.default_input_size,
            }
        )
        if return_visualization:
            return result["depth_visualization"]
        return result["depth"]

    def run_image(
        self,
        img_path: str,
        grayscale: bool = False,
        input_size: Optional[int] = None,
    ) -> DepthResult:
        """Run image for DepthAnything2Pipeline."""
        results: List[Dict] = []

        for filename in tqdm(self.operator.collect_paths(img_path), desc="DepthAnythingV2-Image"):
            try:
                depth_vis = self.process(
                    filename,
                    return_visualization=True,
                    grayscale=grayscale,
                    input_size=input_size,
                )
                basename = os.path.basename(filename)
                stem = basename[: basename.rfind(".")] if "." in basename else basename
                results.append({"image": depth_vis, "filename": filename, "stem": stem})
            except Exception:
                logger.warning("Skipping image %s after processing failure", filename, exc_info=True)
                continue

        return DepthResult(results, data_type="image")

    def run_video(
        self,
        video_path: str,
        grayscale: bool = False,
        input_size: Optional[int] = None,
    ) -> DepthResult:
        """Run video for DepthAnything2Pipeline."""
        results: List[Dict] = []

        for index, filename in enumerate(self.operator.collect_paths(video_path), start=1):
            try:
                raw_frames, metadata = read_video(filename)
            except Exception:
                # Skipped samples silently distort benchmark counts, so record them.
                logger.warning("Skipping unreadable video %s", filename, exc_info=True)
                continue

            frame_height, frame_width = raw_frames.shape[1:3]
            frame_rate = float(metadata.get("fps") or metadata.get("framerate") or 30)
            basename = os.path.basename(filename)
            stem = basename[: basename.rfind(".")] if "." in basename else basename

            frames: List[np.ndarray] = []
            with tqdm(
                total=len(raw_frames),
                desc=f"Video {index}",
                unit="frame",
            ) as progress:
                for raw_frame in raw_frames:
                    depth_vis = self.process(
                        raw_frame,
                        return_visualization=True,
                        grayscale=grayscale,
                        input_size=input_size,
                        color_order="rgb",
                    )
                    frames.append(depth_vis)
                    progress.update(1)

            results.append(
                {
                    "frames": frames,
                    "filename": filename,
                    "stem": stem,
                    "frame_rate": frame_rate,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                }
            )

        return DepthResult(results, data_type="video")

    def __call__(
        self,
        data_path: str,
        grayscale: bool = False,
        input_size: Optional[int] = None,
        **kwargs,
    ) -> DepthResult:
        """Execute the complete pipeline generation flow."""
        del kwargs
        if self.data_type == "image":
            return self.run_image(data_path, grayscale=grayscale, input_size=input_size)
        return self.run_video(data_path, grayscale=grayscale, input_size=input_size)


__all__ = ["DepthAnything2Pipeline", "DepthResult"]
