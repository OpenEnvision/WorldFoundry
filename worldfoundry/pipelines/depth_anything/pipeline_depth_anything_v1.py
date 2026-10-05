"""Depth Anything V1 visual generation pipeline module."""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v1 import (
    DepthAnything1Representation,
)
from worldfoundry.core.io import artifact_root_path, read_video, write_video

from ...operators.depth_anything_operator import DepthAnythingOperator
from ..pipeline_utils import PipelineABC

logger = logging.getLogger(__name__)


def _default_output_dir(name: str = "vis_depth") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


class DepthResult:
    """Container class for depth estimation results."""
    
    def __init__(self, results: List[Dict], data_type: str):
        """
        Initialize depth result container.
        
        Args:
            results: List of dictionaries containing:
                - For images: {'image': np.ndarray, 'filename': str, 'stem': str}
                - For videos: {'frames': List[np.ndarray], 'filename': str, 'stem': str, 
                              'frame_rate': float, 'frame_width': int, 'frame_height': int}
            data_type: Type of data ('image' or 'video')
        """
        self.results = results
        self.data_type = data_type
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save depth results to files.
        
        Args:
            output_dir: Output directory. If None, uses default based on data_type.
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = _default_output_dir(
                "vis_depth" if self.data_type == "image" else "vis_video_depth"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        if self.data_type == "image":
            for result in self.results:
                output_path = os.path.join(output_dir, f"{result['stem']}_depth.png")
                cv2.imwrite(output_path, result['image'])
                saved_files.append(output_path)
        else:  # video
            for result in self.results:
                output_path = os.path.join(output_dir, f"{result['stem']}_depth.mp4")
                write_video(result["frames"], output_path, fps=int(round(result["frame_rate"])))
                saved_files.append(output_path)
        
        return saved_files
    
    def __len__(self):
        """Len for DepthResult."""
        return len(self.results)
    
    def __getitem__(self, idx):
        """Getitem for DepthResult."""
        return self.results[idx]


class DepthAnything1Pipeline(PipelineABC):
    """Pipeline for Depth Anything (V1) depth estimation."""
    
    def __init__(
        self,
        representation: Optional[DepthAnything1Representation] = None,
        operator: Optional[DepthAnythingOperator] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
    ) -> None:
        """
        Args:
            representation: Pre-loaded DepthAnything1Representation instance (optional)
            operator: DepthAnythingOperator instance (optional)
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on ('cuda' or 'cpu')
            data_type: Type of data to process ('image' or 'video')
        """
        if data_type not in {"image", "video"}:
            raise ValueError("data_type must be either 'image' or 'video'")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.data_type = data_type
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
        **kwargs
    ) -> 'DepthAnything1Pipeline':
        """
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            data_type: Type of data to process ('image' or 'video')
            **kwargs: Additional arguments
        """
        # Load representation from pretrained model
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
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})

        representation = DepthAnything1Representation.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            encoder=encoder,
            device=device,
            **kwargs
        )
        
        return cls(
            representation=representation,
            encoder=encoder,
            device=device,
            data_type=data_type,
        )
    
    def process(
        self,
        input_image: Union[str, np.ndarray, torch.Tensor],
        return_visualization: bool = False,
        grayscale: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process input image and return depth map.
        
        Args:
            input_image: Image path, numpy array, or torch tensor
            return_visualization: If True, return visualized depth (uint8), else return raw depth tensor
            grayscale: For visualization, whether to use grayscale (ignored if return_visualization=False)
            
        Returns:
            Depth map as tensor (if return_visualization=False) or visualized array (if return_visualization=True)
        """
        if self.representation is None:
            raise RuntimeError("Representation not loaded. Use from_pretrained() first.")
        
        # Load and preprocess image using operator's process_perception method
        image_rgb = self.operator.process_perception(input_image)
        
        # Get depth representation
        data = {
            'image': image_rgb,
            'return_visualization': return_visualization,
            'grayscale': grayscale,
        }
        result = self.representation.get_representation(data)
        
        # Return based on request
        if return_visualization:
            return result['depth_visualization']
        else:
            return result['depth']
    
    def run_image(
        self,
        img_path: str,
        grayscale: bool = False,
    ) -> DepthResult:
        """
        Process images and return depth maps.
        
        Args:
            img_path: Image file, directory, or txt file with paths
            grayscale: If True, return grayscale depth, else color map
            
        Returns:
            DepthResult object containing processed depth images
        """
        results: List[Dict] = []
        
        for filename in tqdm(self.operator.collect_paths(img_path), desc="DepthAnything-Image"):
            try:
                depth_vis = self.process(filename, return_visualization=True, grayscale=grayscale)
                
                basename = os.path.basename(filename)
                stem = basename[:basename.rfind(".")] if "." in basename else basename
                
                results.append({
                    'image': depth_vis,
                    'filename': filename,
                    'stem': stem,
                })
            except Exception:
                logger.warning("Skipping image %s after processing failure", filename, exc_info=True)
                continue
        
        return DepthResult(results, data_type="image")
    
    def run_video(
        self,
        video_path: str,
    ) -> DepthResult:
        """
        Process videos and return depth video frames.
        
        Args:
            video_path: Video file, directory, or txt file with paths
            
        Returns:
            DepthResult object containing processed depth video frames
        """
        results: List[Dict] = []
        
        for k, filename in enumerate(self.operator.collect_paths(video_path), start=1):
            try:
                raw_frames, metadata = read_video(filename)
            except Exception:
                # Skipped samples silently distort benchmark counts, so record them.
                logger.warning("Skipping unreadable video %s", filename, exc_info=True)
                continue

            frame_height, frame_width = raw_frames.shape[1:3]
            frame_rate = float(metadata.get("fps") or metadata.get("framerate") or 30)
            
            basename = os.path.basename(filename)
            stem = basename[:basename.rfind(".")] if "." in basename else basename
            
            frames: List[np.ndarray] = []
            with tqdm(total=len(raw_frames), desc=f"Video {k}", unit="frame") as pbar:
                for raw_frame in raw_frames:
                    depth_color = self.process(raw_frame, return_visualization=True, grayscale=False)
                    frames.append(depth_color)
                    pbar.update(1)
            
            results.append({
                'frames': frames,
                'filename': filename,
                'stem': stem,
                'frame_rate': frame_rate,
                'frame_width': frame_width,
                'frame_height': frame_height,
            })
        
        return DepthResult(results, data_type="video")
    
    def __call__(
        self,
        data_path: str | None = None,
        grayscale: bool = False,
        *,
        images: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> DepthResult | dict[str, Any]:
        """
        Main call interface for the pipeline.
        
        Args:
            data_path: Path to image/video file, directory, or txt file
            grayscale: For image mode, whether to use grayscale (ignored for video)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            DepthResult object containing processed depth results
        """
        del kwargs
        routed_inputs = dict(operator_kwargs or {})
        source = data_path or images or video
        if source is None:
            source = routed_inputs.get("input") or routed_inputs.get("input_path")
        if source is None:
            raise ValueError("Depth Anything V1 requires an image or video input path")

        result = (
            self.run_image(str(source), grayscale=grayscale)
            if self.data_type == "image"
            else self.run_video(str(source))
        )
        if not return_dict:
            return result
        if output_path is None:
            output_dir = _default_output_dir(
                "vis_depth" if self.data_type == "image" else "vis_video_depth"
            )
        else:
            output_dir = str(Path(output_path).expanduser().parent)
        artifacts = result.save(output_dir)
        if not artifacts:
            raise RuntimeError("Depth Anything V1 produced no output artifacts")
        return {
            "status": "succeeded",
            "artifact_path": artifacts[0],
            "artifacts": artifacts,
        }


__all__ = ["DepthAnything1Pipeline", "DepthResult"]
