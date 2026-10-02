import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dpt import (
    DepthAnything,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from worldfoundry.core.model_loading import load_torch_checkpoint


class DepthAnything1Representation:
    """Representation for Depth Anything V1 depth estimation model."""
    
    def __init__(self, model: Optional[torch.nn.Module] = None, device: Optional[str] = None):
        """
        Initialize DepthAnything1Representation.
        
        Args:
            model: Pre-loaded DepthAnything model (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        # Initialize transform if model is provided
        if self.model is not None:
            self._init_transform()
            self.model = self.model.to(self.device).eval()
    
    def _init_transform(self):
        """Initialize image transformation pipeline."""
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image tensor for model inference."""
        if self.transform is None:
            raise RuntimeError("Transform not initialized. Model must be loaded first.")
        tensor = self.transform({"image": image})["image"]
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
    
    @classmethod
    def load_model(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        Load DepthAnything model from local checkpoint or HuggingFace repository.
        
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            **kwargs: Additional arguments
            
        Returns:
            Loaded DepthAnything model
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from local path or HuggingFace repo
        if pretrained_model_path and Path(pretrained_model_path).exists():
            model_path = Path(pretrained_model_path)
            config_path = model_path / "config.json" if model_path.is_dir() else None
            config = {}
            if config_path is not None and config_path.is_file():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    config = {}
            if config.get("model_type") == "depth_anything":
                from transformers import AutoModelForDepthEstimation

                model = AutoModelForDepthEstimation.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )
            else:
                model = cls._load_from_local(pretrained_model_path, encoder, device)
        else:
            model = cls._load_from_huggingface(pretrained_model_path, encoder, device)
        
        return model.to(device).eval()
    
    @staticmethod
    def _load_from_local(
        pretrained_model_path: str,
        encoder: str,
        device: str
    ) -> DepthAnything:
        """Load model from local checkpoint file."""
        model_path = Path(pretrained_model_path)
        if model_path.is_dir():
            return DepthAnything.from_pretrained(str(model_path))

        # The checkpoint is consumed purely as a state_dict; load weights-only by
        # default and only fall back to unrestricted pickle on legacy wrapped
        # checkpoints that fail the safe path.
        checkpoint = load_torch_checkpoint(
            model_path,
            map_location="cpu",
            weights_only=True,
            allow_unsafe_pickle_fallback=True,
        )

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        detected_encoder = encoder
        if 'pretrained.cls_token' in state_dict:
            embed_dim = state_dict['pretrained.cls_token'].shape[-1]
            if embed_dim == 384:
                detected_encoder = 'vits'
            elif embed_dim == 768:
                detected_encoder = 'vitb'
            elif embed_dim == 1024:
                detected_encoder = 'vitl'
        elif 'pretrained.pos_embed' in state_dict:
            embed_dim = state_dict['pretrained.pos_embed'].shape[-1]
            if embed_dim == 384:
                detected_encoder = 'vits'
            elif embed_dim == 768:
                detected_encoder = 'vitb'
            elif embed_dim == 1024:
                detected_encoder = 'vitl'
        
        detected_out_channels = None
        if 'depth_head.projects.0.weight' in state_dict:
            detected_out_channels = [
                state_dict['depth_head.projects.0.weight'].shape[0],
                state_dict['depth_head.projects.1.weight'].shape[0],
                state_dict['depth_head.projects.2.weight'].shape[0],
                state_dict['depth_head.projects.3.weight'].shape[0],
            ]
        
        # Model configurations
        encoder_configs = {
            'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            if detected_encoder != encoder:
                model_config['encoder'] = detected_encoder
            if detected_out_channels:
                model_config['out_channels'] = detected_out_channels
                if detected_encoder in encoder_configs:
                    model_config['features'] = encoder_configs[detected_encoder]['features']
        else:
            if detected_encoder in encoder_configs:
                base_config = encoder_configs[detected_encoder].copy()
                model_config = {
                    'encoder': detected_encoder,
                    'features': base_config['features'],
                    'out_channels': detected_out_channels or base_config['out_channels'],
                    'use_bn': False,
                    'use_clstoken': False,
                    'localhub': True,
                }
            else:
                model_config = {
                    'encoder': detected_encoder,
                    'features': 256,
                    'out_channels': detected_out_channels or [256, 512, 1024, 1024],
                    'use_bn': False,
                    'use_clstoken': False,
                    'localhub': True,
                }
        
        model = DepthAnything(model_config)
        model.load_state_dict(state_dict, strict=False)
        return model
    
    @staticmethod
    def _load_from_huggingface(
        pretrained_model_path: Optional[str],
        encoder: str,
        device: str
    ) -> DepthAnything:
        """Load model from HuggingFace repository."""
        model_source = pretrained_model_path or f"LiheYoung/depth_anything_{encoder}14"
        return DepthAnything.from_pretrained(model_source)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        **kwargs
    ) -> 'DepthAnything1Representation':
        """
        Create representation instance from pretrained model.
        
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            **kwargs: Additional arguments
            
        Returns:
            DepthAnything1Representation instance
        """
        model = cls.load_model(
            pretrained_model_path=pretrained_model_path,
            encoder=encoder,
            device=device,
            **kwargs
        )
        
        return cls(model=model, device=device)
    
    def get_representation(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get depth representation from input data.
        
        Args:
            data: Dictionary containing:
                - 'image': Input image as numpy array (H, W, 3) in RGB format, normalized to [0, 1]
                - Optional: 'return_visualization': If True, return visualized depth
                - Optional: 'grayscale': For visualization, whether to use grayscale
                
        Returns:
            Dictionary containing:
                - 'depth': Depth map as torch.Tensor (H, W)
                - Optional: 'depth_visualization': Visualized depth as numpy array (if return_visualization=True)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() or load_model() first.")
        
        input_image = data['image']
        return_visualization = data.get('return_visualization', False)
        grayscale = data.get('grayscale', False)
        
        # Prepare tensor and run inference
        tensor = self._prepare_tensor(input_image)
        h, w = input_image.shape[:2]
        
        with torch.no_grad():
            model_output = self.model(tensor)
        depth = self._prediction_tensor(model_output)
        
        # Interpolate to original size
        import torch.nn.functional as F
        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        
        result = {'depth': depth}
        
        # Add visualization if requested
        if return_visualization:
            from worldfoundry.core.io.artifacts import (
                depth_to_colormap_rgb,
                depth_to_uint8,
            )

            depth_uint8 = depth_to_uint8(depth.detach().cpu().numpy())
            if depth_uint8 is None:
                raise ValueError(f"Unexpected depth shape for visualization: {tuple(depth.shape)}")
            if grayscale:
                depth_vis = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = depth_to_colormap_rgb(depth_uint8)
            result['depth_visualization'] = depth_vis
        
        return result

    @staticmethod
    def _prediction_tensor(model_output: Any) -> torch.Tensor:
        """Normalize in-tree and Transformers Depth Anything outputs."""

        if isinstance(model_output, torch.Tensor):
            return model_output
        predicted_depth = getattr(model_output, "predicted_depth", None)
        if isinstance(predicted_depth, torch.Tensor):
            return predicted_depth
        if isinstance(model_output, dict) and isinstance(model_output.get("predicted_depth"), torch.Tensor):
            return model_output["predicted_depth"]
        raise TypeError(
            "Depth Anything model must return a tensor or an object containing predicted_depth"
        )
