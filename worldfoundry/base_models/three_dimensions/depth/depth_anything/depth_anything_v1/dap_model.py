"""Inference-only Depth Any Panoramas model."""

from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v2.dpt import (
    DPTHead,
)

from .dap_dino import DINOv3Adapter

_MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}

_INTERMEDIATE_LAYERS = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
}


class DepthAnythingV2(nn.Module):
    """DAP depth/mask heads on the shared DINOv3 backbone."""

    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: list[int] | None = None,
        use_bn: bool = False,
        use_clstoken: bool = False,
        max_depth: float = 20.0,
    ) -> None:
        super().__init__()
        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unsupported DAP model type {encoder!r}; expected one of {sorted(_MODEL_CONFIGS)}.")
        out_channels = out_channels or list(_MODEL_CONFIGS[encoder]["out_channels"])
        self.intermediate_layer_idx = _INTERMEDIATE_LAYERS
        self.max_depth = max_depth
        self.encoder = encoder
        self.pretrained = DINOv3Adapter(model_name=encoder)
        head_kwargs = {
            "out_channels": out_channels,
            "use_clstoken": use_clstoken,
            "is_metric": True,
        }
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            **head_kwargs,
        )
        self.mask_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            **head_kwargs,
        )
        self.patch_size = self.pretrained.patch_size

    def extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Run the shared DINO encoder once and return DAP feature maps."""
        return list(
            self.pretrained.get_intermediate_layers(
                image,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=False,
            )
        )

    def decode_features(
        self,
        features: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode cached DINO features into depth and invalid-mask maps."""
        patch_height = image_size[0] // self.patch_size
        patch_width = image_size[1] // self.patch_size
        depth = (
            self.depth_head(
                features,
                patch_height,
                patch_width,
                self.patch_size,
            )
            * self.max_depth
        )
        mask = self.mask_head(
            features,
            patch_height,
            patch_width,
            self.patch_size,
        )
        return depth, mask

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, mask = self.decode_features(self.extract_features(image), image.shape[-2:])
        return depth.squeeze(1), mask.squeeze(1)


class DAP(nn.Module):
    """Panorama depth inference model."""

    def __init__(self, model_type: str = "vitl", max_depth: float = 1.0) -> None:
        super().__init__()
        if model_type not in _MODEL_CONFIGS:
            raise ValueError(f"Unsupported DAP model type {model_type!r}; expected one of {sorted(_MODEL_CONFIGS)}.")
        config = _MODEL_CONFIGS[model_type]
        self.max_depth = max_depth
        self.core = DepthAnythingV2(
            encoder=model_type,
            features=int(config["features"]),
            out_channels=list(config["out_channels"]),
            max_depth=1.0,
        )
        self.core.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        depth, invalid_mask = self.core(image)
        return {
            "pred_depth": depth.unsqueeze(1).clamp_min(0.0) * self.max_depth,
            "pred_mask": invalid_mask.unsqueeze(1),
        }


def make_dap_model(
    midas_model_type: str = "vitl",
    max_depth: float = 1.0,
) -> DAP:
    """Build DAP without downloading weights or embedding another backbone."""
    return DAP(model_type=midas_model_type, max_depth=max_depth)
