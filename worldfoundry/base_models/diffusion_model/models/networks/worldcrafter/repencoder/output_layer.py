# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RepEncoderConfig


class OutputLayer(nn.Module):
    def __init__(self, config: RepEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        self.proj = nn.Conv3d(
            self.config.scene_dim,
            self.config.latent_channels,
            kernel_size=self.config.output_kernel,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.scene_dim,
            self.config.target_views,
            self.config.latent_height,
            self.config.latent_width,
        )
        if features.ndim != 5 or tuple(features.shape[1:]) != expected:
            raise ValueError(f"features must be [B,{expected}], got {tuple(features.shape)}")
        temporal, height, width = self.config.output_kernel
        padded = F.pad(
            features,
            (width // 2, width // 2, height // 2, height // 2, temporal // 2, temporal // 2),
            mode="replicate",
        )
        latent = self.proj(padded)
        expected_output = (
            features.shape[0],
            self.config.latent_channels,
            self.config.target_views,
            self.config.latent_height,
            self.config.latent_width,
        )
        if tuple(latent.shape) != expected_output:
            raise RuntimeError(f"memory4 shape {tuple(latent.shape)} != {expected_output}")
        return latent


__all__ = ["OutputLayer"]
