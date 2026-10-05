# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RepEncoderConfig


class InputLayer(nn.Module):
    def __init__(
        self,
        *,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        config: RepEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        mean = torch.zeros(self.config.latent_channels, dtype=torch.float32)
        std = torch.ones(self.config.latent_channels, dtype=torch.float32)
        if latents_mean is not None:
            mean = torch.as_tensor(latents_mean, dtype=torch.float32).reshape(-1)
        if latents_std is not None:
            std = torch.as_tensor(latents_std, dtype=torch.float32).reshape(-1)
        expected = (self.config.latent_channels,)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError("latents_mean and latents_std must each contain 16 values")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("latent statistics must be finite and std must be positive")
        self.register_buffer("latents_mean", mean, persistent=True)
        self.register_buffer("latents_std", std, persistent=True)
        self.proj = nn.Conv2d(
            self.config.latent_channels,
            self.config.dino_dim,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, source_latents: torch.Tensor) -> torch.Tensor:
        expected_tail = (
            self.config.source_views,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )
        if source_latents.ndim != 5 or tuple(source_latents.shape[1:]) != expected_tail:
            raise ValueError(
                f"source_latents must be [B,{','.join(map(str, expected_tail))}], "
                f"got {tuple(source_latents.shape)}"
            )
        if not torch.is_floating_point(source_latents):
            raise TypeError("source_latents must be floating point")
        if not torch.isfinite(source_latents).all():
            raise FloatingPointError("source_latents contain NaN or Inf")
        batch = source_latents.shape[0]
        flat = source_latents.reshape(
            batch * self.config.source_views,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )
        mean = self.latents_mean.to(device=flat.device, dtype=flat.dtype)
        std = self.latents_std.to(device=flat.device, dtype=flat.dtype)
        raw = flat * std[None, :, None, None] + mean[None, :, None, None]
        raw = F.interpolate(raw, size=(44, 74), mode="bilinear", align_corners=True)
        prediction = self.proj(F.pad(raw, (1, 1, 1, 1), mode="replicate"))
        expected = (
            batch * self.config.source_views,
            self.config.dino_dim,
            self.config.patch_height,
            self.config.patch_width,
        )
        if tuple(prediction.shape) != expected:
            raise RuntimeError(f"input_layer output {tuple(prediction.shape)} != {expected}")
        return prediction.permute(0, 2, 3, 1).reshape(
            batch,
            self.config.source_views,
            self.config.patch_tokens,
            self.config.dino_dim,
        )


__all__ = ["InputLayer"]
