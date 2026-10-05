# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RepEncoderConfig
from .blocks import BidirectionalBlock, FinalTargetBlock


class RepFeature(nn.Module):
    def __init__(self, config: RepEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        self.target_embedding = nn.Conv2d(
            6,
            self.config.scene_dim,
            kernel_size=self.config.target_patch_size,
            stride=self.config.target_patch_size,
            bias=False,
        )
        self.target_norm = nn.LayerNorm(self.config.scene_dim, bias=False)
        self.target_register_tokens = nn.Parameter(
            torch.zeros(1, 4, self.config.scene_dim)
        )
        self.blocks = nn.ModuleList(
            [
                BidirectionalBlock(
                    width=self.config.scene_dim,
                    heads=self.config.repfeature_heads,
                )
                for _ in range(self.config.repfeature_depth - 1)
            ]
        )
        self.final_block = FinalTargetBlock(
            width=self.config.scene_dim,
            heads=self.config.repfeature_heads,
        )

    def forward(
        self,
        scene: torch.Tensor,
        target_rays: torch.Tensor,
        *,
        target_microbatch: int = 4,
    ) -> torch.Tensor:
        if scene.ndim != 4 or tuple(scene.shape[1:]) != (
            self.config.source_views,
            self.config.patch_tokens,
            self.config.scene_dim,
        ):
            raise ValueError("scene must be [B,9,814,768]")
        if target_rays.ndim != 5 or tuple(target_rays.shape[1:]) != (
            self.config.target_views,
            6,
            self.config.ray_height,
            self.config.ray_width,
        ):
            raise ValueError("target_rays must be [B,4,6,384,640]")
        if scene.shape[0] != target_rays.shape[0]:
            raise ValueError("scene and target_rays batch sizes differ")
        target_microbatch = int(target_microbatch)
        if target_microbatch <= 0:
            raise ValueError("target_microbatch must be positive")

        batch = scene.shape[0]
        flat_scene = scene.reshape(
            batch,
            self.config.source_views * self.config.patch_tokens,
            self.config.scene_dim,
        )
        pieces: list[torch.Tensor] = []
        for start in range(0, self.config.target_views, target_microbatch):
            rays = target_rays[:, start : start + target_microbatch]
            views = rays.shape[1]
            flat_rays = rays.reshape(
                batch * views, 6, self.config.ray_height, self.config.ray_width
            )
            target = self.target_embedding(flat_rays).flatten(2).transpose(1, 2)
            target = self.target_norm(target)
            registers = self.target_register_tokens.expand(batch * views, -1, -1).to(
                device=target.device, dtype=target.dtype
            )
            target = torch.cat((registers, target), dim=1)
            representation = (
                flat_scene[:, None]
                .expand(batch, views, *flat_scene.shape[1:])
                .reshape(batch * views, *flat_scene.shape[1:])
            )
            for block in self.blocks:
                target, representation = block(target, representation)
            target = self.final_block(target, representation)
            pieces.append(
                target[:, 4:].reshape(
                    batch,
                    views,
                    self.config.latent_height * self.config.latent_width,
                    -1,
                )
            )

        joined = torch.cat(pieces, dim=1)
        expected = (
            batch,
            self.config.target_views,
            self.config.latent_height * self.config.latent_width,
            self.config.scene_dim,
        )
        if tuple(joined.shape) != expected:
            raise RuntimeError(f"repfeature output {tuple(joined.shape)} != {expected}")
        return (
            joined.reshape(
                batch,
                self.config.target_views,
                self.config.latent_height,
                self.config.latent_width,
                self.config.scene_dim,
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )


__all__ = ["RepFeature"]
