# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from ..config import RepEncoderConfig
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import MemEffAttention
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import Block
from worldfoundry.core.attention.rotary.rope_2d import PositionGetter, RotaryPositionEmbedding2D


def _slice_expand_and_flatten(token: torch.Tensor, batch: int, views: int) -> torch.Tensor:
    first = token[:, 0:1].expand(batch, 1, *token.shape[2:])
    others = token[:, 1:].expand(batch, views - 1, *token.shape[2:])
    return torch.cat((first, others), dim=1).reshape(batch * views, *token.shape[2:])


class DinoTail(nn.Module):
    def __init__(self, config: RepEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        self.latent_cls_token = nn.Parameter(torch.zeros(1, 1, self.config.dino_dim))
        self.latent_register_tokens = nn.Parameter(torch.zeros(1, 4, self.config.dino_dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.config.dino_dim,
                    num_heads=self.config.dino_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1.0,
                    norm_layer=norm_layer,
                    attn_class=MemEffAttention,
                    qk_norm=False,
                )
                for _ in range(self.config.dino_depth - self.config.dino_input_block)
            ]
        )
        self.norm = norm_layer(self.config.dino_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.source_views,
            self.config.patch_tokens,
            self.config.dino_dim,
        )
        if patch_tokens.ndim != 4 or tuple(patch_tokens.shape[1:]) != expected:
            raise ValueError(f"DINO input must be [B,{expected}], got {tuple(patch_tokens.shape)}")
        batch = patch_tokens.shape[0]
        flat = patch_tokens.reshape(
            batch * self.config.source_views,
            self.config.patch_tokens,
            self.config.dino_dim,
        )
        prefix = torch.cat(
            (
                self.latent_cls_token.expand(flat.shape[0], -1, -1),
                self.latent_register_tokens.expand(flat.shape[0], -1, -1),
            ),
            dim=1,
        ).to(device=flat.device, dtype=flat.dtype)
        tokens = torch.cat((prefix, flat), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)[:, self.config.latent_special_tokens :]
        return tokens.reshape(batch, *expected)


class RepresentationBackbone(nn.Module):
    def __init__(self, config: RepEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        block_args = dict(
            dim=self.config.dino_dim,
            num_heads=self.config.vggt_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            init_values=0.01,
            qk_norm=True,
            rope=self.rope,
        )
        self.frame_blocks = nn.ModuleList(
            [Block(**block_args) for _ in range(self.config.vggt_depth)]
        )
        self.global_blocks = nn.ModuleList(
            [Block(**block_args) for _ in range(self.config.vggt_depth)]
        )
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, self.config.dino_dim))
        self.register_token = nn.Parameter(
            torch.zeros(1, 2, self.config.vggt_register_tokens, self.config.dino_dim)
        )
        self.camera_mlp = nn.Sequential(
            nn.Linear(11, self.config.dino_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.config.dino_dim, self.config.dino_dim, bias=True),
        )
        self.geo_feature_connector = nn.Linear(
            self.config.dino_dim * 2, self.config.scene_dim, bias=True
        )
        self.geo_feature_norm = nn.LayerNorm(self.config.scene_dim, bias=False)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        camera_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, views, patches, channels = patch_tokens.shape
        expected = (
            self.config.source_views,
            self.config.patch_tokens,
            self.config.dino_dim,
        )
        if tuple(patch_tokens.shape[1:]) != expected:
            raise ValueError(f"VGGT patch input must end in {expected}")
        if tuple(camera_tokens.shape) != (batch, views, 11):
            raise ValueError("camera_tokens must be [B,9,11]")
        flat_patch = patch_tokens.reshape(batch * views, patches, channels)
        projected_camera = self.camera_mlp(camera_tokens).unsqueeze(2)
        camera = _slice_expand_and_flatten(self.camera_token, batch, views)
        camera = camera + projected_camera.reshape(batch * views, 1, channels)
        register = _slice_expand_and_flatten(self.register_token, batch, views)
        tokens = torch.cat((camera, register, flat_patch), dim=1)
        total_tokens = tokens.shape[1]

        pos = self.position_getter(
            batch * views,
            self.config.patch_height,
            self.config.patch_width,
            device=patch_tokens.device,
        )
        pos = pos + 1
        special = torch.zeros(
            batch * views,
            self.config.latent_special_tokens,
            2,
            device=patch_tokens.device,
            dtype=pos.dtype,
        )
        pos = torch.cat((special, pos), dim=1)

        frame_index = 0
        global_index = 0
        last_frame = None
        last_global = None
        for _ in range(self.config.vggt_depth):
            if tokens.shape != (batch * views, total_tokens, channels):
                tokens = tokens.reshape(batch, views, total_tokens, channels).reshape(
                    batch * views, total_tokens, channels
                )
            frame_pos = pos.reshape(batch, views, total_tokens, 2).reshape(
                batch * views, total_tokens, 2
            )
            tokens = self.frame_blocks[frame_index](tokens, pos=frame_pos)
            frame_index += 1
            last_frame = tokens.reshape(batch, views, total_tokens, channels)

            tokens = tokens.reshape(batch, views, total_tokens, channels).reshape(
                batch, views * total_tokens, channels
            )
            global_pos = pos.reshape(batch, views, total_tokens, 2).reshape(
                batch, views * total_tokens, 2
            )
            tokens = self.global_blocks[global_index](tokens, pos=global_pos)
            global_index += 1
            last_global = tokens.reshape(batch, views, total_tokens, channels)

        if last_frame is None or last_global is None:
            raise RuntimeError("VGGT representation path produced no output")
        combined = torch.cat((last_frame, last_global), dim=-1)
        patch = combined[:, :, self.config.latent_special_tokens :]
        scene = self.geo_feature_connector(patch)
        return self.geo_feature_norm(scene)


__all__ = ["DinoTail", "RepresentationBackbone"]
