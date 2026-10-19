# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""WorldPlay2 causal memory compression with shared Wan VAE primitives."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    AttentionBlock,
    AvgDown3D,
    CausalConv3d,
)
from worldfoundry.core.attention import scaled_dot_product_attention


class MemoryAttentionBlock(AttentionBlock):
    """The released compressor's eight-head, per-frame Wan attention."""

    def __init__(self, dim: int, head_dim: int = 8):
        super().__init__(dim)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch, channels, time, height, width = x.shape
        frames = rearrange(x, "b c t h w -> (b t) c h w")
        qkv = self.to_qkv(self.norm(frames))
        qkv = qkv.reshape(batch * time, 1, channels * 3, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        q, k, v = (
            rearrange(item, "b h l (n d) -> b (h n) l d", n=self.head_dim)
            for item in (q, k, v)
        )
        frames = scaled_dot_product_attention(q, k, v)
        frames = rearrange(frames, "b h l d -> b (h d) l")
        frames = self.proj(frames.reshape(batch * time, channels, height, width))
        return rearrange(frames, "(b t) c h w -> b c t h w", t=time) + identity


class CausalCompressConv3D(nn.Module):
    """Causal temporal convolution and channel-folded downsample shortcut."""

    def __init__(self, in_dim, out_dim, spatial_down, temporal_down, non_linearity="silu"):
        super().__init__()
        self.conv = CausalConv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        if non_linearity != "silu":
            raise ValueError("WorldPlay2 memory compression requires SiLU")
        self.nonlinearity = nn.SiLU()
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down
        if spatial_down:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(out_dim, out_dim, 3, stride=2),
            )
        if temporal_down:
            self.time_conv = CausalConv3d(
                out_dim, out_dim, kernel_size=(3, 1, 1),
                stride=(2, 1, 1), padding=(1, 0, 0),
            )
        self.avg_shortcut = (
            AvgDown3D(in_dim, out_dim, 2 if temporal_down else 1, 2 if spatial_down else 1)
            if spatial_down or temporal_down
            else nn.Conv3d(in_dim, out_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.nonlinearity(self.conv(x))
        if self.spatial_down:
            time = hidden.shape[2]
            hidden = rearrange(hidden, "b c t h w -> (b t) c h w")
            hidden = self.resample(hidden)
            hidden = rearrange(hidden, "(b t) c h w -> b c t h w", t=time)
        if self.temporal_down:
            hidden = self.time_conv(hidden)
        return hidden + self.avg_shortcut(x)


class CausalMemCompressModel(nn.Module):
    """Map HR history to tokens aligned with LR Wan patch embeddings."""

    def __init__(
        self,
        output_dim=2048,
        input_dim=65,
        dims=(64, 64, 128, 256, 256, 512, 512),
        spatial_down=(1, 1, 0, 0, 0, 0),
        temporal_down=(1, 0, 0, 0, 0, 0),
        attn_num=2,
        non_linearity="silu",
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, dims[0])
        self.blocks = nn.ModuleList(
            CausalCompressConv3D(
                in_dim, out_dim, spatial_down[index], temporal_down[index], non_linearity,
            )
            for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
        )
        self.attn_blocks = nn.ModuleList(MemoryAttentionBlock(dims[-1]) for _ in range(attn_num))
        self.output_layer = nn.Linear(dims[-1], output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, height, width = x.shape[2:]
        hidden = self.input_layer(rearrange(x, "b c t h w -> b (t h w) c"))
        hidden = rearrange(hidden, "b (t h w) c -> b c t h w", t=time, h=height, w=width)
        for block in self.blocks:
            hidden = block(hidden)
        for block in self.attn_blocks:
            hidden = block(hidden)
        return self.output_layer(rearrange(hidden, "b c t h w -> b (t h w) c"))
