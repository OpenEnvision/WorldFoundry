# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License).
"""Causal ConvNeXt pixel decoders for FastFrames (Sec. 3.2 of arXiv:2611.23226).

Maps pre-head DiT hidden states ``[B, F*H*W, dim]`` and head modulation
``[B, F, 1, dim]`` directly into RGB video ``[B, 4*F, 3, 16*H, 16*W]`` in
``[-1, 1]``, replacing the Wan2.1 VAE decoder. Reuses canonical ``WanLayerNorm``
while preserving exact released checkpoint parameter keys.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import (
    WanLayerNorm,
)

__all__ = [
    "BridgedPixelDecoder",
    "PixelDecoder",
    "align_frames",
    "unpatchify_tokens",
]


def unpatchify_tokens(
    tokens: torch.Tensor,
    grid_sizes: torch.Tensor,
    patch_size: tuple[int, int, int],
    out_dim: int,
) -> torch.Tensor:
    """Unpack ``[B, F, H*W, prod(patch) * C]`` tokens to ``[B, F*p_t, C, H*p_h, W*p_w]``."""
    p_t, p_h, p_w = patch_size
    outputs: list[torch.Tensor] = []
    for item, (f, h, w) in zip(tokens, grid_sizes.tolist(), strict=True):
        unpacked = torch.einsum("fhwpqrc->cfphqwr", item.reshape(f, h, w, p_t, p_h, p_w, out_dim))
        outputs.append(unpacked.reshape(out_dim, f * p_t, h * p_h, w * p_w))
    return torch.stack(outputs, dim=0).permute(0, 2, 1, 3, 4).contiguous()


def align_frames(pixels: torch.Tensor, num_frames: int = 81) -> torch.Tensor:
    """Drop leading warm-up frames so ``pixels`` has ``num_frames`` frames (e.g. 84 -> 81)."""
    if pixels.shape[1] < num_frames:
        raise ValueError(f"Pixel decoder produced {pixels.shape[1]} < {num_frames} frames")
    return pixels[:, pixels.shape[1] - num_frames :]


class _CausalConv3d(nn.Conv3d):
    """3-D convolution with left-only temporal padding and symmetric spatial padding."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.padding, tuple) or len(self.padding) != 3:
            raise ValueError(f"_CausalConv3d requires 3-tuple integer padding, got {self.padding!r}")
        pad_t, pad_h, pad_w = (int(v) for v in self.padding)
        self._pad: tuple[int, int, int, int, int, int] = (pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)
        self.padding = (0, 0, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(input, self._pad))


class _ChannelLayerNorm3d(nn.Module):
    """Channel-wise LayerNorm over ``[B, C, T, H, W]`` feature volumes."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moved = x.movedim(1, -1)
        normed = F.layer_norm(moved, moved.shape[-1:], self.weight, self.bias, self.eps)
        return normed.movedim(-1, 1).contiguous()


class _CausalResDWBlock3d(nn.Module):
    """Causal depthwise Conv3d -> channel LN -> pointwise GELU MLP with zero-init residual."""

    def __init__(self, dim: int, kernel: tuple[int, int, int] = (3, 3, 3), ffn_mult: int = 4) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel)
        self.dw = _CausalConv3d(dim, dim, kernel_size=kernel, padding=padding, groups=dim)
        self.norm = _ChannelLayerNorm3d(dim)
        self.pw1 = nn.Conv3d(dim, dim * ffn_mult, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(dim * ffn_mult, dim, kernel_size=1)
        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class PixelDecoder(nn.Module):
    """AdaLN -> channel LN + 1x1 mixer -> causal ConvNeXt stack -> linear unpatchify."""

    def __init__(
        self,
        dim: int = 1536,
        pixel_patch_size: tuple[int, int, int] = (4, 16, 16),
        out_dim: int = 3,
        n_low_blocks: int = 2,
        n_high_blocks: int = 1,
        ffn_mult: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.patch_size = tuple(pixel_patch_size)
        self.norm = WanLayerNorm(self.dim, eps)
        self.modulation = nn.Parameter(torch.randn(1, 2, self.dim) / self.dim**0.5)
        self.entry_norm = _ChannelLayerNorm3d(self.dim)
        self.entry_pw = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.conv_low = nn.Sequential(*[_CausalResDWBlock3d(self.dim, ffn_mult=ffn_mult) for _ in range(n_low_blocks)])
        self.conv_high = nn.Sequential(
            *[_CausalResDWBlock3d(self.dim, ffn_mult=ffn_mult) for _ in range(n_high_blocks)]
        )
        self.fc_out = nn.Linear(self.dim, math.prod(self.patch_size) * self.out_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        e_mod: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Decode ``x: [B, F*L, dim]`` with ``e_mod: [B, F, 1, dim]`` on ``grid_sizes: [B, 3]``."""
        num_frames = e_mod.shape[1]
        frame_tokens = x.shape[1] // num_frames
        shift, scale = (self.modulation.unsqueeze(1) + e_mod).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (num_frames, frame_tokens)) * (1 + scale) + shift
        f, h, w = grid_sizes[0].tolist()
        value = x.reshape(x.shape[0], f, h, w, self.dim).permute(0, 4, 1, 2, 3).contiguous()
        value = self.entry_pw(self.entry_norm(value))
        value = self.conv_low(value)
        value = self.conv_high(value)
        batch_size, channels, frames, grid_h, grid_w = value.shape
        tokens = self.fc_out(
            value.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, frames, grid_h * grid_w, channels)
        )
        grid = torch.tensor([[frames, grid_h, grid_w]] * batch_size, dtype=torch.long, device=tokens.device)
        return unpatchify_tokens(tokens, grid, self.patch_size, self.out_dim)


class BridgedPixelDecoder(nn.Module):
    """Bias-free projection into a narrower ``PixelDecoder`` (used by FastFrames-L 14B)."""

    def __init__(
        self,
        in_dim: int = 5120,
        dim: int = 1536,
        bridge_seed: int = 20260821,
        *,
        pixel_patch_size: tuple[int, int, int] = (4, 16, 16),
        out_dim: int = 3,
        n_low_blocks: int = 2,
        n_high_blocks: int = 1,
        ffn_mult: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bridge = nn.Linear(in_dim, dim, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(bridge_seed))
            nn.init.normal_(self.bridge.weight, std=in_dim**-0.5)
        self.pixel_head = PixelDecoder(
            dim,
            pixel_patch_size=pixel_patch_size,
            out_dim=out_dim,
            n_low_blocks=n_low_blocks,
            n_high_blocks=n_high_blocks,
            ffn_mult=ffn_mult,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        e_mod: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        return self.pixel_head(self.bridge(x), self.bridge(e_mod), grid_sizes)
