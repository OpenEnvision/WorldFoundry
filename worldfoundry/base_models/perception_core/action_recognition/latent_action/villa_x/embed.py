"""Embedding helpers for Villa-X."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .utils import flatten_internal


class PatchEmbed(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int = 3,
        flatten: bool = True,
    ):
        super().__init__()
        self.resolution, self.patch_size = (
            (resolution, resolution),
            (patch_size, patch_size),
        )
        self.grid_size = (
            self.resolution[0] // self.patch_size[0],
            self.resolution[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, patches: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        @flatten_internal
        def embed(x: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            if self.flatten:
                return rearrange(x, "... c h w -> ... (h w) c")
            return x

        if isinstance(patches, list):
            return [embed(patch) for patch in patches]
        return embed(patches)
