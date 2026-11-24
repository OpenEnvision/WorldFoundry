# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from TencentARC/Open-MAGVIT2 (Apache-2.0).
"""MAGVIT2 convolution and DCR depth-to-space upsampling."""

import torch
from torch import nn


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError("Expecting a channels-first (*CHW) tensor of at least 3 dimensions")
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels")

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class Upsampler(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None) -> None:
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out
