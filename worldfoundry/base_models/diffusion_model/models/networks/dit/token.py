# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2024 Tianhong Li
# Adapted from DiT/MAR (MIT); the default Linear initialization is retained.
"""Random-initialized AdaLN output projection for 2-D denoising tokens."""

import torch
from torch import nn

from worldfoundry.core.nn.transformer.dit import scale_shift


class TokenFinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = scale_shift(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
