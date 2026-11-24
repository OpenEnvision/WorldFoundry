# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Adapted from microsoft/CogACT and facebookresearch/DiT; retain their MIT terms.
"""Shared CogACT/DB-CogACT inference leaves with the released action-head key layout."""

from typing import Any

import torch
from torch import nn

from worldfoundry.core.nn.blocks.layers import Mlp
from worldfoundry.core.nn.transformer.vit_qkv import QKVSelfAttention as Attention


class ActionEmbedder(nn.Module):
    def __init__(self, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class LabelEmbedder(nn.Module):
    """Project conditioning tokens and retain the learned CFG null condition."""

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
        conditions_shape: tuple[int, int, int] = (1, 1, 4096),
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.linear(conditions)


class DiTBlock(nn.Module):
    """
    A DiT block with self-attention conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu() -> nn.GELU:
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        x = self.linear(x)
        return x
