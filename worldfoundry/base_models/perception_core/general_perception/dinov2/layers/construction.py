# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""DINO construction leaves shared by dense vision backbones."""

from torch import Tensor, nn
from torch.nn.init import trunc_normal_


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BlockChunk(nn.ModuleList):
    def forward(self, x: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        for b in self:
            x = b(x)
        return x


def build_dino_head_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int | None = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Module:
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
