# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention


from worldfoundry.core.nn.blocks.layers import Mlp as _SharedMlp


def Mlp(width: int = 768, ratio: int = 4):
    return _SharedMlp(in_features=width, hidden_features=width * ratio, bias=False)


class BidirectionalBlock(nn.Module):
    def __init__(self, width: int = 768, heads: int = 12) -> None:
        super().__init__()
        self.norm1_x = nn.LayerNorm(width, bias=False)
        self.self_attn = Attention(width, heads)
        self.norm2_x = nn.LayerNorm(width, bias=False)
        self.cross_attn_x = Attention(width, heads)
        self.norm3_x = nn.LayerNorm(width, bias=False)
        self.mlp_x = Mlp(width)
        self.norm1_rec = nn.LayerNorm(width, bias=False)
        self.cross_attn_rec = Attention(width, heads)
        self.norm2_rec = nn.LayerNorm(width, bias=False)
        self.mlp_rec = Mlp(width)

    def forward(
        self, target: torch.Tensor, representation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = target + self.self_attn(self.norm1_x(target))
        target_norm = self.norm2_x(target)
        representation_norm = self.norm1_rec(representation)
        target = target + self.cross_attn_x(target_norm, kv=representation_norm)
        representation = representation + self.cross_attn_rec(
            representation_norm, kv=target_norm
        )
        target = target + self.mlp_x(self.norm3_x(target))
        representation = representation + self.mlp_rec(self.norm2_rec(representation))
        return target, representation


class FinalTargetBlock(nn.Module):
    def __init__(self, width: int = 768, heads: int = 12) -> None:
        super().__init__()
        self.norm1_x = nn.LayerNorm(width, bias=False)
        self.self_attn = Attention(width, heads)
        self.norm2_x = nn.LayerNorm(width, bias=False)
        self.cross_attn_x = Attention(width, heads)
        self.norm3_x = nn.LayerNorm(width, bias=False)
        self.mlp_x = Mlp(width)
        self.norm1_rec = nn.LayerNorm(width, bias=False)

    def forward(self, target: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
        target = target + self.self_attn(self.norm1_x(target))
        target_norm = self.norm2_x(target)
        representation_norm = self.norm1_rec(representation)
        target = target + self.cross_attn_x(target_norm, kv=representation_norm)
        return target + self.mlp_x(self.norm3_x(target))


__all__ = ["BidirectionalBlock", "FinalTargetBlock"]
