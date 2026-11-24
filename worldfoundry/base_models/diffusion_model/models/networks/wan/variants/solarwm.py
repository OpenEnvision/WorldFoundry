# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan projections with SolarWM's activation-dtype Q/K normalization."""

import torch

from ..reference_22 import WanRMSNorm, WanSelfAttention


class SolarWanRMSNorm(WanRMSNorm):
    """Keep the RMS reduction in the incoming dtype, including BF16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x).type_as(x) * self.weight


class SolarWanAttentionProjections(WanSelfAttention):
    """Shared Wan parameter layout for Solar's causal and cached attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        if qk_norm:
            self.norm_q.__class__ = SolarWanRMSNorm
            self.norm_k.__class__ = SolarWanRMSNorm
