# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License.
"""Inference-only subtask boundary classifier for Mem-0."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = None if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value if self.skip is None else self.skip(value)
        hidden = self.fc1(self.norm(value))
        hidden = self.dropout(F.gelu(hidden))
        hidden = self.dropout(self.fc2(hidden))
        return hidden + residual


class SubtaskEndClassifier(nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],
        dropout: float,
        pos_weight: float | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(input_dim, output_dim, dropout)
                for input_dim, output_dim in itertools.pairwise(hidden_sizes)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_sizes[-1])
        self.head = nn.Linear(hidden_sizes[-1], 1)
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    @torch.inference_mode()
    def predict(self, fused_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if fused_hidden.ndim == 3:
            fused_hidden = fused_hidden.squeeze(1)
        hidden = fused_hidden
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.head(self.final_norm(hidden)).squeeze(-1)
        return {"logits": logits, "prob": torch.sigmoid(logits)}


__all__ = ["SubtaskEndClassifier"]
