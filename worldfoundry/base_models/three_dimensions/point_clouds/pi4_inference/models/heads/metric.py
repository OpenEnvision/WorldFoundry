# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Metric-scale prediction head for Pi4.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/heads/metric.py``). Upstream code and weights are licensed under CC BY-NC 4.0.
"""

from __future__ import annotations

import torch
from torch import nn

from ..layers import CrossBlock


class MetricHead(nn.Module):
    """Predict a positive sequence-level metric scale scalar per batch element."""

    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, 1024))
        self.projects_x = nn.Linear(1024, 768)
        self.projects_y = nn.Linear(1024, 768)
        self.blocks = nn.ModuleList(CrossBlock() for _ in range(2))
        self.output = nn.Linear(768, 1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        context = self.projects_y(tokens.reshape(batch_size, -1, 1024).float())
        positions = positions.reshape(batch_size, -1, 2)
        features = self.projects_x(self.token.expand(batch_size, -1, -1))
        for block in self.blocks:
            features = block(features, context, positions)
        return self.output(features).reshape(batch_size).exp()
