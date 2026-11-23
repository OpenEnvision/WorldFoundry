# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Camera-to-world pose prediction head for Pi4.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/heads/camera.py``). Upstream code and weights are licensed under CC BY-NC 4.0.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..layers import Block


class CameraResidual(nn.Module):
    """Three-layer ReLU residual MLP operating on per-patch camera features."""

    def __init__(self) -> None:
        super().__init__()
        self.res_conv1 = nn.Linear(512, 512)
        self.res_conv2 = nn.Linear(512, 512)
        self.res_conv3 = nn.Linear(512, 512)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.res_conv1(features))
        residual = F.relu(self.res_conv2(residual))
        return features + F.relu(self.res_conv3(residual))


class CameraHead(nn.Module):
    """Predict SE(3) camera-to-world matrices from geometry decoder tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.projects = nn.Linear(1024, 768)
        self.blocks = nn.ModuleList(Block(768, 12) for _ in range(2))
        self.linear_out = nn.Linear(768, 512)
        self.res_conv = nn.ModuleList(CameraResidual() for _ in range(2))
        self.more_mlps = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.pose = nn.Linear(512, 12)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        features = self.projects(tokens.float())
        for block in self.blocks:
            features = block(features, positions)
        features = self.linear_out(features)[:, 5:]
        for block in self.res_conv:
            features = block(features)
        features = self.more_mlps(features.mean(dim=1))
        raw_pose = self.pose(features.float())
        rotation = raw_pose[:, 3:].reshape(-1, 3, 3)
        left, _, right = torch.svd(F.normalize(rotation, dim=-1).transpose(-2, -1))
        determinant = torch.det(right @ left.transpose(-2, -1))
        rotation = torch.cat(
            (right[:, :, :-1], right[:, :, -1:] * determinant[:, None, None]), dim=-1
        ) @ left.transpose(-2, -1)
        pose = torch.zeros(raw_pose.shape[0], 4, 4, device=tokens.device, dtype=torch.float32)
        pose[:, :3, :3] = rotation
        pose[:, :3, 3] = raw_pose[:, :3]
        pose[:, 3, 3] = 1
        return pose
