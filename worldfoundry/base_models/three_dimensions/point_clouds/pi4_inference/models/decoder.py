# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Geometry and motion looped transformer decoders for Pi4.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/decoder.py``). Upstream code and weights are licensed under CC BY-NC 4.0.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import Block


class TimeGates(nn.Module):
    """Per-iteration gating projection for looped decoder blocks."""

    encoding: torch.Tensor

    def __init__(self, steps: int) -> None:
        super().__init__()
        self.register_buffer("encoding", torch.empty(steps, 128))
        self.proj = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3072))

    def forward(self, step: int, batch_size: int) -> torch.Tensor:
        encoding = self.encoding[step].expand(batch_size, -1).contiguous()
        gates = self.proj(encoding) + 1
        return gates.reshape(batch_size, 3, 1024).permute(1, 0, 2).unsqueeze(2)


class GeometryDecoder(nn.Module):
    """16-step alternating frame/global decoder producing scene geometry tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.register_tokens = nn.Parameter(torch.empty(1, 1, 5, 1024))
        self.frame_block = Block(1024, 16, normalize=True, scaled=True)
        self.global_block = Block(1024, 16, normalize=True, scaled=True)
        self.readout = Block(1024, 16, normalize=True, scaled=True)
        self.frame_gates = TimeGates(16)
        self.global_gates = TimeGates(16)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial = torch.cat(
            (
                self.register_tokens.expand(batch_size, frames, -1, -1).reshape(batch_size * frames, 5, 1024),
                tokens,
            ),
            dim=1,
        )
        features = initial
        token_count = features.shape[1]
        for index in range(16):
            features = self.frame_block(
                features.reshape(batch_size * frames, token_count, 1024),
                positions,
                gates=self.frame_gates(index, batch_size * frames),
            )
            features = self.global_block(
                features.reshape(batch_size, frames * token_count, 1024),
                gates=self.global_gates(index, batch_size),
            )
        features = self.readout(features.reshape(batch_size * frames, token_count, 1024), positions)
        return features, initial


class SpatialCondition(nn.Module):
    """Geometry- and step-conditioned affine modulation generator for motion decoding."""

    time_encoding: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.geometry = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
        self.time = nn.Linear(64, 256)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(256, 4096))
        self.register_buffer("time_encoding", torch.empty(8, 64))

    def forward(self, geometry: torch.Tensor, step: int) -> tuple[torch.Tensor, ...]:
        encoding = self.time_encoding[step].expand(geometry.shape[0], -1).contiguous()
        return self.output(self.geometry(geometry.float()) + self.time(encoding)[:, None]).chunk(4, dim=-1)


class MotionDecoder(nn.Module):
    """8-step geometry-conditioned alternating frame/global motion decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_block = Block(1024, 16, normalize=True, scaled=True)
        self.global_block = Block(1024, 16, normalize=True, scaled=True)
        self.readout = Block(1024, 16, normalize=True, scaled=True)
        self.frame_condition = SpatialCondition()
        self.global_condition = SpatialCondition()
        self.frame_gates = TimeGates(8)
        self.global_gates = TimeGates(8)

    def forward(
        self,
        initial: torch.Tensor,
        geometry: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        frames: int,
    ) -> torch.Tensor:
        token_count = initial.shape[1]
        features = initial
        for index in range(8):
            frame_geometry = geometry.reshape(batch_size * frames, token_count, 1024)
            features = self.frame_block(
                features.reshape_as(frame_geometry),
                positions,
                gates=self.frame_gates(index, batch_size * frames),
                modulation=self.frame_condition(frame_geometry, index),
            )
            global_geometry = geometry.reshape(batch_size, frames * token_count, 1024)
            features = self.global_block(
                features.reshape_as(global_geometry),
                gates=self.global_gates(index, batch_size),
                modulation=self.global_condition(global_geometry, index),
            )
        return self.readout(features.reshape(batch_size * frames, token_count, 1024), positions).float()
