# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo action-chunk conditioning on the canonical Cosmos Predict DiT.

Adapted from NVIDIA/DreamDojo at 02f119b759d5c7f84a399fdeea3c6e82e7ed6cff,
cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py.
"""

from __future__ import annotations

from typing import Any

import torch
from einops import repeat
from torch import nn

from .model import Cosmos25Transformer3DModel


class ActionChunkCosmosTransformer(Cosmos25Transformer3DModel):
    """Add four 384-D robot actions to each future latent frame's AdaLN."""

    def __init__(self, *, action_dim: int = 384, temporal_compression: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.temporal_compression = temporal_compression
        hidden = self.time_norm.weight.numel()
        self.action_embedder_B_D = nn.Sequential()  # Names below match the official Mlp.
        self.action_embedder_B_3D = nn.Sequential()
        for module, output in ((self.action_embedder_B_D, hidden), (self.action_embedder_B_3D, 3 * hidden)):
            module.add_module("fc1", nn.Linear(action_dim * temporal_compression, hidden * 4))
            module.add_module("act", nn.GELU(approximate="tanh"))
            module.add_module("fc2", nn.Linear(hidden * 4, output))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        fps: float | None = None,
        control_hidden_states: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Evaluate one action-conditioned velocity prediction in ``BCTHW``."""
        batch, _, frames, _, _ = hidden_states.shape
        expected = (batch, (frames - 1) * self.temporal_compression, self.action_dim)
        if action is None:
            raise ValueError("DreamDojo requires normalized robot actions")
        if tuple(action.shape) != expected:
            raise ValueError(f"DreamDojo actions must have shape {expected}, got {tuple(action.shape)}")
        tokens, rotary, grid = self._prepare_tokens(
            hidden_states,
            embedder=self.patch_embed,
            fps=None,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
        )
        context = self.text_embed(encoder_hidden_states)
        projected, temb = self.time_embed(tokens, timestep.reshape(batch, -1).flatten())
        grouped = action.reshape(batch, frames - 1, -1).to(tokens)
        action_projected = self.action_embedder_B_D(grouped)
        action_temb = self.action_embedder_B_3D(grouped)
        projected = projected.reshape(batch, -1, projected.shape[-1]) + torch.cat(
            (torch.zeros_like(action_projected[:, :1]), action_projected), dim=1
        )
        temb = temb.reshape(batch, -1, temb.shape[-1]) + torch.cat(
            (torch.zeros_like(action_temb[:, :1]), action_temb), dim=1
        )
        projected = self.time_norm(projected)
        projected, temb = (repeat(value, "b t c -> b (t h w) c", h=grid[1], w=grid[2]) for value in (projected, temb))
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None].to(dtype=torch.bool)
        for index, block in enumerate(self.transformer_blocks):
            tokens = block(tokens, context, projected, temb, rotary, attention_mask)
            if control_hidden_states is not None and index in control_hidden_states:
                tokens = tokens + control_hidden_states[index].to(tokens)
        return self._unpatchify(self.proj_out(self.norm_out(tokens, projected, temb)), grid)
