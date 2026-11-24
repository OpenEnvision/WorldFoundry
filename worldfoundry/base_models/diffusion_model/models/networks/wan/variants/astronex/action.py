# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, models/action.py.
"""Per-frame action conditioning through Wan's timestep modulation.

A 64-wide continuous action vector per latent frame, plus an optional embodiment
id, becomes an additive ``[B, F, 6, dim]`` term on the shift/scale/gate
modulation every block already consumes, so each latent frame receives exactly
its own command.
"""

from __future__ import annotations

import torch
from torch import nn


class ActionEmbedder(nn.Module):
    """``(B, F, action_dim)`` actions [+ embodiment id] -> ``[B, F, 6, dim]`` modulation."""

    def __init__(self, dim: int, action_dim: int = 64, num_embodiments: int = 32, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.dim = dim
        self.action_dim = action_dim
        self.proj = nn.Sequential(nn.Linear(action_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.embodiment = nn.Embedding(num_embodiments, hidden) if num_embodiments > 0 else None
        self.to_modulation = nn.Linear(hidden, dim * 6)

    def forward(self, actions: torch.Tensor, embodiment_id: torch.Tensor | None = None) -> torch.Tensor:
        """``actions``: ``[B, F, action_dim]``; ``embodiment_id``: ``[B]`` or ``[B, F]`` integers."""
        if actions.dim() != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must be [B, F, {self.action_dim}], got {tuple(actions.shape)}")
        hidden = self.proj(actions)
        if self.embodiment is not None and embodiment_id is not None:
            embodiment = self.embodiment(embodiment_id.long())
            if embodiment.dim() == 2:
                embodiment = embodiment.unsqueeze(1)
            hidden = hidden + embodiment.to(hidden.dtype)
        return self.to_modulation(hidden).unflatten(-1, (6, self.dim))


def apply_action_modulation(
    embedder: ActionEmbedder | None,
    e0: torch.Tensor,
    actions: torch.Tensor | None,
    embodiment_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add the action term to the per-frame time modulation ``e0`` (``[B, F, 6, dim]``).

    The embedder runs under CUDA autocast at the activation dtype, so a head held in a
    different parameter dtype than the backbone still serves the activation dtype.
    """
    if embedder is None or actions is None:
        return e0
    target = e0.dtype
    autocast_enabled = actions.is_cuda and target in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=target, enabled=autocast_enabled):
        delta = embedder(actions.to(target), embodiment_id)
    if e0.shape[1] != delta.shape[1]:
        raise ValueError(f"action frames {delta.shape[1]} do not match modulation frames {e0.shape[1]}")
    return e0 + delta.to(e0.dtype)
