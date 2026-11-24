# SPDX-FileCopyrightText: 2026 OpenEnvision
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
# Shared from the SAM and SAM2 source distributions.

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Clone a layer without sharing parameters between the resulting modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000) -> torch.Tensor:
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def concat_points(
    old_point_inputs: dict[str, torch.Tensor] | None, new_points: torch.Tensor, new_labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
