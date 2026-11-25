# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qwen and Hugging Face Transformers rotary embedding helpers (Apache-2.0).
"""Half-split query/key rotation with caller-selected position tables."""

import torch

from worldfoundry.core.attention.rotary.rope import rotate_half
from worldfoundry.core.attention.rotary.triton_rotary_ops import triton_apply_rotary_pos_emb


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K after inserting the head axis into the selected cosine/sine tables."""
    if q.device.type == "cuda":
        triton_out = triton_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        if triton_out is not None:
            return triton_out
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half_chunked(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two Torch chunks, retaining chunk semantics for odd feature widths."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
