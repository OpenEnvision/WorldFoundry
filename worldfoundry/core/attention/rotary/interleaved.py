# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from rotary-embedding-torch (Phil Wang, MIT).
"""Adjacent-pair RoPE with partial feature slices and input-dtype restoration."""

import torch
from einops import rearrange


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs as ``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]``."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@torch.amp.autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float | torch.Tensor = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    """Apply paired angular frequencies to a feature slice, preserving the outer features."""
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], (
        f"Feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    )
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_pairs(t_middle) * freqs.sin() * scale)
    out = torch.cat([t_left, t_transformed, t_right], dim=-1)
    return out.type(dtype)
