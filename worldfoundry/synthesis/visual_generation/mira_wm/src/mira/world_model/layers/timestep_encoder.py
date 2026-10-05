"""Sinusoidal diffusion-time (``tau``) embedding.

Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings, matching Denoising Diffusion Probabilistic Models.

    Args:
        timesteps: A 1-D tensor of N indices, one per batch element. These may be fractional.
        embedding_dim: The dimension of the output.
        flip_sin_to_cos: Whether the embedding order should be ``cos, sin`` (if True) or ``sin, cos``
            (if False).
        downscale_freq_shift: Controls the delta between frequencies between dimensions.
        scale: Scaling factor applied to the embeddings.
        max_period: Controls the maximum frequency of the embeddings.

    Returns:
        An ``[N x embedding_dim]`` tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class DiffusionTimeEmbedding(nn.Module):
    """Embed the per-frame diffusion time ``tau`` into the model's hidden dimension."""

    def __init__(self, dim: int, n_freq: int = 128):
        super().__init__()
        self.n_freq = n_freq

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.n_freq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        batch_size, n_frames = timesteps.shape[:2]
        timesteps = 1000 * rearrange(timesteps, "b t 1 1 1 -> (b t)")
        t_emb = get_timestep_embedding(timesteps, embedding_dim=2 * self.n_freq)
        dtype = self.mlp[0].weight.dtype
        t_emb = self.mlp(t_emb.to(dtype))  # type: ignore
        t_emb = rearrange(t_emb, "(b t) c -> b t 1 1 c", b=batch_size, t=n_frames)
        return t_emb
