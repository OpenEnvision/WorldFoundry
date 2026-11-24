"""Transformer components required by the Villa-X encoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


def _sine_cosine(dim: int, positions: np.ndarray) -> np.ndarray:
    frequencies = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    frequencies = 1.0 / (10000**frequencies)
    values = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate((np.sin(values), np.cos(values)), axis=1)


def get_1D_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return _sine_cosine(embed_dim, np.arange(length))


def get_2D_position_embeddings(embed_dim: int, grid_size: int) -> np.ndarray:
    height, width = (
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
    )
    grid = np.stack(np.meshgrid(width, height), axis=0).reshape(2, 1, grid_size, grid_size)
    return np.concatenate(
        (
            _sine_cosine(embed_dim // 2, grid[0]),
            _sine_cosine(embed_dim // 2, grid[1]),
        ),
        axis=1,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(values, dim=-1, keepdim=True) * self.scale
        return values / norm.clamp(min=self.eps) * self.g


class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(values).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class FlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm
        if qk_norm:
            head_dim = embed_dim // n_heads
            self.q_layernorm = nn.LayerNorm(head_dim)
            self.k_layernorm = nn.LayerNorm(head_dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        batch_size, query_count, embed_dim = q_in.shape
        key_count = kv_in.shape[1]

        key_value = (
            self.kv(kv_in)
            .reshape(
                batch_size,
                key_count,
                2,
                self.n_heads,
                embed_dim // self.n_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query = (
            self.q(q_in)
            .reshape(
                batch_size,
                query_count,
                self.n_heads,
                embed_dim // self.n_heads,
            )
            .transpose(1, 2)
        )
        key, value = key_value.unbind(0)
        if self.qk_norm:
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)

        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch_size, query_count, embed_dim)
        return self.dropout(self.proj(output))


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        output_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim
        self.pre_projection = nn.Linear(embed_dim, embed_dim)
        self.latents = nn.Parameter(torch.zeros(n_latents, embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        norm = RMSNorm if do_rms_norm else nn.LayerNorm
        self.attn_norm = norm(embed_dim)
        self.attn = FlashAttention(
            embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.mlp_norm = norm(embed_dim)

        hidden_dim = int(mlp_ratio * embed_dim)
        hidden_layer: nn.Module
        if do_swish_glu:
            hidden_layer = SwishGLU(embed_dim, hidden_dim)
        else:
            hidden_layer = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU())
        self.mlp = nn.Sequential(
            hidden_layer,
            nn.Linear(hidden_dim, embed_dim),
        )
        self.final_proj = nn.Sequential(nn.Linear(embed_dim, output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        latents = repeat(
            self.latents,
            "n_latents d -> b n_latents d",
            b=values.shape[0],
        )
        latents = self.attn_norm(
            latents
            + self.attn(
                q_in=latents,
                kv_in=self.pre_projection(values),
            )
        )
        latents = self.mlp_norm(latents + self.mlp(latents))
        return self.final_proj(latents.squeeze(dim=1))
