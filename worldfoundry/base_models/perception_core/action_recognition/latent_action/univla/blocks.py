"""Transformer and vector-quantization blocks required by UniVLA."""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        divisor = torch.exp(exponent)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pos_enc", encoding, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.pos_enc[: values.shape[2]]


class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.scale = (model_dim // num_heads) ** -0.5
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, *, is_causal: bool = False) -> Tensor:
        query, key, value = (
            rearrange(
                projection(values),
                "b n (h d) -> b h n d",
                h=self.heads,
            )
            for projection in (self.to_q, self.to_k, self.to_v)
        )
        weights = query @ key.transpose(-2, -1) * self.scale
        if is_causal:
            length = weights.shape[-1]
            mask = torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=weights.device,
            ).triu(diagonal=1)
            weights = weights.masked_fill(mask, float("-inf"))
        output = weights.softmax(dim=-1) @ value
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, values: Tensor, causal_temporal: bool = False) -> Tensor:
        time_length, spatial_length = values.shape[1:3]

        values = rearrange(values, "b t s e -> (b t) s e")
        values = values + self.spatial_attn(self.norm1(values))
        values = rearrange(values, "(b t) s e -> b t s e", t=time_length)

        values = rearrange(values, "b t s e -> (b s) t e")
        values = values + self.temporal_attn(self.norm2(values), is_causal=causal_temporal)
        values = rearrange(values, "(b s) t e -> b t s e", s=spatial_length)
        return values + self.ffn(self.norm3(values))


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        to_out: bool = True,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(model_dim, out_dim) if to_out else nn.Identity()
        self.causal_temporal = causal_temporal

    def forward(self, values: Tensor) -> Tensor:
        values = self.pos_enc(self.ffn(values))
        for block in self.transformer_blocks:
            values = block(values, self.causal_temporal)
        return self.out(values)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, **_: object) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        indices = torch.cdist(values, self.codebook.weight).argmin(dim=-1)
        quantized = self.codebook(indices)
        return quantized, indices
