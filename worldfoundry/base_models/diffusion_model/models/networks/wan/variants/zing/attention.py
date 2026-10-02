# Adapted from seedleap/zing-world-model (Apache-2.0).
"""Zing cache-aware attention, using the canonical Wan projections and core dispatch."""

from __future__ import annotations

import inspect

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_22 import WanSelfAttention, rope_params
from worldfoundry.core.attention.sequence.varlen import varlen_scaled_dot_product_attention


class CompiledSegment:
    compiled = {}
    mode = None

    @classmethod
    def get(cls, function, enabled: bool):
        if not enabled:
            return function
        if function not in cls.compiled:
            options = {}
            if "recompile_limit" in inspect.signature(torch.compile).parameters:
                options["recompile_limit"] = 1024
            cls.compiled[function] = torch.compile(function, dynamic=True, mode=cls.mode, **options)
        return cls.compiled[function]


def make_rope_freqs(dim, num_heads, maximum):
    head_dim = dim // num_heads
    widths = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
    return tuple(torch.view_as_real(rope_params(maximum, width)).float() for width in widths)


def compute_rope(
    positions: torch.Tensor, temporal: torch.Tensor, height: torch.Tensor, width: torch.Tensor
) -> torch.Tensor:
    maxima = positions.max(dim=0).values
    if int(maxima[0]) >= temporal.shape[0] or int(maxima[1]) >= height.shape[0] or int(maxima[2]) >= width.shape[0]:
        raise ValueError("RoPE position exceeds generator.rope_max_seq_len")
    return torch.cat((temporal[positions[:, 0]], height[positions[:, 1]], width[positions[:, 2]]), dim=1)


def apply_rope(value: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sequence = value.shape[-3]
    head_dim = value.shape[-1]
    shaped = rope.reshape(*([1] * (value.dim() - 3)), sequence, 1, head_dim // 2, 2)
    cosine, sine = shaped[..., 0], shaped[..., 1]
    real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
    rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1)
    return rotated.flatten(-2).to(value.dtype)


def flash_attention_varlen(query, key, value, query_lengths, key_lengths, deterministic):
    zero = query_lengths.new_zeros(1)
    return varlen_scaled_dot_product_attention(
        query,
        key,
        value,
        cu_seqlens_q=torch.cat((zero, query_lengths.cumsum(0))).to(torch.int32),
        cu_seqlens_k=torch.cat((zero, key_lengths.cumsum(0))).to(torch.int32),
        max_seqlen_q=int(query_lengths.max()),
        max_seqlen_k=int(key_lengths.max()),
        deterministic=deterministic,
    )


class SelfAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, eps, qk_norm, compile_fusion, deterministic):
        super().__init__(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.deterministic = deterministic

    def forward(
        self,
        hidden: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        history: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        query = self.q(hidden)
        key = self.k(hidden)
        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query.reshape(batch, sequence, self.num_heads, self.head_dim)
        key = key.reshape(batch, sequence, self.num_heads, self.head_dim)
        value = self.v(hidden).reshape(batch, sequence, self.num_heads, self.head_dim)
        history_key, history_value = history
        raw_key = key if history_key is None else torch.cat((history_key, key), dim=1)
        full_value = value if history_value is None else torch.cat((history_value, value), dim=1)
        query = apply_rope(query, query_rope)
        rotated_key = apply_rope(raw_key, key_rope)
        key_sequence = rotated_key.shape[1]
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        key_lengths = torch.full((batch,), key_sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query.reshape(batch * sequence, self.num_heads, self.head_dim),
            rotated_key.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            full_value.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            query_lengths,
            key_lengths,
            self.deterministic,
        )
        attended = attended.reshape(batch, sequence, self.num_heads * self.head_dim)
        return self.o(attended), key, value


class CrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, eps, qk_norm, compile_fusion, deterministic):
        super().__init__(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.deterministic = deterministic

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cached: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, sequence, _ = hidden.shape
        query = self.norm_q(self.q(hidden)).reshape(batch * sequence, self.num_heads, self.head_dim)
        key, value = cached
        created = None
        if key is None:
            key = self.norm_k(self.k(context)).reshape(context.shape[0], self.num_heads, self.head_dim)
            value = self.v(context).reshape(context.shape[0], self.num_heads, self.head_dim)
            created = (key, value)
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query, key, value, query_lengths, context_lengths.to(torch.int32), self.deterministic
        )
        return self.o(attended.reshape(batch, sequence, -1)), created
