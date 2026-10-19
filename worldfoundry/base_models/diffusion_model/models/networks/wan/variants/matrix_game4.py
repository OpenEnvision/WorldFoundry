# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 4 causal Wan graph with a fixed-relative identity sink.

Adapted from SkyworkAI/Matrix-Game, revision
5446525c80da2a4a2f87801addbbc7faa8d2486e, Matrix-Game-4 (Apache-2.0).
The canonical Causal-Forcing Wan graph owns embeddings, blocks and the head;
only the bounded inference attention and extended positional table differ.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from worldfoundry.core.attention.rotary.complex_rope import (
    apply_complex_rotary_embedding,
    assemble_complex_rotary_frequencies_3d,
    complex_rotary_frequencies,
)
from worldfoundry.core.attention.sequence.varlen import attention

from .forcing.causal_forcing import CausalWanModel, CausalWanSelfAttention


def _rotate(value: Tensor, grid_sizes: Tensor, frequencies: Tensor, start: int) -> Tensor:
    """Apply the released FP64 complex RoPE at an absolute latent position."""
    half = value.shape[-1] // 2
    spatial = half // 3
    tables = frequencies.split((half - 2 * spatial, spatial, spatial), dim=1)
    frames, height, width = (int(item) for item in grid_sizes[0].tolist())
    rotations = assemble_complex_rotary_frequencies_3d(tables, (frames, height, width), start_frame=start)
    return apply_complex_rotary_embedding(value.flatten(2), rotations, value.shape[2]).reshape_as(value)


class MatrixGame4SelfAttention(CausalWanSelfAttention):
    """One-latent cached attention; rewrites never advance or evict history."""

    def forward(
        self,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        block_mask: object,
        kv_cache: dict[str, Tensor] | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> Tensor:
        del seq_lens, block_mask, cache_start
        if kv_cache is None or x.shape[0] != 1 or int(grid_sizes[0, 0]) != 1:
            raise ValueError("Matrix-Game 4 attention requires batch one, one latent and explicit KV")
        batch, tokens = x.shape[:2]
        shape = (batch, tokens, self.num_heads, self.head_dim)
        query = self.norm_q(self.q(x)).view(shape)
        key = self.norm_k(self.k(x)).view(shape)
        value = self.v(x).view(shape)
        position = current_start // tokens
        query = _rotate(query, grid_sizes, freqs, position).type_as(value)
        rotated_key = _rotate(key, grid_sizes, freqs, position).type_as(value)

        previous_end = int(kv_cache["global_end_index"].item())
        local_end = int(kv_cache["local_end_index"].item())
        current_end = current_start + tokens
        if current_end not in (previous_end, previous_end + tokens):
            raise ValueError("Matrix-Game 4 KV writes must be consecutive or rewrite the current latent")
        if current_end > previous_end:
            if local_end + tokens > kv_cache["k"].shape[1]:
                # Keep the identity prefix and evict exactly one oldest gameplay latent.
                for name in ("k", "v"):
                    kv_cache[name][:, tokens : local_end - tokens].copy_(
                        kv_cache[name][:, 2 * tokens : local_end].clone()
                    )
                local_end -= tokens
            local_end += tokens
        local_start = local_end - tokens
        kv_cache["k"][:, local_start:local_end].copy_(rotated_key)
        kv_cache["v"][:, local_start:local_end].copy_(value)
        if local_start == 0:
            kv_cache["k_sink_raw"].copy_(key)
        keys = kv_cache["k"][:, :local_end]
        values = kv_cache["v"][:, :local_end]
        if position > self.local_attn_size:
            sink = _rotate(kv_cache["k_sink_raw"], grid_sizes, freqs, position - self.local_attn_size).type_as(value)
            keys = torch.cat((sink, keys[:, tokens:]), dim=1)
        result = attention(query, keys, values)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end)
        return self.o(result.flatten(2))


class MatrixGame4Model(CausalWanModel):
    """Released 148-input / 48-output graph with checkpoint-identical names."""

    def __init__(self, *, max_rope_frames: int = 8193, **kwargs: Any) -> None:
        # TI2V has the same text-only cross attention as canonical causal T2V;
        # low-poly conditioning is concatenated by the denoiser adapter.
        super().__init__(model_type="t2v", **kwargs)
        self.max_rope_frames = int(max_rope_frames)
        for block in self.blocks:
            block.self_attn = MatrixGame4SelfAttention(
                self.dim,
                self.num_heads,
                self.local_attn_size,
                1,
                self.qk_norm,
                self.eps,
            )
        head_dim = self.dim // self.num_heads
        spatial_dim = 2 * (head_dim // 6)
        self.freqs = torch.cat(
            tuple(
                complex_rotary_frequencies(dim, self.max_rope_frames)
                for dim in (head_dim - 2 * spatial_dim, spatial_dim, spatial_dim)
            ),
            dim=1,
        )
