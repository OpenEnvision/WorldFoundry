"""Compatibility helpers for legacy and current diffusers Wan RoPE layouts."""

from __future__ import annotations

from typing import TypeAlias

import torch

RotaryEmbedding: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def split_rotary_embedding(
    rotary_emb: RotaryEmbedding,
    *,
    chunks: int,
    rank: int,
) -> RotaryEmbedding:
    """Split a Wan rotary embedding along its sequence dimension for SP.

    Older diffusers releases returned one complex tensor with layout
    ``[batch, heads, sequence, dim]``. Current releases return real
    ``(cos, sin)`` tensors with layout ``[batch, sequence, heads, dim]``.
    """

    if isinstance(rotary_emb, tuple):
        return tuple(torch.chunk(part, chunks, dim=1)[rank] for part in rotary_emb)
    return torch.chunk(rotary_emb, chunks, dim=2)[rank]


def apply_rotary_embedding_bhld(
    hidden_states: torch.Tensor,
    rotary_emb: RotaryEmbedding,
) -> torch.Tensor:
    """Apply either Wan RoPE representation to a ``[B, H, L, D]`` tensor."""

    if isinstance(rotary_emb, tuple):
        freqs_cos, freqs_sin = rotary_emb
        sequence_first = hidden_states.transpose(1, 2)
        x1, x2 = sequence_first.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        rotated = torch.empty_like(sequence_first)
        rotated[..., 0::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated.transpose(1, 2).type_as(hidden_states)

    complex_states = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
    rotated = torch.view_as_real(complex_states * rotary_emb).flatten(3, 4)
    return rotated.type_as(hidden_states)


__all__ = ["RotaryEmbedding", "apply_rotary_embedding_bhld", "split_rotary_embedding"]
