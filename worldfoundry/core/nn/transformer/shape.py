"""Transformer shape helpers without model-specific state.

Head split/merge, causal masks, sinusoidal 1D tables. Used by DiT and
ViT so each model does not reimplement ``hidden % heads``.

Not this module:
    Block shells live in :mod:`worldfoundry.core.nn.transformer.vit`.
    Attention kernels live in :mod:`worldfoundry.core.attention`.
    DDPM-style timestep tables live in :mod:`worldfoundry.core.nn.diffusion.timestep`.

Public surface:

- :class:`TransformerShapeSpec` / :func:`transformer_shape_spec`
- :func:`split_attention_heads` / :func:`merge_attention_heads`
- :func:`causal_attention_mask` / :func:`sinusoidal_embedding_1d`
- Re-exports of stochastic-depth helpers for older import paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# Shape spec — hidden / heads / MLP width with divisibility checks
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransformerShapeSpec:
    """Common transformer dimensions derived from hidden size and head count."""

    hidden_size: int
    num_heads: int
    head_dim: int
    mlp_hidden_size: int


def attention_head_dim(hidden_size: int, num_heads: int) -> int:
    """Return per-head attention width and validate divisibility."""

    hidden = int(hidden_size)
    heads = int(num_heads)
    if hidden <= 0:
        raise ValueError("hidden_size must be positive.")
    if heads <= 0:
        raise ValueError("num_heads must be positive.")
    if hidden % heads:
        raise ValueError(f"hidden_size {hidden} is not divisible by num_heads {heads}.")
    return hidden // heads


def transformer_shape_spec(
    hidden_size: int,
    num_heads: int,
    *,
    mlp_ratio: float = 4.0,
    multiple_of: int | None = None,
) -> TransformerShapeSpec:
    """Build a reusable shape spec for attention and MLP dimensions."""

    return TransformerShapeSpec(
        hidden_size=int(hidden_size),
        num_heads=int(num_heads),
        head_dim=attention_head_dim(hidden_size, num_heads),
        mlp_hidden_size=mlp_hidden_size(hidden_size, multiplier=mlp_ratio, multiple_of=multiple_of),
    )


def mlp_hidden_size(hidden_size: int, *, multiplier: float = 4.0, multiple_of: int | None = None) -> int:
    """Compute a feed-forward hidden width with optional upward rounding."""

    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be positive.")
    if float(multiplier) <= 0:
        raise ValueError("multiplier must be positive.")
    width = int(round(int(hidden_size) * float(multiplier)))
    if multiple_of is None:
        return max(1, width)
    multiple = int(multiple_of)
    if multiple <= 0:
        raise ValueError("multiple_of must be positive.")
    return ((max(1, width) + multiple - 1) // multiple) * multiple


# ──────────────────────────────────────────────────────────────────────────
# Head split / merge — last-dim hidden ↔ (heads, head_dim)
# ──────────────────────────────────────────────────────────────────────────


def split_attention_heads(value: Any, num_heads: int) -> Any:
    """Reshape ``(..., seq, hidden)`` to ``(..., heads, seq, head_dim)``."""

    shape = tuple(int(item) for item in value.shape)
    if len(shape) < 2:
        raise ValueError("attention input must have at least sequence and hidden dimensions.")
    heads = int(num_heads)
    head_dim = attention_head_dim(shape[-1], heads)
    reshaped = value.reshape(*shape[:-1], heads, head_dim)
    return _swapaxes(reshaped, -3, -2)


def merge_attention_heads(value: Any) -> Any:
    """Invert ``split_attention_heads`` for ``(..., heads, seq, head_dim)`` tensors."""

    shape = tuple(int(item) for item in value.shape)
    if len(shape) < 3:
        raise ValueError("attention-head input must have at least heads, sequence, and head_dim dimensions.")
    merged = _swapaxes(value, -3, -2)
    return merged.reshape(*shape[:-3], shape[-2], shape[-3] * shape[-1])


# ──────────────────────────────────────────────────────────────────────────
# Causal mask and 1D sinusoid — DiT t-embed and decoder attention
# ──────────────────────────────────────────────────────────────────────────


def causal_attention_mask(query_len: int, key_len: int | None = None, *, include_self: bool = True) -> np.ndarray:
    """Return a boolean causal mask for query/key lengths.

    When ``key_len`` is larger than ``query_len``, the mask assumes the query is
    aligned to the tail of a key/value cache.
    """

    q_len = int(query_len)
    k_len = q_len if key_len is None else int(key_len)
    if q_len <= 0:
        raise ValueError("query_len must be positive.")
    if k_len <= 0:
        raise ValueError("key_len must be positive.")
    diagonal = k_len - q_len if include_self else k_len - q_len - 1
    return np.tri(q_len, k_len, k=diagonal, dtype=bool)


def sinusoidal_embedding_1d(
    dim: int,
    position: torch.Tensor,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Return cosine/sine timestep embeddings used by diffusion transformers.

    ``max_period`` is explicit because action-flow policies often use a much
    shorter period than image/video diffusion transformers.
    """

    if int(dim) % 2:
        raise ValueError("dim must be even for sinusoidal embeddings.")
    half = int(dim) // 2
    # Scheduler timesteps are commonly integer tensors.  Casting the
    # embedding back to that dtype truncates almost every sine/cosine value to
    # zero, which silently destroys diffusion conditioning.  Preserve an
    # explicitly floating input dtype, but keep the reference float64 result
    # for integer positions so callers can downcast after the embedding.
    output_dtype = position.dtype if position.is_floating_point() else torch.float64
    position = position.to(torch.float64)
    if float(max_period) <= 0:
        raise ValueError("max_period must be positive")
    frequencies = torch.pow(
        float(max_period),
        -torch.arange(half, device=position.device, dtype=torch.float64).div(half),
    )
    sinusoid = torch.outer(position, frequencies)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(output_dtype)


def _swapaxes(value: Any, axis1: int, axis2: int) -> Any:
    """Swap two axes on numpy or torch without importing either in the helper.

    Raises:
        TypeError: the object exposes neither ``swapaxes`` nor ``transpose``.
    """

    if hasattr(value, "swapaxes"):
        return value.swapaxes(axis1, axis2)
    if hasattr(value, "transpose"):
        return value.transpose(axis1, axis2)
    raise TypeError(f"object of type {type(value).__name__} does not support axis swapping.")


from worldfoundry.core.nn.blocks.stochastic_depth import (  # noqa: E402
    add_residual,
    drop_add_residual_stochastic_depth,
    get_branges_scales,
)

__all__ = [
    "TransformerShapeSpec",
    "add_residual",
    "attention_head_dim",
    "causal_attention_mask",
    "drop_add_residual_stochastic_depth",
    "get_branges_scales",
    "merge_attention_heads",
    "mlp_hidden_size",
    "sinusoidal_embedding_1d",
    "split_attention_heads",
    "transformer_shape_spec",
]
