"""Dense, packed, and hybrid-provider attention adapters.

Some blocks flatten heads into ``hidden``; others already have ``[B,S,H,D]``.
These adapters reshape once and call :func:`attention_forward` so a model
does not fork its own backend list. Hybrid-provider variants exist for
checkpoints that mix exact SDPA with an explicit sparse kernel on one
stream only.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from worldfoundry.core.attention.backends.probe import resolve_attention_backend
from worldfoundry.core.attention.backends.native import scaled_dot_product_attention
from worldfoundry.core.attention.sequence.varlen import flash_attention, varlen_scaled_dot_product_attention

# ──────────────────────────────────────────────────────────────────────────
# Flattened adapters — reshape once, then reuse shared dispatch / SDPA
# ──────────────────────────────────────────────────────────────────────────


def flattened_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: str = "flash",
    dropout_p: float = 0.0,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Run attention on ``[B, S, H, D]`` tensors and flatten the heads.

    Cumulative lengths may describe ordinary packed batches or alternating
    valid/padding segments produced by :func:`get_cu_seqlens`. Backend
    resolution and exact PyTorch fallbacks are framework-owned.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, sequence, heads, head_dim]")
    if key.shape[0] != query.shape[0] or value.shape[:2] != key.shape[:2]:
        raise ValueError("query, key, and value must have matching batch and key/value dimensions")

    if backend.strip().lower().replace("-", "_") == "vanilla":
        output = _vanilla_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=causal,
        )
    else:
        resolved = resolve_attention_backend(backend, query.device)
        version = 3 if resolved == "flash_attention_3" else 2 if resolved == "flash_attention_2" else None
        if cu_seqlens_q is not None or cu_seqlens_k is not None:
            if cu_seqlens_q is None or cu_seqlens_k is None:
                raise ValueError("cu_seqlens_q and cu_seqlens_k must be supplied together")
            if attn_mask is not None:
                raise ValueError("packed attention does not accept a separate attention mask")
            flat_query = query.reshape(-1, *query.shape[2:])
            flat_key = key.reshape(-1, *key.shape[2:])
            flat_value = value.reshape(-1, *value.shape[2:])
            output = varlen_scaled_dot_product_attention(
                flat_query,
                flat_key,
                flat_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                causal=causal,
                version=version,
            ).reshape(query.shape[0], query.shape[1], query.shape[2], query.shape[3])
        elif version is not None and attn_mask is None:
            output = flash_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                causal=causal,
                version=version,
            )
        else:
            output = scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
            ).transpose(1, 2)
    return output.reshape(output.shape[0], output.shape[1], -1)


# ──────────────────────────────────────────────────────────────────────────
# Hybrid provider — sequence-parallel image stream + shared text attention
# ──────────────────────────────────────────────────────────────────────────


def hybrid_provider_attention(
    provider: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    image_query_length: int,
    image_key_length: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Join a sequence-parallel image provider with shared text attention."""

    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be supplied together")
    if cu_seqlens_q is None:
        image_output = provider(
            None,
            query[:, :image_query_length],
            key[:, :image_key_length],
            value[:, :image_key_length],
            dropout_p=0.0,
            causal=False,
        )
        text_output = query.new_empty((query.shape[0], 0, query.shape[2], query.shape[3]))
    else:
        query_split = int(cu_seqlens_q[1].item())
        key_split = int(cu_seqlens_k[1].item())
        image_output = provider(
            None,
            query[:, :image_query_length],
            key[:, :image_key_length],
            value[:, :image_key_length],
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=query[:, image_query_length:query_split],
            joint_tensor_key=key[:, image_key_length:key_split],
            joint_tensor_value=value[:, image_key_length:key_split],
            joint_strategy="rear",
        )
        text_output = flash_attention(
            query[:, query_split:],
            key[:, key_split:],
            value[:, key_split:],
            version=2,
        )
    output = torch.cat((image_output, text_output), dim=1)
    return output.reshape(output.shape[0], output.shape[1], -1)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: str = "flash",
    drop_rate: float = 0.0,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_kv: int | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Compatibility spelling for checkpoint architectures using Q/KV names."""

    del batch_size
    return flattened_attention(
        q,
        k,
        v,
        backend=mode,
        dropout_p=drop_rate,
        attn_mask=attn_mask,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
    )


def parallel_attention(
    hybrid_seq_parallel_attn: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    img_q_len: int,
    img_kv_len: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Compatibility spelling for hybrid sequence-parallel model blocks."""

    return hybrid_provider_attention(
        hybrid_seq_parallel_attn,
        q,
        k,
        v,
        image_query_length=img_q_len,
        image_key_length=img_kv_len,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
    )


# ──────────────────────────────────────────────────────────────────────────
# Vanilla scores — explicit O(S²) path for checkpoints that request it
# ──────────────────────────────────────────────────────────────────────────


def _vanilla_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    causal: bool,
) -> torch.Tensor:
    """Materialize scores in PyTorch when a caller opts out of fused kernels.

    Causal and an explicit mask cannot be combined: the two contracts
    disagree on how the lower triangle interacts with padding, and
    silently OR-ing them would hide a caller bug.
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    bias = query.new_zeros(query.shape[0], query.shape[1], query.shape[2], key.shape[2])
    if causal:
        if attn_mask is not None:
            raise ValueError("causal and explicit attention masks cannot be combined in vanilla attention")
        keep = torch.ones_like(bias, dtype=torch.bool).tril()
        bias.masked_fill_(~keep, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            bias.add_(attn_mask)
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    probabilities = torch.softmax(scores + bias, dim=-1)
    probabilities = torch.dropout(probabilities, p=dropout_p, train=dropout_p > 0)
    return (probabilities @ value).transpose(1, 2)


__all__ = [
    "attention",
    "flattened_attention",
    "hybrid_provider_attention",
    "parallel_attention",
]
