"""Sequence-parallel attention for packed image+video tokens.

Wan-style dual streams (image query length vs video KV) need a 4D all-to-all
that keeps ``cu_seqlens`` consistent across ranks. This module gathers or
reshards those packed sequences, then prefers varlen FlashAttention when
explicitly available and falls back to in-tree SDPA.

Not this module:
    RoPE frequency padding lives in :mod:`.sequence_parallel_rope`.
    Ulysses-only (single stream) lives in :mod:`.ulysses_attention`.
    SageAttention is used only when the caller sets ``use_sage`` *and*
    the probe says it is usable — never as an implicit ``auto`` pick.

Public surface:

- :func:`parallel_attention` — dual-stream SP attention returning
  ``(hidden, None)`` so Hunyuan-style callers can ignore the second
  slot.
"""

import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention
from worldfoundry.core.attention.backends.probe import probe_attention_backends
from worldfoundry.core.distributed.sequence_parallel.runtime import (
    all_gather,
    all_to_all_4D,
    all_to_all_4d_many,
    get_sequence_parallel_state,
    nccl_info,
)


# ──────────────────────────────────────────────────────────────────────────
# Dual-stream SP — image tokens exchange; encoder heads are rank-narrowed
# ──────────────────────────────────────────────────────────────────────────


def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    use_sage,
):
    """Attend packed image+encoder streams under an optional SP mesh.

    Image Q/K/V are all-to-all'd (scatter heads, gather sequence).
    Encoder states are already replicated, so only the local head
    slice is kept. The SDPA fallback loops per sample because a dense
    launch would attend into padding that ``cu_seqlens`` had excluded.
    The trailing ``None`` matches Hunyuan call sites that unpack a
    pair.
    """

    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    if get_sequence_parallel_state():
        query, key, value = all_to_all_4d_many(
            (query, key, value),
            scatter_dim=2,
            gather_dim=1,
        )

        def shrink_head(encoder_state, dim):
            """Keep this rank's head slice of a fully-replicated encoder tensor."""

            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    batch_size = query.shape[0]
    head = query.shape[-2]
    head_dim = query.shape[-1]

    capabilities = probe_attention_backends(query.device)
    use_sage = bool(use_sage and capabilities["sage_attention"].usable)
    if use_sage:
        try:
            from sageattention import sageattn
        except ImportError:
            use_sage = False
        else:
            hidden_states = sageattn(query, key, value, tensor_layout="NHD")
    use_flash = flash_attn_varlen_func is not None and capabilities["flash_attention_2"].usable
    if not use_sage and not use_flash:
        outputs = []
        for batch_idx in range(batch_size):
            q_start = int(cu_seqlens_q[2 * batch_idx].item())
            q_end = int(cu_seqlens_q[2 * batch_idx + 1].item())
            kv_start = int(cu_seqlens_kv[2 * batch_idx].item())
            kv_end = int(cu_seqlens_kv[2 * batch_idx + 1].item())
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            item = _worldfoundry_scaled_dot_product_attention(
                query[batch_idx, :q_len].transpose(0, 1).unsqueeze(0),
                key[batch_idx, :kv_len].transpose(0, 1).unsqueeze(0),
                value[batch_idx, :kv_len].transpose(0, 1).unsqueeze(0),
            )
            item = item.squeeze(0).transpose(0, 1)
            if q_len < max_seqlen_q:
                padding = item.new_zeros(max_seqlen_q - q_len, head, head_dim)
                item = torch.cat([item, padding], dim=0)
            outputs.append(item)
        hidden_states = torch.stack(outputs, dim=0)
    elif not use_sage:
        query, key, value = [x.view(x.shape[0] * x.shape[1], *x.shape[2:]) for x in [query, key, value]]
        hidden_states = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    hidden_states = hidden_states.view(batch_size, max_seqlen_q, head, head_dim).contiguous()
    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
        (sequence_length, encoder_sequence_length),
        dim=1,
    )

    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()

    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    batch_size, seq_len, _, _ = attn.shape
    return attn.reshape(batch_size, seq_len, -1), None
