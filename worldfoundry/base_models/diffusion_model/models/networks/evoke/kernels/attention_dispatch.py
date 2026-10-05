import os

import torch

from worldfoundry.core.attention import varlen_scaled_dot_product_attention


def create_navit_attention_masks(
    batch_size: int,
    original_context_length_list: list,
    history_context_length: int,
    encoder_hidden_states_seq_len: int,
    device: torch.device,
    restrict_self_attn: bool = False,
    guidance_cross_attn: bool = False,
    warp_len_list: list = None,
):
    # Per-stage synchronized warp tokens; layout per stage = [shared_history | warp_s | noise_s].
    # warp_len_list is in the SAME order as original_context_length_list (caller passes both reversed).
    # None/all-zero => legacy fixed_mem (every "+ warp" reduces to +0, mask bit-identical).
    _wl = warp_len_list if warp_len_list is not None else [0] * len(original_context_length_list)
    assert len(_wl) == len(original_context_length_list), (
        f"warp_len_list len {len(_wl)} != original_context_length_list len {len(original_context_length_list)}"
    )
    # Self-attn KV span per stage = noise + shared_history + warp_s.
    _self_kv = [length + history_context_length + w for length, w in zip(original_context_length_list, _wl)]

    # Build navit_hidden_attention_mask.
    if restrict_self_attn:
        cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
        max_seqlen_q = max(original_context_length_list)

        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + kvlen)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(_self_kv)
    else:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + kvlen)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(_self_kv)
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
    navit_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # Build navit_history_hidden_attention_mask.
    navit_history_hidden_attention_mask = None
    if restrict_self_attn:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
        navit_history_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # Build navit_encoder_attention_mask.
    if guidance_cross_attn:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_max_seqlen_q = max(original_context_length_list)
    else:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + kvlen)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_cu_seqlens_q[0] = 0
        cross_max_seqlen_q = max(_self_kv)

    cu_seqlens_kv = [0]
    for _ in range(batch_size):
        for length in original_context_length_list:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + encoder_hidden_states_seq_len)
    cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
    max_seqlen_kv = encoder_hidden_states_seq_len
    navit_encoder_attention_mask = cross_cu_seqlens_q, cu_seqlens_kv, cross_max_seqlen_q, max_seqlen_kv

    return navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask


def _attention_version(device: torch.device) -> int | None:
    """Select an optional FlashAttention ABI; shared core owns all fallbacks."""
    if device.type != "cuda":
        return None
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return None
    if os.environ.get("EVOKE_FORCE_FA2", "0") == "1":
        return 2
    return 3 if major >= 9 else 2


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is None:
        batch, q_length = q.shape[:2]
        k_length = k.shape[1]
        cu_seqlens_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * q_length
        cu_seqlens_kv = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * k_length
        max_seqlen_q, max_seqlen_kv = q_length, k_length
    else:
        batch, q_length = q.shape[:2]
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask

    output = varlen_scaled_dot_product_attention(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_kv),
        version=_attention_version(q.device),
    )
    return output.unflatten(0, (batch, q_length))
