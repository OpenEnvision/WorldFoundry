# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The lingbot-world Authors. Portions derived from:
# https://github.com/Robbyant/lingbot-world/blob/main/wan/distributed/ulysses.py
#
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt at repo root).
#
# Modifications Copyright (c) 2026 SkyworkAI and contributors.
"""Ulysses attention that pads local sequences before the all-to-all.

Standard Ulysses assumes ``L % p == 0``. Causal caches and memory tokens
often violate that. Pad, communicate, then crop so the local kernel still
sees a dense shard. Prefer the unpadded path in
:mod:`worldfoundry.core.attention.parallel.ulysses_attention` when lengths already
divide evenly (less traffic).

Not this module:
    Causal LingBot graphs belong in
    :mod:`.causal_ulysses_attention`. RoPE frequency sharding belongs
    in :mod:`.sequence_parallel_rope`. This file only pads heads /
    exchanges / crops.

Public surface:

- :func:`distributed_attention` — pad heads to ``p``, Ulysses all-to-all,
  local :func:`attention`, crop dummy heads.
"""

import torch
import torch.distributed as dist

from worldfoundry.core.attention import attention
from worldfoundry.core.distributed.sequence_parallel.ops import all_to_all, all_to_all_many


# ──────────────────────────────────────────────────────────────────────────
# Padded Ulysses — head-pad so all-to-all shapes match when Nq % p != 0
# ──────────────────────────────────────────────────────────────────────────


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
    fa_version=None,
):
    """
    Sequence-parallel attention via Ulysses all-to-all (https://arxiv.org/pdf/2309.14509).

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    world_size = dist.get_world_size()
    original_num_heads = q.shape[2]
    padded_num_heads = original_num_heads

    # Ulysses all_to_all gathers along sequence after scattering heads.
    # To keep tensor shapes consistent across ranks, pad heads to a multiple
    # of world_size when needed (e.g. 40 heads on 7 GPUs -> 42 heads).
    if original_num_heads % world_size != 0:
        padded_num_heads = ((original_num_heads + world_size - 1) // world_size) * world_size
        pad_heads = padded_num_heads - original_num_heads
        if pad_heads > 0:
            q = torch.cat([q, q.new_zeros(*q.shape[:2], pad_heads, q.shape[3])], dim=2)
            k = torch.cat([k, k.new_zeros(*k.shape[:2], pad_heads, k.shape[3])], dim=2)
            v = torch.cat([v, v.new_zeros(*v.shape[:2], pad_heads, v.shape[3])], dim=2)

    # gather q/k/v sequence
    q, k, v = all_to_all_many(
        (q, k, v),
        scatter_dim=2,
        gather_dim=1,
    )

    x = attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
        version=fa_version,
    )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    if padded_num_heads != original_num_heads:
        x = x[:, :, :original_num_heads, :].contiguous()
    return x
