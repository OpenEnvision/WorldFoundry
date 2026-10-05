# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-time tensor/context-parallel batch collectives.

Broadcast or gather a *batch* (prompts, latents) so every TP/CP rank
sees the same sample before sharding activations. Not for per-layer
all-reduce — that stays inside the parallel Linear/Attention.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import broadcast, get_process_group_ranks

from worldfoundry.core.distributed.model_parallel.megatron_compat import mpu, parallel_state

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Load-balanced CP chunking — ranks take [i] and [2P-i-1] to even causal load
# ──────────────────────────────────────────────────────────────────────────


def get_batch_on_this_cp_rank(inputs: torch.Tensor) -> torch.Tensor:
    """Return the load-balanced sequence chunks assigned to this CP rank."""
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return inputs

    cp_rank = mpu.get_context_parallel_rank()
    seq_dim = 1
    if inputs.shape[seq_dim] % (2 * cp_size) != 0:
        raise ValueError(
            f"sequence length {inputs.shape[seq_dim]} must be divisible by 2 * context parallel size {cp_size}"
        )
    inputs = inputs.view(
        *inputs.shape[:seq_dim],
        2 * cp_size,
        inputs.shape[seq_dim] // (2 * cp_size),
        *inputs.shape[seq_dim + 1 :],
    )
    index = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1],
        device=inputs.device,
    )
    inputs = inputs.index_select(seq_dim, index)
    return inputs.view(*inputs.shape[:seq_dim], -1, *inputs.shape[seq_dim + 2 :])


def gather_batch_from_cp_ranks(outputs: torch.Tensor) -> torch.Tensor:
    """Gather load-balanced CP chunks and restore their original sequence order."""
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return outputs

    cp_rank = mpu.get_context_parallel_rank()
    seq_dim = 1
    try:
        chunk_size = outputs.shape[seq_dim] // 2
        outputs = outputs.view(*outputs.shape[:seq_dim], 2, chunk_size, *outputs.shape[seq_dim + 1 :])
        gathered = [torch.empty_like(outputs) for _ in range(cp_size)]
        dist.all_gather(gathered, outputs, group=parallel_state.get_context_parallel_group())
        reordered = [None] * (2 * cp_size)
        for rank, rank_output in enumerate(gathered):
            reordered[rank] = rank_output.select(seq_dim, 0)
            reordered[2 * cp_size - rank - 1] = rank_output.select(seq_dim, 1)
        return torch.cat(reordered, dim=seq_dim)
    except Exception as exc:
        logger.exception("[Rank %s] Failed to gather context-parallel output", cp_rank)
        raise RuntimeError("failed to gather context-parallel output") from exc


# ──────────────────────────────────────────────────────────────────────────
# Batch broadcast — every TP/CP rank must see the same sample before sharding
# ──────────────────────────────────────────────────────────────────────────


def broadcast_data_batch_in_tp_cp_group(data_batch: dict) -> None:
    """Broadcast tensor values across the configured TP and CP groups in-place."""
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group() if tp_size > 1 else None
    cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
    tp_ranks = get_process_group_ranks(tp_group) if tp_group is not None else None
    cp_ranks = get_process_group_ranks(cp_group) if cp_group is not None else None

    for key in sorted(data_batch):
        tensor = data_batch[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor = tensor.contiguous()
        if tp_group is not None:
            broadcast(tensor, min(tp_ranks), group=tp_group)
        if cp_group is not None:
            broadcast(tensor, min(cp_ranks), group=cp_group)


__all__ = [
    "broadcast_data_batch_in_tp_cp_group",
    "gather_batch_from_cp_ranks",
    "get_batch_on_this_cp_rank",
]
