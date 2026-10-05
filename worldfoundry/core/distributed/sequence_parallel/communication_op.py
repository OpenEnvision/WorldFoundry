# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py
"""SP/TP collectives: all-reduce, all-gather, and 4D all-to-all.

Thin wrappers over :mod:`parallel_state` groups. TP all-reduce
restores a sharded Linear; SP 4D all-to-all is the Ulysses swap. Call
these instead of ``dist.all_reduce`` so the correct subgroup is used.

Not a process-group factory — groups must already exist via
:func:`~.parallel_state.initialize_model_parallel`. Uneven sequence
lengths are rejected: NCCL all-to-all here assumes equal shard sizes.

Public surface: :func:`tensor_model_parallel_all_reduce`,
:func:`tensor_model_parallel_all_gather`,
:func:`sequence_model_parallel_all_to_all_4D`,
:func:`sequence_model_parallel_all_gather`.
"""

import torch
import torch.distributed

from .parallel_state import get_sp_group, get_tp_group


# ──────────────────────────────────────────────────────────────────────────
# Tensor-parallel collectives — restore a column/row-sharded Linear
# ──────────────────────────────────────────────────────────────────────────


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """Sum shards across the TP group so a row-parallel Linear is whole again.

    Uses :func:`~.parallel_state.get_tp_group`, not WORLD. A WORLD all-reduce
    would mix SP replicas that already hold the same residual.
    """
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Concatenate TP shards along ``dim`` (default last) into a full tensor.

    Bound to the TP subgroup so language-encoder gathers cannot pull DiT
    sequence shards. All TP ranks must pass the same shape.
    """
    return get_tp_group().all_gather(input_, dim)


# ──────────────────────────────────────────────────────────────────────────
# Sequence-parallel collectives — Ulysses head ↔ sequence swap
# ──────────────────────────────────────────────────────────────────────────


def sequence_model_parallel_all_to_all_4D(
    input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
) -> torch.Tensor:
    """All-to-all communication of 4D tensors (e.g. QKV matrices) across sequence parallel group.

    NOTE: Does not support uneven sequence lengths. The scatter dimension must
    be divisible by the SP world size, otherwise an assertion error is raised.
    """
    sp_group = get_sp_group()
    sp_world_size = sp_group.world_size
    if sp_world_size == 1:
        return input_
    scatter_size = input_.shape[scatter_dim]
    assert scatter_size % sp_world_size == 0, (
        f"sequence_model_parallel_all_to_all_4D: input shape[{scatter_dim}]={scatter_size} "
        f"is not divisible by sp_world_size={sp_world_size}. "
        f"Full input shape: {tuple(input_.shape)}"
    )
    return sp_group.all_to_all_4D(input_, scatter_dim, gather_dim)


def sequence_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across sequence parallel group.

    NOTE: All SP ranks must have the same tensor shape. If shapes differ,
    all_gather_into_tensor will fail.
    """
    return get_sp_group().all_gather(input_, dim)
