# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan runtime setup: shard function, dtype, and eval-mode placement.

Applies a caller-supplied ``shard_fn`` (FSDP or device_map) after the
process group exists. Eval mode freezes dropout and skips grad buckets.
This is glue, not a parallel strategy — pick TP/CP/FSDP first.
"""

import os
import warnings

import torch
import torch.distributed as dist

import worldfoundry.core.distributed.runtime.torch_process_group as _torch_process_group

# ──────────────────────────────────────────────────────────────────────────
# Eval-mode placement — shard_fn is caller-supplied (FSDP / device_map)
# ──────────────────────────────────────────────────────────────────────────


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """Freeze, barrier, then shard — or fall back to a single-device ``.to``.

    The barrier before ``shard_fn`` keeps a slow rank from wrapping while
    others have already started the first FSDP all-gather. Without a process
    group the model is moved locally so inference still runs on one GPU.
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


# ──────────────────────────────────────────────────────────────────────────
# Deprecated NCCL init + scalar reductions used by Wan eval loops
# ──────────────────────────────────────────────────────────────────────────


def init_distributed(world_size, local_rank, rank):
    """Initialize an NCCL process group from explicit rank arguments.

    Deprecated: call
    :func:`worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed`
    after binding ``LOCAL_RANK``. Signature, ``set_device(local_rank)``, and
    the nccl/env:// contract are unchanged (CC-17).
    """

    warnings.warn(
        "worldfoundry.core.distributed.runtime.runtime_setup.init_distributed is deprecated; "
        "use worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed",
        DeprecationWarning,
        stacklevel=2,
    )
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    # Canonical device binding reads LOCAL_RANK; publish the explicit argument
    # so a caller-supplied local_rank still wins over a stale env value.
    os.environ["LOCAL_RANK"] = str(local_rank)
    _torch_process_group.init_torch_distributed(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def dist_mean(local_tensor):
    """All-reduce mean in-place; identity when the process group is not up."""

    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor


def dist_max(local_tensor):
    """All-reduce max in-place; identity when the process group is not up."""

    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
