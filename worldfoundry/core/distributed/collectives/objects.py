"""Object and tensor collectives for lightweight distributed jobs.

``broadcast_object`` / gather of Python payloads (configs, path lists).
Do not put large GPU tensors here — use ``tensor_collectives`` so NCCL
stays on device.

Public surface: :func:`all_gather`, :func:`reduce_dict`, :func:`synchronize`,
:data:`LOCAL_PROCESS_GROUP`.
"""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch.utils import data

from worldfoundry.core.distributed.collectives.generic import get_rank, get_world_size
from worldfoundry.core.distributed.collectives.generic import is_master as is_primary

LOCAL_PROCESS_GROUP = None

# ──────────────────────────────────────────────────────────────────────────
# Intra-node group + object collectives — pickle stays off the NCCL tensor path
# ──────────────────────────────────────────────────────────────────────────


def get_local_rank() -> int:
    """Return this process's rank inside :data:`LOCAL_PROCESS_GROUP`.

    Unlike :func:`torch_process_group.get_local_rank`, this requires the
    intra-node group created by :mod:`multiprocess_launch`. A missing group
    is a hard error so a caller cannot silently treat every rank as local 0.
    """

    if not dist.is_available() or not dist.is_initialized():
        return 0
    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("LOCAL_PROCESS_GROUP is None")
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def synchronize() -> None:
    """Barrier on the default group; no-op outside a multi-rank process group."""

    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """In-place all-reduce; identity when ``world_size == 1`` so callers stay branch-free."""

    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(value):
    """Gather picklable Python objects from every rank."""

    from worldfoundry.core.distributed.collectives.evaluation import all_gather as gather

    return gather(value)


def reduce_dict(input_dict, average=True):
    """Reduce tensor values onto rank 0; keys are sorted so ranks agree on order.

    Non-zero ranks still participate in the collective but must not divide —
    only rank 0 holds the accumulated sum.
    """

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys], 0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
    return {key: value for key, value in zip(keys, values)}


def data_sampler(dataset, shuffle, distributed):
    """Pick a sampler that does not duplicate examples across ranks when distributed."""

    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)


__all__ = [
    "LOCAL_PROCESS_GROUP",
    "all_gather",
    "all_reduce",
    "data_sampler",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_primary",
    "reduce_dict",
    "synchronize",
]
