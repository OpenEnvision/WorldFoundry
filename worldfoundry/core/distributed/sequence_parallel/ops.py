# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Sequence-parallel tensor ops: all-to-all, gather, and rank helpers.

Wan Ulysses paths call these instead of raw ``dist.all_to_all``. The
``_many`` variants batch several tensors in one collective to cut NCCL
launch overhead. Gather-forward is for assembling a full sequence on
one rank (decode / loss), not for every attention layer.
"""

import os
import warnings

import torch
import torch.distributed as dist

import worldfoundry.core.distributed.runtime.torch_process_group as _torch_process_group

from worldfoundry.core.distributed.collectives.generic import get_rank, get_world_size  # noqa: F401 - public compatibility exports
from worldfoundry.core.distributed.collectives.tensor import all_gather_concat, all_to_all_concat

# ──────────────────────────────────────────────────────────────────────────
# Default-group init — does NOT create an SP subgroup (that is sequence_parallel_runtime)
# ──────────────────────────────────────────────────────────────────────────


def _distributed_ready():
    """True only when a default process group exists; single-process stays off this path."""

    return dist.is_available() and dist.is_initialized()


def init_distributed_group():
    """Initialize the default NCCL process group if needed.

    Deprecated: this does not create a sequence-parallel group; it only
    initializes the default process group. Call
    :func:`worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed`
    instead (CC-17).
    """

    warnings.warn(
        "worldfoundry.core.distributed.sequence_parallel.ops.init_distributed_group is deprecated; "
        "use worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed",
        DeprecationWarning,
        stacklevel=2,
    )
    if not _distributed_ready():
        _torch_process_group.init_torch_distributed(backend="nccl")


# ──────────────────────────────────────────────────────────────────────────
# Sequence all-to-all / gather — Ulysses hops, not per-layer TP all-reduce
# ──────────────────────────────────────────────────────────────────────────


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = dist.get_world_size(group) if _distributed_ready() else get_world_size()
    if world_size > 1:
        if not kwargs:
            return all_to_all_concat(
                x,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group,
            )
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_to_all_many(tensors, scatter_dim, gather_dim, group=None, **kwargs):
    """Fuse equal-shaped tensors into one destination-major all-to-all.

    Q/K/V exchanges otherwise pay three Python, dispatcher, and NCCL launch
    costs per attention layer.  Packing destination-major avoids the former
    stack-then-movedim copy and maps directly to ``all_to_all_single``. Calls
    larger than ``WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB`` fall back automatically.
    """

    values = tuple(tensors)
    if len(values) < 2 or not _distributed_ready():
        return values
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return values
    first = values[0]
    compatible = all(
        value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in values[1:]
    )
    try:
        max_bytes = max(
            int(float(os.getenv("WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB", "512") or "512") * 1024**2),
            0,
        )
    except ValueError:
        max_bytes = 512 * 1024**2
    total_bytes = sum(value.numel() * value.element_size() for value in values)
    all_to_all_single = getattr(dist, "all_to_all_single", None)
    if not compatible or max_bytes == 0 or total_bytes > max_bytes or kwargs or not callable(all_to_all_single):
        return tuple(all_to_all(value, scatter_dim, gather_dim, group=group, **kwargs) for value in values)

    ndim = first.ndim
    raw_scatter_dim = int(scatter_dim)
    raw_gather_dim = int(gather_dim)
    if ndim < 1 or not -ndim <= raw_scatter_dim < ndim or not -ndim <= raw_gather_dim < ndim:
        return tuple(all_to_all(value, scatter_dim, gather_dim, group=group) for value in values)
    scatter_dim = raw_scatter_dim % ndim
    gather_dim = raw_gather_dim % ndim
    if first.shape[scatter_dim] % world_size:
        return tuple(all_to_all(value, scatter_dim, gather_dim, group=group) for value in values)
    local_scatter = first.shape[scatter_dim] // world_size
    other_dims = [dimension for dimension in range(ndim) if dimension != scatter_dim]
    packed = first.new_empty(
        world_size,
        local_scatter,
        len(values),
        *(first.shape[dimension] for dimension in other_dims),
    )
    for index, value in enumerate(values):
        source = value.movedim(scatter_dim, 0).unflatten(0, (world_size, local_scatter))
        packed[:, :, index].copy_(source)

    exchanged = torch.empty_like(packed)
    all_to_all_single(exchanged, packed, group=group)

    original_axis = {scatter_dim: 1}
    original_axis.update({dimension: axis + 3 for axis, dimension in enumerate(other_dims)})
    permutation = [2]
    for dimension in range(ndim):
        if dimension == gather_dim:
            permutation.append(0)
        permutation.append(original_axis[dimension])
    output_shape = list(first.shape)
    output_shape[scatter_dim] //= world_size
    output_shape[gather_dim] *= world_size
    outputs = exchanged.permute(permutation).contiguous().reshape(len(values), *output_shape)
    return outputs.unbind(dim=0)


def all_gather(tensor):
    """Gather equal tensors into a Python list; identity list when world size is 1."""

    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    """Concatenate sequence shards along ``dim`` for decode / loss, not every layer."""

    # skip if world_size == 1
    world_size = get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    return all_gather_concat(input, dim=dim)
