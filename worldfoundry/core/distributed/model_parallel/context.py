# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor and object split/gather primitives for Context Parallel (CP).

Ranks hold a sequence shard. :func:`split_inputs_cp` / :func:`cat_outputs_cp`
must pair with attention-side Ulysses and RoPE ``shift_t`` CP splits or
positions misalign. Object-list variants pickle Python payloads (masks,
metadata) that are not GPU tensors. ``cat_outputs_cp_with_grad`` keeps
autograd edges for training; inference can use the no-grad cat.
"""

import math
from typing import TypeVar

import torch
from torch import Tensor
from torch.distributed import (
    ProcessGroup,
    all_gather,
    all_gather_object,
    broadcast_object_list,
    get_process_group_ranks,
    get_world_size,
)
from torch.distributed.utils import _verify_param_shape_across_processes

import worldfoundry.core.distributed.runtime.torch_process_group as distributed
from worldfoundry.core.distributed.collectives.tensor import all_gather_concat

try:
    from worldfoundry.core.distributed.model_parallel.megatron_compat import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)

# ──────────────────────────────────────────────────────────────────────────
# Tensor split / gather — must pair with attention Ulysses and RoPE shift_t
# ──────────────────────────────────────────────────────────────────────────


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None) -> Tensor:
    """Slice a tensor along ``seq_dim`` to this rank's CP shard.

    Args:
        x: Input tensor.
        seq_dim: Dimension to split along (negative indexing supported).
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Contiguous slice of length ``x.shape[seq_dim] // cp_size``.

    Raises:
        AssertionError: ``seq_dim`` is not divisible by the CP size.
    """
    if cp_group is None:
        return x

    cp_size = cp_group.size()
    if seq_dim < 0:
        seq_dim = x.ndim + seq_dim  # bring it to positive dimension

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(
        *x.shape[:seq_dim],
        cp_size,
        x.shape[seq_dim] // cp_size,
        *x.shape[(seq_dim + 1) :],
    )
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x.contiguous()


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None) -> Tensor:
    """Gather and concatenate per-rank tensors along ``seq_dim``.

    Args:
        x: This rank's local tensor.
        seq_dim: Concatenation dimension.
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Tensor with the gathered shards concatenated along ``seq_dim``.

    Raises:
        RuntimeError: ``all_gather`` failed.
    """
    if cp_group is None:
        return x

    try:
        return all_gather_concat(x, dim=seq_dim, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather tensors") from e


def cat_outputs_cp_with_grad(x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None) -> Tensor:
    """Gather CP shards while preserving the local rank's autograd graph."""

    if cp_group is None:
        return x

    cp_size = cp_group.size()
    assert cp_size > 0, "cp_size should be greater than 0"
    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]

    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather tensors") from e

    gathered_tensors[cp_group.rank()] = x
    return torch.cat(gathered_tensors, dim=seq_dim)


# ──────────────────────────────────────────────────────────────────────────
# Shape-tolerant broadcast — non-src ranks may start with a dummy shape
# ──────────────────────────────────────────────────────────────────────────


@_disable_compile
def robust_broadcast(
    tensor: torch.Tensor,
    src: int,
    pg: ProcessGroup,
    is_check_shape: bool = False,
) -> torch.Tensor:
    """Broadcast a tensor even when non-source ranks start with different shapes."""

    if tensor.device.type != "cuda" and torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))

    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long, device=tensor.device)
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)
    torch.distributed.broadcast(tensor, src, group=pg)
    return tensor


def broadcast(
    item: torch.Tensor | str | None,
    process_group: ProcessGroup | None = None,
) -> torch.Tensor | str | None:
    """Broadcast a tensor or object from the minimum rank in ``process_group``."""

    if process_group is None:
        return item

    min_rank = min(get_process_group_ranks(process_group))
    if isinstance(item, torch.Tensor):
        return robust_broadcast(item, min_rank, process_group)
    if item is not None:
        broadcastable_list = [item]
        broadcast_object_list(broadcastable_list, min_rank, group=process_group)
        return broadcastable_list[0]
    return item


def broadcast_split_tensor(
    tensor: torch.Tensor | None,
    seq_dim: int,
    process_group: ProcessGroup | None = None,
) -> torch.Tensor | None:
    """Broadcast a tensor from the minimum CP rank, then return this rank's shard."""

    if tensor is None or process_group is None:
        return tensor
    tensor = robust_broadcast(tensor, min(get_process_group_ranks(process_group)), process_group)
    return split_inputs_cp(tensor, seq_dim, process_group)


def find_split(
    shape_tensor: torch.Size,
    cp_size: int,
    patch_values: tuple[int, int, int] = (1, 2, 2),
    view_factor: int = 1,
) -> torch.Size:
    """Find the post-context-parallel temporal/spatial split shape."""

    if not USE_MEGATRON:
        raise ImportError("megatron.core is required for context-parallel split planning.")
    _, _, temporal, height, width = shape_tensor
    splits = []
    assert temporal % view_factor == 0
    temporal = temporal // view_factor
    cp_size_t = 1
    for index, size in enumerate([temporal, height, width]):
        if index == 2 and cp_size > 1:
            raise ValueError("Split by width dimension is not supported; lower the context-parallel size.")
        patch_size = patch_values[index]
        gcd = math.gcd(size // patch_size, cp_size)
        cp_size = cp_size // gcd
        if index == 0:
            cp_size_t = gcd
        splits.append(size // gcd)
    parallel_state.cp_size_t = cp_size_t
    return torch.Size(splits)


T = TypeVar("T")

# ──────────────────────────────────────────────────────────────────────────
# Object-list split / gather — pickle metadata, never large GPU tensors
# ──────────────────────────────────────────────────────────────────────────


def split_inputs_cp_object_list(object_list: list[T], cp_group: ProcessGroup | None = None) -> list[T]:
    """Slice a list to this rank's CP shard.

    Args:
        object_list: List to split.
        cp_group: CP process group; ``None`` returns ``object_list`` unchanged.

    Returns:
        This rank's contiguous slice of length ``len(object_list) // cp_size``.

    Raises:
        AssertionError: ``len(object_list)`` is not divisible by the CP size.
    """
    if cp_group is None:
        return object_list

    cp_size = cp_group.size()
    n_objects = len(object_list)
    assert n_objects % cp_size == 0, f"{n_objects} cannot divide cp_size {cp_size}"

    n_objects_per_rank = n_objects // cp_size
    rank = cp_group.rank()
    start_idx = rank * n_objects_per_rank
    end_idx = start_idx + n_objects_per_rank
    return object_list[start_idx:end_idx]


def cat_outputs_cp_object_list(object_list: list[T], cp_group: ProcessGroup | None = None) -> list[T]:
    """Gather per-rank lists and flatten into a single list.

    Args:
        object_list: This rank's local list.
        cp_group: CP process group; ``None`` returns ``object_list`` unchanged.

    Returns:
        Flattened concatenation of every rank's list.
    """
    if cp_group is None:
        return object_list

    world_size = get_world_size(cp_group)
    gathered_object_list: list[list[T]] = [[] for _ in range(world_size)]

    try:
        all_gather_object(gathered_object_list, object_list, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather objects") from e

    # all_gather_object treats each list as a single object -> flatten.
    return [item for sublist in gathered_object_list for item in sublist]
