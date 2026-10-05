"""Allocation-efficient equal-shape tensor collectives.

The inference attention paths exchange equally shaped sequence/head shards.
Using functional collectives (or ``*_into_tensor`` fallbacks) keeps each
exchange in one contiguous buffer, avoiding a Python list and one allocation
per rank on every transformer block.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

try:
    import torch.distributed._functional_collectives as _functional_collectives
except ImportError:  # pragma: no cover - torch 2.7+ provides this module.
    _functional_collectives = None

# ──────────────────────────────────────────────────────────────────────────
# Dimension / group helpers — negative dims and WORLD fallback
# ──────────────────────────────────────────────────────────────────────────


def _normalise_dim(dim: int, ndim: int, *, name: str) -> int:
    """Convert a possibly-negative dim to ``[0, ndim)`` or raise.

    A 0-D tensor has no gather axis; fail here rather than let NCCL see an
    empty split.
    """

    if ndim < 1:
        raise ValueError(f"{name} requires a tensor with at least one dimension")
    value = int(dim)
    if not -ndim <= value < ndim:
        raise IndexError(f"{name} dimension {dim} is invalid for a {ndim}-D tensor")
    return value % ndim


def _world_size(group: dist.ProcessGroup | None) -> int:
    """Process-group size, or 1 so single-process callers skip the collective."""

    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _functional_group(group: dist.ProcessGroup | None) -> Any:
    """Functional collectives treat ``None`` as WORLD, not "no communication"."""

    return dist.group.WORLD if group is None else group


# ──────────────────────────────────────────────────────────────────────────
# Equal-shape gather / all-to-all — one packed buffer, not a Python list
# ──────────────────────────────────────────────────────────────────────────


def all_gather_concat(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Gather equal tensors and concatenate rank shards along ``dim``."""

    dim = _normalise_dim(dim, tensor.ndim, name="all_gather_concat")
    world_size = _world_size(group)
    if world_size == 1:
        return tensor
    input_tensor = tensor.contiguous()

    functional = getattr(_functional_collectives, "all_gather_tensor", None)
    if callable(functional):
        output = functional(
            input_tensor,
            gather_dim=dim,
            group=_functional_group(group),
        )
        return output.contiguous()

    # Older supported runtimes may lack functional collectives but expose the
    # allocation-efficient c10d primitive. It gathers along dimension zero, so
    # transpose the requested dimension around the operation.
    gather_into = getattr(dist, "all_gather_into_tensor", None)
    if callable(gather_into):
        packed = input_tensor.movedim(dim, 0).contiguous() if dim != 0 else input_tensor
        output_shape = list(packed.shape)
        output_shape[0] *= world_size
        output = packed.new_empty(output_shape)
        gather_into(output, packed, group=group)
        if dim != 0:
            output = output.movedim(0, dim).contiguous()
        return output

    gathered = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, input_tensor, group=group)
    return torch.cat(gathered, dim=dim).contiguous()


def all_to_all_concat(
    tensor: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Exchange equal shards from ``scatter_dim`` and join on ``gather_dim``.

    This is equivalent to list-based ``dist.all_to_all`` followed by
    ``torch.cat(outputs, dim=gather_dim)``, but communicates through one packed
    tensor and therefore maps directly to NCCL ``allToAll``.
    """

    scatter_dim = _normalise_dim(scatter_dim, tensor.ndim, name="all_to_all_concat")
    gather_dim = _normalise_dim(gather_dim, tensor.ndim, name="all_to_all_concat")
    world_size = _world_size(group)
    if world_size == 1:
        return tensor
    if tensor.shape[scatter_dim] % world_size:
        raise ValueError(
            f"scatter dimension {tensor.shape[scatter_dim]} must be divisible by world size {world_size}"
        )

    # Destination chunks occupy consecutive ranges on dimension zero, which is
    # the memory contract expected by all_to_all_single.
    packed = tensor.movedim(scatter_dim, 0).contiguous()
    functional = getattr(_functional_collectives, "all_to_all_single", None)
    if callable(functional):
        exchanged = functional(
            packed,
            output_split_sizes=None,
            input_split_sizes=None,
            group=_functional_group(group),
        )
    else:
        exchanged = torch.empty_like(packed)
        all_to_all_single = getattr(dist, "all_to_all_single", None)
        if not callable(all_to_all_single):
            inputs = [chunk.contiguous() for chunk in torch.chunk(tensor, world_size, dim=scatter_dim)]
            outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
            dist.all_to_all(outputs, inputs, group=group)
            return torch.cat(outputs, dim=gather_dim).contiguous()
        all_to_all_single(exchanged, packed, group=group)

    # The packed result is [source_rank, local_scatter, all_other_dims]. Move
    # source_rank directly before the original gather dimension and merge the
    # pair, reproducing rank-ordered torch.cat semantics for every dim pair.
    local_scatter = tensor.shape[scatter_dim] // world_size
    other_dims = [index for index in range(tensor.ndim) if index != scatter_dim]
    exchanged = exchanged.reshape(
        world_size,
        local_scatter,
        *(tensor.shape[index] for index in other_dims),
    )
    original_axis = {scatter_dim: 1}
    original_axis.update({dimension: axis + 2 for axis, dimension in enumerate(other_dims)})
    permutation: list[int] = []
    for dimension in range(tensor.ndim):
        if dimension == gather_dim:
            permutation.append(0)
        permutation.append(original_axis[dimension])
    exchanged = exchanged.permute(permutation)
    output_shape = list(tensor.shape)
    output_shape[scatter_dim] //= world_size
    output_shape[gather_dim] *= world_size
    return exchanged.reshape(output_shape).contiguous()


__all__ = ["all_gather_concat", "all_to_all_concat"]
