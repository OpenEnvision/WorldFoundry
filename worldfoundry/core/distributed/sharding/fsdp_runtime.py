"""FSDP1 runtime: checkpointing, scoped unsharding, and device meshes.

FSDP shards parameters and all-gathers them for forward. Inference can
unshard a block only for that forward and reshard immediately to cut
peak VRAM. This module is FSDP1 (flat params). Prefer
:mod:`worldfoundry.core.distributed.sharding.fsdp2_sharding` (DTensor / FSDP2)
for new Wan-style block models. Do not wrap the same module with both.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import (
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)
from torch.distributed.utils import _p_assert

import worldfoundry.core.distributed.runtime.torch_process_group as distributed

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Activation checkpointing and scoped unshard — FSDP1 flat-param only
# ──────────────────────────────────────────────────────────────────────────


def apply_fsdp_checkpointing(model, list_block_cls) -> None:
    """Apply non-reentrant activation checkpointing to matching block classes."""

    logger.info("Applying FSDP activation checkpointing.")
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule):
        """Wrap only transformer-block classes listed by the caller."""

        return any(isinstance(submodule, block_cls) for block_cls in list_block_cls)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )


@contextmanager
def possible_fsdp_scope(model: torch.nn.Module):
    """Temporarily unshard an FSDP module in no-grad inference paths."""

    enabled = isinstance(model, FSDP)
    if enabled:
        assert not torch.is_grad_enabled(), "FSDP context should be entered with grad disabled"
        handle = model._handle
        args, kwargs = [0], {"dummy": 0}
        with torch.autograd.profiler.record_function("FullyShardedDataParallel.possible_fsdp_scope"):
            args, kwargs = _root_pre_forward(model, model, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                model,
                handle,
                _pre_forward_unshard,
                model._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == model.compute_device,
                    "Expected FlatParameter on the FSDP compute device.",
                )
    try:
        yield None
    finally:
        if enabled:
            output = {"output": 1}
            _post_forward(model, handle, _post_forward_reshard, model, unused, output)


def hsdp_device_mesh(replica_group_size=None, sharding_group_size=None, device=None):
    """Initialize a two-dimensional device mesh for FSDP hybrid sharding."""

    world_size = distributed.get_world_size()
    if sharding_group_size is None:
        sharding_group_size = min(world_size, 8)
    sharding_group_size = min(sharding_group_size, world_size)
    if replica_group_size is None:
        replica_group_size = world_size // sharding_group_size

    device = device or "cuda"
    if world_size % sharding_group_size != 0:
        raise ValueError(f"World size {world_size} is not divisible by sharding group size {sharding_group_size}.")
    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(f"Replica group size {replica_group_size} does not divide the available replica groups.")

    device_mesh = init_device_mesh(
        device,
        (replica_group_size, sharding_group_size),
        mesh_dim_names=("replicate", "shard"),
    )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid FSDP device mesh.")

    logger.info(
        "Initialized HSDP device mesh with replica group size %s and sharding group size %s.",
        replica_group_size,
        sharding_group_size,
    )
    return device_mesh


# ──────────────────────────────────────────────────────────────────────────
# Public surface
# ──────────────────────────────────────────────────────────────────────────

__all__ = ["apply_fsdp_checkpointing", "hsdp_device_mesh", "possible_fsdp_scope"]
