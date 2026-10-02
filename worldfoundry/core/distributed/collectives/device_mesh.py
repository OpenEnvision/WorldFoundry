"""Collectives built on torch DeviceMesh and DTensor (FSDP2 / EMA).

DTensor-aware EMA and state broadcast need the local tensor, not the
global view. :func:`get_local_tensor_if_dtensor` unwraps before a host
collective. Used with FSDP2 meshes; plain TP/CP groups stay on
``model_parallel_groups``.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from worldfoundry.core.distributed.collectives.tensor import all_to_all_concat

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import Replicate, distribute_tensor
except ImportError:  # pragma: no cover - optional torch feature.
    Replicate = None
    distribute_tensor = None

# ──────────────────────────────────────────────────────────────────────────
# DeviceMesh broadcast — replicate via DTensor, never raw NCCL on a mesh
# ──────────────────────────────────────────────────────────────────────────


def _mesh_device(mesh: DeviceMesh) -> torch.device:
    """Bind CUDA collectives to this process's current device, not ``cuda:0``.

    A bare ``cuda`` device would land every rank on GPU 0 and NCCL would
    report a duplicate GPU.
    """

    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(mesh.device_type)


def broadcast(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    """Replicate ``tensor`` across a CP/TP mesh via DTensor ``Replicate``.

    Requires ``torch.distributed.tensor``. A missing import is a hard error
    rather than a silent no-op so a single-rank fallback cannot look like a
    successful multi-rank broadcast.
    """

    if Replicate is None or distribute_tensor is None:
        raise ImportError("torch.distributed.tensor is required for DeviceMesh broadcast.")
    tensor = tensor.to(_mesh_device(cp_or_tp_mesh))
    if cp_or_tp_mesh.size() > 1:
        tensor = distribute_tensor(tensor, cp_or_tp_mesh, [Replicate()]).to_local()
    return tensor


def broadcast_with_shape_check(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    """Broadcast a tensor and resize non-source ranks when rank-0 shape differs."""

    device = _mesh_device(cp_or_tp_mesh)
    original_shape = torch.tensor(tensor.shape, device=device)
    final_shape = broadcast(torch.tensor(tensor.shape, device=device), cp_or_tp_mesh)
    if final_shape.ne(original_shape).any():
        tensor = torch.zeros(final_shape.tolist(), dtype=tensor.dtype, device=tensor.device)
    return broadcast(tensor, cp_or_tp_mesh)


def get_local_tensor_if_dtensor(tensor):
    """Return the local shard for DTensor inputs; leave regular tensors unchanged."""

    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def all_to_all_tensor(
    tensor: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Exchange equal tensor chunks and concatenate them along another dimension."""

    actual_world_size = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
    if int(world_size) != int(actual_world_size):
        raise ValueError(
            f"configured world size {world_size} does not match process-group size {actual_world_size}"
        )
    return all_to_all_concat(
        tensor,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        group=group,
    )


# ──────────────────────────────────────────────────────────────────────────
# DTensor EMA — operate on local shards so foreach kernels stay on-device
# ──────────────────────────────────────────────────────────────────────────


class DTensorFastEmaModelUpdater:
    """Foreach-based EMA updater that operates on local DTensor shards."""

    def __init__(self) -> None:
        """Start with an empty cache; :meth:`cache` must precede :meth:`restore`."""

        self.is_cached = False

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        """Overwrite target local shards with source local shards (no broadcast)."""

        with torch.no_grad():
            for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
                get_local_tensor_if_dtensor(tgt_params).data.copy_(get_local_tensor_if_dtensor(src_params).data)

    @torch.no_grad()
    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        """In-place EMA on local FP32 shards: ``tgt = beta * tgt + (1 - beta) * src``.

        EMA in a lower dtype silently underflows the ``1 - beta`` term, so the
        target must already be FP32.
        """

        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            local_tgt = get_local_tensor_if_dtensor(tgt_params)
            local_src = get_local_tensor_if_dtensor(src_params)
            assert local_tgt.dtype == torch.float32, f"EMA model only works in FP32 dtype, got {local_tgt.dtype}."
            target_list.append(local_tgt)
            source_list.append(local_src.data)
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)

    @torch.no_grad()
    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        """Snapshot local shards before a temporary overwrite (eval / checkpoint).

        Nested :meth:`cache` without :meth:`restore` is refused so a later
        restore cannot apply the wrong generation of weights.
        """

        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else ("cuda" if torch.cuda.is_available() else None)
        collected = []
        for param in parameters:
            local_param = get_local_tensor_if_dtensor(param)
            cached = local_param.clone()
            collected.append(cached.to(device) if device is not None else cached)
        self.collected_params = collected
        self.is_cached = True

    @torch.no_grad()
    def restore(self, parameters: Any) -> None:
        """Write cached local shards back; ``strict=False`` zip matches FSDP holes."""

        assert self.is_cached, "EMA cache is not taken yet."
        for cached_param, param in zip(self.collected_params, parameters, strict=False):
            local_param = get_local_tensor_if_dtensor(param)
            local_param.copy_(cached_param.data.type_as(local_param))
        self.collected_params = []
        self.is_cached = False


FastEmaModelUpdater = DTensorFastEmaModelUpdater
get_local_tensor_if_DTensor = get_local_tensor_if_dtensor


# ──────────────────────────────────────────────────────────────────────────
# Replicate-mesh state broadcast — CPU tensors hop through CUDA for NCCL
# ──────────────────────────────────────────────────────────────────────────


def broadcast_dtensor_model_states(model: torch.nn.Module, mesh: DeviceMesh) -> None:
    """Broadcast model parameters and buffers from the first rank in the replicate mesh."""

    replicate_group = mesh.get_group("replicate")
    all_ranks = dist.get_process_group_ranks(replicate_group)
    if len(all_ranks) == 1:
        return

    src_rank = all_ranks[0]
    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        local_tensor = get_local_tensor_if_dtensor(tensor)
        if local_tensor.device.type == "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL DTensor broadcast requires CUDA for CPU-resident model state.")
            broadcast_tensor = local_tensor.to(torch.device("cuda", torch.cuda.current_device()))
            dist.broadcast(broadcast_tensor, src=src_rank, group=replicate_group)
            local_tensor.copy_(broadcast_tensor.cpu())
        else:
            dist.broadcast(local_tensor, src=src_rank, group=replicate_group)


__all__ = [
    "DTensorFastEmaModelUpdater",
    "FastEmaModelUpdater",
    "broadcast",
    "broadcast_dtensor_model_states",
    "broadcast_with_shape_check",
    "get_local_tensor_if_dtensor",
    "get_local_tensor_if_DTensor",
]
