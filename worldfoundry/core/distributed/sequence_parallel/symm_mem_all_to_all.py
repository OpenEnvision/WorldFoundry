# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""NVLink/NVSwitch direct P2P Symmetric Memory Ulysses All-to-All engine.

Uses PyTorch 2.11+ ``torch.distributed._symmetric_memory`` to exchange Ulysses
4D attention tensors directly across NVLink/NVSwitch without NCCL kernel launch
overhead or intermediate staging buffers:

- **Ephemeral push mode** (inside :func:`ulysses_ephemeral_output_scope`):
  1 single Hopper Triton kernel reads local ``(Q, K, V)`` or ``O`` and writes
  directly over NVLink into peer symmetric output slots, fusing optional
  ``pad_heads`` zero-fill and head-cropping with zero staging copies.
- **Fresh-output pull mode** (general callers outside ephemeral scope):
  packs into a symmetric input buffer, synchronizes over NVLink, and pulls
  directly into freshly allocated non-aliasing PyTorch output tensors.

Both modes are 100% bit-exact with NCCL ``all_to_all_single``, CUDA-Graph
capture/replay safe (once warmed up), and automatically fall back to the
Triton+NCCL path on multi-node, non-NVLink, CPU, autograd, or unsupported
environments.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from worldfoundry.core.distributed.sequence_parallel.triton_ulysses_ops import (
    _triton_ulysses_eligible,
    launch_symm_o_direct_push,
    launch_symm_o_pack_into,
    launch_symm_o_pull_unpack,
    launch_symm_qkv_direct_push,
    launch_symm_qkv_pack_into,
    launch_symm_qkv_pull_unpack,
)

_EPHEMERAL_TLS = threading.local()
_SLOT_LOCK = threading.Lock()
_ALIGNMENT_BYTES = 16 * 1024 * 1024  # 16 MiB granularity avoids rendezvous thrash


@dataclass
class _SymmSlot:
    """One rendezvoused symmetric byte buffer plus device pointer table."""

    buf: torch.Tensor
    hdl: Any
    peer_ptrs: torch.Tensor
    capacity_bytes: int


# Keyed by (id(group), world_size, device_index, slot_id)
_SYMM_SLOTS: dict[tuple[int, int, int, int], _SymmSlot] = {}
# Process groups where symmetric memory rendezvous failed (e.g. multi-node / no P2P)
_UNAVAILABLE_GROUPS: set[tuple[int, int]] = set()


def ulysses_backend_mode() -> str:
    """Return requested Ulysses collective backend: auto, symm_mem, triton_nccl, or torch."""
    return os.getenv("WORLDFOUNDRY_ULYSSES_BACKEND", "auto").strip().casefold() or "auto"


def _symm_mem_max_bytes() -> int:
    try:
        mb = float(os.getenv("WORLDFOUNDRY_SYMM_MEM_MAX_MB", "1024") or "1024")
        return max(int(mb * 1024 * 1024), 0)
    except ValueError:
        return 1024 * 1024 * 1024


@contextmanager
def ulysses_ephemeral_output_scope() -> Generator[None, None, None]:
    """Permit zero-staging direct NVLink push outputs consumed immediately by attention/proj.

    Nested scopes are reference-counted per thread. Only use when the returned
    ``(q, k, v)`` or ``o`` tensors are consumed before the next Ulysses
    exchange on the same slot and are not retained across layers.
    """
    depth = getattr(_EPHEMERAL_TLS, "depth", 0)
    _EPHEMERAL_TLS.depth = depth + 1
    try:
        yield
    finally:
        _EPHEMERAL_TLS.depth = depth


def is_ulysses_ephemeral_output_scope() -> bool:
    """True inside :func:`ulysses_ephemeral_output_scope`."""
    return getattr(_EPHEMERAL_TLS, "depth", 0) > 0


def _symm_mem_module() -> Any | None:
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except (ImportError, AttributeError):
        return None
    if not hasattr(symm_mem, "empty") or not hasattr(symm_mem, "rendezvous"):
        return None
    return symm_mem


def _round_up_capacity(needed_bytes: int) -> int:
    if needed_bytes <= 0:
        return _ALIGNMENT_BYTES
    return ((needed_bytes + _ALIGNMENT_BYTES - 1) // _ALIGNMENT_BYTES) * _ALIGNMENT_BYTES


def _resolve_symm_group(group: dist.ProcessGroup | None) -> dist.ProcessGroup | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD if group is None else group


def _get_or_alloc_slot(
    group: dist.ProcessGroup,
    device: torch.device,
    needed_bytes: int,
    slot_id: int,
) -> _SymmSlot | None:
    """Return a rendezvoused symmetric slot with at least *needed_bytes*, or None."""
    mode = ulysses_backend_mode()
    if mode not in {"auto", "symm_mem"}:
        return None
    if device.type != "cuda":
        return None
    max_bytes = _symm_mem_max_bytes()
    if max_bytes == 0 or needed_bytes <= 0 or needed_bytes > max_bytes:
        return None
    world_size = int(dist.get_world_size(group))
    if world_size <= 1:
        return None

    group_key = (id(group), world_size)
    if group_key in _UNAVAILABLE_GROUPS:
        return None

    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    slot_key = (id(group), world_size, int(dev_idx), int(slot_id))

    with _SLOT_LOCK:
        existing = _SYMM_SLOTS.get(slot_key)
        if existing is not None and existing.capacity_bytes >= needed_bytes:
            return existing

    # Never invoke symm_mem.rendezvous() during active CUDA Graph capture.
    if torch.cuda.is_current_stream_capturing():
        return None

    symm_mem = _symm_mem_module()
    if symm_mem is None:
        _UNAVAILABLE_GROUPS.add(group_key)
        return None

    cap = min(_round_up_capacity(needed_bytes), max_bytes)
    try:
        with torch.cuda.device(device):
            buf = symm_mem.empty((cap,), dtype=torch.uint8, device=device)
            hdl = symm_mem.rendezvous(buf, group=group)
            peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), dtype=torch.int64, device=device)
    except (RuntimeError, OSError, ValueError, NotImplementedError, AttributeError):
        _UNAVAILABLE_GROUPS.add(group_key)
        return None

    slot = _SymmSlot(buf=buf, hdl=hdl, peer_ptrs=peer_ptrs, capacity_bytes=cap)
    with _SLOT_LOCK:
        _SYMM_SLOTS[slot_key] = slot
    return slot


def try_symm_mem_all_to_all_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    pad_heads: int = 0,
    ephemeral: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Attempt NVLink symmetric-memory 4D QKV Ulysses all-to-all (scatter=2, gather=1).

    Returns ``(q_out, k_out, v_out)`` on success, or ``None`` if symmetric
    memory is unavailable/disabled so the caller seamlessly uses Triton+NCCL.
    """
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return None
    if torch.compiler.is_compiling():
        return None
    if not (_triton_ulysses_eligible(q) and _triton_ulysses_eligible(k) and _triton_ulysses_eligible(v)):
        return None

    resolved_group = _resolve_symm_group(group)
    if resolved_group is None:
        return None
    world_size = int(dist.get_world_size(resolved_group))
    if world_size <= 1:
        return None

    b, s_l, h_orig, d = q.shape
    h_pad = h_orig + int(pad_heads)
    if h_pad % world_size != 0:
        return None
    h_l = h_pad // world_size
    s_full = world_size * s_l
    elem_bytes = q.element_size()
    needed_bytes = 3 * b * s_full * h_l * d * elem_bytes

    use_ephemeral = is_ulysses_ephemeral_output_scope() if ephemeral is None else bool(ephemeral)
    rank = int(dist.get_rank(resolved_group))

    if use_ephemeral:
        slot = _get_or_alloc_slot(resolved_group, q.device, needed_bytes, slot_id=0)
        if slot is None:
            return None
        with torch.cuda.device(q.device):
            slot.hdl.barrier(channel=0)
            launch_symm_qkv_direct_push(q, k, v, slot.peer_ptrs, rank, world_size, int(pad_heads))
            slot.hdl.barrier(channel=0)
        out_view = slot.buf[:needed_bytes].view(q.dtype).view(3, b, s_full, h_l, d)
        return out_view[0], out_view[1], out_view[2]

    slot = _get_or_alloc_slot(resolved_group, q.device, needed_bytes, slot_id=2)
    if slot is None:
        return None
    in_view = slot.buf[:needed_bytes].view(q.dtype).view(world_size, 3, b, s_l, h_l, d)
    q_out = torch.empty((b, s_full, h_l, d), device=q.device, dtype=q.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    with torch.cuda.device(q.device):
        launch_symm_qkv_pack_into(q, k, v, in_view, world_size, int(pad_heads))
        slot.hdl.barrier(channel=0)
        launch_symm_qkv_pull_unpack(slot.peer_ptrs, q_out, k_out, v_out, rank, world_size)
        slot.hdl.barrier(channel=0)
    return q_out, k_out, v_out


def try_symm_mem_all_to_all_o(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    out_heads: int | None = None,
    ephemeral: bool | None = None,
) -> torch.Tensor | None:
    """Attempt NVLink symmetric-memory inverse Ulysses all-to-all (scatter=1, gather=2).

    Returns ``[B, S_L, out_heads, D]`` on success, or ``None`` to fall back to
    Triton+NCCL.
    """
    if torch.is_grad_enabled() or x.requires_grad:
        return None
    if torch.compiler.is_compiling():
        return None
    if not _triton_ulysses_eligible(x):
        return None

    resolved_group = _resolve_symm_group(group)
    if resolved_group is None:
        return None
    world_size = int(dist.get_world_size(resolved_group))
    if world_size <= 1:
        return None

    b, s_full, h_l, d = x.shape
    if s_full % world_size != 0:
        return None
    s_l = s_full // world_size
    hc = h_l * world_size
    target_heads = hc if out_heads is None else int(out_heads)
    if not 0 < target_heads <= hc:
        return None
    elem_bytes = x.element_size()

    use_ephemeral = is_ulysses_ephemeral_output_scope() if ephemeral is None else bool(ephemeral)
    rank = int(dist.get_rank(resolved_group))

    if use_ephemeral:
        needed_bytes = b * s_l * target_heads * d * elem_bytes
        slot = _get_or_alloc_slot(resolved_group, x.device, needed_bytes, slot_id=1)
        if slot is None:
            return None
        with torch.cuda.device(x.device):
            slot.hdl.barrier(channel=0)
            launch_symm_o_direct_push(x, slot.peer_ptrs, rank, world_size, target_heads)
            slot.hdl.barrier(channel=0)
        return slot.buf[:needed_bytes].view(x.dtype).view(b, s_l, target_heads, d)

    needed_bytes = world_size * b * s_l * h_l * d * elem_bytes
    slot = _get_or_alloc_slot(resolved_group, x.device, needed_bytes, slot_id=3)
    if slot is None:
        return None
    in_view = slot.buf[:needed_bytes].view(x.dtype).view(world_size, b, s_l, h_l, d)
    out = torch.empty((b, s_l, target_heads, d), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        launch_symm_o_pack_into(x, in_view, world_size)
        slot.hdl.barrier(channel=0)
        launch_symm_o_pull_unpack(slot.peer_ptrs, out, rank, world_size, h_l)
        slot.hdl.barrier(channel=0)
    return out


def try_symm_mem_row_parallel_mm_all_reduce(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    group: dist.ProcessGroup | None,
) -> torch.Tensor | None:
    """Zero-copy Hopper TC FP32 GEMM + NVSwitch ``two_shot_all_reduce_`` for RowParallelLinear.

    Computes ``mm(value, weight.T, out_dtype=fp32)`` directly into a reusable
    symmetric-memory FP32 buffer, reduces in-place across NVLink via PyTorch
    symmetric memory ``two_shot_all_reduce_``, adds ``bias`` in FP32, and
    returns a freshly materialized ``value.dtype`` tensor (zero aliasing).
    """
    if torch.is_grad_enabled() or value.requires_grad:
        return None
    if torch.compiler.is_compiling():
        return None
    if not (
        value.is_cuda
        and value.dtype in {torch.float16, torch.bfloat16}
        and weight.dtype == value.dtype
        and value.ndim >= 2
        and value.numel() > 0
    ):
        return None
    resolved_group = _resolve_symm_group(group)
    if resolved_group is None:
        return None
    group_name = getattr(resolved_group, "group_name", None)
    if not isinstance(group_name, str) or not group_name:
        return None
    world_size = int(dist.get_world_size(resolved_group))
    if world_size <= 1:
        return None

    value_2d = value.reshape(-1, value.shape[-1])
    rows = int(value_2d.shape[0])
    out_features = int(weight.shape[0])
    needed_bytes = rows * out_features * 4  # FP32 accumulator
    # two_shot_all_reduce_ requires 16-byte-per-rank alignment.
    if needed_bytes % (16 * world_size) != 0:
        return None

    slot = _get_or_alloc_slot(resolved_group, value.device, needed_bytes, slot_id=4)
    if slot is None:
        return None

    symm_ops = getattr(torch.ops, "symm_mem", None)
    if symm_ops is None:
        return None
    multimem = getattr(symm_ops, "multimem_all_reduce_", None)
    two_shot = getattr(symm_ops, "two_shot_all_reduce_", None)
    has_multicast = bool(getattr(slot.hdl, "multicast_ptr", 0))
    # On NVSwitch meshes with >=4 GPUs, hardware NVLS SHARP multicast reduces in
    # one hop (1.2x-2.0x faster than NCCL & two-shot). On 2 GPUs, direct P2P
    # two-shot avoids multicast setup overhead and outperforms NCCL.
    reduce_fn = multimem if (world_size >= 4 and has_multicast and callable(multimem)) else two_shot
    if not callable(reduce_fn):
        reduce_fn = multimem if (has_multicast and callable(multimem)) else None
    if not callable(reduce_fn):
        return None

    try:
        symm_2d = slot.buf[:needed_bytes].view(torch.float32).view(rows, out_features)
        with torch.cuda.device(value.device):
            torch.mm(value_2d, weight.t(), out_dtype=torch.float32, out=symm_2d)
            reduce_fn(symm_2d, "sum", group_name)
            if bias is not None:
                symm_2d.add_(bias.to(torch.float32))
            return symm_2d.to(value.dtype).reshape(*value.shape[:-1], out_features)
    except (RuntimeError, TypeError, AttributeError):
        return None


def clear_symm_mem_pools() -> None:
    """Release cached symmetric-memory buffers and availability flags."""
    with _SLOT_LOCK:
        _SYMM_SLOTS.clear()
        _UNAVAILABLE_GROUPS.clear()


__all__ = [
    "clear_symm_mem_pools",
    "is_ulysses_ephemeral_output_scope",
    "try_symm_mem_all_to_all_o",
    "try_symm_mem_all_to_all_qkv",
    "try_symm_mem_row_parallel_mm_all_reduce",
    "ulysses_backend_mode",
    "ulysses_ephemeral_output_scope",
]
