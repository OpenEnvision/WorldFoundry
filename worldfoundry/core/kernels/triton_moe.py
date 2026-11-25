# SPDX-License-Identifier: Apache-2.0
"""Packed SwiGLU routed-MoE Hopper SM90 WGMMA Triton kernels.

Public entry is :func:`worldfoundry.core.kernels.moe.routed_swiglu_moe`
(lazy ``routed_swiglu_moe_triton``).

Hopper SM90 engineering:
- **Lazy Triton import**: CPU-only environments never import ``triton``.
- **Adaptive Hopper WGMMA tiling**: scales ``(BLOCK_M, BLOCK_I, BLOCK_D)`` up
  to ``128×64×64`` / ``64×128×64`` with ``acc = tl.dot(x, tl.trans(w), acc=acc)``
  and L2-swizzled expert-grouped scheduling for large batches while dropping to
  ``16×32×64`` for tiny token counts, with zero register spills.
- **Vectorized token packing**: packs ``BLOCK_T=128`` tokens per CTA directly
  from ``int32`` or ``int64`` routing indices without an extra cast launch.
- **Top-1 direct store**: when ``top_k == 1``, bypasses output zeroing and FP32
  ``atomic_add`` in favor of direct ``tl.store``.
- **Fast Hopper ``exp2`` SiLU** + strict ``tl.int64`` pointer arithmetic.
- **Compile & CUDA Graph safe**: ``workspace=None`` routes through
  ``worldfoundry::routed_swiglu_moe_fwd`` with a registered fake op, while
  explicit ``workspace`` dicts execute zero-allocation in-place launches for
  CUDA Graph capture/replay.

Not this module:
    Portable PyTorch reference and registry registration live in :mod:`.moe`.
    Callers must not import this file from a DiT block.

Public surface:

- :func:`routed_swiglu_moe_triton` — grouped gate/up then down; optional workspace.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
import torch.version

from worldfoundry.core.kernels.capabilities import QUALIFIED_TRITON_CAPABILITIES, triton_tensor_eligible


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tensor_triton_eligible(tensor: torch.Tensor) -> bool:
    """Dynamo-safe wrapper around :func:`triton_tensor_eligible`."""
    if not torch.compiler.is_compiling():
        return triton_tensor_eligible(tensor)
    if not tensor.is_cuda or bool(getattr(torch.version, "hip", None)):
        return False
    capability = torch.cuda.get_device_capability(tensor.device)
    allow_untested = os.getenv("WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if capability not in QUALIFIED_TRITON_CAPABILITIES and not allow_untested:
        return False
    if tensor.dtype == torch.bfloat16:
        return capability >= (8, 0)
    return tensor.dtype in {torch.float16, torch.float32}


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily configure the Triton cache and compile Hopper MoE kernels."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="moe-triton")

    import triton
    import triton.language as tl

    # ──────────────────────────────────────────────────────────────────────
    # Vectorized token pack — 128 tokens/CTA, direct int32/int64 load
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _moe_pack_selected_kernel(
        selected_ptr,
        counts_ptr,
        rows_ptr,
        slots_ptr,
        T,
        TOPK: tl.constexpr,
        MAX_ROUTES,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Scatter each token's top-k routes into per-expert packed row/slot tables."""
        pid = tl.program_id(0).to(tl.int64)
        t_i64 = tl.cast(T, tl.int64)
        mr_i64 = tl.cast(MAX_ROUTES, tl.int64)

        rows = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        slots = tl.arange(0, BLOCK_K).to(tl.int64)
        mask = (rows[:, None] < t_i64) & (slots[None, :] < TOPK)

        experts = tl.load(selected_ptr + rows[:, None] * TOPK + slots[None, :], mask=mask, other=0).to(tl.int64)
        pos = tl.atomic_add(counts_ptr + experts, 1, sem="relaxed", mask=mask).to(tl.int64)
        store_mask = mask & (pos < mr_i64)
        dest = experts * mr_i64 + pos
        tl.store(rows_ptr + dest, rows[:, None].to(tl.int32), mask=store_mask)
        tl.store(slots_ptr + dest, slots[None, :].to(tl.int32), mask=store_mask)

    # ──────────────────────────────────────────────────────────────────────
    # Grouped Gate + Up WGMMA SwiGLU kernel (L2-swizzled, strict int64)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _moe_gate_up_grouped_kernel(
        x_ptr,
        gate_ptr,
        up_ptr,
        counts_ptr,
        rows_ptr,
        slots_ptr,
        route_ptr,
        inter_ptr,
        D,
        I,  # noqa: E741  # I is intermediate size in T/D/E/I notation
        MAX_ROUTES,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_D: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Grouped ``silu(x @ W_gate^T) * (x @ W_up^T) * route`` into ``[T, K, I]``."""
        pid = tl.program_id(0).to(tl.int64)
        expert = tl.program_id(1).to(tl.int64)

        count = tl.load(counts_ptr + expert).to(tl.int64)
        d_i64 = tl.cast(D, tl.int64)
        i_i64 = tl.cast(I, tl.int64)
        mr_i64 = tl.cast(MAX_ROUTES, tl.int64)

        num_m = (mr_i64 + BLOCK_M - 1) // BLOCK_M
        num_i = (i_i64 + BLOCK_I - 1) // BLOCK_I
        num_pid_in_group = GROUP_M * num_i
        group_id = pid // num_pid_in_group
        first_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_m - first_m, GROUP_M)
        bid_m = first_m + ((pid % num_pid_in_group) % group_size_m)
        bid_i = (pid % num_pid_in_group) // group_size_m

        start_m = bid_m * BLOCK_M
        if start_m >= count:
            return

        route_idx = start_m + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_i = bid_i * BLOCK_I + tl.arange(0, BLOCK_I).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        valid_m = route_idx < count
        valid_i = offs_i < i_i64

        expert_base = expert * mr_i64 + route_idx
        rows = tl.load(rows_ptr + expert_base, mask=valid_m, other=0).to(tl.int64)
        slots = tl.load(slots_ptr + expert_base, mask=valid_m, other=0).to(tl.int64)

        acc_g = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        w_row_base = (expert * i_i64 + offs_i) * d_i64

        for d0 in range(0, D, BLOCK_D):
            ds = tl.cast(d0, tl.int64) + offs_d
            valid_d = ds < d_i64
            x = tl.load(
                x_ptr + rows[:, None] * d_i64 + ds[None, :],
                mask=valid_m[:, None] & valid_d[None, :],
                other=0.0,
            )
            gw = tl.load(
                gate_ptr + w_row_base[:, None] + ds[None, :],
                mask=valid_i[:, None] & valid_d[None, :],
                other=0.0,
            )
            uw = tl.load(
                up_ptr + w_row_base[:, None] + ds[None, :],
                mask=valid_i[:, None] & valid_d[None, :],
                other=0.0,
            )
            acc_g = tl.dot(x, tl.trans(gw), acc=acc_g)
            acc_u = tl.dot(x, tl.trans(uw), acc=acc_u)

        route = tl.load(route_ptr + rows * TOPK + slots, mask=valid_m, other=0.0).to(tl.float32)
        LOG2E: tl.constexpr = 1.4426950408889634
        silu = acc_g / (1.0 + tl.math.exp2(-acc_g * LOG2E))
        val = silu * acc_u * route[:, None]
        tl.store(
            inter_ptr + (rows[:, None] * TOPK + slots[:, None]) * i_i64 + offs_i[None, :],
            val.to(inter_ptr.dtype.element_ty),
            mask=valid_m[:, None] & valid_i[None, :],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Grouped Down-Proj WGMMA kernel (direct store for top_k == 1)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _moe_down_grouped_kernel(
        inter_ptr,
        down_ptr,
        counts_ptr,
        rows_ptr,
        slots_ptr,
        out_ptr,
        D,
        I,  # noqa: E741  # I is intermediate size in T/D/E/I notation
        MAX_ROUTES,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_I: tl.constexpr,
        GROUP_M: tl.constexpr,
        DIRECT_STORE: tl.constexpr,
    ):
        """Grouped down-proj; direct store for ``top_k == 1``, atomic_add otherwise."""
        pid = tl.program_id(0).to(tl.int64)
        expert = tl.program_id(1).to(tl.int64)

        count = tl.load(counts_ptr + expert).to(tl.int64)
        d_i64 = tl.cast(D, tl.int64)
        i_i64 = tl.cast(I, tl.int64)
        mr_i64 = tl.cast(MAX_ROUTES, tl.int64)

        num_m = (mr_i64 + BLOCK_M - 1) // BLOCK_M
        num_d = (d_i64 + BLOCK_D - 1) // BLOCK_D
        num_pid_in_group = GROUP_M * num_d
        group_id = pid // num_pid_in_group
        first_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_m - first_m, GROUP_M)
        bid_m = first_m + ((pid % num_pid_in_group) % group_size_m)
        bid_d = (pid % num_pid_in_group) // group_size_m

        start_m = bid_m * BLOCK_M
        if start_m >= count:
            return

        route_idx = start_m + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_d = bid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        offs_i = tl.arange(0, BLOCK_I).to(tl.int64)
        valid_m = route_idx < count
        valid_d = offs_d < d_i64

        expert_base = expert * mr_i64 + route_idx
        rows = tl.load(rows_ptr + expert_base, mask=valid_m, other=0).to(tl.int64)
        slots = tl.load(slots_ptr + expert_base, mask=valid_m, other=0).to(tl.int64)
        row_slot = rows * TOPK + slots

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        w_row_base = (expert * d_i64 + offs_d) * i_i64

        for i0 in range(0, I, BLOCK_I):
            is_ = tl.cast(i0, tl.int64) + offs_i
            valid_i = is_ < i_i64
            x = tl.load(
                inter_ptr + row_slot[:, None] * i_i64 + is_[None, :],
                mask=valid_m[:, None] & valid_i[None, :],
                other=0.0,
            )
            w = tl.load(
                down_ptr + w_row_base[:, None] + is_[None, :],
                mask=valid_d[:, None] & valid_i[None, :],
                other=0.0,
            )
            acc = tl.dot(x, tl.trans(w), acc=acc)

        out_ptrs = out_ptr + rows[:, None] * d_i64 + offs_d[None, :]
        out_mask = valid_m[:, None] & valid_d[None, :]
        if DIRECT_STORE:
            tl.store(out_ptrs, acc, mask=out_mask)
        else:
            tl.atomic_add(out_ptrs, acc, sem="relaxed", mask=out_mask)

    return {
        "triton": triton,
        "pack": _moe_pack_selected_kernel,
        "gate_up": _moe_gate_up_grouped_kernel,
        "down": _moe_down_grouped_kernel,
    }


def _select_moe_tiles(
    T: int,
    D: int,
    I: int,  # noqa: E741
    E: int,
    top_k: int,
    dtype: torch.dtype,
) -> tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]:
    """Adaptively choose Hopper WGMMA tile sizes for gate/up and down projections."""
    avg_m = (T * top_k) / max(1, E)
    if dtype == torch.float32:
        bm_gu = 64 if avg_m >= 64 else (32 if avg_m > 16 else 16)
        gu_cfg = (bm_gu, 32, 32, 4, 2)
        bm_dn = 64 if avg_m >= 64 else (32 if avg_m > 16 else 16)
        bd_dn = 64 if D >= 64 else 32
        dn_cfg = (bm_dn, bd_dn, 32, 4, 2)
        return gu_cfg, dn_cfg

    # BF16 / FP16 Hopper WGMMA configs (verified n_spills == 0 on SM90).
    if avg_m >= 512 and I >= 64 and D >= 64:
        gu_cfg = (128, 64, 64, 8, 3)
    elif avg_m >= 64 and I >= 64 and D >= 64:
        gu_cfg = (64, 64, 64, 4, 3)
    elif avg_m > 16 and I >= 64 and D >= 64:
        gu_cfg = (32, 64, 64, 4, 3)
    else:
        gu_cfg = (16, 32, 64, 4, 3)

    if avg_m >= 64 and D >= 128 and I >= 64:
        dn_cfg = (64, 128, 64, 4, 3)
    elif avg_m >= 64 and D >= 64 and I >= 64:
        dn_cfg = (64, 64, 64, 4, 3)
    elif avg_m > 16 and D >= 64 and I >= 64:
        dn_cfg = (32, 64, 64, 4, 3)
    else:
        dn_cfg = (16, 64, 64, 4, 3)

    return gu_cfg, dn_cfg


def _run_moe_triton_impl(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    counts: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    inter: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    T, D = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    E, I, _ = int(gate_weight.shape[0]), int(gate_weight.shape[1]), int(gate_weight.shape[2])  # noqa: E741
    top_k = int(selected_experts.shape[1])

    if T == 0:
        return out.reshape_as(hidden_states)
    if top_k == 0 or E == 0 or I == 0 or D == 0:
        out.zero_()
        return out.reshape_as(hidden_states)

    kernels = _load_triton_kernels()
    max_routes = T * top_k
    selected_c = selected_experts if selected_experts.is_contiguous() else selected_experts.contiguous()
    route_c = routing_weights if routing_weights.is_contiguous() else routing_weights.contiguous()

    gu_cfg, dn_cfg = _select_moe_tiles(T, D, I, E, top_k, hidden_states.dtype)
    bm_gu, bi_gu, bd_gu, nw_gu, ns_gu = gu_cfg
    bm_dn, bd_dn, bi_dn, nw_dn, ns_dn = dn_cfg

    with torch.cuda.device(hidden_states.device):
        counts.zero_()
        block_t = 128 if T >= 128 else max(16, _next_power_of_2(T))
        kernels["pack"][(_cdiv(T, block_t),)](
            selected_c,
            counts,
            rows,
            slots,
            T,
            TOPK=top_k,
            MAX_ROUTES=max_routes,
            BLOCK_T=block_t,
            BLOCK_K=_next_power_of_2(top_k),
            num_warps=4 if block_t >= 64 else 1,
        )

        grid_gu = (_cdiv(max_routes, bm_gu) * _cdiv(I, bi_gu), E)
        kernels["gate_up"][grid_gu](
            hidden_states,
            gate_weight,
            up_weight,
            counts,
            rows,
            slots,
            route_c,
            inter,
            D,
            I,
            max_routes,
            TOPK=top_k,
            BLOCK_M=bm_gu,
            BLOCK_I=bi_gu,
            BLOCK_D=bd_gu,
            GROUP_M=4,
            num_warps=nw_gu,
            num_stages=ns_gu,
        )

        direct_store = top_k == 1
        if not direct_store:
            out.zero_()

        grid_dn = (_cdiv(max_routes, bm_dn) * _cdiv(D, bd_dn), E)
        kernels["down"][grid_dn](
            inter,
            down_weight,
            counts,
            rows,
            slots,
            out,
            D,
            I,
            max_routes,
            TOPK=top_k,
            BLOCK_M=bm_dn,
            BLOCK_D=bd_dn,
            BLOCK_I=bi_dn,
            GROUP_M=4,
            DIRECT_STORE=direct_store,
            num_warps=nw_dn,
            num_stages=ns_dn,
        )
    return out.reshape_as(hidden_states)


@torch.library.custom_op("worldfoundry::routed_swiglu_moe_fwd", mutates_args=(), device_types="cuda")
def _routed_swiglu_moe_fwd_op(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    T, D = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    E, I, _ = int(gate_weight.shape[0]), int(gate_weight.shape[1]), int(gate_weight.shape[2])  # noqa: E741
    top_k = int(selected_experts.shape[1])
    max_routes = T * top_k
    device = hidden_states.device

    counts = torch.empty((E,), device=device, dtype=torch.int32)
    rows = torch.empty((E, max_routes), device=device, dtype=torch.int32)
    slots = torch.empty((E, max_routes), device=device, dtype=torch.int32)
    inter = torch.empty((T, top_k, I), device=device, dtype=hidden_states.dtype)
    out = torch.empty((T, D), device=device, dtype=torch.float32)

    return _run_moe_triton_impl(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        counts,
        rows,
        slots,
        inter,
        out,
    )


@_routed_swiglu_moe_fwd_op.register_fake
def _routed_swiglu_moe_fwd_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    del routing_weights, selected_experts, gate_weight, up_weight, down_weight
    return torch.empty(hidden_states.shape, device=hidden_states.device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# Public launcher — workspace shapes are part of the CUDA Graph contract
# ──────────────────────────────────────────────────────────────────────────


def routed_swiglu_moe_triton(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate packed routed SwiGLU experts with Hopper WGMMA Triton matmuls.

    Workspace tensors (when provided for CUDA Graph reuse) must match ``(E,)``
    ``counts`` (int32), ``(E, T*K)`` ``rows``/``slots`` (int32), ``(T, K, I)``
    ``inter`` (activation dtype), and fp32 ``(T, D)`` ``out``. When
    ``workspace`` is ``None``, executes through the registered
    ``worldfoundry::routed_swiglu_moe_fwd`` custom op for ``torch.compile``
    compatibility.
    """
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be 2D, got {tuple(hidden_states.shape)}")
    if selected_experts.ndim != 2 or routing_weights.ndim != 2:
        raise ValueError("selected_experts and routing_weights must be 2D")
    if not hidden_states.is_cuda:
        raise ValueError("routed_swiglu_moe_triton requires CUDA tensors")
    if not _tensor_triton_eligible(hidden_states):
        raise ValueError(f"unsupported GPU/dtype for Triton routed MoE: {hidden_states.device}/{hidden_states.dtype}")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have identical shapes")
    if selected_experts.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors and hidden_states must contain the same number of tokens")

    T, D = hidden_states.shape
    E, I, weight_d = gate_weight.shape  # noqa: E741
    top_k = selected_experts.shape[1]
    if weight_d != D or up_weight.shape != gate_weight.shape or down_weight.shape != (E, D, I):
        raise ValueError(
            "Unexpected MoE weight shapes: "
            f"hidden={tuple(hidden_states.shape)} gate={tuple(gate_weight.shape)} "
            f"up={tuple(up_weight.shape)} down={tuple(down_weight.shape)}"
        )
    tensors = (hidden_states, routing_weights, selected_experts, gate_weight, up_weight, down_weight)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("all routed MoE tensors must share one CUDA device")
    if any(weight.dtype != hidden_states.dtype for weight in (gate_weight, up_weight, down_weight)):
        raise ValueError("packed expert weights and hidden states must share one dtype")
    if not all(tensor.is_contiguous() for tensor in (hidden_states, gate_weight, up_weight, down_weight)):
        raise ValueError("hidden states and packed expert weights must be contiguous")

    if workspace is None:
        return _routed_swiglu_moe_fwd_op(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_weight,
            up_weight,
            down_weight,
        )

    max_routes = T * top_k
    expected_workspace = {
        "counts": ((E,), torch.int32),
        "rows": ((E, max_routes), torch.int32),
        "slots": ((E, max_routes), torch.int32),
        "inter": ((T, top_k, I), hidden_states.dtype),
        "out": ((T, D), torch.float32),
    }
    for name, (shape, dtype) in expected_workspace.items():
        tensor = workspace.get(name)
        if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"invalid routed MoE workspace tensor {name!r}")
        if tensor.device != hidden_states.device:
            raise ValueError(f"routed MoE workspace tensor {name!r} is on the wrong device")

    return _run_moe_triton_impl(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        workspace["counts"],
        workspace["rows"],
        workspace["slots"],
        workspace["inter"],
        workspace["out"],
    )


__all__ = ["routed_swiglu_moe_triton"]
