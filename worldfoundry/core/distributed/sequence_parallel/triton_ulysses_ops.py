# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Hopper SM90 fused Triton kernels for lossless Ulysses 4D all-to-all layouts.

Eliminates redundant ``torch.cat``, ``transpose().contiguous()``, and
``pad_heads`` zero-padding / head-cropping HBM passes around sequence-parallel
collectives, and provides direct NVLink peer-to-peer push/pull kernels used by
:mod:`worldfoundry.core.distributed.sequence_parallel.symm_mem_all_to_all`.

All Triton imports are strictly deferred so CPU-only environments never import
Triton. All tensor index arithmetic uses 64-bit integers (``tl.int64``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_ULYSSES_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _triton_ulysses_eligible(tensor: torch.Tensor) -> bool:
    """Return True when *tensor* can run the Hopper Triton Ulysses kernels."""
    return (
        tensor.is_cuda
        and tensor.dtype in _ULYSSES_DTYPES
        and tensor.numel() > 0
        and tensor.stride(-1) == 1
        and not bool(getattr(torch.version, "hip", None))
    )


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily import Triton and compile the Ulysses pack/unpack/P2P kernels."""
    try:
        from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="ulysses-triton")
    except (ImportError, OSError):
        pass

    import triton
    import triton.language as tl

    @triton.jit
    def _pack_qkv_dst_major_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        B,
        S_L,
        H_ORIG,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Pack 3×[B, S_L, H_ORIG, D] (+ optional head zero-pad) into [P, 3, B, S_L, H_L, D]."""
        pid_bs = tl.program_id(0).to(tl.int64)
        pid_pt = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        dst_p = pid_pt // 3
        t_idx = pid_pt - dst_p * 3
        b_idx = pid_bs // S_L
        s_l = pid_bs - b_idx * S_L

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = dst_p * H_L + h_l
        in_bounds = mask & (h_global < H_ORIG)

        if t_idx == 0:
            src_off = b_idx * stride_qb + s_l * stride_qs + h_global * stride_qh + d_idx
            val = tl.load(q_ptr + src_off, mask=in_bounds, other=0.0)
        elif t_idx == 1:
            src_off = b_idx * stride_kb + s_l * stride_ks + h_global * stride_kh + d_idx
            val = tl.load(k_ptr + src_off, mask=in_bounds, other=0.0)
        else:
            src_off = b_idx * stride_vb + s_l * stride_vs + h_global * stride_vh + d_idx
            val = tl.load(v_ptr + src_off, mask=in_bounds, other=0.0)

        dst_row = ((dst_p * 3 + t_idx) * B + b_idx) * S_L + s_l
        tl.store(out_ptr + dst_row * hd_l + hd_offs, val, mask=mask)

    @triton.jit
    def _unpack_qkv_src_major_kernel(
        in_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        B,
        S_L,
        HD_L,
        P: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Unpack [P, 3, B, S_L, HD_L] into 3 contiguous [B, P*S_L, HD_L] outputs."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_bt = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        t_idx = pid_bt // B
        b_idx = pid_bt - t_idx * B
        src_p = pid_s // S_L
        s_l = pid_s - src_p * S_L

        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < HD_L

        src_row = ((src_p * 3 + t_idx) * B + b_idx) * S_L + s_l
        val = tl.load(in_ptr + src_row * HD_L + hd_offs, mask=mask)

        dst_row = b_idx * (P * S_L) + pid_s
        if t_idx == 0:
            tl.store(q_out_ptr + dst_row * HD_L + hd_offs, val, mask=mask)
        elif t_idx == 1:
            tl.store(k_out_ptr + dst_row * HD_L + hd_offs, val, mask=mask)
        else:
            tl.store(v_out_ptr + dst_row * HD_L + hd_offs, val, mask=mask)

    @triton.jit
    def _pack_single_scatter_head_kernel(
        x_ptr,
        out_ptr,
        stride_xb,
        stride_xs,
        stride_xh,
        B,
        S_L,
        H_ORIG,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Pack [B, S_L, H_ORIG, D] (+ optional head zero-pad) into [P, B, S_L, H_L, D]."""
        pid_bs = tl.program_id(0).to(tl.int64)
        dst_p = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        b_idx = pid_bs // S_L
        s_l = pid_bs - b_idx * S_L

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = dst_p * H_L + h_l
        in_bounds = mask & (h_global < H_ORIG)

        src_off = b_idx * stride_xb + s_l * stride_xs + h_global * stride_xh + d_idx
        val = tl.load(x_ptr + src_off, mask=in_bounds, other=0.0)

        dst_row = (dst_p * B + b_idx) * S_L + s_l
        tl.store(out_ptr + dst_row * hd_l + hd_offs, val, mask=mask)

    @triton.jit
    def _unpack_single_gather_seq_kernel(
        in_ptr,
        out_ptr,
        B,
        S_L,
        HD_L,
        P: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Unpack [P, B, S_L, HD_L] into [B, P*S_L, HD_L] (used when B > 1)."""
        pid_s = tl.program_id(0).to(tl.int64)
        b_idx = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        src_p = pid_s // S_L
        s_l = pid_s - src_p * S_L

        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < HD_L

        src_row = (src_p * B + b_idx) * S_L + s_l
        val = tl.load(in_ptr + src_row * HD_L + hd_offs, mask=mask)

        dst_row = b_idx * (P * S_L) + pid_s
        tl.store(out_ptr + dst_row * HD_L + hd_offs, val, mask=mask)

    @triton.jit
    def _pack_o_scatter_seq_kernel(
        x_ptr,
        out_ptr,
        stride_xb,
        stride_xs,
        stride_xh,
        B,
        S_L,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Pack [B, P*S_L, H_L, D] into [P, B, S_L, H_L, D] (used when B > 1 or strided)."""
        pid_bs = tl.program_id(0).to(tl.int64)
        dst_p = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        b_idx = pid_bs // S_L
        s_l = pid_bs - b_idx * S_L
        s_global = dst_p * S_L + s_l

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D

        src_off = b_idx * stride_xb + s_global * stride_xs + h_l * stride_xh + d_idx
        val = tl.load(x_ptr + src_off, mask=mask)

        dst_row = (dst_p * B + b_idx) * S_L + s_l
        tl.store(out_ptr + dst_row * hd_l + hd_offs, val, mask=mask)

    @triton.jit
    def _unpack_o_gather_head_kernel(
        in_ptr,
        out_ptr,
        B,
        S_L,
        H_OUT,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Unpack [P, B, S_L, H_L, D] into [B, S_L, H_OUT, D] with fused head crop."""
        pid_bs = tl.program_id(0).to(tl.int64)
        src_p = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        b_idx = pid_bs // S_L
        s_l = pid_bs - b_idx * S_L

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = src_p * H_L + h_l
        out_mask = mask & (h_global < H_OUT)

        src_row = (src_p * B + b_idx) * S_L + s_l
        val = tl.load(in_ptr + src_row * hd_l + hd_offs, mask=out_mask)

        dst_off = (pid_bs * H_OUT + h_global) * D + d_idx
        tl.store(out_ptr + dst_off, val, mask=out_mask)

    # ──────────────────────────────────────────────────────────────────────
    # Direct NVLink P2P Symmetric Memory Kernels (Latin-square peer swizzled
    # to eliminate 8-to-1 NVSwitch incast, coarse BLOCK_S streaming tiles)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _symm_qkv_direct_push_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        peer_out_ptrs_ptr,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        src_rank,
        B,
        S_L,
        H_ORIG,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """1-kernel Latin-square NVSwitch push: local (Q,K,V) -> peer[dst][3,B,P*S_L,H_L,D]."""
        pid_s_blk = tl.program_id(0).to(tl.int64)
        pid_pt = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        raw_p = pid_pt // 3
        t_idx = pid_pt - raw_p * 3
        # Latin-square swizzle: every rank targets a distinct peer per wave.
        dst_p = (src_rank + raw_p) % P

        peer_base_i64 = tl.load(peer_out_ptrs_ptr + dst_p)
        peer_ptr = peer_base_i64.to(tl.pointer_type(q_ptr.dtype.element_ty))

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        hd_mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = dst_p * H_L + h_l
        in_h = hd_mask & (h_global < H_ORIG)

        s_full = P * S_L
        total_bs = B * S_L
        s_start = pid_s_blk * BLOCK_S

        for i in range(BLOCK_S):
            bs_idx = s_start + i
            if bs_idx < total_bs:
                b_idx = bs_idx // S_L
                s_l = bs_idx - b_idx * S_L
                if t_idx == 0:
                    src_off = b_idx * stride_qb + s_l * stride_qs + h_global * stride_qh + d_idx
                    val = tl.load(q_ptr + src_off, mask=in_h, other=0.0)
                elif t_idx == 1:
                    src_off = b_idx * stride_kb + s_l * stride_ks + h_global * stride_kh + d_idx
                    val = tl.load(k_ptr + src_off, mask=in_h, other=0.0)
                else:
                    src_off = b_idx * stride_vb + s_l * stride_vs + h_global * stride_vh + d_idx
                    val = tl.load(v_ptr + src_off, mask=in_h, other=0.0)
                dst_row = (t_idx * B + b_idx) * s_full + (src_rank * S_L + s_l)
                tl.store(peer_ptr + dst_row * hd_l + hd_offs, val, mask=hd_mask, cache_modifier=".cs")

    @triton.jit
    def _symm_qkv_pull_unpack_kernel(
        peer_in_ptrs_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        dst_rank,
        B,
        S_L,
        HD_L,
        P: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Latin-square swizzled pull from each peer's [P,3,B,S_L,HD_L] into fresh (Q,K,V)."""
        pid_s_blk = tl.program_id(0).to(tl.int64)
        pid_pbt = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        tb = 3 * B
        raw_p = pid_pbt // tb
        rem = pid_pbt - raw_p * tb
        t_idx = rem // B
        b_idx = rem - t_idx * B
        src_p = (dst_rank + raw_p) % P

        peer_base_i64 = tl.load(peer_in_ptrs_ptr + src_p)
        peer_ptr = peer_base_i64.to(tl.pointer_type(q_out_ptr.dtype.element_ty))

        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        hd_mask = hd_offs < HD_L

        s_full = P * S_L
        s_start = pid_s_blk * BLOCK_S
        src_base_row = ((dst_rank * 3 + t_idx) * B + b_idx) * S_L
        dst_base_row = b_idx * s_full + src_p * S_L

        for i in range(BLOCK_S):
            s_l = s_start + i
            if s_l < S_L:
                val = tl.load(
                    peer_ptr + (src_base_row + s_l) * HD_L + hd_offs,
                    mask=hd_mask,
                    eviction_policy="evict_first",
                )
                if t_idx == 0:
                    tl.store(q_out_ptr + (dst_base_row + s_l) * HD_L + hd_offs, val, mask=hd_mask)
                elif t_idx == 1:
                    tl.store(k_out_ptr + (dst_base_row + s_l) * HD_L + hd_offs, val, mask=hd_mask)
                else:
                    tl.store(v_out_ptr + (dst_base_row + s_l) * HD_L + hd_offs, val, mask=hd_mask)

    @triton.jit
    def _symm_o_direct_push_kernel(
        o_ptr,
        peer_out_ptrs_ptr,
        stride_ob,
        stride_os,
        stride_oh,
        src_rank,
        B,
        S_L,
        H_OUT,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """1-kernel Latin-square NVSwitch push: local O -> peer[dst][B,S_L,H_OUT,D] (fused crop)."""
        pid_s_blk = tl.program_id(0).to(tl.int64)
        raw_p = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        dst_p = (src_rank + raw_p) % P
        peer_base_i64 = tl.load(peer_out_ptrs_ptr + dst_p)
        peer_ptr = peer_base_i64.to(tl.pointer_type(o_ptr.dtype.element_ty))

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        hd_mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = src_rank * H_L + h_l
        out_mask = hd_mask & (h_global < H_OUT)

        total_bs = B * S_L
        s_start = pid_s_blk * BLOCK_S
        for i in range(BLOCK_S):
            bs_idx = s_start + i
            if bs_idx < total_bs:
                b_idx = bs_idx // S_L
                s_l = bs_idx - b_idx * S_L
                s_global = dst_p * S_L + s_l
                src_off = b_idx * stride_ob + s_global * stride_os + h_l * stride_oh + d_idx
                val = tl.load(o_ptr + src_off, mask=out_mask)
                dst_off = (bs_idx * H_OUT + h_global) * D + d_idx
                tl.store(peer_ptr + dst_off, val, mask=out_mask, cache_modifier=".cs")

    @triton.jit
    def _symm_o_pull_unpack_kernel(
        peer_in_ptrs_ptr,
        out_ptr,
        dst_rank,
        B,
        S_L,
        H_OUT,
        H_L,
        P: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Latin-square swizzled pull from each peer's [P,B,S_L,H_L,D] into fresh O[B,S_L,H_OUT,D]."""
        pid_s_blk = tl.program_id(0).to(tl.int64)
        raw_p = tl.program_id(1).to(tl.int64)
        pid_hd = tl.program_id(2).to(tl.int64)

        src_p = (dst_rank + raw_p) % P
        peer_base_i64 = tl.load(peer_in_ptrs_ptr + src_p)
        peer_ptr = peer_base_i64.to(tl.pointer_type(out_ptr.dtype.element_ty))

        hd_l = H_L * D
        hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD).to(tl.int64)
        hd_mask = hd_offs < hd_l
        h_l = hd_offs // D
        d_idx = hd_offs - h_l * D
        h_global = src_p * H_L + h_l
        out_mask = hd_mask & (h_global < H_OUT)

        total_bs = B * S_L
        s_start = pid_s_blk * BLOCK_S
        for i in range(BLOCK_S):
            bs_idx = s_start + i
            if bs_idx < total_bs:
                b_idx = bs_idx // S_L
                s_l = bs_idx - b_idx * S_L
                src_row = (dst_rank * B + b_idx) * S_L + s_l
                val = tl.load(
                    peer_ptr + src_row * hd_l + hd_offs,
                    mask=out_mask,
                    eviction_policy="evict_first",
                )
                dst_off = (bs_idx * H_OUT + h_global) * D + d_idx
                tl.store(out_ptr + dst_off, val, mask=out_mask)

    return {
        "triton": triton,
        "pack_qkv": _pack_qkv_dst_major_kernel,
        "unpack_qkv": _unpack_qkv_src_major_kernel,
        "pack_single": _pack_single_scatter_head_kernel,
        "unpack_single": _unpack_single_gather_seq_kernel,
        "pack_o": _pack_o_scatter_seq_kernel,
        "unpack_o": _unpack_o_gather_head_kernel,
        "symm_qkv_push": _symm_qkv_direct_push_kernel,
        "symm_qkv_pull": _symm_qkv_pull_unpack_kernel,
        "symm_o_push": _symm_o_direct_push_kernel,
        "symm_o_pull": _symm_o_pull_unpack_kernel,
    }


def _block_hd_for(hd_l: int, triton_mod: Any) -> tuple[int, int]:
    """Select coalesced Hopper tile width and warp count for ``H_L * D``."""
    block_hd = min(1024, max(64, int(triton_mod.next_power_of_2(max(hd_l, 1)))))
    num_warps = 8 if block_hd >= 1024 else 4
    return block_hd, num_warps


# ──────────────────────────────────────────────────────────────────────────
# PyTorch custom_op wrappers — compile-safe and auto-covered by Inductor
# ──────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("worldfoundry::ulysses_pack_qkv_fwd", mutates_args=())
def _ulysses_pack_qkv_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    pad_heads: int,
) -> torch.Tensor:
    b, s_l, h_orig, d = q.shape
    h_pad = h_orig + int(pad_heads)
    h_l = h_pad // int(world_size)
    out = torch.empty((int(world_size), 3, b, s_l, h_l, d), device=q.device, dtype=q.dtype)
    if out.numel() == 0:
        return out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size) * 3, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(q.device):
        kernels["pack_qkv"][grid](
            q,
            k,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            b,
            s_l,
            h_orig,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return out


@_ulysses_pack_qkv_op.register_fake
def _ulysses_pack_qkv_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    pad_heads: int,
) -> torch.Tensor:
    del k, v
    b, s_l, h_orig, d = q.shape
    h_l = (h_orig + int(pad_heads)) // int(world_size)
    return torch.empty((int(world_size), 3, b, s_l, h_l, d), device=q.device, dtype=q.dtype)


@torch.library.custom_op("worldfoundry::ulysses_unpack_qkv_fwd", mutates_args=())
def _ulysses_unpack_qkv_op(
    exchanged: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p, _, b, s_l, h_l, d = exchanged.shape
    s_full = p * s_l
    q_out = torch.empty((b, s_full, h_l, d), device=exchanged.device, dtype=exchanged.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    if q_out.numel() == 0:
        return q_out, k_out, v_out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (s_full, 3 * b, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(exchanged.device):
        kernels["unpack_qkv"][grid](
            exchanged,
            q_out,
            k_out,
            v_out,
            b,
            s_l,
            hd_l,
            P=int(world_size),
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out


@_ulysses_unpack_qkv_op.register_fake
def _ulysses_unpack_qkv_fake(
    exchanged: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del world_size
    p, _, b, s_l, h_l, d = exchanged.shape
    shape = (b, p * s_l, h_l, d)
    return (
        torch.empty(shape, device=exchanged.device, dtype=exchanged.dtype),
        torch.empty(shape, device=exchanged.device, dtype=exchanged.dtype),
        torch.empty(shape, device=exchanged.device, dtype=exchanged.dtype),
    )


@torch.library.custom_op("worldfoundry::ulysses_pack_single_fwd", mutates_args=())
def _ulysses_pack_single_op(
    x: torch.Tensor,
    world_size: int,
    pad_heads: int,
) -> torch.Tensor:
    b, s_l, h_orig, d = x.shape
    h_l = (h_orig + int(pad_heads)) // int(world_size)
    out = torch.empty((int(world_size), b, s_l, h_l, d), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(x.device):
        kernels["pack_single"][grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            b,
            s_l,
            h_orig,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return out


@_ulysses_pack_single_op.register_fake
def _ulysses_pack_single_fake(
    x: torch.Tensor,
    world_size: int,
    pad_heads: int,
) -> torch.Tensor:
    b, s_l, h_orig, d = x.shape
    h_l = (h_orig + int(pad_heads)) // int(world_size)
    return torch.empty((int(world_size), b, s_l, h_l, d), device=x.device, dtype=x.dtype)


@torch.library.custom_op("worldfoundry::ulysses_unpack_single_fwd", mutates_args=())
def _ulysses_unpack_single_op(
    exchanged: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    p, b, s_l, h_l, d = exchanged.shape
    s_full = p * s_l
    out = torch.empty((b, s_full, h_l, d), device=exchanged.device, dtype=exchanged.dtype)
    if out.numel() == 0:
        return out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (s_full, b, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(exchanged.device):
        kernels["unpack_single"][grid](
            exchanged,
            out,
            b,
            s_l,
            hd_l,
            P=int(world_size),
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return out


@_ulysses_unpack_single_op.register_fake
def _ulysses_unpack_single_fake(
    exchanged: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    del world_size
    p, b, s_l, h_l, d = exchanged.shape
    return torch.empty((b, p * s_l, h_l, d), device=exchanged.device, dtype=exchanged.dtype)


@torch.library.custom_op("worldfoundry::ulysses_pack_o_fwd", mutates_args=())
def _ulysses_pack_o_op(
    x: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    b, s_full, h_l, d = x.shape
    s_l = s_full // int(world_size)
    out = torch.empty((int(world_size), b, s_l, h_l, d), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(x.device):
        kernels["pack_o"][grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            b,
            s_l,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return out


@_ulysses_pack_o_op.register_fake
def _ulysses_pack_o_fake(
    x: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    b, s_full, h_l, d = x.shape
    return torch.empty((int(world_size), b, s_full // int(world_size), h_l, d), device=x.device, dtype=x.dtype)


@torch.library.custom_op("worldfoundry::ulysses_unpack_o_fwd", mutates_args=())
def _ulysses_unpack_o_op(
    exchanged: torch.Tensor,
    world_size: int,
    out_heads: int,
) -> torch.Tensor:
    _, b, s_l, h_l, d = exchanged.shape
    out = torch.empty((b, s_l, int(out_heads), d), device=exchanged.device, dtype=exchanged.dtype)
    if out.numel() == 0:
        return out
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(exchanged.device):
        kernels["unpack_o"][grid](
            exchanged,
            out,
            b,
            s_l,
            int(out_heads),
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return out


@_ulysses_unpack_o_op.register_fake
def _ulysses_unpack_o_fake(
    exchanged: torch.Tensor,
    world_size: int,
    out_heads: int,
) -> torch.Tensor:
    del world_size
    _, b, s_l, _, d = exchanged.shape
    return torch.empty((b, s_l, int(out_heads), d), device=exchanged.device, dtype=exchanged.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Public Python dispatchers with exact PyTorch fallbacks
# ──────────────────────────────────────────────────────────────────────────


def ulysses_pack_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    *,
    pad_heads: int = 0,
) -> torch.Tensor:
    """Pack ``(q, k, v)`` of shape ``[B, S_L, H, D]`` into ``[P, 3, B, S_L, H_L, D]``."""
    if _triton_ulysses_eligible(q) and _triton_ulysses_eligible(k) and _triton_ulysses_eligible(v):
        return _ulysses_pack_qkv_op(q, k, v, int(world_size), int(pad_heads))
    b, s_l, h_orig, d = q.shape
    h_pad = h_orig + int(pad_heads)
    h_l = h_pad // int(world_size)
    if pad_heads > 0:
        pad = q.new_zeros((b, s_l, int(pad_heads), d))
        q = torch.cat((q, pad), dim=2)
        k = torch.cat((k, pad), dim=2)
        v = torch.cat((v, pad), dim=2)
    stacked = torch.stack((q, k, v), dim=0)  # [3, B, S_L, H_pad, D]
    return stacked.reshape(3, b, s_l, int(world_size), h_l, d).permute(3, 0, 1, 2, 4, 5).contiguous()


def ulysses_unpack_qkv(
    exchanged: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack ``[P, 3, B, S_L, H_L, D]`` into 3 contiguous ``[B, P*S_L, H_L, D]`` tensors."""
    if _triton_ulysses_eligible(exchanged) and exchanged.is_contiguous():
        return _ulysses_unpack_qkv_op(exchanged, int(world_size))
    p, _, b, s_l, h_l, d = exchanged.shape
    unpacked = exchanged.permute(1, 2, 0, 3, 4, 5).contiguous().reshape(3, b, p * s_l, h_l, d)
    return unpacked[0], unpacked[1], unpacked[2]


def ulysses_pack_single_scatter_head(
    x: torch.Tensor,
    world_size: int,
    *,
    pad_heads: int = 0,
) -> torch.Tensor:
    """Pack ``[B, S_L, H, D]`` (+ optional head pad) into ``[P, B, S_L, H_L, D]``."""
    if _triton_ulysses_eligible(x):
        return _ulysses_pack_single_op(x, int(world_size), int(pad_heads))
    b, s_l, h_orig, d = x.shape
    h_pad = h_orig + int(pad_heads)
    h_l = h_pad // int(world_size)
    if pad_heads > 0:
        x = torch.cat((x, x.new_zeros((b, s_l, int(pad_heads), d))), dim=2)
    return x.reshape(b, s_l, int(world_size), h_l, d).permute(2, 0, 1, 3, 4).contiguous()


def ulysses_unpack_single_gather_seq(
    exchanged: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Unpack ``[P, B, S_L, H_L, D]`` into ``[B, P*S_L, H_L, D]`` (zero-copy when B==1)."""
    p, b, s_l, h_l, d = exchanged.shape
    if b == 1 and exchanged.is_contiguous():
        return exchanged.view(1, p * s_l, h_l, d)
    if _triton_ulysses_eligible(exchanged) and exchanged.is_contiguous():
        return _ulysses_unpack_single_op(exchanged, int(world_size))
    return exchanged.transpose(0, 1).contiguous().reshape(b, p * s_l, h_l, d)


def ulysses_pack_o_scatter_seq(
    x: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Pack ``[B, P*S_L, H_L, D]`` into ``[P, B, S_L, H_L, D]`` (zero-copy when B==1)."""
    b, s_full, h_l, d = x.shape
    s_l = s_full // int(world_size)
    if b == 1 and x.is_contiguous():
        return x.view(int(world_size), 1, s_l, h_l, d)
    if _triton_ulysses_eligible(x):
        return _ulysses_pack_o_op(x, int(world_size))
    return x.reshape(b, int(world_size), s_l, h_l, d).transpose(0, 1).contiguous()


def ulysses_unpack_o_gather_head(
    exchanged: torch.Tensor,
    world_size: int,
    *,
    out_heads: int | None = None,
) -> torch.Tensor:
    """Unpack ``[P, B, S_L, H_L, D]`` into ``[B, S_L, out_heads, D]`` (fused head crop)."""
    p, b, s_l, h_l, d = exchanged.shape
    target_heads = p * h_l if out_heads is None else int(out_heads)
    if _triton_ulysses_eligible(exchanged) and exchanged.is_contiguous():
        return _ulysses_unpack_o_op(exchanged, int(world_size), target_heads)
    out = exchanged.permute(1, 2, 0, 3, 4).contiguous().reshape(b, s_l, p * h_l, d)
    if target_heads < p * h_l:
        out = out[:, :, :target_heads, :].contiguous()
    return out


# ──────────────────────────────────────────────────────────────────────────
# Direct NVLink P2P Kernel Launchers (called by symm_mem_all_to_all.py)
# ──────────────────────────────────────────────────────────────────────────


def launch_symm_qkv_pack_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dst_symm: torch.Tensor,
    world_size: int,
    pad_heads: int,
) -> None:
    """Pack ``(q, k, v)`` directly into pre-allocated symmetric ``dst_symm``."""
    b, s_l, h_orig, d = q.shape
    h_l = (h_orig + int(pad_heads)) // int(world_size)
    if b * s_l * h_l * d == 0:
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size) * 3, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(q.device):
        kernels["pack_qkv"][grid](
            q,
            k,
            v,
            dst_symm,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            b,
            s_l,
            h_orig,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


def _symm_block_s_for(rows: int) -> int:
    """Choose coarse row-streaming tile size for NVLink P2P kernels."""
    if rows >= 512:
        return 16
    if rows >= 128:
        return 8
    return 4


def launch_symm_qkv_direct_push(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    peer_out_ptrs: torch.Tensor,
    src_rank: int,
    world_size: int,
    pad_heads: int,
) -> None:
    """1-kernel Latin-square NVSwitch push from local ``(q, k, v)`` into peers' symmetric output."""
    b, s_l, h_orig, d = q.shape
    h_l = (h_orig + int(pad_heads)) // int(world_size)
    if b * s_l * h_l * d == 0:
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, _ = _block_hd_for(hd_l, triton_mod)
    num_warps = 8 if block_hd >= 512 else 4
    block_s = _symm_block_s_for(b * s_l)
    grid = (triton_mod.cdiv(b * s_l, block_s), int(world_size) * 3, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(q.device):
        kernels["symm_qkv_push"][grid](
            q,
            k,
            v,
            peer_out_ptrs,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            int(src_rank),
            b,
            s_l,
            h_orig,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_S=block_s,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


def launch_symm_qkv_pull_unpack(
    peer_in_ptrs: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    dst_rank: int,
    world_size: int,
) -> None:
    """Latin-square swizzled pull from all peers' symmetric input buffers into fresh ``(q_out, k_out, v_out)``."""
    b, s_full, h_l, d = q_out.shape
    s_l = s_full // int(world_size)
    if q_out.numel() == 0:
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, _ = _block_hd_for(hd_l, triton_mod)
    num_warps = 8 if block_hd >= 512 else 4
    block_s = _symm_block_s_for(s_l)
    grid = (triton_mod.cdiv(s_l, block_s), int(world_size) * 3 * b, triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(q_out.device):
        kernels["symm_qkv_pull"][grid](
            peer_in_ptrs,
            q_out,
            k_out,
            v_out,
            int(dst_rank),
            b,
            s_l,
            hd_l,
            P=int(world_size),
            BLOCK_S=block_s,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


def launch_symm_o_pack_into(
    x: torch.Tensor,
    dst_symm: torch.Tensor,
    world_size: int,
) -> None:
    """Pack ``O`` ``[B, P*S_L, H_L, D]`` directly into symmetric ``[P, B, S_L, H_L, D]``."""
    b, s_full, h_l, d = x.shape
    s_l = s_full // int(world_size)
    if x.numel() == 0:
        return
    if b == 1 and x.is_contiguous():
        dst_symm.copy_(x.view(int(world_size), 1, s_l, h_l, d))
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, num_warps = _block_hd_for(hd_l, triton_mod)
    grid = (b * s_l, int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(x.device):
        kernels["pack_o"][grid](
            x,
            dst_symm,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            b,
            s_l,
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


def launch_symm_o_direct_push(
    x: torch.Tensor,
    peer_out_ptrs: torch.Tensor,
    src_rank: int,
    world_size: int,
    out_heads: int,
) -> None:
    """1-kernel Latin-square NVSwitch push from local ``O`` into peers' symmetric output with fused crop."""
    b, s_full, h_l, d = x.shape
    s_l = s_full // int(world_size)
    if x.numel() == 0 or out_heads <= 0:
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = h_l * d
    block_hd, _ = _block_hd_for(hd_l, triton_mod)
    num_warps = 8 if block_hd >= 512 else 4
    block_s = _symm_block_s_for(b * s_l)
    grid = (triton_mod.cdiv(b * s_l, block_s), int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(x.device):
        kernels["symm_o_push"][grid](
            x,
            peer_out_ptrs,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            int(src_rank),
            b,
            s_l,
            int(out_heads),
            h_l,
            P=int(world_size),
            D=d,
            BLOCK_S=block_s,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


def launch_symm_o_pull_unpack(
    peer_in_ptrs: torch.Tensor,
    out: torch.Tensor,
    dst_rank: int,
    world_size: int,
    h_l: int,
) -> None:
    """Latin-square swizzled pull from all peers' symmetric input buffers into fresh ``out[B, S_L, H_OUT, D]``."""
    b, s_l, out_heads, d = out.shape
    if out.numel() == 0:
        return
    kernels = _load_triton_kernels()
    triton_mod = kernels["triton"]
    hd_l = int(h_l) * d
    block_hd, _ = _block_hd_for(hd_l, triton_mod)
    num_warps = 8 if block_hd >= 512 else 4
    block_s = _symm_block_s_for(b * s_l)
    grid = (triton_mod.cdiv(b * s_l, block_s), int(world_size), triton_mod.cdiv(hd_l, block_hd))
    with torch.cuda.device(out.device):
        kernels["symm_o_pull"][grid](
            peer_in_ptrs,
            out,
            int(dst_rank),
            b,
            s_l,
            out_heads,
            int(h_l),
            P=int(world_size),
            D=d,
            BLOCK_S=block_s,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )


__all__ = [
    "_triton_ulysses_eligible",
    "launch_symm_o_direct_push",
    "launch_symm_o_pack_into",
    "launch_symm_o_pull_unpack",
    "launch_symm_qkv_direct_push",
    "launch_symm_qkv_pack_into",
    "launch_symm_qkv_pull_unpack",
    "ulysses_pack_o_scatter_seq",
    "ulysses_pack_qkv",
    "ulysses_pack_single_scatter_head",
    "ulysses_unpack_o_gather_head",
    "ulysses_unpack_qkv",
    "ulysses_unpack_single_gather_seq",
]
