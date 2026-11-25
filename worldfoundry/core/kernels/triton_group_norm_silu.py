# SPDX-License-Identifier: Apache-2.0
#
# Lossless Hopper SM90 in-tree Triton GroupNorm + SiLU implementation.

"""In-tree fused GroupNorm + SiLU Hopper SM90 Triton kernels.

Entry point is selected by :func:`worldfoundry.core.kernels.diffusion.group_norm_silu`.
Do not import this file from DiT blocks — it skips capability gates.

Key properties:
- **Lazy Triton import**: importing this module on CPU-only environments never
  imports ``triton`` or initializes CUDA.
- **Numerically lossless**: uses deterministic shift-stabilized parallel Chan's
  variance reduction with exact ``tl.int64`` total counts (immune to
  catastrophic cancellation on large-mean features) and fast Hopper ``exp2`` SiLU.
- **Strict ``tl.int64`` indexing**: safe on ``>2^31``-element 5D video tensors.
- **Native layouts**: supports contiguous 2D–5D (``NCHW`` / ``NCDHW``) as well
  as ``channels_last`` (4D ``NHWC``) and ``channels_last_3d`` (5D ``NDHWC``)
  with coalesced multi-group tiles.
- **Compile & CUDA Graph safe**: registered as
  ``worldfoundry::core_group_norm_silu_fwd`` with a layout-preserving fake op.

Not this module:
    Eligibility, size thresholds, and PyTorch fallback live in :mod:`.diffusion`.

Public surface:

- :func:`group_norm_silu` — layout-preserving GroupNorm + SiLU CUDA kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

_ONE_PASS_CF_MAX_GROUP_SIZE = 4096
_CF_REDUCE_BLOCK = 4096
_CF_APPLY_BLOCK = 4096


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _is_channels_last_nd(x: torch.Tensor) -> bool:
    """Return True when *x* is strictly stored in 4D NHWC or 5D NDHWC layout."""
    if x.ndim == 5:
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            if not x.is_contiguous():
                return True
            return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
        return False
    if x.ndim == 4:
        if x.is_contiguous(memory_format=torch.channels_last):
            if not x.is_contiguous():
                return True
            return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(3) == x.shape[1]
        return False
    return False


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily configure the Triton cache and compile GroupNorm+SiLU kernels."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="group-norm-triton")

    import triton
    import triton.language as tl

    # ──────────────────────────────────────────────────────────────────────
    # Deterministic parallel Chan's variance combine (exact int64 counts)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _chan_combine_i64(n_a, m_a, m2_a, n_b, m_b, m2_b):
        new_n = n_a + n_b
        denom = tl.maximum(new_n, 1).to(tl.float32)
        delta = m_b - m_a
        has_both = (n_a > 0) & (n_b > 0)
        wb = n_b.to(tl.float32) / denom
        new_m = tl.where(new_n > 0, m_a + delta * wb, 0.0)
        new_m2 = m2_a + m2_b + tl.where(has_both, (delta * delta) * (n_a.to(tl.float32) * wb), 0.0)
        return new_n, new_m, new_m2

    # ──────────────────────────────────────────────────────────────────────
    # Contiguous (NCHW / NCDHW) one-pass kernel for small groups (<= 4096)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _group_norm_silu_cf_1pass_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        channels,
        spatial_size,
        channels_per_group,
        group_size,
        eps,
        BLOCK: tl.constexpr,
    ):
        """Single-pass centered-variance GroupNorm + affine + fast exp2 SiLU."""
        group_id = tl.program_id(0).to(tl.int64)
        batch_id = tl.program_id(1).to(tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        s_i64 = tl.cast(spatial_size, tl.int64)
        cg_i64 = tl.cast(channels_per_group, tl.int64)
        gs_i64 = tl.cast(group_size, tl.int64)

        group_base = batch_id * c_i64 * s_i64 + group_id * gs_i64
        indices = tl.arange(0, BLOCK).to(tl.int64)
        mask = indices < gs_i64

        val = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        inv_group = 1.0 / gs_i64.to(tl.float32)
        mean = tl.sum(val, axis=0) * inv_group
        centered = tl.where(mask, val - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) * inv_group
        rstd = tl.rsqrt(var + eps)

        channel_offsets = group_id * cg_i64 + indices // s_i64
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)

        y = tl.math.fma(val - mean, rstd * weight, bias)
        LOG2E: tl.constexpr = 1.4426950408889634
        y = y / (1.0 + tl.math.exp2(-y * LOG2E))
        tl.store(output_ptr + group_base + indices, y.to(output_ptr.dtype.element_ty), mask=mask)

    # ──────────────────────────────────────────────────────────────────────
    # Contiguous (NCHW / NCDHW) multi-chunk Chan reduce & per-channel apply
    # Single packed workspace layout:
    #   [0 : rows * num_chunks)                  -> chunk_cnt (float32, exact <= 2^24)
    #   [rows * num_chunks : 2 * rows * num_chunks) -> chunk_mean (float32)
    #   [2 * rows * num_chunks : 3 * rows * num_chunks) -> chunk_m2 (float32)
    #   [3 * rows * num_chunks : 3 * rows * num_chunks + 2 * rows) -> stats (mean, rstd)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _group_norm_reduce_cf_kernel(
        x_ptr,
        ws_ptr,
        channels,
        spatial_size,
        num_groups,
        group_size,
        chunk_size,
        num_chunks,
        rows,
        BLOCK: tl.constexpr,
    ):
        """Chunked shift-stabilized moments for contiguous NCHW/NCDHW tensors."""
        k = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)

        c_i64 = tl.cast(channels, tl.int64)
        s_i64 = tl.cast(spatial_size, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        gs_i64 = tl.cast(group_size, tl.int64)
        cs_i64 = tl.cast(chunk_size, tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)
        plane = tl.cast(rows, tl.int64) * nc_i64

        b_idx = row // g_i64
        g_idx = row - b_idx * g_i64
        group_base = b_idx * c_i64 * s_i64 + g_idx * gs_i64

        c_start = k * cs_i64
        c_end = tl.minimum(c_start + cs_i64, gs_i64)
        cnt = tl.maximum(c_end - c_start, 0)
        out_idx = row * nc_i64 + k
        if cnt == 0:
            tl.store(ws_ptr + out_idx, 0.0)
            tl.store(ws_ptr + plane + out_idx, 0.0)
            tl.store(ws_ptr + 2 * plane + out_idx, 0.0)
            return

        offs = tl.arange(0, BLOCK).to(tl.int64)
        idx0 = c_start + offs
        m0 = idx0 < c_end
        n0 = tl.sum(m0.to(tl.int64), axis=0)
        x0 = tl.load(x_ptr + group_base + idx0, mask=m0, other=0.0).to(tl.float32)
        shift = tl.sum(x0, axis=0) / tl.maximum(n0, 1).to(tl.float32)
        d0 = tl.where(m0, x0 - shift, 0.0)
        sum_d = tl.zeros((), dtype=tl.float32)
        sum_d2 = tl.sum(d0 * d0, axis=0)

        for off in range(BLOCK, chunk_size, BLOCK):
            idx = c_start + tl.cast(off, tl.int64) + offs
            m = idx < c_end
            xv = tl.load(x_ptr + group_base + idx, mask=m, other=0.0).to(tl.float32)
            diff = tl.where(m, xv - shift, 0.0)
            sum_d += tl.sum(diff, axis=0)
            sum_d2 += tl.sum(diff * diff, axis=0)

        cnt_f = cnt.to(tl.float32)
        mean_d = sum_d / cnt_f
        m2 = tl.maximum(sum_d2 - sum_d * mean_d, 0.0)
        tl.store(ws_ptr + out_idx, cnt_f)
        tl.store(ws_ptr + plane + out_idx, shift + mean_d)
        tl.store(ws_ptr + 2 * plane + out_idx, m2)

    @triton.jit
    def _group_norm_finalize_chan_kernel(
        ws_ptr,
        num_chunks,
        rows,
        eps,
        BLOCK_K: tl.constexpr,
    ):
        """Deterministic O(log K) parallel Chan variance merge with exact int64 counts."""
        row = tl.program_id(0).to(tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)
        plane = tl.cast(rows, tl.int64) * nc_i64
        base = row * nc_i64
        offs = tl.arange(0, BLOCK_K).to(tl.int64)

        tot_n = tl.zeros((), dtype=tl.int64)
        tot_m = tl.zeros((), dtype=tl.float32)
        tot_m2 = tl.zeros((), dtype=tl.float32)

        for k0 in range(0, num_chunks, BLOCK_K):
            k_idx = tl.cast(k0, tl.int64) + offs
            mask = k_idx < nc_i64
            nk = tl.load(ws_ptr + base + k_idx, mask=mask, other=0.0).to(tl.int64)
            mk = tl.load(ws_ptr + plane + base + k_idx, mask=mask, other=0.0)
            m2k = tl.load(ws_ptr + 2 * plane + base + k_idx, mask=mask, other=0.0)
            bn, bm, bm2 = tl.reduce((nk, mk, m2k), axis=0, combine_fn=_chan_combine_i64)
            tot_n, tot_m, tot_m2 = _chan_combine_i64(tot_n, tot_m, tot_m2, bn, bm, bm2)

        var = tl.maximum(tot_m2 / tl.maximum(tot_n, 1).to(tl.float32), 0.0)
        rstd = tl.rsqrt(var + eps)
        stats_base = 3 * plane + row * 2
        tl.store(ws_ptr + stats_base + 0, tot_m)
        tl.store(ws_ptr + stats_base + 1, rstd)

    @triton.jit
    def _group_norm_apply_cf_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ws_ptr,
        out_ptr,
        channels,
        spatial_size,
        num_groups,
        channels_per_group,
        stats_offset,
        BLOCK_S: tl.constexpr,
        TILES: tl.constexpr,
    ):
        """Per-channel scalar-affine apply + fast exp2 SiLU for NCHW/NCDHW."""
        s_tile = tl.program_id(0).to(tl.int64)
        bc = tl.program_id(1).to(tl.int64)

        c_i64 = tl.cast(channels, tl.int64)
        s_i64 = tl.cast(spatial_size, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        cg_i64 = tl.cast(channels_per_group, tl.int64)
        so_i64 = tl.cast(stats_offset, tl.int64)

        b_idx = bc // c_i64
        c_idx = bc - b_idx * c_i64
        bg = b_idx * g_i64 + (c_idx // cg_i64)

        mean = tl.load(ws_ptr + so_i64 + bg * 2 + 0)
        rstd = tl.load(ws_ptr + so_i64 + bg * 2 + 1)
        wv = tl.load(w_ptr + c_idx).to(tl.float32)
        bv = tl.load(b_ptr + c_idx).to(tl.float32)
        scale = rstd * wv

        base_s = s_tile * (BLOCK_S * TILES)
        lane = tl.arange(0, BLOCK_S).to(tl.int64)
        row_base = bc * s_i64
        LOG2E: tl.constexpr = 1.4426950408889634
        out_ty = out_ptr.dtype.element_ty

        for t in tl.static_range(TILES):
            s_offs = base_s + t * BLOCK_S + lane
            m = s_offs < s_i64
            ptrs = row_base + s_offs
            xv = tl.load(x_ptr + ptrs, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
            y = tl.math.fma(xv - mean, scale, bv)
            y = y / (1.0 + tl.math.exp2(-y * LOG2E))
            tl.store(out_ptr + ptrs, y.to(out_ty), mask=m, eviction_policy="evict_first")

    # ──────────────────────────────────────────────────────────────────────
    # Native Channels-Last (4D NHWC / 5D NDHWC) coalesced multi-group path
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _group_norm_reduce_cl_mg_kernel(
        x_ptr,
        ws_ptr,
        spatial_size,
        channels,
        num_groups,
        CG: tl.constexpr,
        chunk_s,
        num_chunks,
        rows,
        BLOCK_S: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,
    ):
        """Coalesced 128-bit multi-group CL/CL3D reduction (zero inner barriers)."""
        k = tl.program_id(0).to(tl.int64)
        bg_blk = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        cs_i64 = tl.cast(chunk_s, tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)
        plane = tl.cast(rows, tl.int64) * nc_i64
        g_blocks = (g_i64 + GROUPS_PER_CTA - 1) // GROUPS_PER_CTA

        b_idx = bg_blk // g_blocks
        gb_idx = bg_blk - b_idx * g_blocks
        g0 = gb_idx * GROUPS_PER_CTA
        c0 = g0 * CG

        s_start = k * cs_i64
        s_end = tl.minimum(s_start + cs_i64, s_i64)
        cnt_s = tl.maximum(s_end - s_start, 0)

        g_offs = tl.arange(0, GROUPS_PER_CTA).to(tl.int64)
        g_mask = (g0 + g_offs) < g_i64
        out_idx = (b_idx * g_i64 + g0 + g_offs) * nc_i64 + k

        if cnt_s == 0:
            z = tl.zeros((GROUPS_PER_CTA,), dtype=tl.float32)
            tl.store(ws_ptr + out_idx, z, mask=g_mask)
            tl.store(ws_ptr + plane + out_idx, z, mask=g_mask)
            tl.store(ws_ptr + 2 * plane + out_idx, z, mask=g_mask)
            return

        BLOCK_C: tl.constexpr = GROUPS_PER_CTA * CG
        c_offs = tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = (c0 + c_offs) < c_i64
        x_base = b_idx * s_i64 * c_i64 + c0
        s_lane = tl.arange(0, BLOCK_S).to(tl.int64)

        s_vec0 = s_start + s_lane
        s_mask0 = s_vec0 < s_end
        cnt_s0 = tl.sum(s_mask0.to(tl.int64), axis=0)
        m0 = s_mask0[:, None] & c_mask[None, :]
        x0 = tl.load(x_ptr + x_base + s_vec0[:, None] * c_i64 + c_offs[None, :], mask=m0, other=0.0).to(tl.float32)
        shift_g = tl.sum(tl.sum(tl.reshape(x0, (BLOCK_S, GROUPS_PER_CTA, CG)), axis=2), axis=0) / (
            tl.maximum(cnt_s0, 1).to(tl.float32) * CG
        )
        shift_bc = tl.reshape(tl.broadcast_to(shift_g[:, None], (GROUPS_PER_CTA, CG)), (1, BLOCK_C))

        d0 = tl.where(m0, x0 - shift_bc, 0.0)
        acc_d = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        acc_d2 = d0 * d0

        for s_off in range(BLOCK_S, chunk_s, BLOCK_S):
            s_vec = s_start + tl.cast(s_off, tl.int64) + s_lane
            s_mask = s_vec < s_end
            m = s_mask[:, None] & c_mask[None, :]
            xv = tl.load(x_ptr + x_base + s_vec[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            diff = tl.where(m, xv - shift_bc, 0.0)
            acc_d += diff
            acc_d2 += diff * diff

        sum_d = tl.sum(tl.sum(tl.reshape(acc_d, (BLOCK_S, GROUPS_PER_CTA, CG)), axis=2), axis=0)
        sum_d2 = tl.sum(tl.sum(tl.reshape(acc_d2, (BLOCK_S, GROUPS_PER_CTA, CG)), axis=2), axis=0)
        cnt_f = (cnt_s * CG).to(tl.float32)
        mean_d = sum_d / cnt_f
        m2 = tl.maximum(sum_d2 - sum_d * mean_d, 0.0)

        tl.store(ws_ptr + out_idx, tl.full((GROUPS_PER_CTA,), cnt_f, dtype=tl.float32), mask=g_mask)
        tl.store(ws_ptr + plane + out_idx, shift_g + mean_d, mask=g_mask)
        tl.store(ws_ptr + 2 * plane + out_idx, m2, mask=g_mask)

    @triton.jit
    def _group_norm_apply_cl_mg_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ws_ptr,
        out_ptr,
        spatial_size,
        channels,
        num_groups,
        stats_offset,
        CG: tl.constexpr,
        BLOCK_S: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,
    ):
        """Coalesced 128-bit multi-group CL/CL3D affine apply + fast exp2 SiLU."""
        s_tile = tl.program_id(0).to(tl.int64)
        bg_blk = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        so_i64 = tl.cast(stats_offset, tl.int64)
        g_blocks = (g_i64 + GROUPS_PER_CTA - 1) // GROUPS_PER_CTA

        b_idx = bg_blk // g_blocks
        gb_idx = bg_blk - b_idx * g_blocks
        g0 = gb_idx * GROUPS_PER_CTA
        c0 = g0 * CG

        g_offs = tl.arange(0, GROUPS_PER_CTA).to(tl.int64)
        g_mask = (g0 + g_offs) < g_i64
        in_bg = b_idx * g_i64 + g0 + g_offs

        mean_g = tl.load(ws_ptr + so_i64 + in_bg * 2 + 0, mask=g_mask, other=0.0)
        rstd_g = tl.load(ws_ptr + so_i64 + in_bg * 2 + 1, mask=g_mask, other=1.0)

        BLOCK_C: tl.constexpr = GROUPS_PER_CTA * CG
        c_offs = tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = (c0 + c_offs) < c_i64
        wv = tl.load(w_ptr + c0 + c_offs, mask=c_mask, other=1.0).to(tl.float32)
        bv = tl.load(b_ptr + c0 + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        mean_bc = tl.reshape(tl.broadcast_to(mean_g[:, None], (GROUPS_PER_CTA, CG)), (BLOCK_C,))
        rstd_bc = tl.reshape(tl.broadcast_to(rstd_g[:, None], (GROUPS_PER_CTA, CG)), (BLOCK_C,))
        scale = rstd_bc * wv

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64
        m = s_mask[:, None] & c_mask[None, :]
        ptrs = b_idx * s_i64 * c_i64 + s_offs[:, None] * c_i64 + (c0 + c_offs[None, :])
        xv = tl.load(x_ptr + ptrs, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        y = tl.math.fma(xv - mean_bc[None, :], scale[None, :], bv[None, :])
        LOG2E: tl.constexpr = 1.4426950408889634
        y = y / (1.0 + tl.math.exp2(-y * LOG2E))
        tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty), mask=m, eviction_policy="evict_first")

    # ──────────────────────────────────────────────────────────────────────
    # Native Channels-Last generic per-group path (non-pow2 or large CG)
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _group_norm_reduce_cl_generic_kernel(
        x_ptr,
        ws_ptr,
        spatial_size,
        channels,
        num_groups,
        channels_per_group,
        chunk_s,
        num_chunks,
        rows,
        BLOCK_S: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        """Per-group CL/CL3D reduction supporting arbitrary channels_per_group."""
        k = tl.program_id(0).to(tl.int64)
        bg = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        cg_i64 = tl.cast(channels_per_group, tl.int64)
        cs_i64 = tl.cast(chunk_s, tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)
        plane = tl.cast(rows, tl.int64) * nc_i64

        b_idx = bg // g_i64
        g_idx = bg - b_idx * g_i64
        c0 = g_idx * cg_i64

        s_start = k * cs_i64
        s_end = tl.minimum(s_start + cs_i64, s_i64)
        cnt_s = tl.maximum(s_end - s_start, 0)
        out_idx = bg * nc_i64 + k
        if cnt_s == 0:
            tl.store(ws_ptr + out_idx, 0.0)
            tl.store(ws_ptr + plane + out_idx, 0.0)
            tl.store(ws_ptr + 2 * plane + out_idx, 0.0)
            return

        x_base = b_idx * s_i64 * c_i64 + c0
        s_lane = tl.arange(0, BLOCK_S).to(tl.int64)
        c_lane = tl.arange(0, BLOCK_CG).to(tl.int64)

        tot_n = tl.zeros((), dtype=tl.int64)
        tot_m = tl.zeros((), dtype=tl.float32)
        tot_m2 = tl.zeros((), dtype=tl.float32)

        for c_off in range(0, channels_per_group, BLOCK_CG):
            c_vec = tl.cast(c_off, tl.int64) + c_lane
            c_mask = c_vec < cg_i64
            cnt_c = tl.sum(c_mask.to(tl.int64), axis=0)
            if cnt_c > 0:
                s_vec0 = s_start + s_lane
                s_mask0 = s_vec0 < s_end
                cnt_s0 = tl.sum(s_mask0.to(tl.int64), axis=0)
                m0 = s_mask0[:, None] & c_mask[None, :]
                x0 = tl.load(x_ptr + x_base + s_vec0[:, None] * c_i64 + c_vec[None, :], mask=m0, other=0.0).to(
                    tl.float32
                )
                shift = tl.sum(x0) / tl.maximum(cnt_s0 * cnt_c, 1).to(tl.float32)
                d0 = tl.where(m0, x0 - shift, 0.0)
                sum_d = tl.zeros((), dtype=tl.float32)
                sum_d2 = tl.sum(d0 * d0)

                for s_off in range(BLOCK_S, chunk_s, BLOCK_S):
                    s_vec = s_start + tl.cast(s_off, tl.int64) + s_lane
                    s_mask = s_vec < s_end
                    m = s_mask[:, None] & c_mask[None, :]
                    xv = tl.load(x_ptr + x_base + s_vec[:, None] * c_i64 + c_vec[None, :], mask=m, other=0.0).to(
                        tl.float32
                    )
                    diff = tl.where(m, xv - shift, 0.0)
                    sum_d += tl.sum(diff)
                    sum_d2 += tl.sum(diff * diff)

                n_sub = cnt_s * cnt_c
                mean_d = sum_d / n_sub.to(tl.float32)
                m2_sub = tl.maximum(sum_d2 - sum_d * mean_d, 0.0)
                tot_n, tot_m, tot_m2 = _chan_combine_i64(tot_n, tot_m, tot_m2, n_sub, shift + mean_d, m2_sub)

        tl.store(ws_ptr + out_idx, tot_n.to(tl.float32))
        tl.store(ws_ptr + plane + out_idx, tot_m)
        tl.store(ws_ptr + 2 * plane + out_idx, tot_m2)

    @triton.jit
    def _group_norm_apply_cl_generic_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ws_ptr,
        out_ptr,
        spatial_size,
        channels,
        num_groups,
        channels_per_group,
        stats_offset,
        BLOCK_S: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        """Per-group CL/CL3D affine apply + fast exp2 SiLU for arbitrary CG."""
        s_tile = tl.program_id(0).to(tl.int64)
        bg = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        g_i64 = tl.cast(num_groups, tl.int64)
        cg_i64 = tl.cast(channels_per_group, tl.int64)
        so_i64 = tl.cast(stats_offset, tl.int64)

        b_idx = bg // g_i64
        g_idx = bg - b_idx * g_i64
        c0 = g_idx * cg_i64

        mean = tl.load(ws_ptr + so_i64 + bg * 2 + 0)
        rstd = tl.load(ws_ptr + so_i64 + bg * 2 + 1)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64
        c_lane = tl.arange(0, BLOCK_CG).to(tl.int64)
        base = b_idx * s_i64 * c_i64 + c0
        LOG2E: tl.constexpr = 1.4426950408889634
        out_ty = out_ptr.dtype.element_ty

        for c_off in range(0, channels_per_group, BLOCK_CG):
            c_vec = tl.cast(c_off, tl.int64) + c_lane
            c_mask = c_vec < cg_i64
            m = s_mask[:, None] & c_mask[None, :]
            ptrs = base + s_offs[:, None] * c_i64 + c_vec[None, :]
            xv = tl.load(x_ptr + ptrs, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
            wv = tl.load(w_ptr + c0 + c_vec, mask=c_mask, other=1.0).to(tl.float32)
            bv = tl.load(b_ptr + c0 + c_vec, mask=c_mask, other=0.0).to(tl.float32)
            y = tl.math.fma(xv - mean, (rstd * wv)[None, :], bv[None, :])
            y = y / (1.0 + tl.math.exp2(-y * LOG2E))
            tl.store(out_ptr + ptrs, y.to(out_ty), mask=m, eviction_policy="evict_first")

    return {
        "triton": triton,
        "cf_1pass": _group_norm_silu_cf_1pass_kernel,
        "reduce_cf": _group_norm_reduce_cf_kernel,
        "finalize_chan": _group_norm_finalize_chan_kernel,
        "apply_cf": _group_norm_apply_cf_kernel,
        "reduce_cl_mg": _group_norm_reduce_cl_mg_kernel,
        "apply_cl_mg": _group_norm_apply_cl_mg_kernel,
        "reduce_cl_generic": _group_norm_reduce_cl_generic_kernel,
        "apply_cl_generic": _group_norm_apply_cl_generic_kernel,
    }


def _launch_cf(
    kernels: dict[str, Any],
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Launch contiguous NCHW/NCDHW GroupNorm + SiLU with Chan variance."""
    batch_size, channels = int(input.shape[0]), int(input.shape[1])
    spatial_size = math.prod(int(d) for d in input.shape[2:]) if input.ndim > 2 else 1
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial_size

    output = torch.empty_like(input)
    if output.numel() == 0:
        return output

    if group_size <= _ONE_PASS_CF_MAX_GROUP_SIZE:
        block = _next_power_of_2(max(64, group_size))
        num_warps = 8 if block >= 2048 else 4
        kernels["cf_1pass"][(num_groups, batch_size)](
            input,
            weight,
            bias,
            output,
            channels,
            spatial_size,
            channels_per_group,
            group_size,
            eps,
            BLOCK=block,
            num_warps=num_warps,
        )
        return output

    rows = batch_size * num_groups
    block = _CF_REDUCE_BLOCK if group_size >= _CF_REDUCE_BLOCK else _next_power_of_2(max(64, group_size))
    # Ensure chunk_size <= 2^24 (exact in float32) while saturating Hopper SMs.
    min_chunks_for_f32_cnt = _cdiv(group_size, 1 << 23)
    target_chunks = max(_cdiv(512, max(1, rows)), _cdiv(group_size, block * 2), min_chunks_for_f32_cnt)
    num_chunks = min(
        max(target_chunks, min_chunks_for_f32_cnt), max(4096, min_chunks_for_f32_cnt), max(1, _cdiv(group_size, block))
    )
    chunk_size = _cdiv(group_size, num_chunks)

    stats_offset = 3 * rows * num_chunks
    ws = torch.empty((stats_offset + 2 * rows,), device=input.device, dtype=torch.float32)

    kernels["reduce_cf"][(num_chunks, rows)](
        input,
        ws,
        channels,
        spatial_size,
        num_groups,
        group_size,
        chunk_size,
        num_chunks,
        rows,
        BLOCK=block,
        num_warps=4 if block >= 4096 else 8,
        num_stages=3,
    )

    block_k = min(1024, _next_power_of_2(max(16, num_chunks)))
    kernels["finalize_chan"][(rows,)](
        ws,
        num_chunks,
        rows,
        eps,
        BLOCK_K=block_k,
        num_warps=4,
    )

    apply_block = min(_CF_APPLY_BLOCK, _next_power_of_2(max(64, spatial_size)))
    apply_tiles = 2 if spatial_size >= apply_block * 2 else 1
    apply_warps = 4 if (apply_block >= 4096 and apply_tiles == 2) else (8 if apply_block >= 1024 else 4)
    kernels["apply_cf"][(_cdiv(spatial_size, apply_block * apply_tiles), batch_size * channels)](
        input,
        weight,
        bias,
        ws,
        output,
        channels,
        spatial_size,
        num_groups,
        channels_per_group,
        stats_offset,
        BLOCK_S=apply_block,
        TILES=apply_tiles,
        num_warps=apply_warps,
        num_stages=3,
    )
    return output


def _launch_cl(
    kernels: dict[str, Any],
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Launch native channels-last (NHWC / NDHWC) GroupNorm + SiLU."""
    batch_size, channels = int(input.shape[0]), int(input.shape[1])
    spatial_size = math.prod(int(d) for d in input.shape[2:])
    channels_per_group = channels // num_groups
    rows = batch_size * num_groups

    output = torch.empty_like(input)
    if output.numel() == 0:
        return output

    min_chunks_for_f32_cnt = _cdiv(spatial_size * channels_per_group, 1 << 23)

    if _is_power_of_2(channels_per_group) and channels_per_group <= 128:
        gpc = 1
        while gpc * 2 <= num_groups and (gpc * 2 * channels_per_group) <= 128 and num_groups % (gpc * 2) == 0:
            gpc *= 2
        g_blocks = num_groups // gpc
        block_c = gpc * channels_per_group
        block_s = max(16, min(64, 4096 // max(1, block_c)))
        target_chunks = max(
            _cdiv(512, max(1, batch_size * g_blocks)), _cdiv(spatial_size, block_s * 4), min_chunks_for_f32_cnt
        )
        num_chunks = min(
            max(target_chunks, min_chunks_for_f32_cnt),
            max(512, min_chunks_for_f32_cnt),
            max(1, _cdiv(spatial_size, block_s)),
        )
        chunk_s = _cdiv(spatial_size, num_chunks)

        stats_offset = 3 * rows * num_chunks
        ws = torch.empty((stats_offset + 2 * rows,), device=input.device, dtype=torch.float32)

        kernels["reduce_cl_mg"][(num_chunks, batch_size * g_blocks)](
            input,
            ws,
            spatial_size,
            channels,
            num_groups,
            CG=channels_per_group,
            chunk_s=chunk_s,
            num_chunks=num_chunks,
            rows=rows,
            BLOCK_S=block_s,
            GROUPS_PER_CTA=gpc,
            num_warps=8,
            num_stages=2,
        )
        block_k = min(1024, _next_power_of_2(max(16, num_chunks)))
        kernels["finalize_chan"][(rows,)](
            ws,
            num_chunks,
            rows,
            eps,
            BLOCK_K=block_k,
            num_warps=4,
        )
        kernels["apply_cl_mg"][(_cdiv(spatial_size, block_s), batch_size * g_blocks)](
            input,
            weight,
            bias,
            ws,
            output,
            spatial_size,
            channels,
            num_groups,
            stats_offset,
            CG=channels_per_group,
            BLOCK_S=block_s,
            GROUPS_PER_CTA=gpc,
            num_warps=8,
            num_stages=2,
        )
        return output

    block_cg = min(128, _next_power_of_2(max(1, channels_per_group)))
    block_s = max(16, min(128, 2048 // max(1, block_cg)))
    target_chunks = max(_cdiv(512, max(1, rows)), _cdiv(spatial_size, block_s * 4), min_chunks_for_f32_cnt)
    num_chunks = min(
        max(target_chunks, min_chunks_for_f32_cnt),
        max(512, min_chunks_for_f32_cnt),
        max(1, _cdiv(spatial_size, block_s)),
    )
    chunk_s = _cdiv(spatial_size, num_chunks)

    stats_offset = 3 * rows * num_chunks
    ws = torch.empty((stats_offset + 2 * rows,), device=input.device, dtype=torch.float32)
    num_warps = 8 if block_s * block_cg >= 2048 else 4

    kernels["reduce_cl_generic"][(num_chunks, rows)](
        input,
        ws,
        spatial_size,
        channels,
        num_groups,
        channels_per_group,
        chunk_s,
        num_chunks,
        rows,
        BLOCK_S=block_s,
        BLOCK_CG=block_cg,
        num_warps=num_warps,
        num_stages=2,
    )
    block_k = min(1024, _next_power_of_2(max(16, num_chunks)))
    kernels["finalize_chan"][(rows,)](
        ws,
        num_chunks,
        rows,
        eps,
        BLOCK_K=block_k,
        num_warps=4,
    )
    kernels["apply_cl_generic"][(_cdiv(spatial_size, block_s), rows)](
        input,
        weight,
        bias,
        ws,
        output,
        spatial_size,
        channels,
        num_groups,
        channels_per_group,
        stats_offset,
        BLOCK_S=block_s,
        BLOCK_CG=block_cg,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


@torch.library.custom_op("worldfoundry::core_group_norm_silu_fwd", mutates_args=(), device_types="cuda")
def _core_group_norm_silu_fwd_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    b_c = bias if bias.is_contiguous() else bias.contiguous()
    with torch.cuda.device(input.device):
        if _is_channels_last_nd(input):
            return _launch_cl(kernels, input, w_c, b_c, int(num_groups), float(eps))
        x_c = input if input.is_contiguous() else input.contiguous()
        return _launch_cf(kernels, x_c, w_c, b_c, int(num_groups), float(eps))


@_core_group_norm_silu_fwd_op.register_fake
def _core_group_norm_silu_fwd_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    del weight, bias, num_groups, eps
    return torch.empty_like(input)


def group_norm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Launch the in-tree lossless GroupNorm + SiLU Triton custom op."""
    return _core_group_norm_silu_fwd_op(input, weight, bias, int(num_groups), float(eps))


__all__ = ["group_norm_silu"]
