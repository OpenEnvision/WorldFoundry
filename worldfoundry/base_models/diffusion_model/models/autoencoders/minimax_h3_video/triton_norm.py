# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper Triton normalization for the MiniMax H3 video VAE.

Implements :func:`triton_temporal_isolated_group_norm_act`, a compile-safe,
CUDA-Graph-safe custom operator executing per-frame (or full 3-D) GroupNorm +
optional learned affine + optional nearest-upsampled dynamic spatial affine +
activation (``identity`` / ``silu`` / ``relu``) in a single fused pipeline
without ``_merge_time_to_batch`` or ``_split_time_from_batch`` layout copies.

Supports both ``NCDHW`` contiguous and ``NDHWC`` (``channels_last_3d``) inputs,
with lazy Triton import and exact PyTorch CPU/autograd fallback.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ACT_TO_CODE = {"identity": 0, "silu": 1, "relu": 2}
_TRITON_KERNELS: dict[str, Any] | None = None
_TRITON_LOAD_FAILED: bool = False


def _is_channels_last_3d(x: torch.Tensor) -> bool:
    if x.ndim != 5:
        return False
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        if not x.is_contiguous():
            return True
        return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
    return False


def _is_batch_strided_ndhwc(x: torch.Tensor) -> bool:
    if x.ndim != 5:
        return False
    _, c, _, h, w = x.shape
    return x.stride(1) == 1 and x.stride(4) == c and x.stride(3) == w * c and x.stride(2) == h * w * c


def _empty_like_layout(x: torch.Tensor) -> torch.Tensor:
    if _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x):
        return torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _effective_5d_strides(t: torch.Tensor | None) -> tuple[int, int, int, int, int, int, int, int]:
    """Return ``(D, H, W, sN, sC, sD, sH, sW)`` with broadcast dims zeroed."""
    if t is None:
        return (1, 1, 1, 0, 0, 0, 0, 0)
    b, c, d, h, w = t.shape
    sn, sc, sd, sh, sw = t.stride()
    return (
        d,
        h,
        w,
        0 if b == 1 else sn,
        0 if c == 1 else sc,
        0 if d == 1 else sd,
        0 if h == 1 else sh,
        0 if w == 1 else sw,
    )


def _get_triton_kernels() -> dict[str, Any] | None:
    global _TRITON_KERNELS, _TRITON_LOAD_FAILED
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    if _TRITON_LOAD_FAILED:
        return None
    try:
        from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="minimax-h3-vae-triton")
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_LOAD_FAILED = True
        return None

    @triton.jit
    def _apply_act(v, ACT_CODE: tl.constexpr):
        if ACT_CODE == 1:
            return v * tl.sigmoid(v)
        if ACT_CODE == 2:
            return tl.maximum(v, 0.0)
        return v

    # =========================================================================
    # Shared Stage-2 Finalizer (Chan parallel variance)
    # =========================================================================

    @triton.jit
    def _tign_finalize_stats_kernel(
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        mean_ptr,
        rstd_ptr,
        num_chunks,
        eps,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * num_chunks
        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
        mask = k_offs < num_chunks

        mu = tl.load(p_mean_ptr + base + k_offs, mask=mask, other=0.0)
        m2 = tl.load(p_m2_ptr + base + k_offs, mask=mask, other=0.0)
        cnt = tl.load(p_cnt_ptr + base + k_offs, mask=mask, other=0.0)

        total_n = tl.maximum(tl.sum(cnt, axis=0), 1.0)
        global_mean = tl.sum(cnt * mu, axis=0) / total_n
        diff = tl.where(cnt > 0.0, mu - global_mean, 0.0)
        global_m2 = tl.sum(m2 + cnt * diff * diff, axis=0)
        var = tl.maximum(global_m2 / total_n, 0.0)
        rstd = tl.rsqrt(var + eps)

        tl.store(mean_ptr + row, global_mean)
        tl.store(rstd_ptr + row, rstd)

    # =========================================================================
    # NDHWC (channels_last_3d): per-(b,t) frame is contiguous (H*W, C)!
    # =========================================================================

    @triton.jit
    def _tign_ndhwc_pow2_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        stride_xn,
        stride_xt,
        T_eff,
        S_eff,
        C,
        G,
        s_per_chunk,
        num_chunks,
        CG: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        ntg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        nt_id = ntg // num_g_tiles
        g_tile = ntg - nt_id * num_g_tiles
        b_id = nt_id // T_eff
        t_id = nt_id - b_id * T_eff

        g_start = g_tile * BLOCK_G
        c_start = g_start * CG
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        g_mask = g_offs < G
        c_mask = c_offs < C

        chunk_s_start = chunk_id * s_per_chunk
        chunk_s_end = tl.minimum(chunk_s_start + s_per_chunk, S_eff)
        out_idx = (nt_id * G + g_offs) * num_chunks + chunk_id

        if chunk_s_start >= S_eff:
            zeros = tl.zeros((BLOCK_G,), dtype=tl.float32)
            tl.store(p_mean_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_m2_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_cnt_ptr + out_idx, zeros, mask=g_mask)
            return

        x_base = x_ptr + b_id * stride_xn + t_id * stride_xt
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        s0 = chunk_s_start + s_base_offs
        m0 = (s0[:, None] < chunk_s_end) & c_mask[None, :]
        x0 = tl.load(x_base + s0[:, None] * C + c_offs[None, :], mask=m0, other=0.0).to(tl.float32)
        n0_vox = tl.minimum(chunk_s_end - chunk_s_start, BLOCK_S).to(tl.float32)
        n0_elems = tl.maximum(n0_vox * CG, 1.0)
        x0_3d = tl.reshape(x0, (BLOCK_S, BLOCK_G, CG))
        mu0 = tl.sum(tl.sum(x0_3d, axis=2), axis=0) / n0_elems

        m0_3d = tl.reshape(m0, (BLOCK_S, BLOCK_G, CG))
        d0 = tl.where(m0_3d, x0_3d - mu0[None, :, None], 0.0)
        sum_d = tl.sum(tl.sum(d0, axis=2), axis=0)
        sum_d2 = tl.sum(tl.sum(d0 * d0, axis=2), axis=0)

        for s_start in range(chunk_s_start + BLOCK_S, chunk_s_end, BLOCK_S):
            s_offs = s_start + s_base_offs
            m = (s_offs[:, None] < chunk_s_end) & c_mask[None, :]
            xv = tl.load(x_base + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            xv_3d = tl.reshape(xv, (BLOCK_S, BLOCK_G, CG))
            m_3d = tl.reshape(m, (BLOCK_S, BLOCK_G, CG))
            d = tl.where(m_3d, xv_3d - mu0[None, :, None], 0.0)
            sum_d += tl.sum(tl.sum(d, axis=2), axis=0)
            sum_d2 += tl.sum(tl.sum(d * d, axis=2), axis=0)

        n_elems = ((chunk_s_end - chunk_s_start) * CG).to(tl.float32)
        delta = sum_d / n_elems
        chunk_mean = mu0 + delta
        chunk_m2 = tl.maximum(sum_d2 - n_elems * delta * delta, 0.0)

        tl.store(p_mean_ptr + out_idx, chunk_mean, mask=g_mask)
        tl.store(p_m2_ptr + out_idx, chunk_m2, mask=g_mask)
        tl.store(p_cnt_ptr + out_idx, tl.full((BLOCK_G,), n_elems, dtype=tl.float32), mask=g_mask)

    @triton.jit
    def _tign_ndhwc_pow2_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ds_ptr,
        db_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        stride_xn,
        stride_xt,
        stride_yn,
        stride_yt,
        T,
        H,
        W,
        T_eff,
        S_eff,
        C,
        G,
        Ds,
        Hs,
        Ws,
        ss_n,
        ss_c,
        ss_d,
        ss_h,
        ss_w,
        Db,
        Hb,
        Wb,
        sb_n,
        sb_c,
        sb_d,
        sb_h,
        sb_w,
        CG: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DYNAMIC: tl.constexpr,
        SAME_DYN_SHAPE: tl.constexpr,
        TEMPORAL_ISOLATED: tl.constexpr,
        ACT_CODE: tl.constexpr,
        BLOCK_S: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        ntg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        nt_id = ntg // num_g_tiles
        g_tile = ntg - nt_id * num_g_tiles
        b_id = nt_id // T_eff
        t_id = nt_id - b_id * T_eff

        g_start = g_tile * BLOCK_G
        c_start = g_start * CG
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        g_mask = g_offs < G
        c_mask = c_offs < C

        stats_offs = nt_id * G + g_offs
        mean = tl.load(mean_ptr + stats_offs, mask=g_mask, other=0.0)
        rstd = tl.load(rstd_ptr + stats_offs, mask=g_mask, other=1.0)
        mean_c = tl.reshape(tl.broadcast_to(mean[:, None], (BLOCK_G, CG)), (BLOCK_C,))
        rstd_c = tl.reshape(tl.broadcast_to(rstd[:, None], (BLOCK_G, CG)), (BLOCK_C,))

        if HAS_WEIGHT:
            w = tl.load(w_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        x_base = x_ptr + b_id * stride_xn + t_id * stride_xt
        y_base = y_ptr + b_id * stride_yn + t_id * stride_yt
        s_base_i32 = tl.arange(0, BLOCK_S)
        s_base_offs = s_base_i32.to(tl.int64)
        cta_s_start = s_tile * (BLOCK_S * APPLY_ITERS)
        hw = H * W
        if HAS_DYNAMIC and TEMPORAL_ISOLATED:
            ds_d_off = ((t_id * Ds) // T) * ss_d
            db_d_off = ((t_id * Db) // T) * sb_d

        for it in tl.static_range(APPLY_ITERS):
            s_offs = cta_s_start + it * BLOCK_S + s_base_offs
            m = (s_offs[:, None] < S_eff) & c_mask[None, :]
            xv = tl.load(x_base + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean_c[None, :]) * rstd_c[None, :]
            if HAS_WEIGHT:
                yv = yv * w[None, :]
            if HAS_BIAS:
                yv = yv + b[None, :]
            if HAS_DYNAMIC:
                s_i32 = s_offs.to(tl.int32)
                if TEMPORAL_ISOLATED:
                    h_coord = s_i32 // W
                    w_coord = s_i32 - h_coord * W
                    ds_h = ((h_coord * Hs) // H).to(tl.int64)
                    ds_w = ((w_coord * Ws) // W).to(tl.int64)
                    ds_offs = b_id * ss_n + c_offs[None, :] * ss_c + (ds_d_off + ds_h * ss_h + ds_w * ss_w)[:, None]
                    if SAME_DYN_SHAPE:
                        db_offs = b_id * sb_n + c_offs[None, :] * sb_c + (db_d_off + ds_h * sb_h + ds_w * sb_w)[:, None]
                    else:
                        db_h = ((h_coord * Hb) // H).to(tl.int64)
                        db_w = ((w_coord * Wb) // W).to(tl.int64)
                        db_offs = b_id * sb_n + c_offs[None, :] * sb_c + (db_d_off + db_h * sb_h + db_w * sb_w)[:, None]
                else:
                    d_coord = s_i32 // hw
                    rem = s_i32 - d_coord * hw
                    h_coord = rem // W
                    w_coord = rem - h_coord * W
                    ds_d = ((d_coord * Ds) // T).to(tl.int64)
                    ds_h = ((h_coord * Hs) // H).to(tl.int64)
                    ds_w = ((w_coord * Ws) // W).to(tl.int64)
                    ds_offs = b_id * ss_n + c_offs[None, :] * ss_c + (ds_d * ss_d + ds_h * ss_h + ds_w * ss_w)[:, None]
                    if SAME_DYN_SHAPE:
                        db_offs = (
                            b_id * sb_n + c_offs[None, :] * sb_c + (ds_d * sb_d + ds_h * sb_h + ds_w * sb_w)[:, None]
                        )
                    else:
                        db_d = ((d_coord * Db) // T).to(tl.int64)
                        db_h = ((h_coord * Hb) // H).to(tl.int64)
                        db_w = ((w_coord * Wb) // W).to(tl.int64)
                        db_offs = (
                            b_id * sb_n + c_offs[None, :] * sb_c + (db_d * sb_d + db_h * sb_h + db_w * sb_w)[:, None]
                        )
                dyn_s = tl.load(ds_ptr + ds_offs, mask=m, other=1.0).to(tl.float32)
                dyn_b = tl.load(db_ptr + db_offs, mask=m, other=0.0).to(tl.float32)
                yv = yv * dyn_s + dyn_b

            yv = _apply_act(yv, ACT_CODE)
            tl.store(y_base + s_offs[:, None] * C + c_offs[None, :], yv.to(y_ptr.dtype.element_ty), mask=m)

    # Non-pow2 CG NDHWC stats & apply
    @triton.jit
    def _tign_ndhwc_gen_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        stride_xn,
        stride_xt,
        T_eff,
        S_eff,
        C,
        G,
        CG,
        s_per_chunk,
        num_chunks,
        BLOCK_S: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        ntg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        nt_id = ntg // num_g_tiles
        g_tile = ntg - nt_id * num_g_tiles
        b_id = nt_id // T_eff
        t_id = nt_id - b_id * T_eff

        g_start = g_tile * BLOCK_G
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        cg_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        g_mask = g_offs < G
        ch_offs = g_offs[:, None] * CG + cg_offs[None, :]
        ch_mask = g_mask[:, None] & (cg_offs[None, :] < CG)

        chunk_s_start = chunk_id * s_per_chunk
        chunk_s_end = tl.minimum(chunk_s_start + s_per_chunk, S_eff)
        out_idx = (nt_id * G + g_offs) * num_chunks + chunk_id

        if chunk_s_start >= S_eff:
            zeros = tl.zeros((BLOCK_G,), dtype=tl.float32)
            tl.store(p_mean_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_m2_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_cnt_ptr + out_idx, zeros, mask=g_mask)
            return

        x_base = x_ptr + b_id * stride_xn + t_id * stride_xt
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        s0 = chunk_s_start + s_base_offs
        m0 = (s0[:, None, None] < chunk_s_end) & ch_mask[None, :, :]
        x0 = tl.load(x_base + s0[:, None, None] * C + ch_offs[None, :, :], mask=m0, other=0.0).to(tl.float32)
        n0_elems = tl.maximum(tl.minimum(chunk_s_end - chunk_s_start, BLOCK_S).to(tl.float32) * CG, 1.0)
        mu0 = tl.sum(tl.sum(x0, axis=2), axis=0) / n0_elems

        d0 = tl.where(m0, x0 - mu0[None, :, None], 0.0)
        sum_d = tl.sum(tl.sum(d0, axis=2), axis=0)
        sum_d2 = tl.sum(tl.sum(d0 * d0, axis=2), axis=0)

        for s_start in range(chunk_s_start + BLOCK_S, chunk_s_end, BLOCK_S):
            s_offs = s_start + s_base_offs
            m = (s_offs[:, None, None] < chunk_s_end) & ch_mask[None, :, :]
            xv = tl.load(x_base + s_offs[:, None, None] * C + ch_offs[None, :, :], mask=m, other=0.0).to(tl.float32)
            d = tl.where(m, xv - mu0[None, :, None], 0.0)
            sum_d += tl.sum(tl.sum(d, axis=2), axis=0)
            sum_d2 += tl.sum(tl.sum(d * d, axis=2), axis=0)

        n_elems = ((chunk_s_end - chunk_s_start) * CG).to(tl.float32)
        delta = sum_d / n_elems
        chunk_mean = mu0 + delta
        chunk_m2 = tl.maximum(sum_d2 - n_elems * delta * delta, 0.0)
        tl.store(p_mean_ptr + out_idx, chunk_mean, mask=g_mask)
        tl.store(p_m2_ptr + out_idx, chunk_m2, mask=g_mask)
        tl.store(p_cnt_ptr + out_idx, tl.full((BLOCK_G,), n_elems, dtype=tl.float32), mask=g_mask)

    @triton.jit
    def _tign_ndhwc_gen_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ds_ptr,
        db_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        stride_xn,
        stride_xt,
        stride_yn,
        stride_yt,
        T,
        H,
        W,
        T_eff,
        S_eff,
        C,
        G,
        CG,
        Ds,
        Hs,
        Ws,
        ss_n,
        ss_c,
        ss_d,
        ss_h,
        ss_w,
        Db,
        Hb,
        Wb,
        sb_n,
        sb_c,
        sb_d,
        sb_h,
        sb_w,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DYNAMIC: tl.constexpr,
        TEMPORAL_ISOLATED: tl.constexpr,
        ACT_CODE: tl.constexpr,
        BLOCK_S: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        ntg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        nt_id = ntg // num_g_tiles
        g_tile = ntg - nt_id * num_g_tiles
        b_id = nt_id // T_eff
        t_id = nt_id - b_id * T_eff

        g_start = g_tile * BLOCK_G
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        cg_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        g_mask = g_offs < G
        ch_offs = g_offs[:, None] * CG + cg_offs[None, :]
        ch_mask = g_mask[:, None] & (cg_offs[None, :] < CG)

        stats_offs = nt_id * G + g_offs
        mean = tl.load(mean_ptr + stats_offs, mask=g_mask, other=0.0)
        rstd = tl.load(rstd_ptr + stats_offs, mask=g_mask, other=1.0)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + ch_offs, mask=ch_mask, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + ch_offs, mask=ch_mask, other=0.0).to(tl.float32)

        x_base = x_ptr + b_id * stride_xn + t_id * stride_xt
        y_base = y_ptr + b_id * stride_yn + t_id * stride_yt
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)
        cta_s_start = s_tile * (BLOCK_S * APPLY_ITERS)
        hw = H * W

        for it in tl.static_range(APPLY_ITERS):
            s_offs = cta_s_start + it * BLOCK_S + s_base_offs
            m = (s_offs[:, None, None] < S_eff) & ch_mask[None, :, :]
            xv = tl.load(x_base + s_offs[:, None, None] * C + ch_offs[None, :, :], mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean[None, :, None]) * rstd[None, :, None]
            if HAS_WEIGHT:
                yv = yv * w[None, :, :]
            if HAS_BIAS:
                yv = yv + b[None, :, :]
            if HAS_DYNAMIC:
                if TEMPORAL_ISOLATED:
                    d_coord = t_id
                    h_coord = s_offs // W
                    w_coord = s_offs - h_coord * W
                else:
                    d_coord = s_offs // hw
                    rem = s_offs - d_coord * hw
                    h_coord = rem // W
                    w_coord = rem - h_coord * W
                ds_d = (d_coord * Ds) // T
                ds_h = (h_coord * Hs) // H
                ds_w = (w_coord * Ws) // W
                ds_offs = (
                    b_id * ss_n + ch_offs[None, :, :] * ss_c + (ds_d * ss_d + ds_h * ss_h + ds_w * ss_w)[:, None, None]
                )
                dyn_s = tl.load(ds_ptr + ds_offs, mask=m, other=1.0).to(tl.float32)

                db_d = (d_coord * Db) // T
                db_h = (h_coord * Hb) // H
                db_w = (w_coord * Wb) // W
                db_offs = (
                    b_id * sb_n + ch_offs[None, :, :] * sb_c + (db_d * sb_d + db_h * sb_h + db_w * sb_w)[:, None, None]
                )
                dyn_b = tl.load(db_ptr + db_offs, mask=m, other=0.0).to(tl.float32)
                yv = yv * dyn_s + dyn_b

            yv = _apply_act(yv, ACT_CODE)
            tl.store(
                y_base + s_offs[:, None, None] * C + ch_offs[None, :, :],
                yv.to(y_ptr.dtype.element_ty),
                mask=m,
            )

    # =========================================================================
    # NCDHW Contiguous — Temporal-Isolated (stride_c = T*HW, contiguous HW!)
    # =========================================================================

    @triton.jit
    def _tign_ncdhw_onepass_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ds_ptr,
        db_ptr,
        y_ptr,
        C,
        T,
        H,
        W,
        G,
        CG,
        eps,
        Ds,
        Hs,
        Ws,
        ss_n,
        ss_c,
        ss_d,
        ss_h,
        ss_w,
        Db,
        Hb,
        Wb,
        sb_n,
        sb_c,
        sb_d,
        sb_h,
        sb_w,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DYNAMIC: tl.constexpr,
        TEMPORAL_ISOLATED: tl.constexpr,
        ACT_CODE: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        # row indexes (b, t, g) when TEMPORAL_ISOLATED else (b, g)
        row = tl.program_id(0).to(tl.int64)
        hw = H * W
        s_len = hw if TEMPORAL_ISOLATED else T * hw
        if TEMPORAL_ISOLATED:
            g_id = row % G
            bt = row // G
            t_id = bt % T
            b_id = bt // T
        else:
            g_id = row % G
            b_id = row // G
            t_id = 0

        stride_c = tl.cast(T, tl.int64) * hw
        stride_n = tl.cast(C, tl.int64) * stride_c
        c_start = g_id * CG
        base_bt = b_id * stride_n + t_id * hw
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        # Shift by first tile of channel 0
        m0 = s_base_offs < s_len
        x0 = tl.load(x_ptr + base_bt + c_start * stride_c + s_base_offs, mask=m0, other=0.0).to(tl.float32)
        n0 = tl.maximum(tl.minimum(s_len, BLOCK_S).to(tl.float32), 1.0)
        mu0 = tl.sum(x0, axis=0) / n0

        sum_d = tl.zeros((), dtype=tl.float32)
        sum_d2 = tl.zeros((), dtype=tl.float32)
        for cg in range(CG):
            c_base = base_bt + (c_start + cg) * stride_c
            for s_start in range(0, s_len, BLOCK_S):
                s_offs = s_start + s_base_offs
                m = s_offs < s_len
                xv = tl.load(x_ptr + c_base + s_offs, mask=m, other=0.0).to(tl.float32)
                d = tl.where(m, xv - mu0, 0.0)
                sum_d += tl.sum(d, axis=0)
                sum_d2 += tl.sum(d * d, axis=0)

        total_n = (CG * s_len).to(tl.float32)
        delta = sum_d / total_n
        mean = mu0 + delta
        var = tl.maximum(sum_d2 / total_n - delta * delta, 0.0)
        rstd = tl.rsqrt(var + eps)

        for cg in range(CG):
            ch = c_start + cg
            c_base = base_bt + ch * stride_c
            if HAS_WEIGHT:
                w_s = tl.load(w_ptr + ch).to(tl.float32)
            if HAS_BIAS:
                b_s = tl.load(b_ptr + ch).to(tl.float32)
            for s_start in range(0, s_len, BLOCK_S):
                s_offs = s_start + s_base_offs
                m = s_offs < s_len
                xv = tl.load(x_ptr + c_base + s_offs, mask=m, other=0.0).to(tl.float32)
                yv = (xv - mean) * rstd
                if HAS_WEIGHT:
                    yv = yv * w_s
                if HAS_BIAS:
                    yv = yv + b_s
                if HAS_DYNAMIC:
                    if TEMPORAL_ISOLATED:
                        d_coord = t_id
                        h_coord = s_offs // W
                        w_coord = s_offs - h_coord * W
                    else:
                        d_coord = s_offs // hw
                        rem = s_offs - d_coord * hw
                        h_coord = rem // W
                        w_coord = rem - h_coord * W
                    ds_offs = (
                        b_id * ss_n
                        + ch * ss_c
                        + ((d_coord * Ds) // T) * ss_d
                        + ((h_coord * Hs) // H) * ss_h
                        + ((w_coord * Ws) // W) * ss_w
                    )
                    db_offs = (
                        b_id * sb_n
                        + ch * sb_c
                        + ((d_coord * Db) // T) * sb_d
                        + ((h_coord * Hb) // H) * sb_h
                        + ((w_coord * Wb) // W) * sb_w
                    )
                    dyn_s = tl.load(ds_ptr + ds_offs, mask=m, other=1.0).to(tl.float32)
                    dyn_b = tl.load(db_ptr + db_offs, mask=m, other=0.0).to(tl.float32)
                    yv = yv * dyn_s + dyn_b
                yv = _apply_act(yv, ACT_CODE)
                tl.store(y_ptr + c_base + s_offs, yv.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _tign_ncdhw_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        C,
        T,
        H,
        W,
        G,
        CG,
        s_per_chunk,
        num_chunks,
        TEMPORAL_ISOLATED: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        hw = H * W
        s_len = hw if TEMPORAL_ISOLATED else T * hw

        if TEMPORAL_ISOLATED:
            g_id = row % G
            bt = row // G
            t_id = bt % T
            b_id = bt // T
        else:
            g_id = row % G
            b_id = row // G
            t_id = 0

        chunk_s_start = chunk_id * s_per_chunk
        chunk_s_end = tl.minimum(chunk_s_start + s_per_chunk, s_len)
        out_idx = row * num_chunks + chunk_id

        if chunk_s_start >= s_len:
            tl.store(p_mean_ptr + out_idx, 0.0)
            tl.store(p_m2_ptr + out_idx, 0.0)
            tl.store(p_cnt_ptr + out_idx, 0.0)
            return

        stride_c = tl.cast(T, tl.int64) * hw
        stride_n = tl.cast(C, tl.int64) * stride_c
        c_start = g_id * CG
        base_bt = b_id * stride_n + t_id * hw
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        s0 = chunk_s_start + s_base_offs
        m0 = s0 < chunk_s_end
        x0 = tl.load(x_ptr + base_bt + c_start * stride_c + s0, mask=m0, other=0.0).to(tl.float32)
        n0 = tl.maximum(tl.minimum(chunk_s_end - chunk_s_start, BLOCK_S).to(tl.float32), 1.0)
        mu0 = tl.sum(x0, axis=0) / n0

        sum_d = tl.zeros((), dtype=tl.float32)
        sum_d2 = tl.zeros((), dtype=tl.float32)
        for cg in range(CG):
            c_base = base_bt + (c_start + cg) * stride_c
            for s_start in range(chunk_s_start, chunk_s_end, BLOCK_S):
                s_offs = s_start + s_base_offs
                m = s_offs < chunk_s_end
                xv = tl.load(x_ptr + c_base + s_offs, mask=m, other=0.0).to(tl.float32)
                d = tl.where(m, xv - mu0, 0.0)
                sum_d += tl.sum(d, axis=0)
                sum_d2 += tl.sum(d * d, axis=0)

        n_elems = ((chunk_s_end - chunk_s_start) * CG).to(tl.float32)
        delta = sum_d / n_elems
        chunk_mean = mu0 + delta
        chunk_m2 = tl.maximum(sum_d2 - n_elems * delta * delta, 0.0)

        tl.store(p_mean_ptr + out_idx, chunk_mean)
        tl.store(p_m2_ptr + out_idx, chunk_m2)
        tl.store(p_cnt_ptr + out_idx, n_elems)

    @triton.jit
    def _tign_ncdhw_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        ds_ptr,
        db_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        C,
        T,
        H,
        W,
        G,
        CG,
        Ds,
        Hs,
        Ws,
        ss_n,
        ss_c,
        ss_d,
        ss_h,
        ss_w,
        Db,
        Hb,
        Wb,
        sb_n,
        sb_c,
        sb_d,
        sb_h,
        sb_w,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DYNAMIC: tl.constexpr,
        SAME_DYN_SHAPE: tl.constexpr,
        TEMPORAL_ISOLATED: tl.constexpr,
        ACT_CODE: tl.constexpr,
        BLOCK_S: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        hw = H * W
        s_len = hw if TEMPORAL_ISOLATED else T * hw

        if TEMPORAL_ISOLATED:
            g_id = row % G
            bt = row // G
            t_id = bt % T
            b_id = bt // T
        else:
            g_id = row % G
            b_id = row // G
            t_id = 0

        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        stride_c = tl.cast(T, tl.int64) * hw
        stride_n = tl.cast(C, tl.int64) * stride_c
        c_start = g_id * CG
        base_bt = b_id * stride_n + t_id * hw
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)
        cta_s_start = s_tile * (BLOCK_S * APPLY_ITERS)
        if HAS_DYNAMIC and TEMPORAL_ISOLATED:
            ds_d_off = ((t_id * Ds) // T) * ss_d
            db_d_off = ((t_id * Db) // T) * sb_d

        for it in tl.static_range(APPLY_ITERS):
            s_offs = cta_s_start + it * BLOCK_S + s_base_offs
            m = s_offs < s_len
            if HAS_DYNAMIC:
                s_i32 = s_offs.to(tl.int32)
                if TEMPORAL_ISOLATED:
                    h_coord = s_i32 // W
                    w_coord = s_i32 - h_coord * W
                    ds_h = ((h_coord * Hs) // H).to(tl.int64)
                    ds_w = ((w_coord * Ws) // W).to(tl.int64)
                    ds_spatial = b_id * ss_n + ds_d_off + ds_h * ss_h + ds_w * ss_w
                    if SAME_DYN_SHAPE:
                        db_spatial = b_id * sb_n + db_d_off + ds_h * sb_h + ds_w * sb_w
                    else:
                        db_h = ((h_coord * Hb) // H).to(tl.int64)
                        db_w = ((w_coord * Wb) // W).to(tl.int64)
                        db_spatial = b_id * sb_n + db_d_off + db_h * sb_h + db_w * sb_w
                else:
                    d_coord = s_i32 // hw
                    rem = s_i32 - d_coord * hw
                    h_coord = rem // W
                    w_coord = rem - h_coord * W
                    ds_d = ((d_coord * Ds) // T).to(tl.int64)
                    ds_h = ((h_coord * Hs) // H).to(tl.int64)
                    ds_w = ((w_coord * Ws) // W).to(tl.int64)
                    ds_spatial = b_id * ss_n + ds_d * ss_d + ds_h * ss_h + ds_w * ss_w
                    if SAME_DYN_SHAPE:
                        db_spatial = b_id * sb_n + ds_d * sb_d + ds_h * sb_h + ds_w * sb_w
                    else:
                        db_d = ((d_coord * Db) // T).to(tl.int64)
                        db_h = ((h_coord * Hb) // H).to(tl.int64)
                        db_w = ((w_coord * Wb) // W).to(tl.int64)
                        db_spatial = b_id * sb_n + db_d * sb_d + db_h * sb_h + db_w * sb_w

            for cg in range(CG):
                ch = c_start + cg
                c_base = base_bt + ch * stride_c
                xv = tl.load(x_ptr + c_base + s_offs, mask=m, other=0.0).to(tl.float32)
                yv = (xv - mean) * rstd
                if HAS_WEIGHT:
                    w_s = tl.load(w_ptr + ch).to(tl.float32)
                    yv = yv * w_s
                if HAS_BIAS:
                    b_s = tl.load(b_ptr + ch).to(tl.float32)
                    yv = yv + b_s
                if HAS_DYNAMIC:
                    dyn_s = tl.load(ds_ptr + ch * ss_c + ds_spatial, mask=m, other=1.0).to(tl.float32)
                    dyn_b = tl.load(db_ptr + ch * sb_c + db_spatial, mask=m, other=0.0).to(tl.float32)
                    yv = yv * dyn_s + dyn_b
                yv = _apply_act(yv, ACT_CODE)
                tl.store(y_ptr + c_base + s_offs, yv.to(y_ptr.dtype.element_ty), mask=m)

    _TRITON_KERNELS = {
        "finalize_stats": _tign_finalize_stats_kernel,
        "ndhwc_pow2_stats": _tign_ndhwc_pow2_stats_kernel,
        "ndhwc_pow2_apply": _tign_ndhwc_pow2_apply_kernel,
        "ndhwc_gen_stats": _tign_ndhwc_gen_stats_kernel,
        "ndhwc_gen_apply": _tign_ndhwc_gen_apply_kernel,
        "ncdhw_onepass": _tign_ncdhw_onepass_kernel,
        "ncdhw_stats": _tign_ncdhw_stats_kernel,
        "ncdhw_apply": _tign_ncdhw_apply_kernel,
    }
    return _TRITON_KERNELS


# =============================================================================
# Launcher
# =============================================================================


def _launch_temporal_isolated_group_norm_act(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_scale: torch.Tensor | None,
    dynamic_bias: torch.Tensor | None,
    eps: float,
    act_code: int,
    temporal_isolated: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _temporal_isolated_group_norm_act_torch(
            x, num_groups, weight, bias, dynamic_scale, dynamic_bias, eps, act_code, temporal_isolated
        )

    B, C, T, H, W = x.shape
    G = int(num_groups)
    CG = C // G
    has_weight = weight is not None
    has_bias = bias is not None
    has_dynamic = dynamic_scale is not None and dynamic_bias is not None
    w_t = weight.contiguous() if weight is not None else x
    b_t = bias.contiguous() if bias is not None else x
    ds_t = dynamic_scale if dynamic_scale is not None else x
    db_t = dynamic_bias if dynamic_bias is not None else x

    Ds, Hs, Ws, ss_n, ss_c, ss_d, ss_h, ss_w = _effective_5d_strides(dynamic_scale)
    Db, Hb, Wb, sb_n, sb_c, sb_d, sb_h, sb_w = _effective_5d_strides(dynamic_bias)
    same_dyn_shape = (Ds == Db) and (Hs == Hb) and (Ws == Wb)

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    if is_ndhwc:
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        T_eff = T if temporal_isolated else 1
        S_eff = H * W if temporal_isolated else T * H * W
        stride_xn = x.stride(0)
        stride_xt = x.stride(2) if temporal_isolated else 0
        stride_yn = y.stride(0)
        stride_yt = y.stride(2) if temporal_isolated else 0
        rows = B * T_eff * G
        cg_is_pow2 = (CG & (CG - 1)) == 0

        with torch.cuda.device(x.device):
            if cg_is_pow2 and CG <= 256:
                block_g = min(_next_power_of_2(G), max(1, 512 // CG))
                block_c = block_g * CG
                block_s = max(8, min(256, 8192 // block_c))
                num_g_tiles = _cdiv(G, block_g)
                ntg_total = B * T_eff * num_g_tiles
                target_chunks = max(1, 528 // max(1, ntg_total))
                num_chunks = min(_cdiv(S_eff, block_s), min(512, target_chunks))
                s_per_chunk = _cdiv(_cdiv(S_eff, num_chunks), block_s) * block_s
                num_chunks = _cdiv(S_eff, s_per_chunk)

                p_mean = torch.empty((rows, num_chunks), device=x.device, dtype=torch.float32)
                p_m2 = torch.empty_like(p_mean)
                p_cnt = torch.empty_like(p_mean)
                mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
                rstd = torch.empty_like(mean)

                kernels["ndhwc_pow2_stats"][(num_chunks, ntg_total)](
                    x,
                    p_mean,
                    p_m2,
                    p_cnt,
                    stride_xn,
                    stride_xt,
                    T_eff,
                    S_eff,
                    C,
                    G,
                    s_per_chunk,
                    num_chunks,
                    CG,
                    BLOCK_S=block_s,
                    BLOCK_G=block_g,
                    BLOCK_C=block_c,
                    num_warps=8,
                    num_stages=3,
                )
                kernels["finalize_stats"][(rows,)](
                    p_mean,
                    p_m2,
                    p_cnt,
                    mean,
                    rstd,
                    num_chunks,
                    eps,
                    BLOCK_K=_next_power_of_2(num_chunks),
                    num_warps=4,
                    num_stages=2,
                )
                apply_iters = 4
                num_apply = _cdiv(S_eff, block_s * apply_iters)
                kernels["ndhwc_pow2_apply"][(num_apply, ntg_total)](
                    x,
                    w_t,
                    b_t,
                    ds_t,
                    db_t,
                    y,
                    mean,
                    rstd,
                    stride_xn,
                    stride_xt,
                    stride_yn,
                    stride_yt,
                    T,
                    H,
                    W,
                    T_eff,
                    S_eff,
                    C,
                    G,
                    Ds,
                    Hs,
                    Ws,
                    ss_n,
                    ss_c,
                    ss_d,
                    ss_h,
                    ss_w,
                    Db,
                    Hb,
                    Wb,
                    sb_n,
                    sb_c,
                    sb_d,
                    sb_h,
                    sb_w,
                    CG,
                    HAS_WEIGHT=has_weight,
                    HAS_BIAS=has_bias,
                    HAS_DYNAMIC=has_dynamic,
                    SAME_DYN_SHAPE=same_dyn_shape,
                    TEMPORAL_ISOLATED=temporal_isolated,
                    ACT_CODE=act_code,
                    BLOCK_S=block_s,
                    APPLY_ITERS=apply_iters,
                    BLOCK_G=block_g,
                    BLOCK_C=block_c,
                    num_warps=8,
                    num_stages=3,
                )
                return y

            block_cg = _next_power_of_2(CG)
            block_g = min(_next_power_of_2(G), max(1, 256 // block_cg))
            block_s = max(8, min(128, 4096 // (block_g * block_cg)))
            num_g_tiles = _cdiv(G, block_g)
            ntg_total = B * T_eff * num_g_tiles
            target_chunks = max(1, 528 // max(1, ntg_total))
            num_chunks = min(_cdiv(S_eff, block_s), min(512, target_chunks))
            s_per_chunk = _cdiv(_cdiv(S_eff, num_chunks), block_s) * block_s
            num_chunks = _cdiv(S_eff, s_per_chunk)

            p_mean = torch.empty((rows, num_chunks), device=x.device, dtype=torch.float32)
            p_m2 = torch.empty_like(p_mean)
            p_cnt = torch.empty_like(p_mean)
            mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
            rstd = torch.empty_like(mean)

            kernels["ndhwc_gen_stats"][(num_chunks, ntg_total)](
                x,
                p_mean,
                p_m2,
                p_cnt,
                stride_xn,
                stride_xt,
                T_eff,
                S_eff,
                C,
                G,
                CG,
                s_per_chunk,
                num_chunks,
                BLOCK_S=block_s,
                BLOCK_G=block_g,
                BLOCK_CG=block_cg,
                num_warps=8,
                num_stages=3,
            )
            kernels["finalize_stats"][(rows,)](
                p_mean,
                p_m2,
                p_cnt,
                mean,
                rstd,
                num_chunks,
                eps,
                BLOCK_K=_next_power_of_2(num_chunks),
                num_warps=4,
                num_stages=2,
            )
            apply_iters = 4
            num_apply = _cdiv(S_eff, block_s * apply_iters)
            kernels["ndhwc_gen_apply"][(num_apply, ntg_total)](
                x,
                w_t,
                b_t,
                ds_t,
                db_t,
                y,
                mean,
                rstd,
                stride_xn,
                stride_xt,
                stride_yn,
                stride_yt,
                T,
                H,
                W,
                T_eff,
                S_eff,
                C,
                G,
                CG,
                Ds,
                Hs,
                Ws,
                ss_n,
                ss_c,
                ss_d,
                ss_h,
                ss_w,
                Db,
                Hb,
                Wb,
                sb_n,
                sb_c,
                sb_d,
                sb_h,
                sb_w,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                HAS_DYNAMIC=has_dynamic,
                TEMPORAL_ISOLATED=temporal_isolated,
                ACT_CODE=act_code,
                BLOCK_S=block_s,
                APPLY_ITERS=apply_iters,
                BLOCK_G=block_g,
                BLOCK_CG=block_cg,
                num_warps=8,
                num_stages=3,
            )
            return y

    # NCDHW contiguous path
    x_contig = x if x.is_contiguous() else x.contiguous()
    y = torch.empty_like(x_contig)
    T_eff = T if temporal_isolated else 1
    S_eff = H * W if temporal_isolated else T * H * W
    rows = B * T_eff * G

    with torch.cuda.device(x.device):
        if S_eff <= 8192 and rows >= 64:
            block_s = min(2048, _next_power_of_2(max(32, S_eff)))
            kernels["ncdhw_onepass"][(rows,)](
                x_contig,
                w_t,
                b_t,
                ds_t,
                db_t,
                y,
                C,
                T,
                H,
                W,
                G,
                CG,
                eps,
                Ds,
                Hs,
                Ws,
                ss_n,
                ss_c,
                ss_d,
                ss_h,
                ss_w,
                Db,
                Hb,
                Wb,
                sb_n,
                sb_c,
                sb_d,
                sb_h,
                sb_w,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                HAS_DYNAMIC=has_dynamic,
                TEMPORAL_ISOLATED=temporal_isolated,
                ACT_CODE=act_code,
                BLOCK_S=block_s,
                num_warps=8 if block_s >= 1024 else 4,
                num_stages=3,
            )
            return y

        block_s = 4096 if S_eff >= 4096 else (2048 if S_eff >= 2048 else _next_power_of_2(max(32, S_eff)))
        target_chunks = max(1, 2112 // max(1, rows))
        num_chunks = min(_cdiv(S_eff, block_s), min(512, target_chunks))
        s_per_chunk = _cdiv(_cdiv(S_eff, num_chunks), block_s) * block_s
        num_chunks = _cdiv(S_eff, s_per_chunk)

        p_mean = torch.empty((rows, num_chunks), device=x.device, dtype=torch.float32)
        p_m2 = torch.empty_like(p_mean)
        p_cnt = torch.empty_like(p_mean)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)

        kernels["ncdhw_stats"][(num_chunks, rows)](
            x_contig,
            p_mean,
            p_m2,
            p_cnt,
            C,
            T,
            H,
            W,
            G,
            CG,
            s_per_chunk,
            num_chunks,
            TEMPORAL_ISOLATED=temporal_isolated,
            BLOCK_S=block_s,
            num_warps=8 if block_s >= 1024 else 4,
            num_stages=3,
        )
        kernels["finalize_stats"][(rows,)](
            p_mean,
            p_m2,
            p_cnt,
            mean,
            rstd,
            num_chunks,
            eps,
            BLOCK_K=_next_power_of_2(num_chunks),
            num_warps=4,
            num_stages=2,
        )
        apply_iters = 2
        num_apply = _cdiv(S_eff, block_s * apply_iters)
        kernels["ncdhw_apply"][(num_apply, rows)](
            x_contig,
            w_t,
            b_t,
            ds_t,
            db_t,
            y,
            mean,
            rstd,
            C,
            T,
            H,
            W,
            G,
            CG,
            Ds,
            Hs,
            Ws,
            ss_n,
            ss_c,
            ss_d,
            ss_h,
            ss_w,
            Db,
            Hb,
            Wb,
            sb_n,
            sb_c,
            sb_d,
            sb_h,
            sb_w,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            HAS_DYNAMIC=has_dynamic,
            SAME_DYN_SHAPE=same_dyn_shape,
            TEMPORAL_ISOLATED=temporal_isolated,
            ACT_CODE=act_code,
            BLOCK_S=block_s,
            APPLY_ITERS=apply_iters,
            num_warps=8 if block_s >= 1024 else 4,
            num_stages=3,
        )
        return y


# =============================================================================
# PyTorch Reference Fallback
# =============================================================================


def _temporal_isolated_group_norm_act_torch(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_scale: torch.Tensor | None,
    dynamic_bias: torch.Tensor | None,
    eps: float,
    act_code: int,
    temporal_isolated: bool,
) -> torch.Tensor:
    need_reshape = temporal_isolated and x.ndim == 5
    f_size = x.shape[-3:]
    batch = x.shape[0]
    if need_reshape:
        _, channels, depth, height, width = x.shape
        x_in = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * depth, channels, 1, height, width)
    else:
        x_in = x

    out = F.group_norm(x_in, num_groups, weight=weight, bias=bias, eps=eps)
    if dynamic_scale is not None and dynamic_bias is not None:
        scale = dynamic_scale
        dbias = dynamic_bias
        if math.prod(scale.shape[-3:]) * math.prod(dbias.shape[-3:]) > 1:
            scale = F.interpolate(scale, size=f_size, mode="nearest")
            dbias = F.interpolate(dbias, size=f_size, mode="nearest")
        if need_reshape:
            if scale.shape[2] == 1 and depth > 1:
                scale = scale.expand(-1, -1, depth, -1, -1)
            if dbias.shape[2] == 1 and depth > 1:
                dbias = dbias.expand(-1, -1, depth, -1, -1)
            sb, sc, sd, sh, sw = scale.shape
            scale = scale.permute(0, 2, 1, 3, 4).contiguous().view(sb * sd, sc, 1, sh, sw)
            bb, bc, bd, bh, bw = dbias.shape
            dbias = dbias.permute(0, 2, 1, 3, 4).contiguous().view(bb * bd, bc, 1, bh, bw)
        out = out * scale + dbias

    if act_code == 1:
        out = F.silu(out)
    elif act_code == 2:
        out = F.relu(out)

    if need_reshape:
        batch_depth, channels, _, height, width = out.shape
        depth = batch_depth // batch
        out = out.view(batch, depth, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
    return out


# =============================================================================
# Compile-Safe Custom Operator
# =============================================================================


@torch.library.custom_op(
    "worldfoundry::minimax_h3_temporal_isolated_group_norm_act",
    mutates_args=(),
    device_types="cuda",
)
def _minimax_h3_tign_act_op(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_scale: torch.Tensor | None,
    dynamic_bias: torch.Tensor | None,
    eps: float,
    act_code: int,
    temporal_isolated: bool,
) -> torch.Tensor:
    return _launch_temporal_isolated_group_norm_act(
        x, num_groups, weight, bias, dynamic_scale, dynamic_bias, eps, act_code, temporal_isolated
    )


@_minimax_h3_tign_act_op.register_fake
def _minimax_h3_tign_act_fake(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_scale: torch.Tensor | None,
    dynamic_bias: torch.Tensor | None,
    eps: float,
    act_code: int,
    temporal_isolated: bool,
) -> torch.Tensor:
    del num_groups, weight, bias, dynamic_scale, dynamic_bias, eps, act_code, temporal_isolated
    return _empty_like_layout(x)


# =============================================================================
# Public Entry Point
# =============================================================================


def _can_use_triton(x: torch.Tensor, *others: torch.Tensor | None) -> bool:
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.compiler.is_compiling() and x.numel() == 0:
        return False
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, *others)):
        return False
    return _get_triton_kernels() is not None


def triton_temporal_isolated_group_norm_act(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    dynamic_scale: torch.Tensor | None = None,
    dynamic_bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    activation: str = "silu",
    *,
    temporal_isolated: bool = True,
) -> torch.Tensor:
    """Stride-aware 5-D temporal-isolated GroupNorm + optional dynamic spatial affine + activation."""
    if activation not in _ACT_TO_CODE:
        raise ValueError(f"Unsupported activation: {activation}. Supported: {set(_ACT_TO_CODE)}")
    act_code = _ACT_TO_CODE[activation]
    num_groups_i = int(num_groups)
    eps_f = float(eps)

    if (
        x.ndim == 5
        and x.shape[1] % num_groups_i == 0
        and (dynamic_scale is None) == (dynamic_bias is None)
        and (dynamic_scale is None or (dynamic_scale.ndim == 5 and dynamic_scale.is_cuda))
        and (dynamic_bias is None or (dynamic_bias.ndim == 5 and dynamic_bias.is_cuda))
        and (weight is None or weight.is_cuda)
        and (bias is None or bias.is_cuda)
        and _can_use_triton(x, weight, bias, dynamic_scale, dynamic_bias)
    ):
        return _minimax_h3_tign_act_op(
            x,
            num_groups_i,
            weight,
            bias,
            dynamic_scale,
            dynamic_bias,
            eps_f,
            act_code,
            bool(temporal_isolated),
        )
    return _temporal_isolated_group_norm_act_torch(
        x,
        num_groups_i,
        weight,
        bias,
        dynamic_scale,
        dynamic_bias,
        eps_f,
        act_code,
        bool(temporal_isolated),
    )


__all__ = ["triton_temporal_isolated_group_norm_act"]
