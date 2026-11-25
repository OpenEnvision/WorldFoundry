# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper Triton operators for the original HunyuanVideo 3-D VAE.

Implements compile-safe, CUDA-Graph-safe, stride-aware custom operators
supporting BOTH ``NCDHW`` contiguous and ``NDHWC`` (``channels_last_3d``)
tensors without hidden ``.contiguous()`` layout copies:

- :func:`triton_group_norm_3d_silu` — 5-D GroupNorm + optional SiLU with
  deterministic FP32 Chan parallel variance reduction and coalesced channel
  vector loads in ``channels_last_3d``.
- :func:`triton_upsample_causal_3d` — single-pass native BF16/FP16/FP32 causal
  nearest-neighbor upsample (``t_out = 0`` copies ``t_in = 0``; ``t_out >= 1``
  copies ``t_in = 1 + (t_out - 1) // st``) with zero FP32 roundtrips and zero
  ``split``/``cat`` allocations.

Triton is imported lazily on first CUDA invocation; CPU, autograd, and
unsupported dtypes fall back transparently to exact PyTorch math.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TRITON_KERNELS: dict[str, Any] | None = None
_TRITON_LOAD_FAILED: bool = False


def _is_channels_last_3d(x: torch.Tensor) -> bool:
    """Return True when 5-D *x* uses ``channels_last_3d`` (NDHWC) memory order."""
    if x.ndim != 5:
        return False
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        if not x.is_contiguous():
            return True
        return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
    return False


def _is_batch_strided_ndhwc(x: torch.Tensor) -> bool:
    """Return True when 5-D *x* has contiguous inner ``(D, H, W, C)`` slices."""
    if x.ndim != 5:
        return False
    _, c, _, h, w = x.shape
    return x.stride(1) == 1 and x.stride(4) == c and x.stride(3) == w * c and x.stride(2) == h * w * c


def _empty_like_layout(
    x: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = x.dtype if dtype is None else dtype
    if _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x):
        return torch.empty(shape, device=x.device, dtype=out_dtype, memory_format=torch.channels_last_3d)
    return torch.empty(shape, device=x.device, dtype=out_dtype)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_triton_kernels() -> dict[str, Any] | None:
    """Lazily compile and cache Triton JIT kernels for HunyuanVideo VAE."""
    global _TRITON_KERNELS, _TRITON_LOAD_FAILED
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    if _TRITON_LOAD_FAILED:
        return None
    try:
        from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="hunyuan-vae-triton")
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_LOAD_FAILED = True
        return None

    # =========================================================================
    # 1. GroupNorm 3D + Optional SiLU — NDHWC (channels_last_3d)
    # =========================================================================

    @triton.jit
    def _gn3d_ndhwc_pow2_onepass_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        S,
        C,
        G,
        CG: tl.constexpr,
        eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch_id = tl.program_id(0).to(tl.int64)
        g_tile = tl.program_id(1).to(tl.int64)

        g_start = g_tile * BLOCK_G
        c_start = g_start * CG
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        c_mask = c_offs < C
        g_mask = g_offs < G

        x_batch_ptr = x_ptr + batch_id * stride_xn
        y_batch_ptr = y_ptr + batch_id * stride_yn

        # First tile establishes shift mu0 for catastrophic-cancellation-free variance.
        # Accumulate stats in fp64 on the small-tensor 1-pass path so NDHWC and NCDHW
        # reductions yield bit-identical fp32 mean/rstd.
        s0 = s_base_offs
        m0 = (s0[:, None] < S) & c_mask[None, :]
        x0 = tl.load(x_batch_ptr + s0[:, None] * C + c_offs[None, :], mask=m0, other=0.0).to(tl.float64)
        n0_vox = tl.minimum(S, BLOCK_S).to(tl.float64)
        n0_elems = tl.maximum(n0_vox * CG, 1.0)
        x0_3d = tl.reshape(x0, (BLOCK_S, BLOCK_G, CG))
        mu0 = tl.sum(tl.sum(x0_3d, axis=2), axis=0) / n0_elems

        m0_3d = tl.reshape(m0, (BLOCK_S, BLOCK_G, CG))
        d0 = tl.where(m0_3d, x0_3d - mu0[None, :, None], 0.0)
        sum_d = tl.sum(tl.sum(d0, axis=2), axis=0)
        sum_d2 = tl.sum(tl.sum(d0 * d0, axis=2), axis=0)

        for s_start in range(BLOCK_S, S, BLOCK_S):
            s_offs = s_start + s_base_offs
            m = (s_offs[:, None] < S) & c_mask[None, :]
            xv = tl.load(x_batch_ptr + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float64)
            xv_3d = tl.reshape(xv, (BLOCK_S, BLOCK_G, CG))
            m_3d = tl.reshape(m, (BLOCK_S, BLOCK_G, CG))
            d = tl.where(m_3d, xv_3d - mu0[None, :, None], 0.0)
            sum_d += tl.sum(tl.sum(d, axis=2), axis=0)
            sum_d2 += tl.sum(tl.sum(d * d, axis=2), axis=0)

        total_n = (S * CG).to(tl.float64)
        delta = sum_d / total_n
        mean = (mu0 + delta).to(tl.float32)
        var = tl.maximum(sum_d2 / total_n - delta * delta, 0.0).to(tl.float32)
        rstd = tl.rsqrt(var + eps)

        mean_c = tl.reshape(tl.broadcast_to(mean[:, None], (BLOCK_G, CG)), (BLOCK_C,))
        rstd_c = tl.reshape(tl.broadcast_to(rstd[:, None], (BLOCK_G, CG)), (BLOCK_C,))
        if HAS_WEIGHT:
            w = tl.load(w_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + s_base_offs
            m = (s_offs[:, None] < S) & c_mask[None, :]
            xv = tl.load(x_batch_ptr + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean_c[None, :]) * rstd_c[None, :]
            if HAS_WEIGHT:
                yv = yv * w[None, :]
            if HAS_BIAS:
                yv = yv + b[None, :]
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_batch_ptr + s_offs[:, None] * C + c_offs[None, :], yv.to(y_ptr.dtype.element_ty), mask=m)

        # Silence unused g_mask warning if any
        _ = g_mask

    @triton.jit
    def _gn3d_ndhwc_pow2_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        stride_xn,
        S,
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
        bg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        batch_id = bg // num_g_tiles
        g_tile = bg - batch_id * num_g_tiles

        g_start = g_tile * BLOCK_G
        c_start = g_start * CG
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        g_mask = g_offs < G
        c_mask = c_offs < C

        chunk_s_start = chunk_id * s_per_chunk
        chunk_s_end = tl.minimum(chunk_s_start + s_per_chunk, S)
        out_idx = (batch_id * G + g_offs) * num_chunks + chunk_id

        if chunk_s_start >= S:
            zeros = tl.zeros((BLOCK_G,), dtype=tl.float32)
            tl.store(p_mean_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_m2_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_cnt_ptr + out_idx, zeros, mask=g_mask)
            return

        x_batch_ptr = x_ptr + batch_id * stride_xn
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        s0 = chunk_s_start + s_base_offs
        m0 = (s0[:, None] < chunk_s_end) & c_mask[None, :]
        x0 = tl.load(x_batch_ptr + s0[:, None] * C + c_offs[None, :], mask=m0, other=0.0).to(tl.float32)
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
            xv = tl.load(x_batch_ptr + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
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
    def _gn3d_ndhwc_pow2_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        stride_xn,
        stride_yn,
        S,
        C,
        G,
        CG: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        bg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        batch_id = bg // num_g_tiles
        g_tile = bg - batch_id * num_g_tiles

        g_start = g_tile * BLOCK_G
        c_start = g_start * CG
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        g_mask = g_offs < G
        c_mask = c_offs < C

        stats_offs = batch_id * G + g_offs
        mean = tl.load(mean_ptr + stats_offs, mask=g_mask, other=0.0)
        rstd = tl.load(rstd_ptr + stats_offs, mask=g_mask, other=1.0)
        mean_c = tl.reshape(tl.broadcast_to(mean[:, None], (BLOCK_G, CG)), (BLOCK_C,))
        rstd_c = tl.reshape(tl.broadcast_to(rstd[:, None], (BLOCK_G, CG)), (BLOCK_C,))

        if HAS_WEIGHT:
            w = tl.load(w_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        x_batch_ptr = x_ptr + batch_id * stride_xn
        y_batch_ptr = y_ptr + batch_id * stride_yn
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)
        cta_s_start = s_tile * (BLOCK_S * APPLY_ITERS)

        for it in tl.static_range(APPLY_ITERS):
            s_offs = cta_s_start + it * BLOCK_S + s_base_offs
            m = (s_offs[:, None] < S) & c_mask[None, :]
            xv = tl.load(x_batch_ptr + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean_c[None, :]) * rstd_c[None, :]
            if HAS_WEIGHT:
                yv = yv * w[None, :]
            if HAS_BIAS:
                yv = yv + b[None, :]
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_batch_ptr + s_offs[:, None] * C + c_offs[None, :], yv.to(y_ptr.dtype.element_ty), mask=m)

    # General NDHWC kernels for non-power-of-2 channels_per_group (e.g. C=96, G=32 -> CG=3)
    @triton.jit
    def _gn3d_ndhwc_gen_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        stride_xn,
        S,
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
        bg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        batch_id = bg // num_g_tiles
        g_tile = bg - batch_id * num_g_tiles

        g_start = g_tile * BLOCK_G
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        cg_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        g_mask = g_offs < G
        cg_mask = cg_offs < CG
        ch_offs = g_offs[:, None] * CG + cg_offs[None, :]  # (BLOCK_G, BLOCK_CG)
        ch_mask = g_mask[:, None] & cg_mask[None, :]

        chunk_s_start = chunk_id * s_per_chunk
        chunk_s_end = tl.minimum(chunk_s_start + s_per_chunk, S)
        out_idx = (batch_id * G + g_offs) * num_chunks + chunk_id

        if chunk_s_start >= S:
            zeros = tl.zeros((BLOCK_G,), dtype=tl.float32)
            tl.store(p_mean_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_m2_ptr + out_idx, zeros, mask=g_mask)
            tl.store(p_cnt_ptr + out_idx, zeros, mask=g_mask)
            return

        x_batch_ptr = x_ptr + batch_id * stride_xn
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)

        s0 = chunk_s_start + s_base_offs
        m0 = (s0[:, None, None] < chunk_s_end) & ch_mask[None, :, :]
        x0 = tl.load(x_batch_ptr + s0[:, None, None] * C + ch_offs[None, :, :], mask=m0, other=0.0).to(tl.float32)
        n0_vox = tl.minimum(chunk_s_end - chunk_s_start, BLOCK_S).to(tl.float32)
        n0_elems = tl.maximum(n0_vox * CG, 1.0)
        mu0 = tl.sum(tl.sum(x0, axis=2), axis=0) / n0_elems

        d0 = tl.where(m0, x0 - mu0[None, :, None], 0.0)
        sum_d = tl.sum(tl.sum(d0, axis=2), axis=0)
        sum_d2 = tl.sum(tl.sum(d0 * d0, axis=2), axis=0)

        for s_start in range(chunk_s_start + BLOCK_S, chunk_s_end, BLOCK_S):
            s_offs = s_start + s_base_offs
            m = (s_offs[:, None, None] < chunk_s_end) & ch_mask[None, :, :]
            xv = tl.load(x_batch_ptr + s_offs[:, None, None] * C + ch_offs[None, :, :], mask=m, other=0.0).to(
                tl.float32
            )
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
    def _gn3d_ndhwc_gen_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        stride_xn,
        stride_yn,
        S,
        C,
        G,
        CG,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        bg = tl.program_id(1).to(tl.int64)
        num_g_tiles = tl.cdiv(G, BLOCK_G)
        batch_id = bg // num_g_tiles
        g_tile = bg - batch_id * num_g_tiles

        g_start = g_tile * BLOCK_G
        g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
        cg_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        g_mask = g_offs < G
        cg_mask = cg_offs < CG
        ch_offs = g_offs[:, None] * CG + cg_offs[None, :]
        ch_mask = g_mask[:, None] & cg_mask[None, :]

        stats_offs = batch_id * G + g_offs
        mean = tl.load(mean_ptr + stats_offs, mask=g_mask, other=0.0)
        rstd = tl.load(rstd_ptr + stats_offs, mask=g_mask, other=1.0)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + ch_offs, mask=ch_mask, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + ch_offs, mask=ch_mask, other=0.0).to(tl.float32)

        x_batch_ptr = x_ptr + batch_id * stride_xn
        y_batch_ptr = y_ptr + batch_id * stride_yn
        s_base_offs = tl.arange(0, BLOCK_S).to(tl.int64)
        cta_s_start = s_tile * (BLOCK_S * APPLY_ITERS)

        for it in tl.static_range(APPLY_ITERS):
            s_offs = cta_s_start + it * BLOCK_S + s_base_offs
            m = (s_offs[:, None, None] < S) & ch_mask[None, :, :]
            xv = tl.load(x_batch_ptr + s_offs[:, None, None] * C + ch_offs[None, :, :], mask=m, other=0.0).to(
                tl.float32
            )
            yv = (xv - mean[None, :, None]) * rstd[None, :, None]
            if HAS_WEIGHT:
                yv = yv * w[None, :, :]
            if HAS_BIAS:
                yv = yv + b[None, :, :]
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(
                y_batch_ptr + s_offs[:, None, None] * C + ch_offs[None, :, :],
                yv.to(y_ptr.dtype.element_ty),
                mask=m,
            )

    # =========================================================================
    # 2. Shared Stage-2 Chan Parallel Variance Finalization
    # =========================================================================

    @triton.jit
    def _gn3d_finalize_stats_kernel(
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
    # 3. GroupNorm 3D + Optional SiLU — NCDHW Contiguous
    # =========================================================================

    @triton.jit
    def _gn3d_ncdhw_onepass_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        S,
        G,
        CG,
        group_size,
        eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        group_id = row % G
        group_base = row * group_size
        base_offs = tl.arange(0, BLOCK_SIZE).to(tl.int64)

        m0 = base_offs < group_size
        x0 = tl.load(x_ptr + group_base + base_offs, mask=m0, other=0.0).to(tl.float64)
        n0 = tl.maximum(tl.minimum(group_size, BLOCK_SIZE).to(tl.float64), 1.0)
        mu0 = tl.sum(x0, axis=0) / n0
        d0 = tl.where(m0, x0 - mu0, 0.0)
        sum_d = tl.sum(d0, axis=0)
        sum_d2 = tl.sum(d0 * d0, axis=0)

        for start in range(BLOCK_SIZE, group_size, BLOCK_SIZE):
            offs = start + base_offs
            m = offs < group_size
            xv = tl.load(x_ptr + group_base + offs, mask=m, other=0.0).to(tl.float64)
            d = tl.where(m, xv - mu0, 0.0)
            sum_d += tl.sum(d, axis=0)
            sum_d2 += tl.sum(d * d, axis=0)

        total_n = group_size.to(tl.float64)
        delta = sum_d / total_n
        mean = (mu0 + delta).to(tl.float32)
        var = tl.maximum(sum_d2 / total_n - delta * delta, 0.0).to(tl.float32)
        rstd = tl.rsqrt(var + eps)
        ch_base = group_id * CG

        for start in range(0, group_size, BLOCK_SIZE):
            offs = start + base_offs
            m = offs < group_size
            xv = tl.load(x_ptr + group_base + offs, mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean) * rstd
            if HAS_WEIGHT or HAS_BIAS:
                c_offs = ch_base + offs // S
                if HAS_WEIGHT:
                    w = tl.load(w_ptr + c_offs, mask=m, other=1.0).to(tl.float32)
                    yv = yv * w
                if HAS_BIAS:
                    b = tl.load(b_ptr + c_offs, mask=m, other=0.0).to(tl.float32)
                    yv = yv + b
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_ptr + group_base + offs, yv.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _gn3d_ncdhw_stats_kernel(
        x_ptr,
        p_mean_ptr,
        p_m2_ptr,
        p_cnt_ptr,
        group_size,
        elems_per_chunk,
        num_chunks,
        BLOCK_SIZE: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        group_base = row * group_size
        chunk_start = chunk_id * elems_per_chunk
        chunk_end = tl.minimum(chunk_start + elems_per_chunk, group_size)
        out_idx = row * num_chunks + chunk_id

        if chunk_start >= group_size:
            tl.store(p_mean_ptr + out_idx, 0.0)
            tl.store(p_m2_ptr + out_idx, 0.0)
            tl.store(p_cnt_ptr + out_idx, 0.0)
            return

        base_offs = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        offs0 = chunk_start + base_offs
        m0 = offs0 < chunk_end
        x0 = tl.load(x_ptr + group_base + offs0, mask=m0, other=0.0).to(tl.float32)
        n0 = tl.maximum(tl.minimum(chunk_end - chunk_start, BLOCK_SIZE).to(tl.float32), 1.0)
        mu0 = tl.sum(x0, axis=0) / n0
        d0 = tl.where(m0, x0 - mu0, 0.0)
        sum_d = tl.sum(d0, axis=0)
        sum_d2 = tl.sum(d0 * d0, axis=0)

        for start in range(chunk_start + BLOCK_SIZE, chunk_end, BLOCK_SIZE):
            offs = start + base_offs
            m = offs < chunk_end
            xv = tl.load(x_ptr + group_base + offs, mask=m, other=0.0).to(tl.float32)
            d = tl.where(m, xv - mu0, 0.0)
            sum_d += tl.sum(d, axis=0)
            sum_d2 += tl.sum(d * d, axis=0)

        n_elems = (chunk_end - chunk_start).to(tl.float32)
        delta = sum_d / n_elems
        chunk_mean = mu0 + delta
        chunk_m2 = tl.maximum(sum_d2 - n_elems * delta * delta, 0.0)

        tl.store(p_mean_ptr + out_idx, chunk_mean)
        tl.store(p_m2_ptr + out_idx, chunk_m2)
        tl.store(p_cnt_ptr + out_idx, n_elems)

    @triton.jit
    def _gn3d_ncdhw_apply_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        S,
        G,
        CG,
        group_size,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        SCALAR_AFFINE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        APPLY_ITERS: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        group_id = row % G
        group_base = row * group_size
        ch_base = group_id * CG

        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        cta_start = chunk_id * (BLOCK_SIZE * APPLY_ITERS)
        base_offs = tl.arange(0, BLOCK_SIZE).to(tl.int64)

        if SCALAR_AFFINE:
            ch = ch_base + cta_start // S
            if HAS_WEIGHT:
                w_s = tl.load(w_ptr + ch).to(tl.float32)
            if HAS_BIAS:
                b_s = tl.load(b_ptr + ch).to(tl.float32)

        for it in tl.static_range(APPLY_ITERS):
            offs = cta_start + it * BLOCK_SIZE + base_offs
            m = offs < group_size
            xv = tl.load(x_ptr + group_base + offs, mask=m, other=0.0).to(tl.float32)
            yv = (xv - mean) * rstd
            if SCALAR_AFFINE:
                if HAS_WEIGHT:
                    yv = yv * w_s
                if HAS_BIAS:
                    yv = yv + b_s
            else:
                if HAS_WEIGHT or HAS_BIAS:
                    c_offs = ch_base + offs // S
                    if HAS_WEIGHT:
                        w = tl.load(w_ptr + c_offs, mask=m, other=1.0).to(tl.float32)
                        yv = yv * w
                    if HAS_BIAS:
                        b = tl.load(b_ptr + c_offs, mask=m, other=0.0).to(tl.float32)
                        yv = yv + b
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_ptr + group_base + offs, yv.to(y_ptr.dtype.element_ty), mask=m)

    # =========================================================================
    # 4. UpsampleCausal3D — NDHWC & NCDHW (Bit-Exact Single-Pass)
    # =========================================================================

    @triton.jit
    def _upsample_causal_3d_ndhwc_pow2_kernel(
        x_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        T,
        H,
        W,
        C,
        T_out,
        H_out,
        W_out,
        SCALE_T: tl.constexpr,
        SCALE_H: tl.constexpr,
        SCALE_W: tl.constexpr,
        BLOCK_W_IN: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bth = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)
        c_tile = tl.program_id(2).to(tl.int64)

        h_in = bth % H
        bt = bth // H
        t_in = bt % T
        b = bt // T

        w_in_start = w_tile * BLOCK_W_IN
        c_start = c_tile * BLOCK_C
        w_in_offs = w_in_start + tl.arange(0, BLOCK_W_IN).to(tl.int64)
        c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
        in_mask = (w_in_offs[:, None] < W) & (c_offs[None, :] < C)

        in_row_base = b * stride_xn + (t_in * H + h_in) * (W * C)
        val = tl.load(x_ptr + in_row_base + w_in_offs[:, None] * C + c_offs[None, :], mask=in_mask)

        # Replicate along W in registers: (BLOCK_W_IN, BLOCK_C) -> (BLOCK_W_IN * SCALE_W, BLOCK_C)
        if SCALE_W == 1:
            out_val = val
        else:
            out_val = tl.reshape(
                tl.broadcast_to(val[:, None, :], (BLOCK_W_IN, SCALE_W, BLOCK_C)),
                (BLOCK_W_IN * SCALE_W, BLOCK_C),
            )

        w_out_offs = w_in_start * SCALE_W + tl.arange(0, BLOCK_W_IN * SCALE_W).to(tl.int64)
        out_mask = (w_out_offs[:, None] < W_out) & (c_offs[None, :] < C)

        t_out0 = tl.where(t_in == 0, 0, 1 + (t_in - 1) * SCALE_T)
        t_rep = tl.where(t_in == 0, 1, SCALE_T)
        h_out0 = h_in * SCALE_H
        y_batch_base = b * stride_yn
        hwc_stride = H_out * W_out * C
        wc_stride = W_out * C

        for dt in range(t_rep):
            t_base = y_batch_base + (t_out0 + dt) * hwc_stride
            for dh in tl.static_range(SCALE_H):
                row_base = t_base + (h_out0 + dh) * wc_stride
                tl.store(y_ptr + row_base + w_out_offs[:, None] * C + c_offs[None, :], out_val, mask=out_mask)

    @triton.jit
    def _upsample_causal_3d_ndhwc_gen_kernel(
        x_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        T,
        H,
        W,
        C,
        T_out,
        H_out,
        W_out,
        scale_t,
        scale_h,
        scale_w,
        BLOCK_W_OUT: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bth = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)
        c_tile = tl.program_id(2).to(tl.int64)

        h_out = bth % H_out
        bt = bth // H_out
        t_out = bt % T_out
        b = bt // T_out

        t_in = tl.where(t_out == 0, 0, 1 + (t_out - 1) // scale_t)
        h_in = h_out // scale_h

        w_out_offs = w_tile * BLOCK_W_OUT + tl.arange(0, BLOCK_W_OUT).to(tl.int64)
        c_offs = c_tile * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
        w_in_offs = w_out_offs // scale_w
        mask = (w_out_offs[:, None] < W_out) & (c_offs[None, :] < C)

        in_row_base = b * stride_xn + (t_in * H + h_in) * (W * C)
        out_row_base = b * stride_yn + (t_out * H_out + h_out) * (W_out * C)
        val = tl.load(x_ptr + in_row_base + w_in_offs[:, None] * C + c_offs[None, :], mask=mask)
        tl.store(y_ptr + out_row_base + w_out_offs[:, None] * C + c_offs[None, :], val, mask=mask)

    @triton.jit
    def _upsample_causal_3d_ncdhw_pow2_kernel(
        x_ptr,
        y_ptr,
        T,
        H,
        W,
        T_out,
        H_out,
        W_out,
        SCALE_T: tl.constexpr,
        SCALE_H: tl.constexpr,
        SCALE_W: tl.constexpr,
        BLOCK_W_IN: tl.constexpr,
    ):
        bcth = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        h_in = bcth % H
        bct = bcth // H
        t_in = bct % T
        bc = bct // T

        w_in_start = w_tile * BLOCK_W_IN
        w_in_offs = w_in_start + tl.arange(0, BLOCK_W_IN).to(tl.int64)
        in_mask = w_in_offs < W

        in_row_base = ((bc * T + t_in) * H + h_in) * W
        val = tl.load(x_ptr + in_row_base + w_in_offs, mask=in_mask)

        if SCALE_W == 1:
            out_val = val
        else:
            out_val = tl.reshape(
                tl.broadcast_to(val[:, None], (BLOCK_W_IN, SCALE_W)),
                (BLOCK_W_IN * SCALE_W,),
            )

        w_out_offs = w_in_start * SCALE_W + tl.arange(0, BLOCK_W_IN * SCALE_W).to(tl.int64)
        out_mask = w_out_offs < W_out

        t_out0 = tl.where(t_in == 0, 0, 1 + (t_in - 1) * SCALE_T)
        t_rep = tl.where(t_in == 0, 1, SCALE_T)
        h_out0 = h_in * SCALE_H
        bc_base = bc * (T_out * H_out * W_out)
        hw_stride = H_out * W_out

        for dt in range(t_rep):
            t_base = bc_base + (t_out0 + dt) * hw_stride
            for dh in tl.static_range(SCALE_H):
                tl.store(y_ptr + t_base + (h_out0 + dh) * W_out + w_out_offs, out_val, mask=out_mask)

    @triton.jit
    def _upsample_causal_3d_ncdhw_gen_kernel(
        x_ptr,
        y_ptr,
        T,
        H,
        W,
        T_out,
        H_out,
        W_out,
        scale_t,
        scale_h,
        scale_w,
        BLOCK_W_OUT: tl.constexpr,
    ):
        bcth = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        h_out = bcth % H_out
        bct = bcth // H_out
        t_out = bct % T_out
        bc = bct // T_out

        t_in = tl.where(t_out == 0, 0, 1 + (t_out - 1) // scale_t)
        h_in = h_out // scale_h

        w_out_offs = w_tile * BLOCK_W_OUT + tl.arange(0, BLOCK_W_OUT).to(tl.int64)
        w_in_offs = w_out_offs // scale_w
        mask = w_out_offs < W_out

        in_row_base = ((bc * T + t_in) * H + h_in) * W
        out_row_base = ((bc * T_out + t_out) * H_out + h_out) * W_out
        val = tl.load(x_ptr + in_row_base + w_in_offs, mask=mask)
        tl.store(y_ptr + out_row_base + w_out_offs, val, mask=mask)

    _TRITON_KERNELS = {
        "gn3d_ndhwc_pow2_onepass": _gn3d_ndhwc_pow2_onepass_kernel,
        "gn3d_ndhwc_pow2_stats": _gn3d_ndhwc_pow2_stats_kernel,
        "gn3d_ndhwc_pow2_apply": _gn3d_ndhwc_pow2_apply_kernel,
        "gn3d_ndhwc_gen_stats": _gn3d_ndhwc_gen_stats_kernel,
        "gn3d_ndhwc_gen_apply": _gn3d_ndhwc_gen_apply_kernel,
        "gn3d_finalize_stats": _gn3d_finalize_stats_kernel,
        "gn3d_ncdhw_onepass": _gn3d_ncdhw_onepass_kernel,
        "gn3d_ncdhw_stats": _gn3d_ncdhw_stats_kernel,
        "gn3d_ncdhw_apply": _gn3d_ncdhw_apply_kernel,
        "upsample_ndhwc_pow2": _upsample_causal_3d_ndhwc_pow2_kernel,
        "upsample_ndhwc_gen": _upsample_causal_3d_ndhwc_gen_kernel,
        "upsample_ncdhw_pow2": _upsample_causal_3d_ncdhw_pow2_kernel,
        "upsample_ncdhw_gen": _upsample_causal_3d_ncdhw_gen_kernel,
    }
    return _TRITON_KERNELS


# =============================================================================
# Launchers
# =============================================================================


def _launch_group_norm_3d_silu(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _group_norm_3d_silu_torch(x, num_groups, weight, bias, eps, apply_silu)

    B, C, D, H, W = x.shape
    S = D * H * W
    G = int(num_groups)
    CG = C // G
    has_weight = weight is not None
    has_bias = bias is not None
    w_t = weight.contiguous() if weight is not None else x
    b_t = bias.contiguous() if bias is not None else x

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    if is_ndhwc:
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        stride_xn = x.stride(0)
        stride_yn = y.stride(0)
        cg_is_pow2 = (CG & (CG - 1)) == 0

        with torch.cuda.device(x.device):
            if cg_is_pow2 and CG <= 256:
                block_g = min(_next_power_of_2(G), max(1, 512 // CG))
                block_c = block_g * CG
                block_s = max(8, min(256, 8192 // block_c))
                num_g_tiles = _cdiv(G, block_g)

                if S <= block_s * 4:  # noqa: SIM300 - S is runtime spatial size, not a constant.
                    kernels["gn3d_ndhwc_pow2_onepass"][(B, num_g_tiles)](
                        x,
                        w_t,
                        b_t,
                        y,
                        stride_xn,
                        stride_yn,
                        S,
                        C,
                        G,
                        CG,
                        eps,
                        HAS_WEIGHT=has_weight,
                        HAS_BIAS=has_bias,
                        APPLY_SILU=apply_silu,
                        BLOCK_S=block_s,
                        BLOCK_G=block_g,
                        BLOCK_C=block_c,
                        num_warps=8,
                        num_stages=3,
                    )
                    return y

                target_chunks = max(1, 528 // max(1, B * num_g_tiles))
                num_chunks = min(_cdiv(S, block_s), min(512, target_chunks))
                s_per_chunk = _cdiv(_cdiv(S, num_chunks), block_s) * block_s
                num_chunks = _cdiv(S, s_per_chunk)

                p_mean = torch.empty((B * G, num_chunks), device=x.device, dtype=torch.float32)
                p_m2 = torch.empty_like(p_mean)
                p_cnt = torch.empty_like(p_mean)
                mean = torch.empty((B * G,), device=x.device, dtype=torch.float32)
                rstd = torch.empty_like(mean)

                kernels["gn3d_ndhwc_pow2_stats"][(num_chunks, B * num_g_tiles)](
                    x,
                    p_mean,
                    p_m2,
                    p_cnt,
                    stride_xn,
                    S,
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
                kernels["gn3d_finalize_stats"][(B * G,)](
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
                num_apply = _cdiv(S, block_s * apply_iters)
                kernels["gn3d_ndhwc_pow2_apply"][(num_apply, B * num_g_tiles)](
                    x,
                    w_t,
                    b_t,
                    y,
                    mean,
                    rstd,
                    stride_xn,
                    stride_yn,
                    S,
                    C,
                    G,
                    CG,
                    HAS_WEIGHT=has_weight,
                    HAS_BIAS=has_bias,
                    APPLY_SILU=apply_silu,
                    BLOCK_S=block_s,
                    APPLY_ITERS=apply_iters,
                    BLOCK_G=block_g,
                    BLOCK_C=block_c,
                    num_warps=8,
                    num_stages=3,
                )
                return y

            # Non-power-of-2 CG general NDHWC path
            block_cg = _next_power_of_2(CG)
            block_g = min(_next_power_of_2(G), max(1, 256 // block_cg))
            block_s = max(8, min(128, 4096 // (block_g * block_cg)))
            num_g_tiles = _cdiv(G, block_g)
            target_chunks = max(1, 528 // max(1, B * num_g_tiles))
            num_chunks = min(_cdiv(S, block_s), min(512, target_chunks))
            s_per_chunk = _cdiv(_cdiv(S, num_chunks), block_s) * block_s
            num_chunks = _cdiv(S, s_per_chunk)

            p_mean = torch.empty((B * G, num_chunks), device=x.device, dtype=torch.float32)
            p_m2 = torch.empty_like(p_mean)
            p_cnt = torch.empty_like(p_mean)
            mean = torch.empty((B * G,), device=x.device, dtype=torch.float32)
            rstd = torch.empty_like(mean)

            kernels["gn3d_ndhwc_gen_stats"][(num_chunks, B * num_g_tiles)](
                x,
                p_mean,
                p_m2,
                p_cnt,
                stride_xn,
                S,
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
            kernels["gn3d_finalize_stats"][(B * G,)](
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
            num_apply = _cdiv(S, block_s * apply_iters)
            kernels["gn3d_ndhwc_gen_apply"][(num_apply, B * num_g_tiles)](
                x,
                w_t,
                b_t,
                y,
                mean,
                rstd,
                stride_xn,
                stride_yn,
                S,
                C,
                G,
                CG,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                APPLY_SILU=apply_silu,
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
    group_size = CG * S
    rows = B * G

    with torch.cuda.device(x.device):
        if group_size <= 65536:
            block_size = min(4096, _next_power_of_2(max(32, group_size)))
            kernels["gn3d_ncdhw_onepass"][(rows,)](
                x_contig,
                w_t,
                b_t,
                y,
                S,
                G,
                CG,
                group_size,
                eps,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                APPLY_SILU=apply_silu,
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size >= 2048 else 4,
                num_stages=3,
            )
            return y

        block_size = 4096
        target_chunks = max(1, 528 // max(1, rows))
        num_chunks = min(_cdiv(group_size, block_size), min(512, target_chunks))
        elems_per_chunk = _cdiv(_cdiv(group_size, num_chunks), block_size) * block_size
        num_chunks = _cdiv(group_size, elems_per_chunk)

        p_mean = torch.empty((rows, num_chunks), device=x.device, dtype=torch.float32)
        p_m2 = torch.empty_like(p_mean)
        p_cnt = torch.empty_like(p_mean)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)

        kernels["gn3d_ncdhw_stats"][(num_chunks, rows)](
            x_contig,
            p_mean,
            p_m2,
            p_cnt,
            group_size,
            elems_per_chunk,
            num_chunks,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=3,
        )
        kernels["gn3d_finalize_stats"][(rows,)](
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
        chunk_apply = block_size * apply_iters
        num_apply = _cdiv(group_size, chunk_apply)
        scalar_affine = (S % chunk_apply == 0) and (num_apply >= 4)
        kernels["gn3d_ncdhw_apply"][(num_apply, rows)](
            x_contig,
            w_t,
            b_t,
            y,
            mean,
            rstd,
            S,
            G,
            CG,
            group_size,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            APPLY_SILU=apply_silu,
            SCALAR_AFFINE=scalar_affine,
            BLOCK_SIZE=block_size,
            APPLY_ITERS=apply_iters,
            num_warps=4 if scalar_affine else 8,
            num_stages=3,
        )
        return y


def _launch_upsample_causal_3d(
    x: torch.Tensor,
    scale_t: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _upsample_causal_3d_torch(x, scale_t, scale_h, scale_w)

    B, C, T, H, W = x.shape
    T_out = 1 + (T - 1) * scale_t
    H_out = H * scale_h
    W_out = W * scale_w
    out_shape = (B, C, T_out, H_out, W_out)

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    with torch.cuda.device(x.device):
        if is_ndhwc:
            y = torch.empty(out_shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
            stride_xn = x.stride(0)
            stride_yn = y.stride(0)
            block_c = min(_next_power_of_2(C), 256)
            num_c_tiles = _cdiv(C, block_c)

            if scale_w in (1, 2, 4) and scale_h <= 8 and scale_t <= 8:
                block_w_in = max(4, min(64, 4096 // (block_c * scale_w)))
                block_w_in = _next_power_of_2(min(block_w_in, max(1, W)))
                num_w_tiles = _cdiv(W, block_w_in)
                kernels["upsample_ndhwc_pow2"][(B * T * H, num_w_tiles, num_c_tiles)](
                    x,
                    y,
                    stride_xn,
                    stride_yn,
                    T,
                    H,
                    W,
                    C,
                    T_out,
                    H_out,
                    W_out,
                    SCALE_T=scale_t,
                    SCALE_H=scale_h,
                    SCALE_W=scale_w,
                    BLOCK_W_IN=block_w_in,
                    BLOCK_C=block_c,
                    num_warps=8 if block_w_in * scale_w * block_c >= 4096 else 4,
                    num_stages=3,
                )
                return y

            block_w_out = max(8, min(64, 4096 // block_c))
            kernels["upsample_ndhwc_gen"][(B * T_out * H_out, _cdiv(W_out, block_w_out), num_c_tiles)](
                x,
                y,
                stride_xn,
                stride_yn,
                T,
                H,
                W,
                C,
                T_out,
                H_out,
                W_out,
                scale_t,
                scale_h,
                scale_w,
                BLOCK_W_OUT=block_w_out,
                BLOCK_C=block_c,
                num_warps=4,
                num_stages=3,
            )
            return y

        # NCDHW contiguous
        x_contig = x if x.is_contiguous() else x.contiguous()
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        if scale_w in (1, 2, 4) and scale_h <= 8 and scale_t <= 8:
            block_w_in = max(32, min(1024 // scale_w, _next_power_of_2(W)))
            kernels["upsample_ncdhw_pow2"][(B * C * T * H, _cdiv(W, block_w_in))](
                x_contig,
                y,
                T,
                H,
                W,
                T_out,
                H_out,
                W_out,
                SCALE_T=scale_t,
                SCALE_H=scale_h,
                SCALE_W=scale_w,
                BLOCK_W_IN=block_w_in,
                num_warps=4 if block_w_in * scale_w <= 512 else 8,
                num_stages=3,
            )
            return y

        block_w_out = max(64, min(1024, _next_power_of_2(W_out)))
        kernels["upsample_ncdhw_gen"][(B * C * T_out * H_out, _cdiv(W_out, block_w_out))](
            x_contig,
            y,
            T,
            H,
            W,
            T_out,
            H_out,
            W_out,
            scale_t,
            scale_h,
            scale_w,
            BLOCK_W_OUT=block_w_out,
            num_warps=4,
            num_stages=3,
        )
        return y


# =============================================================================
# PyTorch Reference Fallbacks (CPU / Autograd / Non-CUDA / Unsupported Dtype)
# =============================================================================


def _group_norm_3d_silu_torch(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    out = F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    if apply_silu:
        out = F.silu(out)
    return out


def _upsample_causal_3d_torch(
    x: torch.Tensor,
    scale_t: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    dtype = x.dtype
    hidden = x.to(torch.float32) if dtype == torch.bfloat16 else x
    if hidden.shape[0] >= 64:
        hidden = hidden.contiguous()
    _, _, T, _, _ = hidden.shape
    first_h, other_h = hidden.split((1, T - 1), dim=2)
    if T > 1:
        other_h = F.interpolate(other_h, scale_factor=(scale_t, scale_h, scale_w), mode="nearest")
    first_h = F.interpolate(first_h.squeeze(2), scale_factor=(scale_h, scale_w), mode="nearest").unsqueeze(2)
    out = torch.cat((first_h, other_h), dim=2) if T > 1 else first_h
    if dtype == torch.bfloat16:
        out = out.to(dtype)
    return out


# =============================================================================
# Compile-Safe Custom Operators
# =============================================================================


@torch.library.custom_op("worldfoundry::hunyuan_group_norm_3d_silu", mutates_args=(), device_types="cuda")
def _hunyuan_group_norm_3d_silu_op(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    return _launch_group_norm_3d_silu(x, num_groups, weight, bias, eps, apply_silu)


@_hunyuan_group_norm_3d_silu_op.register_fake
def _hunyuan_group_norm_3d_silu_fake(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    del num_groups, weight, bias, eps, apply_silu
    return _empty_like_layout(x, tuple(x.shape))


@torch.library.custom_op("worldfoundry::hunyuan_upsample_causal_3d", mutates_args=(), device_types="cuda")
def _hunyuan_upsample_causal_3d_op(
    x: torch.Tensor,
    scale_t: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    return _launch_upsample_causal_3d(x, scale_t, scale_h, scale_w)


@_hunyuan_upsample_causal_3d_op.register_fake
def _hunyuan_upsample_causal_3d_fake(
    x: torch.Tensor,
    scale_t: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    B, C, T, H, W = x.shape
    out_shape = (B, C, 1 + (T - 1) * scale_t, H * scale_h, W * scale_w)
    return _empty_like_layout(x, out_shape)


# =============================================================================
# Public Entry Points
# =============================================================================


def _can_use_triton(x: torch.Tensor, *others: torch.Tensor | None) -> bool:
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.compiler.is_compiling() and x.numel() == 0:
        return False
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, *others)):
        return False
    return _get_triton_kernels() is not None


def triton_group_norm_3d_silu(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Stride-aware 5-D GroupNorm + optional SiLU supporting ``NCDHW`` and ``channels_last_3d``."""
    num_groups_i = int(num_groups)
    eps_f = float(eps)
    if (
        x.ndim == 5
        and x.shape[1] % num_groups_i == 0
        and _can_use_triton(x, weight, bias)
        and (weight is None or weight.is_cuda)
        and (bias is None or bias.is_cuda)
    ):
        return _hunyuan_group_norm_3d_silu_op(x, num_groups_i, weight, bias, eps_f, bool(apply_silu))
    return _group_norm_3d_silu_torch(x, num_groups_i, weight, bias, eps_f, bool(apply_silu))


def triton_upsample_causal_3d(
    x: torch.Tensor,
    upsample_factor: tuple[int, int, int] = (2, 2, 2),
) -> torch.Tensor:
    """Single-pass bit-exact causal 3-D nearest upsample for ``NCDHW`` and ``channels_last_3d``."""
    st, sh, sw = (int(upsample_factor[0]), int(upsample_factor[1]), int(upsample_factor[2]))
    if (
        x.ndim == 5
        and st >= 1
        and sh >= 1
        and sw >= 1
        and x.shape[2] >= 1
        and math.prod(x.shape) > 0
        and _can_use_triton(x)
    ):
        return _hunyuan_upsample_causal_3d_op(x, st, sh, sw)
    return _upsample_causal_3d_torch(x, st, sh, sw)


__all__ = ["triton_group_norm_3d_silu", "triton_upsample_causal_3d"]
