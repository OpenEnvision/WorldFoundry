# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper Triton operators for LTX-2.3 / LTX-2.5 VideoVAE.

Implements compile-safe, CUDA-Graph-safe, stride-aware custom operators
supporting both ``NCDHW`` contiguous and ``NDHWC`` (``channels_last_3d``)
tensors without hidden layout copies:

- :func:`triton_pixel_norm_adaln_silu` — single-pass per-voxel channel RMSNorm
  (``x / sqrt(mean(x^2, dim=1) + eps)`` computed in FP32, immune to FP16
  squaring overflow) + optional ``*(1 + scale) + shift`` AdaLN + optional
  ``SiLU``. Also accelerates standalone 4-D/5-D :class:`PixelNorm`.
- :func:`triton_depth_to_space_3d` — single-pass fused ``DepthToSpaceUpsample``
  unpacking ``(B, C*p1*p2*p3, D, H, W) -> (B, C, D*p1 - (1 if p1==2 else 0),
  H*p2, W*p3)`` with optional fused residual-branch unpack + repeat + add.

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


def _effective_5d_strides(t: torch.Tensor | None) -> tuple[int, int, int, int, int]:
    if t is None:
        return (0, 0, 0, 0, 0)
    b, c, d, h, w = t.shape
    sn, sc, sd, sh, sw = t.stride()
    return (
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

        configure_persistent_compile_cache(namespace="ltx-vae-triton")
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_LOAD_FAILED = True
        return None

    # =========================================================================
    # 1. PixelNorm + Optional AdaLN + Optional SiLU — NDHWC & NCDHW
    # =========================================================================

    @triton.jit
    def _pixel_norm_ndhwc_onepass_kernel(
        x_ptr,
        sc_ptr,
        sh_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        S,
        H,
        W,
        C,
        eps,
        sc_sn,
        sc_sc,
        sc_sd,
        sc_sh,
        sc_sw,
        sh_sn,
        sh_sc,
        sh_sd,
        sh_sh,
        sh_sw,
        HAS_ADALN: tl.constexpr,
        SPATIAL_ADALN: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        b_id = tl.program_id(1).to(tl.int64)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        c_offs = tl.arange(0, BLOCK_C).to(tl.int64)
        s_mask = s_offs < S
        c_mask = c_offs < C
        m = s_mask[:, None] & c_mask[None, :]

        x_batch = x_ptr + b_id * stride_xn
        y_batch = y_ptr + b_id * stride_yn

        xv = tl.load(x_batch + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
        mean_sq = tl.sum(xv * xv, axis=1) * (1.0 / C)
        rrms = tl.rsqrt(mean_sq + eps)
        yv = xv * rrms[:, None]

        if HAS_ADALN:
            if not SPATIAL_ADALN:
                sc = tl.load(sc_ptr + b_id * sc_sn + c_offs * sc_sc, mask=c_mask, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + b_id * sh_sn + c_offs * sh_sc, mask=c_mask, other=0.0).to(tl.float32)
                yv = yv * (1.0 + sc[None, :]) + sh[None, :]
            else:
                hw = H * W
                d_coord = s_offs // hw
                rem = s_offs - d_coord * hw
                h_coord = rem // W
                w_coord = rem - h_coord * W
                sc_offs = (
                    b_id * sc_sn
                    + c_offs[None, :] * sc_sc
                    + (d_coord * sc_sd + h_coord * sc_sh + w_coord * sc_sw)[:, None]
                )
                sh_offs = (
                    b_id * sh_sn
                    + c_offs[None, :] * sh_sc
                    + (d_coord * sh_sd + h_coord * sh_sh + w_coord * sh_sw)[:, None]
                )
                sc = tl.load(sc_ptr + sc_offs, mask=m, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + sh_offs, mask=m, other=0.0).to(tl.float32)
                yv = yv * (1.0 + sc) + sh

        if APPLY_SILU:
            yv = yv * tl.sigmoid(yv)

        tl.store(y_batch + s_offs[:, None] * C + c_offs[None, :], yv.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _pixel_norm_ndhwc_twopass_kernel(
        x_ptr,
        sc_ptr,
        sh_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        S,
        H,
        W,
        C,
        eps,
        sc_sn,
        sc_sc,
        sc_sd,
        sc_sh,
        sc_sw,
        sh_sn,
        sh_sc,
        sh_sd,
        sh_sh,
        sh_sw,
        HAS_ADALN: tl.constexpr,
        SPATIAL_ADALN: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        b_id = tl.program_id(1).to(tl.int64)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < S
        c_base_offs = tl.arange(0, BLOCK_C).to(tl.int64)

        x_batch = x_ptr + b_id * stride_xn
        y_batch = y_ptr + b_id * stride_yn

        sum_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + c_base_offs
            m = s_mask[:, None] & (c_offs[None, :] < C)
            xv = tl.load(x_batch + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            sum_sq += tl.sum(xv * xv, axis=1)

        rrms = tl.rsqrt(sum_sq * (1.0 / C) + eps)

        hw = H * W
        if HAS_ADALN and SPATIAL_ADALN:
            d_coord = s_offs // hw
            rem = s_offs - d_coord * hw
            h_coord = rem // W
            w_coord = rem - h_coord * W
            sc_spatial = d_coord * sc_sd + h_coord * sc_sh + w_coord * sc_sw
            sh_spatial = d_coord * sh_sd + h_coord * sh_sh + w_coord * sh_sw

        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + c_base_offs
            c_mask = c_offs < C
            m = s_mask[:, None] & c_mask[None, :]
            xv = tl.load(x_batch + s_offs[:, None] * C + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            yv = xv * rrms[:, None]
            if HAS_ADALN:
                if not SPATIAL_ADALN:
                    sc = tl.load(sc_ptr + b_id * sc_sn + c_offs * sc_sc, mask=c_mask, other=0.0).to(tl.float32)
                    sh = tl.load(sh_ptr + b_id * sh_sn + c_offs * sh_sc, mask=c_mask, other=0.0).to(tl.float32)
                    yv = yv * (1.0 + sc[None, :]) + sh[None, :]
                else:
                    sc_offs = b_id * sc_sn + c_offs[None, :] * sc_sc + sc_spatial[:, None]
                    sh_offs = b_id * sh_sn + c_offs[None, :] * sh_sc + sh_spatial[:, None]
                    sc = tl.load(sc_ptr + sc_offs, mask=m, other=0.0).to(tl.float32)
                    sh = tl.load(sh_ptr + sh_offs, mask=m, other=0.0).to(tl.float32)
                    yv = yv * (1.0 + sc) + sh
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_batch + s_offs[:, None] * C + c_offs[None, :], yv.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _pixel_norm_ncdhw_kernel(
        x_ptr,
        sc_ptr,
        sh_ptr,
        y_ptr,
        S,
        H,
        W,
        C,
        eps,
        sc_sn,
        sc_sc,
        sc_sd,
        sc_sh,
        sc_sw,
        sh_sn,
        sh_sc,
        sh_sd,
        sh_sh,
        sh_sw,
        HAS_ADALN: tl.constexpr,
        SPATIAL_ADALN: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        s_tile = tl.program_id(0).to(tl.int64)
        b_id = tl.program_id(1).to(tl.int64)

        s_offs_i32 = (s_tile * BLOCK_S).to(tl.int32) + tl.arange(0, BLOCK_S)
        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < S

        s_i64 = tl.cast(S, tl.int64)
        batch_base = b_id * (tl.cast(C, tl.int64) * s_i64)
        sum_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for c in range(C):
            xv = tl.load(x_ptr + batch_base + tl.cast(c, tl.int64) * s_i64 + s_offs, mask=s_mask, other=0.0).to(
                tl.float32
            )
            sum_sq += xv * xv

        rrms = tl.rsqrt(sum_sq * (1.0 / C) + eps)

        if HAS_ADALN and SPATIAL_ADALN:
            hw = H * W
            d_coord = (s_offs_i32 // hw).to(tl.int64)
            rem = s_offs_i32 - d_coord.to(tl.int32) * hw
            h_coord = (rem // W).to(tl.int64)
            w_coord = (rem - h_coord.to(tl.int32) * W).to(tl.int64)
            sc_spatial = d_coord * sc_sd + h_coord * sc_sh + w_coord * sc_sw
            sh_spatial = d_coord * sh_sd + h_coord * sh_sh + w_coord * sh_sw

        for c in range(C):
            c_i64 = tl.cast(c, tl.int64)
            xv = tl.load(x_ptr + batch_base + c_i64 * s_i64 + s_offs, mask=s_mask, other=0.0).to(tl.float32)
            yv = xv * rrms
            if HAS_ADALN:
                if not SPATIAL_ADALN:
                    sc = tl.load(sc_ptr + b_id * sc_sn + c_i64 * sc_sc).to(tl.float32)
                    sh = tl.load(sh_ptr + b_id * sh_sn + c_i64 * sh_sc).to(tl.float32)
                    yv = yv * (1.0 + sc) + sh
                else:
                    sc = tl.load(sc_ptr + b_id * sc_sn + c_i64 * sc_sc + sc_spatial, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    sh = tl.load(sh_ptr + b_id * sh_sn + c_i64 * sh_sc + sh_spatial, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    yv = yv * (1.0 + sc) + sh
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(y_ptr + batch_base + c_i64 * s_i64 + s_offs, yv.to(y_ptr.dtype.element_ty), mask=s_mask)

    # =========================================================================
    # 2. DepthToSpace 3D (+ Optional Fused Residual Repeat+Add) — NDHWC & NCDHW
    # =========================================================================

    @triton.jit
    def _d2s_ndhwc_fast_kernel(
        x_ptr,
        res_ptr,
        y_ptr,
        stride_xn,
        stride_rn,
        stride_yn,
        D,
        H,
        W,
        C_in,
        C_res_in,
        C_res_unpacked,
        C_out,
        D_out,
        H_out,
        W_out,
        P1: tl.constexpr,
        P2: tl.constexpr,
        P3: tl.constexpr,
        P: tl.constexpr,
        TRIM_FIRST: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        CONTIG_RES_SLICE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)
        c_tile = tl.program_id(2).to(tl.int64)

        h_in = bdh % H
        bd = bdh // H
        d_in = bd % D
        b_id = bd // D

        w0 = w_tile * BLOCK_W
        c0 = c_tile * BLOCK_C
        w_offs = w0 + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < W
        c_offs = c0 + tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = c_offs < C_out

        # Coalesced 2-D load: (BLOCK_W, BLOCK_C * P)
        cp_offs = c0 * P + tl.arange(0, BLOCK_C * P).to(tl.int64)
        cp_mask = w_mask[:, None] & (cp_offs[None, :] < C_in)
        in_row_base = b_id * stride_xn + (d_in * H + h_in) * (W * C_in)
        v = tl.load(x_ptr + in_row_base + w_offs[:, None] * C_in + cp_offs[None, :], mask=cp_mask)

        if HAS_RESIDUAL:
            res_row_base = b_id * stride_rn + (d_in * H + h_in) * (W * C_res_in)
            if CONTIG_RES_SLICE:
                res_c0 = c0 % C_res_unpacked
                rp_offs = res_c0 * P + tl.arange(0, BLOCK_C * P).to(tl.int64)
                r = tl.load(res_ptr + res_row_base + w_offs[:, None] * C_res_in + rp_offs[None, :], mask=cp_mask)
            else:
                p_offs = tl.arange(0, P).to(tl.int64)
                rc_2d = (c_offs % C_res_unpacked)[:, None] * P + p_offs[None, :]
                rc_flat = tl.reshape(rc_2d, (BLOCK_C * P,))
                r = tl.load(res_ptr + res_row_base + w_offs[:, None] * C_res_in + rc_flat[None, :], mask=cp_mask)
            v = (v.to(tl.float32) + r.to(tl.float32)).to(y_ptr.dtype.element_ty)

        y_batch_base = b_id * stride_yn
        hwc_stride = tl.cast(H_out, tl.int64) * W_out * C_out
        wc_stride = tl.cast(W_out, tl.int64) * C_out
        h_out0 = h_in * P2

        if P3 == 2:
            w_out_offs = w0 * 2 + tl.arange(0, BLOCK_W * 2).to(tl.int64)
            out_mask = (w_out_offs[:, None] < W_out) & c_mask[None, :]
        else:
            w_out_offs = w_offs
            out_mask = w_mask[:, None] & c_mask[None, :]

        if P == 1:
            row_base = y_batch_base + d_in * hwc_stride + h_out0 * wc_stride
            tl.store(y_ptr + row_base + w_out_offs[:, None] * C_out + c_offs[None, :], v, mask=out_mask)
        elif P == 2:
            v0, v1 = tl.split(tl.reshape(v, (BLOCK_W, BLOCK_C, 2)))
            if P1 == 2:
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                    tl.store(y_ptr + b0 + w_out_offs[:, None] * C_out + c_offs[None, :], v0, mask=out_mask)
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b1 + w_out_offs[:, None] * C_out + c_offs[None, :], v1, mask=out_mask)
            elif P2 == 2:
                t_base = y_batch_base + d_in * hwc_stride
                tl.store(
                    y_ptr + t_base + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    v0,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t_base + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    v1,
                    mask=out_mask,
                )
            else:
                # (1, 1, 2): combine (v0, v1) -> (BLOCK_W*2, BLOCK_C) contiguous store
                out_row = tl.reshape(tl.permute(tl.join(v0, v1), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                b0 = y_batch_base + d_in * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b0 + w_out_offs[:, None] * C_out + c_offs[None, :], out_row, mask=out_mask)
        elif P == 4:
            va, vb = tl.split(tl.reshape(v, (BLOCK_W, BLOCK_C, 2, 2)))
            v00, v10 = tl.split(va)
            v01, v11 = tl.split(vb)
            if P1 == 1 and P2 == 2 and P3 == 2:
                row0 = tl.reshape(tl.permute(tl.join(v00, v01), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                row1 = tl.reshape(tl.permute(tl.join(v10, v11), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                t_base = y_batch_base + d_in * hwc_stride
                tl.store(
                    y_ptr + t_base + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    row0,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t_base + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    row1,
                    mask=out_mask,
                )
            elif P1 == 2 and P2 == 2 and P3 == 1:
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride
                    tl.store(
                        y_ptr + b0 + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                        v00,
                        mask=out_mask,
                    )
                    tl.store(
                        y_ptr + b0 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                        v01,
                        mask=out_mask,
                    )
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride
                tl.store(
                    y_ptr + b1 + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    v10,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + b1 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    v11,
                    mask=out_mask,
                )
            else:
                # (2, 1, 2)
                row0 = tl.reshape(tl.permute(tl.join(v00, v01), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                row1 = tl.reshape(tl.permute(tl.join(v10, v11), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                    tl.store(y_ptr + b0 + w_out_offs[:, None] * C_out + c_offs[None, :], row0, mask=out_mask)
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b1 + w_out_offs[:, None] * C_out + c_offs[None, :], row1, mask=out_mask)
        else:
            # P == 8: (2, 2, 2)
            vr3_0, vr3_1 = tl.split(tl.reshape(v, (BLOCK_W, BLOCK_C, 4, 2)))
            vr2_0_r3_0, vr2_1_r3_0 = tl.split(tl.reshape(vr3_0, (BLOCK_W, BLOCK_C, 2, 2)))
            vr2_0_r3_1, vr2_1_r3_1 = tl.split(tl.reshape(vr3_1, (BLOCK_W, BLOCK_C, 2, 2)))
            v000, v100 = tl.split(vr2_0_r3_0)
            v010, v110 = tl.split(vr2_1_r3_0)
            v001, v101 = tl.split(vr2_0_r3_1)
            v011, v111 = tl.split(vr2_1_r3_1)

            d_full0 = d_in * 2
            if d_full0 >= TRIM_FIRST:
                t0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride
                row00 = tl.reshape(tl.permute(tl.join(v000, v001), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                row01 = tl.reshape(tl.permute(tl.join(v010, v011), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                tl.store(
                    y_ptr + t0 + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    row00,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t0 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                    row01,
                    mask=out_mask,
                )

            t1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride
            row10 = tl.reshape(tl.permute(tl.join(v100, v101), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
            row11 = tl.reshape(tl.permute(tl.join(v110, v111), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
            tl.store(
                y_ptr + t1 + h_out0 * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                row10,
                mask=out_mask,
            )
            tl.store(
                y_ptr + t1 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * C_out + c_offs[None, :],
                row11,
                mask=out_mask,
            )

    @triton.jit
    def _d2s_ndhwc_gen_kernel(
        x_ptr,
        res_ptr,
        y_ptr,
        stride_xn,
        stride_yn,
        rs_n,
        rs_c,
        rs_d,
        rs_h,
        rs_w,
        D,
        H,
        W,
        C_in,
        C_res_unpacked,
        C_out,
        D_out,
        H_out,
        W_out,
        p1,
        p2,
        p3,
        trim_first,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bdhw = tl.program_id(0).to(tl.int64)
        c_tile = tl.program_id(1).to(tl.int64)

        w_out = bdhw % W_out
        bdh = bdhw // W_out
        h_out = bdh % H_out
        bd = bdh // H_out
        d_out = bd % D_out
        b_id = bd // D_out

        d_full = d_out + trim_first
        d_in = d_full // p1
        r1 = d_full - d_in * p1
        h_in = h_out // p2
        r2 = h_out - h_in * p2
        w_in = w_out // p3
        r3 = w_out - w_in * p3
        P = p1 * p2 * p3
        sub_idx = (r1 * p2 + r2) * p3 + r3

        c_offs = c_tile * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = c_offs < C_out

        in_voxel_base = b_id * stride_xn + ((d_in * H + h_in) * W + w_in) * C_in
        xv = tl.load(x_ptr + in_voxel_base + c_offs * P + sub_idx, mask=c_mask)

        if HAS_RESIDUAL:
            c_res = (c_offs % C_res_unpacked) * P + sub_idx
            res_offs = b_id * rs_n + c_res * rs_c + d_in * rs_d + h_in * rs_h + w_in * rs_w
            rv = tl.load(res_ptr + res_offs, mask=c_mask)
            xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(y_ptr.dtype.element_ty)

        y_off = b_id * stride_yn + ((d_out * H_out + h_out) * W_out + w_out) * C_out + c_offs
        tl.store(y_ptr + y_off, xv, mask=c_mask)

    @triton.jit
    def _d2s_ncdhw_temporal_kernel(
        x_ptr,
        res_ptr,
        y_ptr,
        D,
        HW,
        C_in,
        C_res_in,
        C_res_unpacked,
        C_out,
        D_out,
        P1: tl.constexpr,
        TRIM_FIRST: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        bcd = tl.program_id(0).to(tl.int64)
        hw_tile = tl.program_id(1).to(tl.int64)

        d_in = bcd % D
        bc = bcd // D
        c_out = bc % C_out
        b_id = bc // C_out

        hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
        mask = hw_offs < HW
        dhw_in = D * HW
        dhw_out = D_out * HW
        x_b_base = b_id * (C_in * dhw_in) + d_in * HW
        y_bc_base = bc * dhw_out

        if HAS_RESIDUAL:
            c_res_base = (c_out % C_res_unpacked) * P1
            r_b_base = b_id * (C_res_in * dhw_in) + d_in * HW

        c_x_base = c_out * P1
        for r1 in tl.static_range(P1):
            d_full = d_in * P1 + r1
            if d_full >= TRIM_FIRST:
                d_out = d_full - TRIM_FIRST
                v0 = tl.load(x_ptr + x_b_base + (c_x_base + r1) * dhw_in + hw_offs, mask=mask)
                if HAS_RESIDUAL:
                    res0 = tl.load(res_ptr + r_b_base + (c_res_base + r1) * dhw_in + hw_offs, mask=mask)
                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + y_bc_base + d_out * HW + hw_offs, v0, mask=mask)

    @triton.jit
    def _d2s_ncdhw_fast_kernel(
        x_ptr,
        res_ptr,
        y_ptr,
        D,
        H,
        W,
        C_in,
        C_res_in,
        C_res_unpacked,
        C_out,
        D_out,
        H_out,
        W_out,
        P1: tl.constexpr,
        P2: tl.constexpr,
        P3: tl.constexpr,
        P: tl.constexpr,
        TRIM_FIRST: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        bcdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        num_h_tiles = tl.cdiv(H, BLOCK_H)
        h_tile = bcdh % num_h_tiles
        bcd = bcdh // num_h_tiles
        d_in = bcd % D
        bc = bcd // D
        c_out = bc % C_out
        b_id = bc // C_out

        w_in_start = w_tile * BLOCK_W
        w_in_offs = w_in_start + tl.arange(0, BLOCK_W).to(tl.int64)
        in_mask = w_in_offs < W

        dhw_in = D * H * W
        dhw_out = D_out * H_out * W_out
        hw_out = H_out * W_out
        x_b_base = b_id * (C_in * dhw_in)
        y_bc_base = bc * dhw_out

        if HAS_RESIDUAL:
            c_res_base = (c_out % C_res_unpacked) * P
            r_b_base = b_id * (C_res_in * dhw_in)

        c_x_base = c_out * P
        h0 = h_tile * BLOCK_H

        for dh_it in tl.static_range(BLOCK_H):
            h_in = h0 + dh_it
            if h_in < H:
                dh_in_off = (d_in * H + h_in) * W
                for r1 in tl.static_range(P1):
                    d_full = d_in * P1 + r1
                    if d_full >= TRIM_FIRST:
                        d_out = d_full - TRIM_FIRST
                        for r2 in tl.static_range(P2):
                            h_out = h_in * P2 + r2
                            if P3 == 2:
                                sub0 = (r1 * P2 + r2) * 2
                                sub1 = sub0 + 1
                                v0 = tl.load(
                                    x_ptr + x_b_base + (c_x_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                    mask=in_mask,
                                )
                                v1 = tl.load(
                                    x_ptr + x_b_base + (c_x_base + sub1) * dhw_in + dh_in_off + w_in_offs,
                                    mask=in_mask,
                                )
                                if HAS_RESIDUAL:
                                    res0 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    res1 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub1) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(y_ptr.dtype.element_ty)
                                    v1 = (v1.to(tl.float32) + res1.to(tl.float32)).to(y_ptr.dtype.element_ty)
                                out_row = tl.interleave(v0, v1)
                                w_out_offs = w_in_start * 2 + tl.arange(0, BLOCK_W * 2).to(tl.int64)
                                out_mask = w_out_offs < W_out
                                tl.store(
                                    y_ptr + y_bc_base + d_out * hw_out + h_out * W_out + w_out_offs,
                                    out_row,
                                    mask=out_mask,
                                )
                            else:
                                sub0 = r1 * P2 + r2
                                v0 = tl.load(
                                    x_ptr + x_b_base + (c_x_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                    mask=in_mask,
                                )
                                if HAS_RESIDUAL:
                                    res0 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(y_ptr.dtype.element_ty)
                                tl.store(
                                    y_ptr + y_bc_base + d_out * hw_out + h_out * W_out + w_in_offs,
                                    v0,
                                    mask=in_mask,
                                )

    @triton.jit
    def _d2s_ncdhw_gen_kernel(
        x_ptr,
        res_ptr,
        y_ptr,
        rs_n,
        rs_c,
        rs_d,
        rs_h,
        rs_w,
        D,
        H,
        W,
        C_in,
        C_res_unpacked,
        C_out,
        D_out,
        H_out,
        W_out,
        p1,
        p2,
        p3,
        trim_first,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_W_OUT: tl.constexpr,
    ):
        bcdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        h_out = bcdh % H_out
        bcd = bcdh // H_out
        d_out = bcd % D_out
        bc = bcd // D_out
        c_out = bc % C_out
        b_id = bc // C_out

        d_full = d_out + trim_first
        d_in = d_full // p1
        r1 = d_full - d_in * p1
        h_in = h_out // p2
        r2 = h_out - h_in * p2

        w_out_offs = w_tile * BLOCK_W_OUT + tl.arange(0, BLOCK_W_OUT).to(tl.int64)
        mask = w_out_offs < W_out
        w_in_offs = w_out_offs // p3
        r3 = w_out_offs - w_in_offs * p3
        P = p1 * p2 * p3
        sub_idx = (r1 * p2 + r2) * p3 + r3

        dhw_in = D * H * W
        c_x = c_out * P + sub_idx
        x_offs = b_id * (C_in * dhw_in) + c_x * dhw_in + (d_in * H + h_in) * W + w_in_offs
        xv = tl.load(x_ptr + x_offs, mask=mask)

        if HAS_RESIDUAL:
            c_res = (c_out % C_res_unpacked) * P + sub_idx
            r_offs = b_id * rs_n + c_res * rs_c + d_in * rs_d + h_in * rs_h + w_in_offs * rs_w
            rv = tl.load(res_ptr + r_offs, mask=mask)
            xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(y_ptr.dtype.element_ty)

        y_off = ((bc * D_out + d_out) * H_out + h_out) * W_out + w_out_offs
        tl.store(y_ptr + y_off, xv, mask=mask)

    _TRITON_KERNELS = {
        "pixel_norm_ndhwc_onepass": _pixel_norm_ndhwc_onepass_kernel,
        "pixel_norm_ndhwc_twopass": _pixel_norm_ndhwc_twopass_kernel,
        "pixel_norm_ncdhw": _pixel_norm_ncdhw_kernel,
        "d2s_ndhwc_fast": _d2s_ndhwc_fast_kernel,
        "d2s_ndhwc_gen": _d2s_ndhwc_gen_kernel,
        "d2s_ncdhw_temporal": _d2s_ncdhw_temporal_kernel,
        "d2s_ncdhw_fast": _d2s_ncdhw_fast_kernel,
        "d2s_ncdhw_gen": _d2s_ncdhw_gen_kernel,
    }
    return _TRITON_KERNELS


# =============================================================================
# Launchers
# =============================================================================


def _launch_pixel_norm_adaln_silu(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _pixel_norm_adaln_silu_torch(x, scale, shift, eps, apply_silu)

    B, C, D, H, W = x.shape
    S = D * H * W
    has_adaln = scale is not None and shift is not None
    sc_t = scale if scale is not None else x
    sh_t = shift if shift is not None else x
    sc_sn, sc_sc, sc_sd, sc_sh, sc_sw = _effective_5d_strides(scale)
    sh_sn, sh_sc, sh_sd, sh_sh, sh_sw = _effective_5d_strides(shift)
    spatial_adaln = has_adaln and (sc_sd != 0 or sc_sh != 0 or sc_sw != 0 or sh_sd != 0 or sh_sh != 0 or sh_sw != 0)

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    with torch.cuda.device(x.device):
        if is_ndhwc:
            y = torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
            stride_xn = x.stride(0)
            stride_yn = y.stride(0)

            if C <= 2048:
                block_c = _next_power_of_2(C)
                block_s = max(1, min(64, 4096 // block_c))
                kernels["pixel_norm_ndhwc_onepass"][(_cdiv(S, block_s), B)](
                    x,
                    sc_t,
                    sh_t,
                    y,
                    stride_xn,
                    stride_yn,
                    S,
                    H,
                    W,
                    C,
                    eps,
                    sc_sn,
                    sc_sc,
                    sc_sd,
                    sc_sh,
                    sc_sw,
                    sh_sn,
                    sh_sc,
                    sh_sd,
                    sh_sh,
                    sh_sw,
                    HAS_ADALN=has_adaln,
                    SPATIAL_ADALN=spatial_adaln,
                    APPLY_SILU=apply_silu,
                    BLOCK_S=block_s,
                    BLOCK_C=block_c,
                    num_warps=8 if block_s * block_c >= 4096 else 4,
                    num_stages=3,
                )
                return y

            block_c = 1024
            block_s = 4
            kernels["pixel_norm_ndhwc_twopass"][(_cdiv(S, block_s), B)](
                x,
                sc_t,
                sh_t,
                y,
                stride_xn,
                stride_yn,
                S,
                H,
                W,
                C,
                eps,
                sc_sn,
                sc_sc,
                sc_sd,
                sc_sh,
                sc_sw,
                sh_sn,
                sh_sc,
                sh_sd,
                sh_sh,
                sh_sw,
                HAS_ADALN=has_adaln,
                SPATIAL_ADALN=spatial_adaln,
                APPLY_SILU=apply_silu,
                BLOCK_S=block_s,
                BLOCK_C=block_c,
                num_warps=8,
                num_stages=3,
            )
            return y

        # NCDHW contiguous — 1-D coalesced channel loop with SM-wave-aware BLOCK_S
        x_contig = x if x.is_contiguous() else x.contiguous()
        y = torch.empty_like(x_contig)
        if B * _cdiv(S, 1024) >= 528:
            block_s = 1024
        elif B * _cdiv(S, 512) >= 528:
            block_s = 512
        elif B * _cdiv(S, 256) >= 528:
            block_s = 256
        else:
            block_s = max(64, min(128, _next_power_of_2(S)))
        kernels["pixel_norm_ncdhw"][(_cdiv(S, block_s), B)](
            x_contig,
            sc_t,
            sh_t,
            y,
            S,
            H,
            W,
            C,
            eps,
            sc_sn,
            sc_sc,
            sc_sd,
            sc_sh,
            sc_sw,
            sh_sn,
            sh_sc,
            sh_sd,
            sh_sh,
            sh_sw,
            HAS_ADALN=has_adaln,
            SPATIAL_ADALN=spatial_adaln,
            APPLY_SILU=apply_silu,
            BLOCK_S=block_s,
            num_warps=8 if block_s >= 1024 else 4,
            num_stages=3,
        )
        return y


def _launch_depth_to_space_3d(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _depth_to_space_3d_torch(x, p1, p2, p3, residual, out_channels_reduction_factor)

    B, C_in, D, H, W = x.shape
    P = p1 * p2 * p3
    C_out = C_in // P
    trim_first = 1 if p1 == 2 else 0
    D_out = D * p1 - trim_first
    H_out = H * p2
    W_out = W * p3
    out_shape = (B, C_out, D_out, H_out, W_out)

    has_residual = residual is not None
    res_t = residual if residual is not None else x
    C_res_in = residual.shape[1] if residual is not None else C_in
    C_res_unpacked = C_res_in // P if residual is not None else C_out

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    res_is_ndhwc = (not has_residual) or _is_channels_last_3d(res_t) or _is_batch_strided_ndhwc(res_t)
    res_is_ncdhw = (not has_residual) or res_t.is_contiguous()

    with torch.cuda.device(x.device):
        if is_ndhwc:
            y = torch.empty(out_shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
            stride_xn = x.stride(0)
            stride_yn = y.stride(0)
            fast_stride = p1 in (1, 2) and p2 in (1, 2) and p3 in (1, 2)

            if fast_stride and res_is_ndhwc:
                max_bc = min(256, max(1, 2048 // P))
                block_c = min(max_bc, _next_power_of_2(C_out))
                if has_residual and (C_res_unpacked & (C_res_unpacked - 1)) == 0 and C_res_unpacked < block_c:
                    block_c = C_res_unpacked
                contig_res_slice = (not has_residual) or (C_res_unpacked % block_c == 0)
                block_w = max(1, min(16, 4096 // max(1, block_c * P)))
                block_w = min(block_w, _next_power_of_2(max(1, W)))
                stride_rn = res_t.stride(0)
                kernels["d2s_ndhwc_fast"][(B * D * H, _cdiv(W, block_w), _cdiv(C_out, block_c))](
                    x,
                    res_t,
                    y,
                    stride_xn,
                    stride_rn,
                    stride_yn,
                    D,
                    H,
                    W,
                    C_in,
                    C_res_in,
                    C_res_unpacked,
                    C_out,
                    D_out,
                    H_out,
                    W_out,
                    P1=p1,
                    P2=p2,
                    P3=p3,
                    P=P,
                    TRIM_FIRST=trim_first,
                    HAS_RESIDUAL=has_residual,
                    CONTIG_RES_SLICE=contig_res_slice,
                    BLOCK_W=block_w,
                    BLOCK_C=block_c,
                    num_warps=8 if block_w * block_c * P >= 4096 else 4,
                    num_stages=3,
                )
                return y

            # General NDHWC path (arbitrary p1/p2/p3 or mixed-layout residual)
            rs_n, rs_c, rs_d, rs_h, rs_w = res_t.stride() if has_residual else (0, 0, 0, 0, 0)
            block_c = min(256, _next_power_of_2(C_out))
            kernels["d2s_ndhwc_gen"][(B * D_out * H_out * W_out, _cdiv(C_out, block_c))](
                x,
                res_t,
                y,
                stride_xn,
                stride_yn,
                rs_n,
                rs_c,
                rs_d,
                rs_h,
                rs_w,
                D,
                H,
                W,
                C_in,
                C_res_unpacked,
                C_out,
                D_out,
                H_out,
                W_out,
                p1,
                p2,
                p3,
                trim_first,
                HAS_RESIDUAL=has_residual,
                BLOCK_C=block_c,
                num_warps=4,
                num_stages=3,
            )
            return y

        # NCDHW contiguous
        x_contig = x if x.is_contiguous() else x.contiguous()
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        fast_stride = p1 in (1, 2) and p2 in (1, 2) and p3 in (1, 2)

        if fast_stride and res_is_ncdhw:
            if p2 == 1 and p3 == 1:
                hw = H * W
                block_hw = max(64, min(2048, _next_power_of_2(hw)))
                kernels["d2s_ncdhw_temporal"][(B * C_out * D, _cdiv(hw, block_hw))](
                    x_contig,
                    res_t,
                    y,
                    D,
                    hw,
                    C_in,
                    C_res_in,
                    C_res_unpacked,
                    C_out,
                    D_out,
                    P1=p1,
                    TRIM_FIRST=trim_first,
                    HAS_RESIDUAL=has_residual,
                    BLOCK_HW=block_hw,
                    num_warps=8 if block_hw >= 1024 else 4,
                    num_stages=3,
                )
                return y

            block_w = max(32, min(256, _next_power_of_2(W)))
            block_h = 4 if H >= 8 else 1
            num_h_tiles = _cdiv(H, block_h)
            kernels["d2s_ncdhw_fast"][(B * C_out * D * num_h_tiles, _cdiv(W, block_w))](
                x_contig,
                res_t,
                y,
                D,
                H,
                W,
                C_in,
                C_res_in,
                C_res_unpacked,
                C_out,
                D_out,
                H_out,
                W_out,
                P1=p1,
                P2=p2,
                P3=p3,
                P=P,
                TRIM_FIRST=trim_first,
                HAS_RESIDUAL=has_residual,
                BLOCK_H=block_h,
                BLOCK_W=block_w,
                num_warps=4,
                num_stages=3,
            )
            return y

        rs_n, rs_c, rs_d, rs_h, rs_w = res_t.stride() if has_residual else (0, 0, 0, 0, 0)
        block_w_out = max(64, min(512, _next_power_of_2(W_out)))
        kernels["d2s_ncdhw_gen"][(B * C_out * D_out * H_out, _cdiv(W_out, block_w_out))](
            x_contig,
            res_t,
            y,
            rs_n,
            rs_c,
            rs_d,
            rs_h,
            rs_w,
            D,
            H,
            W,
            C_in,
            C_res_unpacked,
            C_out,
            D_out,
            H_out,
            W_out,
            p1,
            p2,
            p3,
            trim_first,
            HAS_RESIDUAL=has_residual,
            BLOCK_W_OUT=block_w_out,
            num_warps=4,
            num_stages=3,
        )
        return y


# =============================================================================
# PyTorch Reference Fallbacks
# =============================================================================


def _pixel_norm_adaln_silu_torch(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    mean_sq = torch.mean(x**2, dim=1, keepdim=True)
    out = x / torch.sqrt(mean_sq + eps)
    if scale is not None and shift is not None:
        out = out * (1 + scale) + shift
    if apply_silu:
        out = F.silu(out)
    return out


def _depth_to_space_3d_torch(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
) -> torch.Tensor:
    B, C_in, D, H, W = x.shape
    P = p1 * p2 * p3
    C_out = C_in // P
    out = (
        x.reshape(B, C_out, p1, p2, p3, D, H, W)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(B, C_out, D * p1, H * p2, W * p3)
    )
    if p1 == 2:
        out = out[:, :, 1:, :, :]
    if residual is not None:
        C_res_unpacked = residual.shape[1] // P
        res = (
            residual.reshape(B, C_res_unpacked, p1, p2, p3, D, H, W)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .reshape(B, C_res_unpacked, D * p1, H * p2, W * p3)
        )
        num_repeat = P // out_channels_reduction_factor
        res = res.repeat(1, num_repeat, 1, 1, 1)
        if p1 == 2:
            res = res[:, :, 1:, :, :]
        out = out + res
    return out


# =============================================================================
# Compile-Safe Custom Operators
# =============================================================================


@torch.library.custom_op("worldfoundry::ltx_pixel_norm_adaln_silu", mutates_args=(), device_types="cuda")
def _ltx_pixel_norm_adaln_silu_op(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    return _launch_pixel_norm_adaln_silu(x, scale, shift, eps, apply_silu)


@_ltx_pixel_norm_adaln_silu_op.register_fake
def _ltx_pixel_norm_adaln_silu_fake(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    del scale, shift, eps, apply_silu
    return _empty_like_layout(x, tuple(x.shape))


@torch.library.custom_op("worldfoundry::ltx_depth_to_space_3d", mutates_args=(), device_types="cuda")
def _ltx_depth_to_space_3d_op(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
) -> torch.Tensor:
    return _launch_depth_to_space_3d(x, p1, p2, p3, residual, out_channels_reduction_factor)


@_ltx_depth_to_space_3d_op.register_fake
def _ltx_depth_to_space_3d_fake(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
) -> torch.Tensor:
    del residual, out_channels_reduction_factor
    B, C_in, D, H, W = x.shape
    P = p1 * p2 * p3
    trim_first = 1 if p1 == 2 else 0
    out_shape = (B, C_in // P, D * p1 - trim_first, H * p2, W * p3)
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


def triton_pixel_norm_adaln_silu(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    eps: float = 1e-8,
    *,
    apply_silu: bool = False,
) -> torch.Tensor:
    """Single-pass channel RMSNorm + optional ``*(1 + scale) + shift`` + optional SiLU."""
    eps_f = float(eps)
    if (
        x.ndim == 5
        and (scale is None) == (shift is None)
        and (scale is None or (scale.ndim == 5 and scale.is_cuda))
        and (shift is None or (shift.ndim == 5 and shift.is_cuda))
        and _can_use_triton(x, scale, shift)
    ):
        return _ltx_pixel_norm_adaln_silu_op(x, scale, shift, eps_f, bool(apply_silu))
    if x.ndim == 4 and scale is None and shift is None and _can_use_triton(x):
        x5 = x.unsqueeze(2)
        return _ltx_pixel_norm_adaln_silu_op(x5, None, None, eps_f, bool(apply_silu)).squeeze(2)
    return _pixel_norm_adaln_silu_torch(x, scale, shift, eps_f, bool(apply_silu))


def triton_depth_to_space_3d(
    x: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    residual: torch.Tensor | None = None,
    out_channels_reduction_factor: int = 1,
) -> torch.Tensor:
    """Single-pass fused ``DepthToSpaceUpsample`` with optional residual unpack+repeat+add."""
    p1, p2, p3 = (int(stride[0]), int(stride[1]), int(stride[2]))
    P = p1 * p2 * p3
    red = int(out_channels_reduction_factor)
    if (
        x.ndim == 5
        and p1 >= 1
        and p2 >= 1
        and p3 >= 1
        and x.shape[1] % P == 0
        and (x.shape[2] * p1 > (1 if p1 == 2 else 0))
        and math.prod(x.shape) > 0
        and (
            residual is None
            or (
                residual.ndim == 5
                and residual.is_cuda
                and residual.shape[0] == x.shape[0]
                and residual.shape[2:] == x.shape[2:]
                and residual.shape[1] % P == 0
                and P % red == 0
                and (residual.shape[1] // P) * (P // red) == x.shape[1] // P
            )
        )
        and _can_use_triton(x, residual)
    ):
        return _ltx_depth_to_space_3d_op(x, p1, p2, p3, residual, red)
    return _depth_to_space_3d_torch(x, p1, p2, p3, residual, red)


__all__ = ["triton_depth_to_space_3d", "triton_pixel_norm_adaln_silu"]
