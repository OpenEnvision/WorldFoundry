# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper SM90 Triton operators for LTX-2.3 / LTX-2.5 Conv VideoVAE.

Implements compile-safe, CUDA-Graph-safe, strictly ``int64``-indexed custom
operators supporting both ``NCDHW`` contiguous and ``NDHWC``
(``channels_last_3d``) tensors without hidden layout copies:

- :func:`triton_pixel_norm_adaln_silu` — single-pass per-voxel channel RMSNorm
  (``x / sqrt(mean(x^2, dim=1) + eps)`` computed in FP32, immune to FP16
  squaring overflow) + optional ``*(1 + scale) + shift`` AdaLN + optional
  ``SiLU``. Also accelerates standalone 4-D/5-D :class:`PixelNorm`.
- :func:`triton_pixel_norm_adaln_silu_pad_3d` — super-fused ``(optional
  pre_bias) + PixelNorm + (optional AdaLN) + (optional SiLU) + 3-D pad``
  feeding ``CausalConv3d.conv`` directly (eliminates preceding ``conv1.bias``
  pass and temporal/spatial padding copies).
- :func:`triton_ltx_pad_3d` — bit-exact, layout-preserving 5-D pad supporting
  independent spatial & temporal modes.
- :func:`triton_ltx_pad_3d_align_c` — fused 3-D pad + channel zero-padding to
  an 8-channel Tensor Core boundary for ``in_channels < 8`` ``channels_last_3d``
  convolutions.
- :func:`triton_ltx_residual_bias_add` — single-pass coalesced
  ``x + (h + bias)`` (or ``h + bias`` when ``x is None``) eliminating PyTorch's
  unvectorized 5-D ``Conv3d`` bias broadcast pass.
- :func:`triton_depth_to_space_3d` — single-pass fused ``DepthToSpaceUpsample``
  with optional deferred ``h_bias`` and residual-branch unpack + repeat + add.
- :func:`triton_space_to_depth_3d_add` — single-pass fused
  ``SpaceToDepthDownsample`` with optional deferred ``h_bias``, causal
  first-frame duplication, and shortcut group-mean residual add.
- :func:`triton_patchify_3d` / :func:`triton_unpatchify_3d` — layout-aware
  5-D spatial/temporal patchify (direct ``channels_last_3d`` output) and
  unpatchify with optional fused ``conv_out.bias``.

Triton is imported lazily on first CUDA invocation; CPU, autograd, and
unsupported dtypes/shapes fall back transparently to exact PyTorch math.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MODE_TO_CODE = {"constant": 0, "zeros": 0, "replicate": 1, "reflect": 2}
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

        configure_persistent_compile_cache(namespace="ltx-vae-triton-v2")
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_LOAD_FAILED = True
        return None

    @triton.jit
    def _map_coord_i64(idx, pad_left, size, mode: tl.constexpr):
        raw = idx - pad_left
        if mode == 0:  # constant zero
            valid = (raw >= 0) & (raw < size)
            coord = tl.maximum(tl.minimum(raw, size - 1), 0)
            return coord, valid
        elif mode == 1:  # replicate
            coord = tl.maximum(tl.minimum(raw, size - 1), 0)
            return coord, raw == raw
        else:  # reflect (guaranteed pad_left < size and pad_right < size)
            coord = tl.where(raw < 0, -raw, raw)
            coord = tl.where(coord >= size, 2 * (size - 1) - coord, coord)
            return coord, raw == raw

    # =========================================================================
    # 1. PixelNorm + Optional AdaLN + Optional SiLU — NDHWC & NCDHW (int64)
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

        s_i64 = tl.cast(S, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        sxn = tl.cast(stride_xn, tl.int64)
        syn = tl.cast(stride_yn, tl.int64)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        c_offs = tl.arange(0, BLOCK_C).to(tl.int64)
        s_mask = s_offs < s_i64
        c_mask = c_offs < c_i64
        m = s_mask[:, None] & c_mask[None, :]

        x_batch = x_ptr + b_id * sxn
        y_batch = y_ptr + b_id * syn

        xv = tl.load(x_batch + s_offs[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
        mean_sq = tl.sum(xv * xv, axis=1) * (1.0 / C)
        rrms = tl.rsqrt(mean_sq + eps)
        yv = xv * rrms[:, None]

        if HAS_ADALN:
            sc_sn_i64 = tl.cast(sc_sn, tl.int64)
            sc_sc_i64 = tl.cast(sc_sc, tl.int64)
            sh_sn_i64 = tl.cast(sh_sn, tl.int64)
            sh_sc_i64 = tl.cast(sh_sc, tl.int64)
            if not SPATIAL_ADALN:
                sc = tl.load(sc_ptr + b_id * sc_sn_i64 + c_offs * sc_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + b_id * sh_sn_i64 + c_offs * sh_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                yv = yv * (1.0 + sc[None, :]) + sh[None, :]
            else:
                hw = h_i64 * w_i64
                d_coord = s_offs // hw
                rem = s_offs - d_coord * hw
                h_coord = rem // w_i64
                w_coord = rem - h_coord * w_i64
                sc_offs = (
                    b_id * sc_sn_i64
                    + c_offs[None, :] * sc_sc_i64
                    + (
                        d_coord * tl.cast(sc_sd, tl.int64)
                        + h_coord * tl.cast(sc_sh, tl.int64)
                        + w_coord * tl.cast(sc_sw, tl.int64)
                    )[:, None]
                )
                sh_offs = (
                    b_id * sh_sn_i64
                    + c_offs[None, :] * sh_sc_i64
                    + (
                        d_coord * tl.cast(sh_sd, tl.int64)
                        + h_coord * tl.cast(sh_sh, tl.int64)
                        + w_coord * tl.cast(sh_sw, tl.int64)
                    )[:, None]
                )
                sc = tl.load(sc_ptr + sc_offs, mask=m, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + sh_offs, mask=m, other=0.0).to(tl.float32)
                yv = yv * (1.0 + sc) + sh

        if APPLY_SILU:
            yv = yv * tl.sigmoid(yv)

        tl.store(
            y_batch + s_offs[:, None] * c_i64 + c_offs[None, :],
            yv.to(y_ptr.dtype.element_ty),
            mask=m,
            eviction_policy="evict_first",
        )

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

        s_i64 = tl.cast(S, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        sxn = tl.cast(stride_xn, tl.int64)
        syn = tl.cast(stride_yn, tl.int64)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64
        c_base_offs = tl.arange(0, BLOCK_C).to(tl.int64)

        x_batch = x_ptr + b_id * sxn
        y_batch = y_ptr + b_id * syn

        sum_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            c_offs = tl.cast(c_start, tl.int64) + c_base_offs
            m = s_mask[:, None] & (c_offs[None, :] < c_i64)
            xv = tl.load(x_batch + s_offs[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            sum_sq += tl.sum(xv * xv, axis=1)

        rrms = tl.rsqrt(sum_sq * (1.0 / C) + eps)

        sc_sn_i64 = tl.cast(sc_sn, tl.int64)
        sc_sc_i64 = tl.cast(sc_sc, tl.int64)
        sh_sn_i64 = tl.cast(sh_sn, tl.int64)
        sh_sc_i64 = tl.cast(sh_sc, tl.int64)
        if HAS_ADALN and SPATIAL_ADALN:
            hw = h_i64 * w_i64
            d_coord = s_offs // hw
            rem = s_offs - d_coord * hw
            h_coord = rem // w_i64
            w_coord = rem - h_coord * w_i64
            sc_spatial = (
                d_coord * tl.cast(sc_sd, tl.int64)
                + h_coord * tl.cast(sc_sh, tl.int64)
                + w_coord * tl.cast(sc_sw, tl.int64)
            )
            sh_spatial = (
                d_coord * tl.cast(sh_sd, tl.int64)
                + h_coord * tl.cast(sh_sh, tl.int64)
                + w_coord * tl.cast(sh_sw, tl.int64)
            )

        for c_start in range(0, C, BLOCK_C):
            c_offs = tl.cast(c_start, tl.int64) + c_base_offs
            c_mask = c_offs < c_i64
            m = s_mask[:, None] & c_mask[None, :]
            xv = tl.load(x_batch + s_offs[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0).to(tl.float32)
            yv = xv * rrms[:, None]
            if HAS_ADALN:
                if not SPATIAL_ADALN:
                    sc = tl.load(sc_ptr + b_id * sc_sn_i64 + c_offs * sc_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                    sh = tl.load(sh_ptr + b_id * sh_sn_i64 + c_offs * sh_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                    yv = yv * (1.0 + sc[None, :]) + sh[None, :]
                else:
                    sc_offs = b_id * sc_sn_i64 + c_offs[None, :] * sc_sc_i64 + sc_spatial[:, None]
                    sh_offs = b_id * sh_sn_i64 + c_offs[None, :] * sh_sc_i64 + sh_spatial[:, None]
                    sc = tl.load(sc_ptr + sc_offs, mask=m, other=0.0).to(tl.float32)
                    sh = tl.load(sh_ptr + sh_offs, mask=m, other=0.0).to(tl.float32)
                    yv = yv * (1.0 + sc) + sh
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(
                y_batch + s_offs[:, None] * c_i64 + c_offs[None, :],
                yv.to(y_ptr.dtype.element_ty),
                mask=m,
                eviction_policy="evict_first",
            )

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

        s_i64 = tl.cast(S, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_i64 = tl.cast(C, tl.int64)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64

        batch_base = b_id * (c_i64 * s_i64)
        sum_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for c in range(C):
            c_idx = tl.cast(c, tl.int64)
            xv = tl.load(x_ptr + batch_base + c_idx * s_i64 + s_offs, mask=s_mask, other=0.0).to(tl.float32)
            sum_sq += xv * xv

        rrms = tl.rsqrt(sum_sq * (1.0 / C) + eps)

        sc_sn_i64 = tl.cast(sc_sn, tl.int64)
        sc_sc_i64 = tl.cast(sc_sc, tl.int64)
        sh_sn_i64 = tl.cast(sh_sn, tl.int64)
        sh_sc_i64 = tl.cast(sh_sc, tl.int64)
        if HAS_ADALN and SPATIAL_ADALN:
            hw = h_i64 * w_i64
            d_coord = s_offs // hw
            rem = s_offs - d_coord * hw
            h_coord = rem // w_i64
            w_coord = rem - h_coord * w_i64
            sc_spatial = (
                d_coord * tl.cast(sc_sd, tl.int64)
                + h_coord * tl.cast(sc_sh, tl.int64)
                + w_coord * tl.cast(sc_sw, tl.int64)
            )
            sh_spatial = (
                d_coord * tl.cast(sh_sd, tl.int64)
                + h_coord * tl.cast(sh_sh, tl.int64)
                + w_coord * tl.cast(sh_sw, tl.int64)
            )

        for c in range(C):
            c_idx = tl.cast(c, tl.int64)
            xv = tl.load(x_ptr + batch_base + c_idx * s_i64 + s_offs, mask=s_mask, other=0.0).to(tl.float32)
            yv = xv * rrms
            if HAS_ADALN:
                if not SPATIAL_ADALN:
                    sc = tl.load(sc_ptr + b_id * sc_sn_i64 + c_idx * sc_sc_i64).to(tl.float32)
                    sh = tl.load(sh_ptr + b_id * sh_sn_i64 + c_idx * sh_sc_i64).to(tl.float32)
                    yv = yv * (1.0 + sc) + sh
                else:
                    sc = tl.load(sc_ptr + b_id * sc_sn_i64 + c_idx * sc_sc_i64 + sc_spatial, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    sh = tl.load(sh_ptr + b_id * sh_sn_i64 + c_idx * sh_sc_i64 + sh_spatial, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    yv = yv * (1.0 + sc) + sh
            if APPLY_SILU:
                yv = yv * tl.sigmoid(yv)
            tl.store(
                y_ptr + batch_base + c_idx * s_i64 + s_offs,
                yv.to(y_ptr.dtype.element_ty),
                mask=s_mask,
                eviction_policy="evict_first",
            )

    # =========================================================================
    # 2. Super-fused (pre_bias +) PixelNorm (+ AdaLN) (+ SiLU) + 3-D Pad
    # =========================================================================

    @triton.jit
    def _pixel_norm_pad_3d_ndhwc_kernel(
        x_ptr,
        pre_bias_ptr,
        sc_ptr,
        sh_ptr,
        out_ptr,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
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
        eps,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_adaln: tl.constexpr,
        spatial_adaln: tl.constexpr,
        apply_silu: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_in_bounds = w_offs < w_out_i64
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < channels
        out_mask = w_in_bounds[:, None] & c_mask[None, :]

        out_base = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64
        out_ptrs = out_base + w_offs[:, None] * so_w_i64 + c_offs[None, :]

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, pad_h_l_i64, h_in_i64, s_mode)

        if (not d_valid) or (not h_valid):
            tl.store(
                out_ptr + out_ptrs,
                tl.zeros((block_w, block_c), dtype=out_ptr.dtype.element_ty),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            return

        w_i, w_valid = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)
        load_mask = (w_in_bounds & w_valid)[:, None] & c_mask[None, :]

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)

        in_base = b_idx * sx_b_i64 + d_i * sx_d_i64 + h_i * sx_h_i64
        in_ptrs = in_base + w_i[:, None] * sx_w_i64 + c_offs[None, :]

        x_in = tl.load(x_ptr + in_ptrs, mask=load_mask, other=0.0)
        x_f = x_in.to(tl.float32)
        if has_pre_bias:
            pre_b = tl.load(pre_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            x_f = (x_f + pre_b).to(x_in.dtype).to(tl.float32)

        mean_sq = tl.sum(x_f * x_f, axis=1) * (1.0 / channels)
        rrms = tl.rsqrt(mean_sq + eps)
        y = x_f * rrms[:, None]

        if has_adaln:
            sc_sn_i64 = tl.cast(sc_sn, tl.int64)
            sc_sc_i64 = tl.cast(sc_sc, tl.int64)
            sh_sn_i64 = tl.cast(sh_sn, tl.int64)
            sh_sc_i64 = tl.cast(sh_sc, tl.int64)
            if not spatial_adaln:
                sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_offs * sc_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_offs * sh_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                y = y * (1.0 + sc[None, :]) + sh[None, :]
            else:
                sc_offs = (
                    b_idx * sc_sn_i64
                    + c_offs[None, :] * sc_sc_i64
                    + (
                        d_i * tl.cast(sc_sd, tl.int64) + h_i * tl.cast(sc_sh, tl.int64) + w_i * tl.cast(sc_sw, tl.int64)
                    )[:, None]
                )
                sh_offs = (
                    b_idx * sh_sn_i64
                    + c_offs[None, :] * sh_sc_i64
                    + (
                        d_i * tl.cast(sh_sd, tl.int64) + h_i * tl.cast(sh_sh, tl.int64) + w_i * tl.cast(sh_sw, tl.int64)
                    )[:, None]
                )
                sc = tl.load(sc_ptr + sc_offs, mask=load_mask, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + sh_offs, mask=load_mask, other=0.0).to(tl.float32)
                y = y * (1.0 + sc) + sh

        if apply_silu:
            y = y * tl.sigmoid(y)

        y_out = y.to(out_ptr.dtype.element_ty)
        if s_mode == 0:
            zero_v = tl.zeros((1, 1), dtype=out_ptr.dtype.element_ty)
            y_out = tl.where(w_valid[:, None], y_out, zero_v)
        tl.store(out_ptr + out_ptrs, y_out, mask=out_mask, eviction_policy="evict_first")

    @triton.jit
    def _pixel_norm_tpad_3d_ndhwc_kernel(
        x_ptr,
        pre_bias_ptr,
        sc_ptr,
        sh_ptr,
        out_ptr,
        d_in,
        d_out,
        hw_dim,
        w_dim,
        pad_t_l,
        sx_b,
        sx_d,
        so_b,
        so_d,
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
        eps,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_adaln: tl.constexpr,
        spatial_adaln: tl.constexpr,
        apply_silu: tl.constexpr,
        block_s: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Contiguous-(H,W) fast path when spatial pad is 0 (LTX-2.3/2.5 zeros mode)."""
        s_tile = tl.program_id(0).to(tl.int64)
        bd = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        hw_i64 = tl.cast(hw_dim, tl.int64)
        w_i64 = tl.cast(w_dim, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)

        b_idx = bd // d_out_i64
        d_o = bd - b_idx * d_out_i64

        s_offs = s_tile * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < hw_i64
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        m = s_mask[:, None] & c_mask[None, :]

        out_base = b_idx * tl.cast(so_b, tl.int64) + d_o * tl.cast(so_d, tl.int64)
        out_ptrs = out_base + s_offs[:, None] * c_i64 + c_offs[None, :]

        d_i, d_valid = _map_coord_i64(d_o, tl.cast(pad_t_l, tl.int64), d_in_i64, t_mode)
        if not d_valid:
            tl.store(
                out_ptr + out_ptrs,
                tl.zeros((block_s, block_c), dtype=out_ptr.dtype.element_ty),
                mask=m,
                eviction_policy="evict_first",
            )
            return

        in_base = b_idx * tl.cast(sx_b, tl.int64) + d_i * tl.cast(sx_d, tl.int64)
        x_in = tl.load(x_ptr + in_base + s_offs[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0)
        x_f = x_in.to(tl.float32)
        if has_pre_bias:
            pre_b = tl.load(pre_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            x_f = (x_f + pre_b).to(x_in.dtype).to(tl.float32)

        rrms = tl.rsqrt(tl.sum(x_f * x_f, axis=1) * (1.0 / channels) + eps)
        y = x_f * rrms[:, None]

        if has_adaln:
            sc_sn_i64 = tl.cast(sc_sn, tl.int64)
            sc_sc_i64 = tl.cast(sc_sc, tl.int64)
            sh_sn_i64 = tl.cast(sh_sn, tl.int64)
            sh_sc_i64 = tl.cast(sh_sc, tl.int64)
            if not spatial_adaln:
                sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_offs * sc_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_offs * sh_sc_i64, mask=c_mask, other=0.0).to(tl.float32)
                y = y * (1.0 + sc[None, :]) + sh[None, :]
            else:
                h_i = s_offs // w_i64
                w_i = s_offs - h_i * w_i64
                sc_offs = (
                    b_idx * sc_sn_i64
                    + c_offs[None, :] * sc_sc_i64
                    + (
                        d_i * tl.cast(sc_sd, tl.int64) + h_i * tl.cast(sc_sh, tl.int64) + w_i * tl.cast(sc_sw, tl.int64)
                    )[:, None]
                )
                sh_offs = (
                    b_idx * sh_sn_i64
                    + c_offs[None, :] * sh_sc_i64
                    + (
                        d_i * tl.cast(sh_sd, tl.int64) + h_i * tl.cast(sh_sh, tl.int64) + w_i * tl.cast(sh_sw, tl.int64)
                    )[:, None]
                )
                sc = tl.load(sc_ptr + sc_offs, mask=m, other=0.0).to(tl.float32)
                sh = tl.load(sh_ptr + sh_offs, mask=m, other=0.0).to(tl.float32)
                y = y * (1.0 + sc) + sh

        if apply_silu:
            y = y * tl.sigmoid(y)
        tl.store(out_ptr + out_ptrs, y.to(out_ptr.dtype.element_ty), mask=m, eviction_policy="evict_first")

    @triton.jit
    def _pixel_norm_tpad_3d_ncdhw_kernel(
        x_ptr,
        pre_bias_ptr,
        sc_ptr,
        sh_ptr,
        out_ptr,
        d_in,
        d_out,
        hw_dim,
        w_dim,
        pad_t_l,
        sx_b,
        sx_c,
        sx_d,
        so_b,
        so_c,
        so_d,
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
        eps,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_adaln: tl.constexpr,
        spatial_adaln: tl.constexpr,
        apply_silu: tl.constexpr,
        block_s: tl.constexpr,
    ):
        """Contiguous-(H,W) NCDHW fast path when spatial pad is 0."""
        s_tile = tl.program_id(0).to(tl.int64)
        bd = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        hw_i64 = tl.cast(hw_dim, tl.int64)
        w_i64 = tl.cast(w_dim, tl.int64)

        b_idx = bd // d_out_i64
        d_o = bd - b_idx * d_out_i64

        s_offs = s_tile * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < hw_i64

        sx_c_i64 = tl.cast(sx_c, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        out_base = b_idx * tl.cast(so_b, tl.int64) + d_o * tl.cast(so_d, tl.int64)

        d_i, d_valid = _map_coord_i64(d_o, tl.cast(pad_t_l, tl.int64), d_in_i64, t_mode)
        if not d_valid:
            zeros = tl.zeros((block_s,), dtype=out_ptr.dtype.element_ty)
            for c in range(channels):
                tl.store(
                    out_ptr + out_base + tl.cast(c, tl.int64) * so_c_i64 + s_offs,
                    zeros,
                    mask=s_mask,
                    eviction_policy="evict_first",
                )
            return

        in_base = b_idx * tl.cast(sx_b, tl.int64) + d_i * tl.cast(sx_d, tl.int64)
        sq_sum = tl.zeros((block_s,), dtype=tl.float32)
        for c in range(channels):
            c_i64 = tl.cast(c, tl.int64)
            v_raw = tl.load(x_ptr + in_base + c_i64 * sx_c_i64 + s_offs, mask=s_mask, other=0.0)
            v = v_raw.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                v = (v + pb).to(v_raw.dtype).to(tl.float32)
            sq_sum += v * v
        rrms = tl.rsqrt(sq_sum * (1.0 / channels) + eps)

        sc_sn_i64 = tl.cast(sc_sn, tl.int64)
        sc_sc_i64 = tl.cast(sc_sc, tl.int64)
        sh_sn_i64 = tl.cast(sh_sn, tl.int64)
        sh_sc_i64 = tl.cast(sh_sc, tl.int64)
        if has_adaln and spatial_adaln:
            h_i = s_offs // w_i64
            w_i = s_offs - h_i * w_i64
            sc_sp = d_i * tl.cast(sc_sd, tl.int64) + h_i * tl.cast(sc_sh, tl.int64) + w_i * tl.cast(sc_sw, tl.int64)
            sh_sp = d_i * tl.cast(sh_sd, tl.int64) + h_i * tl.cast(sh_sh, tl.int64) + w_i * tl.cast(sh_sw, tl.int64)

        for c in range(channels):
            c_i64 = tl.cast(c, tl.int64)
            x_in = tl.load(x_ptr + in_base + c_i64 * sx_c_i64 + s_offs, mask=s_mask, other=0.0)
            x_f = x_in.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                x_f = (x_f + pb).to(x_in.dtype).to(tl.float32)
            y = x_f * rrms
            if has_adaln:
                if not spatial_adaln:
                    sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64).to(tl.float32)
                    sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64).to(tl.float32)
                    y = y * (1.0 + sc) + sh
                else:
                    sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64 + sc_sp, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64 + sh_sp, mask=s_mask, other=0.0).to(
                        tl.float32
                    )
                    y = y * (1.0 + sc) + sh
            if apply_silu:
                y = y * tl.sigmoid(y)
            tl.store(
                out_ptr + out_base + c_i64 * so_c_i64 + s_offs,
                y.to(out_ptr.dtype.element_ty),
                mask=s_mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _pixel_norm_pad_3d_ncdhw_kernel(
        x_ptr,
        pre_bias_ptr,
        sc_ptr,
        sh_ptr,
        out_ptr,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
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
        eps,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_adaln: tl.constexpr,
        spatial_adaln: tl.constexpr,
        apply_silu: tl.constexpr,
        block_w: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        sc_sn_i64 = tl.cast(sc_sn, tl.int64)
        sc_sc_i64 = tl.cast(sc_sc, tl.int64)
        sh_sn_i64 = tl.cast(sh_sn, tl.int64)
        sh_sc_i64 = tl.cast(sh_sc, tl.int64)

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, pad_h_l_i64, h_in_i64, s_mode)

        in_base = b_idx * sx_b_i64 + d_i * sx_d_i64 + h_i * sx_h_i64
        out_base = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64

        num_interior_w = (w_in_i64 + block_w - 1) // block_w
        if w_blk < num_interior_w:
            w_in_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
            w_mask = w_in_offs < w_in_i64
            w_out_offs = pad_w_l_i64 + w_in_offs

            if (not d_valid) or (not h_valid):
                zeros = tl.zeros((block_w,), dtype=out_ptr.dtype.element_ty)
                for c in range(channels):
                    tl.store(
                        out_ptr + out_base + tl.cast(c, tl.int64) * so_c_i64 + w_out_offs,
                        zeros,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )
                return

            sq_sum = tl.zeros((block_w,), dtype=tl.float32)
            for c in range(channels):
                c_i64 = tl.cast(c, tl.int64)
                v_raw = tl.load(
                    x_ptr + in_base + c_i64 * sx_c_i64 + w_in_offs,
                    mask=w_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                v = v_raw.to(tl.float32)
                if has_pre_bias:
                    pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                    v = (v + pb).to(v_raw.dtype).to(tl.float32)
                sq_sum += v * v
            rrms = tl.rsqrt(sq_sum * (1.0 / channels) + eps)

            if has_adaln and spatial_adaln:
                sc_sp = (
                    d_i * tl.cast(sc_sd, tl.int64)
                    + h_i * tl.cast(sc_sh, tl.int64)
                    + w_in_offs * tl.cast(sc_sw, tl.int64)
                )
                sh_sp = (
                    d_i * tl.cast(sh_sd, tl.int64)
                    + h_i * tl.cast(sh_sh, tl.int64)
                    + w_in_offs * tl.cast(sh_sw, tl.int64)
                )

            for c in range(channels):
                c_i64 = tl.cast(c, tl.int64)
                x_in = tl.load(
                    x_ptr + in_base + c_i64 * sx_c_i64 + w_in_offs,
                    mask=w_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                x_f = x_in.to(tl.float32)
                if has_pre_bias:
                    pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                    x_f = (x_f + pb).to(x_in.dtype).to(tl.float32)
                y = x_f * rrms
                if has_adaln:
                    if not spatial_adaln:
                        sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64).to(tl.float32)
                        sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64).to(tl.float32)
                        y = y * (1.0 + sc) + sh
                    else:
                        sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64 + sc_sp, mask=w_mask, other=0.0).to(
                            tl.float32
                        )
                        sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64 + sh_sp, mask=w_mask, other=0.0).to(
                            tl.float32
                        )
                        y = y * (1.0 + sc) + sh
                if apply_silu:
                    y = y * tl.sigmoid(y)
                tl.store(
                    out_ptr + out_base + c_i64 * so_c_i64 + w_out_offs,
                    y.to(out_ptr.dtype.element_ty),
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
        else:
            pad_w_r_i64 = w_out_i64 - pad_w_l_i64 - w_in_i64
            total_border = pad_w_l_i64 + pad_w_r_i64
            for b_start in range(0, total_border, block_w):
                b_idx_w = tl.cast(b_start, tl.int64) + tl.arange(0, block_w).to(tl.int64)
                b_in_bounds = b_idx_w < total_border
                w_offs = tl.where(b_idx_w < pad_w_l_i64, b_idx_w, pad_w_l_i64 + w_in_i64 + (b_idx_w - pad_w_l_i64))
                if (not d_valid) or (not h_valid) or (s_mode == 0):
                    zeros = tl.zeros((block_w,), dtype=out_ptr.dtype.element_ty)
                    for c in range(channels):
                        tl.store(out_ptr + out_base + tl.cast(c, tl.int64) * so_c_i64 + w_offs, zeros, mask=b_in_bounds)
                else:
                    w_i, _ = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)
                    sq_sum = tl.zeros((block_w,), dtype=tl.float32)
                    for c in range(channels):
                        c_i64 = tl.cast(c, tl.int64)
                        v_raw = tl.load(x_ptr + in_base + c_i64 * sx_c_i64 + w_i, mask=b_in_bounds, other=0.0)
                        v = v_raw.to(tl.float32)
                        if has_pre_bias:
                            pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                            v = (v + pb).to(v_raw.dtype).to(tl.float32)
                        sq_sum += v * v
                    rrms = tl.rsqrt(sq_sum * (1.0 / channels) + eps)

                    if has_adaln and spatial_adaln:
                        sc_sp = (
                            d_i * tl.cast(sc_sd, tl.int64)
                            + h_i * tl.cast(sc_sh, tl.int64)
                            + w_i * tl.cast(sc_sw, tl.int64)
                        )
                        sh_sp = (
                            d_i * tl.cast(sh_sd, tl.int64)
                            + h_i * tl.cast(sh_sh, tl.int64)
                            + w_i * tl.cast(sh_sw, tl.int64)
                        )

                    for c in range(channels):
                        c_i64 = tl.cast(c, tl.int64)
                        x_in = tl.load(x_ptr + in_base + c_i64 * sx_c_i64 + w_i, mask=b_in_bounds, other=0.0)
                        x_f = x_in.to(tl.float32)
                        if has_pre_bias:
                            pb = tl.load(pre_bias_ptr + c_i64).to(tl.float32)
                            x_f = (x_f + pb).to(x_in.dtype).to(tl.float32)
                        y = x_f * rrms
                        if has_adaln:
                            if not spatial_adaln:
                                sc = tl.load(sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64).to(tl.float32)
                                sh = tl.load(sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64).to(tl.float32)
                                y = y * (1.0 + sc) + sh
                            else:
                                sc = tl.load(
                                    sc_ptr + b_idx * sc_sn_i64 + c_i64 * sc_sc_i64 + sc_sp, mask=b_in_bounds, other=0.0
                                ).to(tl.float32)
                                sh = tl.load(
                                    sh_ptr + b_idx * sh_sn_i64 + c_i64 * sh_sc_i64 + sh_sp, mask=b_in_bounds, other=0.0
                                ).to(tl.float32)
                                y = y * (1.0 + sc) + sh
                        if apply_silu:
                            y = y * tl.sigmoid(y)
                        tl.store(
                            out_ptr + out_base + c_i64 * so_c_i64 + w_offs,
                            y.to(out_ptr.dtype.element_ty),
                            mask=b_in_bounds,
                        )

    # =========================================================================
    # 3. Bit-exact 5-D Pad (NDHWC & NCDHW) & Pad + Align-C (int64)
    # =========================================================================

    @triton.jit
    def _ltx_pad_3d_ndhwc_kernel(
        x_ptr,
        out_ptr,
        c_dim,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        c_i64 = tl.cast(c_dim, tl.int64)
        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        d_i, d_valid = _map_coord_i64(d_o, tl.cast(pad_t_l, tl.int64), d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, tl.cast(pad_h_l, tl.int64), h_in_i64, s_mode)

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_in_bounds = w_offs < w_out_i64
        w_i, w_valid = _map_coord_i64(w_offs, tl.cast(pad_w_l, tl.int64), w_in_i64, s_mode)

        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64

        load_mask = (d_valid & h_valid & w_in_bounds & w_valid)[:, None] & c_mask[None, :]
        store_mask = w_in_bounds[:, None] & c_mask[None, :]

        in_base = b_idx * tl.cast(sx_b, tl.int64) + d_i * tl.cast(sx_d, tl.int64) + h_i * tl.cast(sx_h, tl.int64)
        in_ptrs = in_base + w_i[:, None] * tl.cast(sx_w, tl.int64) + c_offs[None, :]
        vals = tl.load(x_ptr + in_ptrs, mask=load_mask, other=0.0, eviction_policy="evict_first")

        out_base = b_idx * tl.cast(so_b, tl.int64) + d_o * tl.cast(so_d, tl.int64) + h_o * tl.cast(so_h, tl.int64)
        out_ptrs = out_base + w_offs[:, None] * tl.cast(so_w, tl.int64) + c_offs[None, :]
        tl.store(out_ptr + out_ptrs, vals, mask=store_mask, eviction_policy="evict_first")

    @triton.jit
    def _ltx_pad_3d_ncdhw_kernel(
        x_ptr,
        out_ptr,
        c_dim,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
    ):
        bcd = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        c_i64 = tl.cast(c_dim, tl.int64)
        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        cd_out = c_i64 * d_out_i64
        b_idx = bcd // cd_out
        rem = bcd - b_idx * cd_out
        c_idx = rem // d_out_i64
        d_o = rem - c_idx * d_out_i64

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_offs = h_blk * block_h + tl.arange(0, block_h).to(tl.int64)
        h_in_bounds = h_offs < h_out_i64
        h_i, h_valid = _map_coord_i64(h_offs, pad_h_l_i64, h_in_i64, s_mode)

        sx_h_i64 = tl.cast(sx_h, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        in_base = b_idx * tl.cast(sx_b, tl.int64) + c_idx * tl.cast(sx_c, tl.int64) + d_i * tl.cast(sx_d, tl.int64)
        out_base = b_idx * tl.cast(so_b, tl.int64) + c_idx * tl.cast(so_c, tl.int64) + d_o * tl.cast(so_d, tl.int64)

        num_interior_w = (w_in_i64 + block_w - 1) // block_w
        if w_blk < num_interior_w:
            w_in_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
            w_mask = w_in_offs < w_in_i64
            w_out_offs = pad_w_l_i64 + w_in_offs
            load_mask = (d_valid & h_in_bounds & h_valid)[:, None] & w_mask[None, :]
            store_mask = h_in_bounds[:, None] & w_mask[None, :]
            vals = tl.load(
                x_ptr + in_base + h_i[:, None] * sx_h_i64 + w_in_offs[None, :],
                mask=load_mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            tl.store(
                out_ptr + out_base + h_offs[:, None] * so_h_i64 + w_out_offs[None, :],
                vals,
                mask=store_mask,
                eviction_policy="evict_first",
            )
        else:
            pad_w_r_i64 = w_out_i64 - pad_w_l_i64 - w_in_i64
            total_border = pad_w_l_i64 + pad_w_r_i64
            for b_start in range(0, total_border, block_w):
                b_idx_w = tl.cast(b_start, tl.int64) + tl.arange(0, block_w).to(tl.int64)
                b_in_bounds = b_idx_w < total_border
                w_offs = tl.where(b_idx_w < pad_w_l_i64, b_idx_w, pad_w_l_i64 + w_in_i64 + (b_idx_w - pad_w_l_i64))
                w_i, w_valid = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)
                load_mask = (d_valid & h_in_bounds & h_valid)[:, None] & (b_in_bounds & w_valid)[None, :]
                store_mask = h_in_bounds[:, None] & b_in_bounds[None, :]
                vals = tl.load(x_ptr + in_base + h_i[:, None] * sx_h_i64 + w_i[None, :], mask=load_mask, other=0.0)
                tl.store(out_ptr + out_base + h_offs[:, None] * so_h_i64 + w_offs[None, :], vals, mask=store_mask)

    @triton.jit
    def _ltx_pad_3d_align_c_kernel(
        x_ptr,
        out_ptr,
        c_in,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        c_align: tl.constexpr,
        block_w: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_in_bounds = w_offs < w_out_i64
        c_offs = tl.arange(0, c_align).to(tl.int64)
        c_valid = c_offs < tl.cast(c_in, tl.int64)

        d_i, d_valid = _map_coord_i64(d_o, tl.cast(pad_t_l, tl.int64), d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, tl.cast(pad_h_l, tl.int64), h_in_i64, s_mode)
        w_i, w_valid = _map_coord_i64(w_offs, tl.cast(pad_w_l, tl.int64), w_in_i64, s_mode)

        load_mask = (d_valid & h_valid & w_in_bounds & w_valid)[:, None] & c_valid[None, :]
        in_ptrs = (
            b_idx * tl.cast(sx_b, tl.int64)
            + d_i * tl.cast(sx_d, tl.int64)
            + h_i * tl.cast(sx_h, tl.int64)
            + w_i[:, None] * tl.cast(sx_w, tl.int64)
            + c_offs[None, :] * tl.cast(sx_c, tl.int64)
        )
        vals = tl.load(x_ptr + in_ptrs, mask=load_mask, other=0.0)
        out_ptrs = (
            b_idx * tl.cast(so_b, tl.int64)
            + d_o * tl.cast(so_d, tl.int64)
            + h_o * tl.cast(so_h, tl.int64)
            + w_offs[:, None] * tl.cast(so_w, tl.int64)
            + c_offs[None, :]
        )
        tl.store(out_ptr + out_ptrs, vals, mask=w_in_bounds[:, None], eviction_policy="evict_first")

    # =========================================================================
    # 4. Coalesced Residual + Conv Bias Add (NDHWC & NCDHW, strict int64)
    # =========================================================================

    @triton.jit
    def _ltx_residual_bias_add_ndhwc_kernel(
        x_ptr,
        h_ptr,
        bias_ptr,
        out_ptr,
        total_voxels,
        channels,
        has_x: tl.constexpr,
        has_bias: tl.constexpr,
        block_v: tl.constexpr,
        block_c: tl.constexpr,
    ):
        v_blk = tl.program_id(0).to(tl.int64)
        c_blk = tl.program_id(1).to(tl.int64)

        total_v_i64 = tl.cast(total_voxels, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        v_offs = v_blk * block_v + tl.arange(0, block_v).to(tl.int64)
        v_mask = v_offs < total_v_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        mask = v_mask[:, None] & c_mask[None, :]

        ptrs = v_offs[:, None] * c_i64 + c_offs[None, :]
        h_val = tl.load(h_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
        if has_bias:
            b_val = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            h_val = (h_val.to(tl.float32) + b_val).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _ltx_residual_bias_add_ncdhw_kernel(
        x_ptr,
        h_ptr,
        bias_ptr,
        out_ptr,
        spatial_size,
        channels,
        has_x: tl.constexpr,
        has_bias: tl.constexpr,
        block_s: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        s_blk = tl.program_id(1).to(tl.int64)

        spatial_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        c_idx = bc % c_i64
        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64

        ptrs = bc * spatial_i64 + s_offs
        h_val = tl.load(h_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
        if has_bias:
            b_val = tl.load(bias_ptr + c_idx).to(tl.float32)
            h_val = (h_val.to(tl.float32) + b_val).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=s_mask, eviction_policy="evict_first")

    # =========================================================================
    # 5. DepthToSpace 3D (+ Optional h_bias + Residual Repeat+Add) — int64
    # =========================================================================

    @triton.jit
    def _d2s_ndhwc_fast_kernel(
        x_ptr,
        res_ptr,
        h_bias_ptr,
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
        HAS_H_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        CONTIG_RES_SLICE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)
        c_tile = tl.program_id(2).to(tl.int64)

        d_i64 = tl.cast(D, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_in_i64 = tl.cast(C_in, tl.int64)
        c_res_in_i64 = tl.cast(C_res_in, tl.int64)
        c_res_unp_i64 = tl.cast(C_res_unpacked, tl.int64)
        c_out_i64 = tl.cast(C_out, tl.int64)
        h_out_i64 = tl.cast(H_out, tl.int64)
        w_out_i64 = tl.cast(W_out, tl.int64)
        sxn = tl.cast(stride_xn, tl.int64)
        srn = tl.cast(stride_rn, tl.int64)
        syn = tl.cast(stride_yn, tl.int64)

        h_in = bdh % h_i64
        bd = bdh // h_i64
        d_in = bd % d_i64
        b_id = bd // d_i64

        w0 = w_tile * BLOCK_W
        c0 = c_tile * BLOCK_C
        w_offs = w0 + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < w_i64
        c_offs = c0 + tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = c_offs < c_out_i64

        cp_offs = c0 * P + tl.arange(0, BLOCK_C * P).to(tl.int64)
        cp_c_mask = cp_offs < c_in_i64
        cp_mask = w_mask[:, None] & cp_c_mask[None, :]
        in_row_base = b_id * sxn + (d_in * h_i64 + h_in) * (w_i64 * c_in_i64)
        v = tl.load(x_ptr + in_row_base + w_offs[:, None] * c_in_i64 + cp_offs[None, :], mask=cp_mask)

        out_ty = y_ptr.dtype.element_ty
        if HAS_H_BIAS:
            hb = tl.load(h_bias_ptr + cp_offs, mask=cp_c_mask, other=0.0).to(tl.float32)[None, :]
            v = (v.to(tl.float32) + hb).to(out_ty)

        if HAS_RESIDUAL:
            res_row_base = b_id * srn + (d_in * h_i64 + h_in) * (w_i64 * c_res_in_i64)
            if CONTIG_RES_SLICE:
                res_c0 = c0 % c_res_unp_i64
                rp_offs = res_c0 * P + tl.arange(0, BLOCK_C * P).to(tl.int64)
                r = tl.load(res_ptr + res_row_base + w_offs[:, None] * c_res_in_i64 + rp_offs[None, :], mask=cp_mask)
            else:
                p_offs = tl.arange(0, P).to(tl.int64)
                rc_2d = (c_offs % c_res_unp_i64)[:, None] * P + p_offs[None, :]
                rc_flat = tl.reshape(rc_2d, (BLOCK_C * P,))
                r = tl.load(res_ptr + res_row_base + w_offs[:, None] * c_res_in_i64 + rc_flat[None, :], mask=cp_mask)
            v = (v.to(tl.float32) + r.to(tl.float32)).to(out_ty)

        y_batch_base = b_id * syn
        hwc_stride = h_out_i64 * w_out_i64 * c_out_i64
        wc_stride = w_out_i64 * c_out_i64
        h_out0 = h_in * P2

        if P3 == 2:
            w_out_offs = w0 * 2 + tl.arange(0, BLOCK_W * 2).to(tl.int64)
            out_mask = (w_out_offs[:, None] < w_out_i64) & c_mask[None, :]
        else:
            w_out_offs = w_offs
            out_mask = w_mask[:, None] & c_mask[None, :]

        if P == 1:
            row_base = y_batch_base + d_in * hwc_stride + h_out0 * wc_stride
            tl.store(y_ptr + row_base + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], v, mask=out_mask)
        elif P == 2:
            v0, v1 = tl.split(tl.reshape(v, (BLOCK_W, BLOCK_C, 2)))
            if P1 == 2:
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                    tl.store(y_ptr + b0 + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], v0, mask=out_mask)
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b1 + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], v1, mask=out_mask)
            elif P2 == 2:
                t_base = y_batch_base + d_in * hwc_stride
                tl.store(
                    y_ptr + t_base + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    v0,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t_base + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    v1,
                    mask=out_mask,
                )
            else:
                out_row = tl.reshape(tl.permute(tl.join(v0, v1), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                b0 = y_batch_base + d_in * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b0 + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], out_row, mask=out_mask)
        elif P == 4:
            va, vb = tl.split(tl.reshape(v, (BLOCK_W, BLOCK_C, 2, 2)))
            v00, v10 = tl.split(va)
            v01, v11 = tl.split(vb)
            if P1 == 1 and P2 == 2 and P3 == 2:
                row0 = tl.reshape(tl.permute(tl.join(v00, v01), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                row1 = tl.reshape(tl.permute(tl.join(v10, v11), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                t_base = y_batch_base + d_in * hwc_stride
                tl.store(
                    y_ptr + t_base + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    row0,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t_base + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    row1,
                    mask=out_mask,
                )
            elif P1 == 2 and P2 == 2 and P3 == 1:
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride
                    tl.store(
                        y_ptr + b0 + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                        v00,
                        mask=out_mask,
                    )
                    tl.store(
                        y_ptr + b0 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                        v01,
                        mask=out_mask,
                    )
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride
                tl.store(
                    y_ptr + b1 + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    v10,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + b1 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    v11,
                    mask=out_mask,
                )
            else:
                row0 = tl.reshape(tl.permute(tl.join(v00, v01), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                row1 = tl.reshape(tl.permute(tl.join(v10, v11), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
                d_full0 = d_in * 2
                if d_full0 >= TRIM_FIRST:
                    b0 = y_batch_base + (d_full0 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                    tl.store(y_ptr + b0 + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], row0, mask=out_mask)
                b1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride + h_out0 * wc_stride
                tl.store(y_ptr + b1 + w_out_offs[:, None] * c_out_i64 + c_offs[None, :], row1, mask=out_mask)
        else:
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
                    y_ptr + t0 + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    row00,
                    mask=out_mask,
                )
                tl.store(
                    y_ptr + t0 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                    row01,
                    mask=out_mask,
                )

            t1 = y_batch_base + (d_in * 2 + 1 - TRIM_FIRST) * hwc_stride
            row10 = tl.reshape(tl.permute(tl.join(v100, v101), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
            row11 = tl.reshape(tl.permute(tl.join(v110, v111), (0, 2, 1)), (BLOCK_W * 2, BLOCK_C))
            tl.store(
                y_ptr + t1 + h_out0 * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                row10,
                mask=out_mask,
            )
            tl.store(
                y_ptr + t1 + (h_out0 + 1) * wc_stride + w_out_offs[:, None] * c_out_i64 + c_offs[None, :],
                row11,
                mask=out_mask,
            )

    @triton.jit
    def _d2s_ndhwc_gen_kernel(
        x_ptr,
        res_ptr,
        h_bias_ptr,
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
        HAS_H_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bdhw = tl.program_id(0).to(tl.int64)
        c_tile = tl.program_id(1).to(tl.int64)

        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_in_i64 = tl.cast(C_in, tl.int64)
        c_res_unp_i64 = tl.cast(C_res_unpacked, tl.int64)
        c_out_i64 = tl.cast(C_out, tl.int64)
        d_out_i64 = tl.cast(D_out, tl.int64)
        h_out_i64 = tl.cast(H_out, tl.int64)
        w_out_i64 = tl.cast(W_out, tl.int64)
        p1_i64 = tl.cast(p1, tl.int64)
        p2_i64 = tl.cast(p2, tl.int64)
        p3_i64 = tl.cast(p3, tl.int64)

        w_out = bdhw % w_out_i64
        bdh = bdhw // w_out_i64
        h_out = bdh % h_out_i64
        bd = bdh // h_out_i64
        d_out = bd % d_out_i64
        b_id = bd // d_out_i64

        d_full = d_out + tl.cast(trim_first, tl.int64)
        d_in = d_full // p1_i64
        r1 = d_full - d_in * p1_i64
        h_in = h_out // p2_i64
        r2 = h_out - h_in * p2_i64
        w_in = w_out // p3_i64
        r3 = w_out - w_in * p3_i64
        p_i64 = p1_i64 * p2_i64 * p3_i64
        sub_idx = (r1 * p2_i64 + r2) * p3_i64 + r3

        c_offs = c_tile * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
        c_mask = c_offs < c_out_i64
        ch_x = c_offs * p_i64 + sub_idx

        in_voxel_base = b_id * tl.cast(stride_xn, tl.int64) + ((d_in * h_i64 + h_in) * w_i64 + w_in) * c_in_i64
        xv = tl.load(x_ptr + in_voxel_base + ch_x, mask=c_mask)

        out_ty = y_ptr.dtype.element_ty
        if HAS_H_BIAS:
            hb = tl.load(h_bias_ptr + ch_x, mask=c_mask, other=0.0).to(tl.float32)
            xv = (xv.to(tl.float32) + hb).to(out_ty)

        if HAS_RESIDUAL:
            c_res = (c_offs % c_res_unp_i64) * p_i64 + sub_idx
            res_offs = (
                b_id * tl.cast(rs_n, tl.int64)
                + c_res * tl.cast(rs_c, tl.int64)
                + d_in * tl.cast(rs_d, tl.int64)
                + h_in * tl.cast(rs_h, tl.int64)
                + w_in * tl.cast(rs_w, tl.int64)
            )
            rv = tl.load(res_ptr + res_offs, mask=c_mask)
            xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(out_ty)

        y_off = (
            b_id * tl.cast(stride_yn, tl.int64) + ((d_out * h_out_i64 + h_out) * w_out_i64 + w_out) * c_out_i64 + c_offs
        )
        tl.store(y_ptr + y_off, xv, mask=c_mask)

    @triton.jit
    def _d2s_ncdhw_temporal_kernel(
        x_ptr,
        res_ptr,
        h_bias_ptr,
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
        HAS_H_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        bcd = tl.program_id(0).to(tl.int64)
        hw_tile = tl.program_id(1).to(tl.int64)

        d_i64 = tl.cast(D, tl.int64)
        hw_i64 = tl.cast(HW, tl.int64)
        c_in_i64 = tl.cast(C_in, tl.int64)
        c_res_in_i64 = tl.cast(C_res_in, tl.int64)
        c_res_unp_i64 = tl.cast(C_res_unpacked, tl.int64)
        c_out_i64 = tl.cast(C_out, tl.int64)
        d_out_i64 = tl.cast(D_out, tl.int64)

        d_in = bcd % d_i64
        bc = bcd // d_i64
        c_out = bc % c_out_i64
        b_id = bc // c_out_i64

        hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
        mask = hw_offs < hw_i64
        dhw_in = d_i64 * hw_i64
        dhw_out = d_out_i64 * hw_i64
        x_b_base = b_id * (c_in_i64 * dhw_in) + d_in * hw_i64
        y_bc_base = bc * dhw_out
        out_ty = y_ptr.dtype.element_ty

        if HAS_RESIDUAL:
            c_res_base = (c_out % c_res_unp_i64) * P1
            r_b_base = b_id * (c_res_in_i64 * dhw_in) + d_in * hw_i64

        c_x_base = c_out * P1
        for r1 in tl.static_range(P1):
            d_full = d_in * P1 + r1
            if d_full >= TRIM_FIRST:
                d_out = d_full - TRIM_FIRST
                ch_x = c_x_base + r1
                v0 = tl.load(x_ptr + x_b_base + ch_x * dhw_in + hw_offs, mask=mask)
                if HAS_H_BIAS:
                    hb = tl.load(h_bias_ptr + ch_x).to(tl.float32)
                    v0 = (v0.to(tl.float32) + hb).to(out_ty)
                if HAS_RESIDUAL:
                    res0 = tl.load(res_ptr + r_b_base + (c_res_base + r1) * dhw_in + hw_offs, mask=mask)
                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(out_ty)
                tl.store(y_ptr + y_bc_base + d_out * hw_i64 + hw_offs, v0, mask=mask)

    @triton.jit
    def _d2s_ncdhw_fast_kernel(
        x_ptr,
        res_ptr,
        h_bias_ptr,
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
        HAS_H_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        bcdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        d_i64 = tl.cast(D, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_in_i64 = tl.cast(C_in, tl.int64)
        c_res_in_i64 = tl.cast(C_res_in, tl.int64)
        c_res_unp_i64 = tl.cast(C_res_unpacked, tl.int64)
        c_out_i64 = tl.cast(C_out, tl.int64)
        d_out_i64 = tl.cast(D_out, tl.int64)
        h_out_i64 = tl.cast(H_out, tl.int64)
        w_out_i64 = tl.cast(W_out, tl.int64)

        num_h_tiles = (h_i64 + BLOCK_H - 1) // BLOCK_H
        h_tile = bcdh % num_h_tiles
        bcd = bcdh // num_h_tiles
        d_in = bcd % d_i64
        bc = bcd // d_i64
        c_out = bc % c_out_i64
        b_id = bc // c_out_i64

        w_in_start = w_tile * BLOCK_W
        w_in_offs = w_in_start + tl.arange(0, BLOCK_W).to(tl.int64)
        in_mask = w_in_offs < w_i64

        dhw_in = d_i64 * h_i64 * w_i64
        dhw_out = d_out_i64 * h_out_i64 * w_out_i64
        hw_out = h_out_i64 * w_out_i64
        x_b_base = b_id * (c_in_i64 * dhw_in)
        y_bc_base = bc * dhw_out
        out_ty = y_ptr.dtype.element_ty

        if HAS_RESIDUAL:
            c_res_base = (c_out % c_res_unp_i64) * P
            r_b_base = b_id * (c_res_in_i64 * dhw_in)

        c_x_base = c_out * P
        h0 = h_tile * BLOCK_H

        for dh_it in tl.static_range(BLOCK_H):
            h_in = h0 + dh_it
            if h_in < h_i64:
                dh_in_off = (d_in * h_i64 + h_in) * w_i64
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
                                if HAS_H_BIAS:
                                    hb0 = tl.load(h_bias_ptr + c_x_base + sub0).to(tl.float32)
                                    hb1 = tl.load(h_bias_ptr + c_x_base + sub1).to(tl.float32)
                                    v0 = (v0.to(tl.float32) + hb0).to(out_ty)
                                    v1 = (v1.to(tl.float32) + hb1).to(out_ty)
                                if HAS_RESIDUAL:
                                    res0 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    res1 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub1) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(out_ty)
                                    v1 = (v1.to(tl.float32) + res1.to(tl.float32)).to(out_ty)
                                out_row = tl.interleave(v0, v1)
                                w_out_offs = w_in_start * 2 + tl.arange(0, BLOCK_W * 2).to(tl.int64)
                                out_mask = w_out_offs < w_out_i64
                                tl.store(
                                    y_ptr + y_bc_base + d_out * hw_out + h_out * w_out_i64 + w_out_offs,
                                    out_row,
                                    mask=out_mask,
                                )
                            else:
                                sub0 = r1 * P2 + r2
                                v0 = tl.load(
                                    x_ptr + x_b_base + (c_x_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                    mask=in_mask,
                                )
                                if HAS_H_BIAS:
                                    hb0 = tl.load(h_bias_ptr + c_x_base + sub0).to(tl.float32)
                                    v0 = (v0.to(tl.float32) + hb0).to(out_ty)
                                if HAS_RESIDUAL:
                                    res0 = tl.load(
                                        res_ptr + r_b_base + (c_res_base + sub0) * dhw_in + dh_in_off + w_in_offs,
                                        mask=in_mask,
                                    )
                                    v0 = (v0.to(tl.float32) + res0.to(tl.float32)).to(out_ty)
                                tl.store(
                                    y_ptr + y_bc_base + d_out * hw_out + h_out * w_out_i64 + w_in_offs,
                                    v0,
                                    mask=in_mask,
                                )

    @triton.jit
    def _d2s_ncdhw_gen_kernel(
        x_ptr,
        res_ptr,
        h_bias_ptr,
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
        HAS_H_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_W_OUT: tl.constexpr,
    ):
        bcdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        d_i64 = tl.cast(D, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        w_i64 = tl.cast(W, tl.int64)
        c_in_i64 = tl.cast(C_in, tl.int64)
        c_res_unp_i64 = tl.cast(C_res_unpacked, tl.int64)
        c_out_i64 = tl.cast(C_out, tl.int64)
        d_out_i64 = tl.cast(D_out, tl.int64)
        h_out_i64 = tl.cast(H_out, tl.int64)
        w_out_i64 = tl.cast(W_out, tl.int64)
        p1_i64 = tl.cast(p1, tl.int64)
        p2_i64 = tl.cast(p2, tl.int64)
        p3_i64 = tl.cast(p3, tl.int64)

        h_out = bcdh % h_out_i64
        bcd = bcdh // h_out_i64
        d_out = bcd % d_out_i64
        bc = bcd // d_out_i64
        c_out = bc % c_out_i64
        b_id = bc // c_out_i64

        d_full = d_out + tl.cast(trim_first, tl.int64)
        d_in = d_full // p1_i64
        r1 = d_full - d_in * p1_i64
        h_in = h_out // p2_i64
        r2 = h_out - h_in * p2_i64

        w_out_offs = w_tile * BLOCK_W_OUT + tl.arange(0, BLOCK_W_OUT).to(tl.int64)
        mask = w_out_offs < w_out_i64
        w_in_offs = w_out_offs // p3_i64
        r3 = w_out_offs - w_in_offs * p3_i64
        p_i64 = p1_i64 * p2_i64 * p3_i64
        sub_idx = (r1 * p2_i64 + r2) * p3_i64 + r3

        dhw_in = d_i64 * h_i64 * w_i64
        c_x = c_out * p_i64 + sub_idx
        x_offs = b_id * (c_in_i64 * dhw_in) + c_x * dhw_in + (d_in * h_i64 + h_in) * w_i64 + w_in_offs
        xv = tl.load(x_ptr + x_offs, mask=mask)
        out_ty = y_ptr.dtype.element_ty

        if HAS_H_BIAS:
            hb = tl.load(h_bias_ptr + c_x, mask=mask, other=0.0).to(tl.float32)
            xv = (xv.to(tl.float32) + hb).to(out_ty)

        if HAS_RESIDUAL:
            c_res = (c_out % c_res_unp_i64) * p_i64 + sub_idx
            r_offs = (
                b_id * tl.cast(rs_n, tl.int64)
                + c_res * tl.cast(rs_c, tl.int64)
                + d_in * tl.cast(rs_d, tl.int64)
                + h_in * tl.cast(rs_h, tl.int64)
                + w_in_offs * tl.cast(rs_w, tl.int64)
            )
            rv = tl.load(res_ptr + r_offs, mask=mask)
            xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(out_ty)

        y_off = ((bc * d_out_i64 + d_out) * h_out_i64 + h_out) * w_out_i64 + w_out_offs
        tl.store(y_ptr + y_off, xv, mask=mask)

    # =========================================================================
    # 6. SpaceToDepth 3D (+ h_bias) + Shortcut Group Mean + Add (NDHWC & NCDHW)
    # =========================================================================

    @triton.jit
    def _s2d_add_ndhwc_fast_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        d_out,
        h_out,
        w_out,
        c_h,
        c_x,
        c_out,
        sh_b,
        sh_d,
        sh_h,
        sh_w,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        P1: tl.constexpr,
        P2: tl.constexpr,
        P3: tl.constexpr,
        P: tl.constexpr,
        DUP_FIRST_T: tl.constexpr,
        HAS_H_BIAS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_CH: tl.constexpr,
    ):
        """100% coalesced NDHWC SpaceToDepth (+ h_bias) + shortcut group-mean add."""
        bth = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)
        ch_tile = tl.program_id(2).to(tl.int64)

        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_h_i64 = tl.cast(c_h, tl.int64)
        c_x_i64 = tl.cast(c_x, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)

        th_out = d_out_i64 * h_out_i64
        b_idx = bth // th_out
        rem = bth - b_idx * th_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        w_offs = w_tile * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < w_out_i64

        ch0 = ch_tile * BLOCK_CH
        ch_offs = ch0 + tl.arange(0, BLOCK_CH).to(tl.int64)
        ch_mask = ch_offs < c_h_i64
        h_mask = w_mask[:, None] & ch_mask[None, :]

        BLOCK_COUT: tl.constexpr = BLOCK_CH * P
        BLOCK_CX: tl.constexpr = BLOCK_CH * GROUP_SIZE
        cx0 = ch0 * GROUP_SIZE
        cx_offs = cx0 + tl.arange(0, BLOCK_CX).to(tl.int64)
        x_mask = w_mask[:, None] & (cx_offs < c_x_i64)[None, :]

        cout_offs = ch0 * P + tl.arange(0, BLOCK_COUT).to(tl.int64)
        out_mask = w_mask[:, None] & (cout_offs < c_out_i64)[None, :]

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sh_w_i64 = tl.cast(sh_w, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        dh0 = d_o * P1
        dh1 = dh0 + 1
        if DUP_FIRST_T:
            dx0 = tl.maximum(dh0 - 1, 0)
            dx1 = tl.maximum(dh1 - 1, 0)
        else:
            dx0 = dh0
            dx1 = dh1

        hh0 = h_o * P2
        hh1 = hh0 + 1
        ww0 = w_offs * P3
        ww1 = ww0 + 1

        hb = tl.zeros((1, BLOCK_CH), dtype=tl.float32)
        if HAS_H_BIAS:
            hb = tl.load(h_bias_ptr + ch_offs, mask=ch_mask, other=0.0).to(tl.float32)[None, :]

        h_b = b_idx * sh_b_i64
        x_b = b_idx * sx_b_i64

        if P == 2:
            if P1 == 2:
                h0 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h1 = tl.load(
                    h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x0 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x1 = tl.load(
                    x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            elif P2 == 2:
                h0 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h1 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x0 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x1 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            else:
                h0 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h1 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x0 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x1 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            if HAS_H_BIAS:
                h0 = (h0.to(tl.float32) + hb).to(out_ty)
                h1 = (h1.to(tl.float32) + hb).to(out_ty)
            h_pack = tl.reshape(tl.join(h0, h1), (BLOCK_W, BLOCK_COUT))
            x_pack = tl.reshape(tl.join(x0, x1), (BLOCK_W, BLOCK_COUT, GROUP_SIZE))
        elif P == 4:
            if P1 == 1 and P2 == 2 and P3 == 2:
                h00 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h01 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h10 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h11 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x00 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x01 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x10 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x11 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            elif P1 == 2 and P2 == 2 and P3 == 1:
                h00 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h01 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h10 = tl.load(
                    h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h11 = tl.load(
                    h_ptr + h_b + dh1 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x00 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x01 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x10 = tl.load(
                    x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x11 = tl.load(
                    x_ptr + x_b + dx1 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            else:
                h00 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h01 = tl.load(
                    h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h10 = tl.load(
                    h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                h11 = tl.load(
                    h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x00 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x01 = tl.load(
                    x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x10 = tl.load(
                    x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
                x11 = tl.load(
                    x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                    mask=x_mask,
                    other=0.0,
                )
            if HAS_H_BIAS:
                h00 = (h00.to(tl.float32) + hb).to(out_ty)
                h01 = (h01.to(tl.float32) + hb).to(out_ty)
                h10 = (h10.to(tl.float32) + hb).to(out_ty)
                h11 = (h11.to(tl.float32) + hb).to(out_ty)
            h_pack = tl.reshape(tl.join(tl.join(h00, h10), tl.join(h01, h11)), (BLOCK_W, BLOCK_COUT))
            x_pack = tl.reshape(tl.join(tl.join(x00, x10), tl.join(x01, x11)), (BLOCK_W, BLOCK_COUT, GROUP_SIZE))
        else:
            # P == 8: (2, 2, 2)
            h000 = tl.load(
                h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h001 = tl.load(
                h_ptr + h_b + dh0 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h010 = tl.load(
                h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h011 = tl.load(
                h_ptr + h_b + dh0 * sh_d_i64 + hh1 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h100 = tl.load(
                h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h101 = tl.load(
                h_ptr + h_b + dh1 * sh_d_i64 + hh0 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h110 = tl.load(
                h_ptr + h_b + dh1 * sh_d_i64 + hh1 * sh_h_i64 + ww0[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            h111 = tl.load(
                h_ptr + h_b + dh1 * sh_d_i64 + hh1 * sh_h_i64 + ww1[:, None] * sh_w_i64 + ch_offs[None, :],
                mask=h_mask,
                other=0.0,
            )
            if HAS_H_BIAS:
                h000 = (h000.to(tl.float32) + hb).to(out_ty)
                h001 = (h001.to(tl.float32) + hb).to(out_ty)
                h010 = (h010.to(tl.float32) + hb).to(out_ty)
                h011 = (h011.to(tl.float32) + hb).to(out_ty)
                h100 = (h100.to(tl.float32) + hb).to(out_ty)
                h101 = (h101.to(tl.float32) + hb).to(out_ty)
                h110 = (h110.to(tl.float32) + hb).to(out_ty)
                h111 = (h111.to(tl.float32) + hb).to(out_ty)
            h_a00 = tl.join(h000, h100)
            h_a01 = tl.join(h001, h101)
            h_a10 = tl.join(h010, h110)
            h_a11 = tl.join(h011, h111)
            h_pack = tl.reshape(tl.join(tl.join(h_a00, h_a10), tl.join(h_a01, h_a11)), (BLOCK_W, BLOCK_COUT))

            x000 = tl.load(
                x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x001 = tl.load(
                x_ptr + x_b + dx0 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x010 = tl.load(
                x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x011 = tl.load(
                x_ptr + x_b + dx0 * sx_d_i64 + hh1 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x100 = tl.load(
                x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x101 = tl.load(
                x_ptr + x_b + dx1 * sx_d_i64 + hh0 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x110 = tl.load(
                x_ptr + x_b + dx1 * sx_d_i64 + hh1 * sx_h_i64 + ww0[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x111 = tl.load(
                x_ptr + x_b + dx1 * sx_d_i64 + hh1 * sx_h_i64 + ww1[:, None] * sx_w_i64 + cx_offs[None, :],
                mask=x_mask,
                other=0.0,
            )
            x_a00 = tl.join(x000, x100)
            x_a01 = tl.join(x001, x101)
            x_a10 = tl.join(x010, x110)
            x_a11 = tl.join(x011, x111)
            x_pack = tl.reshape(
                tl.join(tl.join(x_a00, x_a10), tl.join(x_a01, x_a11)), (BLOCK_W, BLOCK_COUT, GROUP_SIZE)
            )

        if GROUP_SIZE == 1:
            x_mean = tl.reshape(x_pack, (BLOCK_W, BLOCK_COUT))
        else:
            x_mean = (tl.sum(x_pack.to(tl.float32), axis=2) * (1.0 / GROUP_SIZE)).to(out_ty)

        out_val = (h_pack.to(tl.float32) + x_mean.to(tl.float32)).to(out_ty)
        out_base = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64
        tl.store(
            out_ptr + out_base + w_offs[:, None] * so_w_i64 + cout_offs[None, :],
            out_val,
            mask=out_mask,
            eviction_policy="evict_first",
        )

    @triton.jit
    def _s2d_add_ncdhw_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        d_out,
        h_out,
        w_out,
        c_out,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
        P1: tl.constexpr,
        P2: tl.constexpr,
        P3: tl.constexpr,
        P: tl.constexpr,
        DUP_FIRST_T: tl.constexpr,
        HAS_H_BIAS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Coalesced NCDHW SpaceToDepth (+ h_bias) + shortcut group-mean add."""
        bcdh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)

        h_o = bcdh % h_out_i64
        bcd = bcdh // h_out_i64
        d_o = bcd % d_out_i64
        bc = bcd // d_out_i64
        c_o = bc % c_out_i64
        b_idx = bc // c_out_i64

        w_offs = w_tile * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < w_out_i64

        c_h = c_o // P
        sub_h = c_o - c_h * P
        r3_h = sub_h % P3
        tmp_h = sub_h // P3
        r2_h = tmp_h % P2
        r1_h = tmp_h // P2

        d_h = d_o * P1 + r1_h
        h_h = h_o * P2 + r2_h

        out_ty = out_ptr.dtype.element_ty
        h_row = (
            b_idx * tl.cast(sh_b, tl.int64)
            + c_h * tl.cast(sh_c, tl.int64)
            + d_h * tl.cast(sh_d, tl.int64)
            + h_h * tl.cast(sh_h, tl.int64)
        )
        if P3 == 2:
            w2_offs = w_tile * (2 * BLOCK_W) + tl.arange(0, 2 * BLOCK_W).to(tl.int64)
            w2_mask = w2_offs < (2 * w_out_i64)
            h_pair = tl.load(h_ptr + h_row + w2_offs, mask=w2_mask, other=0.0)
            h_even, h_odd = tl.split(tl.reshape(h_pair, (BLOCK_W, 2)))
            hv = tl.where(r3_h == 0, h_even, h_odd)
        else:
            hv = tl.load(h_ptr + h_row + w_offs * P3 + r3_h, mask=w_mask, other=0.0)
        if HAS_H_BIAS:
            hb = tl.load(h_bias_ptr + c_h).to(tl.float32)
            hv = (hv.to(tl.float32) + hb).to(out_ty)

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        x_b = b_idx * sx_b_i64

        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        k_base = c_o * GROUP_SIZE
        for j in tl.static_range(GROUP_SIZE):
            k = k_base + j
            cx = k // P
            sub_x = k - cx * P
            rx3 = sub_x % P3
            tmp_x = sub_x // P3
            rx2 = tmp_x % P2
            rx1 = tmp_x // P2
            d_x_dup = d_o * P1 + rx1
            d_x = tl.maximum(d_x_dup - 1, 0) if DUP_FIRST_T else d_x_dup
            h_x = h_o * P2 + rx2
            x_row = x_b + cx * sx_c_i64 + d_x * sx_d_i64 + h_x * sx_h_i64
            if P3 == 2:
                w2_offs = w_tile * (2 * BLOCK_W) + tl.arange(0, 2 * BLOCK_W).to(tl.int64)
                w2_mask = w2_offs < (2 * w_out_i64)
                x_pair = tl.load(x_ptr + x_row + w2_offs, mask=w2_mask, other=0.0)
                x_even, x_odd = tl.split(tl.reshape(x_pair, (BLOCK_W, 2)))
                xv = tl.where(rx3 == 0, x_even, x_odd)
            else:
                xv = tl.load(x_ptr + x_row + w_offs * P3 + rx3, mask=w_mask, other=0.0)
            acc += xv.to(tl.float32)

        x_mean = (acc * (1.0 / GROUP_SIZE)).to(out_ty)
        out_val = (hv.to(tl.float32) + x_mean.to(tl.float32)).to(out_ty)
        out_ptrs = (
            b_idx * tl.cast(so_b, tl.int64)
            + c_o * tl.cast(so_c, tl.int64)
            + d_o * tl.cast(so_d, tl.int64)
            + h_o * tl.cast(so_h, tl.int64)
            + w_offs
        )
        tl.store(out_ptr + out_ptrs, out_val, mask=w_mask, eviction_policy="evict_first")

    # =========================================================================
    # 7. Patchify 3D (Direct CL3D / NCDHW) & Unpatchify 3D (+ optional bias)
    # =========================================================================

    @triton.jit
    def _patchify_3d_kernel(
        x_ptr,
        out_ptr,
        c_in,
        f_out,
        h_out,
        w_out,
        sx_b,
        sx_c,
        sx_f,
        sx_h,
        sx_w,
        so_b,
        so_c,
        so_f,
        so_h,
        so_w,
        P_T: tl.constexpr,
        P_HW: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """5-D patchify: ``b c (f p) (h q) (w r) -> b (c p r q) f h w`` (strict int64)."""
        bfh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        f_out_i64 = tl.cast(f_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)

        fh = f_out_i64 * h_out_i64
        b_idx = bfh // fh
        rem = bfh - b_idx * fh
        f_o = rem // h_out_i64
        h_o = rem - f_o * h_out_i64

        w_offs = w_tile * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < w_out_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_f_i64 = tl.cast(sx_f, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_f_i64 = tl.cast(so_f, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        x_b = b_idx * sx_b_i64
        o_base = b_idx * so_b_i64 + f_o * so_f_i64 + h_o * so_h_i64 + w_offs * so_w_i64

        for c in range(c_in):
            c_i64 = tl.cast(c, tl.int64)
            for ip in tl.static_range(P_T):
                f_i = f_o * P_T + ip
                for ir in tl.static_range(P_HW):
                    w_i = w_offs * P_HW + ir
                    for iq in tl.static_range(P_HW):
                        h_i = h_o * P_HW + iq
                        c_o = ((c_i64 * P_T + ip) * P_HW + ir) * P_HW + iq
                        v = tl.load(
                            x_ptr + x_b + c_i64 * sx_c_i64 + f_i * sx_f_i64 + h_i * sx_h_i64 + w_i * sx_w_i64,
                            mask=w_mask,
                            other=0.0,
                        )
                        tl.store(
                            out_ptr + o_base + c_o * so_c_i64,
                            v,
                            mask=w_mask,
                            eviction_policy="evict_first",
                        )

    @triton.jit
    def _unpatchify_3d_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        c_out,
        f_in,
        h_in,
        w_in,
        sx_b,
        sx_c,
        sx_f,
        sx_h,
        sx_w,
        so_b,
        so_c,
        so_f,
        so_h,
        so_w,
        P_T: tl.constexpr,
        P_HW: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """5-D unpatchify (+ optional bias): ``b (c p r q) f h w -> b c (f p) (h q) (w r)``."""
        bfh = tl.program_id(0).to(tl.int64)
        w_tile = tl.program_id(1).to(tl.int64)

        f_in_i64 = tl.cast(f_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)

        fh = f_in_i64 * h_in_i64
        b_idx = bfh // fh
        rem = bfh - b_idx * fh
        f_i = rem // h_in_i64
        h_i = rem - f_i * h_in_i64

        w_offs = w_tile * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
        w_mask = w_offs < w_in_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_f_i64 = tl.cast(sx_f, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_f_i64 = tl.cast(so_f, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        x_base = b_idx * sx_b_i64 + f_i * sx_f_i64 + h_i * sx_h_i64 + w_offs * sx_w_i64
        o_b = b_idx * so_b_i64
        out_ty = out_ptr.dtype.element_ty

        for c in range(c_out):
            c_i64 = tl.cast(c, tl.int64)
            for ip in tl.static_range(P_T):
                f_o = f_i * P_T + ip
                for iq in tl.static_range(P_HW):
                    h_o = h_i * P_HW + iq
                    for ir in tl.static_range(P_HW):
                        w_o = w_offs * P_HW + ir
                        c_i = ((c_i64 * P_T + ip) * P_HW + ir) * P_HW + iq
                        v = tl.load(x_ptr + x_base + c_i * sx_c_i64, mask=w_mask, other=0.0)
                        if HAS_BIAS:
                            bv = tl.load(bias_ptr + c_i).to(tl.float32)
                            v = (v.to(tl.float32) + bv).to(out_ty)
                        tl.store(
                            out_ptr + o_b + c_i64 * so_c_i64 + f_o * so_f_i64 + h_o * so_h_i64 + w_o * so_w_i64,
                            v,
                            mask=w_mask,
                            eviction_policy="evict_first",
                        )

    _TRITON_KERNELS = {
        "pixel_norm_ndhwc_onepass": _pixel_norm_ndhwc_onepass_kernel,
        "pixel_norm_ndhwc_twopass": _pixel_norm_ndhwc_twopass_kernel,
        "pixel_norm_ncdhw": _pixel_norm_ncdhw_kernel,
        "pixel_norm_pad_ndhwc": _pixel_norm_pad_3d_ndhwc_kernel,
        "pixel_norm_pad_ncdhw": _pixel_norm_pad_3d_ncdhw_kernel,
        "pixel_norm_tpad_ndhwc": _pixel_norm_tpad_3d_ndhwc_kernel,
        "pixel_norm_tpad_ncdhw": _pixel_norm_tpad_3d_ncdhw_kernel,
        "pad_ndhwc": _ltx_pad_3d_ndhwc_kernel,
        "pad_ncdhw": _ltx_pad_3d_ncdhw_kernel,
        "pad_align_c": _ltx_pad_3d_align_c_kernel,
        "res_bias_ndhwc": _ltx_residual_bias_add_ndhwc_kernel,
        "res_bias_ncdhw": _ltx_residual_bias_add_ncdhw_kernel,
        "d2s_ndhwc_fast": _d2s_ndhwc_fast_kernel,
        "d2s_ndhwc_gen": _d2s_ndhwc_gen_kernel,
        "d2s_ncdhw_temporal": _d2s_ncdhw_temporal_kernel,
        "d2s_ncdhw_fast": _d2s_ncdhw_fast_kernel,
        "d2s_ncdhw_gen": _d2s_ncdhw_gen_kernel,
        "s2d_ndhwc_fast": _s2d_add_ndhwc_fast_kernel,
        "s2d_ncdhw": _s2d_add_ncdhw_kernel,
        "patchify_3d": _patchify_3d_kernel,
        "unpatchify_3d": _unpatchify_3d_kernel,
    }
    return _TRITON_KERNELS


# =============================================================================
# PyTorch Reference Fallbacks
# =============================================================================


def _pad_3d_torch(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
) -> torch.Tensor:
    s_mode = "constant" if spatial_mode == "zeros" else spatial_mode
    t_mode = "constant" if temporal_mode == "zeros" else temporal_mode
    if all(p == 0 for p in pad):
        return x
    if s_mode == t_mode:
        return F.pad(x, pad, mode=s_mode)
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    out = x
    if pw_l or pw_r or ph_l or ph_r:
        out = F.pad(out, (pw_l, pw_r, ph_l, ph_r, 0, 0), mode=s_mode)
    if pt_l or pt_r:
        out = F.pad(out, (0, 0, 0, 0, pt_l, pt_r), mode=t_mode)
    return out


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


def _pixel_norm_adaln_silu_pad_3d_torch(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
    apply_silu: bool = True,
    pre_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    inp = x + pre_bias.view(1, -1, 1, 1, 1).to(x) if pre_bias is not None else x
    out = _pixel_norm_adaln_silu_torch(inp, scale, shift, eps, apply_silu)
    return _pad_3d_torch(out, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)


def _residual_bias_add_torch(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    h_eff = h + bias.view(1, -1, 1, 1, 1).to(h) if bias is not None else h
    return h_eff if x is None else x + h_eff


def _depth_to_space_3d_torch(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x_eff = x + h_bias.view(1, -1, 1, 1, 1).to(x) if h_bias is not None else x
    B, C_in, D, H, W = x_eff.shape
    P = p1 * p2 * p3
    C_out = C_in // P
    out = (
        x_eff.reshape(B, C_out, p1, p2, p3, D, H, W)
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


def _space_to_depth_3d_add_torch(
    h: torch.Tensor,
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    group_size: int,
    *,
    dup_first_t: bool = False,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    h_eff = h + h_bias.view(1, -1, 1, 1, 1).to(h) if h_bias is not None else h
    x_eff = torch.cat([x[:, :, :1, :, :], x], dim=2) if dup_first_t else x
    B, C_x, D_in, H_in, W_in = x_eff.shape
    C_h = h_eff.shape[1]
    P = p1 * p2 * p3
    D_o, H_o, W_o = D_in // p1, H_in // p2, W_in // p3
    x_s2d = (
        x_eff.reshape(B, C_x, D_o, p1, H_o, p2, W_o, p3)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(B, C_x * P // group_size, group_size, D_o, H_o, W_o)
        .mean(dim=2)
    )
    h_s2d = (
        h_eff.reshape(B, C_h, D_o, p1, H_o, p2, W_o, p3)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(B, C_h * P, D_o, H_o, W_o)
    )
    return h_s2d + x_s2d


def _patchify_3d_torch(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int = 1,
    *,
    channels_last_3d: bool = False,
) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x.contiguous(memory_format=torch.channels_last_3d) if channels_last_3d and x.ndim == 5 else x
    b, c, f, h, w = x.shape
    p, q, r = patch_size_t, patch_size_hw, patch_size_hw
    out = (
        x.reshape(b, c, f // p, p, h // q, q, w // r, r)
        .permute(0, 1, 3, 7, 5, 2, 4, 6)
        .reshape(b, c * p * r * q, f // p, h // q, w // r)
    )
    if channels_last_3d:
        return out.contiguous(memory_format=torch.channels_last_3d)
    return out.contiguous()


def _unpatchify_3d_torch(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int = 1,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x_eff = x + bias.view(1, -1, 1, 1, 1).to(x) if bias is not None else x
    if patch_size_hw == 1 and patch_size_t == 1:
        return x_eff
    b, c_in, f, h, w = x_eff.shape
    p, q, r = patch_size_t, patch_size_hw, patch_size_hw
    c = c_in // (p * r * q)
    return (
        x_eff.reshape(b, c, p, r, q, f, h, w)
        .permute(0, 1, 5, 2, 6, 4, 7, 3)
        .reshape(b, c, f * p, h * q, w * r)
        .contiguous()
    )


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

        x_contig = x if x.is_contiguous() else x.contiguous()
        y = torch.empty_like(x_contig)
        if B * _cdiv(S, 1024) >= 528:
            block_s = 1024
        elif B * _cdiv(S, 512) >= 528:
            block_s = 512
        elif B * _cdiv(S, 256) >= 528:
            block_s = 256
        elif B * _cdiv(S, 128) >= 264:
            block_s = 128
        elif B * _cdiv(S, 64) >= 132:
            block_s = 64
        else:
            block_s = max(32, min(64, _next_power_of_2(S)))
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


def _launch_pixel_norm_adaln_silu_pad_3d(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    eps: float,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        code_to_mode = {0: "constant", 1: "replicate", 2: "reflect"}
        return _pixel_norm_adaln_silu_pad_3d_torch(
            x,
            scale,
            shift,
            eps,
            (pad_w_l, pad_w_r, pad_h_l, pad_h_r, pad_t_l, pad_t_r),
            spatial_mode=code_to_mode[s_mode_code],
            temporal_mode=code_to_mode[t_mode_code],
            apply_silu=apply_silu,
            pre_bias=pre_bias,
        )

    B, C, D_in, H_in, W_in = x.shape
    D_out = D_in + pad_t_l + pad_t_r
    H_out = H_in + pad_h_l + pad_h_r
    W_out = W_in + pad_w_l + pad_w_r

    has_pre_bias = pre_bias is not None
    pre_bias_1d = pre_bias.reshape(C).contiguous() if pre_bias is not None else x
    has_adaln = scale is not None and shift is not None
    sc_t = scale if scale is not None else x
    sh_t = shift if shift is not None else x
    sc_sn, sc_sc, sc_sd, sc_sh, sc_sw = _effective_5d_strides(scale)
    sh_sn, sh_sc, sh_sd, sh_sh, sh_sw = _effective_5d_strides(shift)
    spatial_adaln = has_adaln and (sc_sd != 0 or sc_sh != 0 or sc_sw != 0 or sh_sd != 0 or sh_sh != 0 or sh_sw != 0)

    is_ndhwc = _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x)
    no_spatial_pad = pad_w_l == 0 and pad_w_r == 0 and pad_h_l == 0 and pad_h_r == 0
    with torch.cuda.device(x.device):
        if is_ndhwc:
            out = torch.empty(
                (B, C, D_out, H_out, W_out),
                device=x.device,
                dtype=x.dtype,
                memory_format=torch.channels_last_3d,
            )
            if out.numel() == 0:
                return out
            block_c = _next_power_of_2(C)
            if no_spatial_pad:
                hw = H_in * W_in
                block_s = max(1, min(64, 4096 // block_c))
                num_warps = 8 if block_s * block_c >= 4096 else 4
                kernels["pixel_norm_tpad_ndhwc"][(_cdiv(hw, block_s), B * D_out)](
                    x,
                    pre_bias_1d,
                    sc_t,
                    sh_t,
                    out,
                    D_in,
                    D_out,
                    hw,
                    W_in,
                    pad_t_l,
                    x.stride(0),
                    x.stride(2),
                    out.stride(0),
                    out.stride(2),
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
                    eps,
                    channels=C,
                    t_mode=t_mode_code,
                    has_pre_bias=has_pre_bias,
                    has_adaln=has_adaln,
                    spatial_adaln=spatial_adaln,
                    apply_silu=bool(apply_silu),
                    block_s=block_s,
                    block_c=block_c,
                    num_warps=num_warps,
                    num_stages=3,
                )
                return out
            block_w = max(1, min(64, 4096 // block_c))
            num_warps = 8 if block_w * block_c >= 4096 else 4
            grid = (B * D_out * H_out, _cdiv(W_out, block_w))
            kernels["pixel_norm_pad_ndhwc"][grid](
                x,
                pre_bias_1d,
                sc_t,
                sh_t,
                out,
                D_in,
                H_in,
                W_in,
                D_out,
                H_out,
                W_out,
                pad_t_l,
                pad_h_l,
                pad_w_l,
                x.stride(0),
                x.stride(2),
                x.stride(3),
                x.stride(4),
                out.stride(0),
                out.stride(2),
                out.stride(3),
                out.stride(4),
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
                eps,
                channels=C,
                t_mode=t_mode_code,
                s_mode=s_mode_code,
                has_pre_bias=has_pre_bias,
                has_adaln=has_adaln,
                spatial_adaln=spatial_adaln,
                apply_silu=bool(apply_silu),
                block_w=block_w,
                block_c=block_c,
                num_warps=num_warps,
                num_stages=3,
            )
            return out

        x_c = x if x.is_contiguous() else x.contiguous()
        out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
        if out.numel() == 0:
            return out
        if no_spatial_pad:
            hw = H_in * W_in
            bd_out = B * D_out
            if bd_out * _cdiv(hw, 1024) >= 528:
                block_s = 1024
            elif bd_out * _cdiv(hw, 512) >= 528:
                block_s = 512
            elif bd_out * _cdiv(hw, 256) >= 528:
                block_s = 256
            else:
                block_s = max(64, min(128, _next_power_of_2(hw)))
            kernels["pixel_norm_tpad_ncdhw"][(_cdiv(hw, block_s), bd_out)](
                x_c,
                pre_bias_1d,
                sc_t,
                sh_t,
                out,
                D_in,
                D_out,
                hw,
                W_in,
                pad_t_l,
                x_c.stride(0),
                x_c.stride(1),
                x_c.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
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
                eps,
                channels=C,
                t_mode=t_mode_code,
                has_pre_bias=has_pre_bias,
                has_adaln=has_adaln,
                spatial_adaln=spatial_adaln,
                apply_silu=bool(apply_silu),
                block_s=block_s,
                num_warps=8 if block_s >= 1024 else 4,
                num_stages=3,
            )
            return out
        block_w = 256 if (W_in >= 256 and C <= 256) else (128 if W_in >= 128 else _next_power_of_2(max(W_in, 16)))
        num_warps = 8 if block_w >= 256 else 4
        num_w_tiles = _cdiv(W_in, block_w) + (1 if (pad_w_l > 0 or pad_w_r > 0) else 0)
        grid = (B * D_out * H_out, num_w_tiles)
        kernels["pixel_norm_pad_ncdhw"][grid](
            x_c,
            pre_bias_1d,
            sc_t,
            sh_t,
            out,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            pad_t_l,
            pad_h_l,
            pad_w_l,
            x_c.stride(0),
            x_c.stride(1),
            x_c.stride(2),
            x_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
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
            eps,
            channels=C,
            t_mode=t_mode_code,
            s_mode=s_mode_code,
            has_pre_bias=has_pre_bias,
            has_adaln=has_adaln,
            spatial_adaln=spatial_adaln,
            apply_silu=bool(apply_silu),
            block_w=block_w,
            num_warps=num_warps,
        )
        return out


def _launch_depth_to_space_3d(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _depth_to_space_3d_torch(x, p1, p2, p3, residual, out_channels_reduction_factor, h_bias=h_bias)

    B, C_in, D, H, W = x.shape
    P = p1 * p2 * p3
    C_out = C_in // P
    trim_first = 1 if p1 == 2 else 0
    D_out = D * p1 - trim_first
    H_out = H * p2
    W_out = W * p3
    out_shape = (B, C_out, D_out, H_out, W_out)

    has_h_bias = h_bias is not None
    hb_t = h_bias.reshape(C_in).contiguous() if h_bias is not None else x
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
                    hb_t,
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
                    HAS_H_BIAS=has_h_bias,
                    HAS_RESIDUAL=has_residual,
                    CONTIG_RES_SLICE=contig_res_slice,
                    BLOCK_W=block_w,
                    BLOCK_C=block_c,
                    num_warps=8 if block_w * block_c * P >= 4096 else 4,
                    num_stages=3,
                )
                return y

            rs_n, rs_c, rs_d, rs_h, rs_w = res_t.stride() if has_residual else (0, 0, 0, 0, 0)
            block_c = min(256, _next_power_of_2(C_out))
            kernels["d2s_ndhwc_gen"][(B * D_out * H_out * W_out, _cdiv(C_out, block_c))](
                x,
                res_t,
                hb_t,
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
                HAS_H_BIAS=has_h_bias,
                HAS_RESIDUAL=has_residual,
                BLOCK_C=block_c,
                num_warps=4,
                num_stages=3,
            )
            return y

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
                    hb_t,
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
                    HAS_H_BIAS=has_h_bias,
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
                hb_t,
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
                HAS_H_BIAS=has_h_bias,
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
            hb_t,
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
            HAS_H_BIAS=has_h_bias,
            HAS_RESIDUAL=has_residual,
            BLOCK_W_OUT=block_w_out,
            num_warps=4,
            num_stages=3,
        )
        return y


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


@torch.library.custom_op("worldfoundry::ltx_pixel_norm_adaln_silu_pad_3d", mutates_args=(), device_types="cuda")
def _ltx_pixel_norm_adaln_silu_pad_3d_op(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    eps: float,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    apply_silu: bool,
) -> torch.Tensor:
    return _launch_pixel_norm_adaln_silu_pad_3d(
        x,
        scale,
        shift,
        pre_bias,
        eps,
        pad_w_l,
        pad_w_r,
        pad_h_l,
        pad_h_r,
        pad_t_l,
        pad_t_r,
        s_mode_code,
        t_mode_code,
        apply_silu,
    )


@_ltx_pixel_norm_adaln_silu_pad_3d_op.register_fake
def _ltx_pixel_norm_adaln_silu_pad_3d_fake(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    eps: float,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    apply_silu: bool,
) -> torch.Tensor:
    del scale, shift, pre_bias, eps, s_mode_code, t_mode_code, apply_silu
    b, c, d_in, h_in, w_in = x.shape
    return _empty_like_layout(x, (b, c, d_in + pad_t_l + pad_t_r, h_in + pad_h_l + pad_h_r, w_in + pad_w_l + pad_w_r))


@torch.library.custom_op("worldfoundry::ltx_pad_3d", mutates_args=(), device_types="cuda")
def _ltx_pad_3d_op(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        code_to_mode = {0: "constant", 1: "replicate", 2: "reflect"}
        return _pad_3d_torch(
            x,
            (pad_w_l, pad_w_r, pad_h_l, pad_h_r, pad_t_l, pad_t_r),
            spatial_mode=code_to_mode[s_mode_code],
            temporal_mode=code_to_mode[t_mode_code],
        )

    b, c, d_in, h_in, w_in = x.shape
    d_out = d_in + pad_t_l + pad_t_r
    h_out = h_in + pad_h_l + pad_h_r
    w_out = w_in + pad_w_l + pad_w_r

    with torch.cuda.device(x.device):
        if _is_channels_last_3d(x) or _is_batch_strided_ndhwc(x):
            out = torch.empty(
                (b, c, d_out, h_out, w_out),
                device=x.device,
                dtype=x.dtype,
                memory_format=torch.channels_last_3d,
            )
            if out.numel() == 0:
                return out
            block_c = 128 if c >= 128 else _next_power_of_2(max(c, 16))
            block_w = 16 if block_c >= 128 else 32
            num_warps = 8 if block_w * block_c >= 2048 else 4
            grid = (b * d_out * h_out, _cdiv(w_out, block_w), _cdiv(c, block_c))
            kernels["pad_ndhwc"][grid](
                x,
                out,
                c,
                d_in,
                h_in,
                w_in,
                d_out,
                h_out,
                w_out,
                pad_t_l,
                pad_h_l,
                pad_w_l,
                x.stride(0),
                x.stride(2),
                x.stride(3),
                x.stride(4),
                out.stride(0),
                out.stride(2),
                out.stride(3),
                out.stride(4),
                t_mode=t_mode_code,
                s_mode=s_mode_code,
                block_w=block_w,
                block_c=block_c,
                num_warps=num_warps,
            )
            return out

        x_c = x if x.is_contiguous() else x.contiguous()
        out = torch.empty((b, c, d_out, h_out, w_out), device=x.device, dtype=x.dtype)
        if out.numel() == 0:
            return out
        block_w = 256 if w_in >= 256 else (128 if w_in >= 128 else _next_power_of_2(max(w_in, 16)))
        block_h = 8 if h_out >= 8 else _next_power_of_2(max(h_out, 1))
        num_warps = 8 if block_h * block_w >= 2048 else 4
        num_w_tiles = _cdiv(w_in, block_w) + (1 if (pad_w_l > 0 or pad_w_r > 0) else 0)
        grid = (b * c * d_out, _cdiv(h_out, block_h), num_w_tiles)
        kernels["pad_ncdhw"][grid](
            x_c,
            out,
            c,
            d_in,
            h_in,
            w_in,
            d_out,
            h_out,
            w_out,
            pad_t_l,
            pad_h_l,
            pad_w_l,
            x_c.stride(0),
            x_c.stride(1),
            x_c.stride(2),
            x_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            t_mode=t_mode_code,
            s_mode=s_mode_code,
            block_h=block_h,
            block_w=block_w,
            num_warps=num_warps,
        )
        return out


@_ltx_pad_3d_op.register_fake
def _ltx_pad_3d_fake(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
) -> torch.Tensor:
    del s_mode_code, t_mode_code
    b, c, d_in, h_in, w_in = x.shape
    return _empty_like_layout(x, (b, c, d_in + pad_t_l + pad_t_r, h_in + pad_h_l + pad_h_r, w_in + pad_w_l + pad_w_r))


@torch.library.custom_op("worldfoundry::ltx_pad_3d_align_c", mutates_args=(), device_types="cuda")
def _ltx_pad_3d_align_c_op(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    align_c: int,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    b, c_in, d_in, h_in, w_in = x.shape
    d_out = d_in + pad_t_l + pad_t_r
    h_out = h_in + pad_h_l + pad_h_r
    w_out = w_in + pad_w_l + pad_w_r
    if kernels is None:
        code_to_mode = {0: "constant", 1: "replicate", 2: "reflect"}
        padded = _pad_3d_torch(
            x,
            (pad_w_l, pad_w_r, pad_h_l, pad_h_r, pad_t_l, pad_t_r),
            spatial_mode=code_to_mode[s_mode_code],
            temporal_mode=code_to_mode[t_mode_code],
        )
        out = torch.empty(
            (b, align_c, d_out, h_out, w_out),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        ).zero_()
        out[:, :c_in] = padded
        return out

    with torch.cuda.device(x.device):
        out = torch.empty(
            (b, align_c, d_out, h_out, w_out),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
        if out.numel() == 0:
            return out
        block_w = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
        grid = (b * d_out * h_out, _cdiv(w_out, block_w))
        kernels["pad_align_c"][grid](
            x,
            out,
            c_in,
            d_in,
            h_in,
            w_in,
            d_out,
            h_out,
            w_out,
            pad_t_l,
            pad_h_l,
            pad_w_l,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            t_mode=t_mode_code,
            s_mode=s_mode_code,
            c_align=align_c,
            block_w=block_w,
            num_warps=4,
        )
        return out


@_ltx_pad_3d_align_c_op.register_fake
def _ltx_pad_3d_align_c_fake(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    align_c: int,
) -> torch.Tensor:
    del s_mode_code, t_mode_code
    b, _, d_in, h_in, w_in = x.shape
    return torch.empty(
        (b, align_c, d_in + pad_t_l + pad_t_r, h_in + pad_h_l + pad_h_r, w_in + pad_w_l + pad_w_r),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )


@torch.library.custom_op("worldfoundry::ltx_residual_bias_add", mutates_args=(), device_types="cuda")
def _ltx_residual_bias_add_op(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _residual_bias_add_torch(x, h, bias)

    b, c, d, hh, w = h.shape
    has_x = x is not None
    has_bias = bias is not None
    bias_1d = bias.reshape(c).contiguous() if bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or (x is not None and _is_channels_last_3d(x))
    with torch.cuda.device(h.device):
        if use_cl3d:
            h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
            x_cl = (
                (x if _is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d))
                if x is not None
                else h_cl
            )
            out = torch.empty((b, c, d, hh, w), device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
            if out.numel() == 0:
                return out
            total_v = b * d * hh * w
            block_c = 128 if c >= 128 else _next_power_of_2(max(c, 16))
            block_v = max(16, min(64, 4096 // block_c))
            num_warps = 8 if block_v * block_c >= 4096 else 4
            grid = (_cdiv(total_v, block_v), _cdiv(c, block_c))
            kernels["res_bias_ndhwc"][grid](
                x_cl,
                h_cl,
                bias_1d,
                out,
                total_v,
                c,
                has_x=has_x,
                has_bias=has_bias,
                block_v=block_v,
                block_c=block_c,
                num_warps=num_warps,
                num_stages=3,
            )
            return out

        h_c = h if h.is_contiguous() else h.contiguous()
        x_c = (x if x.is_contiguous() else x.contiguous()) if x is not None else h_c
        out = torch.empty((b, c, d, hh, w), device=h.device, dtype=h.dtype)
        if out.numel() == 0:
            return out
        spatial = d * hh * w
        block_s = 1024 if spatial >= 1024 else _next_power_of_2(max(spatial, 64))
        num_warps = 8 if block_s >= 1024 else 4
        grid = (b * c, _cdiv(spatial, block_s))
        kernels["res_bias_ncdhw"][grid](
            x_c,
            h_c,
            bias_1d,
            out,
            spatial,
            c,
            has_x=has_x,
            has_bias=has_bias,
            block_s=block_s,
            num_warps=num_warps,
        )
        return out


@_ltx_residual_bias_add_op.register_fake
def _ltx_residual_bias_add_fake(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    del bias
    if _is_channels_last_3d(h) or (x is not None and _is_channels_last_3d(x)):
        return torch.empty(tuple(h.shape), device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return h.new_empty(tuple(h.shape))


@torch.library.custom_op("worldfoundry::ltx_depth_to_space_3d", mutates_args=(), device_types="cuda")
def _ltx_depth_to_space_3d_op(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _launch_depth_to_space_3d(x, p1, p2, p3, residual, out_channels_reduction_factor, h_bias=h_bias)


@_ltx_depth_to_space_3d_op.register_fake
def _ltx_depth_to_space_3d_fake(
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    residual: torch.Tensor | None,
    out_channels_reduction_factor: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del residual, out_channels_reduction_factor, h_bias
    B, C_in, D, H, W = x.shape
    P = p1 * p2 * p3
    trim_first = 1 if p1 == 2 else 0
    out_shape = (B, C_in // P, D * p1 - trim_first, H * p2, W * p3)
    return _empty_like_layout(x, out_shape)


@torch.library.custom_op("worldfoundry::ltx_space_to_depth_3d_add", mutates_args=(), device_types="cuda")
def _ltx_space_to_depth_3d_add_op(
    h: torch.Tensor,
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    group_size: int,
    dup_first_t: bool,
    h_bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _space_to_depth_3d_add_torch(h, x, p1, p2, p3, group_size, dup_first_t=dup_first_t, h_bias=h_bias)

    b, c_h, d_h, h_in, w_in = h.shape
    c_x = x.shape[1]
    p = p1 * p2 * p3
    c_out = c_h * p
    d_out = d_h // p1
    h_out = h_in // p2
    w_out = w_in // p3

    has_h_bias = h_bias is not None
    hb_1d = h_bias.reshape(c_h).contiguous() if h_bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or _is_channels_last_3d(x)
    fast_stride = p1 in (1, 2) and p2 in (1, 2) and p3 in (1, 2) and p > 1
    is_pow2_g = group_size > 0 and (group_size & (group_size - 1)) == 0

    with torch.cuda.device(h.device):
        if use_cl3d and fast_stride and is_pow2_g:
            h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
            x_cl = x if _is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d)
            out = torch.empty(
                (b, c_out, d_out, h_out, w_out),
                device=h.device,
                dtype=h.dtype,
                memory_format=torch.channels_last_3d,
            )
            if out.numel() == 0:
                return out
            max_ch_by_out = max(1, 256 // p)
            max_ch_by_x = max(1, 256 // group_size)
            block_ch = min(_next_power_of_2(c_h), max_ch_by_out, max_ch_by_x)
            if block_ch >= 1 and (c_h % block_ch == 0 or _next_power_of_2(c_h) <= block_ch):
                eff_c = max(block_ch * p, block_ch * group_size)
                block_w = max(1, min(16, 2048 // max(1, eff_c)))
                block_w = min(block_w, _next_power_of_2(max(1, w_out)))
                num_warps = 8 if block_w * eff_c >= 2048 else 4
                grid = (b * d_out * h_out, _cdiv(w_out, block_w), _cdiv(c_h, block_ch))
                kernels["s2d_ndhwc_fast"][grid](
                    h_cl,
                    x_cl,
                    hb_1d,
                    out,
                    d_out,
                    h_out,
                    w_out,
                    c_h,
                    c_x,
                    c_out,
                    h_cl.stride(0),
                    h_cl.stride(2),
                    h_cl.stride(3),
                    h_cl.stride(4),
                    x_cl.stride(0),
                    x_cl.stride(2),
                    x_cl.stride(3),
                    x_cl.stride(4),
                    out.stride(0),
                    out.stride(2),
                    out.stride(3),
                    out.stride(4),
                    P1=p1,
                    P2=p2,
                    P3=p3,
                    P=p,
                    DUP_FIRST_T=bool(dup_first_t),
                    HAS_H_BIAS=has_h_bias,
                    GROUP_SIZE=int(group_size),
                    BLOCK_W=block_w,
                    BLOCK_CH=block_ch,
                    num_warps=num_warps,
                )
                return out

        h_c = h if h.is_contiguous() else h.contiguous()
        x_c = x if x.is_contiguous() else x.contiguous()
        out = torch.empty((b, c_out, d_out, h_out, w_out), device=h.device, dtype=h.dtype)
        if out.numel() == 0:
            return out
        block_w = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
        grid = (b * c_out * d_out * h_out, _cdiv(w_out, block_w))
        kernels["s2d_ncdhw"][grid](
            h_c,
            x_c,
            hb_1d,
            out,
            d_out,
            h_out,
            w_out,
            c_out,
            h_c.stride(0),
            h_c.stride(1),
            h_c.stride(2),
            h_c.stride(3),
            x_c.stride(0),
            x_c.stride(1),
            x_c.stride(2),
            x_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            P1=p1,
            P2=p2,
            P3=p3,
            P=p,
            DUP_FIRST_T=bool(dup_first_t),
            HAS_H_BIAS=has_h_bias,
            GROUP_SIZE=int(group_size),
            BLOCK_W=block_w,
            num_warps=4,
        )
        return out.contiguous(memory_format=torch.channels_last_3d) if use_cl3d else out


@_ltx_space_to_depth_3d_add_op.register_fake
def _ltx_space_to_depth_3d_add_fake(
    h: torch.Tensor,
    x: torch.Tensor,
    p1: int,
    p2: int,
    p3: int,
    group_size: int,
    dup_first_t: bool,
    h_bias: torch.Tensor | None,
) -> torch.Tensor:
    del group_size, dup_first_t, h_bias
    b, c_h, d_h, h_in, w_in = h.shape
    p = p1 * p2 * p3
    shape = (b, c_h * p, d_h // p1, h_in // p2, w_in // p3)
    if _is_channels_last_3d(h) or _is_channels_last_3d(x):
        return torch.empty(shape, device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return h.new_empty(shape)


@torch.library.custom_op("worldfoundry::ltx_patchify_3d", mutates_args=(), device_types="cuda")
def _ltx_patchify_3d_op(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int,
    channels_last_3d: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _patchify_3d_torch(x, patch_size_hw, patch_size_t, channels_last_3d=channels_last_3d)

    b, c_in, f, h, w = x.shape
    f_out = f // patch_size_t
    h_out = h // patch_size_hw
    w_out = w // patch_size_hw
    c_out = c_in * patch_size_t * patch_size_hw * patch_size_hw
    with torch.cuda.device(x.device):
        if channels_last_3d:
            out = torch.empty(
                (b, c_out, f_out, h_out, w_out),
                device=x.device,
                dtype=x.dtype,
                memory_format=torch.channels_last_3d,
            )
        else:
            out = torch.empty((b, c_out, f_out, h_out, w_out), device=x.device, dtype=x.dtype)
        if out.numel() == 0:
            return out
        block_w = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
        grid = (b * f_out * h_out, _cdiv(w_out, block_w))
        kernels["patchify_3d"][grid](
            x,
            out,
            c_in,
            f_out,
            h_out,
            w_out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            P_T=int(patch_size_t),
            P_HW=int(patch_size_hw),
            BLOCK_W=block_w,
            num_warps=4,
        )
        return out


@_ltx_patchify_3d_op.register_fake
def _ltx_patchify_3d_fake(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int,
    channels_last_3d: bool,
) -> torch.Tensor:
    b, c_in, f, h, w = x.shape
    shape = (
        b,
        c_in * patch_size_t * patch_size_hw * patch_size_hw,
        f // patch_size_t,
        h // patch_size_hw,
        w // patch_size_hw,
    )
    if channels_last_3d:
        return torch.empty(shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    return x.new_empty(shape)


@torch.library.custom_op("worldfoundry::ltx_unpatchify_3d", mutates_args=(), device_types="cuda")
def _ltx_unpatchify_3d_op(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _unpatchify_3d_torch(x, patch_size_hw, patch_size_t, bias=bias)

    b, c_in, f_in, h_in, w_in = x.shape
    factor = patch_size_t * patch_size_hw * patch_size_hw
    c_out = c_in // factor
    f_out = f_in * patch_size_t
    h_out = h_in * patch_size_hw
    w_out = w_in * patch_size_hw
    has_bias = bias is not None
    bias_1d = bias.reshape(c_in).contiguous() if bias is not None else x

    with torch.cuda.device(x.device):
        out = torch.empty((b, c_out, f_out, h_out, w_out), device=x.device, dtype=x.dtype)
        if out.numel() == 0:
            return out
        block_w = 128 if w_in >= 128 else _next_power_of_2(max(w_in, 16))
        grid = (b * f_in * h_in, _cdiv(w_in, block_w))
        kernels["unpatchify_3d"][grid](
            x,
            bias_1d,
            out,
            c_out,
            f_in,
            h_in,
            w_in,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            P_T=int(patch_size_t),
            P_HW=int(patch_size_hw),
            HAS_BIAS=has_bias,
            BLOCK_W=block_w,
            num_warps=4,
        )
        return out


@_ltx_unpatchify_3d_op.register_fake
def _ltx_unpatchify_3d_fake(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    del bias
    b, c_in, f_in, h_in, w_in = x.shape
    c_out = c_in // (patch_size_t * patch_size_hw * patch_size_hw)
    return x.new_empty((b, c_out, f_in * patch_size_t, h_in * patch_size_hw, w_in * patch_size_hw))


# =============================================================================
# Public Entry Points & Eligibility Helpers
# =============================================================================


def _can_use_triton(x: torch.Tensor, *others: torch.Tensor | None) -> bool:
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.compiler.is_compiling() and x.numel() == 0:
        return False
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, *others)):
        return False
    return _get_triton_kernels() is not None


def _can_use_triton_inference(x: torch.Tensor, *others: torch.Tensor | None) -> bool:
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES or torch.is_grad_enabled():
        return False
    if not torch.compiler.is_compiling() and x.numel() == 0:
        return False
    for t in others:
        if t is not None and (not t.is_cuda or t.device != x.device or t.dtype not in _SUPPORTED_DTYPES):
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


def can_use_triton_pixel_norm_adaln_silu_pad_3d(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    pre_bias: torch.Tensor | None = None,
) -> bool:
    if not _can_use_triton_inference(x, scale, shift, pre_bias):
        return False
    if x.ndim != 5 or (scale is None) != (shift is None):
        return False
    if scale is not None and scale.ndim != 5:
        return False
    if shift is not None and shift.ndim != 5:
        return False
    if pre_bias is not None and pre_bias.numel() != x.shape[1]:
        return False
    if spatial_mode not in _MODE_TO_CODE or temporal_mode not in _MODE_TO_CODE:
        return False
    if len(pad) != 6 or any(p < 0 for p in pad):
        return False
    _, c, d_in, h_in, w_in = x.shape
    if not (0 < c <= 2048) or d_in <= 0 or h_in <= 0 or w_in <= 0:
        return False
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    if spatial_mode == "reflect" and (pw_l >= w_in or pw_r >= w_in or ph_l >= h_in or ph_r >= h_in):
        return False
    return not (temporal_mode == "reflect" and (pt_l >= d_in or pt_r >= d_in))


def triton_pixel_norm_adaln_silu_pad_3d(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    eps: float,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
    apply_silu: bool = True,
    pre_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Super-fused ``(pre_bias +) PixelNorm (+ AdaLN) (+ SiLU) + 3-D pad``."""
    eps_f = float(eps)
    if can_use_triton_pixel_norm_adaln_silu_pad_3d(
        x,
        pad,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
        scale=scale,
        shift=shift,
        pre_bias=pre_bias,
    ):
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = (int(p) for p in pad)
        return _ltx_pixel_norm_adaln_silu_pad_3d_op(
            x,
            scale,
            shift,
            pre_bias,
            eps_f,
            pw_l,
            pw_r,
            ph_l,
            ph_r,
            pt_l,
            pt_r,
            _MODE_TO_CODE[spatial_mode],
            _MODE_TO_CODE[temporal_mode],
            bool(apply_silu),
        )
    return _pixel_norm_adaln_silu_pad_3d_torch(
        x,
        scale,
        shift,
        eps_f,
        pad,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
        apply_silu=apply_silu,
        pre_bias=pre_bias,
    )


def can_use_triton_ltx_pad_3d(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
) -> bool:
    if not _can_use_triton_inference(x) or x.ndim != 5:
        return False
    if spatial_mode not in _MODE_TO_CODE or temporal_mode not in _MODE_TO_CODE:
        return False
    if len(pad) != 6 or any(p < 0 for p in pad):
        return False
    _, _, d_in, h_in, w_in = x.shape
    if d_in <= 0 or h_in <= 0 or w_in <= 0:
        return False
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    if spatial_mode == "reflect" and (pw_l >= w_in or pw_r >= w_in or ph_l >= h_in or ph_r >= h_in):
        return False
    return not (temporal_mode == "reflect" and (pt_l >= d_in or pt_r >= d_in))


def triton_ltx_pad_3d(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
) -> torch.Tensor:
    """Bit-exact, layout-preserving, strictly ``int64``-indexed 5-D pad."""
    if all(p == 0 for p in pad):
        return x
    if can_use_triton_ltx_pad_3d(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode):
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = (int(p) for p in pad)
        return _ltx_pad_3d_op(
            x,
            pw_l,
            pw_r,
            ph_l,
            ph_r,
            pt_l,
            pt_r,
            _MODE_TO_CODE[spatial_mode],
            _MODE_TO_CODE[temporal_mode],
        )
    return _pad_3d_torch(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)


def can_use_triton_ltx_pad_3d_align_c(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
    align_c: int = 8,
) -> bool:
    if not can_use_triton_ltx_pad_3d(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode):
        return False
    if align_c <= 0 or (align_c & (align_c - 1)) != 0:
        return False
    return 0 < x.shape[1] < align_c


def triton_ltx_pad_3d_align_c(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "constant",
    temporal_mode: str = "replicate",
    align_c: int = 8,
) -> torch.Tensor:
    """Fuse 3-D pad + channel zero-padding to ``align_c`` in ``channels_last_3d``."""
    if can_use_triton_ltx_pad_3d_align_c(
        x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode, align_c=align_c
    ):
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = (int(p) for p in pad)
        return _ltx_pad_3d_align_c_op(
            x,
            pw_l,
            pw_r,
            ph_l,
            ph_r,
            pt_l,
            pt_r,
            _MODE_TO_CODE[spatial_mode],
            _MODE_TO_CODE[temporal_mode],
            int(align_c),
        )
    padded = _pad_3d_torch(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)
    b, c_in, d_out, h_out, w_out = padded.shape
    out = torch.empty(
        (b, align_c, d_out, h_out, w_out),
        device=padded.device,
        dtype=padded.dtype,
        memory_format=torch.channels_last_3d,
    ).zero_()
    out[:, :c_in] = padded
    return out


def can_use_triton_ltx_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if not _can_use_triton_inference(h, x, bias) or h.ndim != 5:
        return False
    if x is not None and (x.ndim != 5 or x.shape != h.shape or x.dtype != h.dtype):
        return False
    return not (bias is not None and bias.numel() != h.shape[1])


def triton_ltx_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Coalesced single-pass ``x + (h + bias)`` (or ``h + bias`` when ``x is None``)."""
    if x is None and bias is None:
        return h
    if can_use_triton_ltx_residual_bias_add(x, h, bias):
        return _ltx_residual_bias_add_op(x, h, bias)
    return _residual_bias_add_torch(x, h, bias)


def triton_depth_to_space_3d(
    x: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    residual: torch.Tensor | None = None,
    out_channels_reduction_factor: int = 1,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass fused ``DepthToSpaceUpsample`` (+ optional ``h_bias`` & residual)."""
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
            h_bias is None
            or (
                h_bias.is_cuda
                and h_bias.device == x.device
                and h_bias.numel() == x.shape[1]
                and h_bias.dtype in _SUPPORTED_DTYPES
            )
        )
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
        and _can_use_triton(x, residual, h_bias)
    ):
        return _ltx_depth_to_space_3d_op(x, p1, p2, p3, residual, red, h_bias)
    return _depth_to_space_3d_torch(x, p1, p2, p3, residual, red, h_bias=h_bias)


def can_use_triton_space_to_depth_3d_add(
    h: torch.Tensor,
    x: torch.Tensor,
    stride: tuple[int, int, int],
    group_size: int,
    *,
    dup_first_t: bool = False,
    h_bias: torch.Tensor | None = None,
) -> bool:
    if not _can_use_triton_inference(h, x, h_bias):
        return False
    if h.ndim != 5 or x.ndim != 5 or h.dtype != x.dtype:
        return False
    if h_bias is not None and h_bias.numel() != h.shape[1]:
        return False
    p1, p2, p3 = (int(stride[0]), int(stride[1]), int(stride[2]))
    p = p1 * p2 * p3
    if p1 < 1 or p2 < 1 or p3 < 1 or p <= 1 or group_size < 1:
        return False
    b, c_h, d_h, h_in, w_in = h.shape
    bx, c_x, d_x, hx, wx = x.shape
    expected_d_h = d_x + (1 if dup_first_t else 0)
    if b != bx or d_h != expected_d_h or h_in != hx or w_in != wx:
        return False
    if d_h % p1 != 0 or h_in % p2 != 0 or w_in % p3 != 0:
        return False
    return c_x * p == c_h * p * group_size


def triton_space_to_depth_3d_add(
    h: torch.Tensor,
    x: torch.Tensor,
    stride: tuple[int, int, int],
    group_size: int,
    *,
    dup_first_t: bool = False,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass fused ``SpaceToDepthDownsample`` (+ ``h_bias``) + shortcut group-mean add."""
    p1, p2, p3 = (int(stride[0]), int(stride[1]), int(stride[2]))
    g = int(group_size)
    if can_use_triton_space_to_depth_3d_add(h, x, (p1, p2, p3), g, dup_first_t=dup_first_t, h_bias=h_bias):
        return _ltx_space_to_depth_3d_add_op(h, x, p1, p2, p3, g, bool(dup_first_t), h_bias)
    return _space_to_depth_3d_add_torch(h, x, p1, p2, p3, g, dup_first_t=dup_first_t, h_bias=h_bias)


def triton_patchify_3d(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int = 1,
    *,
    channels_last_3d: bool = False,
) -> torch.Tensor:
    """Layout-aware 5-D patchify outputting ``channels_last_3d`` directly when requested."""
    p_hw = int(patch_size_hw)
    p_t = int(patch_size_t)
    if p_hw == 1 and p_t == 1:
        return x.contiguous(memory_format=torch.channels_last_3d) if channels_last_3d and x.ndim == 5 else x
    if (
        x.ndim == 5
        and p_hw >= 1
        and p_t >= 1
        and x.shape[2] % p_t == 0
        and x.shape[3] % p_hw == 0
        and x.shape[4] % p_hw == 0
        and x.shape[1] <= 16
        and _can_use_triton_inference(x)
    ):
        return _ltx_patchify_3d_op(x, p_hw, p_t, bool(channels_last_3d))
    return _patchify_3d_torch(x, p_hw, p_t, channels_last_3d=channels_last_3d)


def triton_unpatchify_3d(
    x: torch.Tensor,
    patch_size_hw: int,
    patch_size_t: int = 1,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Layout-aware 5-D unpatchify with optional fused ``conv_out.bias``."""
    p_hw = int(patch_size_hw)
    p_t = int(patch_size_t)
    if p_hw == 1 and p_t == 1:
        return triton_ltx_residual_bias_add(None, x, bias) if bias is not None else x
    factor = p_t * p_hw * p_hw
    if (
        x.ndim == 5
        and p_hw >= 1
        and p_t >= 1
        and x.shape[1] % factor == 0
        and (x.shape[1] // factor) <= 16
        and (bias is None or bias.numel() == x.shape[1])
        and _can_use_triton_inference(x, bias)
    ):
        return _ltx_unpatchify_3d_op(x, p_hw, p_t, bias)
    return _unpatchify_3d_torch(x, p_hw, p_t, bias=bias)


__all__ = [
    "can_use_triton_ltx_pad_3d",
    "can_use_triton_ltx_pad_3d_align_c",
    "can_use_triton_ltx_residual_bias_add",
    "can_use_triton_pixel_norm_adaln_silu_pad_3d",
    "can_use_triton_space_to_depth_3d_add",
    "triton_depth_to_space_3d",
    "triton_ltx_pad_3d",
    "triton_ltx_pad_3d_align_c",
    "triton_ltx_residual_bias_add",
    "triton_patchify_3d",
    "triton_pixel_norm_adaln_silu",
    "triton_pixel_norm_adaln_silu_pad_3d",
    "triton_space_to_depth_3d_add",
    "triton_unpatchify_3d",
]
