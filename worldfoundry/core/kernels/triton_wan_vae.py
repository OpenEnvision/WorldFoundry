# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for WanVideoVAE / WanVideoVAE38.

Shared by 8 foundation video and streaming world models (Wan2.1-14B,
Wan2.2-TI2V-5B, Wan2.2-T2V-A14B, Matrix-Game-2.0, LongLive-1.3B,
Self-Forcing 1.3B, LingBot-Video, and SANA-Video).

Replaces multi-pass eager PyTorch 4D/5D ops and unvectorized ``Conv3d``/``Conv2d``
bias passes with single-pass, strictly ``int64``-indexed, coalesced Triton
kernels supporting BOTH ``NCDHW``/``NCHW`` and ``channels_last_3d``/``channels_last``
memory formats:

- :func:`wan_vae_rms_norm` — exact ``(optional pre_bias +) RMS_norm (+ SiLU)``.
- :func:`wan_vae_rms_norm_silu_causal_pad` — super-fused ``(optional pre_bias +)
  RMS_norm + SiLU`` + last-``cache_t``-frame cache snapshot + ``cat([cache, y])``
  + zero-padding directly into the ``CausalConv3d`` input buffer.
- :func:`wan_vae_causal_pad_cache` — single-pass causal cache snapshot + concat +
  3D zero-padding with optional ``pre_bias``, optional zero-left cache padding
  (``"Rep"``), and optional channel zero-pad alignment (``C: 3 -> 8`` in
  ``channels_last_3d`` to avoid cuDNN's ``in_channels < 8`` SGEMM cliff).
- :func:`wan_vae_residual_bias_add` — single-pass coalesced
  ``(x + x_bias) + (h + h_bias)`` (or ``h + h_bias`` when ``x is None``) for
  4D and 5D tensors, eliminating PyTorch's unvectorized ``Conv`` bias kernels.
- :func:`wan_vae_nearest_upsample_2x` — bit-exact native BF16/FP16/FP32 2x2
  nearest-neighbor upsampling for 4D ``NCHW``/``NHWC`` (load-once, store-4x).
- :func:`wan_vae_time_interleave_upsample_2x` — super-fused ``time_conv.bias`` +
  temporal 2x interleave + 5D->4D reshape + 2x2 nearest upsample for
  ``Resample`` / ``Resample38`` ``upsample3d`` preserving ``channels_last``.
- :func:`wan_vae_zero_pad_2d_rb` — single-pass ``ZeroPad2d((0, 1, 0, 1))`` for
  ``Resample`` / ``Resample38`` downsampling.
- :func:`wan_vae_dup_up_3d` / :func:`wan_vae_dup_up_3d_add` — single-pass
  ``DupUp3D`` (+ optional ``h + h_bias`` residual fusion) in BOTH
  ``channels_last_3d`` and ``NCDHW``.
- :func:`wan_vae_avg_down_3d` / :func:`wan_vae_avg_down_3d_add` — single-pass
  ``AvgDown3D`` (+ optional ``h + h_bias`` residual fusion) in BOTH
  ``channels_last_3d`` and ``NCDHW``.
- :func:`wan_vae_patchify_2x` / :func:`wan_vae_unpatchify_2x` — single-pass
  2x2 spatial patchify/unpatchify (``WanVideoVAE38``) preserving or converting
  ``channels_last_3d`` / ``NCDHW`` in one pass.

All CUDA ops register ``torch.library.custom_op`` + ``register_fake`` with exact
output shapes and strides for ``torch.compile(fullgraph=True)`` and CUDA Graph
capture safety. Importing this module does not import Triton or initialize CUDA.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def is_channels_last_3d(x: torch.Tensor) -> bool:
    """Return True when 5D *x* uses ``channels_last_3d`` (``NDHWC``) memory order."""
    if x.ndim != 5:
        return False
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        if not x.is_contiguous():
            return True
        return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
    return False


def is_channels_last_2d(x: torch.Tensor) -> bool:
    """Return True when 4D *x* uses ``channels_last`` (``NHWC``) memory order."""
    if x.ndim != 4:
        return False
    if x.is_contiguous(memory_format=torch.channels_last):
        if not x.is_contiguous():
            return True
        return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(3) == x.shape[1]
    return False


def _is_channels_last_nd(x: torch.Tensor) -> bool:
    """Return True when 4D/5D *x* uses channels-last memory order."""
    if x.ndim == 5:
        return is_channels_last_3d(x)
    if x.ndim == 4:
        return is_channels_last_2d(x)
    return False


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _select_cl_tile_w_warps(channels: int) -> tuple[int, int, int]:
    """Return Hopper-tuned ``(block_w, block_c, num_warps)`` for CL3D RMSNorm kernels."""
    block_c = _next_power_of_2(channels)
    if block_c <= 128:
        return 32, block_c, 4
    if block_c == 256:
        return (4 if channels == 256 else 8), 256, 2
    if block_c == 512:
        return (2 if channels >= 384 else 4), 512, 2
    return (1 if channels >= 768 else 2), block_c, 2


def _empty_with_layout(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last:
        if len(shape) == 5:
            return torch.empty(shape, device=device, dtype=dtype, memory_format=torch.channels_last_3d)
        if len(shape) == 4:
            return torch.empty(shape, device=device, dtype=dtype, memory_format=torch.channels_last)
    return torch.empty(shape, device=device, dtype=dtype)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Hopper SM90 Triton kernels on first eligible CUDA call."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="wan-vae-triton-v2")

    import triton
    import triton.language as tl

    # =========================================================================
    # 1. RMS_norm (+ optional pre_bias, bias, SiLU) — CL & CF
    # =========================================================================

    @triton.jit
    def _wan_rms_norm_channels_last_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        out_ptr,
        total_pixels,
        scale,
        channels: tl.constexpr,
        block_p: tl.constexpr,
        block_c: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        apply_silu: tl.constexpr,
    ):
        """Coalesced channels-last ``(pre_bias +) RMS_norm (+ bias) (+ SiLU)``."""
        pid = tl.program_id(0).to(tl.int64)
        total_p_i64 = tl.cast(total_pixels, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)

        p_offs = pid * block_p + tl.arange(0, block_p).to(tl.int64)
        p_mask = p_offs < total_p_i64
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        mask = p_mask[:, None] & c_mask[None, :]
        out_ty = out_ptr.dtype.element_ty

        ptrs = p_offs[:, None] * c_i64 + c_offs[None, :]
        x_in = tl.load(x_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
        x_f = x_in.to(tl.float32)
        if has_pre_bias:
            pb = tl.load(pre_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            x_f = (x_f + pb).to(out_ty).to(tl.float32)

        # Match F.normalize(x, p=2, dim=1, eps=1e-12) exactly.
        norm = tl.maximum(tl.sqrt(tl.sum(x_f * x_f, axis=1)), 1e-12)
        inv_norm = (scale / norm)[:, None]

        gamma = tl.load(gamma_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
        y = x_f * inv_norm * gamma
        if has_bias:
            bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y + bias
        if apply_silu:
            y_cast = y.to(out_ty).to(tl.float32)
            y = y_cast * tl.sigmoid(y_cast)
        tl.store(out_ptr + ptrs, y.to(out_ty), mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_rms_norm_channels_first_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        out_ptr,
        spatial_size,
        scale,
        channels: tl.constexpr,
        block_s: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        apply_silu: tl.constexpr,
    ):
        """Coalesced channels-first ``(pre_bias +) RMS_norm (+ bias) (+ SiLU)``."""
        s_blk = tl.program_id(0).to(tl.int64)
        b_idx = tl.program_id(1).to(tl.int64)
        spatial_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64
        batch_base = b_idx * c_i64 * spatial_i64
        out_ty = out_ptr.dtype.element_ty

        sq_sum = tl.zeros((block_s,), dtype=tl.float32)
        for c in range(channels):
            v_raw = tl.load(
                x_ptr + batch_base + c * spatial_i64 + s_offs,
                mask=s_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            val = v_raw.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                val = (val + pb).to(out_ty).to(tl.float32)
            sq_sum += val * val

        inv_norm = scale / tl.maximum(tl.sqrt(sq_sum), 1e-12)

        for c in range(channels):
            col_ptr = batch_base + c * spatial_i64 + s_offs
            x_in = tl.load(x_ptr + col_ptr, mask=s_mask, other=0.0, eviction_policy="evict_first")
            x_f = x_in.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                x_f = (x_f + pb).to(out_ty).to(tl.float32)
            g = tl.load(gamma_ptr + c).to(tl.float32)
            y = x_f * inv_norm * g
            if has_bias:
                y = y + tl.load(bias_ptr + c).to(tl.float32)
            if apply_silu:
                y_cast = y.to(out_ty).to(tl.float32)
                y = y_cast * tl.sigmoid(y_cast)
            tl.store(out_ptr + col_ptr, y.to(out_ty), mask=s_mask, eviction_policy="evict_first")

    # =========================================================================
    # 2. Super-fused (pre_bias +) RMS_norm + SiLU + Cache Snapshot + Causal Pad
    # =========================================================================

    @triton.jit
    def _wan_rms_norm_silu_causal_pad_cl_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        cache_in_ptr,
        out_pad_ptr,
        cache_out_ptr,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_left,
        t_out,
        h_out,
        w_out,
        ph: tl.constexpr,
        pw: tl.constexpr,
        t_cache_out,
        scale,
        channels: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
    ):
        """Super-fused NDHWC ``(pre_bias +) RMS_norm + SiLU + cache + cat + pad``."""
        w_blk = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        bt = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_ci_i64 = tl.cast(t_cache_in, tl.int64)
        t_pl_i64 = tl.cast(t_pad_left, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        t_co_i64 = tl.cast(t_cache_out, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)

        b = bt // t_out_i64
        to = bt - b * t_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        out_mask = wo_mask[:, None] & c_mask[None, :]
        out_ty = out_pad_ptr.dtype.element_ty

        out_row_base = ((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs
        out_ptrs = out_row_base[:, None] * c_i64 + c_offs[None, :]

        in_spatial_h = (ho >= ph) & (ho < ph + h_in_i64)
        in_spatial_w = (wo_offs >= pw) & (wo_offs < pw + w_in_i64)

        if (to < t_pl_i64) or (not in_spatial_h):
            tl.store(
                out_pad_ptr + out_ptrs,
                tl.zeros((block_w, block_c), dtype=out_ty),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            return

        hi = ho - ph
        wi = wo_offs - pw
        valid_2d = (wo_mask & in_spatial_w)[:, None] & c_mask[None, :]
        zero_v = tl.zeros((1, 1), dtype=out_ty)

        if has_cache_in and (to < t_pl_i64 + t_ci_i64):
            tc = to - t_pl_i64
            c_row = ((b * t_ci_i64 + tc) * h_in_i64 + hi) * w_in_i64 + wi
            c_ptrs = c_row[:, None] * c_i64 + c_offs[None, :]
            val = tl.load(cache_in_ptr + c_ptrs, mask=valid_2d, other=0.0, eviction_policy="evict_first")
            tl.store(
                out_pad_ptr + out_ptrs,
                tl.where(in_spatial_w[:, None], val, zero_v),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            if copy_prev_cache_to_cache_out and (tc == t_ci_i64 - 1):
                co_row = ((b * t_co_i64 + 0) * h_in_i64 + hi) * w_in_i64 + wi
                tl.store(
                    cache_out_ptr + co_row[:, None] * c_i64 + c_offs[None, :],
                    val,
                    mask=valid_2d,
                    eviction_policy="evict_first",
                )
            return

        ti = to - t_pl_i64 - t_ci_i64
        in_row = ((b * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi
        in_ptrs = in_row[:, None] * c_i64 + c_offs[None, :]
        x_in = tl.load(x_ptr + in_ptrs, mask=valid_2d, other=0.0, eviction_policy="evict_first")
        x_f = x_in.to(tl.float32)
        if has_pre_bias:
            pb = tl.load(pre_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            x_f = (x_f + pb).to(out_ty).to(tl.float32)

        sq_sum = tl.sum(x_f * x_f, axis=1)
        inv_norm = (scale * tl.rsqrt(tl.maximum(sq_sum, 1e-24)))[:, None]
        gamma = tl.load(gamma_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
        y = x_f * inv_norm * gamma
        if has_bias:
            bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y + bias
        y_cast = y.to(out_ty).to(tl.float32)
        y_silu = (y_cast * tl.sigmoid(y_cast)).to(out_ty)

        tl.store(
            out_pad_ptr + out_ptrs,
            tl.where(in_spatial_w[:, None], y_silu, zero_v),
            mask=out_mask,
            eviction_policy="evict_first",
        )

        if write_cache_out and (ti >= t_in_i64 - (t_co_i64 - (1 if copy_prev_cache_to_cache_out else 0))):
            co_slot = ti - (t_in_i64 - t_co_i64)
            co_row = ((b * t_co_i64 + co_slot) * h_in_i64 + hi) * w_in_i64 + wi
            tl.store(
                cache_out_ptr + co_row[:, None] * c_i64 + c_offs[None, :],
                y_silu,
                mask=valid_2d,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _wan_rms_norm_silu_causal_pad_cf_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        cache_in_ptr,
        out_pad_ptr,
        cache_out_ptr,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_left,
        t_out,
        h_out,
        w_out,
        ph: tl.constexpr,
        pw: tl.constexpr,
        t_cache_out,
        scale,
        channels: tl.constexpr,
        block_w: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
    ):
        """Interior-aligned coalesced NCDHW ``(pre_bias +) RMS_norm + SiLU + cache + pad``."""
        w_blk = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        bt = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_ci_i64 = tl.cast(t_cache_in, tl.int64)
        t_pl_i64 = tl.cast(t_pad_left, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        t_co_i64 = tl.cast(t_cache_out, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_pad_ptr.dtype.element_ty

        b = bt // t_out_i64
        to = bt - b * t_out_i64

        out_hw = h_out_i64 * w_out_i64
        out_thw = t_out_i64 * out_hw
        out_row = b * c_i64 * out_thw + to * out_hw + ho * w_out_i64

        num_interior_w = (w_in_i64 + block_w - 1) // block_w
        if w_blk >= num_interior_w:
            # Border zero columns: [0, pw) and [pw + w_in, w_out).
            total_border = 2 * pw
            for b_start in range(0, total_border, block_w):
                bi = b_start + tl.arange(0, block_w).to(tl.int64)
                b_mask = bi < total_border
                wo_b = tl.where(bi < pw, bi, pw + w_in_i64 + (bi - pw))
                zeros = tl.zeros((block_w,), dtype=out_ty)
                for c in range(channels):
                    tl.store(out_pad_ptr + out_row + c * out_thw + wo_b, zeros, mask=b_mask)
            return

        wi_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = wi_offs < w_in_i64
        wo_offs = pw + wi_offs
        in_spatial_h = (ho >= ph) & (ho < ph + h_in_i64)

        if (to < t_pl_i64) or (not in_spatial_h):
            zeros = tl.zeros((block_w,), dtype=out_ty)
            for c in range(channels):
                tl.store(
                    out_pad_ptr + out_row + c * out_thw + wo_offs,
                    zeros,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
            return

        hi = ho - ph
        in_hw = h_in_i64 * w_in_i64
        co_thw = t_co_i64 * in_hw

        if has_cache_in and (to < t_pl_i64 + t_ci_i64):
            tc = to - t_pl_i64
            c_thw = t_ci_i64 * in_hw
            c_base = b * c_i64 * c_thw + tc * in_hw + hi * w_in_i64 + wi_offs
            need_cache_copy = copy_prev_cache_to_cache_out and (tc == t_ci_i64 - 1)
            co_base = b * c_i64 * co_thw + hi * w_in_i64 + wi_offs
            for c in range(channels):
                val = tl.load(cache_in_ptr + c_base + c * c_thw, mask=w_mask, other=0.0, eviction_policy="evict_first")
                tl.store(
                    out_pad_ptr + out_row + c * out_thw + wo_offs,
                    val,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
                if need_cache_copy:
                    tl.store(
                        cache_out_ptr + co_base + c * co_thw,
                        val,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )
            return

        ti = to - t_pl_i64 - t_ci_i64
        in_thw = t_in_i64 * in_hw
        in_base = b * c_i64 * in_thw + ti * in_hw + hi * w_in_i64 + wi_offs

        sq_sum = tl.zeros((block_w,), dtype=tl.float32)
        for c in range(channels):
            v_raw = tl.load(x_ptr + in_base + c * in_thw, mask=w_mask, other=0.0, eviction_policy="evict_last")
            v = v_raw.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                v = (v + pb).to(out_ty).to(tl.float32)
            sq_sum += v * v
        inv_norm = scale * tl.rsqrt(tl.maximum(sq_sum, 1e-24))

        save_to_cache = write_cache_out and (ti >= t_in_i64 - (t_co_i64 - (1 if copy_prev_cache_to_cache_out else 0)))
        co_slot = ti - (t_in_i64 - t_co_i64)
        co_base = b * c_i64 * co_thw + co_slot * in_hw + hi * w_in_i64 + wi_offs

        for c in range(channels):
            x_in = tl.load(x_ptr + in_base + c * in_thw, mask=w_mask, other=0.0, eviction_policy="evict_first")
            x_f = x_in.to(tl.float32)
            if has_pre_bias:
                pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                x_f = (x_f + pb).to(out_ty).to(tl.float32)
            g = tl.load(gamma_ptr + c).to(tl.float32)
            y = x_f * inv_norm * g
            if has_bias:
                y = y + tl.load(bias_ptr + c).to(tl.float32)
            y_cast = y.to(out_ty).to(tl.float32)
            y_silu = (y_cast * tl.sigmoid(y_cast)).to(out_ty)
            tl.store(
                out_pad_ptr + out_row + c * out_thw + wo_offs,
                y_silu,
                mask=w_mask,
                eviction_policy="evict_first",
            )
            if save_to_cache:
                tl.store(
                    cache_out_ptr + co_base + c * co_thw,
                    y_silu,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )

    # =========================================================================
    # 3. Single-pass Causal Pad + Cache (+ optional pre_bias & align_c -> CL3D)
    # =========================================================================

    @triton.jit
    def _wan_causal_pad_cache_cl_kernel(
        x_ptr,
        pre_bias_ptr,
        cache_in_ptr,
        out_pad_ptr,
        cache_out_ptr,
        in_channels,
        out_channels,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_left,
        t_out,
        h_out,
        w_out,
        t_cache_out,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        sc_b,
        sc_c,
        sc_d,
        sc_h,
        sc_w,
        ph: tl.constexpr,
        pw: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
        pad_cache_zero_left: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Arbitrary-stride input -> CL3D ``out_pad`` (+ optional CL3D ``cache_out``)."""
        bt_c = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        in_c_i64 = tl.cast(in_channels, tl.int64)
        out_c_i64 = tl.cast(out_channels, tl.int64)
        c_tiles = (out_c_i64 + block_c - 1) // block_c
        c_blk = bt_c % c_tiles
        bt = bt_c // c_tiles

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_ci_i64 = tl.cast(t_cache_in, tl.int64)
        t_pl_i64 = tl.cast(t_pad_left, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        t_co_i64 = tl.cast(t_cache_out, tl.int64)

        b = bt // t_out_i64
        to = bt - b * t_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        out_c_mask = c_offs < out_c_i64
        in_c_mask = c_offs < in_c_i64
        out_mask = wo_mask[:, None] & out_c_mask[None, :]
        out_ty = out_pad_ptr.dtype.element_ty

        out_row = ((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs
        out_ptrs = out_row[:, None] * out_c_i64 + c_offs[None, :]

        in_spatial_h = (ho >= ph) & (ho < ph + h_in_i64)
        in_spatial_w = (wo_offs >= pw) & (wo_offs < pw + w_in_i64)

        if (to < t_pl_i64) or (not in_spatial_h):
            zeros = tl.zeros((block_w, block_c), dtype=out_ty)
            tl.store(out_pad_ptr + out_ptrs, zeros, mask=out_mask, eviction_policy="evict_first")
            if pad_cache_zero_left and in_spatial_h and (to == t_pl_i64 - 1):
                hi_z = ho - ph
                wi_z = wo_offs - pw
                valid_co_z = (wo_mask & in_spatial_w)[:, None] & in_c_mask[None, :]
                co_row_z = ((b * t_co_i64 + 0) * h_in_i64 + hi_z) * w_in_i64 + wi_z
                tl.store(
                    cache_out_ptr + co_row_z[:, None] * in_c_i64 + c_offs[None, :],
                    zeros,
                    mask=valid_co_z,
                    eviction_policy="evict_first",
                )
            return

        hi = ho - ph
        wi = wo_offs - pw
        valid_in = (wo_mask & in_spatial_w)[:, None] & in_c_mask[None, :]
        zero_v = tl.zeros((1, 1), dtype=out_ty)

        if has_cache_in and (to < t_pl_i64 + t_ci_i64):
            tc = to - t_pl_i64
            sc_b_i64 = tl.cast(sc_b, tl.int64)
            sc_c_i64 = tl.cast(sc_c, tl.int64)
            sc_d_i64 = tl.cast(sc_d, tl.int64)
            sc_h_i64 = tl.cast(sc_h, tl.int64)
            sc_w_i64 = tl.cast(sc_w, tl.int64)
            c_ptrs = b * sc_b_i64 + tc * sc_d_i64 + hi * sc_h_i64 + wi[:, None] * sc_w_i64 + c_offs[None, :] * sc_c_i64
            val = tl.load(cache_in_ptr + c_ptrs, mask=valid_in, other=0.0, eviction_policy="evict_first")
            tl.store(
                out_pad_ptr + out_ptrs,
                tl.where(in_spatial_w[:, None] & in_c_mask[None, :], val, zero_v),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            if copy_prev_cache_to_cache_out and (tc == t_ci_i64 - 1):
                co_row = ((b * t_co_i64 + 0) * h_in_i64 + hi) * w_in_i64 + wi
                tl.store(
                    cache_out_ptr + co_row[:, None] * in_c_i64 + c_offs[None, :],
                    val,
                    mask=valid_in,
                    eviction_policy="evict_first",
                )
            return

        ti = to - t_pl_i64 - t_ci_i64
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        in_ptrs = b * sx_b_i64 + ti * sx_d_i64 + hi * sx_h_i64 + wi[:, None] * sx_w_i64 + c_offs[None, :] * sx_c_i64
        val = tl.load(x_ptr + in_ptrs, mask=valid_in, other=0.0, eviction_policy="evict_first")
        if has_pre_bias:
            pb = tl.load(pre_bias_ptr + c_offs, mask=in_c_mask, other=0.0).to(tl.float32)[None, :]
            val = (val.to(tl.float32) + pb).to(out_ty)

        tl.store(
            out_pad_ptr + out_ptrs,
            tl.where(in_spatial_w[:, None] & in_c_mask[None, :], val, zero_v),
            mask=out_mask,
            eviction_policy="evict_first",
        )
        take_from_x = t_co_i64 - (1 if (copy_prev_cache_to_cache_out or pad_cache_zero_left) else 0)
        if write_cache_out and (ti >= t_in_i64 - take_from_x):
            co_slot = ti - (t_in_i64 - t_co_i64)
            co_row = ((b * t_co_i64 + co_slot) * h_in_i64 + hi) * w_in_i64 + wi
            tl.store(
                cache_out_ptr + co_row[:, None] * in_c_i64 + c_offs[None, :],
                val,
                mask=valid_in,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _wan_causal_pad_cache_cf_kernel(
        x_ptr,
        pre_bias_ptr,
        cache_in_ptr,
        out_pad_ptr,
        cache_out_ptr,
        channels,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_left,
        t_out,
        h_out,
        w_out,
        t_cache_out,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        sc_b,
        sc_c,
        sc_d,
        sc_h,
        sc_w,
        ph: tl.constexpr,
        pw: tl.constexpr,
        has_pre_bias: tl.constexpr,
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
        pad_cache_zero_left: tl.constexpr,
        block_w: tl.constexpr,
    ):
        """Arbitrary-stride input -> NCDHW ``out_pad`` (+ optional NCDHW ``cache_out``)."""
        bct = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        c_i64 = tl.cast(channels, tl.int64)
        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_ci_i64 = tl.cast(t_cache_in, tl.int64)
        t_pl_i64 = tl.cast(t_pad_left, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        t_co_i64 = tl.cast(t_cache_out, tl.int64)
        out_ty = out_pad_ptr.dtype.element_ty

        to = bct % t_out_i64
        bc = bct // t_out_i64
        c = bc % c_i64
        b = bc // c_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64

        out_base = (((b * c_i64 + c) * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs
        in_spatial_h = (ho >= ph) & (ho < ph + h_in_i64)
        in_spatial_w = (wo_offs >= pw) & (wo_offs < pw + w_in_i64)

        if (to < t_pl_i64) or (not in_spatial_h):
            zeros = tl.zeros((block_w,), dtype=out_ty)
            tl.store(out_pad_ptr + out_base, zeros, mask=wo_mask, eviction_policy="evict_first")
            if pad_cache_zero_left and in_spatial_h and (to == t_pl_i64 - 1):
                hi_z = ho - ph
                wi_z = wo_offs - pw
                co_base_z = (((b * c_i64 + c) * t_co_i64 + 0) * h_in_i64 + hi_z) * w_in_i64 + wi_z
                tl.store(cache_out_ptr + co_base_z, zeros, mask=wo_mask & in_spatial_w, eviction_policy="evict_first")
            return

        hi = ho - ph
        wi = wo_offs - pw
        valid = wo_mask & in_spatial_w
        zero_v = tl.zeros((1,), dtype=out_ty)

        if has_cache_in and (to < t_pl_i64 + t_ci_i64):
            tc = to - t_pl_i64
            c_ptrs = (
                b * tl.cast(sc_b, tl.int64)
                + c * tl.cast(sc_c, tl.int64)
                + tc * tl.cast(sc_d, tl.int64)
                + hi * tl.cast(sc_h, tl.int64)
                + wi * tl.cast(sc_w, tl.int64)
            )
            val = tl.load(cache_in_ptr + c_ptrs, mask=valid, other=0.0, eviction_policy="evict_first")
            tl.store(
                out_pad_ptr + out_base,
                tl.where(in_spatial_w, val, zero_v),
                mask=wo_mask,
                eviction_policy="evict_first",
            )
            if copy_prev_cache_to_cache_out and (tc == t_ci_i64 - 1):
                co_base = (((b * c_i64 + c) * t_co_i64 + 0) * h_in_i64 + hi) * w_in_i64 + wi
                tl.store(cache_out_ptr + co_base, val, mask=valid, eviction_policy="evict_first")
            return

        ti = to - t_pl_i64 - t_ci_i64
        in_ptrs = (
            b * tl.cast(sx_b, tl.int64)
            + c * tl.cast(sx_c, tl.int64)
            + ti * tl.cast(sx_d, tl.int64)
            + hi * tl.cast(sx_h, tl.int64)
            + wi * tl.cast(sx_w, tl.int64)
        )
        val = tl.load(x_ptr + in_ptrs, mask=valid, other=0.0, eviction_policy="evict_first")
        if has_pre_bias:
            pb = tl.load(pre_bias_ptr + c).to(tl.float32)
            val = (val.to(tl.float32) + pb).to(out_ty)
        tl.store(
            out_pad_ptr + out_base,
            tl.where(in_spatial_w, val, zero_v),
            mask=wo_mask,
            eviction_policy="evict_first",
        )
        take_from_x = t_co_i64 - (1 if (copy_prev_cache_to_cache_out or pad_cache_zero_left) else 0)
        if write_cache_out and (ti >= t_in_i64 - take_from_x):
            co_slot = ti - (t_in_i64 - t_co_i64)
            co_base = (((b * c_i64 + c) * t_co_i64 + co_slot) * h_in_i64 + hi) * w_in_i64 + wi
            tl.store(cache_out_ptr + co_base, val, mask=valid, eviction_policy="evict_first")

    # =========================================================================
    # 4. Coalesced Residual + Bias Add (4D/5D, CL & CF)
    # =========================================================================

    @triton.jit
    def _wan_residual_bias_add_cl_kernel(
        x_ptr,
        h_ptr,
        h_bias_ptr,
        x_bias_ptr,
        out_ptr,
        total_voxels,
        channels,
        has_x: tl.constexpr,
        has_h_bias: tl.constexpr,
        has_x_bias: tl.constexpr,
        block_v: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Coalesced NHWC/NDHWC ``(x + x_bias) + (h + h_bias)``."""
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
        if has_h_bias:
            hb = tl.load(h_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            if has_x_bias:
                xb = tl.load(x_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
                x_val = (x_val.to(tl.float32) + xb).to(out_ty)
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_residual_bias_add_cf_kernel(
        x_ptr,
        h_ptr,
        h_bias_ptr,
        x_bias_ptr,
        out_ptr,
        spatial_size,
        channels,
        has_x: tl.constexpr,
        has_h_bias: tl.constexpr,
        has_x_bias: tl.constexpr,
        block_s: tl.constexpr,
    ):
        """Coalesced NCHW/NCDHW ``(x + x_bias) + (h + h_bias)``."""
        s_blk = tl.program_id(0).to(tl.int64)
        bc = tl.program_id(1).to(tl.int64)

        spatial_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        c_idx = bc % c_i64
        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64

        ptrs = bc * spatial_i64 + s_offs
        h_val = tl.load(h_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
        if has_h_bias:
            hb = tl.load(h_bias_ptr + c_idx).to(tl.float32)
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
            if has_x_bias:
                xb = tl.load(x_bias_ptr + c_idx).to(tl.float32)
                x_val = (x_val.to(tl.float32) + xb).to(out_ty)
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=s_mask, eviction_policy="evict_first")

    # =========================================================================
    # 5. Bit-Exact Nearest Upsample 2x (4D NCHW & NHWC, load-once, int64)
    # =========================================================================

    @triton.jit
    def _wan_nearest_upsample_2x_nchw_kernel(
        x_ptr,
        out_ptr,
        h_in,
        w_in,
        block_w_in: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        """Bit-exact NCHW 2x2 nearest upsample; coalesced load + doubled join store."""
        plane = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        w_out_i64 = w_in_i64 * 2

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        w_mask = wi_offs < w_in_i64
        wo_offs = w_blk * (block_w_in * 2) + tl.arange(0, block_w_in * 2).to(tl.int64)
        wo_mask = wo_offs < w_out_i64

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            hi = h_base + r
            if hi < h_in_i64:
                vals = tl.load(
                    x_ptr + (plane * h_in_i64 + hi) * w_in_i64 + wi_offs,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
                doubled = tl.reshape(tl.join(vals, vals), (block_w_in * 2,))
                out_base0 = (plane * (h_in_i64 * 2) + hi * 2) * w_out_i64
                tl.store(out_ptr + out_base0 + wo_offs, doubled, mask=wo_mask, eviction_policy="evict_first")
                tl.store(
                    out_ptr + out_base0 + w_out_i64 + wo_offs, doubled, mask=wo_mask, eviction_policy="evict_first"
                )

    @triton.jit
    def _wan_nearest_upsample_2x_nhwc_kernel(
        x_ptr,
        out_ptr,
        h_in,
        w_in,
        channels,
        block_w_in: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Bit-exact NHWC 2x2 nearest upsample; loads each input pixel ONCE."""
        bh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        w_out_i64 = w_in_i64 * 2

        b = bh // h_in_i64
        hi = bh - b * h_in_i64

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        wi_mask = wi_offs < w_in_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        mask = wi_mask[:, None] & c_mask[None, :]

        in_rows = (b * h_in_i64 + hi) * w_in_i64 + wi_offs
        vals = tl.load(x_ptr + in_rows[:, None] * c_i64 + c_offs[None, :], mask=mask, eviction_policy="evict_first")

        out_row0 = (b * (h_in_i64 * 2) + hi * 2) * w_out_i64
        out_row1 = out_row0 + w_out_i64
        wo0 = wi_offs * 2
        wo1 = wo0 + 1
        tl.store(
            out_ptr + (out_row0 + wo0)[:, None] * c_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row0 + wo1)[:, None] * c_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row1 + wo0)[:, None] * c_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row1 + wo1)[:, None] * c_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )

    # =========================================================================
    # 6. Super-fused Time-Interleave (+ bias) + 5D->4D + 2x2 Nearest Upsample
    # =========================================================================

    @triton.jit
    def _wan_time_interleave_upsample_2x_cl_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        c_out,
        has_bias: tl.constexpr,
        block_w_in: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """NDHWC ``(b, 2c, t, h, w)`` (+ bias) -> NHWC ``(b*2t, c, 2h, 2w)``."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)
        c_in_i64 = c_out_i64 * 2
        t_out_i64 = t_in_i64 * 2
        w_out_i64 = w_in_i64 * 2
        out_ty = out_ptr.dtype.element_ty

        hi = bth % h_in_i64
        bt = bth // h_in_i64
        b = bt // t_out_i64
        to = bt - b * t_out_i64
        ti = to >> 1
        k = to & 1

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        wi_mask = wi_offs < w_in_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = wi_mask[:, None] & c_mask[None, :]

        in_ch = k * c_out_i64 + c_offs
        in_row = ((b * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi_offs
        vals = tl.load(x_ptr + in_row[:, None] * c_in_i64 + in_ch[None, :], mask=mask, eviction_policy="evict_first")
        if has_bias:
            bv = tl.load(bias_ptr + in_ch, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            vals = (vals.to(tl.float32) + bv).to(out_ty)

        out_row0 = (bt * (h_in_i64 * 2) + hi * 2) * w_out_i64
        out_row1 = out_row0 + w_out_i64
        wo0 = wi_offs * 2
        wo1 = wo0 + 1
        tl.store(
            out_ptr + (out_row0 + wo0)[:, None] * c_out_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row0 + wo1)[:, None] * c_out_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row1 + wo0)[:, None] * c_out_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + (out_row1 + wo1)[:, None] * c_out_i64 + c_offs[None, :],
            vals,
            mask=mask,
            eviction_policy="evict_first",
        )

    @triton.jit
    def _wan_time_interleave_upsample_2x_cf_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        c_out,
        has_bias: tl.constexpr,
        block_w_in: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        """NCDHW ``(b, 2c, t, h, w)`` (+ bias) -> NCHW ``(b*2t, c, 2h, 2w)``."""
        btc = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)
        c_in_i64 = c_out_i64 * 2
        t_out_i64 = t_in_i64 * 2
        w_out_i64 = w_in_i64 * 2
        out_ty = out_ptr.dtype.element_ty

        ci = btc % c_out_i64
        bt = btc // c_out_i64
        b = bt // t_out_i64
        to = bt - b * t_out_i64
        ti = to >> 1
        k = to & 1
        in_ch = k * c_out_i64 + ci

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        w_mask = wi_offs < w_in_i64
        wo_offs = w_blk * (block_w_in * 2) + tl.arange(0, block_w_in * 2).to(tl.int64)
        wo_mask = wo_offs < w_out_i64

        bv = tl.load(bias_ptr + in_ch).to(tl.float32) if has_bias else 0.0
        in_plane = (b * c_in_i64 + in_ch) * t_in_i64 + ti
        out_plane = bt * c_out_i64 + ci

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            hi = h_base + r
            if hi < h_in_i64:
                vals = tl.load(
                    x_ptr + (in_plane * h_in_i64 + hi) * w_in_i64 + wi_offs,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
                if has_bias:
                    vals = (vals.to(tl.float32) + bv).to(out_ty)
                doubled = tl.reshape(tl.join(vals, vals), (block_w_in * 2,))
                out_base0 = (out_plane * (h_in_i64 * 2) + hi * 2) * w_out_i64
                tl.store(out_ptr + out_base0 + wo_offs, doubled, mask=wo_mask, eviction_policy="evict_first")
                tl.store(
                    out_ptr + out_base0 + w_out_i64 + wo_offs, doubled, mask=wo_mask, eviction_policy="evict_first"
                )

    # =========================================================================
    # 7. Single-Pass 2D Right-Bottom Zero Pad (0, 1, 0, 1) — NHWC & NCHW
    # =========================================================================

    @triton.jit
    def _wan_zero_pad_2d_rb_nhwc_kernel(
        x_ptr,
        out_ptr,
        h_in,
        w_in,
        channels,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        nh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        h_out_i64 = h_in_i64 + 1
        w_out_i64 = w_in_i64 + 1
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        n_idx = nh // h_out_i64
        ho = nh - n_idx * h_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        out_mask = wo_mask[:, None] & c_mask[None, :]

        out_ptrs = ((n_idx * h_out_i64 + ho) * w_out_i64 + wo_offs)[:, None] * c_i64 + c_offs[None, :]
        if ho >= h_in_i64:
            tl.store(
                out_ptr + out_ptrs,
                tl.zeros((block_w, block_c), dtype=out_ty),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            return

        in_w = wo_offs < w_in_i64
        in_ptrs = ((n_idx * h_in_i64 + ho) * w_in_i64 + wo_offs)[:, None] * c_i64 + c_offs[None, :]
        vals = tl.load(
            x_ptr + in_ptrs,
            mask=(wo_mask & in_w)[:, None] & c_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        tl.store(out_ptr + out_ptrs, vals, mask=out_mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_zero_pad_2d_rb_nchw_kernel(
        x_ptr,
        out_ptr,
        h_in,
        w_in,
        block_w: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        plane = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        h_out_i64 = h_in_i64 + 1
        w_out_i64 = w_in_i64 + 1
        out_ty = out_ptr.dtype.element_ty

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        in_w = wo_offs < w_in_i64
        zeros = tl.zeros((block_w,), dtype=out_ty)

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            ho = h_base + r
            if ho < h_out_i64:
                out_row = (plane * h_out_i64 + ho) * w_out_i64 + wo_offs
                if ho < h_in_i64:
                    vals = tl.load(
                        x_ptr + (plane * h_in_i64 + ho) * w_in_i64 + wo_offs,
                        mask=wo_mask & in_w,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    tl.store(out_ptr + out_row, vals, mask=wo_mask, eviction_policy="evict_first")
                else:
                    tl.store(out_ptr + out_row, zeros, mask=wo_mask, eviction_policy="evict_first")

    # =========================================================================
    # 8. Super-Fused DupUp3D (+ h + h_bias) — CL3D & NCDHW
    # =========================================================================

    @triton.jit
    def _wan_dup_up_3d_add_cl_fs2_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        in_channels,
        out_channels,
        repeats: tl.constexpr,
        factor_t: tl.constexpr,
        t_trim: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w_in: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """128-bit coalesced CL3D ``DupUp3D`` (+ ``h + h_bias``) for ``factor_s == 2``."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = w_in_i64 * 2
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        th_out = t_out_i64 * h_out_i64
        b = bth // th_out
        rem = bth - b * th_out
        to = rem // h_out_i64
        ho = rem - to * h_out_i64

        to_full = to + t_trim
        ti = to_full // factor_t
        rt = to_full - ti * factor_t
        hi = ho >> 1
        rh = ho & 1

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        wi_mask = wi_offs < w_in_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = wi_mask[:, None] & c_mask[None, :]

        # Map (co, rt, rh, rw=0/1) -> input channel ci.
        flat0 = ((c_offs * factor_t + rt) * 2 + rh) * 2
        ci0 = flat0 // repeats
        x_row = ((b * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi_offs
        x0 = tl.load(x_ptr + x_row[:, None] * c_in_i64 + ci0[None, :], mask=mask, other=0.0)
        if repeats % 2 == 0:
            x1 = x0
        else:
            ci1 = (flat0 + 1) // repeats
            x1 = tl.load(x_ptr + x_row[:, None] * c_in_i64 + ci1[None, :], mask=mask, other=0.0)

        out_base = ((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64
        p0 = (out_base + wi_offs * 2)[:, None] * c_out_i64 + c_offs[None, :]
        p1 = (out_base + wi_offs * 2 + 1)[:, None] * c_out_i64 + c_offs[None, :]

        if has_h:
            h0 = tl.load(h_ptr + p0, mask=mask, other=0.0, eviction_policy="evict_first")
            h1 = tl.load(h_ptr + p1, mask=mask, other=0.0, eviction_policy="evict_first")
            if has_h_bias:
                hb = tl.load(h_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
                h0 = (h0.to(tl.float32) + hb).to(out_ty)
                h1 = (h1.to(tl.float32) + hb).to(out_ty)
            out0 = (h0.to(tl.float32) + x0.to(tl.float32)).to(out_ty)
            out1 = (h1.to(tl.float32) + x1.to(tl.float32)).to(out_ty)
        else:
            out0 = x0
            out1 = x1
        tl.store(out_ptr + p0, out0, mask=mask, eviction_policy="evict_first")
        tl.store(out_ptr + p1, out1, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_dup_up_3d_add_cl_gen_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        in_channels,
        out_channels,
        repeats: tl.constexpr,
        factor_t: tl.constexpr,
        factor_s: tl.constexpr,
        t_trim: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """General CL3D ``DupUp3D`` (+ ``h + h_bias``) for arbitrary ``factor_s``."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        th_out = t_out_i64 * h_out_i64
        b = bth // th_out
        rem = bth - b * th_out
        to = rem // h_out_i64
        ho = rem - to * h_out_i64

        to_full = to + t_trim
        ti = to_full // factor_t
        rt = to_full - ti * factor_t
        hi = ho // factor_s
        rh = ho - hi * factor_s

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        wi = wo_offs // factor_s
        rw = wo_offs - wi * factor_s

        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = wo_mask[:, None] & c_mask[None, :]

        flat_rep_ch = ((c_offs[None, :] * factor_t + rt) * factor_s + rh) * factor_s + rw[:, None]
        ci = flat_rep_ch // repeats
        x_row = ((b * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi
        x_val = tl.load(x_ptr + x_row[:, None] * c_in_i64 + ci, mask=mask, other=0.0)

        out_ptrs = (((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs)[:, None] * c_out_i64 + c_offs[
            None, :
        ]
        if has_h:
            h_val = tl.load(h_ptr + out_ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            if has_h_bias:
                hb = tl.load(h_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
                h_val = (h_val.to(tl.float32) + hb).to(out_ty)
            out_val = (h_val.to(tl.float32) + x_val.to(tl.float32)).to(out_ty)
        else:
            out_val = x_val
        tl.store(out_ptr + out_ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_dup_up_3d_add_cf_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        in_channels,
        out_channels,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        repeats: tl.constexpr,
        factor_t: tl.constexpr,
        factor_s: tl.constexpr,
        t_trim: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        """Coalesced NCDHW ``DupUp3D`` (+ arbitrary-stride ``h`` + ``h_bias``)."""
        bct = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_c_i64 = tl.cast(sh_c, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        to = bct % t_out_i64
        bc = bct // t_out_i64
        co = bc % c_out_i64
        b = bc // c_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = wo_offs < w_out_i64
        wi = wo_offs // factor_s
        rw = wo_offs - wi * factor_s

        to_full = to + t_trim
        ti = to_full // factor_t
        rt = to_full - ti * factor_t

        hb_scalar = tl.load(h_bias_ptr + co).to(tl.float32) if has_h_bias else 0.0
        h_bct = b * sh_b_i64 + co * sh_c_i64 + to * sh_d_i64 + wo_offs
        out_bct = ((b * c_out_i64 + co) * t_out_i64 + to) * h_out_i64

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            ho = h_base + r
            if ho < h_out_i64:
                hi = ho // factor_s
                rh = ho - hi * factor_s
                flat_rep_ch = ((co * factor_t + rt) * factor_s + rh) * factor_s + rw
                ci = flat_rep_ch // repeats
                in_idx = (((b * c_in_i64 + ci) * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi
                x_val = tl.load(x_ptr + in_idx, mask=w_mask, other=0.0)
                if has_h:
                    h_val = tl.load(
                        h_ptr + h_bct + ho * sh_h_i64, mask=w_mask, other=0.0, eviction_policy="evict_first"
                    )
                    if has_h_bias:
                        h_val = (h_val.to(tl.float32) + hb_scalar).to(out_ty)
                    out_val = (h_val.to(tl.float32) + x_val.to(tl.float32)).to(out_ty)
                else:
                    out_val = x_val
                tl.store(
                    out_ptr + (out_bct + ho) * w_out_i64 + wo_offs,
                    out_val,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )

    # =========================================================================
    # 9. Super-Fused AvgDown3D (+ h + h_bias) — CL3D & NCDHW
    # =========================================================================

    @triton.jit
    def _wan_avg_down_3d_add_cl_fast_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        in_channels,
        out_channels,
        pad_t,
        factor_t: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w: tl.constexpr,
        block_c_in: tl.constexpr,
    ):
        """128-bit coalesced CL3D ``AvgDown3D`` (+ ``h + h_bias``) for ``fs=2, group_size=4``."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        pad_t_i64 = tl.cast(pad_t, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        th_out = t_out_i64 * h_out_i64
        b = bth // th_out
        rem = bth - b * th_out
        to = rem // h_out_i64
        ho = rem - to * h_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        ci_offs = c_blk * block_c_in + tl.arange(0, block_c_in).to(tl.int64)
        ci_mask = ci_offs < c_in_i64
        in_mask = wo_mask[:, None] & ci_mask[None, :]

        hi0 = ho * 2
        hi1 = hi0 + 1
        wi0 = wo_offs * 2
        wi1 = wi0 + 1

        if factor_t == 1:
            ti0 = to - pad_t_i64
            base0 = ((b * t_in_i64 + ti0) * h_in_i64 + hi0) * w_in_i64
            base1 = ((b * t_in_i64 + ti0) * h_in_i64 + hi1) * w_in_i64
            m0 = in_mask & (ti0 >= 0)
            v00 = tl.load(x_ptr + (base0 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m0, other=0.0).to(
                tl.float32
            )
            v01 = tl.load(x_ptr + (base0 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m0, other=0.0).to(
                tl.float32
            )
            v10 = tl.load(x_ptr + (base1 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m0, other=0.0).to(
                tl.float32
            )
            v11 = tl.load(x_ptr + (base1 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m0, other=0.0).to(
                tl.float32
            )
            avg = ((v00 + v01 + v10 + v11) * 0.25).to(out_ty)

            out_ptrs = (((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs)[:, None] * c_out_i64 + ci_offs[
                None, :
            ]
            if has_h:
                h_val = tl.load(h_ptr + out_ptrs, mask=in_mask, other=0.0, eviction_policy="evict_first")
                if has_h_bias:
                    hb = tl.load(h_bias_ptr + ci_offs, mask=ci_mask, other=0.0).to(tl.float32)[None, :]
                    h_val = (h_val.to(tl.float32) + hb).to(out_ty)
                out_val = (h_val.to(tl.float32) + avg.to(tl.float32)).to(out_ty)
            else:
                out_val = avg
            tl.store(out_ptr + out_ptrs, out_val, mask=in_mask, eviction_policy="evict_first")
            return

        # factor_t == 2, group_size == 4 => c_out == 2 * c_in: co = 2*ci + rt
        t0 = to * 2 - pad_t_i64
        t1 = t0 + 1
        m_t0 = in_mask & (t0 >= 0)
        m_t1 = in_mask & (t1 >= 0)
        t0_safe = tl.maximum(t0, 0)
        t1_safe = tl.maximum(t1, 0)

        b0_0 = ((b * t_in_i64 + t0_safe) * h_in_i64 + hi0) * w_in_i64
        b0_1 = ((b * t_in_i64 + t0_safe) * h_in_i64 + hi1) * w_in_i64
        a0 = (
            tl.load(x_ptr + (b0_0 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t0, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b0_0 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t0, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b0_1 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t0, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b0_1 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t0, other=0.0).to(tl.float32)
        ) * 0.25

        b1_0 = ((b * t_in_i64 + t1_safe) * h_in_i64 + hi0) * w_in_i64
        b1_1 = ((b * t_in_i64 + t1_safe) * h_in_i64 + hi1) * w_in_i64
        a1 = (
            tl.load(x_ptr + (b1_0 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t1, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b1_0 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t1, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b1_1 + wi0)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t1, other=0.0).to(tl.float32)
            + tl.load(x_ptr + (b1_1 + wi1)[:, None] * c_in_i64 + ci_offs[None, :], mask=m_t1, other=0.0).to(tl.float32)
        ) * 0.25

        avg_inter = tl.reshape(tl.join(a0.to(out_ty), a1.to(out_ty)), (block_w, block_c_in * 2))
        co_offs = c_blk * (block_c_in * 2) + tl.arange(0, block_c_in * 2).to(tl.int64)
        co_mask = co_offs < c_out_i64
        out_mask = wo_mask[:, None] & co_mask[None, :]
        out_ptrs = (((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs)[:, None] * c_out_i64 + co_offs[
            None, :
        ]
        if has_h:
            h_val = tl.load(h_ptr + out_ptrs, mask=out_mask, other=0.0, eviction_policy="evict_first")
            if has_h_bias:
                hb = tl.load(h_bias_ptr + co_offs, mask=co_mask, other=0.0).to(tl.float32)[None, :]
                h_val = (h_val.to(tl.float32) + hb).to(out_ty)
            out_val = (h_val.to(tl.float32) + avg_inter.to(tl.float32)).to(out_ty)
        else:
            out_val = avg_inter
        tl.store(out_ptr + out_ptrs, out_val, mask=out_mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_avg_down_3d_add_cl_gen_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        in_channels,
        out_channels,
        pad_t,
        factor_t: tl.constexpr,
        factor_s: tl.constexpr,
        group_size: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """General CL3D ``AvgDown3D`` (+ ``h + h_bias``) for arbitrary factors/groups."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        pad_t_i64 = tl.cast(pad_t, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        th_out = t_out_i64 * h_out_i64
        b = bth // th_out
        rem = bth - b * th_out
        to = rem // h_out_i64
        ho = rem - to * h_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        co_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        co_mask = co_offs < c_out_i64
        mask = wo_mask[:, None] & co_mask[None, :]

        acc = tl.zeros((block_w, block_c), dtype=tl.float32)
        for j in tl.static_range(0, group_size):
            flat = co_offs * group_size + j
            rw = flat % factor_s
            tmp1 = flat // factor_s
            rh = tmp1 % factor_s
            tmp2 = tmp1 // factor_s
            rt = tmp2 % factor_t
            ci = tmp2 // factor_t
            ti = to * factor_t + rt - pad_t_i64
            hi = ho * factor_s + rh
            wi = wo_offs[:, None] * factor_s + rw[None, :]
            valid = mask & (ti[None, :] >= 0)
            ti_safe = tl.maximum(ti, 0)
            ptrs = (((b * t_in_i64 + ti_safe[None, :]) * h_in_i64 + hi[None, :]) * w_in_i64 + wi) * c_in_i64 + ci[
                None, :
            ]
            acc += tl.load(x_ptr + ptrs, mask=valid, other=0.0).to(tl.float32)

        avg = (acc * (1.0 / group_size)).to(out_ty)
        out_ptrs = (((b * t_out_i64 + to) * h_out_i64 + ho) * w_out_i64 + wo_offs)[:, None] * c_out_i64 + co_offs[
            None, :
        ]
        if has_h:
            h_val = tl.load(h_ptr + out_ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            if has_h_bias:
                hb = tl.load(h_bias_ptr + co_offs, mask=co_mask, other=0.0).to(tl.float32)[None, :]
                h_val = (h_val.to(tl.float32) + hb).to(out_ty)
            out_val = (h_val.to(tl.float32) + avg.to(tl.float32)).to(out_ty)
        else:
            out_val = avg
        tl.store(out_ptr + out_ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _wan_avg_down_3d_add_cf_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        in_channels,
        out_channels,
        pad_t,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        factor_t: tl.constexpr,
        factor_s: tl.constexpr,
        group_size: tl.constexpr,
        has_h: tl.constexpr,
        has_h_bias: tl.constexpr,
        block_w: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        """Coalesced NCDHW ``AvgDown3D`` (+ arbitrary-stride ``h`` + ``h_bias``)."""
        bct = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_in_i64 = tl.cast(t_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(in_channels, tl.int64)
        c_out_i64 = tl.cast(out_channels, tl.int64)
        pad_t_i64 = tl.cast(pad_t, tl.int64)
        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_c_i64 = tl.cast(sh_c, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        to = bct % t_out_i64
        bc = bct // t_out_i64
        co = bc % c_out_i64
        b = bc // c_out_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = wo_offs < w_out_i64
        hb_scalar = tl.load(h_bias_ptr + co).to(tl.float32) if has_h_bias else 0.0
        h_bct = b * sh_b_i64 + co * sh_c_i64 + to * sh_d_i64 + wo_offs
        out_bct = ((b * c_out_i64 + co) * t_out_i64 + to) * h_out_i64

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            ho = h_base + r
            if ho < h_out_i64:
                acc = tl.zeros((block_w,), dtype=tl.float32)
                for j in tl.static_range(0, group_size):
                    flat = co * group_size + j
                    rw = flat % factor_s
                    tmp1 = flat // factor_s
                    rh = tmp1 % factor_s
                    tmp2 = tmp1 // factor_s
                    rt = tmp2 % factor_t
                    ci = tmp2 // factor_t
                    ti = to * factor_t + rt - pad_t_i64
                    if ti >= 0:
                        hi = ho * factor_s + rh
                        wi = wo_offs * factor_s + rw
                        in_idx = (((b * c_in_i64 + ci) * t_in_i64 + ti) * h_in_i64 + hi) * w_in_i64 + wi
                        acc += tl.load(x_ptr + in_idx, mask=w_mask, other=0.0).to(tl.float32)
                avg = (acc * (1.0 / group_size)).to(out_ty)
                if has_h:
                    h_val = tl.load(
                        h_ptr + h_bct + ho * sh_h_i64, mask=w_mask, other=0.0, eviction_policy="evict_first"
                    )
                    if has_h_bias:
                        h_val = (h_val.to(tl.float32) + hb_scalar).to(out_ty)
                    out_val = (h_val.to(tl.float32) + avg.to(tl.float32)).to(out_ty)
                else:
                    out_val = avg
                tl.store(
                    out_ptr + (out_bct + ho) * w_out_i64 + wo_offs,
                    out_val,
                    mask=w_mask,
                    eviction_policy="evict_first",
                )

    # =========================================================================
    # 10. Single-Pass 2x2 Spatial Patchify & Unpatchify (WanVideoVAE38)
    # =========================================================================

    @triton.jit
    def _wan_patchify_2x_kernel(
        x_ptr,
        out_ptr,
        c_in,
        t_dim,
        h_out,
        w_out,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_c,
        so_d,
        so_h,
        so_w,
        block_w: tl.constexpr,
    ):
        """Exact ``b c f (h q) (w r) -> b (c r q) f h w`` (q=r=2) for any layout."""
        bt = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_i64 = tl.cast(t_dim, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        b = bt // t_i64
        f = bt - b * t_i64

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = wo_offs < w_out_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        in_bt = b * sx_b_i64 + f * sx_d_i64
        out_bthw = b * so_b_i64 + f * so_d_i64 + ho * so_h_i64 + wo_offs * so_w_i64

        for ci in range(c_in):
            in_btc = in_bt + ci * sx_c_i64
            for r in tl.static_range(0, 2):
                wi = wo_offs * 2 + r
                for q in tl.static_range(0, 2):
                    hi = ho * 2 + q
                    co = (ci * 2 + r) * 2 + q
                    val = tl.load(
                        x_ptr + in_btc + hi * sx_h_i64 + wi * sx_w_i64,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        out_ptr + out_bthw + co * so_c_i64,
                        val,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )

    @triton.jit
    def _wan_unpatchify_2x_kernel(
        x_ptr,
        pre_bias_ptr,
        out_ptr,
        c_out,
        t_dim,
        h_in,
        w_in,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_c,
        so_d,
        so_h,
        so_w,
        has_pre_bias: tl.constexpr,
        block_w_in: tl.constexpr,
    ):
        """Exact ``b (c r q) f h w -> b c f (h q) (w r)`` (q=r=2) for any layout."""
        bt = tl.program_id(0).to(tl.int64)
        hi = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_i64 = tl.cast(t_dim, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        out_ty = out_ptr.dtype.element_ty
        b = bt // t_i64
        f = bt - b * t_i64

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        w_mask = wi_offs < w_in_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        in_bthw = b * sx_b_i64 + f * sx_d_i64 + hi * sx_h_i64 + wi_offs * sx_w_i64
        out_bt = b * so_b_i64 + f * so_d_i64

        for co in range(c_out):
            out_btc = out_bt + co * so_c_i64
            for r in tl.static_range(0, 2):
                wo = wi_offs * 2 + r
                for q in tl.static_range(0, 2):
                    ho = hi * 2 + q
                    ci = (co * 2 + r) * 2 + q
                    val = tl.load(x_ptr + in_bthw + ci * sx_c_i64, mask=w_mask, eviction_policy="evict_first")
                    if has_pre_bias:
                        pb = tl.load(pre_bias_ptr + ci).to(tl.float32)
                        val = (val.to(tl.float32) + pb).to(out_ty)
                    tl.store(
                        out_ptr + out_btc + ho * so_h_i64 + wo * so_w_i64,
                        val,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )

    return {
        "triton": triton,
        "rms_cl": _wan_rms_norm_channels_last_kernel,
        "rms_cf": _wan_rms_norm_channels_first_kernel,
        "rms_pad_cl": _wan_rms_norm_silu_causal_pad_cl_kernel,
        "rms_pad_cf": _wan_rms_norm_silu_causal_pad_cf_kernel,
        "causal_pad_cl": _wan_causal_pad_cache_cl_kernel,
        "causal_pad_cf": _wan_causal_pad_cache_cf_kernel,
        "res_bias_cl": _wan_residual_bias_add_cl_kernel,
        "res_bias_cf": _wan_residual_bias_add_cf_kernel,
        "up2x_nchw": _wan_nearest_upsample_2x_nchw_kernel,
        "up2x_nhwc": _wan_nearest_upsample_2x_nhwc_kernel,
        "time_up_cl": _wan_time_interleave_upsample_2x_cl_kernel,
        "time_up_cf": _wan_time_interleave_upsample_2x_cf_kernel,
        "zpad2d_nhwc": _wan_zero_pad_2d_rb_nhwc_kernel,
        "zpad2d_nchw": _wan_zero_pad_2d_rb_nchw_kernel,
        "dup_up_cl_fs2": _wan_dup_up_3d_add_cl_fs2_kernel,
        "dup_up_cl_gen": _wan_dup_up_3d_add_cl_gen_kernel,
        "dup_up_cf": _wan_dup_up_3d_add_cf_kernel,
        "avg_dn_cl_fast": _wan_avg_down_3d_add_cl_fast_kernel,
        "avg_dn_cl_gen": _wan_avg_down_3d_add_cl_gen_kernel,
        "avg_dn_cf": _wan_avg_down_3d_add_cf_kernel,
        "patchify_2x": _wan_patchify_2x_kernel,
        "unpatchify_2x": _wan_unpatchify_2x_kernel,
    }


# =============================================================================
# Custom Op 1: worldfoundry::wan_vae_rms_norm
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_rms_norm", mutates_args=(), device_types="cuda")
def _wan_vae_rms_norm_op(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    channels = x.shape[1]
    has_bias = bias is not None
    has_pre_bias = pre_bias is not None
    gamma_1d = gamma.reshape(channels).contiguous()
    bias_1d = bias.reshape(channels).contiguous() if bias is not None else gamma_1d
    pre_bias_1d = pre_bias.reshape(channels).contiguous() if pre_bias is not None else gamma_1d

    if _is_channels_last_nd(x):
        out = _empty_with_layout(tuple(x.shape), device=x.device, dtype=x.dtype, channels_last=True)
        total_pixels = x.numel() // channels
        if total_pixels == 0:
            return out
        block_p, block_c, num_warps = _select_cl_tile_w_warps(channels)
        kernels["rms_cl"][(_cdiv(total_pixels, block_p),)](
            x,
            gamma_1d,
            bias_1d,
            pre_bias_1d,
            out,
            total_pixels,
            float(scale),
            channels=channels,
            block_p=block_p,
            block_c=block_c,
            has_bias=has_bias,
            has_pre_bias=has_pre_bias,
            apply_silu=bool(apply_silu),
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty(x_c.shape, device=x.device, dtype=x.dtype)
    batch_size = x_c.shape[0]
    spatial_size = x_c.numel() // (batch_size * channels)
    if spatial_size == 0 or batch_size == 0:
        return out
    block_s = (
        256
        if (spatial_size >= 256 and channels <= 256)
        else (128 if spatial_size >= 128 else _next_power_of_2(max(spatial_size, 16)))
    )
    num_warps = 8 if block_s >= 256 else 4
    kernels["rms_cf"][(_cdiv(spatial_size, block_s), batch_size)](
        x_c,
        gamma_1d,
        bias_1d,
        pre_bias_1d,
        out,
        spatial_size,
        float(scale),
        channels=channels,
        block_s=block_s,
        has_bias=has_bias,
        has_pre_bias=has_pre_bias,
        apply_silu=bool(apply_silu),
        num_warps=num_warps,
    )
    return out


@_wan_vae_rms_norm_op.register_fake
def _wan_vae_rms_norm_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    del gamma, bias, pre_bias, scale, apply_silu
    return _empty_with_layout(tuple(x.shape), device=x.device, dtype=x.dtype, channels_last=_is_channels_last_nd(x))


def can_use_wan_vae_rms_norm(x: torch.Tensor, *, channel_first: bool = True) -> bool:
    """Return True when the Triton WanVAE RMSNorm(+SiLU) kernel is eligible."""
    if not channel_first or not x.is_cuda or torch.is_grad_enabled():
        return False
    if x.ndim not in (4, 5) or x.dtype not in _SUPPORTED_DTYPES:
        return False
    return 0 < x.shape[1] <= 1024


def wan_vae_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    scale: float | None = None,
    apply_silu: bool = False,
    pre_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run lossless fused WanVAE ``(pre_bias +) RMS_norm`` with optional ``SiLU`` epilogue."""
    eff_scale = math.sqrt(float(x.shape[1])) if scale is None else float(scale)
    if (
        can_use_wan_vae_rms_norm(x, channel_first=True)
        and gamma.device == x.device
        and (bias is None or bias.device == x.device)
        and (pre_bias is None or pre_bias.device == x.device)
    ):
        return _wan_vae_rms_norm_op(x, gamma, bias, pre_bias, eff_scale, bool(apply_silu))
    inp = x + pre_bias.view(1, -1, *([1] * (x.ndim - 2))) if pre_bias is not None else x
    y = F.normalize(inp, dim=1) * eff_scale * gamma
    if bias is not None:
        y = y + bias
    return F.silu(y) if apply_silu else y


# =============================================================================
# Custom Op 2: worldfoundry::wan_vae_rms_norm_silu_causal_pad
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_rms_norm_silu_causal_pad", mutates_args=(), device_types="cuda")
def _wan_vae_rms_norm_silu_causal_pad_op(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    scale: float,
    write_cache: bool,
    cache_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()

    b, c, t_in, h_in, w_in = x.shape
    has_bias = bias is not None
    has_pre_bias = pre_bias is not None
    gamma_1d = gamma.reshape(c).contiguous()
    bias_1d = bias.reshape(c).contiguous() if bias is not None else gamma_1d
    pre_bias_1d = pre_bias.reshape(c).contiguous() if pre_bias is not None else gamma_1d

    has_cache_in = cache_in is not None and pad_t_left > 0
    if has_cache_in:
        assert cache_in is not None
        t_cache_in = int(cache_in.shape[2])
        t_pad_rem = max(0, int(pad_t_left) - t_cache_in)
    else:
        t_cache_in = 0
        t_pad_rem = int(pad_t_left)

    t_out = t_pad_rem + t_cache_in + t_in
    h_out = h_in + 2 * pad_h
    w_out = w_in + 2 * pad_w

    copy_prev_cache_to_cache_out = bool(write_cache and t_in < cache_t and has_cache_in)
    t_cache_out = min(t_in, cache_t) + (1 if copy_prev_cache_to_cache_out else 0) if write_cache else 0

    is_cl = is_channels_last_3d(x)
    out_pad = _empty_with_layout((b, c, t_out, h_out, w_out), device=x.device, dtype=x.dtype, channels_last=is_cl)
    cache_out = (
        _empty_with_layout((b, c, max(t_cache_out, 1), h_in, w_in), device=x.device, dtype=x.dtype, channels_last=is_cl)
        if write_cache
        else out_pad.new_empty((0,))
    )
    if out_pad.numel() == 0:
        return out_pad, cache_out

    if is_cl:
        cache_in_t = (
            cache_in
            if (
                has_cache_in
                and cache_in is not None
                and cache_in.device == x.device
                and cache_in.dtype == x.dtype
                and is_channels_last_3d(cache_in)
            )
            else (
                cache_in.to(device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
                if has_cache_in and cache_in is not None
                else out_pad
            )
        )
        block_w, block_c, num_warps = _select_cl_tile_w_warps(c)
        kernels["rms_pad_cl"][(_cdiv(w_out, block_w), h_out, b * t_out)](
            x,
            gamma_1d,
            bias_1d,
            pre_bias_1d,
            cache_in_t,
            out_pad,
            cache_out,
            t_in,
            h_in,
            w_in,
            t_cache_in,
            t_pad_rem,
            t_out,
            h_out,
            w_out,
            ph=pad_h,
            pw=pad_w,
            t_cache_out=t_cache_out,
            scale=float(scale),
            channels=c,
            block_w=block_w,
            block_c=block_c,
            has_bias=has_bias,
            has_pre_bias=has_pre_bias,
            has_cache_in=has_cache_in,
            write_cache_out=bool(write_cache),
            copy_prev_cache_to_cache_out=copy_prev_cache_to_cache_out,
            num_warps=num_warps,
        )
        return out_pad, cache_out

    x_c = x if x.is_contiguous() else x.contiguous()
    cache_in_t = (
        cache_in.to(device=x.device, dtype=x.dtype).contiguous() if has_cache_in and cache_in is not None else out_pad
    )
    block_w = 256 if (w_in >= 256 and c <= 256) else (128 if w_in >= 128 else _next_power_of_2(max(w_in, 16)))
    num_warps = 8 if block_w >= 256 else 4
    num_w_tiles = _cdiv(w_in, block_w) + (1 if pad_w > 0 else 0)
    kernels["rms_pad_cf"][(num_w_tiles, h_out, b * t_out)](
        x_c,
        gamma_1d,
        bias_1d,
        pre_bias_1d,
        cache_in_t,
        out_pad,
        cache_out,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_rem,
        t_out,
        h_out,
        w_out,
        ph=pad_h,
        pw=pad_w,
        t_cache_out=t_cache_out,
        scale=float(scale),
        channels=c,
        block_w=block_w,
        has_bias=has_bias,
        has_pre_bias=has_pre_bias,
        has_cache_in=has_cache_in,
        write_cache_out=bool(write_cache),
        copy_prev_cache_to_cache_out=copy_prev_cache_to_cache_out,
        num_warps=num_warps,
    )
    return out_pad, cache_out


@_wan_vae_rms_norm_silu_causal_pad_op.register_fake
def _wan_vae_rms_norm_silu_causal_pad_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    scale: float,
    write_cache: bool,
    cache_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gamma, bias, pre_bias, scale
    b, c, t_in, h_in, w_in = x.shape
    has_cache_in = cache_in is not None and pad_t_left > 0
    t_cache_in = int(cache_in.shape[2]) if has_cache_in and cache_in is not None else 0
    t_pad_rem = max(0, int(pad_t_left) - t_cache_in)
    t_out = t_pad_rem + t_cache_in + t_in
    is_cl = is_channels_last_3d(x)
    out_pad = _empty_with_layout(
        (b, c, t_out, h_in + 2 * pad_h, w_in + 2 * pad_w),
        device=x.device,
        dtype=x.dtype,
        channels_last=is_cl,
    )
    copy_prev = bool(write_cache and t_in < cache_t and has_cache_in)
    t_cache_out = min(t_in, cache_t) + (1 if copy_prev else 0)
    cache_out = (
        _empty_with_layout(
            (b, c, max(t_cache_out, 1), h_in, w_in),
            device=x.device,
            dtype=x.dtype,
            channels_last=is_cl,
        )
        if write_cache
        else x.new_empty((0,))
    )
    return out_pad, cache_out


def wan_vae_rms_norm_silu_causal_pad(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    scale: float | None = None,
    write_cache: bool = True,
    cache_t: int = 2,
    pre_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Super-fused ``(pre_bias +) RMS_norm + SiLU + cache snapshot + cat + F.pad``."""
    eff_scale = math.sqrt(float(x.shape[1])) if scale is None else float(scale)
    if (
        can_use_wan_vae_rms_norm(x, channel_first=True)
        and x.ndim == 5
        and gamma.device == x.device
        and (bias is None or bias.device == x.device)
        and (pre_bias is None or pre_bias.device == x.device)
    ):
        return _wan_vae_rms_norm_silu_causal_pad_op(
            x,
            gamma,
            bias,
            pre_bias,
            cache_in,
            int(pad_w),
            int(pad_h),
            int(pad_t_left),
            eff_scale,
            bool(write_cache),
            int(cache_t),
        )
    inp = x + pre_bias.view(1, -1, 1, 1, 1) if pre_bias is not None else x
    y = F.normalize(inp, dim=1) * eff_scale * gamma
    if bias is not None:
        y = y + bias
    y = F.silu(y)
    cache_out = y.new_empty((0,))
    if write_cache:
        cache_out = y[:, :, -cache_t:].clone()
        if cache_out.shape[2] < cache_t and cache_in is not None and pad_t_left > 0:
            cache_out = torch.cat([cache_in[:, :, -1:].to(device=y.device, dtype=y.dtype), cache_out], dim=2)
    t_pad_rem = int(pad_t_left)
    if cache_in is not None and pad_t_left > 0:
        y_cat = torch.cat([cache_in.to(device=y.device, dtype=y.dtype), y], dim=2)
        t_pad_rem = max(0, t_pad_rem - int(cache_in.shape[2]))
    else:
        y_cat = y
    out_pad = F.pad(y_cat, (pad_w, pad_w, pad_h, pad_h, t_pad_rem, 0))
    return out_pad, cache_out


# =============================================================================
# Custom Op 3: worldfoundry::wan_vae_causal_pad_cache
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_causal_pad_cache", mutates_args=(), device_types="cuda")
def _wan_vae_causal_pad_cache_op(
    x: torch.Tensor,
    pre_bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    write_cache: bool,
    cache_t: int,
    pad_cache_zero_left: bool,
    align_c: int,
    output_cl3d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()

    b, c_in, t_in, h_in, w_in = x.shape
    out_c = align_c if align_c > c_in else c_in
    out_is_cl = (
        True if (align_c > c_in or output_cl3d == 1) else (False if output_cl3d == 0 else is_channels_last_3d(x))
    )

    has_pre_bias = pre_bias is not None
    pb_1d = pre_bias.reshape(c_in).contiguous() if pre_bias is not None else x

    has_cache_in = cache_in is not None and pad_t_left > 0
    if has_cache_in:
        assert cache_in is not None
        cache_in_dev = (
            cache_in
            if (cache_in.device == x.device and cache_in.dtype == x.dtype)
            else cache_in.to(device=x.device, dtype=x.dtype)
        )
        t_cache_in = int(cache_in_dev.shape[2])
        t_pad_rem = max(0, int(pad_t_left) - t_cache_in)
    else:
        cache_in_dev = x
        t_cache_in = 0
        t_pad_rem = int(pad_t_left)

    t_out = t_pad_rem + t_cache_in + t_in
    h_out = h_in + 2 * pad_h
    w_out = w_in + 2 * pad_w

    copy_prev = bool(write_cache and t_in < cache_t and has_cache_in)
    zero_left_cache = bool(
        write_cache and pad_cache_zero_left and t_in < cache_t and not has_cache_in and t_pad_rem > 0
    )
    t_cache_out = (min(t_in, cache_t) + (1 if (copy_prev or zero_left_cache) else 0)) if write_cache else 0

    out_pad = _empty_with_layout(
        (b, out_c, t_out, h_out, w_out), device=x.device, dtype=x.dtype, channels_last=out_is_cl
    )
    cache_out = (
        _empty_with_layout(
            (b, c_in, max(t_cache_out, 1), h_in, w_in), device=x.device, dtype=x.dtype, channels_last=out_is_cl
        )
        if write_cache
        else out_pad.new_empty((0,))
    )
    if out_pad.numel() == 0:
        return out_pad, cache_out

    if out_is_cl:
        block_c = 256 if out_c >= 256 else _next_power_of_2(max(out_c, 8))
        block_w = (
            128
            if block_c <= 16
            else (64 if block_c <= 32 else (32 if block_c <= 64 else (16 if block_c <= 128 else 8)))
        )
        num_warps = 4 if block_w * block_c <= 2048 else 8
        c_tiles = _cdiv(out_c, block_c)
        kernels["causal_pad_cl"][(b * t_out * c_tiles, h_out, _cdiv(w_out, block_w))](
            x,
            pb_1d,
            cache_in_dev,
            out_pad,
            cache_out,
            c_in,
            out_c,
            t_in,
            h_in,
            w_in,
            t_cache_in,
            t_pad_rem,
            t_out,
            h_out,
            w_out,
            t_cache_out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            cache_in_dev.stride(0),
            cache_in_dev.stride(1),
            cache_in_dev.stride(2),
            cache_in_dev.stride(3),
            cache_in_dev.stride(4),
            ph=pad_h,
            pw=pad_w,
            has_pre_bias=has_pre_bias,
            has_cache_in=has_cache_in,
            write_cache_out=bool(write_cache),
            copy_prev_cache_to_cache_out=copy_prev,
            pad_cache_zero_left=zero_left_cache,
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out_pad, cache_out

    block_w = 256 if w_out >= 256 else _next_power_of_2(max(w_out, 16))
    kernels["causal_pad_cf"][(b * c_in * t_out, h_out, _cdiv(w_out, block_w))](
        x,
        pb_1d,
        cache_in_dev,
        out_pad,
        cache_out,
        c_in,
        t_in,
        h_in,
        w_in,
        t_cache_in,
        t_pad_rem,
        t_out,
        h_out,
        w_out,
        t_cache_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        cache_in_dev.stride(0),
        cache_in_dev.stride(1),
        cache_in_dev.stride(2),
        cache_in_dev.stride(3),
        cache_in_dev.stride(4),
        ph=pad_h,
        pw=pad_w,
        has_pre_bias=has_pre_bias,
        has_cache_in=has_cache_in,
        write_cache_out=bool(write_cache),
        copy_prev_cache_to_cache_out=copy_prev,
        pad_cache_zero_left=zero_left_cache,
        block_w=block_w,
        num_warps=4,
    )
    return out_pad, cache_out


@_wan_vae_causal_pad_cache_op.register_fake
def _wan_vae_causal_pad_cache_fake(
    x: torch.Tensor,
    pre_bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    write_cache: bool,
    cache_t: int,
    pad_cache_zero_left: bool,
    align_c: int,
    output_cl3d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del pre_bias
    b, c_in, t_in, h_in, w_in = x.shape
    out_c = align_c if align_c > c_in else c_in
    out_is_cl = (
        True if (align_c > c_in or output_cl3d == 1) else (False if output_cl3d == 0 else is_channels_last_3d(x))
    )
    has_cache_in = cache_in is not None and pad_t_left > 0
    t_cache_in = int(cache_in.shape[2]) if has_cache_in and cache_in is not None else 0
    t_pad_rem = max(0, int(pad_t_left) - t_cache_in)
    t_out = t_pad_rem + t_cache_in + t_in
    out_pad = _empty_with_layout(
        (b, out_c, t_out, h_in + 2 * pad_h, w_in + 2 * pad_w),
        device=x.device,
        dtype=x.dtype,
        channels_last=out_is_cl,
    )
    copy_prev = bool(write_cache and t_in < cache_t and has_cache_in)
    zero_left = bool(write_cache and pad_cache_zero_left and t_in < cache_t and not has_cache_in and t_pad_rem > 0)
    t_cache_out = (min(t_in, cache_t) + (1 if (copy_prev or zero_left) else 0)) if write_cache else 0
    cache_out = (
        _empty_with_layout(
            (b, c_in, max(t_cache_out, 1), h_in, w_in),
            device=x.device,
            dtype=x.dtype,
            channels_last=out_is_cl,
        )
        if write_cache
        else x.new_empty((0,))
    )
    return out_pad, cache_out


def can_use_wan_vae_causal_pad_cache(
    x: torch.Tensor,
    *,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    align_c: int = 0,
) -> bool:
    if not x.is_cuda or torch.is_grad_enabled() or x.ndim != 5 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if pad_w < 0 or pad_h < 0 or pad_t_left < 0:
        return False
    if align_c > 0 and (align_c < x.shape[1] or (align_c & (align_c - 1)) != 0):
        return False
    return x.shape[1] > 0 and x.shape[2] > 0 and x.shape[3] > 0 and x.shape[4] > 0


def wan_vae_causal_pad_cache(
    x: torch.Tensor,
    *,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    write_cache: bool = True,
    cache_t: int = 2,
    pre_bias: torch.Tensor | None = None,
    pad_cache_zero_left: bool = False,
    align_c: int = 0,
    output_cl3d: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass causal cache snapshot + concat + 3D zero-pad (+ optional pre_bias / align_c)."""
    out_cl_flag = -1 if output_cl3d is None else (1 if output_cl3d else 0)
    if can_use_wan_vae_causal_pad_cache(x, pad_w=pad_w, pad_h=pad_h, pad_t_left=pad_t_left, align_c=align_c) and (
        pre_bias is None or pre_bias.device == x.device
    ):
        return _wan_vae_causal_pad_cache_op(
            x,
            pre_bias,
            cache_in,
            int(pad_w),
            int(pad_h),
            int(pad_t_left),
            bool(write_cache),
            int(cache_t),
            bool(pad_cache_zero_left),
            int(align_c),
            out_cl_flag,
        )
    x_eff = x + pre_bias.view(1, -1, 1, 1, 1) if pre_bias is not None else x
    cache_out = x_eff.new_empty((0,))
    if write_cache:
        cache_out = x_eff[:, :, -cache_t:].clone()
        if cache_out.shape[2] < cache_t:
            if cache_in is not None and pad_t_left > 0:
                cache_out = torch.cat(
                    [cache_in[:, :, -1:].to(device=x_eff.device, dtype=x_eff.dtype), cache_out], dim=2
                )
            elif pad_cache_zero_left:
                cache_out = torch.cat([torch.zeros_like(cache_out), cache_out], dim=2)
    t_pad_rem = int(pad_t_left)
    if cache_in is not None and pad_t_left > 0:
        x_cat = torch.cat([cache_in.to(device=x_eff.device, dtype=x_eff.dtype), x_eff], dim=2)
        t_pad_rem = max(0, t_pad_rem - int(cache_in.shape[2]))
    else:
        x_cat = x_eff
    out_pad = F.pad(x_cat, (pad_w, pad_w, pad_h, pad_h, t_pad_rem, 0))
    if align_c > out_pad.shape[1]:
        out_pad = F.pad(out_pad, (0, 0, 0, 0, 0, 0, 0, align_c - out_pad.shape[1])).contiguous(
            memory_format=torch.channels_last_3d
        )
    elif output_cl3d is True:
        out_pad = out_pad.contiguous(memory_format=torch.channels_last_3d)
    elif output_cl3d is False:
        out_pad = out_pad.contiguous()
    return out_pad, cache_out


# =============================================================================
# Custom Op 4: worldfoundry::wan_vae_residual_bias_add
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_residual_bias_add", mutates_args=(), device_types="cuda")
def _wan_vae_residual_bias_add_op(
    x: torch.Tensor | None,
    h: torch.Tensor,
    h_bias: torch.Tensor | None,
    x_bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    c = h.shape[1]
    has_x = x is not None
    has_h_bias = h_bias is not None
    has_x_bias = has_x and (x_bias is not None)
    hb_1d = h_bias.reshape(c).contiguous() if h_bias is not None else h
    xb_1d = x_bias.reshape(c).contiguous() if (has_x and x_bias is not None) else h

    use_cl = _is_channels_last_nd(x) if x is not None else _is_channels_last_nd(h)
    if use_cl:
        mem_fmt = torch.channels_last_3d if h.ndim == 5 else torch.channels_last
        h_cl = h if _is_channels_last_nd(h) else h.contiguous(memory_format=mem_fmt)
        x_cl = (x if _is_channels_last_nd(x) else x.contiguous(memory_format=mem_fmt)) if x is not None else h_cl
        out = _empty_with_layout(tuple(h.shape), device=h.device, dtype=h.dtype, channels_last=True)
        total_voxels = h.numel() // c if c > 0 else 0
        if total_voxels == 0 or c == 0:
            return out
        block_c = 256 if c >= 256 else _next_power_of_2(max(c, 16))
        block_v = 16 if block_c <= 256 else 8
        num_warps = 8 if block_v * block_c >= 2048 else 4
        kernels["res_bias_cl"][(_cdiv(total_voxels, block_v), _cdiv(c, block_c))](
            x_cl,
            h_cl,
            hb_1d,
            xb_1d,
            out,
            total_voxels,
            c,
            has_x=has_x,
            has_h_bias=has_h_bias,
            has_x_bias=has_x_bias,
            block_v=block_v,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    h_c = h if h.is_contiguous() else h.contiguous()
    x_c = (x if x.is_contiguous() else x.contiguous()) if x is not None else h_c
    out = torch.empty(h_c.shape, device=h.device, dtype=h.dtype)
    b = h_c.shape[0]
    spatial_size = h_c.numel() // (b * c) if (b > 0 and c > 0) else 0
    if spatial_size == 0 or b == 0 or c == 0:
        return out
    block_s = 256 if spatial_size >= 256 else _next_power_of_2(max(spatial_size, 16))
    kernels["res_bias_cf"][(_cdiv(spatial_size, block_s), b * c)](
        x_c,
        h_c,
        hb_1d,
        xb_1d,
        out,
        spatial_size,
        c,
        has_x=has_x,
        has_h_bias=has_h_bias,
        has_x_bias=has_x_bias,
        block_s=block_s,
        num_warps=4,
    )
    return out


@_wan_vae_residual_bias_add_op.register_fake
def _wan_vae_residual_bias_add_fake(
    x: torch.Tensor | None,
    h: torch.Tensor,
    h_bias: torch.Tensor | None,
    x_bias: torch.Tensor | None,
) -> torch.Tensor:
    del h_bias, x_bias
    use_cl = _is_channels_last_nd(x) if x is not None else _is_channels_last_nd(h)
    return _empty_with_layout(tuple(h.shape), device=h.device, dtype=h.dtype, channels_last=use_cl)


def can_use_wan_vae_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    h_bias: torch.Tensor | None,
    x_bias: torch.Tensor | None = None,
) -> bool:
    if not h.is_cuda or torch.is_grad_enabled() or h.ndim not in (4, 5) or h.dtype not in _SUPPORTED_DTYPES:
        return False
    if h_bias is not None and (h_bias.device != h.device or h_bias.numel() != h.shape[1]):
        return False
    if x is not None:
        if not x.is_cuda or x.device != h.device or x.dtype != h.dtype or x.shape != h.shape:
            return False
        if x_bias is not None and (x_bias.device != h.device or x_bias.numel() != h.shape[1]):
            return False
    return h.shape[1] > 0


def wan_vae_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    h_bias: torch.Tensor | None,
    x_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass coalesced ``(x + x_bias) + (h + h_bias)`` (or ``h + h_bias``) for 4D/5D."""
    if can_use_wan_vae_residual_bias_add(x, h, h_bias, x_bias):
        return _wan_vae_residual_bias_add_op(x, h, h_bias, x_bias)
    b_shape = (1, -1, *([1] * (h.ndim - 2)))
    h_eff = h + h_bias.view(b_shape) if h_bias is not None else h
    if x is None:
        return h_eff
    x_eff = x + x_bias.view(b_shape) if x_bias is not None else x
    return x_eff + h_eff


# =============================================================================
# Custom Op 5: worldfoundry::wan_vae_nearest_upsample_2x
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_nearest_upsample_2x", mutates_args=(), device_types="cuda")
def _wan_vae_nearest_upsample_2x_op(x: torch.Tensor) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c, h, w = x.shape
    is_cl = is_channels_last_2d(x)
    out = _empty_with_layout((b, c, h * 2, w * 2), device=x.device, dtype=x.dtype, channels_last=is_cl)
    if x.numel() == 0:
        return out

    if is_cl:
        block_c = 256 if c >= 256 else _next_power_of_2(max(c, 16))
        block_w_in = 16 if block_c <= 128 else 8
        num_warps = 8 if block_w_in * block_c >= 2048 else 4
        kernels["up2x_nhwc"][(b * h, _cdiv(w, block_w_in), _cdiv(c, block_c))](
            x,
            out,
            h,
            w,
            c,
            block_w_in=block_w_in,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    block_w_in = 256 if w >= 256 else _next_power_of_2(max(w, 16))
    rows_per_cta = 4 if h >= 4 else 1
    kernels["up2x_nchw"][(b * c, _cdiv(h, rows_per_cta), _cdiv(w, block_w_in))](
        x_c,
        out,
        h,
        w,
        block_w_in=block_w_in,
        rows_per_cta=rows_per_cta,
        num_warps=4,
    )
    return out


@_wan_vae_nearest_upsample_2x_op.register_fake
def _wan_vae_nearest_upsample_2x_fake(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return _empty_with_layout(
        (b, c, h * 2, w * 2), device=x.device, dtype=x.dtype, channels_last=is_channels_last_2d(x)
    )


def wan_vae_nearest_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact native BF16/FP16/FP32 2x2 nearest upsample for 4D NCHW/NHWC."""
    if x.is_cuda and not torch.is_grad_enabled() and x.ndim == 4 and x.dtype in _SUPPORTED_DTYPES:
        return _wan_vae_nearest_upsample_2x_op(x)
    return F.interpolate(x.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").type_as(x)


# =============================================================================
# Custom Op 6: worldfoundry::wan_vae_time_interleave_upsample_2x
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_time_interleave_upsample_2x", mutates_args=(), device_types="cuda")
def _wan_vae_time_interleave_upsample_2x_op(
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c2, t, h, w = x.shape
    c_out = c2 // 2
    has_bias = bias is not None
    b_1d = bias.reshape(c2).contiguous() if bias is not None else x

    is_cl = is_channels_last_3d(x)
    out = _empty_with_layout((b * t * 2, c_out, h * 2, w * 2), device=x.device, dtype=x.dtype, channels_last=is_cl)
    if out.numel() == 0:
        return out

    if is_cl:
        block_c = 256 if c_out >= 256 else _next_power_of_2(max(c_out, 16))
        block_w_in = 16 if block_c <= 128 else 8
        num_warps = 8 if block_w_in * block_c >= 2048 else 4
        kernels["time_up_cl"][(b * t * 2 * h, _cdiv(w, block_w_in), _cdiv(c_out, block_c))](
            x,
            b_1d,
            out,
            t,
            h,
            w,
            c_out,
            has_bias=has_bias,
            block_w_in=block_w_in,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    block_w_in = 256 if w >= 256 else _next_power_of_2(max(w, 16))
    rows_per_cta = 4 if h >= 4 else 1
    kernels["time_up_cf"][(b * t * 2 * c_out, _cdiv(h, rows_per_cta), _cdiv(w, block_w_in))](
        x_c,
        b_1d,
        out,
        t,
        h,
        w,
        c_out,
        has_bias=has_bias,
        block_w_in=block_w_in,
        rows_per_cta=rows_per_cta,
        num_warps=4,
    )
    return out


@_wan_vae_time_interleave_upsample_2x_op.register_fake
def _wan_vae_time_interleave_upsample_2x_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    del bias
    b, c2, t, h, w = x.shape
    return _empty_with_layout(
        (b * t * 2, c2 // 2, h * 2, w * 2),
        device=x.device,
        dtype=x.dtype,
        channels_last=is_channels_last_3d(x),
    )


def wan_vae_time_interleave_upsample_2x(
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Super-fused ``(x + bias) -> temporal 2x interleave -> 5D->4D -> 2x2 nearest upsample``."""
    if (
        x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 5
        and x.dtype in _SUPPORTED_DTYPES
        and x.shape[1] >= 2
        and x.shape[1] % 2 == 0
        and (bias is None or (bias.device == x.device and bias.numel() == x.shape[1]))
    ):
        return _wan_vae_time_interleave_upsample_2x_op(x, bias)
    b, c2, t, h, w = x.shape
    c = c2 // 2
    x_eff = x + bias.view(1, -1, 1, 1, 1) if bias is not None else x
    x_eff = x_eff.reshape(b, 2, c, t, h, w)
    x_eff = torch.stack((x_eff[:, 0], x_eff[:, 1]), dim=3).reshape(b, c, t * 2, h, w)
    x_4d = x_eff.permute(0, 2, 1, 3, 4).reshape(b * t * 2, c, h, w)
    return F.interpolate(x_4d.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").type_as(x)


# =============================================================================
# Custom Op 7: worldfoundry::wan_vae_zero_pad_2d_rb
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_zero_pad_2d_rb", mutates_args=(), device_types="cuda")
def _wan_vae_zero_pad_2d_rb_op(x: torch.Tensor) -> torch.Tensor:
    kernels = _load_triton_kernels()

    n, c, h, w = x.shape
    is_cl = is_channels_last_2d(x)
    out = _empty_with_layout((n, c, h + 1, w + 1), device=x.device, dtype=x.dtype, channels_last=is_cl)
    if out.numel() == 0:
        return out

    if is_cl:
        block_c = 256 if c >= 256 else _next_power_of_2(max(c, 16))
        block_w = 16 if block_c <= 128 else 8
        num_warps = 8 if block_w * block_c >= 2048 else 4
        kernels["zpad2d_nhwc"][(n * (h + 1), _cdiv(w + 1, block_w), _cdiv(c, block_c))](
            x,
            out,
            h,
            w,
            c,
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    block_w = 256 if (w + 1) >= 256 else _next_power_of_2(max(w + 1, 16))
    rows_per_cta = 4 if (h + 1) >= 4 else 1
    kernels["zpad2d_nchw"][(n * c, _cdiv(h + 1, rows_per_cta), _cdiv(w + 1, block_w))](
        x_c,
        out,
        h,
        w,
        block_w=block_w,
        rows_per_cta=rows_per_cta,
        num_warps=4,
    )
    return out


@_wan_vae_zero_pad_2d_rb_op.register_fake
def _wan_vae_zero_pad_2d_rb_fake(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    return _empty_with_layout(
        (n, c, h + 1, w + 1), device=x.device, dtype=x.dtype, channels_last=is_channels_last_2d(x)
    )


def wan_vae_zero_pad_2d_rb(x: torch.Tensor) -> torch.Tensor:
    """Single-pass ``ZeroPad2d((0, 1, 0, 1))`` preserving ``NHWC``/``NCHW``."""
    if x.is_cuda and not torch.is_grad_enabled() and x.ndim == 4 and x.dtype in _SUPPORTED_DTYPES:
        return _wan_vae_zero_pad_2d_rb_op(x)
    return F.pad(x, (0, 1, 0, 1))


# =============================================================================
# Custom Op 8: worldfoundry::wan_vae_dup_up_3d_add (and wan_vae_dup_up_3d)
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_dup_up_3d_add", mutates_args=(), device_types="cuda")
def _wan_vae_dup_up_3d_add_op(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_in, t_in, h_in, w_in = x.shape
    repeats = (out_channels * factor_t * factor_s * factor_s) // c_in
    t_trim = (factor_t - 1) if first_chunk else 0
    t_out = t_in * factor_t - t_trim
    h_out = h_in * factor_s
    w_out = w_in * factor_s

    has_h = h is not None
    has_h_bias = has_h and (h_bias is not None)
    hb_1d = h_bias.reshape(out_channels).contiguous() if (has_h and h_bias is not None) else x

    use_cl = is_channels_last_3d(x) or (h is not None and _is_channels_last_nd(h))
    out = _empty_with_layout(
        (b, out_channels, t_out, h_out, w_out), device=x.device, dtype=x.dtype, channels_last=use_cl
    )
    if out.numel() == 0:
        return out

    if use_cl:
        x_cl = x if is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d)
        if h is not None:
            h_mem_fmt = torch.channels_last_3d if h.ndim == 5 else torch.channels_last
            h_cl = h if _is_channels_last_nd(h) else h.contiguous(memory_format=h_mem_fmt)
        else:
            h_cl = out
        block_c = 256 if out_channels >= 256 else _next_power_of_2(max(out_channels, 16))
        if factor_s == 2:
            block_w_in = 8 if block_c >= 256 else 16
            num_warps = 8 if block_w_in * block_c >= 2048 else 4
            kernels["dup_up_cl_fs2"][(b * t_out * h_out, _cdiv(w_in, block_w_in), _cdiv(out_channels, block_c))](
                h_cl,
                x_cl,
                hb_1d,
                out,
                t_in,
                h_in,
                w_in,
                t_out,
                h_out,
                c_in,
                out_channels,
                repeats=repeats,
                factor_t=factor_t,
                t_trim=t_trim,
                has_h=has_h,
                has_h_bias=has_h_bias,
                block_w_in=block_w_in,
                block_c=block_c,
                num_warps=num_warps,
            )
            return out

        block_w = 8 if block_c >= 256 else 16
        num_warps = 8 if block_w * block_c >= 2048 else 4
        kernels["dup_up_cl_gen"][(b * t_out * h_out, _cdiv(w_out, block_w), _cdiv(out_channels, block_c))](
            h_cl,
            x_cl,
            hb_1d,
            out,
            t_in,
            h_in,
            w_in,
            t_out,
            h_out,
            w_out,
            c_in,
            out_channels,
            repeats=repeats,
            factor_t=factor_t,
            factor_s=factor_s,
            t_trim=t_trim,
            has_h=has_h,
            has_h_bias=has_h_bias,
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    if h is not None:
        h_c = h if h.is_contiguous() else h.contiguous()
        if h_c.ndim == 5:
            sh_b, sh_c, sh_d, sh_h = h_c.stride(0), h_c.stride(1), h_c.stride(2), h_c.stride(3)
        else:
            # 4D NCHW (b*t_out, c_out, h_out, w_out)
            sh_d = h_c.stride(0)
            sh_b = t_out * sh_d
            sh_c = h_c.stride(1)
            sh_h = h_c.stride(2)
    else:
        h_c = out
        sh_b = sh_c = sh_d = sh_h = 0

    block_w = 256 if w_out >= 256 else _next_power_of_2(max(w_out, 16))
    rows_per_cta = 4 if h_out >= 4 else 1
    kernels["dup_up_cf"][(b * out_channels * t_out, _cdiv(h_out, rows_per_cta), _cdiv(w_out, block_w))](
        h_c,
        x_c,
        hb_1d,
        out,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        c_in,
        out_channels,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        repeats=repeats,
        factor_t=factor_t,
        factor_s=factor_s,
        t_trim=t_trim,
        has_h=has_h,
        has_h_bias=has_h_bias,
        block_w=block_w,
        rows_per_cta=rows_per_cta,
        num_warps=4,
    )
    return out


@_wan_vae_dup_up_3d_add_op.register_fake
def _wan_vae_dup_up_3d_add_fake(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    del h_bias
    b, _, t_in, h_in, w_in = x.shape
    t_trim = (factor_t - 1) if first_chunk else 0
    use_cl = is_channels_last_3d(x) or (h is not None and _is_channels_last_nd(h))
    return _empty_with_layout(
        (b, out_channels, t_in * factor_t - t_trim, h_in * factor_s, w_in * factor_s),
        device=x.device,
        dtype=x.dtype,
        channels_last=use_cl,
    )


def can_use_wan_vae_dup_up_3d_add(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool = False,
) -> bool:
    if not x.is_cuda or torch.is_grad_enabled() or x.ndim != 5 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    b, c_in, t_in, h_in, w_in = x.shape
    if c_in <= 0 or out_channels <= 0 or factor_t <= 0 or factor_s <= 0:
        return False
    if (out_channels * factor_t * factor_s * factor_s) % c_in != 0:
        return False
    t_out = t_in * factor_t - ((factor_t - 1) if first_chunk else 0)
    h_out = h_in * factor_s
    w_out = w_in * factor_s
    if t_out < 0:
        return False
    if h is not None:
        if not h.is_cuda or h.device != x.device or h.dtype != x.dtype:
            return False
        if h.ndim == 5:
            if h.shape != (b, out_channels, t_out, h_out, w_out):
                return False
        elif h.ndim == 4:
            if h.shape != (b * t_out, out_channels, h_out, w_out):
                return False
        else:
            return False
        if h_bias is not None and (h_bias.device != x.device or h_bias.numel() != out_channels):
            return False
    return True


def _dup_up_3d_torch(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    repeats = (out_channels * factor_t * factor_s * factor_s) // x.shape[1]
    y = x.repeat_interleave(repeats, dim=1)
    y = y.view(x.size(0), out_channels, factor_t, factor_s, factor_s, x.size(2), x.size(3), x.size(4))
    y = y.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    y = y.view(x.size(0), out_channels, x.size(2) * factor_t, x.size(3) * factor_s, x.size(4) * factor_s)
    if first_chunk:
        y = y[:, :, factor_t - 1 :, :, :]
    return y


def wan_vae_dup_up_3d_add(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None = None,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Single-pass ``(h + h_bias) + DupUp3D(x)`` preserving ``channels_last_3d`` / ``NCDHW``."""
    if can_use_wan_vae_dup_up_3d_add(
        h,
        x,
        h_bias,
        out_channels=int(out_channels),
        factor_t=int(factor_t),
        factor_s=int(factor_s),
        first_chunk=bool(first_chunk),
    ):
        return _wan_vae_dup_up_3d_add_op(
            h, x, h_bias, int(out_channels), int(factor_t), int(factor_s), bool(first_chunk)
        )
    sc = _dup_up_3d_torch(
        x, out_channels=int(out_channels), factor_t=int(factor_t), factor_s=int(factor_s), first_chunk=bool(first_chunk)
    )
    if h is None:
        return sc
    h_5d = (
        h.reshape(x.shape[0], sc.shape[2], out_channels, sc.shape[3], sc.shape[4]).permute(0, 2, 1, 3, 4)
        if h.ndim == 4
        else h
    )
    if h_bias is not None:
        h_5d = h_5d + h_bias.view(1, -1, 1, 1, 1)
    return h_5d + sc


def wan_vae_dup_up_3d(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Bit-exact single-pass ``DupUp3D`` shortcut for ``WanVideoVAE38`` (CL3D & NCDHW)."""
    return wan_vae_dup_up_3d_add(
        None,
        x,
        None,
        out_channels=int(out_channels),
        factor_t=int(factor_t),
        factor_s=int(factor_s),
        first_chunk=bool(first_chunk),
    )


# =============================================================================
# Custom Op 9: worldfoundry::wan_vae_avg_down_3d_add (and wan_vae_avg_down_3d)
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_avg_down_3d_add", mutates_args=(), device_types="cuda")
def _wan_vae_avg_down_3d_add_op(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_in, t_in, h_in, w_in = x.shape
    pad_t = (factor_t - t_in % factor_t) % factor_t
    t_out = (t_in + pad_t) // factor_t
    h_out = h_in // factor_s
    w_out = w_in // factor_s
    group_size = (c_in * factor_t * factor_s * factor_s) // out_channels

    has_h = h is not None
    has_h_bias = has_h and (h_bias is not None)
    hb_1d = h_bias.reshape(out_channels).contiguous() if (has_h and h_bias is not None) else x

    use_cl = is_channels_last_3d(x) or (h is not None and _is_channels_last_nd(h))
    out = _empty_with_layout(
        (b, out_channels, t_out, h_out, w_out), device=x.device, dtype=x.dtype, channels_last=use_cl
    )
    if out.numel() == 0:
        return out

    if use_cl:
        x_cl = x if is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d)
        if h is not None:
            h_mem_fmt = torch.channels_last_3d if h.ndim == 5 else torch.channels_last
            h_cl = h if _is_channels_last_nd(h) else h.contiguous(memory_format=h_mem_fmt)
        else:
            h_cl = out
        if factor_s == 2 and group_size == 4 and factor_t in (1, 2) and out_channels == c_in * factor_t:
            block_c_in = 128 if c_in >= 128 else _next_power_of_2(max(c_in, 16))
            block_w = 8 if (block_c_in * factor_t) >= 256 else 16
            num_warps = 8 if block_w * block_c_in * factor_t >= 2048 else 4
            kernels["avg_dn_cl_fast"][(b * t_out * h_out, _cdiv(w_out, block_w), _cdiv(c_in, block_c_in))](
                h_cl,
                x_cl,
                hb_1d,
                out,
                t_in,
                h_in,
                w_in,
                t_out,
                h_out,
                w_out,
                c_in,
                out_channels,
                pad_t,
                factor_t=factor_t,
                has_h=has_h,
                has_h_bias=has_h_bias,
                block_w=block_w,
                block_c_in=block_c_in,
                num_warps=num_warps,
            )
            return out

        block_c = 128 if out_channels >= 128 else _next_power_of_2(max(out_channels, 16))
        block_w = 16 if block_c <= 128 else 8
        num_warps = 4
        kernels["avg_dn_cl_gen"][(b * t_out * h_out, _cdiv(w_out, block_w), _cdiv(out_channels, block_c))](
            h_cl,
            x_cl,
            hb_1d,
            out,
            t_in,
            h_in,
            w_in,
            t_out,
            h_out,
            w_out,
            c_in,
            out_channels,
            pad_t,
            factor_t=factor_t,
            factor_s=factor_s,
            group_size=group_size,
            has_h=has_h,
            has_h_bias=has_h_bias,
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    if h is not None:
        h_c = h if h.is_contiguous() else h.contiguous()
        if h_c.ndim == 5:
            sh_b, sh_c, sh_d, sh_h = h_c.stride(0), h_c.stride(1), h_c.stride(2), h_c.stride(3)
        else:
            sh_d = h_c.stride(0)
            sh_b = t_out * sh_d
            sh_c = h_c.stride(1)
            sh_h = h_c.stride(2)
    else:
        h_c = out
        sh_b = sh_c = sh_d = sh_h = 0

    block_w = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
    rows_per_cta = 4 if h_out >= 4 else 1
    kernels["avg_dn_cf"][(b * out_channels * t_out, _cdiv(h_out, rows_per_cta), _cdiv(w_out, block_w))](
        h_c,
        x_c,
        hb_1d,
        out,
        t_in,
        h_in,
        w_in,
        t_out,
        h_out,
        w_out,
        c_in,
        out_channels,
        pad_t,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        factor_t=factor_t,
        factor_s=factor_s,
        group_size=group_size,
        has_h=has_h,
        has_h_bias=has_h_bias,
        block_w=block_w,
        rows_per_cta=rows_per_cta,
        num_warps=4,
    )
    return out


@_wan_vae_avg_down_3d_add_op.register_fake
def _wan_vae_avg_down_3d_add_fake(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> torch.Tensor:
    del h_bias
    b, _, t_in, h_in, w_in = x.shape
    pad_t = (factor_t - t_in % factor_t) % factor_t
    use_cl = is_channels_last_3d(x) or (h is not None and _is_channels_last_nd(h))
    return _empty_with_layout(
        (b, out_channels, (t_in + pad_t) // factor_t, h_in // factor_s, w_in // factor_s),
        device=x.device,
        dtype=x.dtype,
        channels_last=use_cl,
    )


def can_use_wan_vae_avg_down_3d_add(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> bool:
    if not x.is_cuda or torch.is_grad_enabled() or x.ndim != 5 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    b, c_in, t_in, h_in, w_in = x.shape
    if c_in <= 0 or out_channels <= 0 or factor_t <= 0 or factor_s <= 0:
        return False
    if h_in % factor_s != 0 or w_in % factor_s != 0:
        return False
    factor = factor_t * factor_s * factor_s
    if (c_in * factor) % out_channels != 0:
        return False
    group_size = (c_in * factor) // out_channels
    if not (0 < group_size <= 32):
        return False
    pad_t = (factor_t - t_in % factor_t) % factor_t
    t_out = (t_in + pad_t) // factor_t
    h_out = h_in // factor_s
    w_out = w_in // factor_s
    if h is not None:
        if not h.is_cuda or h.device != x.device or h.dtype != x.dtype:
            return False
        if h.ndim == 5:
            if h.shape != (b, out_channels, t_out, h_out, w_out):
                return False
        elif h.ndim == 4:
            if h.shape != (b * t_out, out_channels, h_out, w_out):
                return False
        else:
            return False
        if h_bias is not None and (h_bias.device != x.device or h_bias.numel() != out_channels):
            return False
    return True


def _avg_down_3d_torch(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> torch.Tensor:
    pad_t = (factor_t - x.shape[2] % factor_t) % factor_t
    y = F.pad(x, (0, 0, 0, 0, pad_t, 0)) if pad_t else x
    b, c, t, h, w = y.shape
    factor = factor_t * factor_s * factor_s
    group_size = (c * factor) // out_channels
    y = y.view(b, c, t // factor_t, factor_t, h // factor_s, factor_s, w // factor_s, factor_s)
    y = y.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    y = y.view(b, out_channels, group_size, t // factor_t, h // factor_s, w // factor_s)
    return y.mean(dim=2)


def wan_vae_avg_down_3d_add(
    h: torch.Tensor | None,
    x: torch.Tensor,
    h_bias: torch.Tensor | None = None,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> torch.Tensor:
    """Single-pass ``(h + h_bias) + AvgDown3D(x)`` preserving ``channels_last_3d`` / ``NCDHW``."""
    if can_use_wan_vae_avg_down_3d_add(
        h,
        x,
        h_bias,
        out_channels=int(out_channels),
        factor_t=int(factor_t),
        factor_s=int(factor_s),
    ):
        return _wan_vae_avg_down_3d_add_op(h, x, h_bias, int(out_channels), int(factor_t), int(factor_s))
    sc = _avg_down_3d_torch(x, out_channels=int(out_channels), factor_t=int(factor_t), factor_s=int(factor_s))
    if h is None:
        return sc
    h_5d = (
        h.reshape(x.shape[0], sc.shape[2], out_channels, sc.shape[3], sc.shape[4]).permute(0, 2, 1, 3, 4)
        if h.ndim == 4
        else h
    )
    if h_bias is not None:
        h_5d = h_5d + h_bias.view(1, -1, 1, 1, 1)
    return h_5d + sc


def wan_vae_avg_down_3d(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> torch.Tensor:
    """Single-pass ``AvgDown3D`` shortcut for ``WanVideoVAE38`` (CL3D & NCDHW)."""
    return wan_vae_avg_down_3d_add(
        None,
        x,
        None,
        out_channels=int(out_channels),
        factor_t=int(factor_t),
        factor_s=int(factor_s),
    )


# =============================================================================
# Custom Op 10: worldfoundry::wan_vae_patchify_2x & wan_vae_unpatchify_2x
# =============================================================================


@torch.library.custom_op("worldfoundry::wan_vae_patchify_2x", mutates_args=(), device_types="cuda")
def _wan_vae_patchify_2x_op(x: torch.Tensor, output_cl3d: int) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_in, t_dim, h_in, w_in = x.shape
    h_out = h_in // 2
    w_out = w_in // 2
    out_is_cl = (output_cl3d == 1) or (output_cl3d == -1 and is_channels_last_3d(x))
    out = _empty_with_layout(
        (b, c_in * 4, t_dim, h_out, w_out), device=x.device, dtype=x.dtype, channels_last=out_is_cl
    )
    if out.numel() == 0:
        return out
    block_w = 256 if w_out >= 256 else _next_power_of_2(max(w_out, 16))
    kernels["patchify_2x"][(b * t_dim, h_out, _cdiv(w_out, block_w))](
        x,
        out,
        c_in,
        t_dim,
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
        block_w=block_w,
        num_warps=4,
    )
    return out


@_wan_vae_patchify_2x_op.register_fake
def _wan_vae_patchify_2x_fake(x: torch.Tensor, output_cl3d: int) -> torch.Tensor:
    b, c_in, t_dim, h_in, w_in = x.shape
    out_is_cl = (output_cl3d == 1) or (output_cl3d == -1 and is_channels_last_3d(x))
    return _empty_with_layout(
        (b, c_in * 4, t_dim, h_in // 2, w_in // 2),
        device=x.device,
        dtype=x.dtype,
        channels_last=out_is_cl,
    )


def wan_vae_patchify_2x(x: torch.Tensor, *, output_cl3d: bool | None = None) -> torch.Tensor:
    """Single-pass ``b c f (h 2) (w 2) -> b (c r q) f h w`` preserving or setting CL3D."""
    out_flag = -1 if output_cl3d is None else (1 if output_cl3d else 0)
    if (
        x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 5
        and x.dtype in _SUPPORTED_DTYPES
        and x.shape[1] <= 32
        and x.shape[3] % 2 == 0
        and x.shape[4] % 2 == 0
    ):
        return _wan_vae_patchify_2x_op(x, out_flag)
    b, c, f, h, w = x.shape
    y = x.view(b, c, f, h // 2, 2, w // 2, 2).permute(0, 1, 6, 4, 2, 3, 5).reshape(b, c * 4, f, h // 2, w // 2)
    if output_cl3d is True:
        return y.contiguous(memory_format=torch.channels_last_3d)
    if output_cl3d is False:
        return y.contiguous()
    return y.contiguous(memory_format=torch.channels_last_3d) if is_channels_last_3d(x) else y.contiguous()


@torch.library.custom_op("worldfoundry::wan_vae_unpatchify_2x", mutates_args=(), device_types="cuda")
def _wan_vae_unpatchify_2x_op(
    x: torch.Tensor,
    pre_bias: torch.Tensor | None,
    output_cl3d: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_in, t_dim, h_in, w_in = x.shape
    c_out = c_in // 4
    has_pre_bias = pre_bias is not None
    pb_1d = pre_bias.reshape(c_in).contiguous() if pre_bias is not None else x
    out_is_cl = (output_cl3d == 1) or (output_cl3d == -1 and is_channels_last_3d(x))
    out = _empty_with_layout(
        (b, c_out, t_dim, h_in * 2, w_in * 2),
        device=x.device,
        dtype=x.dtype,
        channels_last=out_is_cl,
    )
    if out.numel() == 0:
        return out
    block_w_in = 256 if w_in >= 256 else _next_power_of_2(max(w_in, 16))
    kernels["unpatchify_2x"][(b * t_dim, h_in, _cdiv(w_in, block_w_in))](
        x,
        pb_1d,
        out,
        c_out,
        t_dim,
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
        has_pre_bias=has_pre_bias,
        block_w_in=block_w_in,
        num_warps=4,
    )
    return out


@_wan_vae_unpatchify_2x_op.register_fake
def _wan_vae_unpatchify_2x_fake(
    x: torch.Tensor,
    pre_bias: torch.Tensor | None,
    output_cl3d: int,
) -> torch.Tensor:
    del pre_bias
    b, c_in, t_dim, h_in, w_in = x.shape
    out_is_cl = (output_cl3d == 1) or (output_cl3d == -1 and is_channels_last_3d(x))
    return _empty_with_layout(
        (b, c_in // 4, t_dim, h_in * 2, w_in * 2),
        device=x.device,
        dtype=x.dtype,
        channels_last=out_is_cl,
    )


def wan_vae_unpatchify_2x(
    x: torch.Tensor,
    *,
    pre_bias: torch.Tensor | None = None,
    output_cl3d: bool | None = None,
) -> torch.Tensor:
    """Single-pass ``b (c r q) f h w -> b c f (h 2) (w 2)`` (+ optional pre_bias)."""
    out_flag = -1 if output_cl3d is None else (1 if output_cl3d else 0)
    if (
        x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 5
        and x.dtype in _SUPPORTED_DTYPES
        and x.shape[1] % 4 == 0
        and 0 < x.shape[1] <= 128
        and (pre_bias is None or (pre_bias.device == x.device and pre_bias.numel() == x.shape[1]))
    ):
        return _wan_vae_unpatchify_2x_op(x, pre_bias, out_flag)
    x_eff = x + pre_bias.view(1, -1, 1, 1, 1) if pre_bias is not None else x
    b, c_in, f, h, w = x_eff.shape
    y = x_eff.view(b, c_in // 4, 2, 2, f, h, w).permute(0, 1, 4, 5, 3, 6, 2).reshape(b, c_in // 4, f, h * 2, w * 2)
    if output_cl3d is True:
        return y.contiguous(memory_format=torch.channels_last_3d)
    if output_cl3d is False:
        return y.contiguous()
    return y.contiguous(memory_format=torch.channels_last_3d) if is_channels_last_3d(x) else y.contiguous()


__all__ = [
    "can_use_wan_vae_avg_down_3d_add",
    "can_use_wan_vae_causal_pad_cache",
    "can_use_wan_vae_dup_up_3d_add",
    "can_use_wan_vae_residual_bias_add",
    "can_use_wan_vae_rms_norm",
    "is_channels_last_2d",
    "is_channels_last_3d",
    "wan_vae_avg_down_3d",
    "wan_vae_avg_down_3d_add",
    "wan_vae_causal_pad_cache",
    "wan_vae_dup_up_3d",
    "wan_vae_dup_up_3d_add",
    "wan_vae_nearest_upsample_2x",
    "wan_vae_patchify_2x",
    "wan_vae_residual_bias_add",
    "wan_vae_rms_norm",
    "wan_vae_rms_norm_silu_causal_pad",
    "wan_vae_time_interleave_upsample_2x",
    "wan_vae_unpatchify_2x",
    "wan_vae_zero_pad_2d_rb",
]
