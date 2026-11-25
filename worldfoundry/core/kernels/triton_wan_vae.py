# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for WanVideoVAE / WanVideoVAE38.

Shared by 8 foundation video and streaming world models (Wan2.1-14B,
Wan2.2-TI2V-5B, Wan2.2-T2V-A14B, Matrix-Game-2.0, LongLive-1.3B,
Self-Forcing 1.3B, LingBot-Video, and SANA-Video).

Replaces multi-pass eager PyTorch 4D/5D ops with single-pass coalesced Triton
kernels supporting both ``NCDHW``/``NCHW`` and ``channels_last_3d``/``channels_last``
memory formats:

- :func:`wan_vae_rms_norm` — exact ``F.normalize(x, dim=1) * sqrt(C) * gamma + bias``
  with optional fused ``SiLU`` activation (cuts 6-7 HBM passes to 1).
- :func:`wan_vae_rms_norm_silu_causal_pad` — super-fused ``RMS_norm + SiLU`` +
  last-2-frame feature-cache snapshot + ``cat([cache, y])`` + zero-padding
  directly into the ``CausalConv3d`` input buffer (cuts 9 HBM passes to 1).
- :func:`wan_vae_nearest_upsample_2x` — bit-exact native BF16/FP16/FP32 2x2
  nearest-neighbor upsampling eliminating the legacy ``x.float()`` roundtrip.
- :func:`wan_vae_dup_up_3d` — bit-exact single-pass ``DupUp3D`` channel repeat
  + 3D pixel-shuffle shortcut for ``WanVideoVAE38``.

All CUDA ops register ``torch.library.custom_op`` + ``register_fake`` for
``torch.compile`` and CUDA Graph capture safety. Importing this module does
not import Triton or initialize CUDA.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _is_channels_last_nd(x: torch.Tensor) -> bool:
    """Return True when channel dim (dim=1) is stride-1 contiguous."""
    if x.ndim == 5:
        return x.is_contiguous(memory_format=torch.channels_last_3d)
    if x.ndim == 4:
        return x.is_contiguous(memory_format=torch.channels_last)
    return False


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Triton kernels on first eligible CUDA call."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="wan-vae-triton")

    import triton
    import triton.language as tl

    @triton.jit
    def _wan_rms_norm_channels_last_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        out_ptr,
        total_pixels,
        scale,
        channels: tl.constexpr,
        block_p: tl.constexpr,
        block_c: tl.constexpr,
        has_bias: tl.constexpr,
        apply_silu: tl.constexpr,
    ):
        """One CTA processes ``block_p`` contiguous pixels along stride-1 channels."""
        pid = tl.program_id(0).to(tl.int64)
        p_offs = pid * block_p + tl.arange(0, block_p).to(tl.int64)
        p_mask = p_offs < total_pixels
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < channels
        mask = p_mask[:, None] & c_mask[None, :]

        ptrs = p_offs[:, None] * channels + c_offs[None, :]
        x_in = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)

        # Match F.normalize(x, p=2, dim=1, eps=1e-12) exactly.
        norm = tl.maximum(tl.sqrt(tl.sum(x_f * x_f, axis=1)), 1e-12)
        inv_norm = (scale / norm)[:, None]

        gamma = tl.load(gamma_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
        y = x_f * inv_norm * gamma
        if has_bias:
            bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y + bias
        if apply_silu:
            y_cast = y.to(x_in.dtype).to(tl.float32)
            y = y_cast * tl.sigmoid(y_cast)
        tl.store(out_ptr + ptrs, y.to(x_in.dtype), mask=mask)

    @triton.jit
    def _wan_rms_norm_channels_first_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        out_ptr,
        batch_size,
        spatial_size,
        scale,
        channels: tl.constexpr,
        block_s: tl.constexpr,
        has_bias: tl.constexpr,
        apply_silu: tl.constexpr,
    ):
        """One CTA processes ``block_s`` contiguous spatial elements across ``channels``."""
        s_blk = tl.program_id(0).to(tl.int64)
        b_idx = tl.program_id(1).to(tl.int64)
        spatial_i64 = tl.cast(spatial_size, tl.int64)
        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64
        batch_base = b_idx * channels * spatial_i64

        sq_sum = tl.zeros((block_s,), dtype=tl.float32)
        for c in range(channels):
            val = tl.load(x_ptr + batch_base + c * spatial_i64 + s_offs, mask=s_mask, other=0.0).to(tl.float32)
            sq_sum += val * val

        inv_norm = scale / tl.maximum(tl.sqrt(sq_sum), 1e-12)

        for c in range(channels):
            col_ptr = batch_base + c * spatial_i64 + s_offs
            x_in = tl.load(x_ptr + col_ptr, mask=s_mask, other=0.0)
            g = tl.load(gamma_ptr + c).to(tl.float32)
            y = x_in.to(tl.float32) * inv_norm * g
            if has_bias:
                b_val = tl.load(bias_ptr + c).to(tl.float32)
                y = y + b_val
            if apply_silu:
                y_cast = y.to(x_in.dtype).to(tl.float32)
                y = y_cast * tl.sigmoid(y_cast)
            tl.store(out_ptr + col_ptr, y.to(x_in.dtype), mask=s_mask)

    @triton.jit
    def _wan_rms_norm_silu_causal_pad_cf_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
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
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
    ):
        """Super-fused NCDHW ``RMS_norm + SiLU + cache_snapshot + cat + F.pad``.

        Grid: ``(cdiv(w_out, block_w), h_out, b * t_out)``.
        Writes every output voxel exactly once (zeros on spatial/temporal pad
        borders, ``cache_in`` on cached prefix frames, ``SiLU(RMS_norm(x))`` on
        active frames) and simultaneously populates ``cache_out`` for the last
        ``t_cache_out`` active frames.
        """
        w_blk = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        bt = tl.program_id(2).to(tl.int64)
        b = bt // t_out
        to = bt % t_out

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out

        in_spatial_h = (ho >= ph) & (ho < ph + h_in)
        in_spatial_w = (wo_offs >= pw) & (wo_offs < pw + w_in)
        in_spatial = in_spatial_h & in_spatial_w

        out_hw = tl.cast(h_out, tl.int64) * w_out
        out_thw = tl.cast(t_out, tl.int64) * out_hw
        out_base = b * channels * out_thw + to * out_hw + ho * w_out + wo_offs

        # Region 1: Left zero temporal padding.
        if to < t_pad_left:
            zeros = tl.zeros((block_w,), dtype=tl.float32)
            for c in range(channels):
                tl.store(out_pad_ptr + out_base + c * out_thw, zeros, mask=wo_mask)
            return

        # Region 2: Cached prefix frames from cache_in.
        if has_cache_in and (to < t_pad_left + t_cache_in):
            tc = to - t_pad_left
            in_hw = tl.cast(h_in, tl.int64) * w_in
            c_thw = tl.cast(t_cache_in, tl.int64) * in_hw
            hi = ho - ph
            wi = wo_offs - pw
            c_base = b * channels * c_thw + tc * in_hw + hi * w_in + wi
            valid_c = wo_mask & in_spatial

            need_cache_copy = copy_prev_cache_to_cache_out and (tc == t_cache_in - 1)
            co_thw = tl.cast(t_cache_out, tl.int64) * in_hw
            co_base = b * channels * co_thw + hi * w_in + wi
            for c in range(channels):
                val = tl.load(cache_in_ptr + c_base + c * c_thw, mask=valid_c, other=0.0)
                out_v = tl.where(in_spatial, val, 0.0)
                tl.store(out_pad_ptr + out_base + c * out_thw, out_v, mask=wo_mask)
                if need_cache_copy:
                    tl.store(cache_out_ptr + co_base + c * co_thw, val, mask=valid_c)
            return

        # Region 3: Active frames -> RMS_norm + SiLU (+ optional cache_out write).
        ti = to - t_pad_left - t_cache_in
        if not in_spatial_h:
            zeros = tl.zeros((block_w,), dtype=tl.float32)
            for c in range(channels):
                tl.store(out_pad_ptr + out_base + c * out_thw, zeros, mask=wo_mask)
            return

        in_hw = tl.cast(h_in, tl.int64) * w_in
        in_thw = tl.cast(t_in, tl.int64) * in_hw
        hi = ho - ph
        wi = wo_offs - pw
        in_base = b * channels * in_thw + ti * in_hw + hi * w_in + wi
        valid = wo_mask & in_spatial_w

        sq_sum = tl.zeros((block_w,), dtype=tl.float32)
        for c in range(channels):
            v = tl.load(x_ptr + in_base + c * in_thw, mask=valid, other=0.0).to(tl.float32)
            sq_sum += v * v
        inv_norm = scale / tl.maximum(tl.sqrt(sq_sum), 1e-12)

        save_to_cache = write_cache_out and (ti >= t_in - (t_cache_out - (1 if copy_prev_cache_to_cache_out else 0)))
        co_slot = ti - (t_in - t_cache_out)
        co_thw = tl.cast(t_cache_out, tl.int64) * in_hw
        co_base = b * channels * co_thw + co_slot * in_hw + hi * w_in + wi

        for c in range(channels):
            x_in = tl.load(x_ptr + in_base + c * in_thw, mask=valid, other=0.0)
            g = tl.load(gamma_ptr + c).to(tl.float32)
            y = x_in.to(tl.float32) * inv_norm * g
            if has_bias:
                y = y + tl.load(bias_ptr + c).to(tl.float32)
            y_cast = y.to(x_in.dtype).to(tl.float32)
            y_silu = (y_cast * tl.sigmoid(y_cast)).to(x_in.dtype)
            out_v = tl.where(in_spatial_w, y_silu, 0.0)
            tl.store(out_pad_ptr + out_base + c * out_thw, out_v, mask=wo_mask)
            if save_to_cache:
                tl.store(cache_out_ptr + co_base + c * co_thw, y_silu, mask=valid)

    @triton.jit
    def _wan_rms_norm_silu_causal_pad_cl_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
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
        has_cache_in: tl.constexpr,
        write_cache_out: tl.constexpr,
        copy_prev_cache_to_cache_out: tl.constexpr,
    ):
        """Super-fused NDHWC (``channels_last_3d``) ``RMS_norm + SiLU + cache + cat + pad``."""
        w_blk = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        bt = tl.program_id(2).to(tl.int64)
        b = bt // t_out
        to = bt % t_out

        wo_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        wo_mask = wo_offs < w_out
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < channels
        out_mask = wo_mask[:, None] & c_mask[None, :]

        out_row_base = ((b * t_out + to) * h_out + ho) * w_out + wo_offs
        out_ptrs = out_row_base[:, None] * channels + c_offs[None, :]

        in_spatial_h = (ho >= ph) & (ho < ph + h_in)
        in_spatial_w = (wo_offs >= pw) & (wo_offs < pw + w_in)

        if (to < t_pad_left) or (not in_spatial_h):
            tl.store(out_pad_ptr + out_ptrs, tl.zeros((block_w, block_c), dtype=tl.float32), mask=out_mask)
            return

        hi = ho - ph
        wi = wo_offs - pw
        valid_2d = (wo_mask & in_spatial_w)[:, None] & c_mask[None, :]

        if has_cache_in and (to < t_pad_left + t_cache_in):
            tc = to - t_pad_left
            c_row = ((b * t_cache_in + tc) * h_in + hi) * w_in + wi
            c_ptrs = c_row[:, None] * channels + c_offs[None, :]
            val = tl.load(cache_in_ptr + c_ptrs, mask=valid_2d, other=0.0)
            tl.store(out_pad_ptr + out_ptrs, tl.where(in_spatial_w[:, None], val, 0.0), mask=out_mask)
            if copy_prev_cache_to_cache_out and (tc == t_cache_in - 1):
                co_row = ((b * t_cache_out + 0) * h_in + hi) * w_in + wi
                tl.store(cache_out_ptr + co_row[:, None] * channels + c_offs[None, :], val, mask=valid_2d)
            return

        ti = to - t_pad_left - t_cache_in
        in_row = ((b * t_in + ti) * h_in + hi) * w_in + wi
        in_ptrs = in_row[:, None] * channels + c_offs[None, :]
        x_in = tl.load(x_ptr + in_ptrs, mask=valid_2d, other=0.0)
        x_f = x_in.to(tl.float32)

        norm = tl.maximum(tl.sqrt(tl.sum(x_f * x_f, axis=1)), 1e-12)
        inv_norm = (scale / norm)[:, None]
        gamma = tl.load(gamma_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
        y = x_f * inv_norm * gamma
        if has_bias:
            bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y + bias
        y_cast = y.to(x_in.dtype).to(tl.float32)
        y_silu = (y_cast * tl.sigmoid(y_cast)).to(x_in.dtype)

        tl.store(out_pad_ptr + out_ptrs, tl.where(in_spatial_w[:, None], y_silu, 0.0), mask=out_mask)

        if write_cache_out and (ti >= t_in - (t_cache_out - (1 if copy_prev_cache_to_cache_out else 0))):
            co_slot = ti - (t_in - t_cache_out)
            co_row = ((b * t_cache_out + co_slot) * h_in + hi) * w_in + wi
            tl.store(cache_out_ptr + co_row[:, None] * channels + c_offs[None, :], y_silu, mask=valid_2d)

    @triton.jit
    def _wan_nearest_upsample_2x_nchw_kernel(
        x_ptr,
        out_ptr,
        total_planes,
        h_in,
        w_in,
        block_w_in: tl.constexpr,
        rows_per_cta: tl.constexpr,
    ):
        """Bit-exact NCHW 2x2 nearest upsample; coalesced load + doubled join store."""
        w_blk = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        plane = tl.program_id(2).to(tl.int64)

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in)
        w_mask = wi_offs < w_in
        w_out = w_in * 2
        wo_offs = w_blk * (block_w_in * 2) + tl.arange(0, block_w_in * 2)
        wo_mask = wo_offs < w_out

        h_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            hi = h_base + r
            if hi < h_in:
                vals = tl.load(x_ptr + (plane * h_in + hi) * w_in + wi_offs, mask=w_mask)
                doubled = tl.reshape(tl.join(vals, vals), (block_w_in * 2,))
                out_base0 = (plane * (h_in * 2) + hi * 2) * w_out
                tl.store(out_ptr + out_base0 + wo_offs, doubled, mask=wo_mask)
                tl.store(out_ptr + out_base0 + w_out + wo_offs, doubled, mask=wo_mask)

    @triton.jit
    def _wan_nearest_upsample_2x_nhwc_kernel(
        x_ptr,
        out_ptr,
        batch_size,
        h_in,
        w_in,
        channels: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Bit-exact NHWC 2x2 nearest upsample along stride-1 channels."""
        w_blk = tl.program_id(0).to(tl.int64)
        hi = tl.program_id(1).to(tl.int64)
        b = tl.program_id(2).to(tl.int64)

        w_out = w_in * 2
        wo_offs = w_blk * block_w + tl.arange(0, block_w)
        w_mask = wo_offs < w_out
        c_offs = tl.arange(0, block_c)
        c_mask = c_offs < channels
        mask = w_mask[:, None] & c_mask[None, :]
        wi_offs = wo_offs >> 1

        in_rows = (b * h_in + hi) * w_in + wi_offs
        vals = tl.load(x_ptr + in_rows[:, None] * channels + c_offs[None, :], mask=mask)

        out_rows0 = (b * (h_in * 2) + hi * 2) * w_out + wo_offs
        out_rows1 = out_rows0 + w_out
        tl.store(out_ptr + out_rows0[:, None] * channels + c_offs[None, :], vals, mask=mask)
        tl.store(out_ptr + out_rows1[:, None] * channels + c_offs[None, :], vals, mask=mask)

    @triton.jit
    def _wan_dup_up_3d_cf_kernel(
        x_ptr,
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
        block_w: tl.constexpr,
    ):
        """Bit-exact single-pass NCDHW ``DupUp3D`` (repeat_interleave + 3D shuffle + trim)."""
        w_blk = tl.program_id(0).to(tl.int64)
        ho = tl.program_id(1).to(tl.int64)
        bct = tl.program_id(2).to(tl.int64)
        to = bct % t_out
        bc = bct // t_out
        co = bc % out_channels
        b = bc // out_channels

        wo_offs = w_blk * block_w + tl.arange(0, block_w)
        mask = wo_offs < w_out

        to_full = to + t_trim
        ti = to_full // factor_t
        rt = to_full % factor_t
        hi = ho // factor_s
        rh = ho % factor_s
        wi = wo_offs // factor_s
        rw = wo_offs % factor_s

        # Map output (co, rt, rh, rw) back to repeat_interleave channel index.
        flat_rep_ch = ((co * factor_t + rt) * factor_s + rh) * factor_s + rw
        ci = flat_rep_ch // repeats

        in_idx = (((b * in_channels + ci) * t_in + ti) * h_in + hi) * w_in + wi
        vals = tl.load(x_ptr + in_idx, mask=mask)
        out_idx = (((b * out_channels + co) * t_out + to) * h_out + ho) * w_out + wo_offs
        tl.store(out_ptr + out_idx, vals, mask=mask)

    return {
        "triton": triton,
        "rms_cl": _wan_rms_norm_channels_last_kernel,
        "rms_cf": _wan_rms_norm_channels_first_kernel,
        "rms_pad_cf": _wan_rms_norm_silu_causal_pad_cf_kernel,
        "rms_pad_cl": _wan_rms_norm_silu_causal_pad_cl_kernel,
        "up2x_nchw": _wan_nearest_upsample_2x_nchw_kernel,
        "up2x_nhwc": _wan_nearest_upsample_2x_nhwc_kernel,
        "dup_up_cf": _wan_dup_up_3d_cf_kernel,
    }


@torch.library.custom_op("worldfoundry::wan_vae_rms_norm", mutates_args=(), device_types="cuda")
def _wan_vae_rms_norm_op(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    channels = x.shape[1]
    has_bias = bias is not None
    gamma_1d = gamma.reshape(channels).contiguous()
    bias_1d = bias.reshape(channels).contiguous() if bias is not None else gamma_1d

    if _is_channels_last_nd(x):
        out = torch.empty_like(x)
        total_pixels = x.numel() // channels
        if total_pixels == 0:
            return out
        block_c = triton.next_power_of_2(channels)
        block_p = 32 if block_c <= 128 else (16 if block_c <= 512 else 8)
        num_warps = 4 if block_p * block_c <= 2048 else 8
        kernels["rms_cl"][(triton.cdiv(total_pixels, block_p),)](
            x,
            gamma_1d,
            bias_1d,
            out,
            total_pixels,
            float(scale),
            channels=channels,
            block_p=block_p,
            block_c=block_c,
            has_bias=has_bias,
            apply_silu=bool(apply_silu),
            num_warps=num_warps,
        )
        return out

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    batch_size = x_c.shape[0]
    spatial_size = x_c.numel() // (batch_size * channels)
    if spatial_size == 0 or batch_size == 0:
        return out
    block_s = 256 if channels <= 256 else 128
    num_warps = 4 if block_s <= 256 else 8
    kernels["rms_cf"][(triton.cdiv(spatial_size, block_s), batch_size)](
        x_c,
        gamma_1d,
        bias_1d,
        out,
        batch_size,
        spatial_size,
        float(scale),
        channels=channels,
        block_s=block_s,
        has_bias=has_bias,
        apply_silu=bool(apply_silu),
        num_warps=num_warps,
    )
    return out


@_wan_vae_rms_norm_op.register_fake
def _wan_vae_rms_norm_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    del gamma, bias, scale, apply_silu
    return torch.empty_like(x)


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
) -> torch.Tensor:
    """Run lossless fused WanVAE ``RMS_norm`` with optional ``SiLU`` epilogue."""
    eff_scale = math.sqrt(float(x.shape[1])) if scale is None else float(scale)
    return _wan_vae_rms_norm_op(x, gamma, bias, eff_scale, bool(apply_silu))


@torch.library.custom_op("worldfoundry::wan_vae_rms_norm_silu_causal_pad", mutates_args=(), device_types="cuda")
def _wan_vae_rms_norm_silu_causal_pad_op(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    scale: float,
    write_cache: bool,
    cache_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    b, c, t_in, h_in, w_in = x.shape
    has_bias = bias is not None
    gamma_1d = gamma.reshape(c).contiguous()
    bias_1d = bias.reshape(c).contiguous() if bias is not None else gamma_1d

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

    # Determine output cache temporal size matching ResidualBlock semantics:
    # if t_in >= cache_t: last cache_t frames of y;
    # if t_in < cache_t and has_cache_in: [cache_in[:, :, -1:], y];
    # else: y (t_in frames).
    copy_prev_cache_to_cache_out = bool(write_cache and t_in < cache_t and has_cache_in)
    t_cache_out = min(t_in, cache_t) + (1 if copy_prev_cache_to_cache_out else 0) if write_cache else 0

    is_cl = _is_channels_last_nd(x)
    if is_cl:
        out_pad = torch.empty(
            (b, c, t_out, h_out, w_out),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
        cache_out = (
            torch.empty(
                (b, c, max(t_cache_out, 1), h_in, w_in),
                device=x.device,
                dtype=x.dtype,
                memory_format=torch.channels_last_3d,
            )
            if write_cache
            else out_pad.new_empty((0,))
        )
        cache_in_t = (
            cache_in
            if (
                has_cache_in
                and cache_in is not None
                and cache_in.device == x.device
                and cache_in.dtype == x.dtype
                and _is_channels_last_nd(cache_in)
            )
            else (
                cache_in.to(device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
                if has_cache_in and cache_in is not None
                else out_pad
            )
        )
        block_c = triton.next_power_of_2(c)
        block_w = 32 if block_c <= 128 else (16 if block_c <= 256 else 8)
        num_warps = 4 if block_w * block_c <= 2048 else 8
        kernels["rms_pad_cl"][(triton.cdiv(w_out, block_w), h_out, b * t_out)](
            x,
            gamma_1d,
            bias_1d,
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
            has_cache_in=has_cache_in,
            write_cache_out=bool(write_cache),
            copy_prev_cache_to_cache_out=copy_prev_cache_to_cache_out,
            num_warps=num_warps,
        )
        return out_pad, cache_out

    x_c = x.contiguous()
    out_pad = torch.empty((b, c, t_out, h_out, w_out), device=x.device, dtype=x.dtype)
    cache_out = (
        torch.empty((b, c, max(t_cache_out, 1), h_in, w_in), device=x.device, dtype=x.dtype)
        if write_cache
        else out_pad.new_empty((0,))
    )
    cache_in_t = (
        cache_in.to(device=x.device, dtype=x.dtype).contiguous() if has_cache_in and cache_in is not None else out_pad
    )
    block_w = 128 if w_out >= 128 else triton.next_power_of_2(max(w_out, 16))
    kernels["rms_pad_cf"][(triton.cdiv(w_out, block_w), h_out, b * t_out)](
        x_c,
        gamma_1d,
        bias_1d,
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
        has_cache_in=has_cache_in,
        write_cache_out=bool(write_cache),
        copy_prev_cache_to_cache_out=copy_prev_cache_to_cache_out,
        num_warps=4,
    )
    return out_pad, cache_out


@_wan_vae_rms_norm_silu_causal_pad_op.register_fake
def _wan_vae_rms_norm_silu_causal_pad_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    pad_w: int,
    pad_h: int,
    pad_t_left: int,
    scale: float,
    write_cache: bool,
    cache_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gamma, bias, scale
    b, c, t_in, h_in, w_in = x.shape
    has_cache_in = cache_in is not None and pad_t_left > 0
    t_cache_in = int(cache_in.shape[2]) if has_cache_in and cache_in is not None else 0
    t_pad_rem = max(0, int(pad_t_left) - t_cache_in)
    t_out = t_pad_rem + t_cache_in + t_in
    out_pad = x.new_empty((b, c, t_out, h_in + 2 * pad_h, w_in + 2 * pad_w))
    copy_prev = bool(write_cache and t_in < cache_t and has_cache_in)
    t_cache_out = min(t_in, cache_t) + (1 if copy_prev else 0)
    cache_out = x.new_empty((b, c, max(t_cache_out, 1), h_in, w_in)) if write_cache else x.new_empty((0,))
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Super-fused ``RMS_norm + SiLU + cache snapshot + cat + F.pad`` for ``CausalConv3d``."""
    eff_scale = math.sqrt(float(x.shape[1])) if scale is None else float(scale)
    return _wan_vae_rms_norm_silu_causal_pad_op(
        x,
        gamma,
        bias,
        cache_in,
        int(pad_w),
        int(pad_h),
        int(pad_t_left),
        eff_scale,
        bool(write_cache),
        int(cache_t),
    )


@torch.library.custom_op("worldfoundry::wan_vae_nearest_upsample_2x", mutates_args=(), device_types="cuda")
def _wan_vae_nearest_upsample_2x_op(x: torch.Tensor) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    b, c, h, w = x.shape
    if _is_channels_last_nd(x) and c <= 1024:
        out = torch.empty(
            (b, c, h * 2, w * 2),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last,
        )
        if x.numel() == 0:
            return out
        block_c = triton.next_power_of_2(c)
        block_w = 32 if block_c <= 128 else (16 if block_c <= 256 else 8)
        kernels["up2x_nhwc"][(triton.cdiv(w * 2, block_w), h, b)](
            x,
            out,
            b,
            h,
            w,
            channels=c,
            block_w=block_w,
            block_c=block_c,
            num_warps=4,
        )
        return out

    x_c = x.contiguous()
    out = torch.empty((b, c, h * 2, w * 2), device=x.device, dtype=x.dtype)
    if x_c.numel() == 0:
        return out
    block_w_in = 256 if w >= 256 else triton.next_power_of_2(max(w, 16))
    rows_per_cta = 4 if h >= 4 else 1
    kernels["up2x_nchw"][(triton.cdiv(w, block_w_in), triton.cdiv(h, rows_per_cta), b * c)](
        x_c,
        out,
        b * c,
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
    return x.new_empty((b, c, h * 2, w * 2))


def wan_vae_nearest_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact native BF16/FP16/FP32 2x2 nearest upsample for 4D NCHW/NHWC."""
    return _wan_vae_nearest_upsample_2x_op(x)


@torch.library.custom_op("worldfoundry::wan_vae_dup_up_3d", mutates_args=(), device_types="cuda")
def _wan_vae_dup_up_3d_op(
    x: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    x_c = x.contiguous()
    b, c_in, t_in, h_in, w_in = x_c.shape
    repeats = (out_channels * factor_t * factor_s * factor_s) // c_in
    t_trim = (factor_t - 1) if first_chunk else 0
    t_out = t_in * factor_t - t_trim
    h_out = h_in * factor_s
    w_out = w_in * factor_s
    out = torch.empty((b, out_channels, t_out, h_out, w_out), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    block_w = 256 if w_out >= 256 else triton.next_power_of_2(max(w_out, 16))
    kernels["dup_up_cf"][(triton.cdiv(w_out, block_w), h_out, b * out_channels * t_out)](
        x_c,
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
        block_w=block_w,
        num_warps=4,
    )
    return out


@_wan_vae_dup_up_3d_op.register_fake
def _wan_vae_dup_up_3d_fake(
    x: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    b, _, t_in, h_in, w_in = x.shape
    t_trim = (factor_t - 1) if first_chunk else 0
    return x.new_empty((b, out_channels, t_in * factor_t - t_trim, h_in * factor_s, w_in * factor_s))


def wan_vae_dup_up_3d(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Bit-exact single-pass ``DupUp3D`` shortcut for ``WanVideoVAE38``."""
    return _wan_vae_dup_up_3d_op(x, int(out_channels), int(factor_t), int(factor_s), bool(first_chunk))


__all__ = [
    "can_use_wan_vae_rms_norm",
    "wan_vae_dup_up_3d",
    "wan_vae_nearest_upsample_2x",
    "wan_vae_rms_norm",
    "wan_vae_rms_norm_silu_causal_pad",
]
