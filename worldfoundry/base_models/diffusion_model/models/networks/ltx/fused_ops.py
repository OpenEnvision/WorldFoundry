# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels and post-load acceleration transforms for LTX / LTX-2.

Provides inference-focused operators that preserve LTX checkpoint keys and
numerical contracts while removing redundant HBM round-trips and launches:

- Fused full-hidden QK-RMSNorm + Split / Interleaved RoPE on packed ``(B, T, H*D)``
  tokens (eliminates separate norm, transpose, einops rearrange, and sin negation).
- Fused post-self-attention residual-gate add + RMSNorm (computes both the updated
  residual and its RMS-normalized view in one pass).
- Post-load QKV (self-attention) and KV (cross-attention) projection fusion,
  compatible with subsequent Hopper/Blackwell rowwise FP8 W8A8 conversion.
- Fused FP8 GELU-tanh FFN enablement for LTX's ``GELUApprox`` + ``Float8Linear``
  feed-forward layout.

Importing this module does not import Triton or initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.nn import (
    PytorchAdaZeroFunction,
    PytorchGatedAttention,
    PytorchPreAttention,
    TransformerAttentionOps,
    TransformerOpsConfig,
)
from worldfoundry.core.nn.blocks.normalization import rms_norm

from .rope import LTXRopeType, apply_rotary_emb

LTX_DEFAULT_FP8_EXCLUDES: tuple[str, ...] = (
    "adaln_single",
    "prompt_adaln_single",
    "av_ca_",
    "patchify_proj",
    "proj_out",
    "caption_projection",
    "to_gate_logits",
    "scale_shift_table",
    "action_embedder",
    "action_adaln",
)

LTX_AUDIO_FP8_EXCLUDES: tuple[str, ...] = (
    "audio_attn",
    "audio_ff",
    "video_to_audio",
)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Triton kernels on first eligible CUDA call."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="ltx-fused-ops")

    import triton
    import triton.language as tl

    @triton.jit
    def _round_bf16(val):
        bits = val.to(tl.uint32, bitcast=True)
        is_special = (bits & 0x7F800000) == 0x7F800000
        bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = ((bits + bias) & 0xFFFF0000).to(tl.float32, bitcast=True)
        return tl.where(is_special, val, rounded)

    @triton.jit
    def _rmsnorm_split_rope_single_kernel(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        x_stride_b,
        x_stride_t,
        out_stride_b,
        out_stride_t,
        cos_stride_b,
        cos_stride_h,
        cos_stride_t,
        cos_stride_d,
        sin_stride_b,
        sin_stride_h,
        sin_stride_t,
        sin_stride_d,
        seq_len,
        cos_batch_broadcast: tl.constexpr,
        eps,
        inner_dim: tl.constexpr,
        half_dim: tl.constexpr,
        dim_head: tl.constexpr,
        half_head: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        """One program per ``(b, t)`` token: full-hidden RMSNorm + split-half RoPE."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        idx = tl.arange(0, BLOCK_HALF)
        mask = idx < half_dim
        head = idx // half_head
        j = idx % half_head
        col0 = head * dim_head + j
        col1 = col0 + half_head

        x_base = b * x_stride_b + t * x_stride_t
        x0_in = tl.load(x_ptr + x_base + col0, mask=mask, other=0.0)
        x1_in = tl.load(x_ptr + x_base + col1, mask=mask, other=0.0)
        x0_f = x0_in.to(tl.float32)
        x1_f = x1_in.to(tl.float32)

        sq_sum = tl.sum(x0_f * x0_f + x1_f * x1_f, axis=0)
        rstd = tl.rsqrt(sq_sum / inner_dim + eps)

        w0 = tl.load(w_ptr + col0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + col1, mask=mask, other=0.0).to(tl.float32)
        # Match torch.nn.RMSNorm: apply weight in FP32 then round once to input dtype.
        n0 = (x0_f * rstd * w0).to(x0_in.dtype).to(tl.float32)
        n1 = (x1_f * rstd * w1).to(x1_in.dtype).to(tl.float32)

        cos_off = b_freq * cos_stride_b + head * cos_stride_h + t * cos_stride_t + j * cos_stride_d
        sin_off = b_freq * sin_stride_b + head * sin_stride_h + t * sin_stride_t + j * sin_stride_d
        c = tl.load(cos_ptr + cos_off, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + sin_off, mask=mask, other=0.0).to(tl.float32)

        y0 = n0 * c - n1 * s
        y1 = n1 * c + n0 * s

        out_base = b * out_stride_b + t * out_stride_t
        tl.store(out_ptr + out_base + col0, y0.to(x0_in.dtype), mask=mask)
        tl.store(out_ptr + out_base + col1, y1.to(x1_in.dtype), mask=mask)

    @triton.jit
    def _qk_rmsnorm_split_rope_paired_kernel(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_b,
        q_stride_t,
        k_stride_b,
        k_stride_t,
        q_out_stride_b,
        q_out_stride_t,
        k_out_stride_b,
        k_out_stride_t,
        cos_stride_b,
        cos_stride_h,
        cos_stride_t,
        cos_stride_d,
        sin_stride_b,
        sin_stride_h,
        sin_stride_t,
        sin_stride_d,
        batch_seq,
        seq_len,
        cos_batch_broadcast: tl.constexpr,
        eps,
        inner_dim: tl.constexpr,
        half_dim: tl.constexpr,
        dim_head: tl.constexpr,
        half_head: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        """Process Q (pid < B*T) and K (pid >= B*T) in one launch for self-attention."""
        pid = tl.program_id(0).to(tl.int64)
        is_k = pid >= batch_seq
        row = tl.where(is_k, pid - batch_seq, pid)
        b = row // seq_len
        t = row % seq_len
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        idx = tl.arange(0, BLOCK_HALF)
        mask = idx < half_dim
        head = idx // half_head
        j = idx % half_head
        col0 = head * dim_head + j
        col1 = col0 + half_head

        if is_k:
            x_base = b * k_stride_b + t * k_stride_t
            x0_in = tl.load(k_ptr + x_base + col0, mask=mask, other=0.0)
            x1_in = tl.load(k_ptr + x_base + col1, mask=mask, other=0.0)
            w0 = tl.load(kw_ptr + col0, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(kw_ptr + col1, mask=mask, other=0.0).to(tl.float32)
        else:
            x_base = b * q_stride_b + t * q_stride_t
            x0_in = tl.load(q_ptr + x_base + col0, mask=mask, other=0.0)
            x1_in = tl.load(q_ptr + x_base + col1, mask=mask, other=0.0)
            w0 = tl.load(qw_ptr + col0, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(qw_ptr + col1, mask=mask, other=0.0).to(tl.float32)

        x0_f = x0_in.to(tl.float32)
        x1_f = x1_in.to(tl.float32)
        sq_sum = tl.sum(x0_f * x0_f + x1_f * x1_f, axis=0)
        rstd = tl.rsqrt(sq_sum / inner_dim + eps)
        n0 = (x0_f * rstd * w0).to(x0_in.dtype).to(tl.float32)
        n1 = (x1_f * rstd * w1).to(x1_in.dtype).to(tl.float32)

        cos_off = b_freq * cos_stride_b + head * cos_stride_h + t * cos_stride_t + j * cos_stride_d
        sin_off = b_freq * sin_stride_b + head * sin_stride_h + t * sin_stride_t + j * sin_stride_d
        c = tl.load(cos_ptr + cos_off, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + sin_off, mask=mask, other=0.0).to(tl.float32)

        y0 = (n0 * c - n1 * s).to(x0_in.dtype)
        y1 = (n1 * c + n0 * s).to(x1_in.dtype)

        if is_k:
            out_base = b * k_out_stride_b + t * k_out_stride_t
            tl.store(k_out_ptr + out_base + col0, y0, mask=mask)
            tl.store(k_out_ptr + out_base + col1, y1, mask=mask)
        else:
            out_base = b * q_out_stride_b + t * q_out_stride_t
            tl.store(q_out_ptr + out_base + col0, y0, mask=mask)
            tl.store(q_out_ptr + out_base + col1, y1, mask=mask)

    @triton.jit
    def _rmsnorm_interleaved_rope_single_kernel(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        x_stride_b,
        x_stride_t,
        out_stride_b,
        out_stride_t,
        cos_stride_b,
        cos_stride_t,
        sin_stride_b,
        sin_stride_t,
        seq_len,
        cos_batch_broadcast: tl.constexpr,
        eps,
        inner_dim: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        """One program per ``(b, t)`` token: full-hidden RMSNorm + interleaved pair RoPE."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        idx = tl.arange(0, BLOCK_HALF)
        mask = idx < half_dim
        even_col = idx * 2
        odd_col = even_col + 1

        x_base = b * x_stride_b + t * x_stride_t
        xe_in = tl.load(x_ptr + x_base + even_col, mask=mask, other=0.0)
        xo_in = tl.load(x_ptr + x_base + odd_col, mask=mask, other=0.0)
        xe_f = xe_in.to(tl.float32)
        xo_f = xo_in.to(tl.float32)

        sq_sum = tl.sum(xe_f * xe_f + xo_f * xo_f, axis=0)
        rstd = tl.rsqrt(sq_sum / inner_dim + eps)

        we = tl.load(w_ptr + even_col, mask=mask, other=0.0).to(tl.float32)
        wo = tl.load(w_ptr + odd_col, mask=mask, other=0.0).to(tl.float32)
        ne = (xe_f * rstd * we).to(xe_in.dtype).to(tl.float32)
        no = (xo_f * rstd * wo).to(xo_in.dtype).to(tl.float32)

        cos_base = b_freq * cos_stride_b + t * cos_stride_t
        sin_base = b_freq * sin_stride_b + t * sin_stride_t
        ce = tl.load(cos_ptr + cos_base + even_col, mask=mask, other=1.0).to(tl.float32)
        co = tl.load(cos_ptr + cos_base + odd_col, mask=mask, other=1.0).to(tl.float32)
        se = tl.load(sin_ptr + sin_base + even_col, mask=mask, other=0.0).to(tl.float32)
        so = tl.load(sin_ptr + sin_base + odd_col, mask=mask, other=0.0).to(tl.float32)

        ye = (ne * ce - no * se).to(xe_in.dtype)
        yo = (no * co + ne * so).to(xo_in.dtype)

        out_base = b * out_stride_b + t * out_stride_t
        tl.store(out_ptr + out_base + even_col, ye, mask=mask)
        tl.store(out_ptr + out_base + odd_col, yo, mask=mask)

    @triton.jit
    def _post_sa_residual_rmsnorm_kernel(
        res_ptr,
        val_ptr,
        gate_ptr,
        out_ptr,
        norm_ptr,
        res_stride_b,
        res_stride_t,
        val_stride_b,
        val_stride_t,
        gate_stride_b,
        gate_stride_t,
        out_stride_b,
        out_stride_t,
        norm_stride_b,
        norm_stride_t,
        seq_len,
        eps,
        features: tl.constexpr,
        round_bf16: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute ``out = res + val * gate`` and ``norm = rms_norm(out, eps)`` in one pass."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        res_in = tl.load(res_ptr + b * res_stride_b + t * res_stride_t + cols, mask=mask, other=0.0)
        val_in = tl.load(val_ptr + b * val_stride_b + t * val_stride_t + cols, mask=mask, other=0.0)
        gate_in = tl.load(gate_ptr + b * gate_stride_b + t * gate_stride_t + cols, mask=mask, other=0.0)

        prod = val_in.to(tl.float32) * gate_in.to(tl.float32)
        if round_bf16:
            prod = _round_bf16(prod)
        out_val = (res_in.to(tl.float32) + prod).to(res_in.dtype)
        out_f = out_val.to(tl.float32)
        variance = tl.sum(out_f * out_f, axis=0) / features
        norm_val = (out_f * tl.rsqrt(variance + eps)).to(res_in.dtype)

        tl.store(out_ptr + b * out_stride_b + t * out_stride_t + cols, out_val, mask=mask)
        tl.store(norm_ptr + b * norm_stride_b + t * norm_stride_t + cols, norm_val, mask=mask)

    @triton.jit
    def _post_sa_residual_rmsnorm_adaln_kernel(
        res_ptr,
        val_ptr,
        gate_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        mod_ptr,
        res_stride_b,
        res_stride_t,
        val_stride_b,
        val_stride_t,
        gate_stride_b,
        gate_stride_t,
        scale_stride_b,
        scale_stride_t,
        shift_stride_b,
        shift_stride_t,
        out_stride_b,
        out_stride_t,
        mod_stride_b,
        mod_stride_t,
        seq_len,
        eps,
        features: tl.constexpr,
        round_bf16: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse ``out = res + val * gate`` and ``mod = rms_norm(out, eps) * (1 + scale) + shift``."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        res_in = tl.load(res_ptr + b * res_stride_b + t * res_stride_t + cols, mask=mask, other=0.0)
        val_in = tl.load(val_ptr + b * val_stride_b + t * val_stride_t + cols, mask=mask, other=0.0)
        gate_in = tl.load(gate_ptr + b * gate_stride_b + t * gate_stride_t + cols, mask=mask, other=0.0)

        prod = val_in.to(tl.float32) * gate_in.to(tl.float32)
        if round_bf16:
            prod = _round_bf16(prod)
        out_val = (res_in.to(tl.float32) + prod).to(res_in.dtype)
        out_f = out_val.to(tl.float32)
        variance = tl.sum(out_f * out_f, axis=0) / features
        norm_f = (out_f * tl.rsqrt(variance + eps)).to(res_in.dtype).to(tl.float32)

        scale_f = tl.load(scale_ptr + b * scale_stride_b + t * scale_stride_t + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        shift_f = tl.load(shift_ptr + b * shift_stride_b + t * shift_stride_t + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        mod_val = (norm_f * (1.0 + scale_f) + shift_f).to(res_in.dtype)

        tl.store(out_ptr + b * out_stride_b + t * out_stride_t + cols, out_val, mask=mask)
        tl.store(mod_ptr + b * mod_stride_b + t * mod_stride_t + cols, mod_val, mask=mask)

    @triton.jit
    def _gated_attn_kernel(
        x_ptr,
        gate_ptr,
        out_ptr,
        x_stride_b,
        x_stride_t,
        g_stride_b,
        g_stride_t,
        out_stride_b,
        out_stride_t,
        seq_len,
        inner_dim: tl.constexpr,
        dim_head: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse per-head ``x * (2 * sigmoid(gate_logits))`` in one pass."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < inner_dim
        head = cols // dim_head

        x_in = tl.load(x_ptr + b * x_stride_b + t * x_stride_t + cols, mask=mask, other=0.0)
        g_in = tl.load(gate_ptr + b * g_stride_b + t * g_stride_t + head, mask=mask, other=0.0)
        g_val = (2.0 * tl.sigmoid(g_in.to(tl.float32))).to(g_in.dtype).to(tl.float32)
        y = (x_in.to(tl.float32) * g_val).to(x_in.dtype)
        tl.store(out_ptr + b * out_stride_b + t * out_stride_t + cols, y, mask=mask)

    @triton.jit
    def _gated_attn_rowwise_fp8_kernel(
        x_ptr,
        gate_ptr,
        codes_ptr,
        scale_ptr,
        x_stride_b,
        x_stride_t,
        g_stride_b,
        g_stride_t,
        codes_stride_r,
        seq_len,
        fp8_max,
        tiny,
        inner_dim: tl.constexpr,
        dim_head: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Fuse per-head ``2 * sigmoid(gate) * x`` and rowwise FP8 quantization in one pass."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < inner_dim
        head = cols // dim_head

        x_in = tl.load(x_ptr + b * x_stride_b + t * x_stride_t + cols, mask=mask, other=0.0)
        g_in = tl.load(gate_ptr + b * g_stride_b + t * g_stride_t + head, mask=mask, other=0.0)
        g_val = (2.0 * tl.sigmoid(g_in.to(tl.float32))).to(g_in.dtype).to(tl.float32)
        y = (x_in.to(tl.float32) * g_val).to(x_in.dtype).to(tl.float32)
        y = tl.where(mask, y, 0.0)

        amax = tl.max(tl.abs(y), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        q = tl.minimum(tl.maximum(y / scale, -fp8_max_f), fp8_max_f)
        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        tl.store(codes_ptr + row * codes_stride_r + cols, q.to(out_dtype), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _rmsnorm_scale_shift_fp8_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        codes_ptr,
        row_scale_ptr,
        x_stride_b,
        x_stride_t,
        s_stride_b,
        s_stride_t,
        sh_stride_b,
        sh_stride_t,
        codes_stride_r,
        seq_len,
        eps,
        fp8_max,
        tiny,
        features: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Fuse unweighted RMSNorm + AdaLN scale/shift + rowwise FP8 quantization."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        x_in = tl.load(x_ptr + b * x_stride_b + t * x_stride_t + cols, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)
        variance = tl.sum(x_f * x_f, axis=0) / features
        norm_f = (x_f * tl.rsqrt(variance + eps)).to(x_in.dtype).to(tl.float32)

        s_f = tl.load(scale_ptr + b * s_stride_b + t * s_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        sh_f = tl.load(shift_ptr + b * sh_stride_b + t * sh_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        y = (norm_f * (1.0 + s_f) + sh_f).to(x_in.dtype).to(tl.float32)
        y = tl.where(mask, y, 0.0)

        amax = tl.max(tl.abs(y), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        row_scale = tl.maximum(amax / fp8_max_f, tiny_f)
        q = tl.minimum(tl.maximum(y / row_scale, -fp8_max_f), fp8_max_f)
        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        tl.store(codes_ptr + row * codes_stride_r + cols, q.to(out_dtype), mask=mask)
        tl.store(row_scale_ptr + row, row_scale)

    @triton.jit
    def _dual_rmsnorm_scale_shift_kernel(
        x_ptr,
        s1_ptr,
        sh1_ptr,
        s2_ptr,
        sh2_ptr,
        o1_ptr,
        o2_ptr,
        x_stride_b,
        x_stride_t,
        s1_stride_b,
        s1_stride_t,
        sh1_stride_b,
        sh1_stride_t,
        s2_stride_b,
        s2_stride_t,
        sh2_stride_b,
        sh2_stride_t,
        o1_stride_b,
        o1_stride_t,
        o2_stride_b,
        o2_stride_t,
        seq_len,
        eps,
        features: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Read ``x`` and compute RMSNorm once to produce two AdaLN-modulated outputs."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        x_in = tl.load(x_ptr + b * x_stride_b + t * x_stride_t + cols, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)
        variance = tl.sum(x_f * x_f, axis=0) / features
        norm_f = (x_f * tl.rsqrt(variance + eps)).to(x_in.dtype).to(tl.float32)

        s1 = tl.load(s1_ptr + b * s1_stride_b + t * s1_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        sh1 = tl.load(sh1_ptr + b * sh1_stride_b + t * sh1_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(s2_ptr + b * s2_stride_b + t * s2_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        sh2 = tl.load(sh2_ptr + b * sh2_stride_b + t * sh2_stride_t + cols, mask=mask, other=0.0).to(tl.float32)

        y1 = (norm_f * (1.0 + s1) + sh1).to(x_in.dtype)
        y2 = (norm_f * (1.0 + s2) + sh2).to(x_in.dtype)
        tl.store(o1_ptr + b * o1_stride_b + t * o1_stride_t + cols, y1, mask=mask)
        tl.store(o2_ptr + b * o2_stride_b + t * o2_stride_t + cols, y2, mask=mask)

    @triton.jit
    def _scale_shift_kernel(
        x_ptr,
        s_ptr,
        sh_ptr,
        out_ptr,
        x_stride_b,
        x_stride_t,
        s_stride_b,
        s_stride_t,
        sh_stride_b,
        sh_stride_t,
        out_stride_b,
        out_stride_t,
        seq_len,
        features: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse ``x * (1 + scale) + shift`` in one pass."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        t = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        x_in = tl.load(x_ptr + b * x_stride_b + t * x_stride_t + cols, mask=mask, other=0.0)
        s_f = tl.load(s_ptr + b * s_stride_b + t * s_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        sh_f = tl.load(sh_ptr + b * sh_stride_b + t * sh_stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x_in.to(tl.float32) * (1.0 + s_f) + sh_f).to(x_in.dtype)
        tl.store(out_ptr + b * out_stride_b + t * out_stride_t + cols, y, mask=mask)

    @triton.jit
    def _qk_rmsnorm_paired_kernel(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_b,
        q_stride_t,
        k_stride_b,
        k_stride_t,
        qo_stride_b,
        qo_stride_t,
        ko_stride_b,
        ko_stride_t,
        q_rows,
        q_seq,
        k_seq,
        eps,
        features: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse Q and K full-hidden RMSNorm (without RoPE) into a single launch."""
        pid = tl.program_id(0).to(tl.int64)
        is_k = pid >= q_rows
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features

        if is_k:
            row = pid - q_rows
            b = row // k_seq
            t = row % k_seq
            x_in = tl.load(k_ptr + b * k_stride_b + t * k_stride_t + cols, mask=mask, other=0.0)
            w_f = tl.load(kw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            row = pid
            b = row // q_seq
            t = row % q_seq
            x_in = tl.load(q_ptr + b * q_stride_b + t * q_stride_t + cols, mask=mask, other=0.0)
            w_f = tl.load(qw_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x_f = x_in.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=0) / features + eps)
        y = (x_f * rstd * w_f).to(x_in.dtype)

        if is_k:
            tl.store(k_out_ptr + b * ko_stride_b + t * ko_stride_t + cols, y, mask=mask)
        else:
            tl.store(q_out_ptr + b * qo_stride_b + t * qo_stride_t + cols, y, mask=mask)

    @triton.jit
    def _gelu_tanh_fp8_1pass_kernel(
        in_ptr,
        out_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        in_stride,
        out_stride,
        BLOCK_SIZE: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Single-pass GELU-tanh + rowwise FP8 quantization for ``n_cols <= 4096``."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=0.0).to(tl.float32)
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        g = (x * tl.sigmoid(2.0 * inner)).to(in_ptr.dtype.element_ty).to(tl.float32)
        g = tl.where(mask, g, 0.0)
        amax = tl.max(tl.abs(g), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        q = tl.minimum(tl.maximum(g / scale, -fp8_max_f), fp8_max_f)
        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        tl.store(out_ptr + row * out_stride + cols, q.to(out_dtype), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _gelu_tanh_fp8_monotone_kernel(
        in_ptr,
        out_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        in_stride,
        out_stride,
        BLOCK_N: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Tiled GELU-tanh + rowwise FP8 avoiding redundant tanh in pass 1 via monotonicity.

        ``GELU_tanh(x)`` is strictly increasing on ``[0, +inf)`` and bounded in magnitude
        by ``0.17005`` on ``(-inf, 0]``. Whenever a row's finite maximum ``max_x >= 0.4``
        (``GELU_tanh(0.4) > 0.262``), ``max_i |GELU_tanh(x_i)| == GELU_tanh(max_x)``
        exactly, cutting transcendental math in half for wide FFN layers.
        """
        row = tl.program_id(0).to(tl.int64)
        row_in = in_ptr + row * in_stride
        row_out = out_ptr + row * out_stride
        max_x = tl.full((), -1e30, tl.float32)
        for start in tl.range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            x = tl.load(row_in + cols, mask=mask, other=-1e30).to(tl.float32)
            max_x = tl.maximum(max_x, tl.max(x, axis=0))

        if (max_x >= 0.4) & (max_x < 1e30):
            inner_m = 0.7978845608028654 * (max_x + 0.044715 * max_x * max_x * max_x)
            g_m = (max_x * tl.sigmoid(2.0 * inner_m)).to(in_ptr.dtype.element_ty).to(tl.float32)
            amax = tl.abs(g_m)
        else:
            amax = tl.full((), 0.0, tl.float32)
            for start in tl.range(0, n_cols, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < n_cols
                x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
                inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
                g = (x * tl.sigmoid(2.0 * inner)).to(in_ptr.dtype.element_ty).to(tl.float32)
                g = tl.where(mask, g, 0.0)
                amax = tl.maximum(amax, tl.max(tl.abs(g), axis=0))

        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)

        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        for start in tl.range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
            inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            g = (x * tl.sigmoid(2.0 * inner)).to(in_ptr.dtype.element_ty).to(tl.float32)
            q = tl.minimum(tl.maximum(g * inv_scale, -fp8_max_f), fp8_max_f)
            tl.store(row_out + cols, q.to(out_dtype), mask=mask)

    return {
        "triton": triton,
        "split_single": _rmsnorm_split_rope_single_kernel,
        "split_paired": _qk_rmsnorm_split_rope_paired_kernel,
        "interleaved_single": _rmsnorm_interleaved_rope_single_kernel,
        "post_sa": _post_sa_residual_rmsnorm_kernel,
        "post_sa_adaln": _post_sa_residual_rmsnorm_adaln_kernel,
        "gated_attn": _gated_attn_kernel,
        "gated_attn_fp8": _gated_attn_rowwise_fp8_kernel,
        "rmsnorm_adaln_fp8": _rmsnorm_scale_shift_fp8_kernel,
        "dual_rmsnorm_adaln": _dual_rmsnorm_scale_shift_kernel,
        "scale_shift": _scale_shift_kernel,
        "qk_rmsnorm_paired": _qk_rmsnorm_paired_kernel,
        "gelu_fp8_1pass": _gelu_tanh_fp8_1pass_kernel,
        "gelu_fp8_monotone": _gelu_tanh_fp8_monotone_kernel,
    }


def _triton_cuda_eligible(*tensors: torch.Tensor) -> bool:
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    first = tensors[0]
    if first.device.type != "cuda" or bool(getattr(torch.version, "hip", None)):
        return False
    return all(
        t.device == first.device and t.dtype in _FLOAT_DTYPES and t.numel() > 0 and t.stride(-1) == 1 for t in tensors
    )


@torch.library.custom_op(
    "worldfoundry::ltx_qk_rmsnorm_split_rope",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_qk_rmsnorm_split_rope_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    eps: float,
    heads: int,
    dim_head: int,
    paired: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    inner_dim = heads * dim_head
    half_dim = inner_dim // 2
    half_head = dim_head // 2
    block_half = triton.next_power_of_2(half_dim)
    num_warps = 4 if block_half <= 1024 else 8

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    with torch.cuda.device(q.device):
        if paired:
            batch_size, seq_len, _ = q.shape
            batch_seq = batch_size * seq_len
            kernels["split_paired"][(2 * batch_seq,)](
                q,
                k,
                q_weight,
                k_weight,
                q_cos,
                q_sin,
                q_out,
                k_out,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                q_out.stride(0),
                q_out.stride(1),
                k_out.stride(0),
                k_out.stride(1),
                q_cos.stride(0),
                q_cos.stride(1),
                q_cos.stride(2),
                q_cos.stride(3),
                q_sin.stride(0),
                q_sin.stride(1),
                q_sin.stride(2),
                q_sin.stride(3),
                batch_seq,
                seq_len,
                cos_batch_broadcast=q_cos.shape[0] == 1,
                eps=eps,
                inner_dim=inner_dim,
                half_dim=half_dim,
                dim_head=dim_head,
                half_head=half_head,
                BLOCK_HALF=block_half,
                num_warps=num_warps,
            )
        else:
            q_batch, q_seq, _ = q.shape
            kernels["split_single"][(q_batch * q_seq,)](
                q,
                q_weight,
                q_cos,
                q_sin,
                q_out,
                q.stride(0),
                q.stride(1),
                q_out.stride(0),
                q_out.stride(1),
                q_cos.stride(0),
                q_cos.stride(1),
                q_cos.stride(2),
                q_cos.stride(3),
                q_sin.stride(0),
                q_sin.stride(1),
                q_sin.stride(2),
                q_sin.stride(3),
                q_seq,
                cos_batch_broadcast=q_cos.shape[0] == 1,
                eps=eps,
                inner_dim=inner_dim,
                half_dim=half_dim,
                dim_head=dim_head,
                half_head=half_head,
                BLOCK_HALF=block_half,
                num_warps=num_warps,
            )
            k_batch, k_seq, _ = k.shape
            kernels["split_single"][(k_batch * k_seq,)](
                k,
                k_weight,
                k_cos,
                k_sin,
                k_out,
                k.stride(0),
                k.stride(1),
                k_out.stride(0),
                k_out.stride(1),
                k_cos.stride(0),
                k_cos.stride(1),
                k_cos.stride(2),
                k_cos.stride(3),
                k_sin.stride(0),
                k_sin.stride(1),
                k_sin.stride(2),
                k_sin.stride(3),
                k_seq,
                cos_batch_broadcast=k_cos.shape[0] == 1,
                eps=eps,
                inner_dim=inner_dim,
                half_dim=half_dim,
                dim_head=dim_head,
                half_head=half_head,
                BLOCK_HALF=block_half,
                num_warps=num_warps,
            )
    return q_out, k_out


@_ltx_qk_rmsnorm_split_rope_op.register_fake
def _ltx_qk_rmsnorm_split_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    eps: float,
    heads: int,
    dim_head: int,
    paired: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, q_cos, q_sin, k_cos, k_sin, eps, heads, dim_head, paired
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op(
    "worldfoundry::ltx_qk_rmsnorm_interleaved_rope",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_qk_rmsnorm_interleaved_rope_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    eps: float,
    inner_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    half_dim = inner_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    num_warps = 4 if block_half <= 1024 else 8

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    with torch.cuda.device(q.device):
        for tensor, weight, cos_f, sin_f, out in (
            (q, q_weight, q_cos, q_sin, q_out),
            (k, k_weight, k_cos, k_sin, k_out),
        ):
            batch_size, seq_len, _ = tensor.shape
            kernels["interleaved_single"][(batch_size * seq_len,)](
                tensor,
                weight,
                cos_f,
                sin_f,
                out,
                tensor.stride(0),
                tensor.stride(1),
                out.stride(0),
                out.stride(1),
                cos_f.stride(0),
                cos_f.stride(1),
                sin_f.stride(0),
                sin_f.stride(1),
                seq_len,
                cos_batch_broadcast=cos_f.shape[0] == 1,
                eps=eps,
                inner_dim=inner_dim,
                half_dim=half_dim,
                BLOCK_HALF=block_half,
                num_warps=num_warps,
            )
    return q_out, k_out


@_ltx_qk_rmsnorm_interleaved_rope_op.register_fake
def _ltx_qk_rmsnorm_interleaved_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    eps: float,
    inner_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, q_cos, q_sin, k_cos, k_sin, eps, inner_dim
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op(
    "worldfoundry::ltx_post_sa_residual_rmsnorm",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_post_sa_residual_rmsnorm_op(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, features = residual.shape
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    gate_expanded = gate.expand_as(residual)
    out = torch.empty_like(residual)
    normed = torch.empty_like(residual)

    with torch.cuda.device(residual.device):
        kernels["post_sa"][(batch_size * seq_len,)](
            residual,
            value,
            gate_expanded,
            out,
            normed,
            residual.stride(0),
            residual.stride(1),
            value.stride(0),
            value.stride(1),
            gate_expanded.stride(0),
            gate_expanded.stride(1),
            out.stride(0),
            out.stride(1),
            normed.stride(0),
            normed.stride(1),
            seq_len,
            eps,
            features=features,
            round_bf16=torch.promote_types(value.dtype, gate.dtype) == torch.bfloat16,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out, normed


@_ltx_post_sa_residual_rmsnorm_op.register_fake
def _ltx_post_sa_residual_rmsnorm_fake(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del value, gate, eps
    return torch.empty_like(residual), torch.empty_like(residual)


@torch.library.custom_op(
    "worldfoundry::ltx_post_sa_residual_rmsnorm_adaln",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_post_sa_residual_rmsnorm_adaln_op(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale_q: torch.Tensor,
    shift_q: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, features = residual.shape
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    gate_exp = gate.expand_as(residual)
    scale_exp = scale_q.expand_as(residual)
    shift_exp = shift_q.expand_as(residual)
    out = torch.empty_like(residual)
    modulated = torch.empty_like(residual)

    with torch.cuda.device(residual.device):
        kernels["post_sa_adaln"][(batch_size * seq_len,)](
            residual,
            value,
            gate_exp,
            scale_exp,
            shift_exp,
            out,
            modulated,
            residual.stride(0),
            residual.stride(1),
            value.stride(0),
            value.stride(1),
            gate_exp.stride(0),
            gate_exp.stride(1),
            scale_exp.stride(0),
            scale_exp.stride(1),
            shift_exp.stride(0),
            shift_exp.stride(1),
            out.stride(0),
            out.stride(1),
            modulated.stride(0),
            modulated.stride(1),
            seq_len,
            eps,
            features=features,
            round_bf16=torch.promote_types(value.dtype, gate.dtype) == torch.bfloat16,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out, modulated


@_ltx_post_sa_residual_rmsnorm_adaln_op.register_fake
def _ltx_post_sa_residual_rmsnorm_adaln_fake(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale_q: torch.Tensor,
    shift_q: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del value, gate, scale_q, shift_q, eps
    return torch.empty_like(residual), torch.empty_like(residual)


@torch.library.custom_op(
    "worldfoundry::ltx_gated_attention",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_gated_attention_op(
    x: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, inner_dim = x.shape
    block_size = triton.next_power_of_2(inner_dim)
    num_warps = 4 if block_size <= 2048 else 8
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        kernels["gated_attn"][(batch_size * seq_len,)](
            x,
            gate_logits,
            out,
            x.stride(0),
            x.stride(1),
            gate_logits.stride(0),
            gate_logits.stride(1),
            out.stride(0),
            out.stride(1),
            seq_len,
            inner_dim=inner_dim,
            dim_head=dim_head,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    del heads
    return out


@_ltx_gated_attention_op.register_fake
def _ltx_gated_attention_fake(
    x: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    del gate_logits, heads, dim_head
    return torch.empty_like(x)


@torch.library.custom_op(
    "worldfoundry::ltx_gated_attention_rowwise_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_gated_attention_rowwise_fp8_op(
    x: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, inner_dim = x.shape
    rows = batch_size * seq_len
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    block_size = triton.next_power_of_2(inner_dim)
    num_warps = 4 if block_size <= 2048 else 8

    codes = torch.empty((rows, inner_dim), dtype=fp8_dtype, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        kernels["gated_attn_fp8"][(rows,)](
            x,
            gate_logits,
            codes,
            scales,
            x.stride(0),
            x.stride(1),
            gate_logits.stride(0),
            gate_logits.stride(1),
            codes.stride(0),
            seq_len,
            fp8_max,
            tiny,
            inner_dim=inner_dim,
            dim_head=dim_head,
            BLOCK_SIZE=block_size,
            FP8_E5M2=fp8_is_e5m2,
            num_warps=num_warps,
        )
    del heads
    return codes, scales


@_ltx_gated_attention_rowwise_fp8_op.register_fake
def _ltx_gated_attention_rowwise_fp8_fake(
    x: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gate_logits, heads, dim_head
    rows = x.shape[0] * x.shape[1]
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    return (
        torch.empty((rows, x.shape[2]), dtype=fp8_dtype, device=x.device),
        torch.empty((rows, 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "worldfoundry::ltx_rmsnorm_scale_shift_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_rmsnorm_scale_shift_fp8_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, features = x.shape
    rows = batch_size * seq_len
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    scale_exp = scale.expand_as(x)
    shift_exp = shift.expand_as(x)
    codes = torch.empty((rows, features), dtype=fp8_dtype, device=x.device)
    row_scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        kernels["rmsnorm_adaln_fp8"][(rows,)](
            x,
            scale_exp,
            shift_exp,
            codes,
            row_scales,
            x.stride(0),
            x.stride(1),
            scale_exp.stride(0),
            scale_exp.stride(1),
            shift_exp.stride(0),
            shift_exp.stride(1),
            codes.stride(0),
            seq_len,
            eps,
            fp8_max,
            tiny,
            features=features,
            BLOCK_SIZE=block_size,
            FP8_E5M2=fp8_is_e5m2,
            num_warps=num_warps,
        )
    return codes, row_scales


@_ltx_rmsnorm_scale_shift_fp8_op.register_fake
def _ltx_rmsnorm_scale_shift_fp8_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del scale, shift, eps
    rows = x.shape[0] * x.shape[1]
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    return (
        torch.empty((rows, x.shape[2]), dtype=fp8_dtype, device=x.device),
        torch.empty((rows, 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "worldfoundry::ltx_dual_rmsnorm_scale_shift",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_dual_rmsnorm_scale_shift_op(
    x: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, features = x.shape
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    s1_exp = scale1.expand_as(x)
    sh1_exp = shift1.expand_as(x)
    s2_exp = scale2.expand_as(x)
    sh2_exp = shift2.expand_as(x)
    o1 = torch.empty_like(x)
    o2 = torch.empty_like(x)
    with torch.cuda.device(x.device):
        kernels["dual_rmsnorm_adaln"][(batch_size * seq_len,)](
            x,
            s1_exp,
            sh1_exp,
            s2_exp,
            sh2_exp,
            o1,
            o2,
            x.stride(0),
            x.stride(1),
            s1_exp.stride(0),
            s1_exp.stride(1),
            sh1_exp.stride(0),
            sh1_exp.stride(1),
            s2_exp.stride(0),
            s2_exp.stride(1),
            sh2_exp.stride(0),
            sh2_exp.stride(1),
            o1.stride(0),
            o1.stride(1),
            o2.stride(0),
            o2.stride(1),
            seq_len,
            eps,
            features=features,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return o1, o2


@_ltx_dual_rmsnorm_scale_shift_op.register_fake
def _ltx_dual_rmsnorm_scale_shift_fake(
    x: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del scale1, shift1, scale2, shift2, eps
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op(
    "worldfoundry::ltx_scale_shift",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_scale_shift_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    batch_size, seq_len, features = x.shape
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    s_exp = scale.expand_as(x)
    sh_exp = shift.expand_as(x)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        kernels["scale_shift"][(batch_size * seq_len,)](
            x,
            s_exp,
            sh_exp,
            out,
            x.stride(0),
            x.stride(1),
            s_exp.stride(0),
            s_exp.stride(1),
            sh_exp.stride(0),
            sh_exp.stride(1),
            out.stride(0),
            out.stride(1),
            seq_len,
            features=features,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out


@_ltx_scale_shift_op.register_fake
def _ltx_scale_shift_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    del scale, shift
    return torch.empty_like(x)


@torch.library.custom_op(
    "worldfoundry::ltx_qk_rmsnorm_paired",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_qk_rmsnorm_paired_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    q_batch, q_seq, features = q.shape
    k_batch, k_seq, _ = k.shape
    q_rows = q_batch * q_seq
    total_rows = q_rows + k_batch * k_seq
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    with torch.cuda.device(q.device):
        kernels["qk_rmsnorm_paired"][(total_rows,)](
            q,
            k,
            q_weight,
            k_weight,
            q_out,
            k_out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            q_rows,
            q_seq,
            k_seq,
            eps,
            features=features,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return q_out, k_out


@_ltx_qk_rmsnorm_paired_op.register_fake
def _ltx_qk_rmsnorm_paired_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, eps
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op(
    "worldfoundry::ltx_gelu_tanh_rowwise_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _ltx_gelu_tanh_rowwise_fp8_op(
    value: torch.Tensor,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    source = value if value.stride(1) == 1 else value.contiguous()
    rows, cols = source.shape
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    codes = torch.empty((rows, cols), dtype=fp8_dtype, device=source.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=source.device)
    if rows == 0 or cols == 0:
        return codes, scales
    fp8_max = float(torch.finfo(fp8_dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    with torch.cuda.device(source.device):
        if cols <= 4096:
            block_size = triton.next_power_of_2(cols)
            num_warps = 4 if block_size <= 2048 else 8
            kernels["gelu_fp8_1pass"][(rows,)](
                source,
                codes,
                scales,
                cols,
                fp8_max,
                tiny,
                source.stride(0),
                codes.stride(0),
                BLOCK_SIZE=block_size,
                FP8_E5M2=fp8_is_e5m2,
                num_warps=num_warps,
            )
        else:
            kernels["gelu_fp8_monotone"][(rows,)](
                source,
                codes,
                scales,
                cols,
                fp8_max,
                tiny,
                source.stride(0),
                codes.stride(0),
                BLOCK_N=4096,
                FP8_E5M2=fp8_is_e5m2,
                num_warps=8,
            )
    return codes, scales


@_ltx_gelu_tanh_rowwise_fp8_op.register_fake
def _ltx_gelu_tanh_rowwise_fp8_fake(
    value: torch.Tensor,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    return (
        torch.empty(value.shape, dtype=fp8_dtype, device=value.device),
        torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device),
    )


def fused_ltx_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_pe: tuple[torch.Tensor, torch.Tensor],
    k_pe: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    eps: float,
    heads: int,
    dim_head: int,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-hidden Q/K RMSNorm and LTX RoPE in one Triton launch when eligible."""
    q_cos, q_sin = q_pe
    k_cos, k_sin = q_pe if k_pe is None else k_pe
    inner_dim = heads * dim_head

    can_fuse_split = (
        rope_type == LTXRopeType.SPLIT
        and q.ndim == 3
        and k.ndim == 3
        and q.shape[-1] == inner_dim
        and k.shape[-1] == inner_dim
        and q_weight.shape == (inner_dim,)
        and k_weight.shape == (inner_dim,)
        and dim_head % 2 == 0
        and 0 < inner_dim <= 8192
        and q_cos.ndim == 4
        and q_sin.shape == q_cos.shape
        and k_cos.ndim == 4
        and k_sin.shape == k_cos.shape
        and q_cos.shape[0] in (1, q.shape[0])
        and k_cos.shape[0] in (1, k.shape[0])
        and q_cos.shape[1] == heads
        and k_cos.shape[1] == heads
        and q_cos.shape[2] == q.shape[1]
        and k_cos.shape[2] == k.shape[1]
        and q_cos.shape[3] == dim_head // 2
        and k_cos.shape[3] == dim_head // 2
        and _triton_cuda_eligible(q, k, q_weight, k_weight, q_cos, q_sin, k_cos, k_sin)
    )
    if can_fuse_split:
        paired = q.shape == k.shape and q_cos is k_cos and q_sin is k_sin
        try:
            return _ltx_qk_rmsnorm_split_rope_op(
                q,
                k,
                q_weight,
                k_weight,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
                float(eps),
                int(heads),
                int(dim_head),
                bool(paired),
            )
        except (ImportError, OSError):
            pass

    can_fuse_interleaved = (
        rope_type == LTXRopeType.INTERLEAVED
        and q.ndim == 3
        and k.ndim == 3
        and q.shape[-1] == inner_dim
        and k.shape[-1] == inner_dim
        and q_weight.shape == (inner_dim,)
        and k_weight.shape == (inner_dim,)
        and inner_dim % 2 == 0
        and 0 < inner_dim <= 8192
        and q_cos.ndim == 3
        and q_sin.shape == q_cos.shape
        and k_cos.ndim == 3
        and k_sin.shape == k_cos.shape
        and q_cos.shape[0] in (1, q.shape[0])
        and k_cos.shape[0] in (1, k.shape[0])
        and q_cos.shape[1] == q.shape[1]
        and k_cos.shape[1] == k.shape[1]
        and q_cos.shape[2] == inner_dim
        and k_cos.shape[2] == inner_dim
        and _triton_cuda_eligible(q, k, q_weight, k_weight, q_cos, q_sin, k_cos, k_sin)
    )
    if can_fuse_interleaved:
        try:
            return _ltx_qk_rmsnorm_interleaved_rope_op(
                q,
                k,
                q_weight,
                k_weight,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
                float(eps),
                int(inner_dim),
            )
        except (ImportError, OSError):
            pass

    q_normed = F.rms_norm(q, (inner_dim,), weight=q_weight, eps=eps)
    k_normed = F.rms_norm(k, (inner_dim,), weight=k_weight, eps=eps)
    q_rot = apply_rotary_emb(q_normed, q_pe, rope_type)
    k_rot = apply_rotary_emb(k_normed, q_pe if k_pe is None else k_pe, rope_type)
    return q_rot, k_rot


def fused_ltx_post_sa_residual_rms_norm(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``out = residual + value * gate`` and ``rms_norm(out, None, eps)``."""
    eligible = (
        residual.ndim == 3
        and value.shape == residual.shape
        and 0 < residual.shape[-1] <= 8192
        and residual.dtype == value.dtype == gate.dtype
        and gate.ndim == 3
        and gate.shape[0] in (1, residual.shape[0])
        and gate.shape[1] in (1, residual.shape[1])
        and gate.shape[2] == residual.shape[2]
        and _triton_cuda_eligible(residual, value, gate)
    )
    if eligible:
        try:
            return _ltx_post_sa_residual_rmsnorm_op(residual, value, gate, float(eps))
        except (ImportError, OSError):
            pass
    output = residual + value * gate
    return output, rms_norm(output, None, eps=eps)


def fused_ltx_post_sa_residual_rmsnorm_adaln(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale_q: torch.Tensor,
    shift_q: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``out = residual + value * gate`` and ``rms_norm(out, eps) * (1 + scale_q) + shift_q``."""
    eligible = (
        residual.ndim == 3
        and value.shape == residual.shape
        and 0 < residual.shape[-1] <= 8192
        and residual.dtype == value.dtype == gate.dtype == scale_q.dtype == shift_q.dtype
        and gate.ndim == 3
        and scale_q.ndim == 3
        and shift_q.ndim == 3
        and gate.shape[0] in (1, residual.shape[0])
        and gate.shape[1] in (1, residual.shape[1])
        and gate.shape[2] == residual.shape[2]
        and scale_q.shape[0] in (1, residual.shape[0])
        and scale_q.shape[1] in (1, residual.shape[1])
        and scale_q.shape[2] == residual.shape[2]
        and shift_q.shape[0] in (1, residual.shape[0])
        and shift_q.shape[1] in (1, residual.shape[1])
        and shift_q.shape[2] == residual.shape[2]
        and _triton_cuda_eligible(residual, value, gate, scale_q, shift_q)
    )
    if eligible:
        try:
            return _ltx_post_sa_residual_rmsnorm_adaln_op(residual, value, gate, scale_q, shift_q, float(eps))
        except (ImportError, OSError):
            pass
    output = residual + value * gate
    normed = rms_norm(output, None, eps=eps)
    return output, normed * (1 + scale_q) + shift_q


def fused_ltx_gated_attention(
    attention_output: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    """Apply ``2 * sigmoid(gate_logits)`` per head to ``attention_output`` in one pass."""
    inner_dim = heads * dim_head
    eligible = (
        attention_output.ndim == 3
        and gate_logits.ndim == 3
        and attention_output.dtype == gate_logits.dtype
        and attention_output.shape[0] == gate_logits.shape[0]
        and attention_output.shape[1] == gate_logits.shape[1]
        and attention_output.shape[2] == inner_dim
        and gate_logits.shape[2] == heads
        and 0 < inner_dim <= 8192
        and _triton_cuda_eligible(attention_output, gate_logits)
    )
    if eligible:
        try:
            return _ltx_gated_attention_op(attention_output, gate_logits, int(heads), int(dim_head))
        except (ImportError, OSError):
            pass
    batch, tokens, _ = attention_output.shape
    gates = 2.0 * torch.sigmoid(gate_logits)
    return (attention_output.reshape(batch, tokens, heads, dim_head) * gates.unsqueeze(-1)).reshape(
        batch, tokens, inner_dim
    )


def fused_ltx_gated_attention_fp8(
    attention_output: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    heads: int,
    dim_head: int,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse per-head sigmoid gating and rowwise FP8 quantization when CUDA/Triton eligible."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    inner_dim = heads * dim_head
    eligible = (
        attention_output.ndim == 3
        and gate_logits.ndim == 3
        and attention_output.dtype == gate_logits.dtype
        and attention_output.shape[0] == gate_logits.shape[0]
        and attention_output.shape[1] == gate_logits.shape[1]
        and attention_output.shape[2] == inner_dim
        and gate_logits.shape[2] == heads
        and 0 < inner_dim <= 8192
        and _triton_cuda_eligible(attention_output, gate_logits)
    )
    if not eligible:
        return None
    try:
        return _ltx_gated_attention_rowwise_fp8_op(
            attention_output,
            gate_logits,
            int(heads),
            int(dim_head),
            fp8_dtype == torch.float8_e5m2,
        )
    except (ImportError, OSError):
        return None


def fused_ltx_rmsnorm_scale_shift_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse unweighted RMSNorm + AdaLN scale/shift + rowwise FP8 quantization when eligible."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    eligible = (
        x.ndim == 3
        and scale.ndim == 3
        and shift.ndim == 3
        and 0 < x.shape[-1] <= 8192
        and x.dtype == scale.dtype == shift.dtype
        and scale.shape[0] in (1, x.shape[0])
        and scale.shape[1] in (1, x.shape[1])
        and scale.shape[2] == x.shape[2]
        and shift.shape[0] in (1, x.shape[0])
        and shift.shape[1] in (1, x.shape[1])
        and shift.shape[2] == x.shape[2]
        and _triton_cuda_eligible(x, scale, shift)
    )
    if not eligible:
        return None
    try:
        return _ltx_rmsnorm_scale_shift_fp8_op(x, scale, shift, float(eps), fp8_dtype == torch.float8_e5m2)
    except (ImportError, OSError):
        return None


def fused_ltx_dual_rmsnorm_scale_shift(
    x: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute two AdaLN-modulated views of ``rms_norm(x, eps)`` in one HBM read."""
    eligible = (
        x.ndim == 3
        and scale1.ndim == 3
        and shift1.ndim == 3
        and scale2.ndim == 3
        and shift2.ndim == 3
        and 0 < x.shape[-1] <= 8192
        and x.dtype == scale1.dtype == shift1.dtype == scale2.dtype == shift2.dtype
        and all(
            t.shape[0] in (1, x.shape[0]) and t.shape[1] in (1, x.shape[1]) and t.shape[2] == x.shape[2]
            for t in (scale1, shift1, scale2, shift2)
        )
        and _triton_cuda_eligible(x, scale1, shift1, scale2, shift2)
    )
    if eligible:
        try:
            return _ltx_dual_rmsnorm_scale_shift_op(x, scale1, shift1, scale2, shift2, float(eps))
        except (ImportError, OSError):
            pass
    normed = rms_norm(x, None, eps=eps)
    return normed * (1 + scale1) + shift1, normed * (1 + scale2) + shift2


def fused_ltx_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Fuse ``x * (1 + scale) + shift`` in one pass when CUDA/Triton eligible."""
    eligible = (
        x.ndim == 3
        and scale.ndim == 3
        and shift.ndim == 3
        and 0 < x.shape[-1] <= 8192
        and x.dtype == scale.dtype == shift.dtype
        and scale.shape[0] in (1, x.shape[0])
        and scale.shape[1] in (1, x.shape[1])
        and scale.shape[2] == x.shape[2]
        and shift.shape[0] in (1, x.shape[0])
        and shift.shape[1] in (1, x.shape[1])
        and shift.shape[2] == x.shape[2]
        and _triton_cuda_eligible(x, scale, shift)
    )
    if eligible:
        try:
            return _ltx_scale_shift_op(x, scale, shift)
        except (ImportError, OSError):
            pass
    return x * (1 + scale) + shift


def fused_ltx_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-hidden Q and K RMSNorm (without RoPE) in a single Triton launch."""
    features = int(q.shape[-1])
    eligible = (
        q.ndim == 3
        and k.ndim == 3
        and q.shape[-1] == k.shape[-1]
        and q_weight.shape == (features,)
        and k_weight.shape == (features,)
        and 0 < features <= 8192
        and q.dtype == k.dtype
        and _triton_cuda_eligible(q, k, q_weight, k_weight)
    )
    if eligible:
        try:
            return _ltx_qk_rmsnorm_paired_op(q, k, q_weight, k_weight, float(eps))
        except (ImportError, OSError):
            pass
    return (
        F.rms_norm(q, (features,), weight=q_weight, eps=eps),
        F.rms_norm(k, (features,), weight=k_weight, eps=eps),
    )


def fused_ltx_gelu_tanh_rowwise_fp8(
    value: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast GELU-tanh + rowwise FP8 quantization for wide LTX FFN hidden tensors."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
    if value.ndim != 2:
        raise ValueError("fused_ltx_gelu_tanh_rowwise_fp8 expects a 2D [rows, K] tensor")
    rows, cols = value.shape
    if rows == 0 or cols == 0:
        return (
            torch.empty((rows, cols), dtype=fp8_dtype, device=value.device),
            torch.empty((rows, 1), dtype=torch.float32, device=value.device),
        )
    cuda_eligible = (
        value.device.type == "cuda"
        and not bool(getattr(torch.version, "hip", None))
        and value.dtype in _FLOAT_DTYPES
        and not (torch.is_grad_enabled() and value.requires_grad)
    )
    if cuda_eligible:
        try:
            return _ltx_gelu_tanh_rowwise_fp8_op(value, fp8_dtype == torch.float8_e5m2)
        except (ImportError, OSError):
            pass
    activated = F.gelu(value, approximate="tanh")
    fp8_max = float(torch.finfo(fp8_dtype).max)
    scale = (activated.detach().float().abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (activated.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return quantized, scale.float()


class FusedLTXPreAttention:
    """Drop-in ``PreAttentionCallable`` fusing LTX QK-RMSNorm (+ optional RoPE)."""

    def __init__(self) -> None:
        self._fallback = PytorchPreAttention()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_module: nn.Module,
        mask: torch.Tensor | None,
        query_position: torch.Tensor | None,
        key_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mask
        module = cast(Any, attention_module)
        q_norm = getattr(module, "q_norm", None)
        k_norm = getattr(module, "k_norm", None)
        if query_position is None:
            if (
                isinstance(q_norm, nn.RMSNorm)
                and isinstance(k_norm, nn.RMSNorm)
                and q_norm.weight is not None
                and k_norm.weight is not None
            ):
                q_eps = float(q_norm.eps if q_norm.eps is not None else 1e-6)
                k_eps = float(k_norm.eps if k_norm.eps is not None else 1e-6)
                if q_eps == k_eps:
                    return fused_ltx_qk_rmsnorm(query, key, q_norm.weight, k_norm.weight, eps=q_eps)
            return self._fallback(query, key, attention_module, None, None, None)
        rope_type = getattr(module, "rope_type", None)
        if (
            isinstance(q_norm, nn.RMSNorm)
            and isinstance(k_norm, nn.RMSNorm)
            and q_norm.weight is not None
            and k_norm.weight is not None
            and isinstance(rope_type, LTXRopeType)
            and isinstance(query_position, tuple)
            and len(query_position) == 2
        ):
            q_eps = float(q_norm.eps if q_norm.eps is not None else 1e-6)
            k_eps = float(k_norm.eps if k_norm.eps is not None else 1e-6)
            if q_eps == k_eps:
                return fused_ltx_qk_rmsnorm_rope(
                    query,
                    key,
                    q_norm.weight,
                    k_norm.weight,
                    query_position,
                    cast(tuple[torch.Tensor, torch.Tensor] | None, key_position),
                    eps=q_eps,
                    heads=int(module.heads),
                    dim_head=int(module.dim_head),
                    rope_type=rope_type,
                )
        return self._fallback(query, key, attention_module, None, query_position, key_position)


class FusedLTXPostSAFunction:
    """Drop-in ``PostSACallable`` fusing residual-gate add + unweighted RMSNorm."""

    def __call__(
        self,
        residual: torch.Tensor,
        value: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if norm_weights is None:
            return fused_ltx_post_sa_residual_rms_norm(residual, value, gate, eps)
        output = residual + value * gate
        return output, rms_norm(output, norm_weights, eps=eps)


def ltx_fused_transformer_ops(
    base: TransformerOpsConfig | None = None,
) -> TransformerOpsConfig:
    """Return a :class:`TransformerOpsConfig` using fused LTX pre-attention and post-SA ops."""
    attn_ops = base.attention_ops if base is not None else TransformerAttentionOps()
    ada_zero = base.ada_zero_function if base is not None else PytorchAdaZeroFunction()
    return TransformerOpsConfig(
        attention_ops=TransformerAttentionOps(
            attention_function=attn_ops.attention_function,
            masked_attention_function=attn_ops.masked_attention_function,
            preattention_function=FusedLTXPreAttention(),
            gated_attention_function=attn_ops.gated_attention_function or PytorchGatedAttention(),
        ),
        ada_zero_function=ada_zero,
        post_sa_function=FusedLTXPostSAFunction(),
    )


def enable_ltx_fused_ops(model: nn.Module) -> int:
    """Install fused QK-RMSNorm+RoPE, gated-attention+FP8, post-SA+AdaLN, and dual AV ops on ``model``."""
    from .attention import Attention
    from .transformer import BasicAVTransformerBlock

    pre_attn = FusedLTXPreAttention()
    post_sa = FusedLTXPostSAFunction()
    updated_blocks = 0
    for child in model.modules():
        if isinstance(child, Attention):
            child.preattention_function = pre_attn
            child._use_fused_gated_out = True
        if isinstance(child, BasicAVTransformerBlock):
            child.post_sa_function = post_sa
            child._use_fused_residual_gate = True
            updated_blocks += 1
    return updated_blocks


def fuse_ltx_projections(
    model: nn.Module,
    *,
    fuse_self_qkv: bool = True,
    fuse_cross_kv: bool = True,
) -> dict[str, int]:
    """Fuse LTX self-attention Q/K/V and cross-attention K/V projections in place.

    Apply after loading weights and before post-load quantization so merged
    projections convert into single wide ``Float8Linear`` layers.
    """
    from .attention import Attention

    fused_qkv = 0
    fused_kv = 0
    for child in model.modules():
        if not isinstance(child, Attention):
            continue
        if child.is_self_attention:
            if fuse_self_qkv and child.fuse_qkv():
                fused_qkv += 1
        elif fuse_cross_kv and child.fuse_kv():
            fused_kv += 1
    return {"fused_qkv": fused_qkv, "fused_kv": fused_kv}


def enable_ltx_fused_fp8_ffn(model: nn.Module) -> int:
    """Enable fused GELU-tanh + rowwise FP8 quantization on eligible LTX FFNs."""
    from .feed_forward import FeedForward

    enabled = 0
    for child in model.modules():
        if isinstance(child, FeedForward) and child.enable_fused_fp8(True):
            enabled += 1
    return enabled


def enable_ltx_fp8_quantization(
    model: nn.Module,
    *,
    exclude: Iterable[str] = LTX_DEFAULT_FP8_EXCLUDES,
    min_features: int = 1024,
    scaling: str = "rowwise",
    use_fast_accum: bool = True,
    keep_dense_fallback: bool = True,
    fuse_fp8_ffn: bool = True,
) -> dict[str, int]:
    """Apply LTX-calibrated rowwise FP8 W8A8 quantization and optional FFN fusion."""
    from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8

    replaced = replace_linear_with_float8(
        model,
        min_features=min_features,
        keep_dense_fallback=keep_dense_fallback,
        exclude=tuple(exclude),
        scaling=scaling,
        use_fast_accum=use_fast_accum,
    )
    fused_ffn = enable_ltx_fused_fp8_ffn(model) if fuse_fp8_ffn and scaling in {"rowwise", "auto"} else 0
    return {"replaced_linears": replaced, "fused_fp8_ffn": fused_ffn}


__all__ = [
    "LTX_AUDIO_FP8_EXCLUDES",
    "LTX_DEFAULT_FP8_EXCLUDES",
    "FusedLTXPostSAFunction",
    "FusedLTXPreAttention",
    "enable_ltx_fp8_quantization",
    "enable_ltx_fused_fp8_ffn",
    "enable_ltx_fused_ops",
    "fuse_ltx_projections",
    "fused_ltx_dual_rmsnorm_scale_shift",
    "fused_ltx_gated_attention",
    "fused_ltx_gated_attention_fp8",
    "fused_ltx_gelu_tanh_rowwise_fp8",
    "fused_ltx_post_sa_residual_rms_norm",
    "fused_ltx_post_sa_residual_rmsnorm_adaln",
    "fused_ltx_qk_rmsnorm",
    "fused_ltx_qk_rmsnorm_rope",
    "fused_ltx_rmsnorm_scale_shift_fp8",
    "fused_ltx_scale_shift",
    "ltx_fused_transformer_ops",
]
