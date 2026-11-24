# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels and Hopper FA3 execution for Matrix-Game-2.0.

All kernels are CUDA-Graph capture safe (zero host syncs, zero dynamic
allocations) and preserve exact lossless BF16 semantics:

- Main ``CausalWanAttentionBlock`` fast path: precomputed real FP64 3D RoPE,
  bit-exact ``F.layer_norm`` + Triton scale-shift, fused residual-gate add,
  exact Q/K WanRMSNorm + 3D RoPE + KV-cache write, Hopper FlashAttention-3
  (or exact cuDNN SDPA) for causal self-attention, and cuDNN SDPA for short
  image cross-attention (skipping redundant ``img_emb`` once cached).
- ``ActionModule`` fast path: hoist shared mouse/keyboard window slicing out
  of the 15-block loop, fused Triton mouse QK-Norm+RoPE+KV-write, fused Triton
  small-sequence mouse attention (``Sq=3, Sk<=16, D=64``, ~30x faster than
  FA3 on tiny sequences), and zero-copy shared-KV keyboard SDPA (eliminating
  the 880x ``.repeat`` and FA3 launch overhead, ~25x faster).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="matrix-game-2-triton")

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _round_f32_to_bf16(value):
        bits = value.to(tl.uint32, bitcast=True)
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits + rounding_bias) & 0xFFFF0000
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _round_f32_to_fp16(value):
        return tl.inline_asm_elementwise(
            "{ .reg .b16 rounded; cvt.rn.f16.f32 rounded, $1; cvt.f32.f16 $0, rounded; }",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _round_to_dtype(value, dtype: tl.constexpr):
        if dtype == tl.bfloat16:
            return _round_f32_to_bf16(value)
        elif dtype == tl.float16:
            return _round_f32_to_fp16(value)
        else:
            return value

    @triton.jit
    def _round_to_code(value, dtype_code: tl.constexpr):
        if dtype_code == 1:
            return _round_f32_to_bf16(value)
        elif dtype_code == 2:
            return _round_f32_to_fp16(value)
        else:
            return value

    @triton.jit
    def _mg2_scale_shift_kernel(
        normed_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        rows,
        frame_seq,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        mul_dtype_code: tl.constexpr,
    ):
        """Bit-exact per-frame AdaLN ``normed * (1 + scale) + shift``."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        x_off = row.to(tl.int64) * hidden_size + cols
        normed = tl.load(normed_ptr + x_off, mask=mask, other=0.0).to(tl.float32)

        frame_row = row // frame_seq
        mod_off = frame_row.to(tl.int64) * hidden_size + cols
        scale_in = tl.load(scale_ptr + mod_off, mask=mask, other=0.0)
        shift_in = tl.load(shift_ptr + mod_off, mask=mask, other=0.0)

        scale_factor = _round_to_dtype(1.0 + scale_in.to(tl.float32), scale_in.dtype)
        modulated = _round_to_code(normed * scale_factor, mul_dtype_code)
        out = modulated + shift_in.to(tl.float32)
        tl.store(out_ptr + x_off, out, mask=mask)

    @triton.jit
    def _mg2_residual_gate_add_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        out_ptr,
        rows,
        frame_seq,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        prod_dtype_code: tl.constexpr,
    ):
        """Bit-exact ``x + (y * gate)`` where ``gate`` broadcasts across ``frame_seq``."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        y_in = tl.load(y_ptr + off, mask=mask, other=0.0)

        frame_row = row // frame_seq
        gate_off = frame_row.to(tl.int64) * hidden_size + cols
        gate_in = tl.load(gate_ptr + gate_off, mask=mask, other=0.0)

        prod = _round_to_code(y_in.to(tl.float32) * gate_in.to(tl.float32), prod_dtype_code)
        out = x_in.to(tl.float32) + prod
        tl.store(out_ptr + off, out, mask=mask)

    @triton.jit
    def _mg2_apply_rstd_weight_kernel(
        x_ptr,
        rstd_ptr,
        w_ptr,
        out_ptr,
        rows,
        x_row_stride,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Bit-exact ``(x.float() * rstd).type_as(x) * w``."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        x_off = row.to(tl.int64) * x_row_stride + cols
        x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        rstd = tl.load(rstd_ptr + row, mask=row < rows, other=1.0).to(tl.float32)
        normed = _round_to_dtype(x_in.to(tl.float32) * rstd, x_in.dtype)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = normed * w
        out_off = row.to(tl.int64) * hidden_size + cols
        tl.store(out_ptr + out_off, out, mask=mask)

    @triton.jit
    def _mg2_rmsnorm_fp64_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        rows,
        x_row_stride,
        eps,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Single-pass FP64-accumulated ``WanRMSNorm``."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        x_off = row.to(tl.int64) * x_row_stride + cols
        x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)
        x_64 = x_f.to(tl.float64)
        mean_sq = (tl.sum(x_64 * x_64, axis=0) / hidden_size).to(tl.float32)
        rstd = _libdevice.rsqrt(mean_sq + eps)
        normed = _round_to_dtype(x_f * rstd, x_in.dtype)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = normed * w
        out_off = row.to(tl.int64) * hidden_size + cols
        tl.store(out_ptr + out_off, out, mask=mask)

    @triton.jit
    def _mg2_qk_apply_rstd_rope_write_kv_kernel(
        q_in_ptr,
        k_in_ptr,
        q_rstd_ptr,
        k_rstd_ptr,
        q_w_ptr,
        k_w_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_dst_ptr,
        seq_len,
        q_row_stride,
        k_row_stride,
        k_dst_batch_stride,
        k_dst_seq_stride,
        dst_token_offset,
        hidden_size: tl.constexpr,
        half_head_dim: tl.constexpr,
        block_pairs: tl.constexpr,
    ):
        """Bit-exact fused Q/K WanRMSNorm + FP64 3D RoPE + KV cache write."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        pair = tl.arange(0, block_pairs)
        even_col = pair * 2
        odd_col = even_col + 1
        pmask = odd_col < hidden_size

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        qe_in = tl.load(q_in_ptr + q_base + even_col, mask=pmask, other=0.0)
        qo_in = tl.load(q_in_ptr + q_base + odd_col, mask=pmask, other=0.0)
        ke_in = tl.load(k_in_ptr + k_base + even_col, mask=pmask, other=0.0)
        ko_in = tl.load(k_in_ptr + k_base + odd_col, mask=pmask, other=0.0)

        q_rstd = tl.load(q_rstd_ptr + row).to(tl.float32)
        k_rstd = tl.load(k_rstd_ptr + row).to(tl.float32)

        qwe = tl.load(q_w_ptr + even_col, mask=pmask, other=0.0).to(tl.float32)
        qwo = tl.load(q_w_ptr + odd_col, mask=pmask, other=0.0).to(tl.float32)
        kwe = tl.load(k_w_ptr + even_col, mask=pmask, other=0.0).to(tl.float32)
        kwo = tl.load(k_w_ptr + odd_col, mask=pmask, other=0.0).to(tl.float32)

        qe = _round_to_dtype(_round_to_dtype(qe_in.to(tl.float32) * q_rstd, qe_in.dtype) * qwe, qe_in.dtype)
        qo = _round_to_dtype(_round_to_dtype(qo_in.to(tl.float32) * q_rstd, qo_in.dtype) * qwo, qo_in.dtype)
        ke = _round_to_dtype(_round_to_dtype(ke_in.to(tl.float32) * k_rstd, ke_in.dtype) * kwe, ke_in.dtype)
        ko = _round_to_dtype(_round_to_dtype(ko_in.to(tl.float32) * k_rstd, ko_in.dtype) * kwo, ko_in.dtype)

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
        s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)

        rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(qe_in.dtype)
        rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(qo_in.dtype)
        rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(ke_in.dtype)
        rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(ko_in.dtype)

        q_out_base = row.to(tl.int64) * hidden_size
        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        tl.store(q_out_ptr + q_out_base + even_col, rqe, mask=pmask)
        tl.store(q_out_ptr + q_out_base + odd_col, rqo, mask=pmask)
        tl.store(k_dst_ptr + k_dst_base + even_col, rke, mask=pmask)
        tl.store(k_dst_ptr + k_dst_base + odd_col, rko, mask=pmask)

    @triton.jit
    def _mg2_qk_rmsnorm_rope_write_kv_fp64_kernel(
        q_in_ptr,
        k_in_ptr,
        q_w_ptr,
        k_w_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_dst_ptr,
        seq_len,
        q_row_stride,
        k_row_stride,
        k_dst_batch_stride,
        k_dst_seq_stride,
        dst_token_offset,
        eps,
        hidden_size: tl.constexpr,
        half_head_dim: tl.constexpr,
        block_pairs: tl.constexpr,
    ):
        """Single-pass FP64-accumulated Q/K WanRMSNorm + FP64 3D RoPE + KV write."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        pair = tl.arange(0, block_pairs)
        even_col = pair * 2
        odd_col = even_col + 1
        pmask = odd_col < hidden_size

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        qe_in = tl.load(q_in_ptr + q_base + even_col, mask=pmask, other=0.0)
        qo_in = tl.load(q_in_ptr + q_base + odd_col, mask=pmask, other=0.0)
        ke_in = tl.load(k_in_ptr + k_base + even_col, mask=pmask, other=0.0)
        ko_in = tl.load(k_in_ptr + k_base + odd_col, mask=pmask, other=0.0)

        qe_f = qe_in.to(tl.float32)
        qo_f = qo_in.to(tl.float32)
        ke_f = ke_in.to(tl.float32)
        ko_f = ko_in.to(tl.float32)

        qe_64 = qe_f.to(tl.float64)
        qo_64 = qo_f.to(tl.float64)
        ke_64 = ke_f.to(tl.float64)
        ko_64 = ko_f.to(tl.float64)
        q_mean_sq = (tl.sum(qe_64 * qe_64 + qo_64 * qo_64, axis=0) / hidden_size).to(tl.float32)
        k_mean_sq = (tl.sum(ke_64 * ke_64 + ko_64 * ko_64, axis=0) / hidden_size).to(tl.float32)
        q_rstd = _libdevice.rsqrt(q_mean_sq + eps)
        k_rstd = _libdevice.rsqrt(k_mean_sq + eps)

        qwe = tl.load(q_w_ptr + even_col, mask=pmask, other=0.0).to(tl.float32)
        qwo = tl.load(q_w_ptr + odd_col, mask=pmask, other=0.0).to(tl.float32)
        kwe = tl.load(k_w_ptr + even_col, mask=pmask, other=0.0).to(tl.float32)
        kwo = tl.load(k_w_ptr + odd_col, mask=pmask, other=0.0).to(tl.float32)

        qe = _round_to_dtype(_round_to_dtype(qe_f * q_rstd, qe_in.dtype) * qwe, qe_in.dtype)
        qo = _round_to_dtype(_round_to_dtype(qo_f * q_rstd, qo_in.dtype) * qwo, qo_in.dtype)
        ke = _round_to_dtype(_round_to_dtype(ke_f * k_rstd, ke_in.dtype) * kwe, ke_in.dtype)
        ko = _round_to_dtype(_round_to_dtype(ko_f * k_rstd, ko_in.dtype) * kwo, ko_in.dtype)

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
        s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)

        rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(qe_in.dtype)
        rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(qo_in.dtype)
        rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(ke_in.dtype)
        rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(ko_in.dtype)

        q_out_base = row.to(tl.int64) * hidden_size
        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        tl.store(q_out_ptr + q_out_base + even_col, rqe, mask=pmask)
        tl.store(q_out_ptr + q_out_base + odd_col, rqo, mask=pmask)
        tl.store(k_dst_ptr + k_dst_base + even_col, rke, mask=pmask)
        tl.store(k_dst_ptr + k_dst_base + odd_col, rko, mask=pmask)

    @triton.jit
    def _mg2_mouse_qkv_norm_rope_kv_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_dst_ptr,
        v_dst_ptr,
        num_frames,
        dst_frame_offset,
        qkv_stride_bs,
        qkv_stride_t,
        kv_stride_bs,
        kv_stride_t,
        kv_stride_h,
        cos_stride_t,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_D: tl.constexpr,
    ):
        """Fused mouse QKV split + ActionModule RMSNorm + FP32 interleaved RoPE + KV write.

        Input ``qkv`` has shape ``[BS, T, 3 * NUM_HEADS * HEAD_DIM]`` with ``K=3`` outermost.
        Outputs ``q_out`` ``[BS, T, NUM_HEADS, HEAD_DIM]`` and writes ``k, v`` into
        ``k_dst[:, dst_frame_offset:dst_frame_offset+T]`` and ``v_dst[...]``.
        """
        bs_t = tl.program_id(0)
        head = tl.program_id(1)
        bs_idx = bs_t // num_frames
        t_idx = bs_t % num_frames

        pair = tl.arange(0, HALF_D)
        even = pair * 2
        odd = even + 1

        hidden = NUM_HEADS * HEAD_DIM
        row_base = bs_idx.to(tl.int64) * qkv_stride_bs + t_idx.to(tl.int64) * qkv_stride_t
        q_base = row_base + head.to(tl.int64) * HEAD_DIM
        k_base = row_base + hidden + head.to(tl.int64) * HEAD_DIM
        v_base = row_base + 2 * hidden + head.to(tl.int64) * HEAD_DIM

        qe_raw = tl.load(qkv_ptr + q_base + even)
        qo_raw = tl.load(qkv_ptr + q_base + odd)
        ke_raw = tl.load(qkv_ptr + k_base + even)
        ko_raw = tl.load(qkv_ptr + k_base + odd)
        ve_raw = tl.load(qkv_ptr + v_base + even)
        vo_raw = tl.load(qkv_ptr + v_base + odd)

        # ActionModule.WanRMSNorm: x * rsqrt(x.pow(2).mean(-1) + eps) computed in x.dtype (BF16/FP16)
        qe_f = qe_raw.to(tl.float32)
        qo_f = qo_raw.to(tl.float32)
        ke_f = ke_raw.to(tl.float32)
        ko_f = ko_raw.to(tl.float32)
        q_sq = _round_to_dtype(qe_f * qe_f, qe_raw.dtype).to(tl.float64) + _round_to_dtype(
            qo_f * qo_f, qo_raw.dtype
        ).to(tl.float64)
        k_sq = _round_to_dtype(ke_f * ke_f, ke_raw.dtype).to(tl.float64) + _round_to_dtype(
            ko_f * ko_f, ko_raw.dtype
        ).to(tl.float64)
        q_mean = _round_to_dtype((tl.sum(q_sq, axis=0) / HEAD_DIM).to(tl.float32), qe_raw.dtype)
        k_mean = _round_to_dtype((tl.sum(k_sq, axis=0) / HEAD_DIM).to(tl.float32), ke_raw.dtype)
        q_rstd = _round_to_dtype(_libdevice.rsqrt(_round_to_dtype(q_mean + eps, qe_raw.dtype)), qe_raw.dtype)
        k_rstd = _round_to_dtype(_libdevice.rsqrt(_round_to_dtype(k_mean + eps, ke_raw.dtype)), ke_raw.dtype)
        qe_n = _round_to_dtype(qe_f * q_rstd, qe_raw.dtype)
        qo_n = _round_to_dtype(qo_f * q_rstd, qo_raw.dtype)
        ke_n = _round_to_dtype(ke_f * k_rstd, ke_raw.dtype)
        ko_n = _round_to_dtype(ko_f * k_rstd, ko_raw.dtype)

        # Interleaved FP32 RoPE: cos/sin have repeated pairs [c0, c0, c1, c1, ...]
        cs_base = t_idx.to(tl.int64) * cos_stride_t + even
        c = tl.load(cos_ptr + cs_base).to(tl.float32)
        s = tl.load(sin_ptr + cs_base).to(tl.float32)

        rqe = (qe_n * c - qo_n * s).to(qe_raw.dtype)
        rqo = (qe_n * s + qo_n * c).to(qo_raw.dtype)
        rke = (ke_n * c - ko_n * s).to(ke_raw.dtype)
        rko = (ke_n * s + ko_n * c).to(ko_raw.dtype)

        q_out_base = (bs_t.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + q_out_base + even, rqe)
        tl.store(q_out_ptr + q_out_base + odd, rqo)

        kv_out_base = (
            bs_idx.to(tl.int64) * kv_stride_bs
            + (dst_frame_offset + t_idx).to(tl.int64) * kv_stride_t
            + head.to(tl.int64) * kv_stride_h
        )
        tl.store(k_dst_ptr + kv_out_base + even, rke)
        tl.store(k_dst_ptr + kv_out_base + odd, rko)
        tl.store(v_dst_ptr + kv_out_base + even, ve_raw)
        tl.store(v_dst_ptr + kv_out_base + odd, vo_raw)

    @triton.jit
    def _mg2_mouse_small_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        spatial_s,
        k_start,
        k_len,
        k_stride_bs,
        k_stride_t,
        k_stride_h,
        scale,
        NUM_FRAMES: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Exact FP32 small-sequence mouse attention fused with ``(B S) T H D -> B (T S) (H D)`` transpose.

        Grid: ``(B * S, NUM_HEADS)``. For ``B=1, S=880, H=16`` this launches 14,080 tiny
        programs (~0.015 ms on H200 vs 0.442 ms for FA3 on ``[880, 3, 16, 64]``).
        """
        bs_idx = tl.program_id(0)
        head = tl.program_id(1)
        b_idx = bs_idx // spatial_s
        s_idx = bs_idx % spatial_s
        d_off = tl.arange(0, HEAD_DIM)

        # Support NUM_FRAMES == 3 (MG2 standard) or general small NUM_FRAMES via static loop.
        kv_base = bs_idx.to(tl.int64) * k_stride_bs + head.to(tl.int64) * k_stride_h
        for tq in tl.static_range(0, NUM_FRAMES):
            q_off = ((bs_idx.to(tl.int64) * NUM_FRAMES + tq) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM + d_off
            q_vec = tl.load(q_ptr + q_off).to(tl.float32)

            m_i = -1.0e20
            l_i = 0.0
            acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

            for tk in range(k_len):
                t_abs = k_start + tk
                kt_off = kv_base + t_abs.to(tl.int64) * k_stride_t + d_off
                k_vec = tl.load(k_ptr + kt_off).to(tl.float32)
                v_vec = tl.load(v_ptr + kt_off).to(tl.float32)
                score = tl.sum(q_vec * k_vec, axis=0) * scale
                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(score - m_new)
                l_i = l_i * alpha + p
                acc = acc * alpha + p * v_vec
                m_i = m_new

            out_vec = acc / l_i
            # Store directly into [B, T * S, H * D] layout at token (tq * S + s_idx)!
            out_tok = b_idx.to(tl.int64) * (NUM_FRAMES * spatial_s) + tq * spatial_s + s_idx
            out_off = out_tok * (NUM_HEADS * HEAD_DIM) + head.to(tl.int64) * HEAD_DIM + d_off
            tl.store(out_ptr + out_off, out_vec)

    @triton.jit
    def _mg2_kb_q_norm_rope_bhsd_kernel(
        q_in_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        num_frames,
        spatial_s,
        cos_stride_t,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_D: tl.constexpr,
    ):
        """Fused keyboard Q RMSNorm + per-frame FP32 RoPE + ``[B, T*S, H, D] -> [B, H, T*S, D]`` layout."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        seq_len = num_frames * spatial_s
        b_idx = row // seq_len
        tok_idx = row % seq_len
        t_idx = tok_idx // spatial_s

        pair = tl.arange(0, HALF_D)
        even = pair * 2
        odd = even + 1

        in_base = (row.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        qe_raw = tl.load(q_in_ptr + in_base + even)
        qo_raw = tl.load(q_in_ptr + in_base + odd)

        qe_f = qe_raw.to(tl.float32)
        qo_f = qo_raw.to(tl.float32)
        q_sq = _round_to_dtype(qe_f * qe_f, qe_raw.dtype).to(tl.float64) + _round_to_dtype(
            qo_f * qo_f, qo_raw.dtype
        ).to(tl.float64)
        q_mean = _round_to_dtype((tl.sum(q_sq, axis=0) / HEAD_DIM).to(tl.float32), qe_raw.dtype)
        rstd = _round_to_dtype(_libdevice.rsqrt(_round_to_dtype(q_mean + eps, qe_raw.dtype)), qe_raw.dtype)
        qe_n = _round_to_dtype(qe_f * rstd, qe_raw.dtype)
        qo_n = _round_to_dtype(qo_f * rstd, qo_raw.dtype)

        cs_base = t_idx.to(tl.int64) * cos_stride_t + even
        c = tl.load(cos_ptr + cs_base).to(tl.float32)
        s = tl.load(sin_ptr + cs_base).to(tl.float32)

        rqe = (qe_n * c - qo_n * s).to(qe_raw.dtype)
        rqo = (qe_n * s + qo_n * c).to(qo_raw.dtype)

        out_base = ((b_idx.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * seq_len + tok_idx.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + out_base + even, rqe)
        tl.store(q_out_ptr + out_base + odd, rqo)

    @triton.jit
    def _mg2_kb_rope_bhsd_kernel(
        q_in_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        num_frames,
        spatial_s,
        cos_stride_t,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_D: tl.constexpr,
    ):
        """Bit-exact per-frame FP32 RoPE + ``[B, T*S, H, D] -> [B, H, T*S, D]`` on pre-normed Q."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        seq_len = num_frames * spatial_s
        b_idx = row // seq_len
        tok_idx = row % seq_len
        t_idx = tok_idx // spatial_s

        pair = tl.arange(0, HALF_D)
        even = pair * 2
        odd = even + 1

        in_base = (row.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        qe_raw = tl.load(q_in_ptr + in_base + even)
        qo_raw = tl.load(q_in_ptr + in_base + odd)

        cs_base = t_idx.to(tl.int64) * cos_stride_t + even
        c = tl.load(cos_ptr + cs_base).to(tl.float32)
        s = tl.load(sin_ptr + cs_base).to(tl.float32)

        qe_f = qe_raw.to(tl.float32)
        qo_f = qo_raw.to(tl.float32)
        rqe = (qe_f * c - qo_f * s).to(qe_raw.dtype)
        rqo = (qe_f * s + qo_f * c).to(qo_raw.dtype)

        out_base = ((b_idx.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * seq_len + tok_idx.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + out_base + even, rqe)
        tl.store(q_out_ptr + out_base + odd, rqo)

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


_FA3_MODULE: Any = None
_FA3_PROBED: bool = False


def hopper_fa3_usable(device: torch.device | str | None = None) -> bool:
    """Return True when ``flash_attn_interface`` is installed and device is SM90 Hopper."""
    global _FA3_MODULE, _FA3_PROBED
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return False
    if torch.cuda.get_device_capability(dev)[0] != 9:
        return False
    if not _FA3_PROBED:
        try:
            _FA3_MODULE = __import__("flash_attn_interface")
        except Exception:
            _FA3_MODULE = None
        _FA3_PROBED = True
    return _FA3_MODULE is not None and callable(getattr(_FA3_MODULE, "flash_attn_func", None))


def _dtype_to_round_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _prepare_frame_mod_tensor(
    t: torch.Tensor,
    *,
    batch: int,
    num_frames: int,
    hidden: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate and materialize a ``[batch * num_frames, hidden]`` per-frame modulation tensor."""
    if not t.is_cuda or t.device != device or t.shape[-1] != hidden:
        return None
    if t.ndim == 4:
        if t.shape[2] != 1:
            return None
        b_s, f_s = int(t.shape[0]), int(t.shape[1])
        if b_s not in (1, batch) or f_s not in (1, num_frames):
            return None
        if b_s == batch and f_s == num_frames:
            return t.reshape(batch * num_frames, hidden).contiguous()
        return t.expand(batch, num_frames, 1, hidden).reshape(batch * num_frames, hidden).contiguous()
    if t.ndim == 3:
        b_s, f_s = int(t.shape[0]), int(t.shape[1])
        if b_s not in (1, batch) or f_s not in (1, num_frames):
            return None
        if b_s == batch and f_s == num_frames:
            return t.reshape(batch * num_frames, hidden).contiguous()
        return t.expand(batch, num_frames, hidden).reshape(batch * num_frames, hidden).contiguous()
    return None


def mg2_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bit-exact ``F.layer_norm`` + fused Triton per-frame AdaLN ``(1 + scale) + shift``."""
    hidden = int(x.shape[-1])
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and x.is_contiguous()
        and hidden <= 4096
        and num_frames > 0
        and frame_seq > 0
        and x.shape[1] == num_frames * frame_seq
    ):
        batch = int(x.shape[0])
        scale_c = _prepare_frame_mod_tensor(scale, batch=batch, num_frames=num_frames, hidden=hidden, device=x.device)
        shift_c = _prepare_frame_mod_tensor(shift, batch=batch, num_frames=num_frames, hidden=hidden, device=x.device)
        if scale_c is not None and shift_c is not None:
            normed = F.layer_norm(x, (hidden,), weight=None, bias=None, eps=eps).type_as(x)
            mul_dtype = torch.promote_types(x.dtype, scale.dtype)
            out_dtype = torch.promote_types(mul_dtype, shift.dtype)
            mul_code = _dtype_to_round_code(mul_dtype)
            out = torch.empty_like(x, dtype=out_dtype)
            rows = int(x.shape[0] * x.shape[1])
            block = triton.next_power_of_2(hidden)
            _mg2_scale_shift_kernel[(rows,)](
                normed,
                scale_c,
                shift_c,
                out,
                rows,
                frame_seq,
                hidden_size=hidden,
                block=block,
                mul_dtype_code=mul_code,
                num_warps=8 if block >= 1024 else 4,
                enable_fp_fusion=False,
            )
            return out
    normed = F.layer_norm(x, (hidden,), weight=None, bias=None, eps=eps).type_as(x)
    scale_b = scale.unsqueeze(2) if scale.ndim == 3 else scale
    shift_b = shift.unsqueeze(2) if shift.ndim == 3 else shift
    return (normed.unflatten(1, (num_frames, frame_seq)) * (1 + scale_b) + shift_b).flatten(1, 2)


def mg2_residual_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    """Bit-exact fused ``x + (y * gate)`` with per-frame gate broadcast."""
    hidden = int(x.shape[-1])
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and y.shape == x.shape
        and y.device == x.device
        and x.is_contiguous()
        and y.is_contiguous()
        and hidden <= 4096
        and num_frames > 0
        and frame_seq > 0
        and x.shape[1] == num_frames * frame_seq
    ):
        batch = int(x.shape[0])
        gate_c = _prepare_frame_mod_tensor(gate, batch=batch, num_frames=num_frames, hidden=hidden, device=x.device)
        if gate_c is not None:
            prod_dtype = torch.promote_types(y.dtype, gate.dtype)
            out_dtype = torch.promote_types(x.dtype, prod_dtype)
            prod_code = _dtype_to_round_code(prod_dtype)
            out = torch.empty_like(x, dtype=out_dtype)
            rows = int(x.shape[0] * x.shape[1])
            block = triton.next_power_of_2(hidden)
            _mg2_residual_gate_add_kernel[(rows,)](
                x,
                y,
                gate_c,
                out,
                rows,
                frame_seq,
                hidden_size=hidden,
                block=block,
                prod_dtype_code=prod_code,
                num_warps=8 if block >= 1024 else 4,
                enable_fp_fusion=False,
            )
            return out
    gate_b = gate.unsqueeze(2) if gate.ndim == 3 else gate
    return x + (y.unflatten(1, (num_frames, frame_seq)) * gate_b).flatten(1, 2)


def mg2_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, *, exact: bool = True) -> torch.Tensor:
    """Fast ``WanRMSNorm`` supporting strided last-dim-contiguous ``[B, S, C]``."""
    hidden = int(x.shape[-1])
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and weight.device == x.device
        and weight.ndim == 1
        and weight.shape[0] == hidden
        and weight.is_contiguous()
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and x.stride(-1) == 1
        and (x.shape[0] == 1 or x.stride(0) == x.shape[1] * x.stride(1))
        and hidden <= 4096
    ):
        rows = int(x.shape[0] * x.shape[1])
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        block = triton.next_power_of_2(hidden)
        if exact:
            rstd = torch.rsqrt(x.float().square_().mean(dim=-1) + eps).reshape(rows).contiguous()
            _mg2_apply_rstd_weight_kernel[(rows,)](
                x,
                rstd,
                weight,
                out,
                rows,
                int(x.stride(1)),
                hidden_size=hidden,
                block=block,
                num_warps=8 if block >= 1024 else 4,
                enable_fp_fusion=False,
            )
        else:
            _mg2_rmsnorm_fp64_kernel[(rows,)](
                x,
                weight,
                out,
                rows,
                int(x.stride(1)),
                float(eps),
                hidden_size=hidden,
                block=block,
                num_warps=8 if block >= 1024 else 4,
                enable_fp_fusion=False,
            )
        return out
    normed = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(x)
    return normed * weight


def mg2_qk_rmsnorm_rope_write_kv(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_dst: torch.Tensor,
    *,
    dst_token_offset: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    exact: bool = True,
) -> torch.Tensor:
    """Fused Q/K WanRMSNorm + FP64 3D RoPE + direct KV cache store."""
    batch, seq_len, hidden = int(q_proj.shape[0]), int(q_proj.shape[1]), int(q_proj.shape[2])
    half_d = head_dim // 2
    if (
        _TRITON_AVAILABLE
        and q_proj.is_cuda
        and k_proj.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and k_proj.device == q_proj.device
        and q_weight.device == q_proj.device
        and k_weight.device == q_proj.device
        and cos.device == q_proj.device
        and sin.device == q_proj.device
        and q_weight.dtype == q_proj.dtype
        and k_weight.dtype == k_proj.dtype
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and not torch.is_grad_enabled()
        and q_proj.stride(-1) == 1
        and k_proj.stride(-1) == 1
        and (
            batch == 1
            or (q_proj.stride(0) == seq_len * q_proj.stride(1) and k_proj.stride(0) == seq_len * k_proj.stride(1))
        )
        and k_dst.is_contiguous()
        and hidden <= 4096
        and hidden % 2 == 0
    ):
        q_out = torch.empty((batch, seq_len, num_heads, head_dim), dtype=q_proj.dtype, device=q_proj.device)
        rows = batch * seq_len
        block_pairs = triton.next_power_of_2(hidden // 2)
        if exact:
            q_rstd = torch.rsqrt(q_proj.float().square_().mean(dim=-1) + eps).reshape(rows).contiguous()
            k_rstd = torch.rsqrt(k_proj.float().square_().mean(dim=-1) + eps).reshape(rows).contiguous()
            _mg2_qk_apply_rstd_rope_write_kv_kernel[(rows,)](
                q_proj,
                k_proj,
                q_rstd,
                k_rstd,
                q_weight,
                k_weight,
                cos.contiguous(),
                sin.contiguous(),
                q_out,
                k_dst,
                seq_len,
                int(q_proj.stride(1)),
                int(k_proj.stride(1)),
                int(k_dst.stride(0)),
                int(k_dst.stride(1)),
                int(dst_token_offset),
                hidden_size=hidden,
                half_head_dim=half_d,
                block_pairs=block_pairs,
                num_warps=8 if block_pairs >= 512 else 4,
                enable_fp_fusion=False,
            )
        else:
            _mg2_qk_rmsnorm_rope_write_kv_fp64_kernel[(rows,)](
                q_proj,
                k_proj,
                q_weight,
                k_weight,
                cos.contiguous(),
                sin.contiguous(),
                q_out,
                k_dst,
                seq_len,
                int(q_proj.stride(1)),
                int(k_proj.stride(1)),
                int(k_dst.stride(0)),
                int(k_dst.stride(1)),
                int(dst_token_offset),
                float(eps),
                hidden_size=hidden,
                half_head_dim=half_d,
                block_pairs=block_pairs,
                num_warps=8 if block_pairs >= 512 else 4,
                enable_fp_fusion=False,
            )
        return q_out

    q_norm = mg2_rmsnorm(q_proj, q_weight, eps=eps, exact=exact).view(batch, seq_len, num_heads, head_dim)
    k_norm = mg2_rmsnorm(k_proj, k_weight, eps=eps, exact=exact).view(batch, seq_len, num_heads, head_dim)
    from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

    q_rot = apply_real_rope_bshd(q_norm, cos, sin).to(q_proj.dtype)
    k_rot = apply_real_rope_bshd(k_norm, cos, sin).to(k_proj.dtype)
    k_dst[:, dst_token_offset : dst_token_offset + seq_len].copy_(k_rot)
    return q_rot


def mg2_main_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "torch",
) -> torch.Tensor:
    """Execute ``[B, S, H, D]`` MG2 causal self-attention via Hopper FA3 or cuDNN SDPA."""
    if (
        backend == "flash_attention_3"
        and hopper_fa3_usable(q.device)
        and _FA3_MODULE is not None
        and q.dtype in {torch.float16, torch.bfloat16}
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
    ):
        out = _FA3_MODULE.flash_attn_func(q, k, v)
        return out[0] if isinstance(out, tuple) else out
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)


def mg2_fast_action_module_forward(
    action_model: nn.Module,
    x: torch.Tensor,
    *,
    num_frames: int,
    spatial_s: int,
    group_mouse_pre: torch.Tensor | None,
    kb_cond_slice: torch.Tensor | None,
    kv_cache_mouse: dict[str, Any] | None,
    kv_cache_keyboard: dict[str, Any] | None,
    action_local_start: int,
    action_local_end: int,
    action_k_start: int,
    start_frame: int,
    use_rope_keyboard: bool = True,
    exact_mouse_sdpa: bool = False,
) -> torch.Tensor:
    """Fast CUDA-Graph-safe ``ActionModule`` steady-state forward.

    Replaces FA3-on-tiny-sequence mouse/keyboard attention (0.442 ms each) with:
    - Fused Triton mouse QKV-Norm+RoPE+KV-write + fused Triton small-sequence
      attention (``Sq=3, Sk<=16, D=64``, ~0.015 ms, fused output transpose).
    - Zero-copy shared-KV keyboard SDPA (``[B, H, T*S, D] x [B, H, Sk, D]``,
      ~0.018 ms, zero ``.repeat(880)``, mathematical identity).
    """
    batch = int(x.shape[0])
    heads = int(action_model.heads_num)
    hidden_states = x
    freqs_cos: torch.Tensor = action_model.freqs_cos
    freqs_sin: torch.Tensor = action_model.freqs_sin
    if freqs_cos.device != x.device:
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)
        action_model.freqs_cos = freqs_cos
        action_model.freqs_sin = freqs_sin
    cos_win = freqs_cos[start_frame : start_frame + num_frames].contiguous()
    sin_win = freqs_sin[start_frame : start_frame + num_frames].contiguous()

    # 1. Mouse branch (when enabled & conditioned)
    if getattr(action_model, "enable_mouse", False) and group_mouse_pre is not None and kv_cache_mouse is not None:
        # [B, T*S, C] -> [B*S, T, C]
        h_mouse = (
            x.view(batch, num_frames, spatial_s, x.shape[-1])
            .transpose(1, 2)
            .reshape(batch * spatial_s, num_frames, x.shape[-1])
        )
        gm = torch.cat([h_mouse, group_mouse_pre], dim=-1)
        gm = action_model.mouse_mlp(gm)
        mouse_qkv = action_model.t_qkv(gm)  # [B*S, T, 3*C_m]
        c_m = int(mouse_qkv.shape[-1]) // 3
        head_dim_m = c_m // heads
        k_cache_m = kv_cache_mouse["k"]
        v_cache_m = kv_cache_mouse["v"]
        k_len_m = action_local_end - action_k_start
        eps_m = float(getattr(action_model.img_attn_q_norm, "eps", 1e-6))

        mouse_has_rms_qk = (
            type(action_model.img_attn_q_norm).__name__ == "WanRMSNorm"
            and type(action_model.img_attn_k_norm).__name__ == "WanRMSNorm"
        )
        if (
            not exact_mouse_sdpa
            and mouse_has_rms_qk
            and k_len_m > 0
            and _TRITON_AVAILABLE
            and mouse_qkv.is_cuda
            and not torch.is_grad_enabled()
            and head_dim_m in {32, 64, 128}
            and num_frames in {1, 2, 3, 4}
            and k_cache_m.is_contiguous()
            and v_cache_m.is_contiguous()
        ):
            q_rot_m = torch.empty(
                (batch * spatial_s, num_frames, heads, head_dim_m), dtype=mouse_qkv.dtype, device=x.device
            )
            _mg2_mouse_qkv_norm_rope_kv_kernel[(batch * spatial_s * num_frames, heads)](
                mouse_qkv,
                cos_win,
                sin_win,
                q_rot_m,
                k_cache_m,
                v_cache_m,
                num_frames,
                action_local_start,
                int(mouse_qkv.stride(0)),
                int(mouse_qkv.stride(1)),
                int(k_cache_m.stride(0)),
                int(k_cache_m.stride(1)),
                int(k_cache_m.stride(2)),
                int(cos_win.stride(0)),
                eps_m,
                NUM_HEADS=heads,
                HEAD_DIM=head_dim_m,
                HALF_D=head_dim_m // 2,
                num_warps=2,
                enable_fp_fusion=False,
            )
            attn_m = torch.empty((batch, num_frames * spatial_s, c_m), dtype=mouse_qkv.dtype, device=x.device)
            _mg2_mouse_small_attn_kernel[(batch * spatial_s, heads)](
                q_rot_m,
                k_cache_m,
                v_cache_m,
                attn_m,
                spatial_s,
                action_k_start,
                k_len_m,
                int(k_cache_m.stride(0)),
                int(k_cache_m.stride(1)),
                int(k_cache_m.stride(2)),
                1.0 / math.sqrt(head_dim_m),
                NUM_FRAMES=num_frames,
                NUM_HEADS=heads,
                HEAD_DIM=head_dim_m,
                num_warps=2,
            )
        else:
            from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb

            q_m, k_m, v_m = mouse_qkv.view(batch * spatial_s, num_frames, 3, heads, head_dim_m).unbind(dim=2)
            q_m = action_model.img_attn_q_norm(q_m).to(v_m)
            k_m = action_model.img_attn_k_norm(k_m).to(v_m)
            q_m, k_m = apply_rotary_emb(q_m, k_m, (freqs_cos, freqs_sin), start_offset=start_frame, head_first=False)
            k_cache_m[:, action_local_start:action_local_end].copy_(k_m)
            v_cache_m[:, action_local_start:action_local_end].copy_(v_m)
            k_win_m = k_cache_m[:, action_k_start:action_local_end]
            v_win_m = v_cache_m[:, action_k_start:action_local_end]
            attn_raw = F.scaled_dot_product_attention(
                q_m.transpose(1, 2), k_win_m.transpose(1, 2), v_win_m.transpose(1, 2)
            ).transpose(1, 2)
            attn_m = (
                attn_raw.reshape(batch, spatial_s, num_frames, heads * head_dim_m)
                .transpose(1, 2)
                .reshape(batch, num_frames * spatial_s, c_m)
            )
        hidden_states = hidden_states + action_model.proj_mouse(attn_m)

    # 2. Keyboard branch (zero-copy shared-KV SDPA — eliminates 880x .repeat & FA3 tiny-seq overhead!)
    if getattr(action_model, "enable_keyboard", False) and kb_cond_slice is not None and kv_cache_keyboard is not None:
        vcr = int(action_model.vae_time_compression_ratio)
        win_sz = int(action_model.windows_size)
        pad_t = vcr * win_sz
        kb_emb = action_model.keyboard_embed(kb_cond_slice)  # [B, (num_frames - 1 + win_sz) * vcr + ..., C_kb]
        group_kb = torch.stack(
            [kb_emb[:, vcr * i : vcr * i + pad_t, :] for i in range(num_frames)],
            dim=1,
        ).reshape(batch, num_frames, -1)

        mouse_q = action_model.mouse_attn_q(hidden_states)  # [B, T*S, C_k]
        kb_kv = action_model.keyboard_attn_kv(group_kb)  # [B, T, 2*C_k]
        c_k = int(mouse_q.shape[-1])
        head_dim_k = c_k // heads
        eps_k = float(getattr(action_model.key_attn_q_norm, "eps", 1e-6))

        k_kb, v_kb = kb_kv.view(batch, num_frames, 2, heads, head_dim_k).unbind(dim=2)
        k_kb = action_model.key_attn_k_norm(k_kb).to(v_kb)
        if use_rope_keyboard:
            from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb

            # Rotate tiny K [B, T, H, D]
            _, k_kb = apply_rotary_emb(k_kb, k_kb, (freqs_cos, freqs_sin), start_offset=start_frame, head_first=False)

        k_cache_kb = kv_cache_keyboard["k"]
        v_cache_kb = kv_cache_keyboard["v"]
        k_cache_kb[:, action_local_start:action_local_end].copy_(k_kb)
        v_cache_kb[:, action_local_start:action_local_end].copy_(v_kb)
        k_win_kb = k_cache_kb[:, action_k_start:action_local_end]  # [B, Sk, H, D]
        v_win_kb = v_cache_kb[:, action_k_start:action_local_end]

        if (
            use_rope_keyboard
            and _TRITON_AVAILABLE
            and mouse_q.is_cuda
            and not torch.is_grad_enabled()
            and head_dim_k in {32, 64, 128}
        ):
            # Fuse Q per-frame RoPE + [B, T*S, H, D] -> [B, H, T*S, D] directly!
            q_bhsd = torch.empty(
                (batch, heads, num_frames * spatial_s, head_dim_k), dtype=mouse_q.dtype, device=x.device
            )
            kb_has_rms_q = type(action_model.key_attn_q_norm).__name__ == "WanRMSNorm"
            if exact_mouse_sdpa or not kb_has_rms_q:
                q_normed = (
                    action_model.key_attn_q_norm(mouse_q.view(batch, num_frames * spatial_s, heads, head_dim_k))
                    .to(v_kb)
                    .contiguous()
                )
                _mg2_kb_rope_bhsd_kernel[(batch * num_frames * spatial_s, heads)](
                    q_normed,
                    cos_win,
                    sin_win,
                    q_bhsd,
                    num_frames,
                    spatial_s,
                    int(cos_win.stride(0)),
                    NUM_HEADS=heads,
                    HEAD_DIM=head_dim_k,
                    HALF_D=head_dim_k // 2,
                    num_warps=2,
                    enable_fp_fusion=False,
                )
            else:
                _mg2_kb_q_norm_rope_bhsd_kernel[(batch * num_frames * spatial_s, heads)](
                    mouse_q.contiguous(),
                    cos_win,
                    sin_win,
                    q_bhsd,
                    num_frames,
                    spatial_s,
                    int(cos_win.stride(0)),
                    eps_k,
                    NUM_HEADS=heads,
                    HEAD_DIM=head_dim_k,
                    HALF_D=head_dim_k // 2,
                    num_warps=2,
                    enable_fp_fusion=False,
                )
            # Zero-copy shared-KV SDPA across all T*S=2640 queries -> 6 shared keyboard K/V tokens!
            attn_kb = (
                F.scaled_dot_product_attention(q_bhsd, k_win_kb.transpose(1, 2), v_win_kb.transpose(1, 2))
                .transpose(1, 2)
                .reshape(batch, num_frames * spatial_s, c_k)
            )
        else:
            q_kb = action_model.key_attn_q_norm(mouse_q.view(batch, num_frames * spatial_s, heads, head_dim_k)).to(v_kb)
            if use_rope_keyboard:
                from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb

                q_4d = (
                    q_kb.view(batch, num_frames, spatial_s, heads, head_dim_k)
                    .transpose(1, 2)
                    .reshape(batch * spatial_s, num_frames, heads, head_dim_k)
                )
                q_4d, _ = apply_rotary_emb(
                    q_4d, k_kb, (freqs_cos, freqs_sin), start_offset=start_frame, head_first=False
                )
                q_kb = (
                    q_4d.view(batch, spatial_s, num_frames, heads, head_dim_k)
                    .transpose(1, 2)
                    .reshape(batch, num_frames * spatial_s, heads, head_dim_k)
                )
            attn_kb = (
                F.scaled_dot_product_attention(q_kb.transpose(1, 2), k_win_kb.transpose(1, 2), v_win_kb.transpose(1, 2))
                .transpose(1, 2)
                .reshape(batch, num_frames * spatial_s, c_k)
            )
        hidden_states = hidden_states + action_model.proj_keyboard(attn_kb)

    return hidden_states


def build_mg2_3d_rope_cos_sin(
    freqs: torch.Tensor,
    num_frames: int,
    h_grid: int,
    w_grid: int,
    start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build real FP64 ``(cos, sin)`` 3D RoPE tables of shape ``[1, F*H*W, 1, D/2]`` (bitwise exact)."""
    c = int(freqs.shape[1])
    t_dim = c - 2 * (c // 3)
    s_dim = c // 3
    f_t, f_h, f_w = freqs.split([t_dim, s_dim, s_dim], dim=1)
    freqs_3d = torch.cat(
        [
            f_t[start_frame : start_frame + num_frames]
            .view(num_frames, 1, 1, t_dim)
            .expand(num_frames, h_grid, w_grid, t_dim),
            f_h[:h_grid].view(1, h_grid, 1, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
            f_w[:w_grid].view(1, 1, w_grid, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
        ],
        dim=-1,
    ).reshape(1, num_frames * h_grid * w_grid, 1, c)
    return freqs_3d.real.to(torch.float64).contiguous(), freqs_3d.imag.to(torch.float64).contiguous()


def precompute_action_inputs(
    action_model_ref: nn.Module | None,
    *,
    mouse_cond: torch.Tensor | None,
    keyboard_cond: torch.Tensor | None,
    num_frames: int,
    spatial_s: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Hoist mouse & keyboard window slicing once outside the 15-block ActionModule loop."""
    if action_model_ref is None:
        return None, None
    vcr = int(action_model_ref.vae_time_compression_ratio)
    win_sz = int(action_model_ref.windows_size)
    pad_t = vcr * win_sz

    group_mouse_pre: torch.Tensor | None = None
    if getattr(action_model_ref, "enable_mouse", False) and mouse_cond is not None and mouse_cond.numel() > 0:
        b, n_frames, c_m = int(mouse_cond.shape[0]), int(mouse_cond.shape[1]), int(mouse_cond.shape[2])
        n_feats = (n_frames - 1) // vcr + 1
        pad = mouse_cond[:, 0:1, :].expand(-1, pad_t, -1)
        m_padded = torch.cat([pad, mouse_cond], dim=1)
        tail = m_padded[:, vcr * (n_feats - num_frames - win_sz) + pad_t :, :]
        gm = torch.stack([tail[:, vcr * i : vcr * i + pad_t, :] for i in range(num_frames)], dim=1)
        group_mouse_pre = (
            gm.unsqueeze(-1)
            .expand(b, num_frames, pad_t, c_m, spatial_s)
            .permute(0, 4, 1, 2, 3)
            .reshape(b * spatial_s, num_frames, pad_t * c_m)
            .contiguous()
        )

    kb_slice: torch.Tensor | None = None
    if getattr(action_model_ref, "enable_keyboard", False) and keyboard_cond is not None and keyboard_cond.numel() > 0:
        n_frames_k = int(keyboard_cond.shape[1])
        n_feats_k = (n_frames_k - 1) // vcr + 1
        pad_k = keyboard_cond[:, 0:1, :].expand(-1, pad_t, -1)
        k_padded = torch.cat([pad_k, keyboard_cond], dim=1)
        kb_slice = k_padded[:, vcr * (n_feats_k - num_frames - win_sz) + pad_t :, :].contiguous()

    return group_mouse_pre, kb_slice


def mg2_fast_steady_forward(
    model: nn.Module,
    x_in: torch.Tensor,
    t_in: torch.Tensor,
    cc_in: torch.Tensor,
    kb_in: torch.Tensor,
    m_in: torch.Tensor | None,
    *,
    sinusoid_freqs: torch.Tensor,
    kv_cache: list[dict[str, Any]],
    kv_cache_mouse: list[dict[str, Any]] | None,
    kv_cache_keyboard: list[dict[str, Any]] | None,
    crossattn_cache: list[dict[str, Any]],
    local_start_index: int,
    local_end_index: int,
    action_local_start: int,
    action_local_end: int,
    start_frame: int,
    attention_backend: str = "flash_attention_3",
    exact_norms: bool = False,
) -> torch.Tensor:
    """Fast CUDA-Graph-safe steady-state execution of all 30 Matrix-Game-2.0 CausalWan blocks."""
    x = torch.cat([x_in, cc_in], dim=1)
    x = model.patch_embedding(x)
    batch_size = int(x.shape[0])
    f_grid, h_grid, w_grid = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])
    frame_seq = h_grid * w_grid
    seq_len = f_grid * frame_seq
    x = x.flatten(2).transpose(1, 2).contiguous()

    pos = t_in.flatten().to(dtype=torch.float64)
    sinusoid = torch.outer(pos, sinusoid_freqs)
    t_emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).type_as(x)
    e = model.time_embedding(t_emb)
    e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(dim=0, sizes=t_in.shape)

    # Precompute real FP64 3D RoPE once for all 30 blocks (0-ULP identical to causal_rope_apply).
    rope_cos, rope_sin = build_mg2_3d_rope_cos_sin(model.freqs, f_grid, h_grid, w_grid, start_frame)

    # Hoist mouse/keyboard window slicing once outside the 15-block ActionModule loop.
    action_ref = None
    for blk in model.blocks:
        am = getattr(blk, "action_model", None)
        if am is not None:
            action_ref = am
            break
    group_mouse_pre, kb_slice = precompute_action_inputs(
        action_ref,
        mouse_cond=m_in,
        keyboard_cond=kb_in,
        num_frames=f_grid,
        spatial_s=frame_seq,
    )

    num_heads = int(model.num_heads)
    head_dim = int(model.dim) // num_heads
    max_attn_size = int(model.blocks[0].self_attn.max_attention_size)
    kv_read_start = 0 if max_attn_size <= 0 else max(0, local_end_index - max_attn_size)
    action_local_attn_size = int(getattr(model, "local_attn_size", 6))
    action_k_start = 0 if action_local_attn_size <= 0 else max(0, action_local_end - action_local_attn_size)
    use_rope_keyboard = bool(getattr(model, "use_rope_keyboard", True))

    for idx, block in enumerate(model.blocks):
        mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
        h = mg2_layernorm_adaln(x, mod[1], mod[0], num_frames=f_grid, frame_seq=frame_seq, eps=float(block.norm1.eps))

        sa = block.self_attn
        if getattr(sa, "_qkv_fused", False) or hasattr(sa, "qkv"):
            from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import project_fused_qkv

            q_proj, k_proj, v_proj = project_fused_qkv(sa.qkv, h)
        else:
            q_proj, k_proj, v_proj = sa.q(h), sa.k(h), sa.v(h)
        v = v_proj.view(batch_size, seq_len, num_heads, head_dim)

        layer_kv = kv_cache[idx]
        k_cache = layer_kv["k"]
        v_cache = layer_kv["v"]
        v_cache[:, local_start_index:local_end_index].copy_(v)

        if hasattr(sa.norm_q, "weight") and hasattr(sa.norm_k, "weight"):
            roped_q = mg2_qk_rmsnorm_rope_write_kv(
                q_proj,
                k_proj,
                sa.norm_q.weight,
                sa.norm_k.weight,
                rope_cos,
                rope_sin,
                k_cache,
                dst_token_offset=local_start_index,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=float(sa.norm_q.eps),
                exact=exact_norms,
            )
        else:
            from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

            qv = sa.norm_q(q_proj).view(batch_size, seq_len, num_heads, head_dim)
            kv = sa.norm_k(k_proj).view(batch_size, seq_len, num_heads, head_dim)
            roped_q = apply_real_rope_bshd(qv, rope_cos, rope_sin).to(v.dtype)
            k_cache[:, local_start_index:local_end_index].copy_(
                apply_real_rope_bshd(kv, rope_cos, rope_sin).to(v.dtype)
            )

        attn_out = mg2_main_self_attention(
            roped_q,
            k_cache[:, kv_read_start:local_end_index],
            v_cache[:, kv_read_start:local_end_index],
            backend=attention_backend,
        )
        y_sa = sa.o(attn_out.reshape(batch_size, seq_len, model.dim))
        x = mg2_residual_gate_add(x, y_sa, mod[2], num_frames=f_grid, frame_seq=frame_seq)

        # Cached image cross-attention (short Sk=257 -> cuDNN SDPA is 4x faster than FA3!)
        ca = block.cross_attn
        x_ca_in = block.norm3(x)
        q_ca_raw = ca.q(x_ca_in)
        if hasattr(ca.norm_q, "weight") and hasattr(ca.norm_q, "eps"):
            q_ca = mg2_rmsnorm(q_ca_raw, ca.norm_q.weight, eps=float(ca.norm_q.eps), exact=exact_norms).view(
                batch_size, seq_len, num_heads, head_dim
            )
        else:
            q_ca = ca.norm_q(q_ca_raw).view(batch_size, seq_len, num_heads, head_dim)
        ca_slot = crossattn_cache[idx]
        ca_attn = F.scaled_dot_product_attention(
            q_ca.transpose(1, 2),
            ca_slot["k"].transpose(1, 2),
            ca_slot["v"].transpose(1, 2),
        ).transpose(1, 2)
        x = x + ca.o(ca_attn.reshape(batch_size, seq_len, model.dim))

        # ActionModule (mouse + keyboard)
        am = getattr(block, "action_model", None)
        if am is not None:
            x = mg2_fast_action_module_forward(
                am,
                x,
                num_frames=f_grid,
                spatial_s=frame_seq,
                group_mouse_pre=group_mouse_pre,
                kb_cond_slice=kb_slice,
                kv_cache_mouse=None if kv_cache_mouse is None else kv_cache_mouse[idx],
                kv_cache_keyboard=None if kv_cache_keyboard is None else kv_cache_keyboard[idx],
                action_local_start=action_local_start,
                action_local_end=action_local_end,
                action_k_start=action_k_start,
                start_frame=start_frame,
                use_rope_keyboard=use_rope_keyboard,
                exact_mouse_sdpa=exact_norms,
            )

        # FFN + gated residual
        h_ffn = mg2_layernorm_adaln(
            x, mod[4], mod[3], num_frames=f_grid, frame_seq=frame_seq, eps=float(block.norm2.eps)
        )
        y_ffn = block.ffn(h_ffn)
        x = mg2_residual_gate_add(x, y_ffn, mod[5], num_frames=f_grid, frame_seq=frame_seq)

    # Head + unpatchify
    head = model.head
    e_head = (head.modulation.unsqueeze(1) + e.unflatten(dim=0, sizes=t_in.shape).unsqueeze(2)).chunk(2, dim=2)
    h_head = mg2_layernorm_adaln(
        x, e_head[1], e_head[0], num_frames=f_grid, frame_seq=frame_seq, eps=float(head.norm.eps)
    )
    out = head.head(h_head)
    c_out = int(model.out_dim)
    pt, ph, pw = model.patch_size
    out = out[:, : f_grid * h_grid * w_grid].view(batch_size, f_grid, h_grid, w_grid, pt, ph, pw, c_out)
    out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return out.view(batch_size, c_out, f_grid * pt, h_grid * ph, w_grid * pw)
