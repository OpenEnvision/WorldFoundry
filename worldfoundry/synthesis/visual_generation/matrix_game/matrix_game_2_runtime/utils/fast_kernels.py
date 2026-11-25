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
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
    sf_rmsnorm,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import project_fused_qkv
from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="matrix-game-2-triton")

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _round_f32_to_bf16(value, USE_BF16_PTX: tl.constexpr):
        if USE_BF16_PTX:
            return tl.inline_asm_elementwise(
                "{ .reg .b16 rounded; cvt.rn.bf16.f32 rounded, $1; mov.b32 $0, {0, rounded}; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
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
    def _round_to_dtype(value, dtype: tl.constexpr, USE_BF16_PTX: tl.constexpr = True):
        if dtype == tl.bfloat16:
            return _round_f32_to_bf16(value, USE_BF16_PTX)
        elif dtype == tl.float16:
            return _round_f32_to_fp16(value)
        else:
            return value

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
        block: tl.constexpr,
        block_pairs: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Bit-exact coalesced 128-bit Q/K WanRMSNorm + FP64 3D RoPE + KV cache write."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        cols = tl.arange(0, block)
        cmask = cols < hidden_size
        pair = tl.arange(0, block_pairs)
        pmask = pair < (hidden_size // 2)

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        q_in = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
        k_in = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
        q_rstd = tl.load(q_rstd_ptr + row).to(tl.float32)
        k_rstd = tl.load(k_rstd_ptr + row).to(tl.float32)
        qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)

        q_n = _round_to_dtype(
            _round_to_dtype(q_in.to(tl.float32) * q_rstd, q_in.dtype, USE_BF16_PTX) * qw, q_in.dtype, USE_BF16_PTX
        )
        k_n = _round_to_dtype(
            _round_to_dtype(k_in.to(tl.float32) * k_rstd, k_in.dtype, USE_BF16_PTX) * kw, k_in.dtype, USE_BF16_PTX
        )

        qe, qo = tl.split(tl.reshape(q_n, (block_pairs, 2)))
        ke, ko = tl.split(tl.reshape(k_n, (block_pairs, 2)))

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
        s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)

        rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(q_in.dtype)
        rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(q_in.dtype)
        rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(k_in.dtype)
        rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(k_in.dtype)

        rq = tl.reshape(tl.join(rqe, rqo), (block,))
        rk = tl.reshape(tl.join(rke, rko), (block,))

        q_out_base = row.to(tl.int64) * hidden_size
        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        tl.store(q_out_ptr + q_out_base + cols, rq, mask=cmask)
        tl.store(k_dst_ptr + k_dst_base + cols, rk, mask=cmask)

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
        block: tl.constexpr,
        block_pairs: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Single-pass coalesced 128-bit Q/K WanRMSNorm + FP64 3D RoPE + KV write."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        cols = tl.arange(0, block)
        cmask = cols < hidden_size
        pair = tl.arange(0, block_pairs)
        pmask = pair < (hidden_size // 2)

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        q_in = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
        k_in = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
        qf = q_in.to(tl.float32)
        kf = k_in.to(tl.float32)

        q_mean_sq = tl.sum(qf * qf, axis=0) / hidden_size
        k_mean_sq = tl.sum(kf * kf, axis=0) / hidden_size
        q_rstd = _libdevice.rsqrt(q_mean_sq + eps)
        k_rstd = _libdevice.rsqrt(k_mean_sq + eps)

        qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)

        q_n = _round_to_dtype(_round_to_dtype(qf * q_rstd, q_in.dtype, USE_BF16_PTX) * qw, q_in.dtype, USE_BF16_PTX)
        k_n = _round_to_dtype(_round_to_dtype(kf * k_rstd, k_in.dtype, USE_BF16_PTX) * kw, k_in.dtype, USE_BF16_PTX)

        qe, qo = tl.split(tl.reshape(q_n, (block_pairs, 2)))
        ke, ko = tl.split(tl.reshape(k_n, (block_pairs, 2)))

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
        s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)

        rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(q_in.dtype)
        rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(q_in.dtype)
        rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(k_in.dtype)
        rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(k_in.dtype)

        rq = tl.reshape(tl.join(rqe, rqo), (block,))
        rk = tl.reshape(tl.join(rke, rko), (block,))

        q_out_base = row.to(tl.int64) * hidden_size
        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        tl.store(q_out_ptr + q_out_base + cols, rq, mask=cmask)
        tl.store(k_dst_ptr + k_dst_base + cols, rk, mask=cmask)

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
        USE_BF16_PTX: tl.constexpr,
    ):
        """Coalesced 128-bit mouse QKV split + ActionModule RMSNorm + FP32 interleaved RoPE + KV write.

        Input ``qkv`` has shape ``[BS, T, 3 * NUM_HEADS * HEAD_DIM]`` with ``K=3`` outermost.
        Outputs ``q_out`` ``[BS, T, NUM_HEADS, HEAD_DIM]`` and writes ``k, v`` into
        ``k_dst[:, dst_frame_offset:dst_frame_offset+T]`` and ``v_dst[...]``.
        """
        bs_t = tl.program_id(0)
        head = tl.program_id(1)
        bs_idx = bs_t // num_frames
        t_idx = bs_t % num_frames

        d_off = tl.arange(0, HEAD_DIM)
        hidden = NUM_HEADS * HEAD_DIM
        row_base = bs_idx.to(tl.int64) * qkv_stride_bs + t_idx.to(tl.int64) * qkv_stride_t
        q_base = row_base + head.to(tl.int64) * HEAD_DIM
        k_base = row_base + hidden + head.to(tl.int64) * HEAD_DIM
        v_base = row_base + 2 * hidden + head.to(tl.int64) * HEAD_DIM

        q_raw = tl.load(qkv_ptr + q_base + d_off)
        k_raw = tl.load(qkv_ptr + k_base + d_off)
        v_raw = tl.load(qkv_ptr + v_base + d_off)

        kv_out_base = (
            bs_idx.to(tl.int64) * kv_stride_bs
            + (dst_frame_offset + t_idx).to(tl.int64) * kv_stride_t
            + head.to(tl.int64) * kv_stride_h
        )
        tl.store(v_dst_ptr + kv_out_base + d_off, v_raw)

        # ActionModule.WanRMSNorm: x * rsqrt(x.pow(2).mean(-1) + eps) computed in x.dtype (BF16/FP16)
        q_f = q_raw.to(tl.float32)
        k_f = k_raw.to(tl.float32)
        q_sq = _round_to_dtype(q_f * q_f, q_raw.dtype, USE_BF16_PTX)
        k_sq = _round_to_dtype(k_f * k_f, k_raw.dtype, USE_BF16_PTX)
        q_mean = _round_to_dtype(tl.sum(q_sq, axis=0) / HEAD_DIM, q_raw.dtype, USE_BF16_PTX)
        k_mean = _round_to_dtype(tl.sum(k_sq, axis=0) / HEAD_DIM, k_raw.dtype, USE_BF16_PTX)
        q_rstd = _round_to_dtype(
            _libdevice.rsqrt(_round_to_dtype(q_mean + eps, q_raw.dtype, USE_BF16_PTX)), q_raw.dtype, USE_BF16_PTX
        )
        k_rstd = _round_to_dtype(
            _libdevice.rsqrt(_round_to_dtype(k_mean + eps, k_raw.dtype, USE_BF16_PTX)), k_raw.dtype, USE_BF16_PTX
        )
        q_n = _round_to_dtype(q_f * q_rstd, q_raw.dtype, USE_BF16_PTX)
        k_n = _round_to_dtype(k_f * k_rstd, k_raw.dtype, USE_BF16_PTX)

        qe_n, qo_n = tl.split(tl.reshape(q_n, (HALF_D, 2)))
        ke_n, ko_n = tl.split(tl.reshape(k_n, (HALF_D, 2)))

        # Interleaved FP32 RoPE: cos/sin have repeated pairs [c0, c0, c1, c1, ...]
        cs_base = t_idx.to(tl.int64) * cos_stride_t + d_off
        c_full = tl.load(cos_ptr + cs_base).to(tl.float32)
        s_full = tl.load(sin_ptr + cs_base).to(tl.float32)
        c, _ = tl.split(tl.reshape(c_full, (HALF_D, 2)))
        s, _ = tl.split(tl.reshape(s_full, (HALF_D, 2)))

        rqe = (qe_n * c - qo_n * s).to(q_raw.dtype)
        rqo = (qe_n * s + qo_n * c).to(q_raw.dtype)
        rke = (ke_n * c - ko_n * s).to(k_raw.dtype)
        rko = (ke_n * s + ko_n * c).to(k_raw.dtype)

        rq = tl.reshape(tl.join(rqe, rqo), (HEAD_DIM,))
        rk = tl.reshape(tl.join(rke, rko), (HEAD_DIM,))

        q_out_base = (bs_t.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + q_out_base + d_off, rq)
        tl.store(k_dst_ptr + kv_out_base + d_off, rk)

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
        USE_BF16_PTX: tl.constexpr,
    ):
        """Coalesced 128-bit keyboard Q RMSNorm + per-frame FP32 RoPE + ``[B, T*S, H, D] -> [B, H, T*S, D]``."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        seq_len = num_frames * spatial_s
        b_idx = row // seq_len
        tok_idx = row % seq_len
        t_idx = tok_idx // spatial_s

        d_off = tl.arange(0, HEAD_DIM)
        in_base = (row.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        q_raw = tl.load(q_in_ptr + in_base + d_off)

        q_f = q_raw.to(tl.float32)
        q_sq = _round_to_dtype(q_f * q_f, q_raw.dtype, USE_BF16_PTX)
        q_mean = _round_to_dtype(tl.sum(q_sq, axis=0) / HEAD_DIM, q_raw.dtype, USE_BF16_PTX)
        rstd = _round_to_dtype(
            _libdevice.rsqrt(_round_to_dtype(q_mean + eps, q_raw.dtype, USE_BF16_PTX)), q_raw.dtype, USE_BF16_PTX
        )
        q_n = _round_to_dtype(q_f * rstd, q_raw.dtype, USE_BF16_PTX)
        qe_n, qo_n = tl.split(tl.reshape(q_n, (HALF_D, 2)))

        cs_base = t_idx.to(tl.int64) * cos_stride_t + d_off
        c_full = tl.load(cos_ptr + cs_base).to(tl.float32)
        s_full = tl.load(sin_ptr + cs_base).to(tl.float32)
        c, _ = tl.split(tl.reshape(c_full, (HALF_D, 2)))
        s, _ = tl.split(tl.reshape(s_full, (HALF_D, 2)))

        rqe = (qe_n * c - qo_n * s).to(q_raw.dtype)
        rqo = (qe_n * s + qo_n * c).to(q_raw.dtype)
        rq = tl.reshape(tl.join(rqe, rqo), (HEAD_DIM,))

        out_base = ((b_idx.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * seq_len + tok_idx.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + out_base + d_off, rq)

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
        """Coalesced 128-bit per-frame FP32 RoPE + ``[B, T*S, H, D] -> [B, H, T*S, D]`` on pre-normed Q."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        seq_len = num_frames * spatial_s
        b_idx = row // seq_len
        tok_idx = row % seq_len
        t_idx = tok_idx // spatial_s

        d_off = tl.arange(0, HEAD_DIM)
        in_base = (row.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM
        q_raw = tl.load(q_in_ptr + in_base + d_off)
        qe_f, qo_f = tl.split(tl.reshape(q_raw.to(tl.float32), (HALF_D, 2)))

        cs_base = t_idx.to(tl.int64) * cos_stride_t + d_off
        c_full = tl.load(cos_ptr + cs_base).to(tl.float32)
        s_full = tl.load(sin_ptr + cs_base).to(tl.float32)
        c, _ = tl.split(tl.reshape(c_full, (HALF_D, 2)))
        s, _ = tl.split(tl.reshape(s_full, (HALF_D, 2)))

        rqe = (qe_f * c - qo_f * s).to(q_raw.dtype)
        rqo = (qe_f * s + qo_f * c).to(q_raw.dtype)
        rq = tl.reshape(tl.join(rqe, rqo), (HEAD_DIM,))

        out_base = ((b_idx.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * seq_len + tok_idx.to(tl.int64)) * HEAD_DIM
        tl.store(q_out_ptr + out_base + d_off, rq)

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


def _use_bf16_ptx(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def mg2_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
    exact: bool = True,
) -> torch.Tensor:
    """``WanLayerNorm`` + fused Triton per-frame AdaLN ``(1 + scale) + shift`` (zero mod copies)."""
    return sf_layernorm_adaln(
        x,
        scale,
        shift,
        num_frames=num_frames,
        frame_seq=frame_seq,
        eps=eps,
        exact=exact,
        enabled=True,
    )


def mg2_residual_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    """Bit-exact fused ``x + (y * gate)`` with strided per-frame gate broadcast (zero copies)."""
    return sf_residual_gate_add(
        x,
        y,
        gate,
        num_frames=num_frames,
        frame_seq=frame_seq,
        enabled=True,
    )


def mg2_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, *, exact: bool = True) -> torch.Tensor:
    """Fast coalesced ``WanRMSNorm`` supporting strided last-dim-contiguous ``[B, S, C]``."""
    return sf_rmsnorm(x, weight, eps=eps, exact=exact, enabled=True)


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
    v_proj: torch.Tensor | None = None,
    v_dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """Coalesced 128-bit fused Q/K WanRMSNorm + FP64 3D RoPE + direct K(/V) cache store."""
    if v_proj is not None and v_dst is not None:
        return sf_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            v_proj,
            q_weight,
            k_weight,
            cos,
            sin,
            k_dst,
            v_dst,
            dst_start=dst_token_offset,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            rotate_k=True,
            exact=exact,
            enabled=True,
        )
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
        block = max(32, int(triton.next_power_of_2(hidden)))
        block_pairs = block // 2
        use_ptx = _use_bf16_ptx(q_proj.device)
        num_warps = 4 if block >= 1024 else 2
        if block > 2048:
            num_warps = 8
        cos_c = cos.reshape(seq_len, half_d).contiguous()
        sin_c = sin.reshape(seq_len, half_d).contiguous()
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
                cos_c,
                sin_c,
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
                block=block,
                block_pairs=block_pairs,
                USE_BF16_PTX=use_ptx,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
        else:
            _mg2_qk_rmsnorm_rope_write_kv_fp64_kernel[(rows,)](
                q_proj,
                k_proj,
                q_weight,
                k_weight,
                cos_c,
                sin_c,
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
                block=block,
                block_pairs=block_pairs,
                USE_BF16_PTX=use_ptx,
                num_warps=num_warps,
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
    action_model: Any,
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
    return_deferred_residual: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Fast CUDA-Graph-safe ``ActionModule`` steady-state forward.

    Replaces FA3-on-tiny-sequence mouse/keyboard attention (0.442 ms each) with:
    - Coalesced 128-bit Triton mouse QKV-Norm+RoPE+KV-write + fused Triton small-sequence
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
    use_ptx = _use_bf16_ptx(x.device)

    run_mouse = bool(
        getattr(action_model, "enable_mouse", False) and group_mouse_pre is not None and kv_cache_mouse is not None
    )
    run_keyboard = bool(
        getattr(action_model, "enable_keyboard", False) and kb_cond_slice is not None and kv_cache_keyboard is not None
    )
    deferred_update: torch.Tensor | None = None

    # 1. Mouse branch (when enabled & conditioned)
    if run_mouse:
        assert group_mouse_pre is not None and kv_cache_mouse is not None
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
                USE_BF16_PTX=use_ptx,
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
        mouse_proj = action_model.proj_mouse(attn_m)
        if return_deferred_residual and not run_keyboard:
            deferred_update = mouse_proj
        else:
            hidden_states = hidden_states + mouse_proj

    # 2. Keyboard branch (zero-copy shared-KV SDPA — eliminates 880x .repeat & FA3 tiny-seq overhead!)
    if run_keyboard:
        assert kb_cond_slice is not None and kv_cache_keyboard is not None
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
                    USE_BF16_PTX=use_ptx,
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
        kb_proj = action_model.proj_keyboard(attn_kb)
        if return_deferred_residual:
            deferred_update = kb_proj
        else:
            hidden_states = hidden_states + kb_proj

    if return_deferred_residual:
        return hidden_states, deferred_update
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
    action_model_ref: Any,
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
    model: Any,
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
    if model.freqs.device != x.device:
        model.freqs = model.freqs.to(x.device)
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
        h = sf_layernorm_adaln(
            x,
            mod[1],
            mod[0],
            num_frames=f_grid,
            frame_seq=frame_seq,
            eps=float(block.norm1.eps),
            exact=exact_norms,
            enabled=True,
        )

        sa = block.self_attn
        if getattr(sa, "_qkv_fused", False) or hasattr(sa, "qkv"):
            q_proj, k_proj, v_proj = project_fused_qkv(sa.qkv, h)
        else:
            q_proj, k_proj, v_proj = sa.q(h), sa.k(h), sa.v(h)

        layer_kv = kv_cache[idx]
        k_cache = layer_kv["k"]
        v_cache = layer_kv["v"]

        if hasattr(sa.norm_q, "weight") and hasattr(sa.norm_k, "weight"):
            roped_q = sf_qk_rmsnorm_rope_write_kv(
                q_proj,
                k_proj,
                v_proj,
                sa.norm_q.weight,
                sa.norm_k.weight,
                rope_cos,
                rope_sin,
                k_cache,
                v_cache,
                dst_start=local_start_index,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=float(sa.norm_q.eps),
                rotate_k=True,
                exact=exact_norms,
                enabled=True,
            )
        else:
            from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

            v = v_proj.view(batch_size, seq_len, num_heads, head_dim)
            v_cache[:, local_start_index:local_end_index].copy_(v)
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

        # 2-stage super-fusion: x = x + y_sa * mod[2] AND x_ca_in = norm3(x) in one HBM pass!
        norm3 = block.norm3
        if hasattr(norm3, "weight") and hasattr(norm3, "eps"):
            x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y_sa.dtype, mod[2].dtype))
            w3 = norm3.weight if norm3.weight.dtype == x_out_dtype else norm3.weight.to(x_out_dtype)
            b3_raw = getattr(norm3, "bias", None)
            b3 = None if b3_raw is None else (b3_raw if b3_raw.dtype == x_out_dtype else b3_raw.to(x_out_dtype))
            x, x_ca_in = sf_residual_gate_add_layernorm(
                x,
                y_sa,
                mod[2],
                w3,
                b3,
                num_frames=f_grid,
                frame_seq=frame_seq,
                eps=float(norm3.eps),
                exact=exact_norms,
                enabled=True,
            )
        else:
            x = sf_residual_gate_add(x, y_sa, mod[2], num_frames=f_grid, frame_seq=frame_seq, enabled=True)
            x_ca_in = norm3(x)

        # Cached image cross-attention (short Sk=257 -> cuDNN SDPA is 4x faster than FA3!)
        ca = block.cross_attn
        q_ca_raw = ca.q(x_ca_in)
        if hasattr(ca.norm_q, "weight") and hasattr(ca.norm_q, "eps"):
            q_ca = sf_rmsnorm(
                q_ca_raw, ca.norm_q.weight, eps=float(ca.norm_q.eps), exact=exact_norms, enabled=True
            ).view(batch_size, seq_len, num_heads, head_dim)
        else:
            q_ca = ca.norm_q(q_ca_raw).view(batch_size, seq_len, num_heads, head_dim)
        ca_slot = crossattn_cache[idx]
        ca_attn = F.scaled_dot_product_attention(
            q_ca.transpose(1, 2),
            ca_slot["k"].transpose(1, 2),
            ca_slot["v"].transpose(1, 2),
        ).transpose(1, 2)
        ca_out = ca.o(ca_attn.reshape(batch_size, seq_len, model.dim))

        # ActionModule (mouse + keyboard) + 2-stage super-fused residual add + norm2 + AdaLN across ALL 30 blocks!
        am = getattr(block, "action_model", None)
        if am is not None:
            x = x + ca_out
            am_res = mg2_fast_action_module_forward(
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
                return_deferred_residual=True,
            )
            assert isinstance(am_res, tuple)
            x, deferred_update = am_res
            if deferred_update is not None:
                x, h_ffn = sf_residual_add_layernorm_adaln(
                    x,
                    deferred_update,
                    mod[4],
                    mod[3],
                    num_frames=f_grid,
                    frame_seq=frame_seq,
                    eps=float(block.norm2.eps),
                    exact=exact_norms,
                    enabled=True,
                )
            else:
                h_ffn = sf_layernorm_adaln(
                    x,
                    mod[4],
                    mod[3],
                    num_frames=f_grid,
                    frame_seq=frame_seq,
                    eps=float(block.norm2.eps),
                    exact=exact_norms,
                    enabled=True,
                )
        else:
            x, h_ffn = sf_residual_add_layernorm_adaln(
                x,
                ca_out,
                mod[4],
                mod[3],
                num_frames=f_grid,
                frame_seq=frame_seq,
                eps=float(block.norm2.eps),
                exact=exact_norms,
                enabled=True,
            )

        y_ffn = block.ffn(h_ffn)
        x = sf_residual_gate_add(x, y_ffn, mod[5], num_frames=f_grid, frame_seq=frame_seq, enabled=True)

    # Head + unpatchify
    head = model.head
    e_head = (head.modulation.unsqueeze(1) + e.unflatten(dim=0, sizes=t_in.shape).unsqueeze(2)).chunk(2, dim=2)
    h_head = sf_layernorm_adaln(
        x,
        e_head[1],
        e_head[0],
        num_frames=f_grid,
        frame_seq=frame_seq,
        eps=float(head.norm.eps),
        exact=exact_norms,
        enabled=True,
    )
    out = head.head(h_head)
    c_out = int(model.out_dim)
    pt, ph, pw = model.patch_size
    out = out[:, : f_grid * h_grid * w_grid].view(batch_size, f_grid, h_grid, w_grid, pt, ph, pw, c_out)
    out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return out.view(batch_size, c_out, f_grid * pt, h_grid * ph, w_grid * pw)
