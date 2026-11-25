# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for SANA-Video ``LiteLAReLURope``.

Fuses the entire ReLU-kernel 3D-RoPE linear attention body directly from the
packed ``qkv = self.qkv(x)`` output ``[B, N, 3*C]`` without materializing
intermediate normalized, ReLU'd, RoPE'd, or FP32-cast Q/K/V tensors in HBM:

1. Optional ``_sana_qk_rstd_kernel`` computes per-token full-hidden RMSNorm
   reciprocal std ``[B, N]`` for Q and K in one pass (only 524 KB written).
2. ``_sana_litela_kv_splitk_kernel`` streams strided K/V tiles per ``(b, h)``
   and split-K partition, applies RMSNorm + ReLU + interleaved 3D RoPE (FP64
   or FP32 exact) in registers, accumulates ``k_sum = sum_n(k_relu)`` and
   ``KV = K_rot^T @ V`` via Hopper Tensor Core ``tl.dot`` in FP32, and writes
   deterministic split-K partials (no atomic non-determinism).
3. ``_sana_litela_q_apply_div_kernel`` loads the tiny ``(block_d, block_d)``
   KV matrix and ``(block_d,)`` ``k_sum`` once per CTA, applies Q RMSNorm +
   ReLU + 3D RoPE in registers, evaluates ``(Q_rot @ KV) / (Q_relu @ k_sum + eps)``
   (using Dual-BF16 split Tensor Cores for FP32-exact BF16 attention), and
   writes the contiguous ``[B, N, C]`` output ready for ``self.proj``.

Supports any even ``head_dim`` in ``[16, 128]`` (including SANA-Video 2B's
non-power-of-two ``head_dim = 112, num_heads = 20`` via zero-masked
``block_d = next_power_of_2(head_dim)`` tiles).

All CUDA ops are registered via ``torch.library.custom_op`` with fake tensor
implementations for seamless ``torch.compile`` and CUDA Graph composition.
Importing this module does not import Triton or initialize CUDA.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Triton kernels on first eligible CUDA call."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="sana-litela-triton")

    import triton
    import triton.language as tl

    @triton.jit
    def _sana_qk_rstd_kernel(
        qkv_ptr,
        q_rstd_ptr,
        k_rstd_ptr,
        total_tokens,
        stride_row,
        q_eps,
        k_eps,
        hidden_dim: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Compute full-hidden RMSNorm ``rsqrt(mean(x^2) + eps)`` for Q and K."""
        pid = tl.program_id(0).to(tl.int64)
        is_k = pid >= total_tokens
        row = tl.where(is_k, pid - total_tokens, pid)
        base = row * stride_row + tl.where(is_k, hidden_dim, 0)

        sq_sum = tl.zeros((), dtype=tl.float32)
        for start in tl.static_range(0, hidden_dim, block_c):
            cols = start + tl.arange(0, block_c)
            mask = cols < hidden_dim
            x = tl.load(qkv_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
            sq_sum += tl.sum(x * x, axis=0)

        eps = tl.where(is_k, k_eps, q_eps)
        rstd = tl.rsqrt(sq_sum / hidden_dim + eps)
        if is_k:
            tl.store(k_rstd_ptr + row, rstd)
        else:
            tl.store(q_rstd_ptr + row, rstd)

    @triton.jit
    def _sana_litela_kv_splitk_kernel(
        qkv_ptr,
        k_rstd_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        partial_kv_ptr,
        partial_ksum_ptr,
        seq_len,
        stride_qkv_b,
        stride_qkv_n,
        stride_cos_b,
        stride_cos_n,
        stride_sin_b,
        stride_sin_n,
        hidden_dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_half: tl.constexpr,
        num_splits: tl.constexpr,
        block_n: tl.constexpr,
        cos_batch_broadcast: tl.constexpr,
        has_k_norm: tl.constexpr,
        use_fp64_rope: tl.constexpr,
        use_fp32_dot: tl.constexpr,
    ):
        """Accumulate deterministic split-K ``KV = K_rot^T @ V`` and ``k_sum = sum(K_relu)``."""
        bh = tl.program_id(0).to(tl.int64)
        split_id = tl.program_id(1).to(tl.int64)
        b = bh // num_heads
        h = bh % num_heads
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        tokens_per_split = (seq_len + num_splits - 1) // num_splits
        n_start = split_id * tokens_per_split
        n_end = tl.minimum(n_start + tokens_per_split, seq_len)

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        offs_half = tl.arange(0, block_half)
        half_mask = offs_half < half_dim

        if has_k_norm:
            kw_d = tl.load(k_weight_ptr + h * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        kv_acc = tl.zeros((block_d, block_d), dtype=tl.float32)
        ksum_acc = tl.zeros((block_d,), dtype=tl.float32)

        qkv_b_base = b * stride_qkv_b
        k_head_col = hidden_dim + h * head_dim
        v_head_col = 2 * hidden_dim + h * head_dim
        cos_b_base = b_freq * stride_cos_b
        sin_b_base = b_freq * stride_sin_b
        rstd_b_base = b * seq_len

        for n_blk in range(n_start, n_end, block_n):
            offs_n = n_blk + tl.arange(0, block_n)
            n_mask = offs_n < n_end
            nd_mask = n_mask[:, None] & d_mask[None, :]
            nh_mask = n_mask[:, None] & half_mask[None, :]
            row_ptrs = qkv_b_base + offs_n.to(tl.int64)[:, None] * stride_qkv_n

            # Coalesced (block_n, block_d) load with zero padding on d >= head_dim.
            k_in = tl.load(qkv_ptr + row_ptrs + (k_head_col + offs_d)[None, :], mask=nd_mask, other=0.0)

            if has_k_norm:
                rstd = tl.load(k_rstd_ptr + rstd_b_base + offs_n, mask=n_mask, other=0.0)[:, None]
                k_n = (k_in.to(tl.float32) * rstd * kw_d[None, :]).to(k_in.dtype)
            else:
                k_n = k_in

            k_relu = tl.where(nd_mask, tl.maximum(k_n.to(tl.float32), 0.0), 0.0)
            ksum_acc += tl.sum(k_relu, axis=0)

            ke_relu, ko_relu = tl.split(tl.reshape(k_relu, (block_n, block_half, 2)))
            cos_offs = cos_b_base + offs_n.to(tl.int64)[:, None] * stride_cos_n + offs_half[None, :]
            sin_offs = sin_b_base + offs_n.to(tl.int64)[:, None] * stride_sin_n + offs_half[None, :]
            if use_fp64_rope:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float64)
                ke_64 = ke_relu.to(tl.float64)
                ko_64 = ko_relu.to(tl.float64)
                kr_e = (ke_64 * c - ko_64 * s).to(k_in.dtype)
                kr_o = (ko_64 * c + ke_64 * s).to(k_in.dtype)
            else:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float32)
                kr_e = (ke_relu * c - ko_relu * s).to(k_in.dtype)
                kr_o = (ko_relu * c + ke_relu * s).to(k_in.dtype)

            k_rot = tl.reshape(tl.join(kr_e, kr_o), (block_n, block_d))
            v_tile = tl.load(qkv_ptr + row_ptrs + (v_head_col + offs_d)[None, :], mask=nd_mask, other=0.0)

            if use_fp32_dot and (k_in.dtype == tl.float32):
                kv_acc = tl.dot(tl.trans(k_rot), v_tile, acc=kv_acc, input_precision="ieee")
            else:
                kv_acc = tl.dot(tl.trans(k_rot), v_tile, acc=kv_acc)

        kv_base = (bh * num_splits + split_id) * (block_d * block_d)
        tl.store(partial_kv_ptr + kv_base + offs_d[:, None] * block_d + offs_d[None, :], kv_acc)

        ksum_base = (bh * num_splits + split_id) * block_d
        tl.store(partial_ksum_ptr + ksum_base + offs_d, ksum_acc)

    @triton.jit
    def _sana_litela_q_apply_div_kernel(
        qkv_ptr,
        q_rstd_ptr,
        q_weight_ptr,
        cos_ptr,
        sin_ptr,
        kv_ptr,
        ksum_ptr,
        out_ptr,
        seq_len,
        stride_qkv_b,
        stride_qkv_n,
        stride_out_b,
        stride_out_n,
        stride_cos_b,
        stride_cos_n,
        stride_sin_b,
        stride_sin_n,
        eps,
        hidden_dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_half: tl.constexpr,
        block_n: tl.constexpr,
        tiles_per_cta: tl.constexpr,
        cos_batch_broadcast: tl.constexpr,
        has_q_norm: tl.constexpr,
        use_fp64_rope: tl.constexpr,
        use_fp32_dot: tl.constexpr,
    ):
        """Compute ``out = (Q_rot @ KV) / (Q_relu @ k_sum + eps)`` directly to ``[B, N, C]``."""
        bh = tl.program_id(0).to(tl.int64)
        cta_tile = tl.program_id(1).to(tl.int64)
        b = bh // num_heads
        h = bh % num_heads
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        offs_half = tl.arange(0, block_half)
        half_mask = offs_half < half_dim

        # Load padded (block_d, block_d) KV state and (block_d,) k_sum ONCE per CTA.
        kv_base = bh * (block_d * block_d)
        kv_mat = tl.load(kv_ptr + kv_base + offs_d[:, None] * block_d + offs_d[None, :]).to(tl.float32)
        ksum_base = bh * block_d
        ksum_d = tl.load(ksum_ptr + ksum_base + offs_d).to(tl.float32)

        q_head_col = h * head_dim
        if has_q_norm:
            qw_d = tl.load(q_weight_ptr + q_head_col + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        in_ty = qkv_ptr.dtype.element_ty
        if use_fp32_dot and (in_ty == tl.bfloat16):
            kv_hi_bf16 = kv_mat.to(tl.bfloat16)
            kv_lo_bf16 = (kv_mat - kv_hi_bf16.to(tl.float32)).to(tl.bfloat16)
        elif not use_fp32_dot:
            kv_low = kv_mat.to(in_ty)

        qkv_b_base = b * stride_qkv_b
        out_b_base = b * stride_out_b
        cos_b_base = b_freq * stride_cos_b
        sin_b_base = b_freq * stride_sin_b
        rstd_b_base = b * seq_len

        base_n = cta_tile * (tiles_per_cta * block_n)
        for t_idx in tl.static_range(0, tiles_per_cta):
            offs_n = base_n + t_idx * block_n + tl.arange(0, block_n)
            n_mask = offs_n < seq_len
            nd_mask = n_mask[:, None] & d_mask[None, :]
            nh_mask = n_mask[:, None] & half_mask[None, :]
            row_ptrs = qkv_b_base + offs_n.to(tl.int64)[:, None] * stride_qkv_n
            q_in = tl.load(qkv_ptr + row_ptrs + (q_head_col + offs_d)[None, :], mask=nd_mask, other=0.0)

            if has_q_norm:
                rstd = tl.load(q_rstd_ptr + rstd_b_base + offs_n, mask=n_mask, other=0.0)[:, None]
                q_n = (q_in.to(tl.float32) * rstd * qw_d[None, :]).to(q_in.dtype)
            else:
                q_n = q_in

            q_relu = tl.where(nd_mask, tl.maximum(q_n.to(tl.float32), 0.0), 0.0)
            denom = tl.sum(q_relu * ksum_d[None, :], axis=1) + eps

            qe_relu, qo_relu = tl.split(tl.reshape(q_relu, (block_n, block_half, 2)))
            cos_offs = cos_b_base + offs_n.to(tl.int64)[:, None] * stride_cos_n + offs_half[None, :]
            sin_offs = sin_b_base + offs_n.to(tl.int64)[:, None] * stride_sin_n + offs_half[None, :]
            if use_fp64_rope:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float64)
                qe_64 = qe_relu.to(tl.float64)
                qo_64 = qo_relu.to(tl.float64)
                qr_e = (qe_64 * c - qo_64 * s).to(q_in.dtype)
                qr_o = (qo_64 * c + qe_64 * s).to(q_in.dtype)
            else:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float32)
                qr_e = (qe_relu * c - qo_relu * s).to(q_in.dtype)
                qr_o = (qo_relu * c + qe_relu * s).to(q_in.dtype)

            q_rot = tl.reshape(tl.join(qr_e, qr_o), (block_n, block_d))
            if use_fp32_dot and (in_ty == tl.float32):
                num = tl.dot(q_rot, kv_mat, input_precision="ieee")
            elif use_fp32_dot and (in_ty == tl.bfloat16):
                num = tl.dot(q_rot, kv_hi_bf16) + tl.dot(q_rot, kv_lo_bf16)
            elif use_fp32_dot:
                num = tl.dot(q_rot, kv_mat.to(in_ty)).to(tl.float32)
            else:
                num = tl.dot(q_rot, kv_low).to(tl.float32)

            out_val = (num / denom[:, None]).to(q_in.dtype)
            out_ptrs = out_b_base + offs_n.to(tl.int64)[:, None] * stride_out_n + (q_head_col + offs_d)[None, :]
            tl.store(out_ptr + out_ptrs, out_val, mask=nd_mask)

    @triton.jit
    def _sana_rms_norm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_x,
        stride_out,
        eps,
        features: tl.constexpr,
        block: tl.constexpr,
    ):
        """Single-pass Hopper RMSNorm matching ``(x.float() * rstd * w.float()).to(x.dtype)``."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, block)
        mask = cols < features
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / features) + eps)
        y = (x * rstd * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + cols, y, mask=mask)

    @triton.jit
    def _sana_litela_prep_kv_swizzled_kernel(
        qkv_ptr,
        k_rstd_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        k_rot_ptr,
        v_out_ptr,
        ksum_part_ptr,
        seq_len,
        stride_qkv_b,
        stride_qkv_n,
        stride_cos_b,
        stride_cos_n,
        stride_sin_b,
        stride_sin_n,
        hidden_dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_half: tl.constexpr,
        block_n: tl.constexpr,
        tiles_per_cta: tl.constexpr,
        num_chunks: tl.constexpr,
        cos_batch_broadcast: tl.constexpr,
        has_k_norm: tl.constexpr,
        use_fp64_rope: tl.constexpr,
    ):
        """L2-swizzled ``K`` norm+ReLU+RoPE, ``V`` pack, and ``k_sum`` partial reduction."""
        bh = tl.program_id(0).to(tl.int64)
        chunk_id = tl.program_id(1).to(tl.int64)
        b = bh // num_heads
        h = bh % num_heads
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        n_base = chunk_id * (block_n * tiles_per_cta)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        offs_half = tl.arange(0, block_half)
        half_mask = offs_half < half_dim

        if has_k_norm:
            kw_d = tl.load(k_weight_ptr + h * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        ksum_acc = tl.zeros((block_d,), dtype=tl.float32)
        qkv_b = b * stride_qkv_b
        k_col = hidden_dim + h * head_dim
        v_col = 2 * hidden_dim + h * head_dim
        cos_b = b_freq * stride_cos_b
        sin_b = b_freq * stride_sin_b
        rstd_b = b * seq_len
        out_bh = bh * seq_len * head_dim

        for ti in tl.static_range(0, tiles_per_cta):
            offs_n = n_base + ti * block_n + tl.arange(0, block_n)
            n_mask = offs_n < seq_len
            nd_mask = n_mask[:, None] & d_mask[None, :]
            nh_mask = n_mask[:, None] & half_mask[None, :]
            offs_n64 = offs_n.to(tl.int64)
            row_ptrs = qkv_b + offs_n64[:, None] * stride_qkv_n

            k_in = tl.load(qkv_ptr + row_ptrs + (k_col + offs_d)[None, :], mask=nd_mask, other=0.0)
            v_in = tl.load(qkv_ptr + row_ptrs + (v_col + offs_d)[None, :], mask=nd_mask, other=0.0)

            if has_k_norm:
                rstd = tl.load(k_rstd_ptr + rstd_b + offs_n64, mask=n_mask, other=0.0)[:, None]
                k_n = (k_in.to(tl.float32) * rstd * kw_d[None, :]).to(k_in.dtype)
            else:
                k_n = k_in

            k_relu = tl.where(nd_mask, tl.maximum(k_n.to(tl.float32), 0.0), 0.0)
            ksum_acc += tl.sum(k_relu, axis=0)

            ke, ko = tl.split(tl.reshape(k_relu, (block_n, block_half, 2)))
            cos_offs = cos_b + offs_n64[:, None] * stride_cos_n + offs_half[None, :]
            sin_offs = sin_b + offs_n64[:, None] * stride_sin_n + offs_half[None, :]
            if use_fp64_rope:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float64)
                ke_64 = ke.to(tl.float64)
                ko_64 = ko.to(tl.float64)
                kr_e = (ke_64 * c - ko_64 * s).to(k_in.dtype)
                kr_o = (ko_64 * c + ke_64 * s).to(k_in.dtype)
            else:
                c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float32)
                kr_e = (ke * c - ko * s).to(k_in.dtype)
                kr_o = (ko * c + ke * s).to(k_in.dtype)

            k_rot = tl.reshape(tl.join(kr_e, kr_o), (block_n, block_d))
            out_ptrs = out_bh + offs_n64[:, None] * head_dim + offs_d[None, :]
            tl.store(k_rot_ptr + out_ptrs, k_rot, mask=nd_mask)
            tl.store(v_out_ptr + out_ptrs, v_in, mask=nd_mask)

        tl.store(ksum_part_ptr + (bh * num_chunks + chunk_id) * head_dim + offs_d, ksum_acc, mask=d_mask)

    @triton.jit
    def _sana_litela_prep_q_denom_swizzled_kernel(
        qkv_ptr,
        q_rstd_ptr,
        q_weight_ptr,
        cos_ptr,
        sin_ptr,
        ksum_ptr,
        q_rot_ptr,
        inv_denom_ptr,
        seq_len,
        stride_qkv_b,
        stride_qkv_n,
        stride_cos_b,
        stride_cos_n,
        stride_sin_b,
        stride_sin_n,
        eps,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_half: tl.constexpr,
        block_n: tl.constexpr,
        cos_batch_broadcast: tl.constexpr,
        has_q_norm: tl.constexpr,
        use_fp64_rope: tl.constexpr,
    ):
        """L2-swizzled ``Q`` norm+ReLU+RoPE and inverse linear-attention denominator."""
        bh = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        b = bh // num_heads
        h = bh % num_heads
        b_freq = tl.zeros((), dtype=tl.int64) if cos_batch_broadcast else b

        offs_n = pid_n * block_n + tl.arange(0, block_n)
        n_mask = offs_n < seq_len
        offs_n64 = offs_n.to(tl.int64)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        offs_half = tl.arange(0, block_half)
        half_mask = offs_half < half_dim
        nd_mask = n_mask[:, None] & d_mask[None, :]
        nh_mask = n_mask[:, None] & half_mask[None, :]

        ksum_d = tl.load(ksum_ptr + bh * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        q_col = h * head_dim
        row_ptrs = b * stride_qkv_b + offs_n64[:, None] * stride_qkv_n
        q_in = tl.load(qkv_ptr + row_ptrs + (q_col + offs_d)[None, :], mask=nd_mask, other=0.0)
        if has_q_norm:
            qw_d = tl.load(q_weight_ptr + q_col + offs_d, mask=d_mask, other=0.0).to(tl.float32)
            rstd = tl.load(q_rstd_ptr + b * seq_len + offs_n64, mask=n_mask, other=0.0)[:, None]
            q_n = (q_in.to(tl.float32) * rstd * qw_d[None, :]).to(q_in.dtype)
        else:
            q_n = q_in

        q_relu = tl.where(nd_mask, tl.maximum(q_n.to(tl.float32), 0.0), 0.0)
        denom = tl.sum(q_relu * ksum_d[None, :], axis=1) + eps
        tl.store(inv_denom_ptr + bh * seq_len + offs_n64, 1.0 / denom, mask=n_mask)

        qe, qo = tl.split(tl.reshape(q_relu, (block_n, block_half, 2)))
        cos_offs = b_freq * stride_cos_b + offs_n64[:, None] * stride_cos_n + offs_half[None, :]
        sin_offs = b_freq * stride_sin_b + offs_n64[:, None] * stride_sin_n + offs_half[None, :]
        if use_fp64_rope:
            c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float64)
            s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float64)
            qe_64 = qe.to(tl.float64)
            qo_64 = qo.to(tl.float64)
            qr_e = (qe_64 * c - qo_64 * s).to(q_in.dtype)
            qr_o = (qo_64 * c + qe_64 * s).to(q_in.dtype)
        else:
            c = tl.load(cos_ptr + cos_offs, mask=nh_mask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + sin_offs, mask=nh_mask, other=0.0).to(tl.float32)
            qr_e = (qe * c - qo * s).to(q_in.dtype)
            qr_o = (qo * c + qe * s).to(q_in.dtype)

        q_rot = tl.reshape(tl.join(qr_e, qr_o), (block_n, block_d))
        out_ptrs = (bh * seq_len + offs_n64)[:, None] * head_dim + offs_d[None, :]
        tl.store(q_rot_ptr + out_ptrs, q_rot, mask=nd_mask)

    @triton.jit
    def _sana_litela_div_transpose_swizzled_kernel(
        num_hi_ptr,
        num_lo_ptr,
        inv_denom_ptr,
        out_ptr,
        seq_len,
        stride_out_b,
        stride_out_n,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_n: tl.constexpr,
        has_lo: tl.constexpr,
    ):
        """Combine Dual-BF16 numerator, divide by denominator, and scatter to ``[B, N, C]``."""
        bh = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        b = bh // num_heads
        h = bh % num_heads

        offs_n = pid_n * block_n + tl.arange(0, block_n)
        n_mask = offs_n < seq_len
        offs_n64 = offs_n.to(tl.int64)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        nd_mask = n_mask[:, None] & d_mask[None, :]

        in_ptrs = (bh * seq_len + offs_n64)[:, None] * head_dim + offs_d[None, :]
        num = tl.load(num_hi_ptr + in_ptrs, mask=nd_mask, other=0.0).to(tl.float32)
        if has_lo:
            num += tl.load(num_lo_ptr + in_ptrs, mask=nd_mask, other=0.0).to(tl.float32)
        inv_d = tl.load(inv_denom_ptr + bh * seq_len + offs_n64, mask=n_mask, other=0.0)[:, None]
        val = (num * inv_d).to(out_ptr.dtype.element_ty)

        out_ptrs = b * stride_out_b + offs_n64[:, None] * stride_out_n + (h * head_dim + offs_d)[None, :]
        tl.store(out_ptr + out_ptrs, val, mask=nd_mask)

    return {
        "triton": triton,
        "rms_norm": _sana_rms_norm_kernel,
        "qk_rstd": _sana_qk_rstd_kernel,
        "kv_splitk": _sana_litela_kv_splitk_kernel,
        "q_apply_div": _sana_litela_q_apply_div_kernel,
        "prep_kv": _sana_litela_prep_kv_swizzled_kernel,
        "prep_q_denom": _sana_litela_prep_q_denom_swizzled_kernel,
        "div_transpose": _sana_litela_div_transpose_swizzled_kernel,
    }


def _select_num_splits(batch_heads: int, seq_len: int) -> int:
    """Choose deterministic split-K count to keep H200 SMs saturated."""
    if seq_len <= 256:
        return 1
    if seq_len <= 1024:
        return 4
    if batch_heads * 16 >= 1024:
        return 16
    if batch_heads * 32 >= 1024:
        return 32
    return 64


@torch.library.custom_op("worldfoundry::sana_rms_norm_fwd", mutates_args=(), device_types="cuda")
def _sana_rms_norm_fwd_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]
    features = int(x.shape[-1])
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out
    if x.ndim == 2:
        mat = x
    elif x.is_contiguous():
        mat = x.view(-1, features)
    elif x.ndim == 3 and (x.shape[0] == 1 or x.stride(0) == x.shape[1] * x.stride(1)):
        mat = x.as_strided((x.shape[0] * x.shape[1], features), (x.stride(1), 1))
    else:
        mat = x.contiguous().view(-1, features)
    rows = int(mat.shape[0])
    out_2d = out.view(rows, features)
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    block = max(16, triton.next_power_of_2(features))
    num_warps = 4 if block <= 1024 else 8
    kernels["rms_norm"][(rows,)](
        mat,
        w_c,
        out_2d,
        mat.stride(0),
        out_2d.stride(0),
        float(eps),
        features=features,
        block=block,
        num_warps=num_warps,
    )
    return out


@_sana_rms_norm_fwd_op.register_fake
def _sana_rms_norm_fwd_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    del weight, eps
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def can_use_sana_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return True when the single-pass Hopper Triton SANA RMSNorm op is eligible."""
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    if x.ndim < 2 or x.dtype not in (torch.bfloat16, torch.float16) or x.stride(-1) != 1:
        return False
    features = int(x.shape[-1])
    if features <= 0 or features > 4096:
        return False
    return (
        weight.device == x.device
        and weight.ndim == 1
        and weight.shape[0] == features
        and weight.dtype in _SUPPORTED_DTYPES
    )


def fused_sana_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Single-pass lossless Hopper Triton RMSNorm matching FP32 weight-mul semantics."""
    return _sana_rms_norm_fwd_op(x, weight, float(eps))


@torch.library.custom_op("worldfoundry::sana_litela_relu_rope_fwd", mutates_args=(), device_types="cuda")
def _sana_litela_relu_rope_fwd_op(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    q_eps: float,
    k_eps: float,
    attn_eps: float,
    use_fp32_attention: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    b, n, three_c = qkv.shape
    c = three_c // 3
    out = torch.empty((b, n, c), device=qkv.device, dtype=qkv.dtype)
    if b == 0 or n == 0:
        return out

    qkv_c = qkv if qkv.is_contiguous() else qkv.contiguous()
    cos_3d = cos.squeeze(2) if cos.ndim == 4 else cos
    sin_3d = sin.squeeze(2) if sin.ndim == 4 else sin
    cos_3d = cos_3d if cos_3d.stride(-1) == 1 else cos_3d.contiguous()
    sin_3d = sin_3d if sin_3d.stride(-1) == 1 else sin_3d.contiguous()

    has_q_norm = q_weight is not None
    has_k_norm = k_weight is not None
    if has_q_norm or has_k_norm:
        q_rstd = torch.empty((b * n,), device=qkv.device, dtype=torch.float32)
        k_rstd = torch.empty((b * n,), device=qkv.device, dtype=torch.float32)
        block_c = min(2048, triton.next_power_of_2(c))
        num_warps_rstd = 4 if block_c <= 512 else 8
        kernels["qk_rstd"][(2 * b * n,)](
            qkv_c,
            q_rstd,
            k_rstd,
            b * n,
            qkv_c.stride(1),
            float(q_eps),
            float(k_eps),
            hidden_dim=c,
            block_c=block_c,
            num_warps=num_warps_rstd,
        )
    else:
        q_rstd = out
        k_rstd = out

    qw_c = q_weight.contiguous() if q_weight is not None else out
    kw_c = k_weight.contiguous() if k_weight is not None else out

    bh = b * num_heads
    block_d = triton.next_power_of_2(head_dim)
    block_half = block_d // 2
    use_fp64_rope = cos_3d.dtype == torch.float64 or sin_3d.dtype == torch.float64
    cos_batch_broadcast = cos_3d.shape[0] == 1 and b > 1

    # On long sequences (N >= 1024), Hopper cuBLAS ``bmm(..., out_dtype=fp32)``
    # with L2-swizzled Triton prep/scatter reaches ~2x higher throughput than
    # streaming Triton ``tl.dot`` while keeping FP32 Dual-BF16 accuracy (>120 dB).
    # Restrict Dual-split to BF16 (8-bit exponent matches FP32, zero overflow risk).
    use_hybrid_cublas = n >= 1024 and (
        qkv_c.dtype == torch.bfloat16 or (qkv_c.dtype == torch.float16 and not use_fp32_attention)
    )
    if use_hybrid_cublas:
        k_rot = torch.empty((bh, n, head_dim), device=qkv.device, dtype=qkv_c.dtype)
        v_out = torch.empty((bh, n, head_dim), device=qkv.device, dtype=qkv_c.dtype)
        bn_prep = 32
        tpc_prep = 8
        num_chunks = triton.cdiv(n, bn_prep * tpc_prep)
        ksum_part = torch.empty((bh, num_chunks, head_dim), device=qkv.device, dtype=torch.float32)
        kernels["prep_kv"][(bh, num_chunks)](
            qkv_c,
            k_rstd,
            kw_c,
            cos_3d,
            sin_3d,
            k_rot,
            v_out,
            ksum_part,
            n,
            qkv_c.stride(0),
            qkv_c.stride(1),
            cos_3d.stride(0),
            cos_3d.stride(1),
            sin_3d.stride(0),
            sin_3d.stride(1),
            hidden_dim=c,
            num_heads=num_heads,
            head_dim=head_dim,
            half_dim=head_dim // 2,
            block_d=block_d,
            block_half=block_half,
            block_n=bn_prep,
            tiles_per_cta=tpc_prep,
            num_chunks=num_chunks,
            cos_batch_broadcast=cos_batch_broadcast,
            has_k_norm=has_k_norm,
            use_fp64_rope=use_fp64_rope,
            num_warps=4,
        )
        ksum = ksum_part.sum(dim=1).contiguous()
        kv_f32 = torch.bmm(k_rot.transpose(1, 2), v_out, out_dtype=torch.float32)

        q_rot = torch.empty((bh, n, head_dim), device=qkv.device, dtype=qkv_c.dtype)
        inv_denom = torch.empty((bh, n), device=qkv.device, dtype=torch.float32)
        kernels["prep_q_denom"][(bh, triton.cdiv(n, bn_prep))](
            qkv_c,
            q_rstd,
            qw_c,
            cos_3d,
            sin_3d,
            ksum,
            q_rot,
            inv_denom,
            n,
            qkv_c.stride(0),
            qkv_c.stride(1),
            cos_3d.stride(0),
            cos_3d.stride(1),
            sin_3d.stride(0),
            sin_3d.stride(1),
            float(attn_eps),
            num_heads=num_heads,
            head_dim=head_dim,
            half_dim=head_dim // 2,
            block_d=block_d,
            block_half=block_half,
            block_n=bn_prep,
            cos_batch_broadcast=cos_batch_broadcast,
            has_q_norm=has_q_norm,
            use_fp64_rope=use_fp64_rope,
            num_warps=4,
        )

        if use_fp32_attention and qkv_c.dtype == torch.bfloat16:
            kv_hi = kv_f32.to(torch.bfloat16)
            kv_lo = (kv_f32 - kv_hi.float()).to(torch.bfloat16)
            num_hi = torch.bmm(q_rot, kv_hi, out_dtype=torch.float32)
            num_lo = torch.bmm(q_rot, kv_lo)
            has_lo = True
        else:
            kv_low = kv_f32.to(qkv_c.dtype)
            num_hi = torch.bmm(q_rot, kv_low, out_dtype=torch.float32)
            num_lo = num_hi
            has_lo = False

        kernels["div_transpose"][(bh, triton.cdiv(n, bn_prep))](
            num_hi,
            num_lo,
            inv_denom,
            out,
            n,
            out.stride(0),
            out.stride(1),
            num_heads=num_heads,
            head_dim=head_dim,
            block_d=block_d,
            block_n=bn_prep,
            has_lo=has_lo,
            num_warps=4,
        )
        return out

    num_splits = _select_num_splits(bh, n)
    block_n = 128 if block_d <= 64 else 64
    num_warps_dot = 4 if block_d <= 64 else 8

    partial_kv = torch.empty((b, num_heads, num_splits, block_d, block_d), device=qkv.device, dtype=torch.float32)
    partial_ksum = torch.empty((b, num_heads, num_splits, block_d), device=qkv.device, dtype=torch.float32)

    kernels["kv_splitk"][(bh, num_splits)](
        qkv_c,
        k_rstd,
        kw_c,
        cos_3d,
        sin_3d,
        partial_kv,
        partial_ksum,
        n,
        qkv_c.stride(0),
        qkv_c.stride(1),
        cos_3d.stride(0),
        cos_3d.stride(1),
        sin_3d.stride(0),
        sin_3d.stride(1),
        hidden_dim=c,
        num_heads=num_heads,
        head_dim=head_dim,
        half_dim=head_dim // 2,
        block_d=block_d,
        block_half=block_half,
        num_splits=num_splits,
        block_n=block_n,
        cos_batch_broadcast=cos_batch_broadcast,
        has_k_norm=has_k_norm,
        use_fp64_rope=use_fp64_rope,
        use_fp32_dot=bool(use_fp32_attention or qkv.dtype == torch.float32),
        num_warps=num_warps_dot,
    )

    if num_splits == 1:
        kv = partial_kv.squeeze(2).contiguous()
        ksum = partial_ksum.squeeze(2).contiguous()
    else:
        kv = partial_kv.sum(dim=2).contiguous()
        ksum = partial_ksum.sum(dim=2).contiguous()

    tiles_per_cta = 4 if n >= 2048 else 1
    kernels["q_apply_div"][(bh, triton.cdiv(n, block_n * tiles_per_cta))](
        qkv_c,
        q_rstd,
        qw_c,
        cos_3d,
        sin_3d,
        kv,
        ksum,
        out,
        n,
        qkv_c.stride(0),
        qkv_c.stride(1),
        out.stride(0),
        out.stride(1),
        cos_3d.stride(0),
        cos_3d.stride(1),
        sin_3d.stride(0),
        sin_3d.stride(1),
        float(attn_eps),
        hidden_dim=c,
        num_heads=num_heads,
        head_dim=head_dim,
        half_dim=head_dim // 2,
        block_d=block_d,
        block_half=block_half,
        block_n=block_n,
        tiles_per_cta=tiles_per_cta,
        cos_batch_broadcast=cos_batch_broadcast,
        has_q_norm=has_q_norm,
        use_fp64_rope=use_fp64_rope,
        use_fp32_dot=bool(use_fp32_attention or qkv.dtype == torch.float32),
        num_warps=num_warps_dot,
    )
    return out


@_sana_litela_relu_rope_fwd_op.register_fake
def _sana_litela_relu_rope_fwd_fake(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    q_eps: float,
    k_eps: float,
    attn_eps: float,
    use_fp32_attention: bool,
) -> torch.Tensor:
    del cos, sin, q_weight, k_weight, num_heads, head_dim, q_eps, k_eps, attn_eps, use_fp32_attention
    b, n, three_c = qkv.shape
    return qkv.new_empty((b, n, three_c // 3))


def can_use_triton_litela_relu_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Return True when the fused Triton SANA LiteLA-ReLU-RoPE op is eligible."""
    if not qkv.is_cuda or torch.is_grad_enabled():
        return False
    if qkv.ndim != 3 or qkv.dtype not in _SUPPORTED_DTYPES:
        return False
    if head_dim < 16 or head_dim > 128 or head_dim % 2 != 0 or num_heads * head_dim * 3 != qkv.shape[-1]:
        return False
    if cos.device != qkv.device or sin.device != qkv.device:
        return False
    if cos.shape[-1] != head_dim // 2 or sin.shape[-1] != head_dim // 2:
        return False
    if cos.ndim not in (3, 4) or sin.ndim not in (3, 4):
        return False
    if cos.ndim == 4 and cos.shape[2] != 1:
        return False
    if sin.ndim == 4 and sin.shape[2] != 1:
        return False
    return cos.shape[0] in (1, qkv.shape[0]) and cos.shape[1] == qkv.shape[1] and sin.shape == cos.shape


def fused_sana_litela_relu_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_eps: float = 1e-5,
    k_eps: float = 1e-5,
    attn_eps: float = 1e-15,
    use_fp32_attention: bool = False,
) -> torch.Tensor:
    """Run fused lossless Hopper Triton ``LiteLAReLURope`` attention body."""
    return _sana_litela_relu_rope_fwd_op(
        qkv,
        cos,
        sin,
        q_weight,
        k_weight,
        int(num_heads),
        int(head_dim),
        float(q_eps),
        float(k_eps),
        float(attn_eps),
        bool(use_fp32_attention),
    )


__all__ = [
    "can_use_sana_rms_norm",
    "can_use_triton_litela_relu_rope",
    "fused_sana_litela_relu_rope",
    "fused_sana_rms_norm",
]
