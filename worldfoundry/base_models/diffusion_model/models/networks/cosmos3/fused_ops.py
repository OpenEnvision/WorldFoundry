# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton fused kernels and Hopper FA3 custom ops for Cosmos3-Super MoT layers.

All CUDA custom ops are registered via ``torch.library.custom_op`` with fake
tensor implementations so they compose cleanly with Piecewise CUDA Graphs and
``torch.compile`` while retaining exact PyTorch fallbacks for CPU and autograd.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from worldfoundry.core.attention import rotate_half as _rotate_half

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_rope_gqa_kernel(
        Q_ptr,
        K_ptr,
        QW_ptr,
        KW_ptr,
        COS_ptr,
        SIN_ptr,
        Q_out_ptr,
        K_out_ptr,
        stride_q_s,
        stride_q_h,
        stride_k_s,
        stride_k_h,
        stride_qo_s,
        stride_qo_h,
        stride_ko_s,
        stride_ko_h,
        stride_cos_s,
        stride_sin_s,
        S,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        Q_BLOCKS: tl.constexpr,
        HALF_D: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_H: tl.constexpr,
        eps,
    ):
        """Head-tiled per-head FP32 RMSNorm + rotate_half RoPE across ``BLOCK_H`` heads per CTA."""
        s_idx = tl.program_id(0).to(tl.int64)
        blk_idx = tl.program_id(1)
        if s_idx >= S:
            return

        offs = tl.arange(0, BLOCK_HALF)
        d_mask = offs < HALF_D
        cos0 = tl.load(COS_ptr + s_idx * stride_cos_s + offs, mask=d_mask, other=1.0).to(tl.float32)[None, :]
        cos1 = tl.load(COS_ptr + s_idx * stride_cos_s + HALF_D + offs, mask=d_mask, other=1.0).to(tl.float32)[None, :]
        sin0 = tl.load(SIN_ptr + s_idx * stride_sin_s + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        sin1 = tl.load(SIN_ptr + s_idx * stride_sin_s + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

        if blk_idx < Q_BLOCKS:
            hs = blk_idx * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = hs < H_Q
            base_in = Q_ptr + s_idx * stride_q_s + hs[:, None] * stride_q_h
            base_out = Q_out_ptr + s_idx * stride_qo_s + hs[:, None] * stride_qo_h
            w0 = tl.load(QW_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
            w1 = tl.load(QW_ptr + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        else:
            hs = (blk_idx - Q_BLOCKS) * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = hs < H_K
            base_in = K_ptr + s_idx * stride_k_s + hs[:, None] * stride_k_h
            base_out = K_out_ptr + s_idx * stride_ko_s + hs[:, None] * stride_ko_h
            w0 = tl.load(KW_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
            w1 = tl.load(KW_ptr + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

        mask = h_mask[:, None] & d_mask[None, :]
        x0 = tl.load(base_in + offs[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(base_in + HALF_D + offs[None, :], mask=mask, other=0.0).to(tl.float32)

        # Exact nn.RMSNorm semantics: fp32 mean-square, rsqrt, then weight multiply
        sq_sum = tl.sum(x0 * x0 + x1 * x1, axis=1)
        inv_rms = tl.rsqrt(sq_sum / (2.0 * HALF_D) + eps)[:, None]
        n0 = (x0 * inv_rms) * w0
        n1 = (x1 * inv_rms) * w1

        # Exact _rotate_half RoPE: [n0 * cos0 - n1 * sin0, n1 * cos1 + n0 * sin1]
        r0 = n0 * cos0 - n1 * sin0
        r1 = n1 * cos1 + n0 * sin1

        tl.store(base_out + offs[None, :], r0, mask=mask)
        tl.store(base_out + HALF_D + offs[None, :], r1, mask=mask)

    @triton.jit
    def _rmsnorm_rope_gqa_1cta_kernel(
        Q_ptr,
        K_ptr,
        QW_ptr,
        KW_ptr,
        COS_ptr,
        SIN_ptr,
        Q_out_ptr,
        K_out_ptr,
        stride_q_s,
        stride_q_h,
        stride_k_s,
        stride_k_h,
        stride_qo_s,
        stride_qo_h,
        stride_ko_s,
        stride_ko_h,
        stride_cos_s,
        stride_sin_s,
        S,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        HALF_D: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_H: tl.constexpr,
        eps,
    ):
        """One CTA per token: load ``cos``/``sin`` and weights once and tile across all Q and K heads."""
        s_idx = tl.program_id(0).to(tl.int64)
        if s_idx >= S:
            return

        offs = tl.arange(0, BLOCK_HALF)
        d_mask = offs < HALF_D
        cos0 = tl.load(COS_ptr + s_idx * stride_cos_s + offs, mask=d_mask, other=1.0).to(tl.float32)[None, :]
        cos1 = tl.load(COS_ptr + s_idx * stride_cos_s + HALF_D + offs, mask=d_mask, other=1.0).to(tl.float32)[None, :]
        sin0 = tl.load(SIN_ptr + s_idx * stride_sin_s + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        sin1 = tl.load(SIN_ptr + s_idx * stride_sin_s + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

        qw0 = tl.load(QW_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        qw1 = tl.load(QW_ptr + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        q_tok_in = Q_ptr + s_idx * stride_q_s
        q_tok_out = Q_out_ptr + s_idx * stride_qo_s
        for h_start in tl.static_range(0, H_Q, BLOCK_H):
            hs = h_start + tl.arange(0, BLOCK_H)
            mask = (hs < H_Q)[:, None] & d_mask[None, :]
            base_in = q_tok_in + hs[:, None] * stride_q_h
            base_out = q_tok_out + hs[:, None] * stride_qo_h
            x0 = tl.load(base_in + offs[None, :], mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(base_in + HALF_D + offs[None, :], mask=mask, other=0.0).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, axis=1) / (2.0 * HALF_D) + eps)[:, None]
            n0 = (x0 * inv_rms) * qw0
            n1 = (x1 * inv_rms) * qw1
            tl.store(base_out + offs[None, :], n0 * cos0 - n1 * sin0, mask=mask)
            tl.store(base_out + HALF_D + offs[None, :], n1 * cos1 + n0 * sin1, mask=mask)

        kw0 = tl.load(KW_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        kw1 = tl.load(KW_ptr + HALF_D + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        k_tok_in = K_ptr + s_idx * stride_k_s
        k_tok_out = K_out_ptr + s_idx * stride_ko_s
        for h_start in tl.static_range(0, H_K, BLOCK_H):
            hs = h_start + tl.arange(0, BLOCK_H)
            mask = (hs < H_K)[:, None] & d_mask[None, :]
            base_in = k_tok_in + hs[:, None] * stride_k_h
            base_out = k_tok_out + hs[:, None] * stride_ko_h
            x0 = tl.load(base_in + offs[None, :], mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(base_in + HALF_D + offs[None, :], mask=mask, other=0.0).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, axis=1) / (2.0 * HALF_D) + eps)[:, None]
            n0 = (x0 * inv_rms) * kw0
            n1 = (x1 * inv_rms) * kw1
            tl.store(base_out + offs[None, :], n0 * cos0 - n1 * sin0, mask=mask)
            tl.store(base_out + HALF_D + offs[None, :], n1 * cos1 + n0 * sin1, mask=mask)

    @triton.jit
    def _residual_add_rmsnorm_1pass_kernel(
        R_ptr,
        U_ptr,
        W_ptr,
        RES_OUT_ptr,
        NORM_OUT_ptr,
        stride_r,
        stride_u,
        stride_ro,
        stride_no,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        eps,
    ):
        """Single-pass register-resident ``res_out = residual + update`` and ``rms_norm(res_out, w, eps)``.

        Eliminates the pass-2 HBM reload of ``RES_OUT_ptr`` while preserving exact
        input-dtype rounding on ``res_out`` before the RMS reduction.
        """
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_COLS)
        mask = offs < N_COLS
        r_in = tl.load(R_ptr + row * stride_r + offs, mask=mask, other=0.0)
        u_in = tl.load(U_ptr + row * stride_u + offs, mask=mask, other=0.0)
        s_out = (r_in.to(tl.float32) + u_in.to(tl.float32)).to(r_in.dtype)
        tl.store(RES_OUT_ptr + row * stride_ro + offs, s_out, mask=mask)
        s_f = tl.where(mask, s_out.to(tl.float32), 0.0)
        inv_rms = tl.rsqrt(tl.sum(s_f * s_f, axis=0) / N_COLS + eps)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(NORM_OUT_ptr + row * stride_no + offs, (s_f * inv_rms) * w, mask=mask)

    @triton.jit
    def _residual_add_rmsnorm_kernel(
        R_ptr,
        U_ptr,
        W_ptr,
        RES_OUT_ptr,
        NORM_OUT_ptr,
        stride_r,
        stride_u,
        stride_ro,
        stride_no,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        eps,
    ):
        """Tiled fallback for ``N_COLS > 8192``: ``res_out = residual + update`` and ``rms_norm``."""
        row = tl.program_id(0).to(tl.int64)
        sq_acc = tl.zeros((), dtype=tl.float32)
        for start in tl.static_range(0, N_COLS, BLOCK_COLS):
            offs = start + tl.arange(0, BLOCK_COLS)
            mask = offs < N_COLS
            r_in = tl.load(R_ptr + row * stride_r + offs, mask=mask, other=0.0)
            u_in = tl.load(U_ptr + row * stride_u + offs, mask=mask, other=0.0)
            s_out = (r_in.to(tl.float32) + u_in.to(tl.float32)).to(r_in.dtype)
            tl.store(RES_OUT_ptr + row * stride_ro + offs, s_out, mask=mask)
            s_f = tl.where(mask, s_out.to(tl.float32), 0.0)
            sq_acc += tl.sum(s_f * s_f, axis=0)
        inv_rms = tl.rsqrt(sq_acc / N_COLS + eps)
        for start in tl.static_range(0, N_COLS, BLOCK_COLS):
            offs = start + tl.arange(0, BLOCK_COLS)
            mask = offs < N_COLS
            s = tl.load(RES_OUT_ptr + row * stride_ro + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(NORM_OUT_ptr + row * stride_no + offs, (s * inv_rms) * w, mask=mask)


@torch.library.custom_op("worldfoundry::cosmos3_rmsnorm_rope_gqa", mutates_args=(), device_types="cuda")
def _cosmos3_rmsnorm_rope_gqa_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    s, h_q, d = q.shape
    h_k = k.shape[1]
    half_d = d // 2
    block_half = triton.next_power_of_2(half_d)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if s >= 512:
        max_bh = 4 if d >= 256 else (8 if h_q >= 64 else 16)
        block_h = min(max_bh, triton.next_power_of_2(max(h_q, h_k)))
        _rmsnorm_rope_gqa_1cta_kernel[(s,)](
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
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
            cos.stride(0),
            sin.stride(0),
            s,
            H_Q=h_q,
            H_K=h_k,
            HALF_D=half_d,
            BLOCK_HALF=block_half,
            BLOCK_H=block_h,
            eps=float(eps),
            num_warps=4,
        )
    else:
        max_bh = 4 if d >= 256 else 8
        block_h = min(max_bh, triton.next_power_of_2(max(h_q, h_k)))
        q_blocks = triton.cdiv(h_q, block_h)
        k_blocks = triton.cdiv(h_k, block_h)
        _rmsnorm_rope_gqa_kernel[(s, q_blocks + k_blocks)](
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
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
            cos.stride(0),
            sin.stride(0),
            s,
            H_Q=h_q,
            H_K=h_k,
            Q_BLOCKS=q_blocks,
            HALF_D=half_d,
            BLOCK_HALF=block_half,
            BLOCK_H=block_h,
            eps=float(eps),
            num_warps=4,
        )
    return q_out, k_out


@_cosmos3_rmsnorm_rope_gqa_op.register_fake
def _cosmos3_rmsnorm_rope_gqa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, eps
    return torch.empty_like(q), torch.empty_like(k)


def fused_rmsnorm_rope_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head RMSNorm and ``_rotate_half`` RoPE for GQA ``[S, Hq, D]`` & ``[S, Hkv, D]``."""
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and q.is_cuda
        and k.is_cuda
        and q_weight.device == q.device
        and k_weight.device == q.device
        and cos.device == q.device
        and sin.device == q.device
        and q.ndim == 3
        and k.ndim == 3
        and q.shape[0] == k.shape[0]
        and q.shape[2] == k.shape[2]
        and cos.shape == (q.shape[0], q.shape[2])
        and sin.shape == (q.shape[0], q.shape[2])
        and q_weight.shape == (q.shape[2],)
        and k_weight.shape == (q.shape[2],)
        and q.shape[2] % 2 == 0
        and q.shape[0] > 0
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
        and q.dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return _cosmos3_rmsnorm_rope_gqa_op(
            q,
            k,
            q_weight.contiguous(),
            k_weight.contiguous(),
            cos,
            sin,
            float(eps),
        )

    # Portable exact fallback
    q_n = F.rms_norm(q, (q.shape[-1],), weight=q_weight, eps=eps)
    k_n = F.rms_norm(k, (k.shape[-1],), weight=k_weight, eps=eps)
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q_out = q_n * cos_u + _rotate_half(q_n) * sin_u
    k_out = k_n * cos_u + _rotate_half(k_n) * sin_u
    return q_out, k_out


@torch.library.custom_op("worldfoundry::cosmos3_residual_add_rmsnorm", mutates_args=(), device_types="cuda")
def _cosmos3_residual_add_rmsnorm_op(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = residual.numel() // residual.shape[-1]
    cols = residual.shape[-1]
    r2 = residual.reshape(rows, cols).contiguous()
    u2 = update.reshape(rows, cols).contiguous()
    res_out = torch.empty_like(r2)
    norm_out = torch.empty_like(r2)
    if cols <= 8192:
        block_cols = triton.next_power_of_2(cols)
        _residual_add_rmsnorm_1pass_kernel[(rows,)](
            r2,
            u2,
            weight.contiguous(),
            res_out,
            norm_out,
            r2.stride(0),
            u2.stride(0),
            res_out.stride(0),
            norm_out.stride(0),
            N_COLS=cols,
            BLOCK_COLS=block_cols,
            eps=float(eps),
            num_warps=4,
        )
    else:
        block_cols = 2048
        _residual_add_rmsnorm_kernel[(rows,)](
            r2,
            u2,
            weight.contiguous(),
            res_out,
            norm_out,
            r2.stride(0),
            u2.stride(0),
            res_out.stride(0),
            norm_out.stride(0),
            N_COLS=cols,
            BLOCK_COLS=block_cols,
            eps=float(eps),
            num_warps=8,
        )
    return res_out.reshape(residual.shape), norm_out.reshape(residual.shape)


@_cosmos3_residual_add_rmsnorm_op.register_fake
def _cosmos3_residual_add_rmsnorm_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del update, weight, eps
    return torch.empty_like(residual), torch.empty_like(residual)


def fused_residual_add_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + update, rms_norm(residual + update, weight, eps))`` in one pass."""
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and residual.is_cuda
        and update.is_cuda
        and weight.device == residual.device
        and update.device == residual.device
        and residual.shape == update.shape
        and weight.shape == (residual.shape[-1],)
        and residual.numel() > 0
        and residual.shape[-1] <= 8192
        and residual.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and update.dtype == residual.dtype
    ):
        return _cosmos3_residual_add_rmsnorm_op(residual, update, weight, float(eps))
    res = residual + update
    return res, F.rms_norm(res, (res.shape[-1],), weight=weight, eps=eps)


@torch.library.custom_op("worldfoundry::cosmos3_fa3_attn", mutates_args=(), device_types="cuda")
def _cosmos3_fa3_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    import flash_attn_interface as fa3

    out = fa3.flash_attn_func(q, k, v, causal=bool(causal))
    return out[0] if isinstance(out, tuple) else out


@_cosmos3_fa3_attn_op.register_fake
def _cosmos3_fa3_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    del k, causal
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def cosmos3_fa3_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Run Hopper FlashAttention-3 directly on ``[S, H, D]`` GQA tensors (zero transpose)."""
    out = _cosmos3_fa3_attn_op(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        bool(causal),
    )
    return out.squeeze(0).flatten(-2, -1)


__all__ = [
    "cosmos3_fa3_attention",
    "fused_residual_add_rmsnorm",
    "fused_rmsnorm_rope_gqa",
]
