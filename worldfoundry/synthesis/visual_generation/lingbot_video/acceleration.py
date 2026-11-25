# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for LingBot-Video Dense & MoE transformers.

Composes four orthogonal inference acceleration pillars:

1. **Kernel Fusion & Broadcast AdaLN**:
   - Lossless projection fusion: collapses ``to_q/to_k/to_v`` into one
     contiguous ``qkv_proj`` (eliminating per-forward weight concatenation) and
     collapses ``gate_proj/up_proj`` into ``gate_up_proj`` + Triton ``silu_and_mul``.
   - Compact broadcast AdaLN modulation: runs ``time_modulation`` on ``(B, D)``
     instead of expanding ``(B, S, 6D)`` across 30k+ tokens (eliminating tens
     of gigabytes of HBM traffic per step).
   - Triton fused per-head QK-RMSNorm + interleaved 3D RoPE
     (``worldfoundry::lingbot_qk_rmsnorm_rope``), Triton ``residual_gate_add``,
     and Triton ``layer_norm_scale_shift`` for the output norm.
   - Zero-sync cached 3D RoPE tables ``(cos, sin)`` on unpadded ``B=1`` paths.
2. **Hopper FlashAttention-3 + In-Tree PISA Sparse Attention**:
   - Native Hopper FA3 dispatch (``worldfoundry::lingbot_fa3_attn``) for dense
     self-attention blocks and dense warmup/boundary steps.
   - In-tree PISA (``worldfoundry.core.attention.sparse.piecewise.piecewise_attention``)
     for refiner / long-sequence interior blocks with configurable density,
     block size, dense edge-block guards (e.g. ``(0, 1, -1)``), and dense
     warmup/tail step guards.
3. **Calibrated EasyCache Cross-Step Residual Reuse**:
   - Request/branch-isolated online strided EasyCache over the transformer
     block stack with warmup, dense-last, and max-skip boundary protection.
4. **Piecewise Block-Stack CUDA Graph**:
   - Captures pure-tensor dense and sparse block-stack segments separately in a
     :class:`PiecewiseBlockGraphRunner` outside Python EasyCache gating.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import torch
from torch import nn

from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.attention.sparse.piecewise import (
    piecewise_attention,
    piecewise_attention_available,
)
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _lb_round_f32_to_bf16(value, USE_BF16_PTX: tl.constexpr):
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
    def _lb_round_f32_to_fp16(value):
        return tl.inline_asm_elementwise(
            "{ .reg .b16 rounded; cvt.rn.f16.f32 rounded, $1; cvt.f32.f16 $0, rounded; }",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _lb_round_to_dtype(value, dtype: tl.constexpr, USE_BF16_PTX: tl.constexpr = True):
        if dtype == tl.bfloat16:
            return _lb_round_f32_to_bf16(value, USE_BF16_PTX)
        elif dtype == tl.float16:
            return _lb_round_f32_to_fp16(value)
        else:
            return value

    @triton.jit
    def _lingbot_qk_rmsnorm_rope_kernel(
        Q_ptr,
        K_ptr,
        QW_ptr,
        KW_ptr,
        COS_ptr,
        SIN_ptr,
        Q_out_ptr,
        K_out_ptr,
        stride_q_b,
        stride_q_s,
        stride_q_h,
        stride_k_b,
        stride_k_s,
        stride_k_h,
        stride_qo_b,
        stride_qo_s,
        stride_qo_h,
        stride_ko_b,
        stride_ko_s,
        stride_ko_h,
        stride_cos_b,
        stride_cos_s,
        stride_sin_b,
        stride_sin_s,
        S,
        H: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAIRS: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        eps,
    ):
        """Coalesced 128-bit per-head FP32 RMSNorm + interleaved complex RoPE for [B, S, H, D]."""
        bs_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        b_idx = bs_idx // S
        s_idx = bs_idx % S

        cols = tl.arange(0, BLOCK)
        cmask = cols < HEAD_DIM
        pair_offs = tl.arange(0, BLOCK_PAIRS)
        pmask = pair_offs < PAIRS

        cos = tl.load(COS_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s + pair_offs, mask=pmask, other=1.0).to(
            tl.float32
        )
        sin = tl.load(SIN_ptr + b_idx * stride_sin_b + s_idx * stride_sin_s + pair_offs, mask=pmask, other=0.0).to(
            tl.float32
        )

        is_q = head_idx < H
        if is_q:
            h = head_idx
            base_in = Q_ptr + b_idx * stride_q_b + s_idx * stride_q_s + h * stride_q_h
            base_out = Q_out_ptr + b_idx * stride_qo_b + s_idx * stride_qo_s + h * stride_qo_h
            w_vec = tl.load(QW_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        else:
            h = head_idx - H
            base_in = K_ptr + b_idx * stride_k_b + s_idx * stride_k_s + h * stride_k_h
            base_out = K_out_ptr + b_idx * stride_ko_b + s_idx * stride_ko_s + h * stride_ko_h
            w_vec = tl.load(KW_ptr + cols, mask=cmask, other=0.0).to(tl.float32)

        x_in = tl.load(base_in + cols, mask=cmask, other=0.0)
        x_f = x_in.to(tl.float32)

        # Match LingBotVideoRMSNorm: fp32 variance & weight multiply, then interleaved RoPE.
        sq_sum = tl.sum(x_f * x_f, axis=0)
        inv_rms = _libdevice.rsqrt(sq_sum / HEAD_DIM + eps)
        normed = (x_f * inv_rms) * w_vec

        n_even, n_odd = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
        r_even = n_even * cos - n_odd * sin
        r_odd = n_even * sin + n_odd * cos
        r_vec = tl.reshape(tl.join(r_even, r_odd), (BLOCK,))

        tl.store(base_out + cols, r_vec.to(x_in.dtype), mask=cmask)

    @triton.jit
    def _lingbot_rmsnorm_adaln_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        rows,
        seq_len,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        eps,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Single-pass ``(LingBotVideoRMSNorm(x, w) * scale + shift).to(out_dtype)``."""
        row = tl.program_id(0)
        b_idx = row // seq_len
        s_idx = row % seq_len
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        x_off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        x_f = x_in.to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        inv_rms = _libdevice.rsqrt(tl.sum(x_f * x_f, axis=0) / hidden_size + eps)
        normed = _lb_round_to_dtype((x_f * inv_rms) * w, x_in.dtype, USE_BF16_PTX)

        scale_off = b_idx.to(tl.int64) * scale_sb + s_idx.to(tl.int64) * scale_ss + cols
        shift_off = b_idx.to(tl.int64) * shift_sb + s_idx.to(tl.int64) * shift_ss + cols
        sc = tl.load(scale_ptr + scale_off, mask=mask, other=1.0).to(tl.float32)
        sh = tl.load(shift_ptr + shift_off, mask=mask, other=0.0).to(tl.float32)
        out = normed * sc + sh
        tl.store(out_ptr + x_off, out, mask=mask)

    @triton.jit
    def _lingbot_post_norm_residual_gate_rmsnorm_adaln_kernel(
        x_ptr,
        attn_ptr,
        post_w_ptr,
        gate_ptr,
        norm2_w_ptr,
        scale_ptr,
        shift_ptr,
        x_out_ptr,
        ffn_in_ptr,
        rows,
        seq_len,
        gate_sb,
        gate_ss,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        eps_post,
        eps_norm2,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """2-stage super-fusion: post-attn RMSNorm + gated residual + norm2 RMSNorm + FFN AdaLN."""
        row = tl.program_id(0)
        b_idx = row // seq_len
        s_idx = row % seq_len
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        a_in = tl.load(attn_ptr + off, mask=mask, other=0.0)
        post_w = tl.load(post_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate_off = b_idx.to(tl.int64) * gate_sb + s_idx.to(tl.int64) * gate_ss + cols
        gate = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

        # Stage 1: a_normed = (post_w * (a_f * rsqrt(mean(a_f^2) + eps_post))).to(a.dtype)
        a_f = a_in.to(tl.float32)
        a_inv = _libdevice.rsqrt(tl.sum(a_f * a_f, axis=0) / hidden_size + eps_post)
        a_normed = _lb_round_to_dtype((a_f * a_inv) * post_w, a_in.dtype, USE_BF16_PTX)
        # x_new = x + (gate * a_normed).to(x.dtype)
        upd = _lb_round_to_dtype(gate * a_normed, x_in.dtype, USE_BF16_PTX)
        x_new = _lb_round_to_dtype(x_in.to(tl.float32) + upd, x_in.dtype, USE_BF16_PTX)
        tl.store(x_out_ptr + off, x_new, mask=mask)

        # Stage 2: ffn_in = (LingBotVideoRMSNorm(x_new, norm2_w) * scale_mlp + shift_mlp).to(bulk_dtype)
        norm2_w = tl.load(norm2_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_inv = _libdevice.rsqrt(tl.sum(x_new * x_new, axis=0) / hidden_size + eps_norm2)
        x_normed = _lb_round_to_dtype((x_new * x_inv) * norm2_w, x_in.dtype, USE_BF16_PTX)

        scale_off = b_idx.to(tl.int64) * scale_sb + s_idx.to(tl.int64) * scale_ss + cols
        shift_off = b_idx.to(tl.int64) * shift_sb + s_idx.to(tl.int64) * shift_ss + cols
        sc = tl.load(scale_ptr + scale_off, mask=mask, other=1.0).to(tl.float32)
        sh = tl.load(shift_ptr + shift_off, mask=mask, other=0.0).to(tl.float32)
        ffn_in = x_normed * sc + sh
        tl.store(ffn_in_ptr + off, ffn_in, mask=mask)

    @triton.jit
    def _lingbot_post_norm_residual_gate_add_kernel(
        x_ptr,
        upd_ptr,
        post_w_ptr,
        gate_ptr,
        x_out_ptr,
        rows,
        seq_len,
        gate_sb,
        gate_ss,
        eps_post,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Fused ``x + (gate * LingBotVideoRMSNorm(upd, post_w)).to(x.dtype)``."""
        row = tl.program_id(0)
        b_idx = row // seq_len
        s_idx = row % seq_len
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_in = tl.load(x_ptr + off, mask=mask, other=0.0)
        u_in = tl.load(upd_ptr + off, mask=mask, other=0.0)
        post_w = tl.load(post_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate_off = b_idx.to(tl.int64) * gate_sb + s_idx.to(tl.int64) * gate_ss + cols
        gate = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

        u_f = u_in.to(tl.float32)
        u_inv = _libdevice.rsqrt(tl.sum(u_f * u_f, axis=0) / hidden_size + eps_post)
        u_normed = _lb_round_to_dtype((u_f * u_inv) * post_w, u_in.dtype, USE_BF16_PTX)
        gated = _lb_round_to_dtype(gate * u_normed, x_in.dtype, USE_BF16_PTX)
        x_new = x_in.to(tl.float32) + gated
        tl.store(x_out_ptr + off, x_new, mask=mask)

    @triton.jit
    def _lingbot_layernorm_adaln_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        rows,
        seq_len,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        eps,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Single-pass FP32 LayerNorm (no affine) + AdaLN ``*(1+scale)+shift`` directly to output dtype."""
        row = tl.program_id(0)
        b_idx = row // seq_len
        s_idx = row % seq_len
        cols = tl.arange(0, block)
        mask = (row < rows) & (cols < hidden_size)

        off = row.to(tl.int64) * hidden_size + cols
        x_f = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x_f, axis=0) / hidden_size
        diff = tl.where(mask, x_f - mean, 0.0)
        inv_std = _libdevice.rsqrt(tl.sum(diff * diff, axis=0) / hidden_size + eps)
        normed = diff * inv_std

        scale_off = b_idx.to(tl.int64) * scale_sb + s_idx.to(tl.int64) * scale_ss + cols
        shift_off = b_idx.to(tl.int64) * shift_sb + s_idx.to(tl.int64) * shift_ss + cols
        sc = tl.load(scale_ptr + scale_off, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + shift_off, mask=mask, other=0.0).to(tl.float32)
        out = normed * (1.0 + sc) + sh
        tl.store(out_ptr + off, out, mask=mask)


def _lb_use_bf16_ptx(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def _lb_mod_strides(t: torch.Tensor, batch: int, seq_len: int) -> tuple[int, int]:
    sb = 0 if int(t.shape[0]) == 1 and batch > 1 else int(t.stride(0))
    ss = 0 if int(t.shape[1]) == 1 and seq_len > 1 else int(t.stride(1))
    return sb, ss


def _lb_optimal_warps(block: int) -> int:
    if block <= 2048:
        return 2
    if block <= 4096:
        return 4
    return 8


@torch.library.custom_op("worldfoundry::lingbot_qk_rmsnorm_rope", mutates_args=(), device_types="cuda")
def _lingbot_qk_rmsnorm_rope_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    output_bhsd: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    pairs = d // 2
    block = max(32, int(triton.next_power_of_2(d)))
    block_pairs = block // 2
    if output_bhsd:
        q_out = torch.empty((b, h, s, d), device=q.device, dtype=q.dtype)
        k_out = torch.empty((b, h, s, d), device=k.device, dtype=k.dtype)
        qo_sb, qo_ss, qo_sh = q_out.stride(0), q_out.stride(2), q_out.stride(1)
        ko_sb, ko_ss, ko_sh = k_out.stride(0), k_out.stride(2), k_out.stride(1)
    else:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        qo_sb, qo_ss, qo_sh = q_out.stride(0), q_out.stride(1), q_out.stride(2)
        ko_sb, ko_ss, ko_sh = k_out.stride(0), k_out.stride(1), k_out.stride(2)
    grid = (b * s, 2 * h)
    _lingbot_qk_rmsnorm_rope_kernel[grid](
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
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        qo_sb,
        qo_ss,
        qo_sh,
        ko_sb,
        ko_ss,
        ko_sh,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        s,
        H=h,
        HEAD_DIM=d,
        PAIRS=pairs,
        BLOCK=block,
        BLOCK_PAIRS=block_pairs,
        eps=float(eps),
        num_warps=2 if block <= 128 else 4,
    )
    return q_out, k_out


@_lingbot_qk_rmsnorm_rope_op.register_fake
def _lingbot_qk_rmsnorm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    output_bhsd: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, eps
    if output_bhsd:
        b, s, h, d = q.shape
        return q.new_empty((b, h, s, d)), k.new_empty((b, h, s, d))
    return torch.empty_like(q), torch.empty_like(k)


def _ensure_fp32_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    return t.float().contiguous()


def fused_lingbot_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-6,
    *,
    output_bhsd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coalesced 128-bit per-head FP32 QK-RMSNorm + interleaved 3D RoPE for ``[B, S, H, D]``."""
    if isinstance(rotary_emb, tuple):
        cos, sin = rotary_emb
    else:
        cos = rotary_emb.real.contiguous()
        sin = rotary_emb.imag.contiguous()

    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and q.is_cuda
        and k.is_cuda
        and q.ndim == 4
        and k.shape == q.shape
        and q.shape[-1] % 2 == 0
        and q.numel() > 0
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q.dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        cos_2d = cos if cos.ndim == 3 else cos.reshape(q.shape[0], q.shape[1], -1)
        sin_2d = sin if sin.ndim == 3 else sin.reshape(q.shape[0], q.shape[1], -1)
        return _lingbot_qk_rmsnorm_rope_op(
            q,
            k,
            _ensure_fp32_contig(q_weight),
            _ensure_fp32_contig(k_weight),
            _ensure_fp32_contig(cos_2d),
            _ensure_fp32_contig(sin_2d),
            float(eps),
            bool(output_bhsd),
        )

    # Exact portable fallback matching LingBotVideoRMSNorm + apply_rotary_emb.
    def _norm(val: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        vf = val.float()
        return (weight.float() * (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps))).to(val.dtype)

    qn = _norm(q, q_weight)
    kn = _norm(k, k_weight)
    cos_u = cos.float().unsqueeze(2)
    sin_u = sin.float().unsqueeze(2)

    def _rot(val: torch.Tensor) -> torch.Tensor:
        pairs = val.float().reshape(*val.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        out = torch.stack((even * cos_u - odd * sin_u, even * sin_u + odd * cos_u), dim=-1).flatten(-2)
        return out.to(val.dtype)

    q_res, k_res = _rot(qn), _rot(kn)
    if output_bhsd:
        return q_res.transpose(1, 2).contiguous(), k_res.transpose(1, 2).contiguous()
    return q_res, k_res


@torch.library.custom_op("worldfoundry::lingbot_rmsnorm_adaln", mutates_args=(), device_types="cuda")
def _lingbot_rmsnorm_adaln_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rows = b * s
    out = torch.empty((b, s, h), device=x.device, dtype=out_dtype)
    scale_sb, scale_ss = _lb_mod_strides(scale, b, s)
    shift_sb, shift_ss = _lb_mod_strides(shift, b, s)
    block = max(32, int(triton.next_power_of_2(h)))
    _lingbot_rmsnorm_adaln_kernel[(rows,)](
        x,
        weight,
        scale,
        shift,
        out,
        rows,
        s,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        float(eps),
        hidden_size=h,
        block=block,
        USE_BF16_PTX=_lb_use_bf16_ptx(x.device),
        num_warps=_lb_optimal_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_lingbot_rmsnorm_adaln_op.register_fake
def _lingbot_rmsnorm_adaln_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del weight, scale, shift, eps
    return torch.empty_like(x, dtype=out_dtype)


def fused_lingbot_rmsnorm_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused ``(LingBotVideoRMSNorm(x, weight) * scale + shift).to(out_dtype)``."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and x.is_cuda
        and weight.is_cuda
        and scale.is_cuda
        and shift.is_cuda
        and x.ndim == 3
        and x.is_contiguous()
        and x.numel() > 0
        and 0 < int(x.shape[-1]) <= 8192
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and scale.ndim == 3
        and shift.ndim == 3
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and scale.shape[0] in (1, x.shape[0])
        and scale.shape[1] in (1, x.shape[1])
        and shift.shape[0] in (1, x.shape[0])
        and shift.shape[1] in (1, x.shape[1])
        and x.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and target_dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return _lingbot_rmsnorm_adaln_op(x, weight, scale, shift, float(eps), target_dtype)
    xf = x.float()
    normed = (weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + float(eps)))).to(x.dtype)
    return (normed * scale + shift).to(target_dtype)


@torch.library.custom_op(
    "worldfoundry::lingbot_post_norm_residual_gate_rmsnorm_adaln", mutates_args=(), device_types="cuda"
)
def _lingbot_post_norm_residual_gate_rmsnorm_adaln_op(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    post_weight: torch.Tensor,
    gate_msa: torch.Tensor,
    norm2_weight: torch.Tensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    eps_post: float,
    eps_norm2: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rows = b * s
    x_out = torch.empty_like(x)
    ffn_in = torch.empty((b, s, h), device=x.device, dtype=out_dtype)
    gate_sb, gate_ss = _lb_mod_strides(gate_msa, b, s)
    scale_sb, scale_ss = _lb_mod_strides(scale_mlp, b, s)
    shift_sb, shift_ss = _lb_mod_strides(shift_mlp, b, s)
    block = max(32, int(triton.next_power_of_2(h)))
    _lingbot_post_norm_residual_gate_rmsnorm_adaln_kernel[(rows,)](
        x,
        attn_out,
        post_weight,
        gate_msa,
        norm2_weight,
        scale_mlp,
        shift_mlp,
        x_out,
        ffn_in,
        rows,
        s,
        gate_sb,
        gate_ss,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        float(eps_post),
        float(eps_norm2),
        hidden_size=h,
        block=block,
        USE_BF16_PTX=_lb_use_bf16_ptx(x.device),
        num_warps=_lb_optimal_warps(block),
        enable_fp_fusion=False,
    )
    return x_out, ffn_in


@_lingbot_post_norm_residual_gate_rmsnorm_adaln_op.register_fake
def _lingbot_post_norm_residual_gate_rmsnorm_adaln_fake(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    post_weight: torch.Tensor,
    gate_msa: torch.Tensor,
    norm2_weight: torch.Tensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    eps_post: float,
    eps_norm2: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del attn_out, post_weight, gate_msa, norm2_weight, scale_mlp, shift_mlp, eps_post, eps_norm2
    return torch.empty_like(x), torch.empty_like(x, dtype=out_dtype)


def fused_lingbot_post_norm_residual_gate_rmsnorm_adaln(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    post_weight: torch.Tensor,
    gate_msa: torch.Tensor,
    norm2_weight: torch.Tensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    *,
    eps_post: float = 1e-6,
    eps_norm2: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-stage super-fusion: ``x_new = x + (gate_msa * RMSNorm(attn_out)).to(x.dtype)`` and ``ffn_in``."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and x.is_cuda
        and attn_out.is_cuda
        and post_weight.is_cuda
        and gate_msa.is_cuda
        and norm2_weight.is_cuda
        and scale_mlp.is_cuda
        and shift_mlp.is_cuda
        and x.ndim == 3
        and attn_out.shape == x.shape
        and x.is_contiguous()
        and attn_out.is_contiguous()
        and x.numel() > 0
        and 0 < int(x.shape[-1]) <= 8192
        and post_weight.shape == (x.shape[-1],)
        and post_weight.is_contiguous()
        and norm2_weight.shape == (x.shape[-1],)
        and norm2_weight.is_contiguous()
        and gate_msa.ndim == 3
        and scale_mlp.ndim == 3
        and shift_mlp.ndim == 3
        and gate_msa.stride(-1) == 1
        and scale_mlp.stride(-1) == 1
        and shift_mlp.stride(-1) == 1
        and gate_msa.shape[0] in (1, x.shape[0])
        and gate_msa.shape[1] in (1, x.shape[1])
        and scale_mlp.shape[0] in (1, x.shape[0])
        and scale_mlp.shape[1] in (1, x.shape[1])
        and shift_mlp.shape[0] in (1, x.shape[0])
        and shift_mlp.shape[1] in (1, x.shape[1])
        and x.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and attn_out.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and target_dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return _lingbot_post_norm_residual_gate_rmsnorm_adaln_op(
            x,
            attn_out,
            post_weight,
            gate_msa,
            norm2_weight,
            scale_mlp,
            shift_mlp,
            float(eps_post),
            float(eps_norm2),
            target_dtype,
        )
    af = attn_out.float()
    a_normed = (post_weight * (af * torch.rsqrt(af.pow(2).mean(-1, keepdim=True) + float(eps_post)))).to(attn_out.dtype)
    x_new = x + (gate_msa * a_normed).to(x.dtype)
    xf = x_new.float()
    x_normed = (norm2_weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + float(eps_norm2)))).to(x_new.dtype)
    ffn_in = (x_normed * scale_mlp + shift_mlp).to(target_dtype)
    return x_new, ffn_in


@torch.library.custom_op("worldfoundry::lingbot_post_norm_residual_gate_add", mutates_args=(), device_types="cuda")
def _lingbot_post_norm_residual_gate_add_op(
    x: torch.Tensor,
    upd: torch.Tensor,
    post_weight: torch.Tensor,
    gate: torch.Tensor,
    eps_post: float,
) -> torch.Tensor:
    b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rows = b * s
    x_out = torch.empty_like(x)
    gate_sb, gate_ss = _lb_mod_strides(gate, b, s)
    block = max(32, int(triton.next_power_of_2(h)))
    _lingbot_post_norm_residual_gate_add_kernel[(rows,)](
        x,
        upd,
        post_weight,
        gate,
        x_out,
        rows,
        s,
        gate_sb,
        gate_ss,
        float(eps_post),
        hidden_size=h,
        block=block,
        USE_BF16_PTX=_lb_use_bf16_ptx(x.device),
        num_warps=_lb_optimal_warps(block),
        enable_fp_fusion=False,
    )
    return x_out


@_lingbot_post_norm_residual_gate_add_op.register_fake
def _lingbot_post_norm_residual_gate_add_fake(
    x: torch.Tensor,
    upd: torch.Tensor,
    post_weight: torch.Tensor,
    gate: torch.Tensor,
    eps_post: float,
) -> torch.Tensor:
    del upd, post_weight, gate, eps_post
    return torch.empty_like(x)


def fused_lingbot_post_norm_residual_gate_add(
    x: torch.Tensor,
    upd: torch.Tensor,
    post_weight: torch.Tensor,
    gate: torch.Tensor,
    *,
    eps_post: float = 1e-6,
) -> torch.Tensor:
    """Fused ``x + (gate * LingBotVideoRMSNorm(upd, post_weight)).to(x.dtype)``."""
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and x.is_cuda
        and upd.is_cuda
        and post_weight.is_cuda
        and gate.is_cuda
        and x.ndim == 3
        and upd.shape == x.shape
        and x.is_contiguous()
        and upd.is_contiguous()
        and x.numel() > 0
        and 0 < int(x.shape[-1]) <= 8192
        and post_weight.shape == (x.shape[-1],)
        and post_weight.is_contiguous()
        and gate.ndim == 3
        and gate.stride(-1) == 1
        and gate.shape[0] in (1, x.shape[0])
        and gate.shape[1] in (1, x.shape[1])
        and x.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and upd.dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return _lingbot_post_norm_residual_gate_add_op(x, upd, post_weight, gate, float(eps_post))
    uf = upd.float()
    u_normed = (post_weight * (uf * torch.rsqrt(uf.pow(2).mean(-1, keepdim=True) + float(eps_post)))).to(upd.dtype)
    return x + (gate * u_normed).to(x.dtype)


@torch.library.custom_op("worldfoundry::lingbot_layernorm_adaln", mutates_args=(), device_types="cuda")
def _lingbot_layernorm_adaln_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rows = b * s
    out = torch.empty((b, s, h), device=x.device, dtype=out_dtype)
    scale_sb, scale_ss = _lb_mod_strides(scale, b, s)
    shift_sb, shift_ss = _lb_mod_strides(shift, b, s)
    block = max(32, int(triton.next_power_of_2(h)))
    _lingbot_layernorm_adaln_kernel[(rows,)](
        x,
        scale,
        shift,
        out,
        rows,
        s,
        scale_sb,
        scale_ss,
        shift_sb,
        shift_ss,
        float(eps),
        hidden_size=h,
        block=block,
        num_warps=_lb_optimal_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_lingbot_layernorm_adaln_op.register_fake
def _lingbot_layernorm_adaln_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del scale, shift, eps
    return torch.empty_like(x, dtype=out_dtype)


def fused_lingbot_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Single-pass FP32 LayerNorm (no affine) + AdaLN ``*(1 + scale) + shift`` directly to ``out_dtype``."""
    target_dtype = scale.dtype if out_dtype is None else out_dtype
    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and x.is_cuda
        and scale.is_cuda
        and shift.is_cuda
        and x.ndim == 3
        and x.is_contiguous()
        and x.numel() > 0
        and 0 < int(x.shape[-1]) <= 8192
        and scale.ndim == 3
        and shift.ndim == 3
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and scale.shape[0] in (1, x.shape[0])
        and scale.shape[1] in (1, x.shape[1])
        and shift.shape[0] in (1, x.shape[0])
        and shift.shape[1] in (1, x.shape[1])
        and x.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and target_dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return _lingbot_layernorm_adaln_op(x, scale, shift, float(eps), target_dtype)
    normed = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight=None, bias=None, eps=float(eps))
    return (normed * (1.0 + scale) + shift).to(target_dtype)


@torch.library.custom_op("worldfoundry::lingbot_fa3_attn", mutates_args=(), device_types="cuda")
def _lingbot_fa3_attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import flash_attn_interface as fa3

    out = fa3.flash_attn_func(q, k, v, causal=False)
    return out[0] if isinstance(out, tuple) else out


@_lingbot_fa3_attn_op.register_fake
def _lingbot_fa3_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    del k
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def lingbot_fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run Hopper FlashAttention-3 on ``[B, S, H, D]`` tensors."""
    return _lingbot_fa3_attn_op(q.contiguous(), k.contiguous(), v.contiguous())


def lingbot_pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    density: float = 0.25,
    block_size: int = 64,
    min_sequence_length: int = 1024,
    strict: bool = False,
    q_k_bhsd: bool = False,
) -> torch.Tensor:
    """Run in-tree PISA piecewise sparse attention returning ``[B, S, H, D]``."""
    q_in = q if q_k_bhsd else q.transpose(1, 2).contiguous()
    k_in = k if q_k_bhsd else k.transpose(1, 2).contiguous()
    v_in = v.transpose(1, 2).contiguous()
    out_bhsd = piecewise_attention(
        q_in,
        k_in,
        v_in,
        density=float(density),
        block_size=int(block_size),
        min_sequence_length=int(min_sequence_length),
        strict=bool(strict),
    )
    return out_bhsd.transpose(1, 2).contiguous()


@dataclass(frozen=True, slots=True)
class LingBotAccelerationConfig:
    """Declarative SOL-Engine acceleration recipe for LingBot-Video."""

    fuse_qkv: bool = True
    fuse_swiglu: bool = True
    compact_adaln: bool = True
    fused_kernels: bool = True
    attention_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3"
    # In-tree PISA sparse attention for refiner / long-sequence interior blocks.
    pisa_enabled: bool = False
    pisa_density: float = 0.25
    pisa_block_size: int = 64
    pisa_min_sequence_length: int = 1024
    pisa_dense_steps: int = 6
    pisa_dense_last_steps: int = 2
    pisa_dense_blocks: tuple[int, ...] = (0, 1, -1)
    pisa_strict: bool = False
    # Calibrated EasyCache cross-step residual reuse.
    easycache: bool = False
    easycache_threshold: float = 0.12
    easycache_warmup_steps: int = 4
    easycache_dense_last: int = 2
    easycache_max_skip_steps: int = 3
    easycache_subsample_stride: int = 64
    # Piecewise CUDA Graph over the block stack.
    cuda_graph: Literal["off", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pisa_dense_blocks"] = list(self.pisa_dense_blocks)
        payload["notes"] = list(self.notes)
        return payload


class LingBotEasyCache:
    """Branch-isolated online strided EasyCache over LingBot's transformer block stack."""

    algorithm = "easycache"

    def __init__(
        self,
        *,
        threshold: float = 0.12,
        total_steps: int = 40,
        warmup_steps: int = 4,
        dense_last: int = 2,
        max_skip_steps: int = 3,
        subsample_stride: int = 64,
        eps: float = 1e-8,
    ) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("LingBotEasyCache threshold must be finite and non-negative")
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_skip_steps = int(max_skip_steps)
        self.subsample_stride = int(subsample_stride)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._previous_input: torch.Tensor | None = None
        self._dense_input: torch.Tensor | None = None
        self._dense_output: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._factor: torch.Tensor | None = None
        self._accumulator: float = 0.0
        self._skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device, int] | None = None
        self.events: list[BlockCacheEvent] = []

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.detach().reshape(value.shape[0], -1)
        if self.subsample_stride > 1 and flat.shape[1] >= self.subsample_stride:
            flat = flat[:, :: self.subsample_stride]
        return flat.float().clone()

    def run_stack(
        self,
        *,
        step: int,
        total_steps: int | None,
        value: torch.Tensor,
        block_count: int,
        run_stack_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        count = self.total_steps if total_steps is None else int(total_steps)
        layout = (tuple(value.shape), value.dtype, value.device, int(block_count))
        probe = self._probe(value)
        reason: str | None = None
        if torch.is_grad_enabled():
            self._residual = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._residual = None
            self._factor = None
            self._dense_input = None
            self._dense_output = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif step >= max(count - self.dense_last, 0):
            reason = "dense-last"
        elif (
            self._residual is None or self._factor is None or self._previous_input is None or self._dense_output is None
        ):
            reason = "seed"
        elif self._skips >= self.max_skip_steps:
            reason = "max-skip"

        if reason is None:
            delta = (probe - cast(torch.Tensor, self._previous_input)).abs().mean(dim=1)
            norm = cast(torch.Tensor, self._dense_output).abs().mean(dim=1).clamp_min(self.eps)
            estimate = float((cast(torch.Tensor, self._factor) * delta / norm).max().item())
            self._accumulator += estimate
            if not math.isfinite(self._accumulator) or self._accumulator >= self.threshold:
                reason = "threshold"

        hit = reason is None
        if hit:
            output = value + cast(torch.Tensor, self._residual)
            self._skips += 1
        else:
            output = run_stack_fn(value)
            if not torch.is_grad_enabled():
                out_probe = self._probe(output)
                if self._dense_input is not None and self._dense_output is not None:
                    in_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.eps)
                    self._factor = (out_probe - self._dense_output).abs().mean(dim=1) / in_delta
                self._dense_input = probe
                self._dense_output = out_probe
                self._residual = (output - value).detach().clone()
                self._layout = layout
            self._accumulator = 0.0
            self._skips = 0

        if not torch.is_grad_enabled():
            self._previous_input = probe
        indices = tuple(range(int(block_count)))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return output, hit

    @property
    def hits(self) -> int:
        return sum(1 for ev in self.events if ev.hit)

    @property
    def dense_block_calls(self) -> int:
        return sum(ev.dense_block_calls for ev in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(ev.skipped_block_calls for ev in self.events)

    def receipt(self) -> dict[str, Any]:
        total = len(self.events)
        hits = self.hits
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_skip_steps": self.max_skip_steps,
            "subsample_stride": self.subsample_stride,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


def _fuse_mlp_gate_up(mlp: nn.Module) -> bool:
    if (
        getattr(mlp, "gate_up_proj", None) is None
        and isinstance(getattr(mlp, "gate_proj", None), nn.Linear)
        and isinstance(getattr(mlp, "up_proj", None), nn.Linear)
    ):
        gate_proj = cast(nn.Linear, mlp.gate_proj)
        up_proj = cast(nn.Linear, mlp.up_proj)
        with torch.no_grad():
            w = torch.cat((gate_proj.weight.data, up_proj.weight.data), dim=0).contiguous()
        proj = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        proj.weight = nn.Parameter(w, requires_grad=False)
        mlp.gate_up_proj = proj
        return True
    return False


def fuse_lingbot_projections(
    model: nn.Module,
    *,
    fuse_qkv: bool = True,
    fuse_swiglu: bool = True,
) -> dict[str, int]:
    """Losslessly pre-fuse LingBot attention QKV and dense/shared-expert MLP gate+up projections."""
    qkv_fused = 0
    swiglu_fused = 0
    for block in getattr(model, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None:
            if not fuse_qkv:
                attn.qkv_proj = None
            elif (
                getattr(attn, "qkv_proj", None) is None
                and isinstance(getattr(attn, "to_q", None), nn.Linear)
                and isinstance(getattr(attn, "to_k", None), nn.Linear)
                and isinstance(getattr(attn, "to_v", None), nn.Linear)
            ):
                with torch.no_grad():
                    w = torch.cat(
                        (attn.to_q.weight.data, attn.to_k.weight.data, attn.to_v.weight.data), dim=0
                    ).contiguous()
                    b = (
                        torch.cat((attn.to_q.bias.data, attn.to_k.bias.data, attn.to_v.bias.data), dim=0).contiguous()
                        if attn.to_q.bias is not None and attn.to_k.bias is not None and attn.to_v.bias is not None
                        else None
                    )
                proj = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device=w.device, dtype=w.dtype)
                proj.weight = nn.Parameter(w, requires_grad=False)
                if b is not None:
                    proj.bias = nn.Parameter(b, requires_grad=False)
                attn.qkv_proj = proj
                qkv_fused += 1
            elif getattr(attn, "qkv_proj", None) is not None:
                qkv_fused += 1

        ffn = getattr(block, "ffn", None)
        if ffn is not None:
            candidates = [ffn]
            shared = getattr(ffn, "shared_experts", None)
            if isinstance(shared, nn.Module):
                candidates.append(shared)
            for mlp in candidates:
                if not fuse_swiglu:
                    if hasattr(mlp, "gate_up_proj"):
                        mlp.gate_up_proj = None
                elif _fuse_mlp_gate_up(mlp) or getattr(mlp, "gate_up_proj", None) is not None:
                    swiglu_fused += 1

    return {"qkv_fused": qkv_fused, "swiglu_fused": swiglu_fused}


def _resolve_fa3_usable(sm_major: int) -> bool:
    if int(sm_major) < 9:
        return False
    from worldfoundry.core.attention.backends.probe import attention_backend_capability

    return bool(attention_backend_capability("flash_attention_3").usable)


def lingbot_sol_profile(
    tier: Literal["exact", "pisa", "sol_high_fidelity", "sol_balanced", "refiner_sol"] | str,
    *,
    sm_major: int = 9,
    cuda_graph: Literal["off", "piecewise"] | None = None,
) -> LingBotAccelerationConfig:
    """Return a validated SOL-Engine preset for LingBot-Video."""
    normalized = str(tier).strip().lower()
    fa3_ok = _resolve_fa3_usable(sm_major)
    attn_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3" if fa3_ok else "torch"

    if normalized == "exact":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=False,
            easycache=False,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"lossless LingBot QKV/SwiGLU/AdaLN/QK-RoPE fusion + {attn_backend} + piecewise graph",),
        )
    if normalized == "pisa":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.25,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=6,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=False,
            cuda_graph=cuda_graph or "piecewise",
            notes=(
                f"LingBot kernel fusion + {attn_backend} + PISA(density=0.25, edge 0/1/-1 dense) + piecewise graph",
            ),
        )
    if normalized == "sol_high_fidelity":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.35,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=6,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=True,
            easycache_threshold=0.06,
            easycache_warmup_steps=5,
            easycache_dense_last=3,
            easycache_max_skip_steps=2,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"LingBot kernel fusion + PISA(0.35) + EasyCache(0.06) + {attn_backend} + piecewise graph",),
        )
    if normalized in {"sol_balanced", "refiner_sol"}:
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.25,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=4 if normalized == "refiner_sol" else 5,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=True,
            easycache_threshold=0.18,
            easycache_warmup_steps=4,
            easycache_dense_last=2,
            easycache_max_skip_steps=3,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"LingBot kernel fusion + PISA(0.25) + EasyCache(0.18) + {attn_backend} + piecewise graph",),
        )
    raise ValueError(
        f"unsupported LingBot acceleration tier {tier!r}; "
        "expected one of: exact, pisa, sol_high_fidelity, sol_balanced, refiner_sol"
    )


def accelerate_lingbot_vae(
    vae: nn.Module,
    *,
    channels_last_3d: bool = True,
    cast_bf16: bool = True,
) -> dict[str, Any]:
    """Accelerate LingBot's Wan VideoVAE with BF16 decode and ``channels_last_3d`` weights."""
    param = next(vae.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    bf16_applied = False
    if cast_bf16 and is_cuda:
        vae.to(dtype=torch.bfloat16)
        bf16_applied = True
    converted_3d = 0
    if channels_last_3d:
        for module in vae.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.to(memory_format=torch.channels_last_3d)
                converted_3d += 1
    return {
        "channels_last_3d": bool(channels_last_3d),
        "conv3d_layers_converted": converted_3d,
        "vae_bf16": bf16_applied,
        "fallbacks": [],
    }


def accelerate_lingbot_video(
    model: nn.Module,
    config: LingBotAccelerationConfig | str,
    *,
    sm_major: int | None = None,
    vae: nn.Module | None = None,
) -> dict[str, Any]:
    """Install the LingBot-Video SOL-Engine acceleration stack and return an auditable receipt."""
    param = next(model.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    detected_sm = (
        torch.cuda.get_device_capability(param.device)[0]
        if is_cuda and param is not None
        else (sm_major if sm_major is not None else 0)
    )
    resolved = lingbot_sol_profile(config, sm_major=detected_sm) if isinstance(config, str) else config
    fallbacks: list[str] = []

    # 1. Projection fusion (QKV + SwiGLU).
    fusion_receipt = fuse_lingbot_projections(
        model,
        fuse_qkv=resolved.fuse_qkv,
        fuse_swiglu=resolved.fuse_swiglu,
    )

    # 2. Resolve Hopper FA3 and in-tree PISA eligibility.
    effective_attn = resolved.attention_backend
    if effective_attn == "flash_attention_3" and not (is_cuda and _resolve_fa3_usable(detected_sm)):
        effective_attn = "torch"
        if is_cuda:
            fallbacks.append("attention:flash_attention_3->torch")

    pisa_ready = bool(
        resolved.pisa_enabled and is_cuda and param is not None and piecewise_attention_available(param.device)
    )
    if resolved.pisa_enabled and is_cuda and not pisa_ready:
        if resolved.pisa_strict:
            raise RuntimeError("PISA sparse attention requested with strict=True but TMA kernel is unavailable")
        fallbacks.append("sparse_attn:pisa->dense")

    blocks = getattr(model, "blocks", ())
    num_blocks = len(blocks)
    has_moe = any(
        type(getattr(block, "ffn", None)).__name__ == "LingBotVideoSparseMoeBlock"
        or hasattr(getattr(block, "ffn", None), "router")
        for block in blocks
    )
    resolved_dense_blocks: set[int] = set()
    for b in resolved.pisa_dense_blocks:
        idx = num_blocks + int(b) if int(b) < 0 else int(b)
        if 0 <= idx < num_blocks:
            resolved_dense_blocks.add(idx)

    # 3. Configure per-block kernel fusion, FA3, and PISA metadata.
    model._lingbot_has_moe = has_moe
    model._lingbot_compact_adaln = bool(resolved.compact_adaln)
    model._lingbot_fused_kernels = bool(resolved.fused_kernels)
    model._lingbot_pisa_policy = (
        {
            "enabled": bool(pisa_ready),
            "requested": True,
            "tma_available": pisa_ready,
            "density": float(resolved.pisa_density),
            "block_size": int(resolved.pisa_block_size),
            "min_sequence_length": int(resolved.pisa_min_sequence_length),
            "dense_steps": int(resolved.pisa_dense_steps),
            "dense_last_steps": int(resolved.pisa_dense_last_steps),
            "dense_blocks": sorted(resolved_dense_blocks),
            "strict": bool(resolved.pisa_strict),
        }
        if resolved.pisa_enabled
        else None
    )
    for idx, block in enumerate(blocks):
        block._use_fused_kernels = bool(resolved.fused_kernels)
        attn = getattr(block, "attn", None)
        if attn is not None:
            attn._attention_backend = effective_attn
            attn._use_fused_kernels = bool(resolved.fused_kernels)
            attn._pisa_block_eligible = bool(pisa_ready and idx not in resolved_dense_blocks)
            attn._pisa_density = float(resolved.pisa_density)
            attn._pisa_block_size = int(resolved.pisa_block_size)
            attn._pisa_min_sequence_length = int(resolved.pisa_min_sequence_length)
            attn._pisa_strict = bool(resolved.pisa_strict)
            attn._fa3_calls = 0
            attn._pisa_calls = 0
            attn._sdpa_calls = 0

    # 4. Calibrated EasyCache spec.
    if resolved.easycache:
        model._lingbot_easycache_spec = {
            "threshold": float(resolved.easycache_threshold),
            "warmup_steps": int(resolved.easycache_warmup_steps),
            "dense_last": int(resolved.easycache_dense_last),
            "max_skip_steps": int(resolved.easycache_max_skip_steps),
            "subsample_stride": int(resolved.easycache_subsample_stride),
        }
        model._lingbot_easycaches = {}
    else:
        model._lingbot_easycache_spec = None
        model._lingbot_easycaches = {}

    # 5. Piecewise CUDA Graph runner for dense/sparse block stack segments.
    # Sparse MoE blocks perform dynamic token routing that is not CUDA-graph safe,
    # so keep MoE models cleanly eager (zero capture failures / zero RNG corruption).
    effective_cuda_graph = "off" if has_moe else resolved.cuda_graph
    if effective_cuda_graph == "piecewise":
        model._lingbot_piecewise_graph = PiecewiseBlockGraphRunner(
            warmup=resolved.cuda_graph_warmup,
            max_graphs_per_segment=resolved.cuda_graph_max_graphs,
            clone_outputs=resolved.cuda_graph_clone_outputs,
            enabled=True,
        )
        graph_receipt: dict[str, Any] = {"mode": "piecewise"}
    else:
        model._lingbot_piecewise_graph = None
        graph_receipt = (
            {"mode": "off", "reason": "moe-dynamic-routing"}
            if has_moe and resolved.cuda_graph == "piecewise"
            else {"mode": "off"}
        )

    model._lingbot_fa3_replay_calls = 0
    model._lingbot_pisa_replay_calls = 0
    vae_receipt = accelerate_lingbot_vae(vae) if vae is not None else None

    receipt: dict[str, Any] = {
        "engine": "worldfoundry-sol-lingbot",
        "config": resolved.to_dict(),
        "projection_fusion": fusion_receipt,
        "attention": {
            "dense_backend": effective_attn,
            "fused_kernels": bool(resolved.fused_kernels),
            "compact_adaln": bool(resolved.compact_adaln),
            "pisa": model._lingbot_pisa_policy or {"enabled": False},
        },
        "easycache": dict(model._lingbot_easycache_spec) if model._lingbot_easycache_spec else {"enabled": False},
        "cuda_graph": graph_receipt,
        "vae": vae_receipt,
        "fallbacks": fallbacks,
    }
    model._lingbot_acceleration_receipt = receipt
    return receipt


def lingbot_runtime_receipt(model: nn.Module) -> dict[str, Any]:
    """Collect auditable live runtime receipts from an accelerated LingBot-Video transformer."""
    base = dict(
        getattr(model, "_lingbot_acceleration_receipt", {"engine": "worldfoundry-sol-lingbot", "fallbacks": []})
    )
    fallbacks = list(base.get("fallbacks", []))
    fa3_calls = 0
    pisa_calls = 0
    sdpa_calls = 0
    for block in getattr(model, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None:
            fa3_calls += int(getattr(attn, "_fa3_calls", 0))
            pisa_calls += int(getattr(attn, "_pisa_calls", 0))
            sdpa_calls += int(getattr(attn, "_sdpa_calls", 0))

    caches = getattr(model, "_lingbot_easycaches", {}) or {}
    piecewise = getattr(model, "_lingbot_piecewise_graph", None)
    graph_report = piecewise.report() if piecewise is not None else None
    if graph_report is not None:
        failed = sum(int(seg.get("capture_failed", 0)) for seg in graph_report.get("segments", {}).values())
        if failed > 0:
            fallbacks.append(f"cuda_graph:capture_failed={failed}")

    base["fallbacks"] = fallbacks
    effective_fa3 = fa3_calls + int(getattr(model, "_lingbot_fa3_replay_calls", 0))
    effective_pisa = pisa_calls + int(getattr(model, "_lingbot_pisa_replay_calls", 0))
    base["runtime"] = {
        "fa3_calls": effective_fa3,
        "pisa_calls": effective_pisa,
        "sdpa_calls": sdpa_calls,
        "easycache_branches": {branch: cache.receipt() for branch, cache in caches.items()},
        "piecewise_graph": graph_report,
    }
    return base


__all__ = [
    "LingBotAccelerationConfig",
    "LingBotEasyCache",
    "accelerate_lingbot_vae",
    "accelerate_lingbot_video",
    "fuse_lingbot_projections",
    "fused_lingbot_qk_rmsnorm_rope",
    "lingbot_fa3_attention",
    "lingbot_pisa_attention",
    "lingbot_runtime_receipt",
    "lingbot_sol_profile",
]
