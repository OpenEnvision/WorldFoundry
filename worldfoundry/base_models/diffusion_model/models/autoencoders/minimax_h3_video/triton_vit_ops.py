# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper SM90 Triton operators for the MiniMax H3 ViT3D VideoVAE decoder.

Fuses the high-overhead per-layer normalization, rotary embedding, and scaled
residual paths across the 36-layer ViT3D decoder:

- :func:`vit_fused_qk_rmsnorm_rope` — single-launch strided per-head Q/K RMSNorm
  + partial NeoX 3-D RoPE on ``[B, S, H, D]`` slices directly from fused
  ``to_qkv`` without intermediate casts, slices, or ``torch.cat`` copies.
- :func:`vit_scaled_residual_add_rmsnorm` — single-pass register-resident
  ``h_out = residual + update * scale`` + FP32-accumulated ``RMSNorm(h_out)``
  supporting both intra-block (``attn -> norm2``) and inter-block
  (``ff -> next_block.norm1``) fusion without FP32 tensor roundtrips.
- :func:`vit_rmsnorm` — single-pass FP32-accumulated ``RMSNorm`` for block-0
  entry without ``hidden_states.float()`` allocation.

All CUDA custom ops register ``torch.library.custom_op`` + ``register_fake``
for ``torch.compile`` and CUDA Graph safety. Triton is imported lazily on first
CUDA call; CPU and autograd fall back transparently to exact PyTorch math.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Triton kernels on first eligible CUDA invocation."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="minimax-h3-vit-vae-triton")

    import triton
    import triton.language as tl

    @triton.jit
    def _vit_qk_rmsnorm_rope_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        qw_ptr,
        kw_ptr,
        qo_ptr,
        ko_ptr,
        q_sb,
        q_ss,
        q_sh,
        k_sb,
        k_ss,
        k_sh,
        cos_sb,
        cos_ss,
        sin_sb,
        sin_ss,
        qo_sb,
        qo_ss,
        qo_sh,
        ko_sb,
        ko_ss,
        ko_sh,
        seq_len,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROT_HALF: tl.constexpr,
        PASS_DIM: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_PASS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        COS_BATCH_BROADCAST: tl.constexpr,
    ):
        """One CTA per ``(b, s)`` token: Q/K per-head RMSNorm + partial NeoX RoPE."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        s = row % seq_len
        b_cs = tl.zeros((), dtype=tl.int64) if COS_BATCH_BROADCAST else b

        offs_h = tl.arange(0, BLOCK_HALF)
        rot_mask = offs_h < ROT_HALF

        cos_base = cos_ptr + b_cs * cos_sb + s * cos_ss
        sin_base = sin_ptr + b_cs * sin_sb + s * sin_ss
        out_dtype = qo_ptr.dtype.element_ty
        # Match _apply_rotary_pos_emb_impl: cos/sin cast to query dtype before multiply.
        cos_h = tl.load(cos_base + offs_h, mask=rot_mask, other=1.0).to(out_dtype).to(tl.float32)[None, :]
        sin_h = tl.load(sin_base + offs_h, mask=rot_mask, other=0.0).to(out_dtype).to(tl.float32)[None, :]

        if HAS_WEIGHT:
            qw1 = tl.load(qw_ptr + offs_h, mask=rot_mask, other=1.0).to(tl.float32)[None, :]
            qw2 = tl.load(qw_ptr + ROT_HALF + offs_h, mask=rot_mask, other=1.0).to(tl.float32)[None, :]
            kw1 = tl.load(kw_ptr + offs_h, mask=rot_mask, other=1.0).to(tl.float32)[None, :]
            kw2 = tl.load(kw_ptr + ROT_HALF + offs_h, mask=rot_mask, other=1.0).to(tl.float32)[None, :]

        if PASS_DIM > 0:
            offs_p = tl.arange(0, BLOCK_PASS)
            pass_mask = offs_p < PASS_DIM
            rot_dim: tl.constexpr = 2 * ROT_HALF
            if HAS_WEIGHT:
                qwp = tl.load(qw_ptr + rot_dim + offs_p, mask=pass_mask, other=1.0).to(tl.float32)[None, :]
                kwp = tl.load(kw_ptr + rot_dim + offs_p, mask=pass_mask, other=1.0).to(tl.float32)[None, :]

        q_tok = q_ptr + b * q_sb + s * q_ss
        k_tok = k_ptr + b * k_sb + s * k_ss
        qo_tok = qo_ptr + b * qo_sb + s * qo_ss
        ko_tok = ko_ptr + b * ko_sb + s * ko_ss
        inv_hd: tl.constexpr = 1.0 / HEAD_DIM

        for h0 in tl.static_range(0, NUM_HEADS, BLOCK_H):
            hs = h0 + tl.arange(0, BLOCK_H)
            h_mask = hs < NUM_HEADS
            mask_h = h_mask[:, None] & rot_mask[None, :]
            qb = q_tok + hs[:, None] * q_sh
            kb = k_tok + hs[:, None] * k_sh
            qob = qo_tok + hs[:, None] * qo_sh
            kob = ko_tok + hs[:, None] * ko_sh

            q1 = tl.load(qb + offs_h[None, :], mask=mask_h, other=0.0).to(tl.float32)
            q2 = tl.load(qb + ROT_HALF + offs_h[None, :], mask=mask_h, other=0.0).to(tl.float32)
            k1 = tl.load(kb + offs_h[None, :], mask=mask_h, other=0.0).to(tl.float32)
            k2 = tl.load(kb + ROT_HALF + offs_h[None, :], mask=mask_h, other=0.0).to(tl.float32)

            q_sq = tl.sum(q1 * q1 + q2 * q2, axis=1)
            k_sq = tl.sum(k1 * k1 + k2 * k2, axis=1)

            if PASS_DIM > 0:
                mask_p = h_mask[:, None] & pass_mask[None, :]
                qp = tl.load(qb + rot_dim + offs_p[None, :], mask=mask_p, other=0.0).to(tl.float32)
                kp = tl.load(kb + rot_dim + offs_p[None, :], mask=mask_p, other=0.0).to(tl.float32)
                q_sq += tl.sum(qp * qp, axis=1)
                k_sq += tl.sum(kp * kp, axis=1)

            qr = tl.rsqrt(q_sq * inv_hd + eps)[:, None]
            kr = tl.rsqrt(k_sq * inv_hd + eps)[:, None]

            q1_n = q1 * qr
            q2_n = q2 * qr
            k1_n = k1 * kr
            k2_n = k2 * kr
            if HAS_WEIGHT:
                q1_n = q1_n * qw1
                q2_n = q2_n * qw2
                k1_n = k1_n * kw1
                k2_n = k2_n * kw2
            q1_n = q1_n.to(out_dtype).to(tl.float32)
            q2_n = q2_n.to(out_dtype).to(tl.float32)
            k1_n = k1_n.to(out_dtype).to(tl.float32)
            k2_n = k2_n.to(out_dtype).to(tl.float32)

            tl.store(qob + offs_h[None, :], (q1_n * cos_h - q2_n * sin_h).to(out_dtype), mask=mask_h)
            tl.store(qob + ROT_HALF + offs_h[None, :], (q2_n * cos_h + q1_n * sin_h).to(out_dtype), mask=mask_h)
            tl.store(kob + offs_h[None, :], (k1_n * cos_h - k2_n * sin_h).to(out_dtype), mask=mask_h)
            tl.store(kob + ROT_HALF + offs_h[None, :], (k2_n * cos_h + k1_n * sin_h).to(out_dtype), mask=mask_h)

            if PASS_DIM > 0:
                qp_n = qp * qr
                kp_n = kp * kr
                if HAS_WEIGHT:
                    qp_n = qp_n * qwp
                    kp_n = kp_n * kwp
                tl.store(qob + rot_dim + offs_p[None, :], qp_n.to(out_dtype), mask=mask_p)
                tl.store(kob + rot_dim + offs_p[None, :], kp_n.to(out_dtype), mask=mask_p)

    @triton.jit
    def _vit_scaled_residual_add_rmsnorm_kernel(
        res_ptr,
        upd_ptr,
        scale_ptr,
        w_ptr,
        res_out_ptr,
        norm_out_ptr,
        res_s0,
        upd_s0,
        ro_s0,
        no_s0,
        eps,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        """Single-pass ``res_out = res + upd * scale`` and FP32 ``RMSNorm(res_out)``."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < N_COLS

        r_in = tl.load(res_ptr + row * res_s0 + cols, mask=mask, other=0.0)
        u_in = tl.load(upd_ptr + row * upd_s0 + cols, mask=mask, other=0.0)
        res_dtype = res_out_ptr.dtype.element_ty
        norm_dtype = norm_out_ptr.dtype.element_ty

        if HAS_SCALE:
            sc = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            # Match eager ``residual + x * scale`` (scale is typically FP32 parameter -> FP32 mul/add).
            h_f = r_in.to(tl.float32) + u_in.to(tl.float32) * sc
        else:
            h_f = r_in.to(tl.float32) + u_in.to(tl.float32)

        h_out = h_f.to(res_dtype)
        tl.store(res_out_ptr + row * ro_s0 + cols, h_out, mask=mask)

        # Match ``self.norm(h_out.float()).to(norm_dtype)``: FP32 variance & weight mul on rounded h_out.
        hf = tl.where(mask, h_out.to(tl.float32), 0.0)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) / N_COLS + eps)
        y = hf * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        tl.store(norm_out_ptr + row * no_s0 + cols, y.to(norm_dtype), mask=mask)

    @triton.jit
    def _vit_rmsnorm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        x_s0,
        o_s0,
        eps,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        """Single-pass FP32-accumulated RMSNorm without ``x.float()`` copy."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < N_COLS

        x_in = tl.load(x_ptr + row * x_s0 + cols, mask=mask, other=0.0)
        xf = tl.where(mask, x_in.to(tl.float32), 0.0)
        rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / N_COLS + eps)
        y = xf * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        tl.store(out_ptr + row * o_s0 + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

    return {
        "triton": triton,
        "qk_rmsnorm_rope": _vit_qk_rmsnorm_rope_kernel,
        "scaled_res_rmsnorm": _vit_scaled_residual_add_rmsnorm_kernel,
        "rmsnorm": _vit_rmsnorm_kernel,
    }


@torch.library.custom_op("worldfoundry::h3_vit_qk_rmsnorm_rope", mutates_args=(), device_types="cuda")
def _h3_vit_qk_rmsnorm_rope_op(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    b, s, h, d = q.shape
    rot_dim = int(cos.shape[-1])
    rot_half = rot_dim // 2
    pass_dim = d - rot_dim
    q_out = torch.empty((b, s, h, d), device=q.device, dtype=q.dtype)
    k_out = torch.empty((b, s, h, d), device=k.device, dtype=k.dtype)
    if b == 0 or s == 0 or h == 0:
        return q_out, k_out

    has_weight = q_weight is not None and k_weight is not None
    qw = q_weight.contiguous() if q_weight is not None else q
    kw = k_weight.contiguous() if k_weight is not None else k
    block_half = triton.next_power_of_2(rot_half)
    block_pass = triton.next_power_of_2(max(pass_dim, 1))
    max_bh = 8 if d <= 64 else 4
    block_h = min(triton.next_power_of_2(h), max_bh)

    with torch.cuda.device(q.device):
        kernels["qk_rmsnorm_rope"][(b * s,)](
            q,
            k,
            cos,
            sin,
            qw,
            kw,
            q_out,
            k_out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            s,
            float(eps),
            NUM_HEADS=h,
            HEAD_DIM=d,
            ROT_HALF=rot_half,
            PASS_DIM=pass_dim,
            BLOCK_HALF=block_half,
            BLOCK_PASS=block_pass,
            BLOCK_H=block_h,
            HAS_WEIGHT=has_weight,
            COS_BATCH_BROADCAST=cos.shape[0] == 1 and b > 1,
            num_warps=4,
        )
    return q_out, k_out


@_h3_vit_qk_rmsnorm_rope_op.register_fake
def _h3_vit_qk_rmsnorm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cos, sin, q_weight, k_weight, eps
    return q.new_empty(q.shape), k.new_empty(k.shape)


def can_use_vit_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return True when the fused ViT3D Q/K RMSNorm+RoPE Triton op is eligible."""
    if torch.is_grad_enabled() or not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if q.device != k.device or q.device != cos.device or q.device != sin.device:
        return False
    if q.ndim != 4 or k.shape != q.shape or cos.ndim != 4 or sin.shape != cos.shape:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype:
        return False
    if cos.shape[0] not in (1, q.shape[0]) or cos.shape[1] != q.shape[1] or cos.shape[2] != 1:
        return False
    d = q.shape[3]
    rot_dim = cos.shape[3]
    return (
        0 < d <= 256
        and 0 < rot_dim <= d
        and rot_dim % 2 == 0
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )


def vit_fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused per-head Q/K RMSNorm + partial NeoX RoPE on ``[B, S, H, D]``."""
    return _h3_vit_qk_rmsnorm_rope_op(q, k, cos, sin, q_weight, k_weight, float(eps))


@torch.library.custom_op("worldfoundry::h3_vit_scaled_residual_rmsnorm", mutates_args=(), device_types="cuda")
def _h3_vit_scaled_residual_rmsnorm_op(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    eps: float,
    norm_out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    orig_shape = residual.shape
    cols = orig_shape[-1]
    rows = residual.numel() // cols
    r2 = residual.reshape(rows, cols)
    u2 = update.reshape(rows, cols)
    if r2.stride(-1) != 1:
        r2 = r2.contiguous()
    if u2.stride(-1) != 1:
        u2 = u2.contiguous()

    res_out = torch.empty((rows, cols), device=residual.device, dtype=residual.dtype)
    norm_out = torch.empty((rows, cols), device=residual.device, dtype=norm_out_dtype)
    if rows == 0 or cols == 0:
        return res_out.reshape(orig_shape), norm_out.reshape(orig_shape)

    has_scale = scale is not None
    has_weight = norm_weight is not None
    sc = scale.contiguous() if scale is not None else r2
    nw = norm_weight.contiguous() if norm_weight is not None else r2
    block_cols = triton.next_power_of_2(cols)
    num_warps = 4 if block_cols <= 2048 else 8

    with torch.cuda.device(residual.device):
        kernels["scaled_res_rmsnorm"][(rows,)](
            r2,
            u2,
            sc,
            nw,
            res_out,
            norm_out,
            r2.stride(0),
            u2.stride(0),
            res_out.stride(0),
            norm_out.stride(0),
            float(eps),
            N_COLS=cols,
            BLOCK_COLS=block_cols,
            HAS_SCALE=has_scale,
            HAS_WEIGHT=has_weight,
            num_warps=num_warps,
        )
    return res_out.reshape(orig_shape), norm_out.reshape(orig_shape)


@_h3_vit_scaled_residual_rmsnorm_op.register_fake
def _h3_vit_scaled_residual_rmsnorm_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    eps: float,
    norm_out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del update, scale, norm_weight, eps
    return torch.empty_like(residual), torch.empty(residual.shape, device=residual.device, dtype=norm_out_dtype)


def can_use_vit_scaled_residual_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
) -> bool:
    """Return True when single-pass scaled residual + RMSNorm is eligible."""
    if torch.is_grad_enabled() or not (residual.is_cuda and update.is_cuda):
        return False
    if residual.device != update.device or residual.shape != update.shape:
        return False
    if residual.dtype not in _SUPPORTED_DTYPES or update.dtype not in _SUPPORTED_DTYPES:
        return False
    cols = residual.shape[-1]
    if cols <= 0 or cols > 4096 or residual.stride(-1) != 1 or update.stride(-1) != 1:
        return False
    if scale is not None and (not scale.is_cuda or scale.device != residual.device or scale.shape != (cols,)):
        return False
    return not (
        norm_weight is not None
        and (not norm_weight.is_cuda or norm_weight.device != residual.device or norm_weight.shape != (cols,))
    )


def vit_scaled_residual_add_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    norm_out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(residual + update * scale, rms_norm(out))`` in one HBM pass."""
    target_norm_dtype = residual.dtype if norm_out_dtype is None else norm_out_dtype
    if can_use_vit_scaled_residual_rmsnorm(residual, update, scale, norm_weight):
        return _h3_vit_scaled_residual_rmsnorm_op(residual, update, scale, norm_weight, float(eps), target_norm_dtype)
    out = residual + update * scale if scale is not None else residual + update
    out = out.to(residual.dtype)
    norm_w = norm_weight.float() if norm_weight is not None else None
    normed = F.rms_norm(out.float(), (out.shape[-1],), weight=norm_w, eps=eps).to(target_norm_dtype)
    return out, normed


@torch.library.custom_op("worldfoundry::h3_vit_rmsnorm", mutates_args=(), device_types="cuda")
def _h3_vit_rmsnorm_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.numel() // cols
    x2 = x.reshape(rows, cols)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty((rows, cols), device=x.device, dtype=out_dtype)
    if rows == 0 or cols == 0:
        return out.reshape(orig_shape)

    has_weight = norm_weight is not None
    nw = norm_weight.contiguous() if norm_weight is not None else x2
    block_cols = triton.next_power_of_2(cols)
    num_warps = 4 if block_cols <= 2048 else 8

    with torch.cuda.device(x.device):
        kernels["rmsnorm"][(rows,)](
            x2,
            nw,
            out,
            x2.stride(0),
            out.stride(0),
            float(eps),
            N_COLS=cols,
            BLOCK_COLS=block_cols,
            HAS_WEIGHT=has_weight,
            num_warps=num_warps,
        )
    return out.reshape(orig_shape)


@_h3_vit_rmsnorm_op.register_fake
def _h3_vit_rmsnorm_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del norm_weight, eps
    return torch.empty(x.shape, device=x.device, dtype=out_dtype)


def vit_rmsnorm(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Single-pass FP32-accumulated RMSNorm without ``x.float()`` copy."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if (
        not torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and target_dtype in _SUPPORTED_DTYPES
        and 0 < x.shape[-1] <= 4096
        and x.stride(-1) == 1
        and (norm_weight is None or (norm_weight.is_cuda and norm_weight.device == x.device))
    ):
        return _h3_vit_rmsnorm_op(x, norm_weight, float(eps), target_dtype)
    norm_w = norm_weight.float() if norm_weight is not None else None
    return F.rms_norm(x.float(), (x.shape[-1],), weight=norm_w, eps=eps).to(target_dtype)


__all__ = [
    "can_use_vit_qk_rmsnorm_rope",
    "can_use_vit_scaled_residual_rmsnorm",
    "vit_fused_qk_rmsnorm_rope",
    "vit_rmsnorm",
    "vit_scaled_residual_add_rmsnorm",
]
