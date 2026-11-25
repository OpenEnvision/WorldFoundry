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

    return {
        "triton": triton,
        "qk_rstd": _sana_qk_rstd_kernel,
        "kv_splitk": _sana_litela_kv_splitk_kernel,
        "q_apply_div": _sana_litela_q_apply_div_kernel,
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
    num_splits = _select_num_splits(bh, n)
    block_n = 128 if block_d <= 64 else 64
    num_warps_dot = 4 if block_d <= 64 else 8
    use_fp64_rope = cos_3d.dtype == torch.float64 or sin_3d.dtype == torch.float64
    cos_batch_broadcast = cos_3d.shape[0] == 1 and b > 1

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


__all__ = ["can_use_triton_litela_relu_rope", "fused_sana_litela_relu_rope"]
