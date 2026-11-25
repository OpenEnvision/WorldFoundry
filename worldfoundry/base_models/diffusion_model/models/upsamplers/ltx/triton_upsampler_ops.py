# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper SM90 Triton operators for the LTX Spatial/Temporal LatentUpsampler.

Fuses ``(optional pre_bias) + GroupNorm + (optional post-norm residual) +
(optional SiLU)`` into a deterministic 2-stage Chan-variance Hopper Triton
operator supporting both ``NCDHW`` / ``NCHW`` and ``channels_last_3d`` /
``channels_last`` layouts with strict ``tl.int64`` pointer arithmetic.

All CUDA ops are registered via ``torch.library.custom_op`` + ``register_fake``
for ``torch.compile`` and CUDA Graph capture safety. Triton is imported lazily
on first CUDA invocation; CPU, autograd, and unsupported dtypes/shapes fall
back transparently to exact PyTorch math.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TRITON_KERNELS: dict[str, Any] | None = None
_TRITON_LOAD_FAILED: bool = False


def _is_channels_last_nd(x: torch.Tensor) -> bool:
    if x.ndim == 5:
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            if not x.is_contiguous():
                return True
            return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
        return False
    if x.ndim == 4:
        if x.is_contiguous(memory_format=torch.channels_last):
            if not x.is_contiguous():
                return True
            return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(3) == x.shape[1]
        return False
    return False


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_triton_kernels() -> dict[str, Any] | None:
    global _TRITON_KERNELS, _TRITON_LOAD_FAILED
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    if _TRITON_LOAD_FAILED:
        return None
    try:
        from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="ltx-upsampler-triton-v1")
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_LOAD_FAILED = True
        return None

    # =========================================================================
    # 1. Stage-1 Chunked Chan Moments (CL & CF, strict int64)
    # =========================================================================

    @triton.jit
    def _gn_reduce_cl_kernel(
        x_ptr,
        pre_bias_ptr,
        partial_ptr,
        S,
        C,
        G,
        CG,
        chunk_s,
        num_chunks,
        HAS_PRE_BIAS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        bg = tl.program_id(0).to(tl.int64)
        k = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        g_i64 = tl.cast(G, tl.int64)
        cg_i64 = tl.cast(CG, tl.int64)
        cs_i64 = tl.cast(chunk_s, tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)

        b_idx = bg // g_i64
        g_idx = bg - b_idx * g_i64
        c0 = g_idx * cg_i64

        s_start = k * cs_i64
        s_end = tl.minimum(s_start + cs_i64, s_i64)
        cnt_s = tl.maximum(s_end - s_start, 0)

        out_base = (bg * nc_i64 + k) * 3
        if cnt_s == 0:
            tl.store(partial_ptr + out_base + 0, 0.0)
            tl.store(partial_ptr + out_base + 1, 0.0)
            tl.store(partial_ptr + out_base + 2, 0.0)
            return

        c_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        c_mask = c_offs < cg_i64
        pb = tl.zeros((1, BLOCK_CG), dtype=tl.float32)
        if HAS_PRE_BIAS:
            pb = tl.load(pre_bias_ptr + c0 + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]

        x_b = b_idx * s_i64 * c_i64 + c0
        # Shift value from first voxel in chunk for numerical stability.
        shift_raw = tl.load(x_ptr + x_b + s_start * c_i64).to(tl.float32)
        if HAS_PRE_BIAS:
            pb0 = tl.load(pre_bias_ptr + c0).to(tl.float32)
            shift = shift_raw + pb0
        else:
            shift = shift_raw

        sum_d = 0.0
        sum_d2 = 0.0
        for s_off in range(0, chunk_s, BLOCK_S):
            s_vec = s_start + tl.cast(s_off, tl.int64) + tl.arange(0, BLOCK_S).to(tl.int64)
            s_mask = s_vec < s_end
            m = s_mask[:, None] & c_mask[None, :]
            xv = tl.load(x_ptr + x_b + s_vec[:, None] * c_i64 + c_offs[None, :], mask=m, other=0.0)
            xf = xv.to(tl.float32)
            if HAS_PRE_BIAS:
                xf = xf + pb
            diff = tl.where(m, xf - shift, 0.0)
            sum_d += tl.sum(diff)
            sum_d2 += tl.sum(diff * diff)

        cnt = (cnt_s * cg_i64).to(tl.float32)
        mean_d = sum_d / cnt
        m2 = tl.maximum(sum_d2 - sum_d * mean_d, 0.0)
        tl.store(partial_ptr + out_base + 0, cnt)
        tl.store(partial_ptr + out_base + 1, shift + mean_d)
        tl.store(partial_ptr + out_base + 2, m2)

    @triton.jit
    def _gn_apply_cl_kernel(
        x_ptr,
        pre_bias_ptr,
        w_ptr,
        b_ptr,
        res_ptr,
        partial_ptr,
        out_ptr,
        S,
        C,
        G,
        CG,
        eps,
        NUM_CHUNKS: tl.constexpr,
        HAS_PRE_BIAS: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_CG: tl.constexpr,
    ):
        bg = tl.program_id(0).to(tl.int64)
        s_tile = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        g_i64 = tl.cast(G, tl.int64)
        cg_i64 = tl.cast(CG, tl.int64)

        b_idx = bg // g_i64
        g_idx = bg - b_idx * g_i64
        c0 = g_idx * cg_i64

        # Merge chunk moments in registers via Chan's parallel variance formula.
        p_base = bg * (NUM_CHUNKS * 3)
        tot_cnt = 0.0
        tot_mean = 0.0
        tot_m2 = 0.0
        for k in tl.static_range(NUM_CHUNKS):
            cnt_k = tl.load(partial_ptr + p_base + k * 3 + 0)
            mean_k = tl.load(partial_ptr + p_base + k * 3 + 1)
            m2_k = tl.load(partial_ptr + p_base + k * 3 + 2)
            new_cnt = tot_cnt + cnt_k
            delta = mean_k - tot_mean
            has_both = (tot_cnt > 0.0) & (cnt_k > 0.0)
            tot_mean = tl.where(new_cnt > 0.0, tot_mean + delta * (cnt_k / tl.maximum(new_cnt, 1.0)), tot_mean)
            tot_m2 = (
                tot_m2 + m2_k + tl.where(has_both, delta * delta * (tot_cnt * cnt_k / tl.maximum(new_cnt, 1.0)), 0.0)
            )
            tot_cnt = new_cnt

        var = tot_m2 / tl.maximum(tot_cnt, 1.0)
        rstd = tl.rsqrt(var + eps)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64
        c_offs = tl.arange(0, BLOCK_CG).to(tl.int64)
        c_mask = c_offs < cg_i64
        m = s_mask[:, None] & c_mask[None, :]

        base = b_idx * s_i64 * c_i64 + c0
        ptrs = base + s_offs[:, None] * c_i64 + c_offs[None, :]
        out_ty = out_ptr.dtype.element_ty

        xv = tl.load(x_ptr + ptrs, mask=m, other=0.0, eviction_policy="evict_first")
        xf = xv.to(tl.float32)
        if HAS_PRE_BIAS:
            pb = tl.load(pre_bias_ptr + c0 + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            xf = xf + pb

        y = (xf - tot_mean) * rstd
        if HAS_AFFINE:
            wv = tl.load(w_ptr + c0 + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
            bv = tl.load(b_ptr + c0 + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y * wv + bv

        if HAS_RESIDUAL:
            rv = tl.load(res_ptr + ptrs, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
            y = y + rv

        if APPLY_SILU:
            y = y * tl.sigmoid(y)

        tl.store(out_ptr + ptrs, y.to(out_ty), mask=m, eviction_policy="evict_first")

    @triton.jit
    def _gn_reduce_cf_kernel(
        x_ptr,
        pre_bias_ptr,
        partial_ptr,
        S,
        C,
        G,
        CG,
        chunk_s,
        num_chunks,
        HAS_PRE_BIAS: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        bg = tl.program_id(0).to(tl.int64)
        k = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        g_i64 = tl.cast(G, tl.int64)
        cg_i64 = tl.cast(CG, tl.int64)
        cs_i64 = tl.cast(chunk_s, tl.int64)
        nc_i64 = tl.cast(num_chunks, tl.int64)

        b_idx = bg // g_i64
        g_idx = bg - b_idx * g_i64
        c0 = g_idx * cg_i64

        s_start = k * cs_i64
        s_end = tl.minimum(s_start + cs_i64, s_i64)
        cnt_s = tl.maximum(s_end - s_start, 0)

        out_base = (bg * nc_i64 + k) * 3
        if cnt_s == 0:
            tl.store(partial_ptr + out_base + 0, 0.0)
            tl.store(partial_ptr + out_base + 1, 0.0)
            tl.store(partial_ptr + out_base + 2, 0.0)
            return

        b_base = b_idx * c_i64 * s_i64
        shift_raw = tl.load(x_ptr + b_base + c0 * s_i64 + s_start).to(tl.float32)
        if HAS_PRE_BIAS:
            pb0 = tl.load(pre_bias_ptr + c0).to(tl.float32)
            shift = shift_raw + pb0
        else:
            shift = shift_raw

        sum_d = 0.0
        sum_d2 = 0.0
        for ci in range(CG):
            c_idx = c0 + tl.cast(ci, tl.int64)
            c_base = b_base + c_idx * s_i64
            pb = tl.load(pre_bias_ptr + c_idx).to(tl.float32) if HAS_PRE_BIAS else 0.0
            for s_off in range(0, chunk_s, BLOCK_S):
                s_vec = s_start + tl.cast(s_off, tl.int64) + tl.arange(0, BLOCK_S).to(tl.int64)
                s_mask = s_vec < s_end
                xv = tl.load(x_ptr + c_base + s_vec, mask=s_mask, other=0.0)
                xf = xv.to(tl.float32)
                if HAS_PRE_BIAS:
                    xf = xf + pb
                diff = tl.where(s_mask, xf - shift, 0.0)
                sum_d += tl.sum(diff)
                sum_d2 += tl.sum(diff * diff)

        cnt = (cnt_s * cg_i64).to(tl.float32)
        mean_d = sum_d / cnt
        m2 = tl.maximum(sum_d2 - sum_d * mean_d, 0.0)
        tl.store(partial_ptr + out_base + 0, cnt)
        tl.store(partial_ptr + out_base + 1, shift + mean_d)
        tl.store(partial_ptr + out_base + 2, m2)

    @triton.jit
    def _gn_apply_cf_kernel(
        x_ptr,
        pre_bias_ptr,
        w_ptr,
        b_ptr,
        res_ptr,
        partial_ptr,
        out_ptr,
        S,
        C,
        G,
        CG,
        eps,
        NUM_CHUNKS: tl.constexpr,
        HAS_PRE_BIAS: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        s_tile = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        c_i64 = tl.cast(C, tl.int64)
        g_i64 = tl.cast(G, tl.int64)
        cg_i64 = tl.cast(CG, tl.int64)

        b_idx = bc // c_i64
        c_idx = bc - b_idx * c_i64
        g_idx = c_idx // cg_i64
        bg = b_idx * g_i64 + g_idx

        p_base = bg * (NUM_CHUNKS * 3)
        tot_cnt = 0.0
        tot_mean = 0.0
        tot_m2 = 0.0
        for k in tl.static_range(NUM_CHUNKS):
            cnt_k = tl.load(partial_ptr + p_base + k * 3 + 0)
            mean_k = tl.load(partial_ptr + p_base + k * 3 + 1)
            m2_k = tl.load(partial_ptr + p_base + k * 3 + 2)
            new_cnt = tot_cnt + cnt_k
            delta = mean_k - tot_mean
            has_both = (tot_cnt > 0.0) & (cnt_k > 0.0)
            tot_mean = tl.where(new_cnt > 0.0, tot_mean + delta * (cnt_k / tl.maximum(new_cnt, 1.0)), tot_mean)
            tot_m2 = (
                tot_m2 + m2_k + tl.where(has_both, delta * delta * (tot_cnt * cnt_k / tl.maximum(new_cnt, 1.0)), 0.0)
            )
            tot_cnt = new_cnt

        var = tot_m2 / tl.maximum(tot_cnt, 1.0)
        rstd = tl.rsqrt(var + eps)

        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_offs < s_i64
        ptrs = bc * s_i64 + s_offs
        out_ty = out_ptr.dtype.element_ty

        xv = tl.load(x_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
        xf = xv.to(tl.float32)
        if HAS_PRE_BIAS:
            pb = tl.load(pre_bias_ptr + c_idx).to(tl.float32)
            xf = xf + pb

        y = (xf - tot_mean) * rstd
        if HAS_AFFINE:
            wv = tl.load(w_ptr + c_idx).to(tl.float32)
            bv = tl.load(b_ptr + c_idx).to(tl.float32)
            y = y * wv + bv

        if HAS_RESIDUAL:
            rv = tl.load(res_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            y = y + rv

        if APPLY_SILU:
            y = y * tl.sigmoid(y)

        tl.store(out_ptr + ptrs, y.to(out_ty), mask=s_mask, eviction_policy="evict_first")

    _TRITON_KERNELS = {
        "gn_reduce_cl": _gn_reduce_cl_kernel,
        "gn_apply_cl": _gn_apply_cl_kernel,
        "gn_reduce_cf": _gn_reduce_cf_kernel,
        "gn_apply_cf": _gn_apply_cf_kernel,
    }
    return _TRITON_KERNELS


def _upsampler_gn_silu_torch(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    pre_bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    apply_silu: bool,
) -> torch.Tensor:
    if pre_bias is not None:
        view_shape = (1, -1) + (1,) * (x.ndim - 2)
        x = x + pre_bias.view(view_shape).to(x)
    y = F.group_norm(
        x,
        num_groups,
        weight.to(x) if weight is not None else None,
        bias.to(x) if bias is not None else None,
        eps,
    )
    if residual is not None:
        y = y + residual
    if apply_silu:
        y = F.silu(y)
    return y


@torch.library.custom_op("worldfoundry::ltx_upsampler_group_norm_silu", mutates_args=(), device_types="cuda")
def _ltx_upsampler_gn_silu_op(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    pre_bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _get_triton_kernels()
    if kernels is None:
        return _upsampler_gn_silu_torch(x, num_groups, weight, bias, eps, pre_bias, residual, apply_silu)

    b, c = x.shape[0], x.shape[1]
    s = 1
    for dim in x.shape[2:]:
        s *= dim
    cg = c // num_groups
    has_pre_bias = pre_bias is not None
    has_affine = weight is not None and bias is not None
    has_residual = residual is not None

    pb_1d = pre_bias.reshape(c).contiguous() if pre_bias is not None else x
    w_1d = weight.reshape(c).contiguous() if weight is not None else x
    b_1d = bias.reshape(c).contiguous() if bias is not None else x

    use_cl = _is_channels_last_nd(x) or (residual is not None and _is_channels_last_nd(residual))
    target_chunks = min(16, max(1, _cdiv(256, max(1, b * num_groups))))
    num_chunks = min(target_chunks, max(1, _cdiv(s, 256)))
    chunk_s = _cdiv(s, num_chunks)

    with torch.cuda.device(x.device):
        partial = torch.empty((b * num_groups, num_chunks, 3), device=x.device, dtype=torch.float32)
        if use_cl:
            mem_fmt = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
            x_cl = x if _is_channels_last_nd(x) else x.contiguous(memory_format=mem_fmt)
            res_cl = (
                (residual if _is_channels_last_nd(residual) else residual.contiguous(memory_format=mem_fmt))
                if residual is not None
                else x_cl
            )
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=mem_fmt)
            if out.numel() == 0:
                return out
            block_cg = _next_power_of_2(cg)
            block_s = max(16, min(128, 4096 // max(1, block_cg)))
            num_warps = 8 if block_s * block_cg >= 4096 else 4
            kernels["gn_reduce_cl"][(b * num_groups, num_chunks)](
                x_cl,
                pb_1d,
                partial,
                s,
                c,
                num_groups,
                cg,
                chunk_s,
                num_chunks,
                HAS_PRE_BIAS=has_pre_bias,
                BLOCK_S=block_s,
                BLOCK_CG=block_cg,
                num_warps=num_warps,
            )
            kernels["gn_apply_cl"][(b * num_groups, _cdiv(s, block_s))](
                x_cl,
                pb_1d,
                w_1d,
                b_1d,
                res_cl,
                partial,
                out,
                s,
                c,
                num_groups,
                cg,
                float(eps),
                NUM_CHUNKS=num_chunks,
                HAS_PRE_BIAS=has_pre_bias,
                HAS_AFFINE=has_affine,
                HAS_RESIDUAL=has_residual,
                APPLY_SILU=bool(apply_silu),
                BLOCK_S=block_s,
                BLOCK_CG=block_cg,
                num_warps=num_warps,
            )
            return out

        x_c = x if x.is_contiguous() else x.contiguous()
        res_c = (residual if residual.is_contiguous() else residual.contiguous()) if residual is not None else x_c
        out = torch.empty_like(x_c)
        if out.numel() == 0:
            return out
        block_s = 1024 if chunk_s >= 1024 else _next_power_of_2(max(chunk_s, 64))
        num_warps = 8 if block_s >= 1024 else 4
        kernels["gn_reduce_cf"][(b * num_groups, num_chunks)](
            x_c,
            pb_1d,
            partial,
            s,
            c,
            num_groups,
            cg,
            chunk_s,
            num_chunks,
            HAS_PRE_BIAS=has_pre_bias,
            BLOCK_S=block_s,
            num_warps=num_warps,
        )
        apply_bs = 1024 if s >= 1024 else _next_power_of_2(max(s, 64))
        kernels["gn_apply_cf"][(b * c, _cdiv(s, apply_bs))](
            x_c,
            pb_1d,
            w_1d,
            b_1d,
            res_c,
            partial,
            out,
            s,
            c,
            num_groups,
            cg,
            float(eps),
            NUM_CHUNKS=num_chunks,
            HAS_PRE_BIAS=has_pre_bias,
            HAS_AFFINE=has_affine,
            HAS_RESIDUAL=has_residual,
            APPLY_SILU=bool(apply_silu),
            BLOCK_S=apply_bs,
            num_warps=8 if apply_bs >= 1024 else 4,
        )
        return out


@_ltx_upsampler_gn_silu_op.register_fake
def _ltx_upsampler_gn_silu_fake(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    pre_bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    apply_silu: bool,
) -> torch.Tensor:
    del num_groups, weight, bias, eps, pre_bias, apply_silu
    if _is_channels_last_nd(x) or (residual is not None and _is_channels_last_nd(residual)):
        mem_fmt = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
        return torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype, memory_format=mem_fmt)
    return x.new_empty(tuple(x.shape))


def can_use_triton_upsampler_group_norm_silu(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    pre_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> bool:
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES or torch.is_grad_enabled():
        return False
    if x.ndim not in (4, 5):
        return False
    if not torch.compiler.is_compiling() and x.numel() == 0:
        return False
    c = x.shape[1]
    if num_groups <= 0 or c <= 0 or c % num_groups != 0:
        return False
    if (c // num_groups) > 512:
        return False
    if (weight is None) != (bias is None):
        return False
    for t in (weight, bias, pre_bias, residual):
        if t is not None and (not t.is_cuda or t.device != x.device or t.dtype not in _SUPPORTED_DTYPES):
            return False
    if weight is not None and weight.numel() != c:
        return False
    if bias is not None and bias.numel() != c:
        return False
    if pre_bias is not None and pre_bias.numel() != c:
        return False
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype):
        return False
    return _get_triton_kernels() is not None


def triton_upsampler_group_norm_silu(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    pre_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Fused ``(pre_bias +) GroupNorm (+ residual) (+ SiLU)`` for 4-D / 5-D tensors."""
    if can_use_triton_upsampler_group_norm_silu(x, int(num_groups), weight, bias, pre_bias, residual):
        return _ltx_upsampler_gn_silu_op(
            x,
            int(num_groups),
            weight,
            bias,
            float(eps),
            pre_bias,
            residual,
            bool(apply_silu),
        )
    return _upsampler_gn_silu_torch(x, int(num_groups), weight, bias, float(eps), pre_bias, residual, bool(apply_silu))


__all__ = [
    "can_use_triton_upsampler_group_norm_silu",
    "triton_upsampler_group_norm_silu",
]
