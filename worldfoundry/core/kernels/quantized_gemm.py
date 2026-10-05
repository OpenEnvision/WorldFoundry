# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The LightX2V Authors. Portions derived from:
# https://github.com/ModelTC/LightX2V/blob/main/lightx2v/common/ops/mm/triton_kernels.py
#
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt at repo root).
#
# Modifications Copyright (c) 2026 SkyworkAI and contributors.
"""INT8 / quantized GEMM dispatch for Wan-style layers.

Public quantized matmul entry with a PyTorch fallback. Triton
implementations stay private to this file.
"""

import torch
from triton import Config, autotune, cdiv, jit, next_power_of_2
from triton import language as tl

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


# ──────────────────────────────────────────────────────────────────────────
# Row-wise quantize — BLOCK_SIZE = next_pow2(H); one program per flattened row
# ──────────────────────────────────────────────────────────────────────────


@jit
def gelu(x):
    """Gelu.

    Args:
        x: The x.
    """
    return x * tl.sigmoid(x * 1.702)


@jit
def int8_quantize_kernel(X, OUT, SCALES, HDIM, BLOCK_SIZE: tl.constexpr):
    """Int8 quantize kernel.

    Args:
        X: The x.
        OUT: The out.
        SCALES: The scales.
        HDIM: The hdim.
        BLOCK_SIZE: The block size.
    """
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * HDIM
    out_ptr = OUT + row_idx * HDIM
    h_offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + h_offset, mask=h_offset < HDIM).to(tl.float32)
    x_scale = 127.0 / tl.max(tl.abs(x))
    x_scaled = x * x_scale
    # Symmetric INT8: away-from-zero 0.5 rounding, then store 1/scale for dequant.
    x_scaled += (0.5 * tl.where(x_scaled >= 0, 1, -1)).to(tl.int8)
    tl.store(out_ptr + h_offset, x_scaled, mask=h_offset < HDIM)
    tl.store(SCALES + row_idx, 1 / x_scale)


def int8_quantize_triton(x):
    """Int8 quantize triton.

    Args:
        x: The x.
    """
    x_shape_orig = x.shape
    x = x.view(-1, x_shape_orig[-1])
    out = torch.empty(x_shape_orig, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    BLOCK_SIZE = next_power_of_2(x_shape_orig[-1])
    grid = (x.shape[0],)
    int8_quantize_kernel[grid](x, out, scales, x_shape_orig[-1], BLOCK_SIZE, num_warps=8)
    return out.view(x_shape_orig), scales.view(x_shape_orig[:-1])


@jit
def fp8_quantize_kernel(X, OUT, SCALES, HDIM, BLOCK_SIZE: tl.constexpr, FP8_MAX_VAL: tl.constexpr):
    """Fp8 quantize kernel.

    Args:
        X: The x.
        OUT: The out.
        SCALES: The scales.
        HDIM: The hdim.
        BLOCK_SIZE: The block size.
        FP8_MAX_VAL: The fp8 max val.
    """
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * HDIM
    out_ptr = OUT + row_idx * HDIM
    h_offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + h_offset, mask=h_offset < HDIM).to(tl.float32)
    absmax = tl.max(tl.abs(x))
    eps = 1e-8
    absmax = tl.maximum(absmax, eps)
    # e4m3 max is 448; store the scale (not its reciprocal) unlike INT8.
    x_scale = absmax / FP8_MAX_VAL
    x_scaled = x / x_scale
    x_scaled = tl.clamp(x_scaled, -FP8_MAX_VAL, FP8_MAX_VAL)
    tl.store(out_ptr + h_offset, x_scaled, mask=h_offset < HDIM)
    tl.store(SCALES + row_idx, x_scale)


def fp8_quantize_triton(x):
    """Fp8 quantize triton.

    Args:
        x: The x.
    """
    x_shape_orig = x.shape
    x = x.view(-1, x_shape_orig[-1])
    out_scaled = torch.empty(x_shape_orig, dtype=torch.float32, device=x.device)
    scales = torch.empty(x.shape[0], dtype=torch.bfloat16, device=x.device)
    BLOCK_SIZE = next_power_of_2(x_shape_orig[-1])
    grid = (x.shape[0],)
    FP8_MAX = 448.0
    fp8_quantize_kernel[grid](x, out_scaled, scales, x_shape_orig[-1], BLOCK_SIZE, FP8_MAX_VAL=FP8_MAX, num_warps=8)
    quantized = out_scaled.to(torch.float8_e4m3fn)
    return quantized.view(x_shape_orig), scales.view(x_shape_orig[:-1])


# ──────────────────────────────────────────────────────────────────────────
# Accumulator dtype — FP8 promotes to fp16 before the GEMM type lattice
# ──────────────────────────────────────────────────────────────────────────


def upcast_if_fp8(a):
    """Upcast if fp8.

    Args:
        a: The a.
    """
    if "fp8" in str(a):
        return torch.float16
    return a


def get_higher_dtype(a, b):
    """Get higher dtype.

    Args:
        a: The a.
        b: The b.
    """
    a = upcast_if_fp8(a)
    b = upcast_if_fp8(b)
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


# ──────────────────────────────────────────────────────────────────────────
# INT8 GEMM — autotune key is (M, N, K); GROUP_M=8; SPLIT_K>1 uses atomic_add
# ──────────────────────────────────────────────────────────────────────────


@autotune(
    configs=[
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@jit
def int8_gemm_bias_kernel(
    A,
    B,
    BIAS,
    A_SCALES,
    B_SCALES,
    C,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    acc_dtype: tl.constexpr,  #
    fuse_gelu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,  #
):
    """Int8 gemm bias kernel.

    Args:
        A: The a.
        B: The b.
        BIAS: The bias.
        A_SCALES: The a scales.
        B_SCALES: The b scales.
        C: The c.
        M: The m.
        N: The n.
        K: The k.
        stride_am: The stride am.
        stride_ak: The stride ak.
        stride_bk: The stride bk.
        stride_bn: The stride bn.
        stride_cm: The stride cm.
        stride_cn: The stride cn.
        acc_dtype: The acc dtype.
        fuse_gelu: The fuse gelu.
        BLOCK_M: The block m.
        BLOCK_N: The block n.
        BLOCK_K: The block k.
        GROUP_M: The group m.
        SPLIT_K: The split k.
        EVEN_K: The even k.
        AB_DTYPE: The ab dtype.
    """
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=None)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(tl.float32)
    a_scales_ptr = A_SCALES + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_scales_ptr = B_SCALES + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scales = tl.load(a_scales_ptr)  # [BM]
    b_scales = tl.load(b_scales_ptr)  # [BN]
    # [BM, BN] * [BM, 1] * [1, BN]

    bias_ptr = BIAS + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr)
    if fuse_gelu:
        acc = gelu(((acc * a_scales[:, None]) * b_scales[None, :]) + bias[None, :])
    else:
        acc = ((acc * a_scales[:, None]) * b_scales[None, :]) + bias[None, :]

    acc = acc.to(C.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# @torch.compiler.disable()
def int8_gemm_bias_triton(a, b, bias, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    """Int8 gemm bias triton.

    Args:
        a: The a.
        b: The b.
        bias: The bias.
        a_scales: The a scales.
        b_scales: The b scales.
        fuse_gelu: The fuse gelu.
        output_dtype: The output dtype.
    """
    device = a.device
    # handle non-contiguous inputs if necessary
    a_orig_shape = a.shape
    a = a.view(-1, a.shape[-1])
    b = b.t()

    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], f"incompatible dimensions {a.shape} and {b.shape}"
    M, K = a.shape
    _, N = b.shape
    out_shape = a_orig_shape[:-1] + (N,)
    # common type between a and b
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)

    # allocates output
    if output_dtype is None:
        output_dtype = ab_dtype

    c = torch.empty((M, N), device=device, dtype=output_dtype)

    # Allowed types for acc_type given the types of a and b.
    supported_acc_dtypes = {
        torch.float16: (torch.float32, torch.float16),
        torch.bfloat16: (torch.float32, torch.bfloat16),
        torch.float32: (torch.float32,),
        torch.int8: (torch.int32,),
    }

    acc_dtype = supported_acc_dtypes[ab_dtype][0]

    def to_tl_type(ty):
        """To tl type.

        Args:
            ty: The ty.
        """
        return getattr(tl, str(ty).split(".")[-1])

    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    # Tensor cores support input with mixed float8 types.
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [
        tl.float8e4nv,
        tl.float8e5,
    ]:
        ab_dtype = None
    # launch kernel
    grid = lambda META: (  # noqa E731
        cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )  # noqa E731
    int8_gemm_bias_kernel[grid](
        a,
        b,
        bias,
        a_scales,
        b_scales,
        c,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        acc_dtype=acc_dtype,  #
        fuse_gelu=fuse_gelu,
        GROUP_M=8,
        EVEN_K=True,
        AB_DTYPE=ab_dtype,
    )
    return c.view(*out_shape)


# ──────────────────────────────────────────────────────────────────────────
# INT8 GEMM without bias — linear-layer layout only (B is transposed)
# ──────────────────────────────────────────────────────────────────────────


@autotune(
    configs=[
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@jit
def int8_gemm_kernel(
    A,
    B,
    A_SCALES,
    B_SCALES,
    C,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    acc_dtype: tl.constexpr,  #
    fuse_gelu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,  #
):
    """Int8 gemm kernel.

    Args:
        A: The a.
        B: The b.
        A_SCALES: The a scales.
        B_SCALES: The b scales.
        C: The c.
        M: The m.
        N: The n.
        K: The k.
        stride_am: The stride am.
        stride_ak: The stride ak.
        stride_bk: The stride bk.
        stride_bn: The stride bn.
        stride_cm: The stride cm.
        stride_cn: The stride cn.
        acc_dtype: The acc dtype.
        fuse_gelu: The fuse gelu.
        BLOCK_M: The block m.
        BLOCK_N: The block n.
        BLOCK_K: The block k.
        GROUP_M: The group m.
        SPLIT_K: The split k.
        EVEN_K: The even k.
        AB_DTYPE: The ab dtype.
    """
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=None)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(tl.float32)
    a_scales_ptr = A_SCALES + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_scales_ptr = B_SCALES + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scales = tl.load(a_scales_ptr)  # [BM]
    b_scales = tl.load(b_scales_ptr)  # [BN]
    # [BM, BN] * [BM, 1] * [1, BN]
    if fuse_gelu:
        acc = gelu((acc * a_scales[:, None]) * b_scales[None, :])
    else:
        acc = (acc * a_scales[:, None]) * b_scales[None, :]

    acc = acc.to(C.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# @torch.compiler.disable()
def int8_gemm_triton(a, b, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    """Int8 gemm triton.

    Args:
        a: The a.
        b: The b.
        a_scales: The a scales.
        b_scales: The b scales.
        fuse_gelu: The fuse gelu.
        output_dtype: The output dtype.
    """
    device = a.device
    # handle non-contiguous inputs if necessary
    # USE ONLY IN linear layer. NOT GENERAL MATRIX MULTIPLY
    a_orig_shape = a.shape
    a = a.view(-1, a.shape[-1])
    b = b.t()

    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], f"incompatible dimensions {a.shape} and {b.shape}"
    M, K = a.shape
    _, N = b.shape
    out_shape = a_orig_shape[:-1] + (N,)
    # common type between a and b
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)

    # allocates output
    if output_dtype is None:
        output_dtype = ab_dtype

    c = torch.empty((M, N), device=device, dtype=output_dtype)

    # Allowed types for acc_type given the types of a and b.
    supported_acc_dtypes = {
        torch.float16: (torch.float32, torch.float16),
        torch.bfloat16: (torch.float32, torch.bfloat16),
        torch.float32: (torch.float32,),
        torch.int8: (torch.int32,),
    }

    acc_dtype = supported_acc_dtypes[ab_dtype][0]

    def to_tl_type(ty):
        """To tl type.

        Args:
            ty: The ty.
        """
        return getattr(tl, str(ty).split(".")[-1])

    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    # Tensor cores support input with mixed float8 types.
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [
        tl.float8e4nv,
        tl.float8e5,
    ]:
        ab_dtype = None
    # launch kernel
    grid = lambda META: (  # noqa E731
        cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )  # noqa E731
    int8_gemm_kernel[grid](
        a,
        b,
        a_scales,
        b_scales,
        c,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        acc_dtype=acc_dtype,  #
        fuse_gelu=fuse_gelu,
        EVEN_K=True,
        GROUP_M=8,
        AB_DTYPE=ab_dtype,
    )
    return c.view(*out_shape)


# ──────────────────────────────────────────────────────────────────────────
# FP8 e4m3 GEMM — same (M, N, K) autotune key; CUDA-only launchers
# ──────────────────────────────────────────────────────────────────────────


@autotune(
    configs=[
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@jit
def fp8_gemm_bias_kernel(
    A,
    B,
    BIAS,
    A_SCALES,
    B_SCALES,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    fuse_gelu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fp8 gemm bias kernel.

    Args:
        A: The a.
        B: The b.
        BIAS: The bias.
        A_SCALES: The a scales.
        B_SCALES: The b scales.
        C: The c.
        M: The m.
        N: The n.
        K: The k.
        stride_am: The stride am.
        stride_ak: The stride ak.
        stride_bk: The stride bk.
        stride_bn: The stride bn.
        stride_cm: The stride cm.
        stride_cn: The stride cn.
        fuse_gelu: The fuse gelu.
        BLOCK_M: The block m.
        BLOCK_N: The block n.
        BLOCK_K: The block k.
        GROUP_M: The group m.
        SPLIT_K: The split k.
        EVEN_K: The even k.
    """
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A_ptr = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A_ptr, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B_ptr, mask=rk[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision=None)

        A_ptr += BLOCK_K * SPLIT_K * stride_ak
        B_ptr += BLOCK_K * SPLIT_K * stride_bk

    a_scales_ptr = A_SCALES + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_scales_ptr = B_SCALES + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scales = tl.load(a_scales_ptr).to(tl.float32)  # [BM]
    b_scales = tl.load(b_scales_ptr).to(tl.float32)  # [BN]

    bias_ptr = BIAS + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr).to(tl.float32)  # [BN]

    out = (acc * a_scales[:, None]) * b_scales[None, :] + bias[None, :]
    if fuse_gelu:
        out = gelu(out)

    out = out.to(C.dtype.element_ty)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    if SPLIT_K == 1:
        tl.store(C_ptr, out, mask=mask)
    else:
        tl.atomic_add(C_ptr, out, mask=mask)


def fp8_gemm_bias_triton(a, b, bias, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    """Fp8 gemm bias triton.

    Args:
        a: The a.
        b: The b.
        bias: The bias.
        a_scales: The a scales.
        b_scales: The b scales.
        fuse_gelu: The fuse gelu.
        output_dtype: The output dtype.
    """
    assert a.is_cuda and b.is_cuda, "This kernel is for CUDA"
    assert a.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e4m3fnuz", None)), f"a.dtype={a.dtype} is not FP8 E4M3"
    assert b.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e4m3fnuz", None)), f"b.dtype={b.dtype} is not FP8 E4M3"

    a_orig_shape = a.shape
    a2 = a.view(-1, a.shape[-1])
    b2 = b.t()

    if a2.stride(0) > 1 and a2.stride(1) > 1:
        a2 = a2.contiguous()
    if b2.stride(0) > 1 and b2.stride(1) > 1:
        b2 = b2.contiguous()

    M, K = a2.shape
    _, N = b2.shape
    out_shape = a_orig_shape[:-1] + (N,)

    if output_dtype is None:
        output_dtype = torch.float16

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    grid = lambda META: (cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]), META["SPLIT_K"])  # noqa E731
    even_k = K % 128 == 0

    fp8_gemm_bias_kernel[grid](
        a2,
        b2,
        bias,
        a_scales,
        b_scales,
        c,
        M,
        N,
        K,
        a2.stride(0),
        a2.stride(1),
        b2.stride(0),
        b2.stride(1),
        c.stride(0),
        c.stride(1),
        fuse_gelu=fuse_gelu,
        GROUP_M=8,
        EVEN_K=even_k,
    )
    return c.view(*out_shape)


# ──────────────────────────────────────────────────────────────────────────
# FP8 GEMM without bias — e4m3fn / e4m3fnuz only; EVEN_K from K % BLOCK_K
# ──────────────────────────────────────────────────────────────────────────


@autotune(
    configs=[
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@jit
def fp8_gemm_kernel(
    A,
    B,
    A_SCALES,
    B_SCALES,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    fuse_gelu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fp8 gemm kernel.

    Args:
        A: The a.
        B: The b.
        A_SCALES: The a scales.
        B_SCALES: The b scales.
        C: The c.
        M: The m.
        N: The n.
        K: The k.
        stride_am: The stride am.
        stride_ak: The stride ak.
        stride_bk: The stride bk.
        stride_bn: The stride bn.
        stride_cm: The stride cm.
        stride_cn: The stride cn.
        fuse_gelu: The fuse gelu.
        BLOCK_M: The block m.
        BLOCK_N: The block n.
        BLOCK_K: The block k.
        GROUP_M: The group m.
        SPLIT_K: The split k.
        EVEN_K: The even k.
    """
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A_ptr = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A_ptr, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B_ptr, mask=rk[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision=None)

        A_ptr += BLOCK_K * SPLIT_K * stride_ak
        B_ptr += BLOCK_K * SPLIT_K * stride_bk

    a_scales_ptr = A_SCALES + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_scales_ptr = B_SCALES + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scales = tl.load(a_scales_ptr).to(tl.float32)  # [BM]
    b_scales = tl.load(b_scales_ptr).to(tl.float32)  # [BN]

    out = (acc * a_scales[:, None]) * b_scales[None, :]
    if fuse_gelu:
        out = gelu(out)

    out = out.to(C.dtype.element_ty)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    if SPLIT_K == 1:
        tl.store(C_ptr, out, mask=mask)
    else:
        tl.atomic_add(C_ptr, out, mask=mask)


def fp8_gemm_triton(a, b, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    """Fp8 gemm triton.

    Args:
        a: The a.
        b: The b.
        a_scales: The a scales.
        b_scales: The b scales.
        fuse_gelu: The fuse gelu.
        output_dtype: The output dtype.
    """
    assert a.is_cuda and b.is_cuda
    e4m3_ok = []
    if hasattr(torch, "float8_e4m3fn"):
        e4m3_ok.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e4m3fnuz"):
        e4m3_ok.append(torch.float8_e4m3fnuz)
    e4m3_ok = tuple(e4m3_ok)

    assert a.dtype in e4m3_ok, f"a.dtype={a.dtype} is not FP8 E4M3"
    assert b.dtype in e4m3_ok, f"b.dtype={b.dtype} is not FP8 E4M3"

    a_orig_shape = a.shape
    a2 = a.view(-1, a.shape[-1])
    b2 = b.t()

    if a2.stride(0) > 1 and a2.stride(1) > 1:
        a2 = a2.contiguous()
    if b2.stride(0) > 1 and b2.stride(1) > 1:
        b2 = b2.contiguous()

    M, K = a2.shape
    _, N = b2.shape
    out_shape = a_orig_shape[:-1] + (N,)

    if output_dtype is None:
        output_dtype = torch.float16

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    grid = lambda META: (  # noqa E731
        cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )  # noqa E731
    even_k = K % 128 == 0

    fp8_gemm_kernel[grid](
        a2,
        b2,
        a_scales,
        b_scales,
        c,
        M,
        N,
        K,
        a2.stride(0),
        a2.stride(1),
        b2.stride(0),
        b2.stride(1),
        c.stride(0),
        c.stride(1),
        fuse_gelu=fuse_gelu,
        GROUP_M=8,
        EVEN_K=even_k,
    )
    return c.view(*out_shape)
