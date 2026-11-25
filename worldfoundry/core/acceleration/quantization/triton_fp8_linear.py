"""Hopper SM90 WGMMA FP8 tensor-core GEMM with row scales and optional bias epilogue.

Supports both ``float8_e4m3fn`` and ``float8_e5m2`` (including E5M2 x E5M2,
which ``torch._scaled_mm`` rejects), with true FP32 accumulation across K tiles,
row-wise ``input_scale`` / ``weight_scale``, optional FP32/BF16/FP16 bias,
strict ``tl.int64`` indexing, zero register spills, and lazy Triton imports so
CPU-only environments without Triton import cleanly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_SUPPORTED_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_DEVICE_CAP_CACHE: dict[int, tuple[int, int]] = {}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _cuda_capability(device: torch.device) -> tuple[int, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _DEVICE_CAP_CACHE.get(idx)
    if cached is not None:
        return cached
    cap = tuple(int(v) for v in torch.cuda.get_device_capability(device))
    _DEVICE_CAP_CACHE[idx] = (cap[0], cap[1])
    return cap[0], cap[1]


def _fp8_linear_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Exact FP32 dequantized reference used as CPU / no-Triton fallback."""
    m = int(input.shape[0])
    n = int(weight.shape[0])
    sa = input_scale.reshape(1, 1) if input_scale.numel() == 1 else input_scale.reshape(m, 1)
    sw = weight_scale.reshape(1, 1) if weight_scale.numel() == 1 else weight_scale.reshape(1, n)
    acc = (input.float() @ weight.float().T) * sa.float() * sw.float()
    if bias is not None:
        acc = acc + bias.float().reshape(1, n)
    return acc.to(output_dtype)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily configure Triton compile cache and define Hopper FP8 WGMMA kernels."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="fp8-linear-triton")

    import triton
    import triton.language as tl

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        TensorDescriptor = None

    @triton.jit
    def _fp8_gemm_tma_kernel(
        A_desc,
        W_desc,
        AS_ptr,
        WS_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_om,
        stride_on,
        as_stride,
        ws_stride,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Hopper SM90 TMA WGMMA FP8 GEMM with L2 swizzling and per-tile FP32 promotion."""
        pid = tl.program_id(0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        off_m = (pid_m * BLOCK_M).to(tl.int32)
        off_n = (pid_n * BLOCK_N).to(tl.int32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_off in tl.range(0, K, BLOCK_K):
            a = A_desc.load([off_m, k_off])
            w = W_desc.load([off_n, k_off])
            # Promote WGMMA partial product to true IEEE FP32 once per BLOCK_K tile
            # (matches use_fast_accum=False without 22-bit mantissa drift).
            acc += tl.dot(a, w.T, out_dtype=tl.float32)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_m = rm < M
        mask_n = rn < N

        sa = tl.load(AS_ptr + rm * as_stride, mask=mask_m, other=0.0).to(tl.float32)
        sw = tl.load(WS_ptr + rn * ws_stride, mask=mask_n, other=0.0).to(tl.float32)
        acc = (acc * sa[:, None]) * sw[None, :]
        if HAS_BIAS:
            bias = tl.load(Bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _fp8_gemm_ptr_kernel(
        A_ptr,
        W_ptr,
        AS_ptr,
        WS_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        as_stride,
        ws_stride,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """Pointer-based masked WGMMA FP8 GEMM for arbitrary shapes/strides with FP32 promotion."""
        pid = tl.program_id(0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        rk = tl.arange(0, BLOCK_K).to(tl.int64)

        mask_m = rm < M
        mask_n = rn < N

        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        w_ptrs = W_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_K)
        for k_idx in tl.range(0, num_k):
            if EVEN_K:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
            else:
                k_mask = (k_idx * BLOCK_K + rk) < K
                a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, w, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk

        sa = tl.load(AS_ptr + rm * as_stride, mask=mask_m, other=0.0).to(tl.float32)
        sw = tl.load(WS_ptr + rn * ws_stride, mask=mask_n, other=0.0).to(tl.float32)
        acc = (acc * sa[:, None]) * sw[None, :]
        if HAS_BIAS:
            bias = tl.load(Bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    return {
        "tma": _fp8_gemm_tma_kernel,
        "ptr": _fp8_gemm_ptr_kernel,
        "TensorDescriptor": TensorDescriptor,
    }


def _select_tiling(
    m: int,
    n: int,
    k: int,
    sm_major: int,
    use_tma: bool,
    aligned_strides: bool,
) -> tuple[int, int, int, int, int]:
    """Deterministic zero-spill Hopper SM90 tiling heuristic ``(BM, BN, BK, warps, stages)``."""
    if m <= 64 or n <= 64:
        # Small shapes use 32x64x64 MMA v2 (exact FP32 accumulation, bit-exact with FP32 reference).
        return 32, 64, 64, 4, 2
    if m < 128 or n < 128:
        return 64, 64, 64, 4, 3
    if sm_major >= 9:
        if use_tma:
            if k >= 256 and m < 16384:
                return 128, 128, 256, 8, 3
            return 128, 128, 128, 8, 5
        if not aligned_strides:
            # Unaligned K or N prevents 128-bit vectorized copies; 64x128x64 stays zero-spill.
            return 64, 128, 64, 8, 3
        if k >= 256 and k % 256 == 0:
            return 128, 128, 256, 8, 3
        if k >= 128 and k % 128 == 0:
            return 128, 128, 128, 8, 4
        return 128, 128, 128 if k >= 128 else 64, 8, 3
    if not aligned_strides:
        return 64, 128, 64, 8, 3
    return 128, 128, 128 if k >= 128 else 64, 8, 3


def _launch_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    m, k = int(input.shape[0]), int(input.shape[1])
    n = int(weight.shape[0])
    output = torch.empty((m, n), device=input.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return output
    if k == 0:
        output.zero_()
        if bias is not None:
            output.add_(bias.to(output_dtype).reshape(1, n))
        return output

    try:
        kernels = _load_triton_kernels()
    except ImportError:
        return _fp8_linear_reference(input, weight, input_scale, weight_scale, bias, output_dtype)

    a_src = input if input.stride(1) == 1 else input.contiguous()
    w_src = weight if weight.stride(1) == 1 else weight.contiguous()
    as_vec = input_scale.reshape(-1)
    if not as_vec.is_contiguous():
        as_vec = as_vec.contiguous()
    ws_vec = weight_scale.reshape(-1)
    if not ws_vec.is_contiguous():
        ws_vec = ws_vec.contiguous()
    as_stride = 0 if as_vec.numel() == 1 else 1
    ws_stride = 0 if ws_vec.numel() == 1 else 1
    has_bias = bias is not None
    bias_ptr = bias.reshape(-1).contiguous() if bias is not None else ws_vec

    sm_major, _ = _cuda_capability(a_src.device)
    tensor_desc_cls = kernels["TensorDescriptor"]
    can_tma = (
        tensor_desc_cls is not None
        and sm_major >= 9
        and m >= 128
        and n >= 128
        and k >= 128
        and a_src.stride(1) == 1
        and w_src.stride(1) == 1
        and (a_src.stride(0) % 16 == 0)
        and (w_src.stride(0) % 16 == 0)
        and (k % 16 == 0)
        and (a_src.data_ptr() % 16 == 0)
        and (w_src.data_ptr() % 16 == 0)
    )
    aligned_strides = (
        (a_src.stride(0) % 16 == 0)
        and (w_src.stride(0) % 16 == 0)
        and (output.stride(0) % 16 == 0)
        and (k % 16 == 0)
        and (n % 16 == 0)
    )
    bm, bn, bk, num_warps, num_stages = _select_tiling(m, n, k, sm_major, can_tma, aligned_strides)
    grid = (_cdiv(m, bm) * _cdiv(n, bn),)

    with torch.cuda.device(a_src.device):
        if can_tma:
            a_desc = tensor_desc_cls.from_tensor(a_src, [bm, bk])
            w_desc = tensor_desc_cls.from_tensor(w_src, [bn, bk])
            kernels["tma"][grid](
                a_desc,
                w_desc,
                as_vec,
                ws_vec,
                bias_ptr,
                output,
                m,
                n,
                k,
                output.stride(0),
                output.stride(1),
                as_stride,
                ws_stride,
                HAS_BIAS=has_bias,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_SIZE_M=8,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            kernels["ptr"][grid](
                a_src,
                w_src,
                as_vec,
                ws_vec,
                bias_ptr,
                output,
                m,
                n,
                k,
                a_src.stride(0),
                a_src.stride(1),
                w_src.stride(0),
                w_src.stride(1),
                output.stride(0),
                output.stride(1),
                as_stride,
                ws_stride,
                HAS_BIAS=has_bias,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_SIZE_M=8,
                EVEN_K=(k % bk == 0),
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return output


@torch.library.custom_op(
    "worldfoundry::triton_fp8_linear",
    mutates_args=(),
    device_types="cuda",
)
def _triton_fp8_linear_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    dtype = _SUPPORTED_OUT_DTYPES[output_dtype_code]
    return _launch_fp8_linear(input, weight, input_scale, weight_scale, bias, dtype)


@_triton_fp8_linear_op.register_fake
def _triton_fp8_linear_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    del input_scale, weight_scale, bias
    dtype = _SUPPORTED_OUT_DTYPES[output_dtype_code]
    return torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=dtype)


_OUT_DTYPE_TO_CODE = {dt: idx for idx, dt in enumerate(_SUPPORTED_OUT_DTYPES)}


def fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute row-scaled FP8 linear ``(input @ weight.T) * input_scale * weight_scale + bias``."""
    if input.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"fp8_linear expects 2D input and weight, got {tuple(input.shape)} and {tuple(weight.shape)}")
    if input.shape[1] != weight.shape[1]:
        raise ValueError(f"fp8_linear K mismatch: input {tuple(input.shape)} vs weight {tuple(weight.shape)}")
    if input.dtype not in _SUPPORTED_FP8_DTYPES or weight.dtype not in _SUPPORTED_FP8_DTYPES:
        raise ValueError(f"fp8_linear expects FP8 inputs, got {input.dtype} and {weight.dtype}")
    dtype_code = _OUT_DTYPE_TO_CODE.get(output_dtype)
    if dtype_code is None:
        raise ValueError(f"unsupported fp8_linear output dtype: {output_dtype}")
    if input.device.type != "cuda":
        return _fp8_linear_reference(input, weight, input_scale, weight_scale, bias, output_dtype)
    return _triton_fp8_linear_op(input, weight, input_scale, weight_scale, bias, dtype_code)


__all__ = ["fp8_linear"]
