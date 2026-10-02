"""Fused per-token (row-wise) FP8 activation quantization in Triton.

Responsibility: one-pass row-wise amax + decode scale + clamped FP8
codes so dynamic activation quantization does not dominate the FP8 GEMM.

This module is not a Linear layer and is not a public package export.
Callers go through :func:`_quantize_rowwise_fp8` in ``quantization.py``,
which falls back to the portable Torch path when Triton is missing.

Public surface:
- :func:`quantize_rowwise_fp8_triton` — ``[rows, K]`` FP8 + ``[rows, 1]``
  FP32 scales, bit-compatible with :func:`_quantize_rowwise_fp8`.

The pure-PyTorch row-wise quantizer (``abs -> amax -> divide -> clamp -> cast``)
makes several passes over HBM and, for large ``K``, costs more than the FP8 GEMM
it feeds. Numerics match ``_quantize_rowwise_fp8`` exactly (symmetric,
round-to-nearest-even via hardware FP8 cast, ``scale = amax / fp8_max`` clamped
to the smallest positive FP32).
"""

from __future__ import annotations

import torch

from worldfoundry.core.execution.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="fp8-triton")

# ──────────────────────────────────────────────────────────────────────────
# Kernel — one row per program; BLOCK_SIZE is next_power_of_2(K)
# ──────────────────────────────────────────────────────────────────────────

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_FP8_TL_DTYPE = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}


@triton.jit
def _quantize_rowwise_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_cols,
    fp8_max,
    tiny,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    """One program per row: amax, scale, clamp, hardware FP8 store."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    in_ptrs = input_ptr + row * input_row_stride + col_offsets
    values = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(amax / fp8_max, tiny)
    quantized = values / scale
    quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)
    out_ptrs = output_ptr + row * output_row_stride + col_offsets
    tl.store(out_ptrs, quantized.to(FP8_DTYPE), mask=mask)
    tl.store(scale_ptr + row, scale)


# ──────────────────────────────────────────────────────────────────────────
# Launch — BLOCK_SIZE = next_power_of_2(K); empty rows skip the kernel
# ──────────────────────────────────────────────────────────────────────────


def quantize_rowwise_fp8_triton(
    value: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise FP8 quantization of a 2D ``[rows, K]`` tensor.

    Returns row-major ``[rows, K]`` FP8 codes and an FP32 ``[rows, 1]`` decode
    scale, bit-compatible with :func:`_quantize_rowwise_fp8`.
    """

    if dtype not in _FP8_TL_DTYPE:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    if value.ndim != 2:
        raise ValueError("row-wise FP8 quantization expects a 2D [rows, K] tensor")
    source = value if value.is_contiguous() else value.contiguous()
    rows, cols = source.shape
    output = torch.empty_like(source, dtype=dtype)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=source.device)
    if rows == 0:
        return output, scale
    block_size = triton.next_power_of_2(cols)
    num_warps = 4 if block_size <= 2048 else (8 if block_size <= 8192 else 16)
    _quantize_rowwise_fp8_kernel[(rows,)](
        source,
        output,
        scale,
        cols,
        float(torch.finfo(dtype).max),
        float(torch.finfo(torch.float32).tiny),
        source.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
        FP8_DTYPE=_FP8_TL_DTYPE[dtype],
        num_warps=num_warps,
    )
    return output, scale


__all__ = ["quantize_rowwise_fp8_triton"]
