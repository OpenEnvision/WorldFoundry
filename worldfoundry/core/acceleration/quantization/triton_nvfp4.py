"""Portable Triton encoder for NVFP4 1x16 activation quantization.

Responsibility: pack activations and already-swizzled E4M3 block
scales for :class:`NVFP4Linear` with the same RNE / two-level scale
contract as :func:`quantize_nvfp4`.

This module is not a Linear layer and is not part of the package
``__all__``. Callers use ``nvfp4.py`` public APIs; OOM must propagate
so the linear does not silently fall back after an allocator failure.

Public surface:
- :func:`triton_quantize_nvfp4` — packed FP4 bytes + swizzled FP8 scales.
"""

from __future__ import annotations

import torch

from worldfoundry.core.execution.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="nvfp4-triton")

# ──────────────────────────────────────────────────────────────────────────
# Kernel — 1x16 blocks; E4M3 local scale + RNE E2M1; cuBLAS 128x4 swizzle
# ──────────────────────────────────────────────────────────────────────────

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402


@triton.jit
def _quantize_nvfp4_kernel(
    input_ptr,
    global_scale_ptr,
    packed_ptr,
    swizzled_scale_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    scale_columns: tl.constexpr,
    padded_rows: tl.constexpr,
    padded_scale_columns: tl.constexpr,
):
    """One 1x16 block: E4M3 local scale, RNE E2M1 codes, cuBLAS swizzle store."""
    row = tl.program_id(0)
    scale_column = tl.program_id(1)
    valid_block = (row < rows) & (scale_column < scale_columns)
    offsets = tl.arange(0, 16)
    input_offsets = row * width + scale_column * 16 + offsets
    values = tl.load(input_ptr + input_offsets, mask=valid_block, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    block_amax = tl.max(tl.abs(values), axis=0)
    local_scale = (block_amax / 6.0) / global_scale
    # Match ``quantize_nvfp4`` exactly.  Using the smallest subnormal here
    # would also avoid division by zero, but it gives zero/very-small blocks a
    # different encoded scale from the reference and pre-quantized weights.
    local_scale = tl.minimum(tl.maximum(local_scale, 2**-6), 448.0)
    is_subnormal = local_scale < 2**-6
    subnormal_mantissa = libdevice.rint(local_scale * 512.0).to(tl.int32)
    subnormal_mantissa = tl.minimum(tl.maximum(subnormal_mantissa, 1), 7)
    exponent = tl.floor(tl.log2(local_scale)).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, -6), 8)
    step = tl.exp2(exponent.to(tl.float32) - 3.0)
    significand = libdevice.rint(local_scale / step).to(tl.int32)
    carry = significand >= 16
    exponent = tl.where(carry, exponent + 1, exponent)
    significand = tl.where(carry, 8, significand)
    step = tl.where(carry, step * 2.0, step)
    exponent = tl.minimum(exponent, 8)
    significand = tl.where(exponent == 8, tl.minimum(significand, 14), significand)
    normal_code = ((exponent + 7) << 3) | (significand - 8)
    local_scale_code = tl.where(is_subnormal, subnormal_mantissa, normal_code)
    decoded_local_scale = tl.where(
        is_subnormal,
        subnormal_mantissa.to(tl.float32) * (2**-9),
        significand.to(tl.float32) * step,
    )

    # FP4 codes are selected at a small set of exact midpoints.  Triton's
    # regular division may use an approximate reciprocal and flip RNE ties;
    # the reference path uses correctly rounded FP32 division.
    normalized = libdevice.div_rn(values, global_scale * decoded_local_scale)
    normalized = tl.minimum(tl.maximum(normalized, -6.0), 6.0)
    magnitude = tl.abs(normalized)

    # Positive E2M1 values are {0,.5,1,1.5,2,3,4,6}. Comparisons at exact
    # midpoints alternate to implement round-to-nearest, ties-to-even.
    code = tl.zeros((16,), dtype=tl.int32)
    code = tl.where(magnitude > 0.25, 1, code)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    code = code | tl.where(normalized < 0.0, 8, 0)

    pair = tl.arange(0, 8)
    code_pairs = tl.reshape(code, (8, 2))
    nibble_weights = tl.reshape(1 << (tl.arange(0, 2) * 4), (1, 2))
    packed = tl.sum(code_pairs * nibble_weights, axis=1)
    packed_offsets = row * (width // 2) + scale_column * 8 + pair
    tl.store(packed_ptr + packed_offsets, packed, mask=valid_block)

    row_block = row // 128
    local_row = row % 128
    column_block = scale_column // 4
    local_column = scale_column % 4
    column_blocks = padded_scale_columns // 4
    tile_base = (row_block * column_blocks + column_block) * 512
    swizzled_offset = tile_base + (local_row % 32) * 16 + (local_row // 32) * 4 + local_column
    scale_value = tl.where(valid_block, local_scale_code, 0)
    tl.store(swizzled_scale_ptr + swizzled_offset, scale_value)


# ──────────────────────────────────────────────────────────────────────────
# Launch — pad to 128×4 so the swizzle store matches cuBLAS layout
# ──────────────────────────────────────────────────────────────────────────


def triton_quantize_nvfp4(
    value: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed FP4 bytes and already-swizzled FP8 block scales."""

    if value.ndim != 2 or not value.is_cuda or value.shape[1] % 16:
        raise ValueError("Triton NVFP4 quantization expects CUDA [rows, K] with K % 16 == 0")
    rows, width = (int(dim) for dim in value.shape)
    scale_columns = width // 16
    padded_rows = triton.cdiv(rows, 128) * 128
    padded_scale_columns = triton.cdiv(scale_columns, 4) * 4
    packed = torch.empty((rows, width // 2), device=value.device, dtype=torch.uint8)
    scales = torch.empty(
        padded_rows * padded_scale_columns,
        device=value.device,
        dtype=torch.uint8,
    )
    _quantize_nvfp4_kernel[(padded_rows, padded_scale_columns)](
        value,
        global_scale,
        packed,
        scales,
        rows=rows,
        width=width,
        scale_columns=scale_columns,
        padded_rows=padded_rows,
        padded_scale_columns=padded_scale_columns,
        num_warps=1,
        num_stages=1,
    )
    return packed, scales.view(torch.float8_e4m3fn)


__all__ = ["triton_quantize_nvfp4"]
