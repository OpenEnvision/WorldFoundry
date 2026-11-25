"""Fused per-token (row-wise) FP8 activation quantization in Triton.

Responsibility: one-pass row-wise amax + decode scale + clamped FP8
codes so dynamic activation quantization does not dominate the FP8 GEMM.

This module is not a Linear layer and is not a public package export.
Callers go through :func:`_quantize_rowwise_fp8` in ``quantization.py``,
which falls back to the portable Torch path when Triton is missing.

Public surface:
- :func:`quantize_rowwise_fp8_triton` — ``[rows, K]`` FP8 + ``[rows, 1]``
  FP32 scales (or ``[rows, K // 2]`` for ``activation="silu_mul"``),
  bit-compatible with :func:`_quantize_rowwise_fp8`.

Key properties:
- **Lazy Triton import**: importing this module on CPU-only environments never
  imports ``triton`` or initializes CUDA.
- **Exact non-pow2 2-chunk single-pass**: decomposes non-power-of-2 FFN widths
  (e.g., ``5120 = 4096 + 1024``, ``20480 = 16384 + 4096``, ``24576 = 16384 + 8192``)
  into two exact register-resident tiles with zero register spills, avoiding
  wasted masked transcendental ops and second HBM passes.
- **Hoisted-reduction wide tiled path**: rows wider than ``24576`` hoist the
  per-tile max reduction out of the loop, apply L2 eviction hints, and use the
  exact 3-point unimodal extremum theorem for GELU/GELU-tanh (zero extra
  transcendental pass).
- **Fused SwiGLU (``activation="silu_mul"``)**: consumes packed ``[rows, 2*K]``
  ``gate | up`` and emits ``[rows, K]`` FP8 codes + ``[rows, 1]`` FP32 scales,
  preserving PyTorch's ``F.silu(gate) * up`` input-dtype rounding boundaries.
- **Compile & CUDA Graph safe**: registered as ``worldfoundry::quantize_rowwise_fp8``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_SUPPORTED_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_SUPPORTED_ACTIVATIONS = {"identity", "gelu", "gelu_tanh", "silu_mul"}
_FAST_TANH_CAP_CACHE: dict[int, bool] = {}


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _prev_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _supports_fast_tanh(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _FAST_TANH_CAP_CACHE.get(idx)
    if cached is not None:
        return cached
    supported = torch.cuda.get_device_capability(device) >= (7, 5)
    _FAST_TANH_CAP_CACHE[idx] = supported
    return supported


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily configure Triton compile cache and define FP8 quantization kernels."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="fp8-triton")

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    fp8_tl_dtype_map = {
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e5m2: tl.float8e5,
    }

    @triton.jit
    def _tanh_approx_f32(value):
        """Single-instruction SM75+ PTX fast tanh matching PyTorch's BF16 GELU(tanh)."""
        return tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _eval_gelu_scalar(x, in_ty: tl.constexpr, ACTIVATION: tl.constexpr, USE_FAST_TANH: tl.constexpr):
        if ACTIVATION == "gelu_tanh":
            inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            tanh_val = _tanh_approx_f32(inner) if USE_FAST_TANH else libdevice.tanh(inner)
            y = 0.5 * x * (1.0 + tanh_val)
        else:
            y = 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))
        return y.to(in_ty).to(tl.float32)

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
        ACTIVATION: tl.constexpr = "identity",
        USE_FAST_TANH: tl.constexpr = False,
    ):
        """One program per row (1-chunk single pass): optional fused activation, amax, scale, FP8 store."""
        row = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = col_offsets < n_cols
        in_ptrs = input_ptr + row * input_row_stride + col_offsets
        values = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "gelu_tanh":
            inner = 0.7978845608028654 * (values + 0.044715 * values * values * values)
            if USE_FAST_TANH:
                values = 0.5 * values * (1.0 + _tanh_approx_f32(inner))
            else:
                values = 0.5 * values * (1.0 + libdevice.tanh(inner))
        elif ACTIVATION == "gelu":
            values = 0.5 * values * (1.0 + libdevice.erf(values * 0.7071067811865476))
        if ACTIVATION != "identity":
            # Preserve nn.GELU's output rounding before the following Linear's
            # row quantization; there is no materialized activation in HBM.
            values = values.to(input_ptr.dtype.element_ty).to(tl.float32)
            values = tl.where(mask, values, 0.0)
        amax = tl.max(tl.abs(values), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        quantized = tl.minimum(tl.maximum(values * inv_scale, -fp8_max_f), fp8_max_f)
        out_ptrs = output_ptr + row * output_row_stride + col_offsets
        tl.store(out_ptrs, quantized.to(FP8_DTYPE), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _quantize_rowwise_fp8_2chunk_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        input_row_stride,
        output_row_stride,
        B0: tl.constexpr,
        B1: tl.constexpr,
        FP8_DTYPE: tl.constexpr,
        ACTIVATION: tl.constexpr = "identity",
        USE_FAST_TANH: tl.constexpr = False,
    ):
        """Register-resident 2-chunk single-pass quantizer for non-pow2 and wide rows (<= 24576)."""
        row = tl.program_id(0).to(tl.int64)
        row_in = input_ptr + row * input_row_stride
        row_out = output_ptr + row * output_row_stride
        in_ty = input_ptr.dtype.element_ty

        c0 = tl.arange(0, B0).to(tl.int64)
        x0 = tl.load(row_in + c0).to(tl.float32)
        if ACTIVATION == "gelu_tanh":
            inner0 = 0.7978845608028654 * (x0 + 0.044715 * x0 * x0 * x0)
            if USE_FAST_TANH:
                x0 = 0.5 * x0 * (1.0 + _tanh_approx_f32(inner0))
            else:
                x0 = 0.5 * x0 * (1.0 + libdevice.tanh(inner0))
            x0 = x0.to(in_ty).to(tl.float32)
        elif ACTIVATION == "gelu":
            x0 = (0.5 * x0 * (1.0 + libdevice.erf(x0 * 0.7071067811865476))).to(in_ty).to(tl.float32)

        c1 = B0 + tl.arange(0, B1).to(tl.int64)
        m1 = c1 < n_cols
        x1 = tl.load(row_in + c1, mask=m1, other=0.0).to(tl.float32)
        if ACTIVATION == "gelu_tanh":
            inner1 = 0.7978845608028654 * (x1 + 0.044715 * x1 * x1 * x1)
            if USE_FAST_TANH:
                x1 = 0.5 * x1 * (1.0 + _tanh_approx_f32(inner1))
            else:
                x1 = 0.5 * x1 * (1.0 + libdevice.tanh(inner1))
            x1 = tl.where(m1, x1.to(in_ty).to(tl.float32), 0.0)
        elif ACTIVATION == "gelu":
            x1 = tl.where(
                m1,
                (0.5 * x1 * (1.0 + libdevice.erf(x1 * 0.7071067811865476))).to(in_ty).to(tl.float32),
                0.0,
            )

        amax = tl.maximum(tl.max(tl.abs(x0), axis=0), tl.max(tl.abs(x1), axis=0))
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)
        tl.store(row_out + c0, tl.minimum(tl.maximum(x0 * inv_scale, -fp8_max_f), fp8_max_f).to(FP8_DTYPE))
        tl.store(row_out + c1, tl.minimum(tl.maximum(x1 * inv_scale, -fp8_max_f), fp8_max_f).to(FP8_DTYPE), mask=m1)

    @triton.jit
    def _quantize_rowwise_fp8_tiled_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        input_row_stride,
        output_row_stride,
        BLOCK_N: tl.constexpr,
        FP8_DTYPE: tl.constexpr,
        ACTIVATION: tl.constexpr = "identity",
        USE_FAST_TANH: tl.constexpr = False,
    ):
        """Two-pass hoisted-reduction tiled quantizer for ultra-wide rows.

        Hoists the cross-warp max reduction out of the column loop, hints L2
        retention on pass 1 and streaming eviction on pass 2, and applies the
        exact 3-point unimodal extremum theorem for GELU/GELU-tanh so pass 1
        runs zero vector transcendentals.
        """
        row = tl.program_id(0).to(tl.int64)
        row_in = input_ptr + row * input_row_stride
        row_out = output_ptr + row * output_row_stride
        in_ty = input_ptr.dtype.element_ty

        if ACTIVATION == "identity":
            amax_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for start in tl.range(0, n_cols, BLOCK_N):
                col = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
                mask = col < n_cols
                values = tl.load(row_in + col, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
                amax_vec = tl.maximum(amax_vec, tl.abs(values))
            amax = tl.max(amax_vec, axis=0)
        else:
            # GELU and GELU-tanh are strictly increasing on [0, +inf) and strictly
            # unimodal on (-inf, 0] with unique global minimum x_star. Tracking the
            # 3 extremal inputs across the row yields the exact row amax with 3
            # scalar evaluations instead of N vector transcendentals.
            x_star: tl.constexpr = -0.7525740765 if ACTIVATION == "gelu_tanh" else -0.7517915241
            neg_inf = tl.full((BLOCK_N,), float("-inf"), tl.float32)
            pos_inf = tl.full((BLOCK_N,), float("inf"), tl.float32)
            x_max_v = neg_inf
            x_left_v = neg_inf
            x_right_v = pos_inf
            for start in tl.range(0, n_cols, BLOCK_N):
                col = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
                mask = col < n_cols
                x = tl.load(row_in + col, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
                x_max_v = tl.maximum(x_max_v, tl.where(mask, x, neg_inf))
                x_left_v = tl.maximum(x_left_v, tl.where(mask & (x <= x_star), x, neg_inf))
                x_right_v = tl.minimum(x_right_v, tl.where(mask & (x >= x_star), x, pos_inf))
            x_max = tl.max(x_max_v, axis=0)
            x_left = tl.max(x_left_v, axis=0)
            x_right = tl.min(x_right_v, axis=0)
            a_max = tl.where(x_max > 0.0, tl.abs(_eval_gelu_scalar(x_max, in_ty, ACTIVATION, USE_FAST_TANH)), 0.0)
            a_left = tl.where(
                x_left > float("-inf"),
                tl.abs(_eval_gelu_scalar(x_left, in_ty, ACTIVATION, USE_FAST_TANH)),
                0.0,
            )
            a_right = tl.where(
                x_right < 0.0,
                tl.abs(_eval_gelu_scalar(x_right, in_ty, ACTIVATION, USE_FAST_TANH)),
                0.0,
            )
            amax = tl.where(x_max != x_max, x_max, tl.maximum(a_max, tl.maximum(a_left, a_right)))

        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)

        for start in tl.range(0, n_cols, BLOCK_N):
            col = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
            mask = col < n_cols
            values = tl.load(row_in + col, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            if ACTIVATION == "gelu_tanh":
                inner = 0.7978845608028654 * (values + 0.044715 * values * values * values)
                if USE_FAST_TANH:
                    values = 0.5 * values * (1.0 + _tanh_approx_f32(inner))
                else:
                    values = 0.5 * values * (1.0 + libdevice.tanh(inner))
                values = values.to(in_ty).to(tl.float32)
            elif ACTIVATION == "gelu":
                values = (0.5 * values * (1.0 + libdevice.erf(values * 0.7071067811865476))).to(in_ty).to(tl.float32)
            quantized = tl.minimum(tl.maximum(values * inv_scale, -fp8_max_f), fp8_max_f)
            tl.store(row_out + col, quantized.to(FP8_DTYPE), mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _quantize_rowwise_fp8_silu_mul_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        half_cols,
        fp8_max,
        tiny,
        input_row_stride,
        output_row_stride,
        BLOCK_SIZE: tl.constexpr,
        FP8_DTYPE: tl.constexpr,
    ):
        """Single-pass fused SwiGLU (``gate | up``) + rowwise FP8 quantizer (``K <= 16384``)."""
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = col < half_cols
        gate_row = input_ptr + row * input_row_stride
        up_row = gate_row + half_cols
        in_ty = input_ptr.dtype.element_ty

        gate = tl.load(gate_row + col, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_row + col, mask=mask, other=0.0).to(tl.float32)
        # Preserve PyTorch's F.silu(gate) * up input-dtype rounding boundaries.
        silu_gate = (gate / (1.0 + tl.math.exp2(-gate * 1.4426950408889634))).to(in_ty).to(tl.float32)
        values = tl.where(mask, (silu_gate * up).to(in_ty).to(tl.float32), 0.0)

        amax = tl.max(tl.abs(values), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        quantized = tl.minimum(tl.maximum(values * inv_scale, -fp8_max_f), fp8_max_f)
        tl.store(output_ptr + row * output_row_stride + col, quantized.to(FP8_DTYPE), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _quantize_rowwise_fp8_silu_mul_tiled_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        half_cols,
        fp8_max,
        tiny,
        input_row_stride,
        output_row_stride,
        BLOCK_N: tl.constexpr,
        FP8_DTYPE: tl.constexpr,
    ):
        """Two-pass hoisted-reduction fused SwiGLU + rowwise FP8 quantizer (``K > 16384``)."""
        row = tl.program_id(0).to(tl.int64)
        gate_row = input_ptr + row * input_row_stride
        up_row = gate_row + half_cols
        out_row = output_ptr + row * output_row_stride
        in_ty = input_ptr.dtype.element_ty

        amax_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for start in tl.range(0, half_cols, BLOCK_N):
            col = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
            mask = col < half_cols
            gate = tl.load(gate_row + col, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
            up = tl.load(up_row + col, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
            silu_gate = (gate / (1.0 + tl.math.exp2(-gate * 1.4426950408889634))).to(in_ty).to(tl.float32)
            values = tl.where(mask, (silu_gate * up).to(in_ty).to(tl.float32), 0.0)
            amax_vec = tl.maximum(amax_vec, tl.abs(values))

        amax = tl.max(amax_vec, axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)

        for start in tl.range(0, half_cols, BLOCK_N):
            col = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
            mask = col < half_cols
            gate = tl.load(gate_row + col, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            up = tl.load(up_row + col, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            silu_gate = (gate / (1.0 + tl.math.exp2(-gate * 1.4426950408889634))).to(in_ty).to(tl.float32)
            values = (silu_gate * up).to(in_ty).to(tl.float32)
            quantized = tl.minimum(tl.maximum(values * inv_scale, -fp8_max_f), fp8_max_f)
            tl.store(out_row + col, quantized.to(FP8_DTYPE), mask=mask, eviction_policy="evict_first")

    return {
        "single": _quantize_rowwise_fp8_kernel,
        "two_chunk": _quantize_rowwise_fp8_2chunk_kernel,
        "tiled": _quantize_rowwise_fp8_tiled_kernel,
        "silu_mul_single": _quantize_rowwise_fp8_silu_mul_kernel,
        "silu_mul_tiled": _quantize_rowwise_fp8_silu_mul_tiled_kernel,
        "fp8_tl_dtype": fp8_tl_dtype_map,
    }


def _quantize_rowwise_fp8_reference(
    source: torch.Tensor,
    dtype: torch.dtype,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback used only if Triton is unavailable at runtime."""
    if activation == "gelu":
        x = torch.nn.functional.gelu(source, approximate="none")
    elif activation == "gelu_tanh":
        x = torch.nn.functional.gelu(source, approximate="tanh")
    elif activation == "silu_mul":
        half = source.shape[1] // 2
        x = torch.nn.functional.silu(source[:, :half]) * source[:, half:]
    else:
        x = source
    fp8_max = float(torch.finfo(dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    xf = x.float()
    scale = (xf.abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(tiny)
    codes = (xf / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return codes, scale


def _launch_quantize_rowwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = value if value.stride(1) == 1 else value.contiguous()
    rows, cols = int(source.shape[0]), int(source.shape[1])
    out_cols = cols // 2 if activation == "silu_mul" else cols
    output = torch.empty((rows, out_cols), dtype=dtype, device=source.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=source.device)
    if rows == 0 or out_cols == 0:
        if rows > 0:
            scale.fill_(float(torch.finfo(torch.float32).tiny))
        return output, scale

    try:
        kernels = _load_triton_kernels()
    except ImportError:
        return _quantize_rowwise_fp8_reference(source, dtype, activation)
    fp8_max = float(torch.finfo(dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    fp8_tl_dtype = kernels["fp8_tl_dtype"][dtype]

    with torch.cuda.device(source.device):
        if activation == "silu_mul":
            if out_cols <= 16384:
                block_size = _next_power_of_2(out_cols)
                num_warps = 4 if block_size <= 2048 else 8
                kernels["silu_mul_single"][(rows,)](
                    source,
                    output,
                    scale,
                    out_cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    BLOCK_SIZE=block_size,
                    FP8_DTYPE=fp8_tl_dtype,
                    num_warps=num_warps,
                )
            else:
                kernels["silu_mul_tiled"][(rows,)](
                    source,
                    output,
                    scale,
                    out_cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    BLOCK_N=4096,
                    FP8_DTYPE=fp8_tl_dtype,
                    num_warps=16,
                )
            return output, scale

        use_fast_tanh = activation == "gelu_tanh" and _supports_fast_tanh(source.device)

        if cols <= 16384:
            block_size = _next_power_of_2(cols)
            # Non-power-of-2 medium/large rows benefit significantly from 2-chunk
            # register decomposition (e.g. 5120 -> 4096+1024 avoids 37.5% wasted tanh/erf;
            # 13824 -> 8192+8192 improves gelu/gelu_tanh scheduling).
            if cols > 2048 and cols != block_size and (activation != "identity" or cols <= 8192):
                b0 = _prev_power_of_2(cols)
                b1 = _next_power_of_2(cols - b0)
                num_warps = 4 if (b0 + b1) <= 8192 else 8
                kernels["two_chunk"][(rows,)](
                    source,
                    output,
                    scale,
                    cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    B0=b0,
                    B1=b1,
                    FP8_DTYPE=fp8_tl_dtype,
                    ACTIVATION=activation,
                    USE_FAST_TANH=use_fast_tanh,
                    num_warps=num_warps,
                )
            else:
                num_warps = 4 if (block_size <= 2048 or (block_size <= 8192 and activation == "identity")) else 8
                kernels["single"][(rows,)](
                    source,
                    output,
                    scale,
                    cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    BLOCK_SIZE=block_size,
                    FP8_DTYPE=fp8_tl_dtype,
                    ACTIVATION=activation,
                    USE_FAST_TANH=use_fast_tanh,
                    num_warps=num_warps,
                )
        elif cols <= 24576:
            # 2-chunk single-pass covers ultra-wide FFN widths up to 24576 in a
            # single HBM pass (e.g., 20480 = 16384+4096, 24576 = 16384+8192) with
            # zero register spills at 16 warps; for identity at cols <= 20480,
            # hoisted-reduction tiled BN=8192 nw=16 is slightly faster still.
            if activation == "identity" and cols <= 20480:
                kernels["tiled"][(rows,)](
                    source,
                    output,
                    scale,
                    cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    BLOCK_N=8192,
                    FP8_DTYPE=fp8_tl_dtype,
                    ACTIVATION=activation,
                    USE_FAST_TANH=False,
                    num_warps=16,
                )
            else:
                b0 = 16384
                b1 = _next_power_of_2(cols - b0)
                kernels["two_chunk"][(rows,)](
                    source,
                    output,
                    scale,
                    cols,
                    fp8_max,
                    tiny,
                    source.stride(0),
                    output.stride(0),
                    B0=b0,
                    B1=b1,
                    FP8_DTYPE=fp8_tl_dtype,
                    ACTIVATION=activation,
                    USE_FAST_TANH=use_fast_tanh,
                    num_warps=16,
                )
        else:
            block_n = 8192 if activation == "identity" else 2048
            num_warps = 16 if activation == "identity" else 4
            kernels["tiled"][(rows,)](
                source,
                output,
                scale,
                cols,
                fp8_max,
                tiny,
                source.stride(0),
                output.stride(0),
                BLOCK_N=block_n,
                FP8_DTYPE=fp8_tl_dtype,
                ACTIVATION=activation,
                USE_FAST_TANH=use_fast_tanh,
                num_warps=num_warps,
            )
    return output, scale


@torch.library.custom_op(
    "worldfoundry::quantize_rowwise_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _quantize_rowwise_fp8_op(
    value: torch.Tensor,
    is_e5m2: bool,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e5m2 if is_e5m2 else torch.float8_e4m3fn
    return _launch_quantize_rowwise_fp8(value, dtype, activation)


@_quantize_rowwise_fp8_op.register_fake
def _quantize_rowwise_fp8_op_fake(
    value: torch.Tensor,
    is_e5m2: bool,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e5m2 if is_e5m2 else torch.float8_e4m3fn
    rows, cols = value.shape[0], value.shape[1]
    out_cols = cols // 2 if activation == "silu_mul" else cols
    return (
        torch.empty((rows, out_cols), dtype=dtype, device=value.device),
        torch.empty((rows, 1), dtype=torch.float32, device=value.device),
    )


def quantize_rowwise_fp8_triton(
    value: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
    *,
    activation: str = "identity",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise FP8 quantization of a 2D ``[rows, K]`` tensor.

    Returns row-major ``[rows, K]`` FP8 codes (or ``[rows, K // 2]`` when
    ``activation="silu_mul"``) and an FP32 ``[rows, 1]`` decode scale,
    bit-compatible with :func:`_quantize_rowwise_fp8`.
    """

    if dtype not in _SUPPORTED_FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported fused activation: {activation}")
    if value.device.type != "cuda" or value.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Triton FP8 quantization requires floating-point CUDA input")
    if value.ndim != 2:
        raise ValueError("row-wise FP8 quantization expects a 2D [rows, K] tensor")
    if activation == "silu_mul" and value.shape[1] % 2 != 0:
        raise ValueError(f"activation='silu_mul' expects even last dimension [rows, 2*K], got {tuple(value.shape)}")
    return _quantize_rowwise_fp8_op(value, dtype == torch.float8_e5m2, activation)


__all__ = ["quantize_rowwise_fp8_triton"]
