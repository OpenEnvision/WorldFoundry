# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for core diffusion & vision/DiT layers.

Provides fused, zero-spill (``n_spills == 0``), strict-``tl.int64``-indexed
Hopper Triton kernels registered via ``@torch.library.custom_op`` +
``.register_fake`` for:

- ``worldfoundry::layer_norm_2d_fwd`` (:func:`triton_layer_norm_2d`):
  Single-launch 2D channel-wise LayerNorm supporting both NCHW (shifted
  2D spatial-tile statistics with L1/L2 reuse) and NHWC (``channels_last``,
  single-pass register-resident channel reduction), preserving exact input
  memory layout and strides.
- ``worldfoundry::quick_gelu_fwd`` (:func:`triton_quick_gelu`):
  Single-pass ``x * sigmoid(1.702 * x)`` with exact IEEE BF16/FP16 intermediate
  rounding boundaries and dense ``channels_last``/``channels_last_3d`` support.
- ``worldfoundry::mish_fwd`` (:func:`triton_mish`):
  Single-pass ``x * tanh(softplus(x))`` with exact IEEE BF16/FP16 intermediate
  rounding boundaries and dense ``channels_last``/``channels_last_3d`` support.
- ``worldfoundry::scheduler_linear_comb_fp64`` (:func:`triton_scheduler_linear_comb_fp64`,
  :func:`triton_scheduler_convert_fp64`):
  Single-pass in-register FP64 arithmetic (``c1 * a + c2 * b`` or direct
  ``alphas_cumprod``-indexed x0/noise/velocity conversion) using strict IEEE
  FP64 instructions without materializing FP64 HBM temporaries.
- ``worldfoundry::flow_match_euler_step`` (:func:`triton_flow_match_step`,
  :func:`triton_flow_match_return_to_timestep`, :func:`triton_flow_match_add_noise`):
  Single-launch fused nearest-neighbor timestep lookup + FlowMatch Euler step /
  velocity recovery / forward noise corruption with zero host-device syncs
  (CUDA-Graph & ``torch.compile`` safe) and exact PyTorch dtype/rounding rules.
- ``worldfoundry::sinusoidal_timestep_embedding_fwd``
  (:func:`triton_sinusoidal_timestep_embedding`):
  Single-kernel DDPM sinusoidal embedding with zero-spill Cody-Waite FP32
  ``sincos``, fused ``flip_sin_to_cos``, ``scale``, and odd-dimension zero pad.

All Triton imports are strictly deferred until first eligible CUDA launch so
CPU-only processes (even without Triton installed) import and run cleanly.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Any

import torch

_FLOAT_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_FP64_INPUT_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32, torch.float64})
_INT_INDEX_DTYPES = frozenset({torch.int64, torch.int32, torch.int16, torch.uint8})
_TIMESTEP_INPUT_DTYPES = frozenset(
    {
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.uint8,
    }
)
_TORCH_BACKENDS = frozenset({"torch", "pytorch", "native", "off", "disabled"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_TORCH_BACKENDS_BYTES = frozenset({b"torch", b"pytorch", b"native", b"off", b"disabled"})
_FALSY_VALUES_BYTES = frozenset({b"0", b"false", b"no", b"off", b"disabled"})

_TRITON_SUITE: _TritonCoreLayersSuite | None = None
_TRITON_SPEC_AVAILABLE: bool | None = None
_BF16_CAP_CACHE: dict[int, bool] = {}

# Bound lazily inside _TritonCoreLayersSuite.__init__ so Triton's AST compiler
# resolves helpers from module globals without importing triton at module load.
tl: Any = None
_round_bf16: Any = None
_round_fp16: Any = None
_round_code: Any = None
_tanh_approx: Any = None
_cw_sincos_fp32: Any = None
_fp32_mul_rn: Any = None
_fp32_add_rn: Any = None
_fp32_sub_rn: Any = None
_fp32_div_rn: Any = None
_fp64_mul_rn: Any = None
_fp64_add_rn: Any = None
_fp64_div_rn: Any = None
_fp64_sqrt_rn: Any = None

_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None),
    "assume_constant_result",
    lambda fn: fn,
)


@_assume_constant_result
def core_layers_triton_ops_enabled() -> bool:
    """Return True when Hopper Triton core layer/scheduler kernels are enabled."""
    env_data = getattr(os.environ, "_data", None)
    if isinstance(env_data, dict):
        b_raw = env_data.get(b"WORLDFOUNDRY_KERNEL_BACKEND")
        if b_raw is not None and b_raw != b"auto":
            if isinstance(b_raw, bytes):
                if b_raw.strip().lower() in _TORCH_BACKENDS_BYTES:
                    return False
            elif str(b_raw).strip().casefold() in _TORCH_BACKENDS:
                return False
        r_raw = env_data.get(b"WORLDFOUNDRY_CORE_LAYERS_TRITON_OPS")
        if r_raw is None or r_raw == b"1":
            return True
        if isinstance(r_raw, bytes):
            return r_raw.strip().lower() not in _FALSY_VALUES_BYTES
        return str(r_raw).strip().casefold() not in _FALSY_VALUES
    backend = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold()
    if backend in _TORCH_BACKENDS:
        return False
    raw = os.getenv("WORLDFOUNDRY_CORE_LAYERS_TRITON_OPS", "1").strip().casefold()
    return raw not in _FALSY_VALUES


@_assume_constant_result
def _has_triton() -> bool:
    global _TRITON_SPEC_AVAILABLE
    if "triton" in sys.modules:
        return sys.modules["triton"] is not None
    if _TRITON_SPEC_AVAILABLE is None:
        try:
            _TRITON_SPEC_AVAILABLE = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            _TRITON_SPEC_AVAILABLE = False
    return _TRITON_SPEC_AVAILABLE


@_assume_constant_result
def _device_supports_bf16(device_index: int) -> bool:
    cached = _BF16_CAP_CACHE.get(device_index)
    if cached is not None:
        return cached
    major, _ = torch.cuda.get_device_capability(device_index)
    supported = major >= 8
    _BF16_CAP_CACHE[device_index] = supported
    return supported


def _is_cuda_inference_ready(*tensors: torch.Tensor) -> bool:
    """Common CUDA, backend, Triton availability, autograd, and device guard."""
    if not tensors or not core_layers_triton_ops_enabled():
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    first = tensors[0]
    if first.device.type != "cuda" or bool(getattr(torch.version, "hip", None)):
        return False
    if not _has_triton():
        return False
    dev = first.device
    for t in tensors:
        if t.device != dev:
            return False
    if any(t.dtype == torch.bfloat16 for t in tensors):
        dev_idx = 0 if dev.index is None else int(dev.index)
        if not _device_supports_bf16(dev_idx):
            return False
    return True


def _dtype_round_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_channels_last_2d(x: torch.Tensor) -> bool:
    """Return True when 4D tensor *x* should take the NHWC (channel-innermost) path."""
    if x.ndim != 4:
        return False
    if not x.is_contiguous(memory_format=torch.channels_last):
        return False
    return x.stride(1) == 1 and (not x.is_contiguous() or x.shape[2] * x.shape[3] == 1)


def _is_dense_supported_layout(x: torch.Tensor) -> bool:
    """Return True for contiguous, 4D channels_last, or 5D channels_last_3d."""
    if x.is_contiguous():
        return True
    if x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return True
    return x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d)


def _same_dense_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True when *a* and *b* share identical shape and dense memory order."""
    if a.shape != b.shape:
        return False
    if a.is_contiguous() and b.is_contiguous():
        return True
    if (
        a.ndim == 4
        and a.is_contiguous(memory_format=torch.channels_last)
        and b.is_contiguous(memory_format=torch.channels_last)
    ):
        return True
    return (
        a.ndim == 5
        and a.is_contiguous(memory_format=torch.channels_last_3d)
        and b.is_contiguous(memory_format=torch.channels_last_3d)
    )


def _get_triton_suite() -> _TritonCoreLayersSuite:
    global _TRITON_SUITE
    if _TRITON_SUITE is None:
        _TRITON_SUITE = _TritonCoreLayersSuite()
    return _TRITON_SUITE


class _TritonCoreLayersSuite:
    """Lazily-compiled Hopper SM90 Triton kernels for core NN & diffusion ops."""

    def __init__(self) -> None:
        global tl, _round_bf16, _round_fp16, _round_code, _tanh_approx, _cw_sincos_fp32
        global _fp32_mul_rn, _fp32_add_rn, _fp32_sub_rn, _fp32_div_rn
        global _fp64_mul_rn, _fp64_add_rn, _fp64_div_rn, _fp64_sqrt_rn
        try:
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="core-layers-triton")
        except (ImportError, OSError):
            pass

        import triton
        import triton.language as _tl

        tl = _tl
        self.triton = triton

        @triton.jit
        def _round_bf16_fn(value):
            return tl.inline_asm_elementwise(
                "{ .reg .b16 r; cvt.rn.bf16.f32 r, $1; mov.b32 $0, {0, r}; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _round_fp16_fn(value):
            return tl.inline_asm_elementwise(
                "{ .reg .b16 r; cvt.rn.f16.f32 r, $1; cvt.f32.f16 $0, r; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _round_code_fn(value, CODE: tl.constexpr):
            if CODE == 1:
                return _round_bf16(value)
            elif CODE == 2:
                return _round_fp16(value)
            else:
                return value

        @triton.jit
        def _tanh_approx_fn(value):
            return tl.inline_asm_elementwise(
                "tanh.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp32_mul_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[x, y],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp32_add_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[x, y],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp32_sub_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "sub.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[x, y],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp32_div_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "div.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[x, y],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp64_mul_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "mul.rn.f64 $0, $1, $2;",
                constraints="=d,d,d",
                args=[x, y],
                dtype=tl.float64,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp64_add_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "add.rn.f64 $0, $1, $2;",
                constraints="=d,d,d",
                args=[x, y],
                dtype=tl.float64,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp64_div_rn_fn(x, y):
            return tl.inline_asm_elementwise(
                "div.rn.f64 $0, $1, $2;",
                constraints="=d,d,d",
                args=[x, y],
                dtype=tl.float64,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _fp64_sqrt_rn_fn(x):
            return tl.inline_asm_elementwise(
                "sqrt.rn.f64 $0, $1;",
                constraints="=d,d",
                args=[x],
                dtype=tl.float64,
                is_pure=True,
                pack=1,
            )

        # Zero-spill FP32 sincos: 3-part Cody-Waite mod (pi/2) + minimax poly.
        # Avoids NVVM libdevice Payne-Hanek .local stack depot (n_spills == 0).
        @triton.jit
        def _cw_sincos_fp32_fn(f):
            INV_PI2: tl.constexpr = 0.6366197723675814
            P1: tl.constexpr = 1.5703125
            P2: tl.constexpr = 0.0004837512969970703
            P3: tl.constexpr = 7.549790126404332e-08
            kf = tl.extra.cuda.libdevice.rint(f * INV_PI2)
            r = tl.math.fma(-kf, P1, f)
            r = tl.math.fma(-kf, P2, r)
            r = tl.math.fma(-kf, P3, r)
            r2 = r * r

            s_poly = tl.math.fma(r2, 2.755731922398589e-06, -0.0001984126984126984)
            s_poly = tl.math.fma(r2, s_poly, 0.008333333333333333)
            s_poly = tl.math.fma(r2, s_poly, -0.16666666666666666)
            s_r = tl.math.fma(r * r2, s_poly, r)

            c_poly = tl.math.fma(r2, 2.48015873015873e-05, -0.001388888888888889)
            c_poly = tl.math.fma(r2, c_poly, 0.041666666666666664)
            c_poly = tl.math.fma(r2, c_poly, -0.5)
            c_r = tl.math.fma(r2, c_poly, 1.0)

            q = kf.to(tl.int32) & 3
            swap = (q & 1) != 0
            sin_v = tl.where(swap, c_r, s_r)
            cos_v = tl.where(swap, s_r, c_r)
            sin_v = tl.where((q & 2) != 0, -sin_v, sin_v)
            cos_v = tl.where(((q + 1) & 2) != 0, -cos_v, cos_v)
            return cos_v, sin_v

        _round_bf16 = _round_bf16_fn
        _round_fp16 = _round_fp16_fn
        _round_code = _round_code_fn
        _tanh_approx = _tanh_approx_fn
        _cw_sincos_fp32 = _cw_sincos_fp32_fn
        _fp32_mul_rn = _fp32_mul_rn_fn
        _fp32_add_rn = _fp32_add_rn_fn
        _fp32_sub_rn = _fp32_sub_rn_fn
        _fp32_div_rn = _fp32_div_rn_fn
        _fp64_mul_rn = _fp64_mul_rn_fn
        _fp64_add_rn = _fp64_add_rn_fn
        _fp64_div_rn = _fp64_div_rn_fn
        _fp64_sqrt_rn = _fp64_sqrt_rn_fn

        # ──────────────────────────────────────────────────────────────────
        # 1A. LayerNorm2d — NHWC (channels_last) single-pass & chunked
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _layer_norm_2d_nhwc_kernel(
            x_ptr,
            w_ptr,
            b_ptr,
            out_ptr,
            N: tl.int64,
            C: tl.int64,
            eps,
            BLOCK_C: tl.constexpr,
            ROWS_PER_CTA: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            c_i64 = z64 + C
            inv_c = 1.0 / c_i64.to(tl.float32)

            c_idx = tl.arange(0, BLOCK_C).to(tl.int64)
            c_mask = c_idx < c_i64

            w = tl.load(w_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)

            row_start = pid * ROWS_PER_CTA
            for r in range(ROWS_PER_CTA):
                row = row_start + r
                if row < n_i64:
                    row_off = row * c_i64 + c_idx
                    x_f = tl.load(x_ptr + row_off, mask=c_mask, other=0.0).to(tl.float32)
                    mean = tl.sum(x_f, axis=0) * inv_c
                    diff = tl.where(c_mask, x_f - mean, 0.0)
                    var = tl.sum(diff * diff, axis=0) * inv_c
                    rstd = tl.rsqrt(var + eps)
                    y = diff * rstd * w + b
                    tl.store(out_ptr + row_off, y, mask=c_mask)

        @triton.jit
        def _layer_norm_2d_nhwc_chunked_kernel(
            x_ptr,
            w_ptr,
            b_ptr,
            out_ptr,
            N: tl.int64,
            C: tl.int64,
            eps,
            BLOCK_C: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            c_i64 = z64 + C
            if row >= n_i64:
                return

            base_x = x_ptr + row * c_i64
            base_y = out_ptr + row * c_i64
            c_local = tl.arange(0, BLOCK_C).to(tl.int64)

            # Pass 1: Chan's parallel Welford over channel chunks.
            mean = tl.zeros([], dtype=tl.float32)
            m2 = tl.zeros([], dtype=tl.float32)
            count = tl.zeros([], dtype=tl.float32)
            for c_base in range(0, C, BLOCK_C):
                c_idx = c_base + c_local
                c_mask = c_idx < c_i64
                x_f = tl.load(base_x + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                rem = c_i64 - c_base
                cnt_b = tl.where(rem < BLOCK_C, rem, BLOCK_C).to(tl.float32)
                m_b = tl.sum(x_f, axis=0) / cnt_b
                d_b = tl.where(c_mask, x_f - m_b, 0.0)
                m2_b = tl.sum(d_b * d_b, axis=0)
                new_count = count + cnt_b
                delta = m_b - mean
                mean = mean + delta * (cnt_b / new_count)
                m2 = m2 + m2_b + delta * delta * (count * cnt_b / new_count)
                count = new_count

            rstd = tl.rsqrt(m2 / c_i64.to(tl.float32) + eps)

            # Pass 2: Normalize + affine + store.
            for c_base in range(0, C, BLOCK_C):
                c_idx = c_base + c_local
                c_mask = c_idx < c_i64
                x_f = tl.load(base_x + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                b = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                y = (x_f - mean) * rstd * w + b
                tl.store(base_y + c_idx, y, mask=c_mask)

        # ──────────────────────────────────────────────────────────────────
        # 1B. LayerNorm2d — NCHW L1-resident shifted 2D spatial tile
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _layer_norm_2d_nchw_kernel(
            x_ptr,
            w_ptr,
            b_ptr,
            out_ptr,
            C: tl.int64,
            HW: tl.int64,
            hw_blocks: tl.int64,
            eps,
            BLOCK_C: tl.constexpr,
            BLOCK_HW: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            c_i64 = z64 + C
            hw_i64 = z64 + HW
            hb_i64 = z64 + hw_blocks

            b_idx = pid // hb_i64
            hw_block = pid - b_idx * hb_i64

            hw_idx = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
            hw_mask = hw_idx < hw_i64

            batch_off = b_idx * c_i64 * hw_i64
            base_x = x_ptr + batch_off
            base_y = out_ptr + batch_off

            # Shift by channel 0 so large DC offsets cancel before squaring.
            x0 = tl.load(base_x + hw_idx, mask=hw_mask, other=0.0).to(tl.float32)
            c_local = tl.arange(0, BLOCK_C).to(tl.int64)

            s1 = tl.zeros([BLOCK_HW], dtype=tl.float32)
            s2 = tl.zeros([BLOCK_HW], dtype=tl.float32)
            for c_base in range(0, C, BLOCK_C):
                c_idx = c_base + c_local
                c_mask = c_idx < c_i64
                tile_mask = c_mask[:, None] & hw_mask[None, :]
                x_f = tl.load(
                    base_x + c_idx[:, None] * hw_i64 + hw_idx[None, :],
                    mask=tile_mask,
                    other=0.0,
                ).to(tl.float32)
                d = tl.where(c_mask[:, None], x_f - x0[None, :], 0.0)
                s1 += tl.sum(d, axis=0)
                s2 += tl.sum(d * d, axis=0)

            inv_c = 1.0 / c_i64.to(tl.float32)
            d_mean = s1 * inv_c
            mean = x0 + d_mean
            var = tl.maximum(s2 * inv_c - d_mean * d_mean, 0.0)
            rstd = tl.rsqrt(var + eps)

            for c_base in range(0, C, BLOCK_C):
                c_idx = c_base + c_local
                c_mask = c_idx < c_i64
                tile_mask = c_mask[:, None] & hw_mask[None, :]
                off = c_idx[:, None] * hw_i64 + hw_idx[None, :]
                x_f = tl.load(base_x + off, mask=tile_mask, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                b = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                y = (x_f - mean[None, :]) * rstd[None, :] * w[:, None] + b[:, None]
                tl.store(base_y + off, y, mask=tile_mask)

        # ──────────────────────────────────────────────────────────────────
        # 2. QuickGELU — single-pass with exact input-dtype rounding
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _quick_gelu_fwd_kernel(
            x_ptr,
            out_ptr,
            N: tl.int64,
            ROUND_CODE: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            mask = offsets < n_i64
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            x_f = x.to(tl.float32)
            scaled = _round_code(_fp32_mul_rn(tl.full([], 1.702, tl.float32), x_f), ROUND_CODE)
            if ROUND_CODE == 1:
                sig = _round_bf16(tl.sigmoid(scaled))
            else:
                one_f32 = tl.full([], 1.0, tl.float32)
                sig_raw = _fp32_div_rn(one_f32, _fp32_add_rn(one_f32, tl.extra.cuda.libdevice.exp(-scaled)))
                sig = _round_code(sig_raw, ROUND_CODE)
            tl.store(out_ptr + offsets, _fp32_mul_rn(x_f, sig), mask=mask)

        # ──────────────────────────────────────────────────────────────────
        # 3. Mish — single-pass with exact softplus/tanh rounding
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _mish_fwd_kernel(
            x_ptr,
            out_ptr,
            N: tl.int64,
            ROUND_CODE: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            mask = offsets < n_i64
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            x_f = x.to(tl.float32)
            sp_in = tl.minimum(x_f, 20.0)
            if ROUND_CODE == 1:
                # BF16: fast ex2 + tanh.approx.f32 is bit-exact after BF16 rounding.
                sp_raw = tl.where(x_f > 20.0, x_f, tl.extra.cuda.libdevice.log1p(tl.exp(sp_in)))
                sp = _round_bf16(sp_raw)
                th = _round_bf16(_tanh_approx(sp))
            else:
                sp_raw = tl.where(
                    x_f > 20.0,
                    x_f,
                    tl.extra.cuda.libdevice.log1p(tl.extra.cuda.libdevice.exp(sp_in)),
                )
                sp = _round_code(sp_raw, ROUND_CODE)
                th = _round_code(tl.extra.cuda.libdevice.tanh(sp), ROUND_CODE)
            tl.store(out_ptr + offsets, x_f * th, mask=mask)

        # ──────────────────────────────────────────────────────────────────
        # 4. SchedulerInterface — single-pass in-register FP64 conversion
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _scheduler_linear_comb_fp64_kernel(
            a_ptr,
            b_ptr,
            p1_ptr,
            p2_ptr,
            out_ptr,
            N: tl.int64,
            K: tl.int64,
            inner_stride: tl.int64,
            alphas_len: tl.int64,
            MODE: tl.constexpr,
            SCALAR_COEFF: tl.constexpr,
            HAS_INDEX: tl.constexpr,
            OUT_IS_LOW_PREC: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            """Compute FP64 linear combination or alpha_cumprod conversion in registers.

            MODE == 0: ``(a - sqrt(alpha) * b) / sqrt(1 - alpha)``  (x0 -> noise)
            MODE == 1: ``(a - sqrt(1 - alpha) * b) / sqrt(alpha)``  (noise -> x0)
            MODE == 2: ``sqrt(alpha) * a - sqrt(1 - alpha) * b``    (velocity -> x0)
            MODE == 3: ``c1 * a + c2 * b``                          (general FP64 linear comb)
            """
            pid = tl.program_id(0).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            k_i64 = z64 + K
            inner_i64 = z64 + inner_stride
            alen_i64 = z64 + alphas_len

            offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            mask = offsets < n_i64

            a_v = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
            b_v = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float64)

            if MODE == 3:
                if SCALAR_COEFF:
                    c1_v = tl.load(p1_ptr).to(tl.float64)
                    c2_v = tl.load(p2_ptr).to(tl.float64)
                else:
                    k_idx = (offsets // inner_i64) % k_i64
                    c1_v = tl.load(p1_ptr + k_idx, mask=mask, other=0.0).to(tl.float64)
                    c2_v = tl.load(p2_ptr + k_idx, mask=mask, other=0.0).to(tl.float64)
                acc = _fp64_add_rn(_fp64_mul_rn(c1_v, a_v), _fp64_mul_rn(c2_v, b_v))
            else:
                one_f64 = tl.full([], 1.0, tl.float64)
                if SCALAR_COEFF:
                    if HAS_INDEX:
                        tid = tl.load(p2_ptr).to(tl.int64)
                        tid = tl.where(tid < 0, tid + alen_i64, tid)
                        alpha = tl.load(p1_ptr + tid).to(tl.float64)
                    else:
                        alpha = tl.load(p1_ptr).to(tl.float64)
                else:
                    k_idx = (offsets // inner_i64) % k_i64
                    if HAS_INDEX:
                        tid = tl.load(p2_ptr + k_idx, mask=mask, other=0).to(tl.int64)
                        tid = tl.where(tid < 0, tid + alen_i64, tid)
                        alpha = tl.load(p1_ptr + tid, mask=mask, other=0.5).to(tl.float64)
                    else:
                        alpha = tl.load(p1_ptr + k_idx, mask=mask, other=0.5).to(tl.float64)

                beta = _fp64_add_rn(one_f64, -alpha)
                sqrt_a = _fp64_sqrt_rn(alpha)
                sqrt_b = _fp64_sqrt_rn(beta)

                if MODE == 0:
                    acc = _fp64_div_rn(_fp64_add_rn(a_v, _fp64_mul_rn(-sqrt_a, b_v)), sqrt_b)
                elif MODE == 1:
                    acc = _fp64_div_rn(_fp64_add_rn(a_v, _fp64_mul_rn(-sqrt_b, b_v)), sqrt_a)
                else:
                    acc = _fp64_add_rn(_fp64_mul_rn(sqrt_a, a_v), _fp64_mul_rn(-sqrt_b, b_v))

            if OUT_IS_LOW_PREC:
                # Match PyTorch float64 -> float32 -> bf16/fp16 cast semantics.
                tl.store(out_ptr + offsets, acc.to(tl.float32), mask=mask)
            else:
                tl.store(out_ptr + offsets, acc, mask=mask)

        # ──────────────────────────────────────────────────────────────────
        # 5. FlowMatchScheduler — fused argmin + Euler step / ret / add_noise
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _flow_match_euler_step_kernel(
            a_ptr,
            b_ptr,
            t_val_ptr,
            timesteps_ptr,
            sigmas_ptr,
            out_ptr,
            N_inner: tl.int64,
            L: tl.int64,
            timestep_scalar,
            terminal,
            TO_FINAL: tl.constexpr,
            MODE: tl.constexpr,
            SCALAR_TIMESTEP: tl.constexpr,
            HAS_T_TENSOR: tl.constexpr,
            OUT_ROUND: tl.constexpr,
            DIFF_ROUND: tl.constexpr,
            BLOCK_T: tl.constexpr,
            BLOCK: tl.constexpr,
            ITERS_PER_CTA: tl.constexpr,
        ):
            pid_x = tl.program_id(0).to(tl.int64)
            pid_b = tl.program_id(1).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_inner_i64 = z64 + N_inner
            l_i64 = z64 + L

            t_range = tl.arange(0, BLOCK_T).to(tl.int64)
            t_mask = t_range < l_i64
            ts_vec = tl.load(timesteps_ptr + t_range, mask=t_mask, other=0.0).to(tl.float32)

            if SCALAR_TIMESTEP:
                if HAS_T_TENSOR:
                    t_target = tl.load(t_val_ptr).to(tl.float32)
                else:
                    t_target = tl.full([], timestep_scalar, tl.float32)
                batch_base = z64
            else:
                t_target = tl.load(t_val_ptr + pid_b).to(tl.float32)
                batch_base = pid_b * n_inner_i64

            dists = tl.where(t_mask, tl.abs(ts_vec - t_target), float("inf"))
            best_idx = tl.argmin(dists, axis=0, tie_break_left=True).to(tl.int64)
            sigma = tl.load(sigmas_ptr + best_idx).to(tl.float32)

            if MODE == 0:
                next_idx = tl.minimum(best_idx + 1, l_i64 - 1)
                next_sigma_raw = tl.load(sigmas_ptr + next_idx).to(tl.float32)
                is_term = (best_idx + 1 >= l_i64) | TO_FINAL
                sigma_next = tl.where(is_term, terminal, next_sigma_raw)
                delta = sigma_next - sigma
                coeff = _round_code(delta, OUT_ROUND) if SCALAR_TIMESTEP else delta
            elif MODE == 1:
                coeff = _round_code(sigma, OUT_ROUND) if SCALAR_TIMESTEP else sigma
            else:
                one_minus_sigma = 1.0 - sigma
                if SCALAR_TIMESTEP:
                    c1 = _round_code(one_minus_sigma, OUT_ROUND)
                    c2 = _round_code(sigma, OUT_ROUND)
                else:
                    c1 = one_minus_sigma
                    c2 = sigma

            local = tl.arange(0, BLOCK).to(tl.int64)
            base_idx = pid_x * (BLOCK * ITERS_PER_CTA)
            for it in range(ITERS_PER_CTA):
                idx = base_idx + it * BLOCK + local
                mask = idx < n_inner_i64
                off = batch_base + idx
                a_f = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
                b_f = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
                if MODE == 0:
                    if SCALAR_TIMESTEP and OUT_ROUND != 0:
                        prod = _round_code(b_f * coeff, OUT_ROUND)
                        out_v = a_f + prod
                    else:
                        prod = _fp32_mul_rn(b_f, coeff)
                        out_v = _fp32_add_rn(a_f, prod)
                elif MODE == 1:
                    diff = _round_code(_fp32_sub_rn(a_f, b_f), DIFF_ROUND)
                    out_v = _fp32_div_rn(diff, coeff)
                else:
                    if SCALAR_TIMESTEP and OUT_ROUND != 0:
                        p1 = _round_code(c1 * a_f, OUT_ROUND)
                        p2 = _round_code(c2 * b_f, OUT_ROUND)
                        out_v = p1 + p2
                    else:
                        p1 = _fp32_mul_rn(c1, a_f)
                        p2 = _fp32_mul_rn(c2, b_f)
                        out_v = _fp32_add_rn(p1, p2)
                tl.store(out_ptr + off, out_v, mask=mask)

        # ──────────────────────────────────────────────────────────────────
        # 6. Sinusoidal Timestep Embedding — single kernel, zero spills
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _sinusoidal_timestep_embedding_kernel(
            timesteps_ptr,
            out_ptr,
            N: tl.int64,
            stride_t: tl.int64,
            embedding_dim: tl.int64,
            half_dim: tl.int64,
            neg_log_max_period: tl.float32,
            inv_freq_denom: tl.float32,
            scale: tl.float32,
            FLIP_SIN_TO_COS: tl.constexpr,
            IS_ODD: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_n = tl.program_id(0).to(tl.int64)
            pid_d = tl.program_id(1).to(tl.int64)
            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            st_i64 = z64 + stride_t
            dim_i64 = z64 + embedding_dim
            half_i64 = z64 + half_dim

            n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            d_idx = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
            n_mask = n_idx < n_i64
            d_mask = d_idx < half_i64

            t_f = tl.load(timesteps_ptr + n_idx * st_i64, mask=n_mask, other=0.0).to(tl.float32)

            z32 = tl.zeros([], dtype=tl.float32)
            neg_log_f32 = z32 + neg_log_max_period
            inv_denom_f32 = z32 + inv_freq_denom
            scale_f32 = z32 + scale
            exponent = _fp32_mul_rn(_fp32_mul_rn(neg_log_f32, d_idx.to(tl.float32)), inv_denom_f32)
            freq = tl.extra.cuda.libdevice.exp(exponent)
            arg = _fp32_mul_rn(scale_f32, _fp32_mul_rn(t_f[:, None], freq[None, :]))
            cos_v, sin_v = _cw_sincos_fp32(arg)

            if FLIP_SIN_TO_COS:
                first_v = cos_v
                second_v = sin_v
            else:
                first_v = sin_v
                second_v = cos_v

            tile_mask = n_mask[:, None] & d_mask[None, :]
            row_base = n_idx[:, None] * dim_i64
            tl.store(out_ptr + row_base + d_idx[None, :], first_v, mask=tile_mask)
            tl.store(out_ptr + row_base + half_i64 + d_idx[None, :], second_v, mask=tile_mask)

            if IS_ODD and pid_d == 0:
                zeros = tl.zeros([BLOCK_N], dtype=tl.float32)
                tl.store(out_ptr + n_idx * dim_i64 + (dim_i64 - 1), zeros, mask=n_mask)

        self.layer_norm_2d_nhwc_kernel = _layer_norm_2d_nhwc_kernel
        self.layer_norm_2d_nhwc_chunked_kernel = _layer_norm_2d_nhwc_chunked_kernel
        self.layer_norm_2d_nchw_kernel = _layer_norm_2d_nchw_kernel
        self.quick_gelu_fwd_kernel = _quick_gelu_fwd_kernel
        self.mish_fwd_kernel = _mish_fwd_kernel
        self.scheduler_linear_comb_fp64_kernel = _scheduler_linear_comb_fp64_kernel
        self.flow_match_euler_step_kernel = _flow_match_euler_step_kernel
        self.sinusoidal_timestep_embedding_kernel = _sinusoidal_timestep_embedding_kernel


# ──────────────────────────────────────────────────────────────────────────
# Kernel launch implementations + @torch.library.custom_op registrations
# ──────────────────────────────────────────────────────────────────────────


def _layer_norm_2d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out_dtype = torch.promote_types(torch.promote_types(x.dtype, weight.dtype), bias.dtype)
    out = torch.empty_like(x, dtype=out_dtype)
    if x.numel() == 0:
        return out

    b, c, h, w = x.shape
    hw = h * w
    suite = _get_triton_suite()

    if _is_channels_last_2d(x):
        n = b * hw
        if c <= 4096:
            block_c = max(16, _next_pow2(c))
            if block_c <= 256 and n >= 1024:
                rows_per_cta = 4
                num_warps = 4
            elif block_c >= 2048 and x.element_size() >= 4:
                rows_per_cta = 2 if n >= 512 else 1
                num_warps = 4
            else:
                rows_per_cta = 1
                num_warps = 2
            grid = (_cdiv(n, rows_per_cta),)
            suite.layer_norm_2d_nhwc_kernel[grid](
                x,
                weight,
                bias,
                out,
                n,
                c,
                float(eps),
                BLOCK_C=block_c,
                ROWS_PER_CTA=rows_per_cta,
                num_warps=num_warps,
            )
        else:
            suite.layer_norm_2d_nhwc_chunked_kernel[(n,)](
                x,
                weight,
                bias,
                out,
                n,
                c,
                float(eps),
                BLOCK_C=1024,
                num_warps=4,
            )
        return out

    # NCHW contiguous layout: 128-byte spatial tile (64 for 2B, 32 for 4B)
    if hw <= 64:
        block_hw = max(16, _next_pow2(hw))
        num_warps = 4
    elif x.element_size() >= 4:
        block_hw = 32
        num_warps = 4
    elif b * _cdiv(hw, 64) >= 128:
        block_hw = 64
        num_warps = 2
    else:
        block_hw = 32
        num_warps = 4
    block_c = 16 if c <= 16 else 32
    hw_blocks = _cdiv(hw, block_hw)
    suite.layer_norm_2d_nchw_kernel[(b * hw_blocks,)](
        x,
        weight,
        bias,
        out,
        c,
        hw,
        hw_blocks,
        float(eps),
        BLOCK_C=block_c,
        BLOCK_HW=block_hw,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::layer_norm_2d_fwd", mutates_args=())
def _layer_norm_2d_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _layer_norm_2d_fwd_impl(x, weight, bias, eps)


@_layer_norm_2d_fwd_op.register_fake
def _layer_norm_2d_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    out_dtype = torch.promote_types(torch.promote_types(x.dtype, weight.dtype), bias.dtype)
    return torch.empty_like(x, dtype=out_dtype)


def _quick_gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    suite = _get_triton_suite()
    block = 1024 if n < 4 * 1024 * 1024 else 2048
    num_warps = 4 if block == 1024 else 8
    suite.quick_gelu_fwd_kernel[(_cdiv(n, block),)](
        x,
        out,
        n,
        ROUND_CODE=_dtype_round_code(x.dtype),
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::quick_gelu_fwd", mutates_args=())
def _quick_gelu_fwd_op(x: torch.Tensor) -> torch.Tensor:
    return _quick_gelu_fwd_impl(x)


@_quick_gelu_fwd_op.register_fake
def _quick_gelu_fwd_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _mish_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    suite = _get_triton_suite()
    block = 1024
    num_warps = 4
    suite.mish_fwd_kernel[(_cdiv(n, block),)](
        x,
        out,
        n,
        ROUND_CODE=_dtype_round_code(x.dtype),
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::mish_fwd", mutates_args=())
def _mish_fwd_op(x: torch.Tensor) -> torch.Tensor:
    return _mish_fwd_impl(x)


@_mish_fwd_op.register_fake
def _mish_fwd_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _scheduler_linear_comb_fp64_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor | None,
    mode: int,
    k: int,
    inner_stride: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty_like(a, dtype=out_dtype)
    n = a.numel()
    if n == 0:
        return out
    suite = _get_triton_suite()
    block = 1024
    num_warps = 4
    suite.scheduler_linear_comb_fp64_kernel[(_cdiv(n, block),)](
        a,
        b,
        p1,
        p2 if p2 is not None else p1,
        out,
        n,
        int(k),
        int(inner_stride),
        int(p1.numel()),
        MODE=int(mode),
        SCALAR_COEFF=(int(k) == 1),
        HAS_INDEX=(p2 is not None and int(mode) != 3),
        OUT_IS_LOW_PREC=(out_dtype in (torch.bfloat16, torch.float16)),
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::scheduler_linear_comb_fp64", mutates_args=())
def _scheduler_linear_comb_fp64_op(
    a: torch.Tensor,
    b: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor | None,
    mode: int,
    k: int,
    inner_stride: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return _scheduler_linear_comb_fp64_impl(a, b, p1, p2, mode, k, inner_stride, out_dtype)


@_scheduler_linear_comb_fp64_op.register_fake
def _scheduler_linear_comb_fp64_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor | None,
    mode: int,
    k: int,
    inner_stride: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del b, p1, p2, mode, k, inner_stride
    return torch.empty_like(a, dtype=out_dtype)


def _flow_match_euler_step_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    timestep_val: torch.Tensor | None,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    timestep_scalar: float,
    terminal: float,
    to_final: bool,
    mode: int,
    scalar_timestep: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty_like(a, dtype=out_dtype)
    total = a.numel()
    if total == 0:
        return out

    sched_len = int(timesteps.numel())
    block_t = max(16, _next_pow2(sched_len))
    suite = _get_triton_suite()

    if scalar_timestep:
        batch = 1
        n_inner = total
    else:
        batch = int(a.shape[0])
        n_inner = total // batch

    block = 1024
    iters_per_cta = 4 if n_inner >= 256 * 1024 else 1
    num_warps = 4
    grid = (_cdiv(n_inner, block * iters_per_cta), batch)

    diff_dtype = torch.promote_types(a.dtype, b.dtype)
    suite.flow_match_euler_step_kernel[grid](
        a,
        b,
        timestep_val if timestep_val is not None else timesteps,
        timesteps,
        sigmas,
        out,
        n_inner,
        sched_len,
        float(timestep_scalar),
        float(terminal),
        TO_FINAL=bool(to_final),
        MODE=int(mode),
        SCALAR_TIMESTEP=bool(scalar_timestep),
        HAS_T_TENSOR=(timestep_val is not None),
        OUT_ROUND=_dtype_round_code(out_dtype),
        DIFF_ROUND=_dtype_round_code(diff_dtype),
        BLOCK_T=block_t,
        BLOCK=block,
        ITERS_PER_CTA=iters_per_cta,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::flow_match_euler_step", mutates_args=())
def _flow_match_euler_step_op(
    a: torch.Tensor,
    b: torch.Tensor,
    timestep_val: torch.Tensor | None,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    timestep_scalar: float,
    terminal: float,
    to_final: bool,
    mode: int,
    scalar_timestep: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return _flow_match_euler_step_impl(
        a,
        b,
        timestep_val,
        timesteps,
        sigmas,
        timestep_scalar,
        terminal,
        to_final,
        mode,
        scalar_timestep,
        out_dtype,
    )


@_flow_match_euler_step_op.register_fake
def _flow_match_euler_step_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    timestep_val: torch.Tensor | None,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    timestep_scalar: float,
    terminal: float,
    to_final: bool,
    mode: int,
    scalar_timestep: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del b, timestep_val, timesteps, sigmas, timestep_scalar, terminal, to_final, mode, scalar_timestep
    return torch.empty_like(a, dtype=out_dtype)


def _sinusoidal_timestep_embedding_fwd_impl(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
) -> torch.Tensor:
    n = int(timesteps.shape[0])
    dim = int(embedding_dim)
    out = torch.empty((n, dim), device=timesteps.device, dtype=torch.float32)
    if n == 0 or dim == 0:
        return out

    half_dim = dim // 2
    if half_dim == 0:
        out.zero_()
        return out

    neg_log_max_period = float(-math.log(max_period))
    inv_freq_denom = float(1.0 / (half_dim - downscale_freq_shift))
    suite = _get_triton_suite()

    if n == 1:
        block_n = 1
        block_d = min(256, max(16, _next_pow2(half_dim)))
        num_warps = 2 if block_d <= 128 else 4
    elif n <= 8:
        block_n = _next_pow2(n)
        block_d = min(256, max(16, _next_pow2(half_dim)))
        num_warps = 4
    else:
        block_n = 16
        block_d = min(128, max(16, _next_pow2(half_dim)))
        num_warps = 4

    grid = (_cdiv(n, block_n), _cdiv(half_dim, block_d))
    suite.sinusoidal_timestep_embedding_kernel[grid](
        timesteps,
        out,
        n,
        int(timesteps.stride(0)),
        dim,
        half_dim,
        neg_log_max_period,
        inv_freq_denom,
        float(scale),
        FLIP_SIN_TO_COS=bool(flip_sin_to_cos),
        IS_ODD=(dim % 2 == 1),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


@torch.library.custom_op("worldfoundry::sinusoidal_timestep_embedding_fwd", mutates_args=())
def _sinusoidal_timestep_embedding_fwd_op(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
) -> torch.Tensor:
    return _sinusoidal_timestep_embedding_fwd_impl(
        timesteps,
        embedding_dim,
        flip_sin_to_cos,
        downscale_freq_shift,
        scale,
        max_period,
    )


@_sinusoidal_timestep_embedding_fwd_op.register_fake
def _sinusoidal_timestep_embedding_fwd_fake(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
) -> torch.Tensor:
    del flip_sin_to_cos, downscale_freq_shift, scale, max_period
    return torch.empty((timesteps.shape[0], int(embedding_dim)), device=timesteps.device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# Public Python dispatch helpers (return None when caller must use PyTorch)
# ──────────────────────────────────────────────────────────────────────────


def triton_layer_norm_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Apply 2D channel-wise LayerNorm via Hopper Triton when eligible, else ``None``."""
    if (
        x.ndim != 4
        or weight.ndim != 1
        or bias.ndim != 1
        or weight.shape[0] != x.shape[1]
        or bias.shape[0] != x.shape[1]
        or x.shape[1] <= 0
        or x.dtype not in _FLOAT_DTYPES
        or weight.dtype not in _FLOAT_DTYPES
        or bias.dtype not in _FLOAT_DTYPES
        or not weight.is_contiguous()
        or not bias.is_contiguous()
        or not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
        or not _is_cuda_inference_ready(x, weight, bias)
    ):
        return None
    if torch.compiler.is_compiling():
        return _layer_norm_2d_fwd_op(x, weight, bias, float(eps))
    return _layer_norm_2d_fwd_impl(x, weight, bias, float(eps))


def triton_quick_gelu(x: torch.Tensor) -> torch.Tensor | None:
    """Apply CLIP QuickGELU via Hopper Triton when eligible, else ``None``."""
    if x.dtype not in _FLOAT_DTYPES or not _is_dense_supported_layout(x) or not _is_cuda_inference_ready(x):
        return None
    if torch.compiler.is_compiling():
        return _quick_gelu_fwd_op(x)
    return _quick_gelu_fwd_impl(x)


def triton_mish(x: torch.Tensor) -> torch.Tensor | None:
    """Apply Mish via Hopper Triton when eligible, else ``None``."""
    if x.dtype not in _FLOAT_DTYPES or not _is_dense_supported_layout(x) or not _is_cuda_inference_ready(x):
        return None
    if torch.compiler.is_compiling():
        return _mish_fwd_op(x)
    return _mish_fwd_impl(x)


def _validate_scheduler_slice_layout(a: torch.Tensor, b: torch.Tensor, k: int) -> int | None:
    """Validate matching dense layout for scheduler conversions; return inner_stride or None."""
    if a.ndim < 4 or not _same_dense_layout(a, b) or k <= 0:
        return None
    if k == 1:
        return 1
    if k != int(a.shape[-4]):
        return None
    if not (a.is_contiguous() or (a.ndim == 4 and a.is_contiguous(memory_format=torch.channels_last))):
        return None
    inner_stride = int(a.shape[-3] * a.shape[-2] * a.shape[-1])
    return inner_stride if inner_stride > 0 else None


def triton_scheduler_linear_comb_fp64(
    a: torch.Tensor,
    b: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Compute ``c1 * a + c2 * b`` in FP64 registers without FP64 HBM temporaries."""
    target_dtype = a.dtype if out_dtype is None else out_dtype
    if (
        a.dtype not in _FP64_INPUT_DTYPES
        or b.dtype not in _FP64_INPUT_DTYPES
        or target_dtype not in _FP64_INPUT_DTYPES
        or c1.dtype != torch.float64
        or c2.dtype != torch.float64
        or not c1.is_contiguous()
        or not c2.is_contiguous()
        or c1.ndim != 1
        or c2.ndim != 1
        or c1.numel() != c2.numel()
        or not _is_cuda_inference_ready(a, b, c1, c2)
    ):
        return None
    k = int(c2.numel())
    inner_stride = _validate_scheduler_slice_layout(a, b, k)
    if inner_stride is None:
        return None
    if torch.compiler.is_compiling():
        return _scheduler_linear_comb_fp64_op(a, b, c1, c2, 3, k, inner_stride, target_dtype)
    return _scheduler_linear_comb_fp64_impl(a, b, c1, c2, 3, k, inner_stride, target_dtype)


def triton_scheduler_convert_fp64(
    alphas_cumprod: torch.Tensor,
    first: torch.Tensor,
    xt: torch.Tensor,
    timestep: Any,
    *,
    mode: int,
) -> torch.Tensor | None:
    """Single-kernel FP64 ``SchedulerInterface.convert_*`` without FP64 HBM temporaries.

    ``mode=0``: ``convert_x0_to_noise`` ``(xt - sqrt(alpha) * x0) / sqrt(1 - alpha)``
    ``mode=1``: ``convert_noise_to_x0`` ``(xt - sqrt(1 - alpha) * noise) / sqrt(alpha)``
    ``mode=2``: ``convert_velocity_to_x0`` ``sqrt(alpha) * xt - sqrt(1 - alpha) * velocity``
    """
    if (
        mode not in (0, 1, 2)
        or first.dtype not in _FP64_INPUT_DTYPES
        or xt.dtype not in _FP64_INPUT_DTYPES
        or alphas_cumprod.ndim != 1
        or alphas_cumprod.dtype not in _FP64_INPUT_DTYPES
        or not _is_cuda_inference_ready(first, xt)
    ):
        return None

    device = first.device
    if isinstance(timestep, torch.Tensor):
        if timestep.dtype not in _INT_INDEX_DTYPES:
            return None
        if alphas_cumprod.device == device:
            alphas_dev = alphas_cumprod if alphas_cumprod.is_contiguous() else alphas_cumprod.contiguous()
            t_idx = (timestep if timestep.device == device else timestep.to(device=device)).reshape(-1).contiguous()
            k = int(t_idx.numel())
            p1, p2 = alphas_dev, t_idx
        elif timestep.device == alphas_cumprod.device:
            # Index on CPU first so only K scalars transfer to GPU.
            selected = alphas_cumprod[timestep].to(device=device, dtype=torch.float64).reshape(-1).contiguous()
            k = int(selected.numel())
            p1, p2 = selected, None
        else:
            alphas_dev = alphas_cumprod.to(device=device).contiguous()
            t_idx = timestep.to(device=device).reshape(-1).contiguous()
            k = int(t_idx.numel())
            p1, p2 = alphas_dev, t_idx
    elif isinstance(timestep, int):
        if alphas_cumprod.device == device:
            selected = alphas_cumprod[timestep : timestep + 1] if timestep >= 0 else alphas_cumprod[timestep].reshape(1)
            p1, p2, k = selected.contiguous(), None, 1
        else:
            selected = alphas_cumprod[timestep].to(device=device, dtype=torch.float64).reshape(1).contiguous()
            p1, p2, k = selected, None, 1
    else:
        return None

    inner_stride = _validate_scheduler_slice_layout(xt, first, k)
    if inner_stride is None:
        return None
    if torch.compiler.is_compiling():
        return _scheduler_linear_comb_fp64_op(xt, first, p1, p2, int(mode), k, inner_stride, first.dtype)
    return _scheduler_linear_comb_fp64_impl(xt, first, p1, p2, int(mode), k, inner_stride, first.dtype)


def _prepare_flow_match_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    timestep: Any,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor | None, float, bool] | None:
    """Validate FlowMatch tensor/schedule invariants and unpack timestep without GPU sync."""
    if (
        a.ndim < 1
        or not _same_dense_layout(a, b)
        or a.dtype not in _FLOAT_DTYPES
        or b.dtype not in _FLOAT_DTYPES
        or timesteps.ndim != 1
        or sigmas.ndim != 1
        or timesteps.shape != sigmas.shape
        or timesteps.dtype != torch.float32
        or sigmas.dtype != torch.float32
        or not timesteps.is_contiguous()
        or not sigmas.is_contiguous()
    ):
        return None
    sched_len = int(timesteps.numel())
    if sched_len <= 0 or sched_len > 1024:
        return None
    if not _is_cuda_inference_ready(a, b, timesteps, sigmas):
        return None

    device = a.device
    compiling = torch.compiler.is_compiling()

    if isinstance(timestep, torch.Tensor):
        if torch.is_grad_enabled() and timestep.requires_grad:
            return None
        scalar = timestep.ndim == 0 or timestep.numel() == 1
        if not scalar:
            if timestep.numel() != a.shape[0]:
                return None
            t_cuda = timestep.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
            return t_cuda, 0.0, False
        if timestep.device == device:
            t_cuda = (
                (timestep if timestep.dtype == torch.float32 else timestep.to(dtype=torch.float32))
                .reshape(-1)
                .contiguous()
            )
            return t_cuda, 0.0, True
        if not compiling and timestep.device.type == "cpu":
            return None, float(timestep.reshape(()).item()), True
        t_cuda = timestep.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
        return t_cuda, 0.0, True

    if isinstance(timestep, (int, float)):
        return None, float(timestep), True

    t_tensor = torch.as_tensor(timestep, device=device, dtype=torch.float32)
    scalar = t_tensor.ndim == 0 or t_tensor.numel() == 1
    if not scalar and t_tensor.numel() != a.shape[0]:
        return None
    return t_tensor.reshape(-1).contiguous(), 0.0, scalar


def triton_flow_match_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: Any,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    terminal: float,
    to_final: bool = False,
) -> torch.Tensor | None:
    """Execute one fused FlowMatch Euler step via Hopper Triton when eligible, else ``None``."""
    prepared = _prepare_flow_match_dispatch(sample, model_output, timestep, timesteps, sigmas)
    if prepared is None:
        return None
    t_val, t_scalar, scalar = prepared
    base_dtype = torch.promote_types(sample.dtype, model_output.dtype)
    out_dtype = base_dtype if scalar else torch.promote_types(base_dtype, sigmas.dtype)
    if torch.compiler.is_compiling():
        return _flow_match_euler_step_op(
            sample,
            model_output,
            t_val,
            timesteps,
            sigmas,
            t_scalar,
            float(terminal),
            bool(to_final),
            0,
            scalar,
            out_dtype,
        )
    return _flow_match_euler_step_impl(
        sample,
        model_output,
        t_val,
        timesteps,
        sigmas,
        t_scalar,
        float(terminal),
        bool(to_final),
        0,
        scalar,
        out_dtype,
    )


def triton_flow_match_return_to_timestep(
    sample: torch.Tensor,
    stabilized_sample: torch.Tensor,
    timestep: Any,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor | None:
    """Execute fused ``(sample - stabilized_sample) / sigma`` via Hopper Triton when eligible."""
    prepared = _prepare_flow_match_dispatch(sample, stabilized_sample, timestep, timesteps, sigmas)
    if prepared is None:
        return None
    t_val, t_scalar, scalar = prepared
    base_dtype = torch.promote_types(sample.dtype, stabilized_sample.dtype)
    out_dtype = base_dtype if scalar else torch.promote_types(base_dtype, sigmas.dtype)
    if torch.compiler.is_compiling():
        return _flow_match_euler_step_op(
            sample,
            stabilized_sample,
            t_val,
            timesteps,
            sigmas,
            t_scalar,
            0.0,
            False,
            1,
            scalar,
            out_dtype,
        )
    return _flow_match_euler_step_impl(
        sample,
        stabilized_sample,
        t_val,
        timesteps,
        sigmas,
        t_scalar,
        0.0,
        False,
        1,
        scalar,
        out_dtype,
    )


def triton_flow_match_add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timestep: Any,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor | None:
    """Execute fused ``((1 - sigma) * original_samples + sigma * noise).type_as(noise)``."""
    prepared = _prepare_flow_match_dispatch(original_samples, noise, timestep, timesteps, sigmas)
    if prepared is None:
        return None
    t_val, t_scalar, scalar = prepared
    if torch.compiler.is_compiling():
        return _flow_match_euler_step_op(
            original_samples,
            noise,
            t_val,
            timesteps,
            sigmas,
            t_scalar,
            0.0,
            False,
            2,
            scalar,
            noise.dtype,
        )
    return _flow_match_euler_step_impl(
        original_samples,
        noise,
        t_val,
        timesteps,
        sigmas,
        t_scalar,
        0.0,
        False,
        2,
        scalar,
        noise.dtype,
    )


def triton_sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor | None:
    """Compute DDPM sinusoidal timestep embeddings in one Hopper Triton launch."""
    dim = int(embedding_dim)
    half_dim = dim // 2
    if (
        timesteps.ndim != 1
        or timesteps.dtype not in _TIMESTEP_INPUT_DTYPES
        or dim < 0
        or max_period <= 0
        or (half_dim > 0 and (half_dim - float(downscale_freq_shift)) == 0.0)
        or not _is_cuda_inference_ready(timesteps)
    ):
        return None
    if torch.compiler.is_compiling():
        return _sinusoidal_timestep_embedding_fwd_op(
            timesteps,
            dim,
            bool(flip_sin_to_cos),
            float(downscale_freq_shift),
            float(scale),
            int(max_period),
        )
    return _sinusoidal_timestep_embedding_fwd_impl(
        timesteps,
        dim,
        bool(flip_sin_to_cos),
        float(downscale_freq_shift),
        float(scale),
        int(max_period),
    )


__all__ = [
    "core_layers_triton_ops_enabled",
    "triton_flow_match_add_noise",
    "triton_flow_match_return_to_timestep",
    "triton_flow_match_step",
    "triton_layer_norm_2d",
    "triton_mish",
    "triton_quick_gelu",
    "triton_scheduler_convert_fp64",
    "triton_scheduler_linear_comb_fp64",
    "triton_sinusoidal_timestep_embedding",
]
