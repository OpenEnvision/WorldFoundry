# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton block-fusion operators for Wan and HunyuanVideo DiTs.

All operators preserve exact IEEE BF16/FP16/FP32 reference semantics (0–1 ULP vs
eager PyTorch, zero algorithmic approximation), compile with zero local-memory
register spills (``n_spills == 0``) at wide hidden widths (``H = 3072, 5120``),
use strict ``int64`` tensor indexing, import Triton lazily so CPU-only processes
never load Triton or initialize CUDA, and register ``@torch.library.custom_op``
+ ``.register_fake`` entries for graph-break-free ``torch.compile`` and CUDA
Graph capture/replay.

Public operators
----------------
Wan2.1 / Wan2.2-5B / Wan2.2-A14B:
- :func:`fused_wan_modulation_ln_adaln`: fuses ``modulation + t_mod`` + affine-free
  LayerNorm + AdaLN ``*(1 + scale) + shift`` (both ``[B, 6, C]`` and per-token
  ``[B, L, 6, C]``, eliminating the 6×C materialization).
- :func:`fused_wan_res_gate_ln_affine`: fuses ``gate = modulation[2] + t_mod[...,2,:]``,
  ``x_out = x + y * gate``, and affine ``LayerNorm(x_out, w, b)`` in one HBM pass.
- :func:`fused_wan_res_add_ln_adaln`: fuses ``x_out = x + cross_out`` and
  ``LayerNorm(x_out) * (1 + scale2) + shift2`` (with inline ``modulation + t_mod``
  or explicit modifiers) in one HBM pass.
- :func:`fused_wan_res_gate_add`: fuses ``gate = modulation[5] + t_mod[...,5,:]``
  and ``x + y * gate`` in one HBM pass.
- :func:`fused_wan_rmsnorm`: single-pass zero-spill Hopper ``WanRMSNorm``
  (replaces multi-kernel eager RMSNorm in ``CrossAttention`` / ``SelfAttention``).
- :func:`fused_wan_head_ln_adaln`: single-pass FP32 ``Head`` LayerNorm + AdaLN
  converting directly to ``head_dtype`` with zero FP32 HBM temporaries.

HunyuanVideo-13B (double-stream & single-stream):
- :func:`fused_hunyuan_qkv_rmsnorm_rope_hopper`: 128-bit coalesced, head-tiled,
  split-Q/K/V single-stream QKV-split + per-head QK-RMSNorm + interleaved 3D RoPE
  (supports strided ``linear1`` slices directly, zero ``torch.cat``).
- :func:`fused_hunyuan_double_qkv_rmsnorm_rope_hopper`: 128-bit coalesced,
  head-tiled, split-Q/K/V double-stream direct-write into joint ``[B, L, H, D]``
  buffers (zero ``torch.cat``, zero warp divergence).
- :func:`fused_strided_gelu_tanh`: 128-bit coalesced row-wise ``GELU(tanh)``
  supporting strided ``linear1`` MLP views and contiguous FFN tensors.
- :func:`fused_dit_layernorm_adaln`: single-pass affine-free LayerNorm + AdaLN.
- :func:`fused_dit_res_gate_ln_adaln`: single-pass ``x_out = x + y * gate`` +
  ``LayerNorm(x_out) * (1 + scale) + shift``.
- :func:`fused_dit_dual_res_gate_add`: single-pass ``x + (a + b) * gate``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_TORCH_BACKENDS = frozenset({"torch", "pytorch", "native", "off", "disabled"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_MAX_ROW_FEATURES = 8192

_TRITON_SUITE: _TritonDitBlockSuite | None = None
_TRITON_SPEC_AVAILABLE: bool | None = None

# Bound lazily inside _TritonDitBlockSuite.__init__ so Triton's AST compiler
# resolves them from module globals without importing triton at module load.
tl: Any = None
_libdevice: Any = None
_round_f32_to_bf16_ptx: Any = None
_round_f32_to_fp16_ptx: Any = None
_round_code: Any = None

_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None),
    "assume_constant_result",
    lambda fn: fn,
)


@_assume_constant_result
def dit_block_ops_enabled() -> bool:
    """Return True when the Hopper Triton DiT block ops are enabled."""
    backend = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold()
    if backend in _TORCH_BACKENDS:
        return False
    raw = os.getenv("WORLDFOUNDRY_DIT_BLOCK_OPS", "1").strip().casefold()
    return raw not in _FALSY_VALUES


@_assume_constant_result
def _has_triton() -> bool:
    global _TRITON_SPEC_AVAILABLE
    if _TRITON_SPEC_AVAILABLE is None:
        _TRITON_SPEC_AVAILABLE = importlib.util.find_spec("triton") is not None
    return _TRITON_SPEC_AVAILABLE


@_assume_constant_result
def _device_supports_bf16(device_index: int) -> bool:
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _cuda_eligible(*tensors: torch.Tensor) -> bool:
    """Validate that all tensors are on the same Triton-eligible CUDA device in inference mode."""
    if not tensors or not dit_block_ops_enabled():
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
        if t.device != dev or t.dtype not in _SUPPORTED_DTYPES or t.stride(-1) != 1:
            return False
    if any(t.dtype == torch.bfloat16 for t in tensors):
        dev_idx = 0 if dev.index is None else int(dev.index)
        if not _device_supports_bf16(dev_idx):
            return False
    return True


def _storage_overlaps(first: torch.Tensor, second: torch.Tensor) -> bool:
    if torch.compiler.is_compiling():
        return True
    try:
        return bool(torch._C._overlaps(first, second))  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError):
        return True


def _dtype_round_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _row_num_warps(features: int, block: int) -> int:
    """Hopper-calibrated warp count keeping n_spills == 0 at H=3072 (4 warps) and H=5120 (8 warps)."""
    if block <= 256:
        return 4
    if features <= 1536:
        return 2
    if features <= 4096:
        return 4
    return 8


def _get_triton_suite() -> _TritonDitBlockSuite:
    global _TRITON_SUITE
    if _TRITON_SUITE is None:
        _TRITON_SUITE = _TritonDitBlockSuite()
    return _TRITON_SUITE


class _TritonDitBlockSuite:
    """Lazily-compiled Hopper SM90 Triton kernel suite for Wan & HunyuanVideo DiTs."""

    def __init__(self) -> None:
        global tl, _libdevice, _round_f32_to_bf16_ptx, _round_f32_to_fp16_ptx, _round_code
        try:
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="dit-block-triton")
        except (ImportError, OSError):
            pass

        import triton
        import triton.language as _tl
        from triton.language.extra.cuda import libdevice as _ld

        tl = _tl
        _libdevice = _ld
        self.triton = triton

        @triton.jit
        def _round_bf16(value):
            # Packed SM80+/SM90 cvt.rn.bf16x2.f32 prevents LLVM from eliding
            # round-to-BF16 boundaries while keeping 3.6+ TB/s HBM throughput.
            return tl.inline_asm_elementwise(
                "{ .reg .b32 p, lo, hi; cvt.rn.bf16x2.f32 p, $3, $2; shl.b32 lo, p, 16; and.b32 hi, p, 0xffff0000; mov.b32 $0, lo; mov.b32 $1, hi; }",
                constraints="=f,=f,f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=2,
            )

        @triton.jit
        def _round_fp16(value):
            return tl.inline_asm_elementwise(
                "{ .reg .b16 r0, r1; cvt.rn.f16.f32 r0, $2; cvt.rn.f16.f32 r1, $3; cvt.f32.f16 $0, r0; cvt.f32.f16 $1, r1; }",
                constraints="=f,=f,f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=2,
            )

        _round_f32_to_bf16_ptx = _round_bf16
        _round_f32_to_fp16_ptx = _round_fp16

        @triton.jit
        def _round_by_code(value, code: tl.constexpr):
            if code == 1:
                return _round_f32_to_bf16_ptx(value)
            elif code == 2:
                return _round_f32_to_fp16_ptx(value)
            else:
                return value

        _round_code = _round_by_code

        # ------------------------------------------------------------------
        # 1. Wan: fused (modulation + t_mod) + affine-free LayerNorm + AdaLN
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_mod_ln_adaln_kernel(
            x_ptr,
            mod_ptr,
            t_mod_ptr,
            out_ptr,
            x_stride_b,
            x_stride_s,
            mod_stride_slot,
            t_stride_b,
            t_stride_s,
            t_stride_slot,
            out_stride_b,
            out_stride_s,
            seq_len,
            eps,
            SLOT_SHIFT: tl.constexpr,
            SLOT_SCALE: tl.constexpr,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_X: tl.constexpr,
            ROUND_TMOD: tl.constexpr,
            ROUND_PROD: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_off = b * x_stride_b + s * x_stride_s + cols
            x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            x_f = tl.where(mask, x_in.to(tl.float32), 0.0)

            mean = tl.sum(x_f, axis=0) / FEATURES
            diff = tl.where(mask, x_f - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            rstd = tl.rsqrt(var + eps)
            normed = _round_code(diff * rstd, ROUND_X)

            t_row = b * t_stride_b + s * t_stride_s
            t_dtype: tl.constexpr = t_mod_ptr.dtype.element_ty
            m_sc = (
                tl.load(mod_ptr + SLOT_SCALE * mod_stride_slot + cols, mask=mask, other=0.0).to(t_dtype).to(tl.float32)
            )
            t_sc = tl.load(t_mod_ptr + t_row + SLOT_SCALE * t_stride_slot + cols, mask=mask, other=0.0).to(tl.float32)
            scale = _round_code(m_sc + t_sc, ROUND_TMOD)
            scale_factor = _round_code(1.0 + scale, ROUND_TMOD)

            m_sh = (
                tl.load(mod_ptr + SLOT_SHIFT * mod_stride_slot + cols, mask=mask, other=0.0).to(t_dtype).to(tl.float32)
            )
            t_sh = tl.load(t_mod_ptr + t_row + SLOT_SHIFT * t_stride_slot + cols, mask=mask, other=0.0).to(tl.float32)
            shift = _round_code(m_sh + t_sh, ROUND_TMOD)

            product = _round_code(normed * scale_factor, ROUND_PROD)
            out_off = b * out_stride_b + s * out_stride_s + cols
            tl.store(out_ptr + out_off, product + shift, mask=mask)

        # ------------------------------------------------------------------
        # 2. Wan: fused (mod[2] + t_mod[2]) + residual gate + affine LayerNorm
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_res_gate_ln_affine_kernel(
            x_ptr,
            y_ptr,
            mod_ptr,
            t_mod_ptr,
            weight_ptr,
            bias_ptr,
            x_out_ptr,
            norm_out_ptr,
            x_stride_b,
            x_stride_s,
            y_stride_b,
            y_stride_s,
            mod_stride_slot,
            t_stride_b,
            t_stride_s,
            t_stride_slot,
            x_out_stride_b,
            x_out_stride_s,
            norm_out_stride_b,
            norm_out_stride_s,
            seq_len,
            eps,
            SLOT_GATE: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_TMOD: tl.constexpr,
            ROUND_PROD: tl.constexpr,
            ROUND_X_OUT: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            y_f = tl.load(y_ptr + b * y_stride_b + s * y_stride_s + cols, mask=mask, other=0.0).to(tl.float32)

            t_dtype: tl.constexpr = t_mod_ptr.dtype.element_ty
            m_g = tl.load(mod_ptr + SLOT_GATE * mod_stride_slot + cols, mask=mask, other=0.0).to(t_dtype).to(tl.float32)
            t_g = tl.load(
                t_mod_ptr + b * t_stride_b + s * t_stride_s + SLOT_GATE * t_stride_slot + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            gate = _round_code(m_g + t_g, ROUND_TMOD)

            prod = _round_code(y_f * gate, ROUND_PROD)
            x_new = _round_code(x_f + prod, ROUND_X_OUT)
            tl.store(x_out_ptr + b * x_out_stride_b + s * x_out_stride_s + cols, x_new, mask=mask)

            x_m = tl.where(mask, x_new, 0.0)
            mean = tl.sum(x_m, axis=0) / FEATURES
            diff = tl.where(mask, x_m - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            normed = diff * tl.rsqrt(var + eps)
            if HAS_WEIGHT:
                w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
                normed = normed * w
            if HAS_BIAS:
                bi = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = normed + bi
            tl.store(norm_out_ptr + b * norm_out_stride_b + s * norm_out_stride_s + cols, normed, mask=mask)

        # ------------------------------------------------------------------
        # 3. Wan: fused (x + cross_out) + LN(x) * (1 + (mod[4]+t[4])) + (mod[3]+t[3])
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_res_add_ln_adaln_kernel(
            x_ptr,
            y_ptr,
            mod_ptr,
            t_mod_ptr,
            x_out_ptr,
            norm_out_ptr,
            x_stride_b,
            x_stride_s,
            y_stride_b,
            y_stride_s,
            mod_stride_slot,
            t_stride_b,
            t_stride_s,
            t_stride_slot,
            x_out_stride_b,
            x_out_stride_s,
            norm_out_stride_b,
            norm_out_stride_s,
            seq_len,
            eps,
            SLOT_SHIFT: tl.constexpr,
            SLOT_SCALE: tl.constexpr,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_X_OUT: tl.constexpr,
            ROUND_TMOD: tl.constexpr,
            ROUND_PROD: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            y_f = tl.load(y_ptr + b * y_stride_b + s * y_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            x_new = _round_code(x_f + y_f, ROUND_X_OUT)
            tl.store(x_out_ptr + b * x_out_stride_b + s * x_out_stride_s + cols, x_new, mask=mask)

            x_m = tl.where(mask, x_new, 0.0)
            mean = tl.sum(x_m, axis=0) / FEATURES
            diff = tl.where(mask, x_m - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            normed = _round_code(diff * tl.rsqrt(var + eps), ROUND_X_OUT)

            t_row = b * t_stride_b + s * t_stride_s
            t_dtype: tl.constexpr = t_mod_ptr.dtype.element_ty
            m_sc = (
                tl.load(mod_ptr + SLOT_SCALE * mod_stride_slot + cols, mask=mask, other=0.0).to(t_dtype).to(tl.float32)
            )
            t_sc = tl.load(t_mod_ptr + t_row + SLOT_SCALE * t_stride_slot + cols, mask=mask, other=0.0).to(tl.float32)
            scale = _round_code(m_sc + t_sc, ROUND_TMOD)
            scale_factor = _round_code(1.0 + scale, ROUND_TMOD)

            m_sh = (
                tl.load(mod_ptr + SLOT_SHIFT * mod_stride_slot + cols, mask=mask, other=0.0).to(t_dtype).to(tl.float32)
            )
            t_sh = tl.load(t_mod_ptr + t_row + SLOT_SHIFT * t_stride_slot + cols, mask=mask, other=0.0).to(tl.float32)
            shift = _round_code(m_sh + t_sh, ROUND_TMOD)

            product = _round_code(normed * scale_factor, ROUND_PROD)
            tl.store(norm_out_ptr + b * norm_out_stride_b + s * norm_out_stride_s + cols, product + shift, mask=mask)

        # ------------------------------------------------------------------
        # 4. Wan: fused (mod[5] + t_mod[5]) + residual gate add: x + y * gate
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_res_gate_add_kernel(
            x_ptr,
            y_ptr,
            mod_ptr,
            t_mod_ptr,
            out_ptr,
            x_stride_b,
            x_stride_s,
            y_stride_b,
            y_stride_s,
            mod_stride_slot,
            t_stride_b,
            t_stride_s,
            t_stride_slot,
            out_stride_b,
            out_stride_s,
            seq_len,
            SLOT_GATE: tl.constexpr,
            HAS_MOD: tl.constexpr,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_TMOD: tl.constexpr,
            ROUND_PROD: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            y_f = tl.load(y_ptr + b * y_stride_b + s * y_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            t_g = tl.load(
                t_mod_ptr + b * t_stride_b + s * t_stride_s + SLOT_GATE * t_stride_slot + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_MOD:
                t_dtype: tl.constexpr = t_mod_ptr.dtype.element_ty
                m_g = (
                    tl.load(mod_ptr + SLOT_GATE * mod_stride_slot + cols, mask=mask, other=0.0)
                    .to(t_dtype)
                    .to(tl.float32)
                )
                gate = _round_code(m_g + t_g, ROUND_TMOD)
            else:
                gate = t_g
            prod = _round_code(y_f * gate, ROUND_PROD)
            tl.store(out_ptr + b * out_stride_b + s * out_stride_s + cols, x_f + prod, mask=mask)

        # ------------------------------------------------------------------
        # 5. Wan: zero-spill single-pass WanRMSNorm (CrossAttention / SelfAttention)
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_rmsnorm_kernel(
            x_ptr,
            w_ptr,
            out_ptr,
            x_stride_r,
            out_stride_r,
            eps,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_IN: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES
            x_in = tl.load(x_ptr + row * x_stride_r + cols, mask=mask, other=0.0)
            x_f = tl.where(mask, x_in.to(tl.float32), 0.0)
            rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=0) / FEATURES + eps)
            w_f = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            normed = _round_code(x_f * rstd, ROUND_IN) * w_f
            tl.store(out_ptr + row * out_stride_r + cols, normed, mask=mask)

        # ------------------------------------------------------------------
        # 6. Wan: single-pass FP32 Head LayerNorm + AdaLN -> head_dtype
        # ------------------------------------------------------------------
        @triton.jit
        def _wan_head_ln_adaln_kernel(
            x_ptr,
            mod_ptr,
            t_ptr,
            out_ptr,
            x_stride_b,
            x_stride_s,
            mod_stride_slot,
            t_stride_b,
            t_stride_s,
            out_stride_b,
            out_stride_s,
            seq_len,
            eps,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.where(
                mask,
                tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32),
                0.0,
            )
            mean = tl.sum(x_f, axis=0) / FEATURES
            diff = tl.where(mask, x_f - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            normed = diff * tl.rsqrt(var + eps)

            t_f = tl.load(t_ptr + b * t_stride_b + s * t_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            m0 = tl.load(mod_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            m1 = tl.load(mod_ptr + mod_stride_slot + cols, mask=mask, other=0.0).to(tl.float32)
            out_f = normed * (1.0 + (m1 + t_f)) + (m0 + t_f)
            tl.store(out_ptr + b * out_stride_b + s * out_stride_s + cols, out_f, mask=mask)

        # ------------------------------------------------------------------
        # 7. HunyuanVideo-13B: 128-bit coalesced 3D-grid split-Q/K/V head-tiled
        #    single-stream QKV-split + per-head QK-RMSNorm + interleaved 3D RoPE
        # ------------------------------------------------------------------
        @triton.jit
        def _hy_qkv_rmsnorm_rope_hopper_kernel(
            qkv_ptr,
            qw_ptr,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_out_ptr,
            v_out_ptr,
            qkv_stride_b,
            qkv_stride_s,
            cos_stride_s,
            sin_stride_s,
            out_stride_b,
            out_stride_s,
            out_stride_h,
            seq_len,
            rope_len,
            eps,
            NUM_HEADS: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            HALF_DIM: tl.constexpr,
            BLOCK_H: tl.constexpr,
            HAS_ROPE: tl.constexpr,
        ):
            tok = tl.program_id(0).to(tl.int64)
            h_blk = tl.program_id(1).to(tl.int64)
            kind = tl.program_id(2)
            b = tok // seq_len
            s = tok % seq_len

            hidden_dim: tl.constexpr = NUM_HEADS * HEAD_DIM
            d = tl.arange(0, HEAD_DIM)
            hs = h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = (hs < NUM_HEADS)[:, None]
            in_tok = b * qkv_stride_b + s * qkv_stride_s
            out_tok = b * out_stride_b + s * out_stride_s

            if kind == 2:
                v_tile = tl.load(
                    qkv_ptr + in_tok + 2 * hidden_dim + hs[:, None] * HEAD_DIM + d[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                tl.store(
                    v_out_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                    v_tile,
                    mask=h_mask,
                )
            else:
                if kind == 0:
                    src_base = in_tok
                    dst_ptr = q_out_ptr
                    w_vec = tl.load(qw_ptr + d).to(tl.float32)[None, :]
                else:
                    src_base = in_tok + hidden_dim
                    dst_ptr = k_out_ptr
                    w_vec = tl.load(kw_ptr + d).to(tl.float32)[None, :]

                x_in = tl.load(
                    qkv_ptr + src_base + hs[:, None] * HEAD_DIM + d[None, :],
                    mask=h_mask,
                    other=0.0,
                )
                x_f = x_in.to(tl.float32)
                rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=1) / HEAD_DIM + eps)[:, None]
                x_norm = ((x_f * rstd).to(x_in.dtype).to(tl.float32) * w_vec).to(x_in.dtype)

                if HAS_ROPE:  # noqa: SIM102
                    if s < rope_len:
                        cos_v = tl.load(cos_ptr + s * cos_stride_s + d).to(tl.float32)
                        sin_v = tl.load(sin_ptr + s * sin_stride_s + d).to(tl.float32)
                        c_e, c_o = tl.split(tl.reshape(cos_v, (HALF_DIM, 2)))
                        s_e, s_o = tl.split(tl.reshape(sin_v, (HALF_DIM, 2)))
                        xe, xo = tl.split(tl.reshape(x_norm, (BLOCK_H, HALF_DIM, 2)))
                        xe_f = xe.to(tl.float32)
                        xo_f = xo.to(tl.float32)
                        re = (xe_f * c_e[None, :] - xo_f * s_e[None, :]).to(x_in.dtype)
                        ro = (xo_f * c_o[None, :] + xe_f * s_o[None, :]).to(x_in.dtype)
                        x_norm = tl.reshape(tl.join(re, ro), (BLOCK_H, HEAD_DIM))

                tl.store(
                    dst_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                    x_norm,
                    mask=h_mask,
                )

        # ------------------------------------------------------------------
        # 8. HunyuanVideo-13B: 128-bit coalesced 3D-grid split-Q/K/V head-tiled
        #    double-stream QKV -> direct joint [B, img_len+txt_len, H, D]
        # ------------------------------------------------------------------
        @triton.jit
        def _hy_double_qkv_rmsnorm_rope_hopper_kernel(
            qkv_img_ptr,
            qkv_txt_ptr,
            qw_img_ptr,
            kw_img_ptr,
            qw_txt_ptr,
            kw_txt_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_out_ptr,
            v_out_ptr,
            img_stride_b,
            img_stride_s,
            txt_stride_b,
            txt_stride_s,
            cos_stride_s,
            sin_stride_s,
            out_stride_b,
            out_stride_s,
            out_stride_h,
            img_len,
            total_len,
            eps,
            NUM_HEADS: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            HALF_DIM: tl.constexpr,
            BLOCK_H: tl.constexpr,
            HAS_ROPE: tl.constexpr,
        ):
            tok = tl.program_id(0).to(tl.int64)
            h_blk = tl.program_id(1).to(tl.int64)
            kind = tl.program_id(2)
            b = tok // total_len
            s = tok % total_len

            hidden_dim: tl.constexpr = NUM_HEADS * HEAD_DIM
            d = tl.arange(0, HEAD_DIM)
            hs = h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = (hs < NUM_HEADS)[:, None]
            out_tok = b * out_stride_b + s * out_stride_s

            if s < img_len:
                in_tok = b * img_stride_b + s * img_stride_s
                if kind == 2:
                    v_tile = tl.load(
                        qkv_img_ptr + in_tok + 2 * hidden_dim + hs[:, None] * HEAD_DIM + d[None, :],
                        mask=h_mask,
                        other=0.0,
                    )
                    tl.store(
                        v_out_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                        v_tile,
                        mask=h_mask,
                    )
                else:
                    if kind == 0:
                        src_base = in_tok
                        dst_ptr = q_out_ptr
                        w_vec = tl.load(qw_img_ptr + d).to(tl.float32)[None, :]
                    else:
                        src_base = in_tok + hidden_dim
                        dst_ptr = k_out_ptr
                        w_vec = tl.load(kw_img_ptr + d).to(tl.float32)[None, :]

                    x_in = tl.load(
                        qkv_img_ptr + src_base + hs[:, None] * HEAD_DIM + d[None, :],
                        mask=h_mask,
                        other=0.0,
                    )
                    x_f = x_in.to(tl.float32)
                    rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=1) / HEAD_DIM + eps)[:, None]
                    x_norm = ((x_f * rstd).to(x_in.dtype).to(tl.float32) * w_vec).to(x_in.dtype)

                    if HAS_ROPE:
                        cos_v = tl.load(cos_ptr + s * cos_stride_s + d).to(tl.float32)
                        sin_v = tl.load(sin_ptr + s * sin_stride_s + d).to(tl.float32)
                        c_e, c_o = tl.split(tl.reshape(cos_v, (HALF_DIM, 2)))
                        s_e, s_o = tl.split(tl.reshape(sin_v, (HALF_DIM, 2)))
                        xe, xo = tl.split(tl.reshape(x_norm, (BLOCK_H, HALF_DIM, 2)))
                        xe_f = xe.to(tl.float32)
                        xo_f = xo.to(tl.float32)
                        re = (xe_f * c_e[None, :] - xo_f * s_e[None, :]).to(x_in.dtype)
                        ro = (xo_f * c_o[None, :] + xe_f * s_o[None, :]).to(x_in.dtype)
                        x_norm = tl.reshape(tl.join(re, ro), (BLOCK_H, HEAD_DIM))

                    tl.store(
                        dst_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                        x_norm,
                        mask=h_mask,
                    )
            else:
                st = s - img_len
                in_tok = b * txt_stride_b + st * txt_stride_s
                if kind == 2:
                    v_tile = tl.load(
                        qkv_txt_ptr + in_tok + 2 * hidden_dim + hs[:, None] * HEAD_DIM + d[None, :],
                        mask=h_mask,
                        other=0.0,
                    )
                    tl.store(
                        v_out_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                        v_tile,
                        mask=h_mask,
                    )
                else:
                    if kind == 0:
                        src_base = in_tok
                        dst_ptr = q_out_ptr
                        w_vec = tl.load(qw_txt_ptr + d).to(tl.float32)[None, :]
                    else:
                        src_base = in_tok + hidden_dim
                        dst_ptr = k_out_ptr
                        w_vec = tl.load(kw_txt_ptr + d).to(tl.float32)[None, :]
                    x_in = tl.load(
                        qkv_txt_ptr + src_base + hs[:, None] * HEAD_DIM + d[None, :],
                        mask=h_mask,
                        other=0.0,
                    )
                    x_f = x_in.to(tl.float32)
                    rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=1) / HEAD_DIM + eps)[:, None]
                    x_norm = ((x_f * rstd).to(x_in.dtype).to(tl.float32) * w_vec).to(x_in.dtype)
                    tl.store(
                        dst_ptr + out_tok + hs[:, None] * out_stride_h + d[None, :],
                        x_norm,
                        mask=h_mask,
                    )

        # ------------------------------------------------------------------
        # 9. Strided 128-bit coalesced GELU(approximate="tanh")
        # ------------------------------------------------------------------
        @triton.jit
        def _strided_gelu_tanh_kernel(
            in_ptr,
            out_ptr,
            in_stride_r,
            out_stride_r,
            n_cols,
            BLOCK_N: tl.constexpr,
            FAST_TANH: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            in_row = row * in_stride_r
            out_row = row * out_stride_r
            for c0 in range(0, n_cols, BLOCK_N):
                cols = c0 + tl.arange(0, BLOCK_N)
                mask = cols < n_cols
                x_in = tl.load(in_ptr + in_row + cols, mask=mask, other=0.0)
                x = x_in.to(tl.float32)
                inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
                if FAST_TANH:
                    th = tl.inline_asm_elementwise(
                        "tanh.approx.f32 $0, $1;",
                        "=f,f",
                        [inner],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    th = _libdevice.tanh(inner)
                out = 0.5 * x * (1.0 + th)
                tl.store(out_ptr + out_row + cols, out.to(x_in.dtype), mask=mask)

        # ------------------------------------------------------------------
        # 10. Generic DiT affine-free LN + AdaLN (HunyuanVideo & Wan modifiers)
        # ------------------------------------------------------------------
        @triton.jit
        def _dit_ln_adaln_kernel(
            x_ptr,
            scale_ptr,
            shift_ptr,
            out_ptr,
            x_stride_b,
            x_stride_s,
            sc_stride_b,
            sc_stride_s,
            sh_stride_b,
            sh_stride_s,
            out_stride_b,
            out_stride_s,
            seq_len,
            eps,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_X: tl.constexpr,
            ROUND_SCALE: tl.constexpr,
            ROUND_PROD: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_in = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0)
            x_f = tl.where(mask, x_in.to(tl.float32), 0.0)
            mean = tl.sum(x_f, axis=0) / FEATURES
            diff = tl.where(mask, x_f - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            normed = _round_code(diff * tl.rsqrt(var + eps), ROUND_X)

            sc_f = tl.load(scale_ptr + b * sc_stride_b + s * sc_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            sh_f = tl.load(shift_ptr + b * sh_stride_b + s * sh_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            scale_factor = _round_code(1.0 + sc_f, ROUND_SCALE)
            product = _round_code(normed * scale_factor, ROUND_PROD)
            tl.store(out_ptr + b * out_stride_b + s * out_stride_s + cols, product + sh_f, mask=mask)

        # ------------------------------------------------------------------
        # 11. Generic DiT residual-gate + LN + AdaLN (HunyuanVideo double block)
        # ------------------------------------------------------------------
        @triton.jit
        def _dit_res_gate_ln_adaln_kernel(
            x_ptr,
            upd_ptr,
            gate_ptr,
            scale_ptr,
            shift_ptr,
            x_out_ptr,
            norm_out_ptr,
            x_stride_b,
            x_stride_s,
            upd_stride_b,
            upd_stride_s,
            gate_stride_b,
            scale_stride_b,
            shift_stride_b,
            x_out_stride_b,
            x_out_stride_s,
            norm_out_stride_b,
            norm_out_stride_s,
            seq_len,
            eps,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
            ROUND_X: tl.constexpr,
            ROUND_PROD: tl.constexpr,
            ROUND_SCALE: tl.constexpr,
            ROUND_MOD_PROD: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            b = row // seq_len
            s = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            upd_f = tl.load(upd_ptr + b * upd_stride_b + s * upd_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            gate_f = tl.load(gate_ptr + b * gate_stride_b + cols, mask=mask, other=0.0).to(tl.float32)

            x_new = _round_code(x_f + _round_code(upd_f * gate_f, ROUND_PROD), ROUND_X)
            tl.store(x_out_ptr + b * x_out_stride_b + s * x_out_stride_s + cols, x_new, mask=mask)

            x_m = tl.where(mask, x_new, 0.0)
            mean = tl.sum(x_m, axis=0) / FEATURES
            diff = tl.where(mask, x_m - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / FEATURES
            normed = _round_code(diff * tl.rsqrt(var + eps), ROUND_X)

            sc = tl.load(scale_ptr + b * scale_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + b * shift_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
            scale_factor = _round_code(1.0 + sc, ROUND_SCALE)
            mod_out = _round_code(normed * scale_factor, ROUND_MOD_PROD) + sh
            tl.store(norm_out_ptr + b * norm_out_stride_b + s * norm_out_stride_s + cols, mod_out, mask=mask)

        # ------------------------------------------------------------------
        # 12. Generic DiT dual residual-gate add (HunyuanVideo single block)
        # ------------------------------------------------------------------
        @triton.jit
        def _dit_dual_res_gate_add_kernel(
            x_ptr,
            a_ptr,
            b_ptr,
            gate_ptr,
            out_ptr,
            x_stride_b,
            x_stride_s,
            a_stride_b,
            a_stride_s,
            b_stride_b,
            b_stride_s,
            gate_stride_b,
            out_stride_b,
            out_stride_s,
            seq_len,
            FEATURES: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0).to(tl.int64)
            batch = row // seq_len
            seq = row % seq_len
            cols = tl.arange(0, BLOCK)
            mask = cols < FEATURES

            x_f = tl.load(x_ptr + batch * x_stride_b + seq * x_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            a_f = tl.load(a_ptr + batch * a_stride_b + seq * a_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            b_f = tl.load(b_ptr + batch * b_stride_b + seq * b_stride_s + cols, mask=mask, other=0.0).to(tl.float32)
            g_f = tl.load(gate_ptr + batch * gate_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + batch * out_stride_b + seq * out_stride_s + cols, x_f + (a_f + b_f) * g_f, mask=mask)

        self.wan_mod_ln_adaln = _wan_mod_ln_adaln_kernel
        self.wan_res_gate_ln_affine = _wan_res_gate_ln_affine_kernel
        self.wan_res_add_ln_adaln = _wan_res_add_ln_adaln_kernel
        self.wan_res_gate_add = _wan_res_gate_add_kernel
        self.wan_rmsnorm = _wan_rmsnorm_kernel
        self.wan_head_ln_adaln = _wan_head_ln_adaln_kernel
        self.hy_qkv_rmsnorm_rope = _hy_qkv_rmsnorm_rope_hopper_kernel
        self.hy_double_qkv_rmsnorm_rope = _hy_double_qkv_rmsnorm_rope_hopper_kernel
        self.strided_gelu_tanh = _strided_gelu_tanh_kernel
        self.dit_ln_adaln = _dit_ln_adaln_kernel
        self.dit_res_gate_ln_adaln = _dit_res_gate_ln_adaln_kernel
        self.dit_dual_res_gate_add = _dit_dual_res_gate_add_kernel


def _t_mod_strides(t_mod: torch.Tensor, batch: int) -> tuple[int, int, int]:
    """Return ``(stride_b, stride_s, stride_slot)`` for 3D ``[B, 6, C]`` or 4D ``[B, L, 6, C]``."""
    if t_mod.ndim == 4:
        sb = 0 if t_mod.shape[0] == 1 and batch > 1 else int(t_mod.stride(0))
        return sb, int(t_mod.stride(1)), int(t_mod.stride(2))
    sb = 0 if t_mod.shape[0] == 1 and batch > 1 else int(t_mod.stride(0))
    return sb, 0, int(t_mod.stride(1))


def _mod_slot_tensor_valid(
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    batch: int,
    seq_len: int,
    features: int,
) -> bool:
    if modulation.ndim != 3 or modulation.shape != (1, 6, features):
        return False
    if t_mod.ndim == 3:
        return t_mod.shape[0] in (1, batch) and t_mod.shape[1] == 6 and t_mod.shape[2] == features
    if t_mod.ndim == 4:
        return (
            t_mod.shape[0] in (1, batch)
            and t_mod.shape[1] == seq_len
            and t_mod.shape[2] == 6
            and t_mod.shape[3] == features
        )
    return False


# ==========================================================================
# Registered torch.library.custom_op CUDA Entry Points + Fake Kernels
# ==========================================================================


@torch.library.custom_op("worldfoundry::wan_mod_ln_adaln", mutates_args=(), device_types="cuda")
def _wan_mod_ln_adaln_op(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> torch.Tensor:
    suite = _get_triton_suite()
    b, s, features = x.shape
    product_dtype = torch.promote_types(x.dtype, t_mod.dtype)
    out = torch.empty_like(x, dtype=product_dtype)
    if b == 0 or s == 0:
        return out
    block = max(2, suite.triton.next_power_of_2(features))
    t_sb, t_ss, t_ssl = _t_mod_strides(t_mod, b)
    with torch.cuda.device(x.device):
        suite.wan_mod_ln_adaln[(b * s,)](
            x,
            modulation,
            t_mod,
            out,
            x.stride(0),
            x.stride(1),
            modulation.stride(1),
            t_sb,
            t_ss,
            t_ssl,
            out.stride(0),
            out.stride(1),
            s,
            float(eps),
            SLOT_SHIFT=slot_shift,
            SLOT_SCALE=slot_scale,
            FEATURES=features,
            BLOCK=block,
            ROUND_X=_dtype_round_code(x.dtype),
            ROUND_TMOD=_dtype_round_code(t_mod.dtype),
            ROUND_PROD=_dtype_round_code(product_dtype),
            num_warps=_row_num_warps(features, block),
        )
    return out


@_wan_mod_ln_adaln_op.register_fake
def _wan_mod_ln_adaln_fake(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> torch.Tensor:
    del modulation, eps, slot_shift, slot_scale
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, t_mod.dtype))


def _launch_wan_res_gate_ln_affine(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    x_out: torch.Tensor,
    norm_out: torch.Tensor,
    eps: float,
    slot_gate: int,
) -> None:
    suite = _get_triton_suite()
    b, s, features = x.shape
    if b == 0 or s == 0:
        return
    block = max(2, suite.triton.next_power_of_2(features))
    t_sb, t_ss, t_ssl = _t_mod_strides(t_mod, b)
    prod_dtype = torch.promote_types(y.dtype, t_mod.dtype)
    with torch.cuda.device(x.device):
        suite.wan_res_gate_ln_affine[(b * s,)](
            x,
            y,
            modulation,
            t_mod,
            x if weight is None else weight,
            x if bias is None else bias,
            x_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            modulation.stride(1),
            t_sb,
            t_ss,
            t_ssl,
            x_out.stride(0),
            x_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            s,
            float(eps),
            SLOT_GATE=slot_gate,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            FEATURES=features,
            BLOCK=block,
            ROUND_TMOD=_dtype_round_code(t_mod.dtype),
            ROUND_PROD=_dtype_round_code(prod_dtype),
            ROUND_X_OUT=_dtype_round_code(x_out.dtype),
            num_warps=_row_num_warps(features, block),
        )


@torch.library.custom_op("worldfoundry::wan_res_gate_ln_affine", mutates_args=(), device_types="cuda")
def _wan_res_gate_ln_affine_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    slot_gate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prod_dtype = torch.promote_types(y.dtype, t_mod.dtype)
    x_out_dtype = torch.promote_types(x.dtype, prod_dtype)
    x_out = torch.empty_like(x, dtype=x_out_dtype)
    norm_out = torch.empty_like(x, dtype=x_out_dtype)
    _launch_wan_res_gate_ln_affine(x, y, modulation, t_mod, weight, bias, x_out, norm_out, eps, slot_gate)
    return x_out, norm_out


@_wan_res_gate_ln_affine_op.register_fake
def _wan_res_gate_ln_affine_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    slot_gate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del modulation, weight, bias, eps, slot_gate
    x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, t_mod.dtype))
    return torch.empty_like(x, dtype=x_out_dtype), torch.empty_like(x, dtype=x_out_dtype)


@torch.library.custom_op("worldfoundry::wan_res_gate_ln_affine_inplace", mutates_args=("x",), device_types="cuda")
def _wan_res_gate_ln_affine_inplace_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    slot_gate: int,
) -> torch.Tensor:
    norm_out = torch.empty_like(x)
    _launch_wan_res_gate_ln_affine(x, y, modulation, t_mod, weight, bias, x, norm_out, eps, slot_gate)
    return norm_out


@_wan_res_gate_ln_affine_inplace_op.register_fake
def _wan_res_gate_ln_affine_inplace_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    slot_gate: int,
) -> torch.Tensor:
    del y, modulation, t_mod, weight, bias, eps, slot_gate
    return torch.empty_like(x)


def _launch_wan_res_add_ln_adaln(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    x_out: torch.Tensor,
    norm_out: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> None:
    suite = _get_triton_suite()
    b, s, features = x.shape
    if b == 0 or s == 0:
        return
    block = max(2, suite.triton.next_power_of_2(features))
    t_sb, t_ss, t_ssl = _t_mod_strides(t_mod, b)
    prod_dtype = torch.promote_types(x_out.dtype, t_mod.dtype)
    with torch.cuda.device(x.device):
        suite.wan_res_add_ln_adaln[(b * s,)](
            x,
            y,
            modulation,
            t_mod,
            x_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            modulation.stride(1),
            t_sb,
            t_ss,
            t_ssl,
            x_out.stride(0),
            x_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            s,
            float(eps),
            SLOT_SHIFT=slot_shift,
            SLOT_SCALE=slot_scale,
            FEATURES=features,
            BLOCK=block,
            ROUND_X_OUT=_dtype_round_code(x_out.dtype),
            ROUND_TMOD=_dtype_round_code(t_mod.dtype),
            ROUND_PROD=_dtype_round_code(prod_dtype),
            num_warps=_row_num_warps(features, block),
        )


@torch.library.custom_op("worldfoundry::wan_res_add_ln_adaln", mutates_args=(), device_types="cuda")
def _wan_res_add_ln_adaln_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_out_dtype = torch.promote_types(x.dtype, y.dtype)
    norm_out_dtype = torch.promote_types(x_out_dtype, t_mod.dtype)
    x_out = torch.empty_like(x, dtype=x_out_dtype)
    norm_out = torch.empty_like(x, dtype=norm_out_dtype)
    _launch_wan_res_add_ln_adaln(x, y, modulation, t_mod, x_out, norm_out, eps, slot_shift, slot_scale)
    return x_out, norm_out


@_wan_res_add_ln_adaln_op.register_fake
def _wan_res_add_ln_adaln_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del modulation, eps, slot_shift, slot_scale
    x_out_dtype = torch.promote_types(x.dtype, y.dtype)
    norm_out_dtype = torch.promote_types(x_out_dtype, t_mod.dtype)
    return torch.empty_like(x, dtype=x_out_dtype), torch.empty_like(x, dtype=norm_out_dtype)


@torch.library.custom_op("worldfoundry::wan_res_add_ln_adaln_inplace", mutates_args=("x",), device_types="cuda")
def _wan_res_add_ln_adaln_inplace_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> torch.Tensor:
    norm_out_dtype = torch.promote_types(x.dtype, t_mod.dtype)
    norm_out = torch.empty_like(x, dtype=norm_out_dtype)
    _launch_wan_res_add_ln_adaln(x, y, modulation, t_mod, x, norm_out, eps, slot_shift, slot_scale)
    return norm_out


@_wan_res_add_ln_adaln_inplace_op.register_fake
def _wan_res_add_ln_adaln_inplace_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    slot_shift: int,
    slot_scale: int,
) -> torch.Tensor:
    del y, modulation, eps, slot_shift, slot_scale
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, t_mod.dtype))


def _launch_wan_res_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    out: torch.Tensor,
    slot_gate: int,
) -> None:
    suite = _get_triton_suite()
    b, s, features = x.shape
    if b == 0 or s == 0:
        return
    block = max(2, suite.triton.next_power_of_2(features))
    if modulation is not None:
        t_sb, t_ss, t_ssl = _t_mod_strides(t_mod, b)
        mod_ptr = modulation
        mod_stride_slot = modulation.stride(1)
        has_mod = True
    else:
        if t_mod.ndim == 3:
            t_sb = 0 if t_mod.shape[0] == 1 and b > 1 else int(t_mod.stride(0))
            t_ss = 0 if t_mod.shape[1] == 1 and s > 1 else int(t_mod.stride(1))
        else:
            t_sb = 0 if t_mod.shape[0] == 1 and b > 1 else int(t_mod.stride(0))
            t_ss = 0
        t_ssl = 0
        mod_ptr = t_mod
        mod_stride_slot = 0
        has_mod = False
    prod_dtype = torch.promote_types(y.dtype, t_mod.dtype)
    with torch.cuda.device(x.device):
        suite.wan_res_gate_add[(b * s,)](
            x,
            y,
            mod_ptr,
            t_mod,
            out,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            mod_stride_slot,
            t_sb,
            t_ss,
            t_ssl,
            out.stride(0),
            out.stride(1),
            s,
            SLOT_GATE=slot_gate if has_mod else 0,
            HAS_MOD=has_mod,
            FEATURES=features,
            BLOCK=block,
            ROUND_TMOD=_dtype_round_code(t_mod.dtype),
            ROUND_PROD=_dtype_round_code(prod_dtype),
            num_warps=_row_num_warps(features, block),
        )


@torch.library.custom_op("worldfoundry::wan_res_gate_add", mutates_args=(), device_types="cuda")
def _wan_res_gate_add_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    slot_gate: int,
) -> torch.Tensor:
    out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, t_mod.dtype))
    out = torch.empty_like(x, dtype=out_dtype)
    _launch_wan_res_gate_add(x, y, modulation, t_mod, out, slot_gate)
    return out


@_wan_res_gate_add_op.register_fake
def _wan_res_gate_add_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    slot_gate: int,
) -> torch.Tensor:
    del modulation, slot_gate
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, torch.promote_types(y.dtype, t_mod.dtype)))


@torch.library.custom_op("worldfoundry::wan_res_gate_add_inplace", mutates_args=("x",), device_types="cuda")
def _wan_res_gate_add_inplace_op(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    slot_gate: int,
) -> None:
    _launch_wan_res_gate_add(x, y, modulation, t_mod, x, slot_gate)


@_wan_res_gate_add_inplace_op.register_fake
def _wan_res_gate_add_inplace_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    slot_gate: int,
) -> None:
    del x, y, modulation, t_mod, slot_gate


@torch.library.custom_op("worldfoundry::wan_rmsnorm", mutates_args=(), device_types="cuda")
def _wan_rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    suite = _get_triton_suite()
    features = int(x.shape[-1])
    mat = x.reshape(-1, features)
    rows = int(mat.shape[0])
    out_dtype = torch.promote_types(x.dtype, weight.dtype)
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if rows == 0 or features == 0:
        return out
    out_2d = out.view(rows, features)
    block = max(2, suite.triton.next_power_of_2(features))
    with torch.cuda.device(x.device):
        suite.wan_rmsnorm[(rows,)](
            mat,
            weight,
            out_2d,
            mat.stride(0),
            out_2d.stride(0),
            float(eps),
            FEATURES=features,
            BLOCK=block,
            ROUND_IN=_dtype_round_code(x.dtype),
            num_warps=_row_num_warps(features, block),
        )
    return out


@_wan_rmsnorm_op.register_fake
def _wan_rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    del eps
    return torch.empty(x.shape, dtype=torch.promote_types(x.dtype, weight.dtype), device=x.device)


@torch.library.custom_op("worldfoundry::wan_head_ln_adaln", mutates_args=(), device_types="cuda")
def _wan_head_ln_adaln_op(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    suite = _get_triton_suite()
    b, s, features = x.shape
    out = torch.empty((b, s, features), dtype=out_dtype, device=x.device)
    if b == 0 or s == 0:
        return out
    block = max(2, suite.triton.next_power_of_2(features))
    if t_mod.ndim == 3:
        t_sb = 0 if t_mod.shape[0] == 1 and b > 1 else int(t_mod.stride(0))
        t_ss = int(t_mod.stride(1))
    else:
        t_sb = 0 if t_mod.shape[0] == 1 and b > 1 else int(t_mod.stride(0))
        t_ss = 0
    with torch.cuda.device(x.device):
        suite.wan_head_ln_adaln[(b * s,)](
            x,
            modulation,
            t_mod,
            out,
            x.stride(0),
            x.stride(1),
            modulation.stride(1),
            t_sb,
            t_ss,
            out.stride(0),
            out.stride(1),
            s,
            float(eps),
            FEATURES=features,
            BLOCK=block,
            num_warps=_row_num_warps(features, block),
        )
    return out


@_wan_head_ln_adaln_op.register_fake
def _wan_head_ln_adaln_fake(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del modulation, t_mod, eps
    return torch.empty(x.shape, dtype=out_dtype, device=x.device)


@torch.library.custom_op("worldfoundry::hy_qkv_rmsnorm_rope_hopper", mutates_args=(), device_types="cuda")
def _hy_qkv_rmsnorm_rope_hopper_op(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    rope_len: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    suite = _get_triton_suite()
    b, s, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    q_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if b == 0 or s == 0:
        return q_out, k_out, v_out
    max_bh = 8 if head_dim <= 128 else 4
    block_h = min(max_bh, suite.triton.next_power_of_2(num_heads))
    grid = (b * s, suite.triton.cdiv(num_heads, block_h), 3)
    with torch.cuda.device(qkv.device):
        suite.hy_qkv_rmsnorm_rope[grid](
            qkv,
            q_weight,
            k_weight,
            cos,
            sin,
            q_out,
            k_out,
            v_out,
            qkv.stride(0),
            qkv.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            s,
            rope_len,
            float(eps),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=head_dim // 2,
            BLOCK_H=block_h,
            HAS_ROPE=has_rope,
            num_warps=2,
        )
    return q_out, k_out, v_out


@_hy_qkv_rmsnorm_rope_hopper_op.register_fake
def _hy_qkv_rmsnorm_rope_hopper_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    rope_len: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, rope_len, has_rope, eps
    b, s, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    shape = (b, s, num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
    )


@torch.library.custom_op("worldfoundry::hy_double_qkv_rmsnorm_rope_hopper", mutates_args=(), device_types="cuda")
def _hy_double_qkv_rmsnorm_rope_hopper_op(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    suite = _get_triton_suite()
    b, img_len, three_c = qkv_img.shape
    total_len = img_len + qkv_txt.shape[1]
    head_dim = three_c // (3 * num_heads)
    q_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    k_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    v_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    if b == 0 or total_len == 0:
        return q_out, k_out, v_out
    max_bh = 8 if head_dim <= 128 else 4
    block_h = min(max_bh, suite.triton.next_power_of_2(num_heads))
    grid = (b * total_len, suite.triton.cdiv(num_heads, block_h), 3)
    with torch.cuda.device(qkv_img.device):
        suite.hy_double_qkv_rmsnorm_rope[grid](
            qkv_img,
            qkv_txt,
            qw_img,
            kw_img,
            qw_txt,
            kw_txt,
            cos,
            sin,
            q_out,
            k_out,
            v_out,
            qkv_img.stride(0),
            qkv_img.stride(1),
            qkv_txt.stride(0),
            qkv_txt.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            img_len,
            total_len,
            float(eps),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=head_dim // 2,
            BLOCK_H=block_h,
            HAS_ROPE=has_rope,
            num_warps=2,
        )
    return q_out, k_out, v_out


@_hy_double_qkv_rmsnorm_rope_hopper_op.register_fake
def _hy_double_qkv_rmsnorm_rope_hopper_fake(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del qw_img, kw_img, qw_txt, kw_txt, cos, sin, has_rope, eps
    b, img_len, three_c = qkv_img.shape
    total_len = img_len + qkv_txt.shape[1]
    head_dim = three_c // (3 * num_heads)
    shape = (b, total_len, num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
    )


@torch.library.custom_op("worldfoundry::strided_gelu_tanh", mutates_args=(), device_types="cuda")
def _strided_gelu_tanh_op(mat: torch.Tensor) -> torch.Tensor:
    suite = _get_triton_suite()
    rows, cols = mat.shape
    out = torch.empty((rows, cols), dtype=mat.dtype, device=mat.device)
    if rows == 0 or cols == 0:
        return out
    block_n = 2048 if cols >= 2048 else max(256, suite.triton.next_power_of_2(cols))
    num_warps = 8 if block_n >= 2048 else 4
    fast_tanh = mat.dtype in (torch.bfloat16, torch.float16)
    with torch.cuda.device(mat.device):
        suite.strided_gelu_tanh[(rows,)](
            mat,
            out,
            mat.stride(0),
            out.stride(0),
            cols,
            BLOCK_N=block_n,
            FAST_TANH=fast_tanh,
            num_warps=num_warps,
        )
    return out


@_strided_gelu_tanh_op.register_fake
def _strided_gelu_tanh_fake(mat: torch.Tensor) -> torch.Tensor:
    return torch.empty(mat.shape, dtype=mat.dtype, device=mat.device)


@torch.library.custom_op("worldfoundry::dit_ln_adaln", mutates_args=(), device_types="cuda")
def _dit_ln_adaln_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    suite = _get_triton_suite()
    b, s, features = x.shape
    prod_dtype = torch.promote_types(x.dtype, scale.dtype)
    out_dtype = torch.promote_types(prod_dtype, shift.dtype)
    out = torch.empty_like(x, dtype=out_dtype)
    if b == 0 or s == 0:
        return out
    block = max(2, suite.triton.next_power_of_2(features))
    sc_sb = 0 if scale.shape[0] == 1 and b > 1 else int(scale.stride(0))
    sc_ss = 0 if (scale.ndim == 2 or scale.shape[1] == 1) else int(scale.stride(1))
    sh_sb = 0 if shift.shape[0] == 1 and b > 1 else int(shift.stride(0))
    sh_ss = 0 if (shift.ndim == 2 or shift.shape[1] == 1) else int(shift.stride(1))
    with torch.cuda.device(x.device):
        suite.dit_ln_adaln[(b * s,)](
            x,
            scale,
            shift,
            out,
            x.stride(0),
            x.stride(1),
            sc_sb,
            sc_ss,
            sh_sb,
            sh_ss,
            out.stride(0),
            out.stride(1),
            s,
            float(eps),
            FEATURES=features,
            BLOCK=block,
            ROUND_X=_dtype_round_code(x.dtype),
            ROUND_SCALE=_dtype_round_code(scale.dtype),
            ROUND_PROD=_dtype_round_code(prod_dtype),
            num_warps=_row_num_warps(features, block),
        )
    return out


@_dit_ln_adaln_op.register_fake
def _dit_ln_adaln_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    return torch.empty_like(x, dtype=torch.promote_types(torch.promote_types(x.dtype, scale.dtype), shift.dtype))


@torch.library.custom_op("worldfoundry::dit_res_gate_ln_adaln", mutates_args=(), device_types="cuda")
def _dit_res_gate_ln_adaln_op(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    suite = _get_triton_suite()
    b, s, features = x.shape
    x_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)
    if b == 0 or s == 0:
        return x_out, norm_out
    block = max(2, suite.triton.next_power_of_2(features))
    g_sb = 0 if gate.shape[0] == 1 and b > 1 else int(gate.stride(0))
    sc_sb = 0 if scale.shape[0] == 1 and b > 1 else int(scale.stride(0))
    sh_sb = 0 if shift.shape[0] == 1 and b > 1 else int(shift.stride(0))
    with torch.cuda.device(x.device):
        suite.dit_res_gate_ln_adaln[(b * s,)](
            x,
            update,
            gate,
            scale,
            shift,
            x_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            update.stride(0),
            update.stride(1),
            g_sb,
            sc_sb,
            sh_sb,
            x_out.stride(0),
            x_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            s,
            float(eps),
            FEATURES=features,
            BLOCK=block,
            ROUND_X=_dtype_round_code(x.dtype),
            ROUND_PROD=_dtype_round_code(torch.promote_types(update.dtype, gate.dtype)),
            ROUND_SCALE=_dtype_round_code(scale.dtype),
            ROUND_MOD_PROD=_dtype_round_code(torch.promote_types(x.dtype, scale.dtype)),
            num_warps=_row_num_warps(features, block),
        )
    return x_out, norm_out


@_dit_res_gate_ln_adaln_op.register_fake
def _dit_res_gate_ln_adaln_fake(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del update, gate, scale, shift, eps
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("worldfoundry::dit_dual_res_gate_add", mutates_args=(), device_types="cuda")
def _dit_dual_res_gate_add_op(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    suite = _get_triton_suite()
    batch, seq_len, features = x.shape
    out = torch.empty_like(x)
    if batch == 0 or seq_len == 0:
        return out
    block = max(2, suite.triton.next_power_of_2(features))
    g_sb = 0 if gate.shape[0] == 1 and batch > 1 else int(gate.stride(0))
    with torch.cuda.device(x.device):
        suite.dit_dual_res_gate_add[(batch * seq_len,)](
            x,
            a,
            b_term,
            gate,
            out,
            x.stride(0),
            x.stride(1),
            a.stride(0),
            a.stride(1),
            b_term.stride(0),
            b_term.stride(1),
            g_sb,
            out.stride(0),
            out.stride(1),
            seq_len,
            FEATURES=features,
            BLOCK=block,
            num_warps=_row_num_warps(features, block),
        )
    return out


@_dit_dual_res_gate_add_op.register_fake
def _dit_dual_res_gate_add_fake(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    del a, b_term, gate
    return torch.empty_like(x)


# ==========================================================================
# Public High-Level Wrappers with Exact Portable PyTorch Fallbacks
# ==========================================================================


def _wan_slot(modulation: torch.Tensor, t_mod: torch.Tensor, slot: int) -> torch.Tensor:
    mod_s = modulation[:, slot : slot + 1, :].to(dtype=t_mod.dtype, device=t_mod.device)
    if t_mod.ndim == 4:
        return mod_s + t_mod[:, :, slot, :]
    return mod_s + t_mod[:, slot : slot + 1, :]


def fused_wan_modulation_ln_adaln(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    *,
    eps: float = 1e-6,
    slot_shift: int = 0,
    slot_scale: int = 1,
) -> torch.Tensor:
    """Fuse ``(modulation + t_mod)`` + affine-free ``LayerNorm(x, eps) * (1 + scale) + shift``."""
    if x.ndim == 3 and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        if _mod_slot_tensor_valid(modulation, t_mod, b, s, features) and _cuda_eligible(x, modulation, t_mod):
            return _wan_mod_ln_adaln_op(x, modulation, t_mod, float(eps), int(slot_shift), int(slot_scale))
    shift = _wan_slot(modulation, t_mod, slot_shift)
    scale = _wan_slot(modulation, t_mod, slot_scale)
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps) * (1 + scale) + shift


def fused_wan_res_gate_ln_affine(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    eps: float = 1e-6,
    slot_gate: int = 2,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``x_out = x + y * (mod[2] + t_mod[2])`` and ``LayerNorm(x_out, weight, bias, eps)``."""
    if x.ndim == 3 and y.shape == x.shape and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, t_mod.dtype))
        w_ok = weight is None or (
            weight.shape == (features,) and weight.dtype == x_out_dtype and weight.is_contiguous()
        )
        b_ok = bias is None or (bias.shape == (features,) and bias.dtype == x_out_dtype and bias.is_contiguous())
        tensors = [x, y, modulation, t_mod]
        if weight is not None:
            tensors.append(weight)
        if bias is not None:
            tensors.append(bias)
        if w_ok and b_ok and _mod_slot_tensor_valid(modulation, t_mod, b, s, features) and _cuda_eligible(*tensors):
            can_inplace = (
                inplace
                and not torch.is_grad_enabled()
                and x_out_dtype == x.dtype
                and not _storage_overlaps(x, y)
                and not _storage_overlaps(x, modulation)
                and not _storage_overlaps(x, t_mod)
                and (weight is None or not _storage_overlaps(x, weight))
                and (bias is None or not _storage_overlaps(x, bias))
            )
            if can_inplace:
                norm_out = _wan_res_gate_ln_affine_inplace_op(
                    x, y, modulation, t_mod, weight, bias, float(eps), int(slot_gate)
                )
                return x, norm_out
            return _wan_res_gate_ln_affine_op(x, y, modulation, t_mod, weight, bias, float(eps), int(slot_gate))

    gate = _wan_slot(modulation, t_mod, slot_gate)
    x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, gate.dtype))
    x_out = x.add_(y * gate) if inplace and not torch.is_grad_enabled() and x_out_dtype == x.dtype else x + y * gate
    norm_out = F.layer_norm(x_out, (x_out.shape[-1],), weight=weight, bias=bias, eps=eps)
    return x_out, norm_out


def fused_wan_res_add_ln_adaln(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    *,
    eps: float = 1e-6,
    slot_shift: int = 3,
    slot_scale: int = 4,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``x_out = x + y`` and ``LayerNorm(x_out, eps) * (1 + scale2) + shift2``."""
    if x.ndim == 3 and y.shape == x.shape and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        x_out_dtype = torch.promote_types(x.dtype, y.dtype)
        if _mod_slot_tensor_valid(modulation, t_mod, b, s, features) and _cuda_eligible(x, y, modulation, t_mod):
            can_inplace = (
                inplace
                and not torch.is_grad_enabled()
                and x_out_dtype == x.dtype
                and not _storage_overlaps(x, y)
                and not _storage_overlaps(x, modulation)
                and not _storage_overlaps(x, t_mod)
            )
            if can_inplace:
                norm_out = _wan_res_add_ln_adaln_inplace_op(
                    x, y, modulation, t_mod, float(eps), int(slot_shift), int(slot_scale)
                )
                return x, norm_out
            return _wan_res_add_ln_adaln_op(x, y, modulation, t_mod, float(eps), int(slot_shift), int(slot_scale))

    x_out_dtype = torch.promote_types(x.dtype, y.dtype)
    x_out = x.add_(y) if inplace and not torch.is_grad_enabled() and x_out_dtype == x.dtype else x + y
    shift = _wan_slot(modulation, t_mod, slot_shift)
    scale = _wan_slot(modulation, t_mod, slot_scale)
    norm_out = F.layer_norm(x_out, (x_out.shape[-1],), weight=None, bias=None, eps=eps) * (1 + scale) + shift
    return x_out, norm_out


def fused_wan_res_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    modulation: torch.Tensor | None,
    t_mod: torch.Tensor,
    *,
    slot_gate: int = 5,
    inplace: bool = False,
) -> torch.Tensor:
    """Fuse ``x + y * (mod[slot_gate] + t_mod[slot_gate])`` or ``x + y * gate``."""
    if x.ndim == 3 and y.shape == x.shape and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        if modulation is not None:
            valid = _mod_slot_tensor_valid(modulation, t_mod, b, s, features) and _cuda_eligible(
                x, y, modulation, t_mod
            )
        else:
            valid = (
                (
                    (t_mod.ndim == 3 and t_mod.shape[0] in (1, b) and t_mod.shape[1] in (1, s))
                    or (t_mod.ndim == 2 and t_mod.shape[0] in (1, b))
                )
                and t_mod.shape[-1] == features
                and _cuda_eligible(x, y, t_mod)
            )
        if valid:
            out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, t_mod.dtype))
            can_inplace = (
                inplace
                and not torch.is_grad_enabled()
                and out_dtype == x.dtype
                and not _storage_overlaps(x, y)
                and not _storage_overlaps(x, t_mod)
                and (modulation is None or not _storage_overlaps(x, modulation))
            )
            if can_inplace:
                _wan_res_gate_add_inplace_op(x, y, modulation, t_mod, int(slot_gate))
                return x
            return _wan_res_gate_add_op(x, y, modulation, t_mod, int(slot_gate))

    gate = (
        _wan_slot(modulation, t_mod, slot_gate)
        if modulation is not None
        else (t_mod.unsqueeze(1) if t_mod.ndim == 2 else t_mod)
    )
    out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, gate.dtype))
    if inplace and not torch.is_grad_enabled() and out_dtype == x.dtype:
        return x.add_(y * gate)
    return x + y * gate


def fused_wan_rmsnorm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Zero-spill Hopper single-pass ``WanRMSNorm`` matching FP32-reduction semantics."""
    features = int(x.shape[-1])
    if (
        x.ndim >= 2
        and 0 < features <= _MAX_ROW_FEATURES
        and weight.shape == (features,)
        and weight.is_contiguous()
        and _cuda_eligible(x, weight)
        and (
            x.is_contiguous()
            or x.ndim == 2
            or (x.ndim == 3 and (x.shape[0] == 1 or x.stride(0) == x.shape[1] * x.stride(1)))
        )
    ):
        if x.ndim == 3 and not x.is_contiguous():
            b, s, c = x.shape
            mat = x[0] if b == 1 else x.as_strided((b * s, c), (x.stride(1), 1))
            return _wan_rmsnorm_op(mat, weight, float(eps)).view(b, s, c)
        return _wan_rmsnorm_op(x, weight, float(eps))
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def fused_wan_head_ln_adaln(
    x: torch.Tensor,
    modulation: torch.Tensor,
    t_mod: torch.Tensor,
    *,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Single-pass FP32 ``Head`` LayerNorm + AdaLN converting directly to ``out_dtype``."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if x.ndim == 3 and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        t_valid = (t_mod.ndim == 2 and t_mod.shape[0] in (1, b) and t_mod.shape[1] == features) or (
            t_mod.ndim == 3 and t_mod.shape[0] in (1, b) and t_mod.shape[1] == s and t_mod.shape[2] == features
        )
        if (
            t_valid
            and modulation.shape == (1, 2, features)
            and target_dtype in _SUPPORTED_DTYPES
            and _cuda_eligible(x, modulation, t_mod)
        ):
            return _wan_head_ln_adaln_op(x, modulation, t_mod, float(eps), target_dtype)

    with torch.autocast(device_type=x.device.type, enabled=False):
        if t_mod.ndim == 3:
            shift, scale = (
                modulation.unsqueeze(0).to(device=t_mod.device, dtype=torch.float32) + t_mod.float().unsqueeze(2)
            ).chunk(2, dim=2)
            out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        else:
            shift, scale = (modulation.to(device=t_mod.device, dtype=torch.float32) + t_mod.float().unsqueeze(1)).chunk(
                2, dim=1
            )
            out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * (1 + scale) + shift
        return out.to(dtype=target_dtype)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (x.float() * rstd).to(x.dtype) * weight


def _apply_interleaved_rope_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c_e = cos[:rope_len, 0::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    c_o = cos[:rope_len, 1::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    s_e = sin[:rope_len, 0::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    s_o = sin[:rope_len, 1::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    def _rot(t: torch.Tensor) -> torch.Tensor:
        prefix = t[:, :rope_len]
        pf = prefix.float()
        e, o = pf[..., 0::2], pf[..., 1::2]
        out_p = torch.stack((e * c_e - o * s_e, o * c_o + e * s_o), dim=-1).flatten(-2).to(t.dtype)
        if rope_len == t.shape[1]:
            return out_p
        return torch.cat((out_p, t[:, rope_len:]), dim=1)

    return _rot(q), _rot(k)


def fused_hunyuan_qkv_rmsnorm_rope_hopper(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    num_heads: int,
    rope_len: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """128-bit coalesced split-Q/K/V head-tiled QKV split + QK-RMSNorm + interleaved 3D RoPE."""
    b, seq_len, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    has_rope = freqs_cis is not None and (rope_len is None or rope_len > 0)
    eff_rope_len = seq_len if rope_len is None else min(int(rope_len), seq_len)

    if (
        head_dim >= 2
        and head_dim <= 256
        and (head_dim & (head_dim - 1)) == 0
        and qkv.ndim == 3
        and q_weight.shape == (head_dim,)
        and k_weight.shape == (head_dim,)
        and _cuda_eligible(qkv, q_weight, k_weight)
    ):
        if has_rope:
            assert freqs_cis is not None
            cos, sin = freqs_cis
            cos_ok = (
                cos.device == qkv.device
                and sin.device == qkv.device
                and cos.ndim == 2
                and sin.shape == cos.shape
                and cos.shape[0] >= eff_rope_len
                and cos.shape[1] == head_dim
                and cos.stride(-1) == 1
                and sin.stride(-1) == 1
            )
            if cos_ok:
                return _hy_qkv_rmsnorm_rope_hopper_op(
                    qkv, q_weight, k_weight, cos, sin, int(num_heads), int(eff_rope_len), True, float(eps)
                )
        else:
            dummy = q_weight.unsqueeze(0)
            return _hy_qkv_rmsnorm_rope_hopper_op(
                qkv, q_weight, k_weight, dummy, dummy, int(num_heads), 0, False, float(eps)
            )

    q, k, v = qkv.reshape(b, seq_len, 3, num_heads, head_dim).unbind(dim=2)
    q = _rms_norm_ref(q, q_weight, eps)
    k = _rms_norm_ref(k, k_weight, eps)
    if has_rope:
        assert freqs_cis is not None
        q, k = _apply_interleaved_rope_ref(q, k, freqs_cis[0], freqs_cis[1], eff_rope_len)
    return q.contiguous(), k.contiguous(), v.contiguous()


def fused_hunyuan_double_qkv_rmsnorm_rope_hopper(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    num_heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """128-bit coalesced split-Q/K/V double-stream QKV -> direct joint ``[B, L, H, D]``."""
    b, img_len, three_c = qkv_img.shape
    head_dim = three_c // (3 * num_heads)
    has_rope = freqs_cis is not None and img_len > 0

    if (
        head_dim >= 2
        and head_dim <= 256
        and (head_dim & (head_dim - 1)) == 0
        and qkv_img.ndim == 3
        and qkv_txt.ndim == 3
        and qkv_img.shape[0] == qkv_txt.shape[0]
        and qkv_img.shape[2] == qkv_txt.shape[2]
        and qkv_img.dtype == qkv_txt.dtype
        and qw_img.shape == (head_dim,)
        and kw_img.shape == (head_dim,)
        and qw_txt.shape == (head_dim,)
        and kw_txt.shape == (head_dim,)
        and _cuda_eligible(qkv_img, qkv_txt, qw_img, kw_img, qw_txt, kw_txt)
    ):
        if has_rope:
            assert freqs_cis is not None
            cos, sin = freqs_cis
            cos_ok = (
                cos.device == qkv_img.device
                and sin.device == qkv_img.device
                and cos.ndim == 2
                and sin.shape == cos.shape
                and cos.shape[0] >= img_len
                and cos.shape[1] == head_dim
                and cos.stride(-1) == 1
                and sin.stride(-1) == 1
            )
            if cos_ok:
                return _hy_double_qkv_rmsnorm_rope_hopper_op(
                    qkv_img, qkv_txt, qw_img, kw_img, qw_txt, kw_txt, cos, sin, int(num_heads), True, float(eps)
                )
        else:
            dummy = qw_img.unsqueeze(0)
            return _hy_double_qkv_rmsnorm_rope_hopper_op(
                qkv_img, qkv_txt, qw_img, kw_img, qw_txt, kw_txt, dummy, dummy, int(num_heads), False, float(eps)
            )

    q_a, k_a, v_a = fused_hunyuan_qkv_rmsnorm_rope_hopper(
        qkv_img, qw_img, kw_img, freqs_cis, num_heads=num_heads, rope_len=img_len, eps=eps
    )
    q_b, k_b, v_b = fused_hunyuan_qkv_rmsnorm_rope_hopper(
        qkv_txt, qw_txt, kw_txt, None, num_heads=num_heads, rope_len=0, eps=eps
    )
    return (
        torch.cat((q_a, q_b), dim=1),
        torch.cat((k_a, k_b), dim=1),
        torch.cat((v_a, v_b), dim=1),
    )


def fused_strided_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """128-bit coalesced row-wise ``GELU(approximate='tanh')`` supporting strided 2D/3D slices."""
    if x.ndim == 3 and _cuda_eligible(x):
        b, s, c = x.shape
        if b == 1:
            mat = x[0]
            return _strided_gelu_tanh_op(mat).unsqueeze(0)
        if x.stride(0) == s * x.stride(1):
            mat = x.as_strided((b * s, c), (x.stride(1), 1))
            return _strided_gelu_tanh_op(mat).view(b, s, c)
    elif x.ndim == 2 and _cuda_eligible(x):
        return _strided_gelu_tanh_op(x)
    return F.gelu(x, approximate="tanh")


def fused_dit_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-pass compile-safe affine-free ``LayerNorm(x, eps) * (1 + scale) + shift``."""
    if x.ndim == 3 and 0 < x.shape[-1] <= _MAX_ROW_FEATURES:
        b, s, features = x.shape
        sc_ok = (
            (scale.ndim == 2 and scale.shape == (b, features))
            or (scale.ndim == 2 and scale.shape == (1, features))
            or (
                scale.ndim == 3 and scale.shape[0] in (1, b) and scale.shape[1] in (1, s) and scale.shape[2] == features
            )
        )
        sh_ok = (
            (shift.ndim == 2 and shift.shape == (b, features))
            or (shift.ndim == 2 and shift.shape == (1, features))
            or (
                shift.ndim == 3 and shift.shape[0] in (1, b) and shift.shape[1] in (1, s) and shift.shape[2] == features
            )
        )
        if sc_ok and sh_ok and _cuda_eligible(x, scale, shift):
            return _dit_ln_adaln_op(x, scale, shift, float(eps))
    sc_v = scale.unsqueeze(1) if scale.ndim == 2 else scale
    sh_v = shift.unsqueeze(1) if shift.ndim == 2 else shift
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps) * (1 + sc_v) + sh_v


def fused_dit_res_gate_ln_adaln(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass ``x_out = x + update * gate`` and ``LN(x_out) * (1 + scale) + shift``."""
    features = int(x.shape[-1])
    if (
        x.ndim == 3
        and update.shape == x.shape
        and gate.ndim == 2
        and scale.ndim == 2
        and shift.ndim == 2
        and gate.shape[0] in (1, x.shape[0])
        and scale.shape[0] in (1, x.shape[0])
        and shift.shape[0] in (1, x.shape[0])
        and gate.shape[1] == scale.shape[1] == shift.shape[1] == features
        and 0 < features <= _MAX_ROW_FEATURES
        and x.dtype == update.dtype
        and _cuda_eligible(x, update, gate, scale, shift)
    ):
        return _dit_res_gate_ln_adaln_op(x, update, gate, scale, shift, float(eps))
    x_out = x + update * gate.unsqueeze(1)
    norm_out = F.layer_norm(x_out, (features,), eps=eps) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x_out, norm_out


def fused_dit_dual_res_gate_add(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Single-pass ``x + (a + b_term) * gate``."""
    features = int(x.shape[-1])
    if (
        x.ndim == 3
        and a.shape == x.shape
        and b_term.shape == x.shape
        and gate.ndim == 2
        and gate.shape[0] in (1, x.shape[0])
        and gate.shape[1] == features
        and 0 < features <= _MAX_ROW_FEATURES
        and x.dtype == a.dtype == b_term.dtype
        and _cuda_eligible(x, a, b_term, gate)
    ):
        return _dit_dual_res_gate_add_op(x, a, b_term, gate)
    return x + (a + b_term) * gate.unsqueeze(1)


__all__ = [
    "dit_block_ops_enabled",
    "fused_dit_dual_res_gate_add",
    "fused_dit_layernorm_adaln",
    "fused_dit_res_gate_ln_adaln",
    "fused_hunyuan_double_qkv_rmsnorm_rope_hopper",
    "fused_hunyuan_qkv_rmsnorm_rope_hopper",
    "fused_strided_gelu_tanh",
    "fused_wan_head_ln_adaln",
    "fused_wan_modulation_ln_adaln",
    "fused_wan_res_add_ln_adaln",
    "fused_wan_res_gate_add",
    "fused_wan_res_gate_ln_affine",
    "fused_wan_rmsnorm",
]
