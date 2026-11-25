# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper Triton operator suite for Self-Forcing 1.3B (Causal Wan).

All kernels preserve exact IEEE BF16/FP16/FP32/FP64 reference semantics (0–1 ULP
vs eager PyTorch, zero algorithmic approximation), are CUDA-Graph capture safe
(zero host syncs, zero dynamic allocations), register ``torch.library.custom_op``
+ ``.register_fake`` entries for graph-break-free ``torch.compile`` composition,
and import Triton lazily so CPU-only environments stay untouched:

- :func:`sf_layernorm_adaln` — single-pass affine-free ``WanLayerNorm`` +
  per-frame AdaLN ``(1 + scale) + shift`` with SM80+ PTX IEEE rounding.
- :func:`sf_qk_rmsnorm_rope_write_kv` — single-pass coalesced 128-bit strided
  Q/K full-hidden ``WanRMSNorm`` + FP64/FP32 interleaved 3D RoPE + direct
  in-place KV cache writes for both ``K`` (rotated or un-rotated raw) and ``V``
  directly from fused ``qkv(x).chunk(3, dim=-1)`` views with zero copies.
- :func:`sf_apply_rope_bshd` — coalesced 128-bit ``[B, S, H, D]`` real-valued
  FP64/FP32 3D RoPE (bit-exact with ``causal_rope_apply``).
- :func:`sf_residual_gate_add_layernorm` — 2-stage super-fusion computing both
  ``x_new = x + y * gate`` and ``norm3_out = LayerNorm(x_new, w, b)`` in one
  HBM pass.
- :func:`sf_residual_add_layernorm_adaln` — 2-stage super-fusion computing both
  ``x_new = x + cross_out`` and ``ffn_in = LayerNorm(x_new) * (1 + scale) + shift``
  in one HBM pass.
- :func:`sf_residual_gate_add` — bit-exact per-frame gated residual add
  ``x + y * gate`` accepting strided modulation views directly.
- :func:`sf_rmsnorm` — single-pass coalesced 128-bit ``WanRMSNorm``.
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
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_MAX_HIDDEN_SIZE = 4096

_TRITON_KERNELS: _TritonKernelSuite | None = None
_TRITON_SPEC_AVAILABLE: bool | None = None
# Bound lazily inside _TritonKernelSuite.__init__ so Triton's AST compiler sees them in module globals
# without importing triton at module load time.
tl: Any = None
_libdevice: Any = None
_round_f32_to_bf16_ptx: Any = None
_round_f32_to_fp16_ptx: Any = None
_round_to_dtype: Any = None
_round_to_code: Any = None


def sf_fast_kernels_enabled(override: bool | None = None) -> bool:
    """Return True when the Self-Forcing Hopper Triton fast path is enabled."""
    if override is not None:
        return bool(override)
    backend = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold()
    if backend in _TORCH_BACKENDS:
        return False
    raw = os.getenv("WORLDFOUNDRY_SF_FAST_KERNELS", "1").strip().casefold()
    return raw not in _FALSY_VALUES


def sf_exact_norms_enabled(
    attention_backend: str | None = None,
    override: bool | None = None,
) -> bool:
    """Return True when 0-ULP bit-exact norm reductions should be preserved.

    In ``auto`` mode (default), ``attention_backend in (None, "torch")`` preserves
    0-ULP bitwise parity (PSNR = inf dB across full 81-frame rollouts), while
    non-torch backends such as Hopper ``flash_attention_3`` activate the full
    single-pass 2-stage super-fusions. Override via ``WORLDFOUNDRY_SF_EXACT_NORMS=0|1``.
    """
    if override is not None:
        return bool(override)
    raw = os.getenv("WORLDFOUNDRY_SF_EXACT_NORMS", "auto").strip().casefold()
    if raw in _TRUTHY_VALUES:
        return True
    if raw in _FALSY_VALUES:
        return False
    return attention_backend in (None, "torch")


def set_sf_fast_kernels_enabled(module: torch.nn.Module, enabled: bool | None) -> None:
    """Enable, disable, or reset (``None``) Self-Forcing fast kernels on *module* and submodules."""
    for sub in module.modules():
        if hasattr(sub, "_worldfoundry_sf_fast_kernels") or type(sub).__name__ in {
            "CausalWanModel",
            "CausalWanAttentionBlock",
            "CausalWanSelfAttention",
            "CausalHead",
        }:
            sub.__dict__["_worldfoundry_sf_fast_kernels"] = enabled


def set_sf_exact_norms(module: torch.nn.Module, exact: bool | None) -> None:
    """Force 0-ULP bit-exact norms (``True``), single-pass super-fusions (``False``), or ``auto`` (``None``)."""
    for sub in module.modules():
        if hasattr(sub, "_worldfoundry_sf_exact_norms") or type(sub).__name__ in {
            "CausalWanModel",
            "CausalWanAttentionBlock",
            "CausalWanSelfAttention",
            "CausalHead",
        }:
            sub.__dict__["_worldfoundry_sf_exact_norms"] = exact


def _has_triton() -> bool:
    """Check Triton availability lazily without importing triton or initializing CUDA."""
    global _TRITON_SPEC_AVAILABLE
    if _TRITON_SPEC_AVAILABLE is None:
        _TRITON_SPEC_AVAILABLE = importlib.util.find_spec("triton") is not None
    return _TRITON_SPEC_AVAILABLE


def _cuda_triton_eligible(tensor: torch.Tensor) -> bool:
    """Return True when *tensor* resides on a CUDA device eligible for Triton execution."""
    if tensor.device.type != "cuda" or bool(getattr(torch.version, "hip", None)):
        return False
    if tensor.dtype not in _SUPPORTED_DTYPES:
        return False
    if not _has_triton():
        return False
    if tensor.dtype == torch.bfloat16:
        major, _ = torch.cuda.get_device_capability(tensor.device)
        if major < 8:
            return False
    return True


def _dtype_to_round_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _frame_mod_valid(
    t: torch.Tensor,
    *,
    batch: int,
    num_frames: int,
    hidden: int,
    device: torch.device,
) -> bool:
    """Validate a per-frame modulation tensor ``[B_m, F_m, (1,) C]`` with stride(-1) == 1."""
    if t.device != device or t.dtype not in _SUPPORTED_DTYPES or t.stride(-1) != 1:
        return False
    if t.ndim == 4:
        return (
            int(t.shape[2]) == 1
            and int(t.shape[3]) == hidden
            and int(t.shape[0]) in (1, batch)
            and int(t.shape[1]) in (1, num_frames)
        )
    if t.ndim == 3:
        return int(t.shape[2]) == hidden and int(t.shape[0]) in (1, batch) and int(t.shape[1]) in (1, num_frames)
    return False


def _frame_mod_strides(t: torch.Tensor) -> tuple[int, int]:
    """Return ``(stride_b, stride_f)`` with 0 for broadcast dimensions (zero copy)."""
    b_s, f_s = int(t.shape[0]), int(t.shape[1])
    return (0 if b_s == 1 else int(t.stride(0))), (0 if f_s == 1 else int(t.stride(1)))


def _optimal_num_warps(block: int) -> int:
    """Select Hopper-tuned warp count (2 warps at BLOCK<=2048 maximizes CTA residency & ILP)."""
    if block <= 2048:
        return 2
    if block <= 4096:
        return 4
    return 8


class _TritonKernelSuite:
    """Lazily-imported Triton kernels so CPU-only imports never load triton or CUDA."""

    def __init__(self) -> None:
        global tl, _libdevice, _round_f32_to_bf16_ptx, _round_f32_to_fp16_ptx, _round_to_dtype, _round_to_code
        try:
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="self-forcing-triton")
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
            """Single-instruction SM80+ PTX IEEE round-to-nearest-even BF16 boundary."""
            return tl.inline_asm_elementwise(
                "{ .reg .b16 r; .reg .b32 b; cvt.rn.bf16.f32 r, $1; cvt.u32.u16 b, r; shl.b32 $0, b, 16; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        @triton.jit
        def _round_fp16(value):
            """PTX IEEE round-to-nearest-even FP16 boundary."""
            return tl.inline_asm_elementwise(
                "{ .reg .b16 rounded; cvt.rn.f16.f32 rounded, $1; cvt.f32.f16 $0, rounded; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

        _round_f32_to_bf16_ptx = _round_bf16
        _round_f32_to_fp16_ptx = _round_fp16

        @triton.jit
        def _round_dtype(value, dtype: tl.constexpr):
            if dtype == tl.bfloat16:
                return _round_f32_to_bf16_ptx(value)
            elif dtype == tl.float16:
                return _round_f32_to_fp16_ptx(value)
            else:
                return value

        @triton.jit
        def _round_code(value, dtype_code: tl.constexpr):
            if dtype_code == 1:
                return _round_f32_to_bf16_ptx(value)
            elif dtype_code == 2:
                return _round_f32_to_fp16_ptx(value)
            else:
                return value

        _round_to_dtype = _round_dtype
        _round_to_code = _round_code

        @triton.jit
        def _sf_layernorm_adaln_kernel(
            x_ptr,
            scale_ptr,
            shift_ptr,
            out_ptr,
            rows,
            seq_len,
            frame_seq,
            scale_sb,
            scale_sf,
            shift_sb,
            shift_sf,
            eps,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
            mul_dtype_code: tl.constexpr,
        ):
            """Single-pass affine-free LayerNorm + strided per-frame AdaLN ``(1 + scale) + shift``."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            x_off = row * hidden_size + cols
            x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            x_f = x_in.to(tl.float32)

            mean = tl.sum(x_f, axis=0) / hidden_size
            diff = tl.where(mask, x_f - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / hidden_size
            rstd = _libdevice.rsqrt(var + eps)
            normed = _round_to_dtype(diff * rstd, x_in.dtype)

            batch_idx = row // seq_len
            frame_idx = (row % seq_len) // frame_seq
            scale_off = batch_idx * scale_sb + frame_idx * scale_sf + cols
            shift_off = batch_idx * shift_sb + frame_idx * shift_sf + cols
            scale_in = tl.load(scale_ptr + scale_off, mask=mask, other=0.0)
            shift_in = tl.load(shift_ptr + shift_off, mask=mask, other=0.0)

            scale_factor = _round_to_dtype(1.0 + scale_in.to(tl.float32), scale_in.dtype)
            modulated = _round_to_code(normed * scale_factor, mul_dtype_code)
            out = modulated + shift_in.to(tl.float32)
            tl.store(out_ptr + x_off, out, mask=mask)

        @triton.jit
        def _sf_residual_gate_add_kernel(
            x_ptr,
            y_ptr,
            gate_ptr,
            out_ptr,
            rows,
            seq_len,
            frame_seq,
            gate_sb,
            gate_sf,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
            prod_dtype_code: tl.constexpr,
        ):
            """Bit-exact ``x + y * gate`` accepting strided per-frame ``gate`` directly."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            off = row * hidden_size + cols
            x_f = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            y_f = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)

            batch_idx = row // seq_len
            frame_idx = (row % seq_len) // frame_seq
            gate_off = batch_idx * gate_sb + frame_idx * gate_sf + cols
            gate_f = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

            prod = _round_to_code(y_f * gate_f, prod_dtype_code)
            tl.store(out_ptr + off, x_f + prod, mask=mask)

        @triton.jit
        def _sf_residual_gate_add_layernorm_kernel(
            x_ptr,
            y_ptr,
            gate_ptr,
            w_ptr,
            b_ptr,
            x_out_ptr,
            norm_out_ptr,
            rows,
            seq_len,
            frame_seq,
            gate_sb,
            gate_sf,
            eps,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
            prod_dtype_code: tl.constexpr,
            x_out_dtype_code: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
        ):
            """Single-pass ``x_out = x + y * gate`` AND ``norm_out = LayerNorm(x_out, w, b)``."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            off = row * hidden_size + cols
            x_f = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            y_f = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)

            batch_idx = row // seq_len
            frame_idx = (row % seq_len) // frame_seq
            gate_off = batch_idx * gate_sb + frame_idx * gate_sf + cols
            gate_f = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

            prod = _round_to_code(y_f * gate_f, prod_dtype_code)
            x_new = _round_to_code(x_f + prod, x_out_dtype_code)
            tl.store(x_out_ptr + off, x_new, mask=mask)

            x_masked = tl.where(mask, x_new, 0.0)
            mean = tl.sum(x_masked, axis=0) / hidden_size
            diff = tl.where(mask, x_masked - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / hidden_size
            rstd = _libdevice.rsqrt(var + eps)
            normed = diff * rstd
            if HAS_WEIGHT:
                w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
                normed = normed * w
            if HAS_BIAS:
                b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = normed + b
            tl.store(norm_out_ptr + off, normed, mask=mask)

        @triton.jit
        def _sf_residual_add_layernorm_adaln_kernel(
            x_ptr,
            upd_ptr,
            scale_ptr,
            shift_ptr,
            x_out_ptr,
            norm_out_ptr,
            rows,
            seq_len,
            frame_seq,
            scale_sb,
            scale_sf,
            shift_sb,
            shift_sf,
            eps,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
            x_out_dtype_code: tl.constexpr,
            mul_dtype_code: tl.constexpr,
        ):
            """Single-pass ``x_out = x + upd`` AND ``norm_out = LayerNorm(x_out)*(1+scale)+shift``."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            off = row * hidden_size + cols
            x_f = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            upd_f = tl.load(upd_ptr + off, mask=mask, other=0.0).to(tl.float32)

            x_new = _round_to_code(x_f + upd_f, x_out_dtype_code)
            tl.store(x_out_ptr + off, x_new, mask=mask)

            x_masked = tl.where(mask, x_new, 0.0)
            mean = tl.sum(x_masked, axis=0) / hidden_size
            diff = tl.where(mask, x_masked - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / hidden_size
            rstd = _libdevice.rsqrt(var + eps)
            normed = _round_to_code(diff * rstd, x_out_dtype_code)

            batch_idx = row // seq_len
            frame_idx = (row % seq_len) // frame_seq
            scale_off = batch_idx * scale_sb + frame_idx * scale_sf + cols
            shift_off = batch_idx * shift_sb + frame_idx * shift_sf + cols
            scale_in = tl.load(scale_ptr + scale_off, mask=mask, other=0.0)
            shift_in = tl.load(shift_ptr + shift_off, mask=mask, other=0.0)

            scale_factor = _round_to_dtype(1.0 + scale_in.to(tl.float32), scale_in.dtype)
            modulated = _round_to_code(normed * scale_factor, mul_dtype_code)
            tl.store(norm_out_ptr + off, modulated + shift_in.to(tl.float32), mask=mask)

        @triton.jit
        def _sf_rmsnorm_kernel(
            x_ptr,
            w_ptr,
            out_ptr,
            rows,
            seq_len,
            x_sb,
            x_ss,
            eps,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
        ):
            """Single-pass coalesced ``WanRMSNorm``: ``(x.float() * rstd).type_as(x) * w``."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            batch_idx = row // seq_len
            tok_idx = row % seq_len
            x_off = batch_idx * x_sb + tok_idx * x_ss + cols
            x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            x_f = x_in.to(tl.float32)

            mean_sq = tl.sum(x_f * x_f, axis=0) / hidden_size
            rstd = _libdevice.rsqrt(mean_sq + eps)
            normed = _round_to_dtype(x_f * rstd, x_in.dtype)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out_off = row * hidden_size + cols
            tl.store(out_ptr + out_off, normed * w, mask=mask)

        @triton.jit
        def _sf_qk_rmsnorm_rope_write_kv_kernel(
            q_in_ptr,
            k_in_ptr,
            v_in_ptr,
            q_w_ptr,
            k_w_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_dst_ptr,
            v_dst_ptr,
            seq_len,
            q_sb,
            q_ss,
            k_sb,
            k_ss,
            v_sb,
            v_ss,
            k_dst_sb,
            k_dst_ss,
            v_dst_sb,
            v_dst_ss,
            dst_start,
            eps,
            hidden_size: tl.constexpr,
            half_head_dim: tl.constexpr,
            block: tl.constexpr,
            block_pairs: tl.constexpr,
            q_norm_code: tl.constexpr,
            k_norm_code: tl.constexpr,
            HAS_QK_NORM: tl.constexpr,
            ROTATE_K: tl.constexpr,
            USE_FP64: tl.constexpr,
        ):
            """Coalesced 128-bit Q/K WanRMSNorm + 3D RoPE + direct K/V cache writes."""
            row = tl.program_id(0).to(tl.int64)
            batch_idx = row // seq_len
            tok_idx = row % seq_len

            cols = tl.arange(0, block)
            cmask = cols < hidden_size

            # Copy V straight from strided QKV view into v_cache first to minimize peak register usage.
            v_base = batch_idx * v_sb + tok_idx * v_ss
            v_dst_base = batch_idx * v_dst_sb + (dst_start + tok_idx) * v_dst_ss
            v_raw = tl.load(v_in_ptr + v_base + cols, mask=cmask, other=0.0)
            tl.store(v_dst_ptr + v_dst_base + cols, v_raw, mask=cmask)

            q_base = batch_idx * q_sb + tok_idx * q_ss
            k_base = batch_idx * k_sb + tok_idx * k_ss
            q_raw = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
            k_raw = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
            q_f = q_raw.to(tl.float32)
            k_f = k_raw.to(tl.float32)

            if HAS_QK_NORM:
                q_rstd = _libdevice.rsqrt(tl.sum(q_f * q_f, axis=0) / hidden_size + eps)
                k_rstd = _libdevice.rsqrt(tl.sum(k_f * k_f, axis=0) / hidden_size + eps)
                qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
                kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
                q_norm = _round_to_code(_round_to_dtype(q_f * q_rstd, q_raw.dtype) * qw, q_norm_code)
                k_norm = _round_to_code(_round_to_dtype(k_f * k_rstd, k_raw.dtype) * kw, k_norm_code)
            else:
                q_norm = q_f
                k_norm = k_f

            k_dst_base = batch_idx * k_dst_sb + (dst_start + tok_idx) * k_dst_ss
            if not ROTATE_K:
                # kv_cache_relative_rope=True: write un-rotated normalized K directly to k_cache.
                tl.store(k_dst_ptr + k_dst_base + cols, k_norm, mask=cmask)

            qe, qo = tl.split(tl.reshape(q_norm, (block_pairs, 2)))
            pair = tl.arange(0, block_pairs)
            pmask = pair < (hidden_size // 2)
            cs_off = tok_idx * half_head_dim + (pair % half_head_dim)
            if USE_FP64:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
                rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(v_raw.dtype)
                rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(v_raw.dtype)
            else:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
                rqe = (qe * c - qo * s).to(v_raw.dtype)
                rqo = (qe * s + qo * c).to(v_raw.dtype)

            q_out_base = row * hidden_size
            tl.store(q_out_ptr + q_out_base + cols, tl.reshape(tl.join(rqe, rqo), (block,)), mask=cmask)

            if ROTATE_K:
                ke, ko = tl.split(tl.reshape(k_norm, (block_pairs, 2)))
                if USE_FP64:
                    rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(v_raw.dtype)
                    rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(v_raw.dtype)
                else:
                    rke = (ke * c - ko * s).to(v_raw.dtype)
                    rko = (ke * s + ko * c).to(v_raw.dtype)
                tl.store(k_dst_ptr + k_dst_base + cols, tl.reshape(tl.join(rke, rko), (block,)), mask=cmask)

        @triton.jit
        def _sf_apply_rope_row_kernel(
            x_ptr,
            cos_ptr,
            sin_ptr,
            out_ptr,
            seq_len,
            x_sb,
            x_ss,
            hidden_size: tl.constexpr,
            half_head_dim: tl.constexpr,
            block: tl.constexpr,
            block_pairs: tl.constexpr,
            USE_FP64: tl.constexpr,
        ):
            """Row-coalesced 128-bit real-valued 3D RoPE across all heads of a token."""
            row = tl.program_id(0).to(tl.int64)
            batch_idx = row // seq_len
            tok_idx = row % seq_len

            cols = tl.arange(0, block)
            cmask = cols < hidden_size
            x_base = batch_idx * x_sb + tok_idx * x_ss
            x_raw = tl.load(x_ptr + x_base + cols, mask=cmask, other=0.0)

            xe, xo = tl.split(tl.reshape(x_raw, (block_pairs, 2)))
            pair = tl.arange(0, block_pairs)
            pmask = pair < (hidden_size // 2)
            cs_off = tok_idx * half_head_dim + (pair % half_head_dim)
            if USE_FP64:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
                rxe = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(tl.float32).to(x_raw.dtype)
                rxo = (xe.to(tl.float64) * s + xo.to(tl.float64) * c).to(tl.float32).to(x_raw.dtype)
            else:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
                rxe = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                rxo = (xe.to(tl.float32) * s + xo.to(tl.float32) * c).to(x_raw.dtype)

            out_base = row * hidden_size
            tl.store(out_ptr + out_base + cols, tl.reshape(tl.join(rxe, rxo), (block,)), mask=cmask)

        @triton.jit
        def _sf_apply_rope_head_kernel(
            x_ptr,
            cos_ptr,
            sin_ptr,
            out_ptr,
            seq_len,
            x_sb,
            x_ss,
            x_sh,
            out_sb,
            out_ss,
            out_sh,
            head_dim: tl.constexpr,
            half_head_dim: tl.constexpr,
            block: tl.constexpr,
            block_pairs: tl.constexpr,
            USE_FP64: tl.constexpr,
        ):
            """Per-head coalesced real-valued 3D RoPE fallback for wide/strided ``[B, S, H, D]``."""
            row = tl.program_id(0).to(tl.int64)
            head = tl.program_id(1).to(tl.int64)
            batch_idx = row // seq_len
            tok_idx = row % seq_len

            cols = tl.arange(0, block)
            cmask = cols < head_dim
            x_base = batch_idx * x_sb + tok_idx * x_ss + head * x_sh
            x_raw = tl.load(x_ptr + x_base + cols, mask=cmask, other=0.0)

            xe, xo = tl.split(tl.reshape(x_raw, (block_pairs, 2)))
            pair = tl.arange(0, block_pairs)
            pmask = pair < half_head_dim
            cs_off = tok_idx * half_head_dim + pair
            if USE_FP64:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
                rxe = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(tl.float32).to(x_raw.dtype)
                rxo = (xe.to(tl.float64) * s + xo.to(tl.float64) * c).to(tl.float32).to(x_raw.dtype)
            else:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
                rxe = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                rxo = (xe.to(tl.float32) * s + xo.to(tl.float32) * c).to(x_raw.dtype)

            out_base = batch_idx * out_sb + tok_idx * out_ss + head * out_sh
            tl.store(out_ptr + out_base + cols, tl.reshape(tl.join(rxe, rxo), (block,)), mask=cmask)

        @triton.jit
        def _sf_scale_shift_kernel(
            normed_ptr,
            scale_ptr,
            shift_ptr,
            out_ptr,
            rows,
            seq_len,
            frame_seq,
            scale_sb,
            scale_sf,
            shift_sb,
            shift_sf,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
            mul_dtype_code: tl.constexpr,
        ):
            """Bit-exact per-frame AdaLN ``normed * (1 + scale) + shift`` on pre-normed input."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            x_off = row * hidden_size + cols
            normed = tl.load(normed_ptr + x_off, mask=mask, other=0.0).to(tl.float32)

            batch_idx = row // seq_len
            frame_idx = (row % seq_len) // frame_seq
            scale_off = batch_idx * scale_sb + frame_idx * scale_sf + cols
            shift_off = batch_idx * shift_sb + frame_idx * shift_sf + cols
            scale_in = tl.load(scale_ptr + scale_off, mask=mask, other=0.0)
            shift_in = tl.load(shift_ptr + shift_off, mask=mask, other=0.0)

            scale_factor = _round_to_dtype(1.0 + scale_in.to(tl.float32), scale_in.dtype)
            modulated = _round_to_code(normed * scale_factor, mul_dtype_code)
            tl.store(out_ptr + x_off, modulated + shift_in.to(tl.float32), mask=mask)

        @triton.jit
        def _sf_apply_rstd_weight_kernel(
            x_ptr,
            rstd_ptr,
            w_ptr,
            out_ptr,
            rows,
            seq_len,
            x_sb,
            x_ss,
            hidden_size: tl.constexpr,
            block: tl.constexpr,
        ):
            """Bit-exact ``(x.float() * rstd).type_as(x) * w`` using exact PyTorch ``rstd``."""
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, block)
            mask = (row < rows) & (cols < hidden_size)

            batch_idx = row // seq_len
            tok_idx = row % seq_len
            x_off = batch_idx * x_sb + tok_idx * x_ss + cols
            x_in = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            rstd = tl.load(rstd_ptr + row, mask=row < rows, other=1.0).to(tl.float32)
            normed = _round_to_dtype(x_in.to(tl.float32) * rstd, x_in.dtype)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + row * hidden_size + cols, normed * w, mask=mask)

        @triton.jit
        def _sf_qk_apply_rstd_rope_write_kv_kernel(
            q_in_ptr,
            k_in_ptr,
            v_in_ptr,
            q_rstd_ptr,
            k_rstd_ptr,
            q_w_ptr,
            k_w_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_dst_ptr,
            v_dst_ptr,
            seq_len,
            q_sb,
            q_ss,
            k_sb,
            k_ss,
            v_sb,
            v_ss,
            k_dst_sb,
            k_dst_ss,
            v_dst_sb,
            v_dst_ss,
            dst_start,
            hidden_size: tl.constexpr,
            half_head_dim: tl.constexpr,
            block: tl.constexpr,
            block_pairs: tl.constexpr,
            q_norm_code: tl.constexpr,
            k_norm_code: tl.constexpr,
            ROTATE_K: tl.constexpr,
            USE_FP64: tl.constexpr,
        ):
            """0-ULP bit-exact coalesced Q/K rstd-scale + weight + 3D RoPE + direct K/V cache write."""
            row = tl.program_id(0).to(tl.int64)
            batch_idx = row // seq_len
            tok_idx = row % seq_len

            cols = tl.arange(0, block)
            cmask = cols < hidden_size

            v_base = batch_idx * v_sb + tok_idx * v_ss
            v_dst_base = batch_idx * v_dst_sb + (dst_start + tok_idx) * v_dst_ss
            v_raw = tl.load(v_in_ptr + v_base + cols, mask=cmask, other=0.0)
            tl.store(v_dst_ptr + v_dst_base + cols, v_raw, mask=cmask)

            q_base = batch_idx * q_sb + tok_idx * q_ss
            k_base = batch_idx * k_sb + tok_idx * k_ss
            q_raw = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
            k_raw = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
            q_rstd = tl.load(q_rstd_ptr + row).to(tl.float32)
            k_rstd = tl.load(k_rstd_ptr + row).to(tl.float32)
            qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            q_norm = _round_to_code(_round_to_dtype(q_raw.to(tl.float32) * q_rstd, q_raw.dtype) * qw, q_norm_code)
            k_norm = _round_to_code(_round_to_dtype(k_raw.to(tl.float32) * k_rstd, k_raw.dtype) * kw, k_norm_code)

            k_dst_base = batch_idx * k_dst_sb + (dst_start + tok_idx) * k_dst_ss
            if not ROTATE_K:
                tl.store(k_dst_ptr + k_dst_base + cols, k_norm, mask=cmask)

            qe, qo = tl.split(tl.reshape(q_norm, (block_pairs, 2)))
            pair = tl.arange(0, block_pairs)
            pmask = pair < (hidden_size // 2)
            cs_off = tok_idx * half_head_dim + (pair % half_head_dim)
            if USE_FP64:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
                rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(v_raw.dtype)
                rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(v_raw.dtype)
            else:
                c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
                s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
                rqe = (qe * c - qo * s).to(v_raw.dtype)
                rqo = (qe * s + qo * c).to(v_raw.dtype)

            tl.store(q_out_ptr + row * hidden_size + cols, tl.reshape(tl.join(rqe, rqo), (block,)), mask=cmask)

            if ROTATE_K:
                ke, ko = tl.split(tl.reshape(k_norm, (block_pairs, 2)))
                if USE_FP64:
                    rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(v_raw.dtype)
                    rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(v_raw.dtype)
                else:
                    rke = (ke * c - ko * s).to(v_raw.dtype)
                    rko = (ke * s + ko * c).to(v_raw.dtype)
                tl.store(k_dst_ptr + k_dst_base + cols, tl.reshape(tl.join(rke, rko), (block,)), mask=cmask)

        self.layernorm_adaln = _sf_layernorm_adaln_kernel
        self.scale_shift = _sf_scale_shift_kernel
        self.residual_gate_add = _sf_residual_gate_add_kernel
        self.residual_gate_add_layernorm = _sf_residual_gate_add_layernorm_kernel
        self.residual_add_layernorm_adaln = _sf_residual_add_layernorm_adaln_kernel
        self.rmsnorm = _sf_rmsnorm_kernel
        self.apply_rstd_weight = _sf_apply_rstd_weight_kernel
        self.qk_rmsnorm_rope_write_kv = _sf_qk_rmsnorm_rope_write_kv_kernel
        self.qk_apply_rstd_rope_write_kv = _sf_qk_apply_rstd_rope_write_kv_kernel
        self.apply_rope_row = _sf_apply_rope_row_kernel
        self.apply_rope_head = _sf_apply_rope_head_kernel


def _get_triton_kernels() -> _TritonKernelSuite:
    global _TRITON_KERNELS
    if _TRITON_KERNELS is None:
        _TRITON_KERNELS = _TritonKernelSuite()
    return _TRITON_KERNELS


# ──────────────────────────────────────────────────────────────────────────
# torch.library.custom_op registrations (CUDA Graph + torch.compile safe)
# ──────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("worldfoundry::sf_layernorm_adaln", mutates_args=(), device_types="cuda")
def _sf_layernorm_adaln_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> torch.Tensor:
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    mul_dtype = torch.promote_types(x.dtype, scale.dtype)
    out_dtype = torch.promote_types(mul_dtype, shift.dtype)
    out = torch.empty_like(x, dtype=out_dtype)
    scale_sb, scale_sf = _frame_mod_strides(scale)
    shift_sb, shift_sf = _frame_mod_strides(shift)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.layernorm_adaln[(rows,)](
        x,
        scale,
        shift,
        out,
        rows,
        seq_len,
        int(frame_seq),
        scale_sb,
        scale_sf,
        shift_sb,
        shift_sf,
        float(eps),
        hidden_size=hidden,
        block=block,
        mul_dtype_code=_dtype_to_round_code(mul_dtype),
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_sf_layernorm_adaln_op.register_fake
def _sf_layernorm_adaln_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> torch.Tensor:
    del num_frames, frame_seq, eps
    out_dtype = torch.promote_types(torch.promote_types(x.dtype, scale.dtype), shift.dtype)
    return torch.empty_like(x, dtype=out_dtype)


@torch.library.custom_op("worldfoundry::sf_scale_shift", mutates_args=(), device_types="cuda")
def _sf_scale_shift_op(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    del num_frames
    suite = _get_triton_kernels()
    seq_len = int(normed.shape[1])
    hidden = int(normed.shape[2])
    rows = int(normed.shape[0]) * seq_len
    mul_dtype = torch.promote_types(normed.dtype, scale.dtype)
    out_dtype = torch.promote_types(mul_dtype, shift.dtype)
    out = torch.empty_like(normed, dtype=out_dtype)
    scale_sb, scale_sf = _frame_mod_strides(scale)
    shift_sb, shift_sf = _frame_mod_strides(shift)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.scale_shift[(rows,)](
        normed,
        scale,
        shift,
        out,
        rows,
        seq_len,
        int(frame_seq),
        scale_sb,
        scale_sf,
        shift_sb,
        shift_sf,
        hidden_size=hidden,
        block=block,
        mul_dtype_code=_dtype_to_round_code(mul_dtype),
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_sf_scale_shift_op.register_fake
def _sf_scale_shift_fake(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    del num_frames, frame_seq
    out_dtype = torch.promote_types(torch.promote_types(normed.dtype, scale.dtype), shift.dtype)
    return torch.empty_like(normed, dtype=out_dtype)


@torch.library.custom_op("worldfoundry::sf_apply_rstd_weight", mutates_args=(), device_types="cuda")
def _sf_apply_rstd_weight_op(x: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    out_dtype = torch.promote_types(x.dtype, weight.dtype)
    out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    rstd_c = rstd.reshape(rows).contiguous()
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.apply_rstd_weight[(rows,)](
        x,
        rstd_c,
        weight,
        out,
        rows,
        seq_len,
        int(x.stride(0)),
        int(x.stride(1)),
        hidden_size=hidden,
        block=block,
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_sf_apply_rstd_weight_op.register_fake
def _sf_apply_rstd_weight_fake(x: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    del rstd
    return torch.empty(x.shape, device=x.device, dtype=torch.promote_types(x.dtype, weight.dtype))


@torch.library.custom_op(
    "worldfoundry::sf_qk_apply_rstd_rope_write_kv",
    mutates_args=("k_cache", "v_cache"),
    device_types="cuda",
)
def _sf_qk_apply_rstd_rope_write_kv_op(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    dst_start: int,
    num_heads: int,
    head_dim: int,
    rotate_k: bool,
) -> torch.Tensor:
    suite = _get_triton_kernels()
    batch, seq_len, hidden = int(q_proj.shape[0]), int(q_proj.shape[1]), int(q_proj.shape[2])
    half_d = int(head_dim) // 2
    rows = batch * seq_len
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    block_pairs = block // 2
    q_out = torch.empty((batch, seq_len, int(num_heads), int(head_dim)), device=q_proj.device, dtype=v_proj.dtype)
    q_rstd_c = q_rstd.reshape(rows).contiguous()
    k_rstd_c = k_rstd.reshape(rows).contiguous()
    cos_c = cos.reshape(seq_len, half_d).contiguous()
    sin_c = sin.reshape(seq_len, half_d).contiguous()
    q_norm_dtype = torch.promote_types(q_proj.dtype, q_weight.dtype)
    k_norm_dtype = torch.promote_types(k_proj.dtype, k_weight.dtype)
    num_warps = 4 if block >= 1024 else 2
    if block > 2048:
        num_warps = 8
    suite.qk_apply_rstd_rope_write_kv[(rows,)](
        q_proj,
        k_proj,
        v_proj,
        q_rstd_c,
        k_rstd_c,
        q_weight,
        k_weight,
        cos_c,
        sin_c,
        q_out,
        k_cache,
        v_cache,
        seq_len,
        int(q_proj.stride(0)),
        int(q_proj.stride(1)),
        int(k_proj.stride(0)),
        int(k_proj.stride(1)),
        int(v_proj.stride(0)),
        int(v_proj.stride(1)),
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        int(v_cache.stride(0)),
        int(v_cache.stride(1)),
        int(dst_start),
        hidden_size=hidden,
        half_head_dim=half_d,
        block=block,
        block_pairs=block_pairs,
        q_norm_code=_dtype_to_round_code(q_norm_dtype),
        k_norm_code=_dtype_to_round_code(k_norm_dtype),
        ROTATE_K=bool(rotate_k),
        USE_FP64=cos.dtype == torch.float64,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return q_out


@_sf_qk_apply_rstd_rope_write_kv_op.register_fake
def _sf_qk_apply_rstd_rope_write_kv_fake(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    dst_start: int,
    num_heads: int,
    head_dim: int,
    rotate_k: bool,
) -> torch.Tensor:
    del k_proj, q_rstd, k_rstd, q_weight, k_weight, cos, sin, k_cache, v_cache, dst_start, rotate_k
    return torch.empty(
        (q_proj.shape[0], q_proj.shape[1], int(num_heads), int(head_dim)), device=q_proj.device, dtype=v_proj.dtype
    )


@torch.library.custom_op("worldfoundry::sf_residual_gate_add", mutates_args=(), device_types="cuda")
def _sf_residual_gate_add_op(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    del num_frames
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    prod_dtype = torch.promote_types(y.dtype, gate.dtype)
    out_dtype = torch.promote_types(x.dtype, prod_dtype)
    out = torch.empty_like(x, dtype=out_dtype)
    gate_sb, gate_sf = _frame_mod_strides(gate)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.residual_gate_add[(rows,)](
        x,
        y,
        gate,
        out,
        rows,
        seq_len,
        int(frame_seq),
        gate_sb,
        gate_sf,
        hidden_size=hidden,
        block=block,
        prod_dtype_code=_dtype_to_round_code(prod_dtype),
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_sf_residual_gate_add_op.register_fake
def _sf_residual_gate_add_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    del num_frames, frame_seq
    out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, gate.dtype))
    return torch.empty_like(x, dtype=out_dtype)


@torch.library.custom_op("worldfoundry::sf_residual_gate_add_layernorm", mutates_args=(), device_types="cuda")
def _sf_residual_gate_add_layernorm_op(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_frames
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    prod_dtype = torch.promote_types(y.dtype, gate.dtype)
    x_out_dtype = torch.promote_types(x.dtype, prod_dtype)
    x_out = torch.empty_like(x, dtype=x_out_dtype)
    norm_out = torch.empty_like(x, dtype=x_out_dtype)
    gate_sb, gate_sf = _frame_mod_strides(gate)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.residual_gate_add_layernorm[(rows,)](
        x,
        y,
        gate,
        x if weight is None else weight,
        x if bias is None else bias,
        x_out,
        norm_out,
        rows,
        seq_len,
        int(frame_seq),
        gate_sb,
        gate_sf,
        float(eps),
        hidden_size=hidden,
        block=block,
        prod_dtype_code=_dtype_to_round_code(prod_dtype),
        x_out_dtype_code=_dtype_to_round_code(x_out_dtype),
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return x_out, norm_out


@_sf_residual_gate_add_layernorm_op.register_fake
def _sf_residual_gate_add_layernorm_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, bias, num_frames, frame_seq, eps
    x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, gate.dtype))
    return torch.empty_like(x, dtype=x_out_dtype), torch.empty_like(x, dtype=x_out_dtype)


@torch.library.custom_op("worldfoundry::sf_residual_add_layernorm_adaln", mutates_args=(), device_types="cuda")
def _sf_residual_add_layernorm_adaln_op(
    x: torch.Tensor,
    residual_update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_frames
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    x_out_dtype = torch.promote_types(x.dtype, residual_update.dtype)
    mul_dtype = torch.promote_types(x_out_dtype, scale.dtype)
    ffn_out_dtype = torch.promote_types(mul_dtype, shift.dtype)
    x_out = torch.empty_like(x, dtype=x_out_dtype)
    norm_out = torch.empty_like(x, dtype=ffn_out_dtype)
    scale_sb, scale_sf = _frame_mod_strides(scale)
    shift_sb, shift_sf = _frame_mod_strides(shift)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.residual_add_layernorm_adaln[(rows,)](
        x,
        residual_update,
        scale,
        shift,
        x_out,
        norm_out,
        rows,
        seq_len,
        int(frame_seq),
        scale_sb,
        scale_sf,
        shift_sb,
        shift_sf,
        float(eps),
        hidden_size=hidden,
        block=block,
        x_out_dtype_code=_dtype_to_round_code(x_out_dtype),
        mul_dtype_code=_dtype_to_round_code(mul_dtype),
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return x_out, norm_out


@_sf_residual_add_layernorm_adaln_op.register_fake
def _sf_residual_add_layernorm_adaln_fake(
    x: torch.Tensor,
    residual_update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seq: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_frames, frame_seq, eps
    x_out_dtype = torch.promote_types(x.dtype, residual_update.dtype)
    ffn_out_dtype = torch.promote_types(torch.promote_types(x_out_dtype, scale.dtype), shift.dtype)
    return torch.empty_like(x, dtype=x_out_dtype), torch.empty_like(x, dtype=ffn_out_dtype)


@torch.library.custom_op("worldfoundry::sf_rmsnorm", mutates_args=(), device_types="cuda")
def _sf_rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    suite = _get_triton_kernels()
    seq_len = int(x.shape[1])
    hidden = int(x.shape[2])
    rows = int(x.shape[0]) * seq_len
    out_dtype = torch.promote_types(x.dtype, weight.dtype)
    out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    suite.rmsnorm[(rows,)](
        x,
        weight,
        out,
        rows,
        seq_len,
        int(x.stride(0)),
        int(x.stride(1)),
        float(eps),
        hidden_size=hidden,
        block=block,
        num_warps=_optimal_num_warps(block),
        enable_fp_fusion=False,
    )
    return out


@_sf_rmsnorm_op.register_fake
def _sf_rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    del eps
    return torch.empty(x.shape, device=x.device, dtype=torch.promote_types(x.dtype, weight.dtype))


@torch.library.custom_op(
    "worldfoundry::sf_qk_rmsnorm_rope_write_kv",
    mutates_args=("k_cache", "v_cache"),
    device_types="cuda",
)
def _sf_qk_rmsnorm_rope_write_kv_op(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    dst_start: int,
    num_heads: int,
    head_dim: int,
    eps: float,
    rotate_k: bool,
) -> torch.Tensor:
    suite = _get_triton_kernels()
    batch, seq_len, hidden = int(q_proj.shape[0]), int(q_proj.shape[1]), int(q_proj.shape[2])
    half_d = int(head_dim) // 2
    rows = batch * seq_len
    block = max(32, int(suite.triton.next_power_of_2(hidden)))
    block_pairs = block // 2
    q_out = torch.empty((batch, seq_len, int(num_heads), int(head_dim)), device=q_proj.device, dtype=v_proj.dtype)
    cos_c = cos.reshape(seq_len, half_d).contiguous()
    sin_c = sin.reshape(seq_len, half_d).contiguous()
    has_qk_norm = q_weight is not None and k_weight is not None
    q_norm_dtype = torch.promote_types(q_proj.dtype, q_weight.dtype) if q_weight is not None else q_proj.dtype
    k_norm_dtype = torch.promote_types(k_proj.dtype, k_weight.dtype) if k_weight is not None else k_proj.dtype
    num_warps = 4 if block >= 1024 else 2
    if block > 2048:
        num_warps = 8
    suite.qk_rmsnorm_rope_write_kv[(rows,)](
        q_proj,
        k_proj,
        v_proj,
        q_proj if q_weight is None else q_weight,
        k_proj if k_weight is None else k_weight,
        cos_c,
        sin_c,
        q_out,
        k_cache,
        v_cache,
        seq_len,
        int(q_proj.stride(0)),
        int(q_proj.stride(1)),
        int(k_proj.stride(0)),
        int(k_proj.stride(1)),
        int(v_proj.stride(0)),
        int(v_proj.stride(1)),
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        int(v_cache.stride(0)),
        int(v_cache.stride(1)),
        int(dst_start),
        float(eps),
        hidden_size=hidden,
        half_head_dim=half_d,
        block=block,
        block_pairs=block_pairs,
        q_norm_code=_dtype_to_round_code(q_norm_dtype),
        k_norm_code=_dtype_to_round_code(k_norm_dtype),
        HAS_QK_NORM=has_qk_norm,
        ROTATE_K=bool(rotate_k),
        USE_FP64=cos.dtype == torch.float64,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return q_out


@_sf_qk_rmsnorm_rope_write_kv_op.register_fake
def _sf_qk_rmsnorm_rope_write_kv_fake(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    dst_start: int,
    num_heads: int,
    head_dim: int,
    eps: float,
    rotate_k: bool,
) -> torch.Tensor:
    del k_proj, q_weight, k_weight, cos, sin, k_cache, v_cache, dst_start, eps, rotate_k
    return torch.empty(
        (q_proj.shape[0], q_proj.shape[1], int(num_heads), int(head_dim)), device=q_proj.device, dtype=v_proj.dtype
    )


@torch.library.custom_op("worldfoundry::sf_apply_rope_bshd", mutates_args=(), device_types="cuda")
def _sf_apply_rope_bshd_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    suite = _get_triton_kernels()
    batch, seq_len, num_heads, head_dim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    half_d = head_dim // 2
    hidden = num_heads * head_dim
    out = torch.empty((batch, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    cos_c = cos.reshape(seq_len, half_d).contiguous()
    sin_c = sin.reshape(seq_len, half_d).contiguous()
    use_fp64 = cos.dtype == torch.float64
    if x.stride(2) == head_dim and hidden <= _MAX_HIDDEN_SIZE:
        block = max(32, int(suite.triton.next_power_of_2(hidden)))
        suite.apply_rope_row[(batch * seq_len,)](
            x,
            cos_c,
            sin_c,
            out,
            seq_len,
            int(x.stride(0)),
            int(x.stride(1)),
            hidden_size=hidden,
            half_head_dim=half_d,
            block=block,
            block_pairs=block // 2,
            USE_FP64=use_fp64,
            num_warps=4 if block >= 1024 else 2,
            enable_fp_fusion=False,
        )
    else:
        block = max(32, int(suite.triton.next_power_of_2(head_dim)))
        suite.apply_rope_head[(batch * seq_len, num_heads)](
            x,
            cos_c,
            sin_c,
            out,
            seq_len,
            int(x.stride(0)),
            int(x.stride(1)),
            int(x.stride(2)),
            int(out.stride(0)),
            int(out.stride(1)),
            int(out.stride(2)),
            head_dim=head_dim,
            half_head_dim=half_d,
            block=block,
            block_pairs=block // 2,
            USE_FP64=use_fp64,
            num_warps=2 if block <= 128 else 4,
            enable_fp_fusion=False,
        )
    return out


@_sf_apply_rope_bshd_op.register_fake
def _sf_apply_rope_bshd_fake(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    del cos, sin
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Public API with exact PyTorch fallbacks
# ──────────────────────────────────────────────────────────────────────────


def sf_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
    exact: bool = False,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Single-pass ``WanLayerNorm`` + per-frame AdaLN ``(1 + scale) + shift``.

    When ``exact=True``, computes the LayerNorm reduction via PyTorch ATen
    ``F.layer_norm`` and fuses the per-frame AdaLN modulation in Triton so the
    output is 0-ULP bit-exact with eager PyTorch.
    """
    hidden = int(x.shape[-1])
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(x)
        and x.ndim == 3
        and x.is_contiguous()
        and x.numel() > 0
        and 0 < hidden <= _MAX_HIDDEN_SIZE
        and num_frames > 0
        and frame_seq > 0
        and int(x.shape[1]) == int(num_frames) * int(frame_seq)
        and _frame_mod_valid(scale, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
        and _frame_mod_valid(shift, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
    ):
        if exact:
            normed = F.layer_norm(x, (hidden,), weight=None, bias=None, eps=float(eps)).type_as(x)
            return _sf_scale_shift_op(normed, scale, shift, int(num_frames), int(frame_seq))
        return _sf_layernorm_adaln_op(x, scale, shift, int(num_frames), int(frame_seq), float(eps))
    normed = F.layer_norm(x, (hidden,), weight=None, bias=None, eps=float(eps)).type_as(x)
    scale_b = scale.unsqueeze(2) if scale.ndim == 3 else scale
    shift_b = shift.unsqueeze(2) if shift.ndim == 3 else shift
    return (normed.unflatten(1, (int(num_frames), int(frame_seq))) * (1 + scale_b) + shift_b).flatten(1, 2)


def sf_residual_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Bit-exact per-frame gated residual ``x + (y.unflatten(1, (F, HW)) * gate).flatten(1, 2)``."""
    hidden = int(x.shape[-1])
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(x)
        and y.device == x.device
        and y.dtype in _SUPPORTED_DTYPES
        and x.ndim == 3
        and y.shape == x.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.numel() > 0
        and 0 < hidden <= _MAX_HIDDEN_SIZE
        and num_frames > 0
        and frame_seq > 0
        and int(x.shape[1]) == int(num_frames) * int(frame_seq)
        and _frame_mod_valid(gate, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
    ):
        return _sf_residual_gate_add_op(x, y, gate, int(num_frames), int(frame_seq))
    gate_b = gate.unsqueeze(2) if gate.ndim == 3 else gate
    return x + (y.unflatten(1, (int(num_frames), int(frame_seq))) * gate_b).flatten(1, 2)


def sf_residual_gate_add_layernorm(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
    exact: bool = False,
    enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-stage super-fusion: ``x_new = x + y * gate`` AND ``norm3_out = WanLayerNorm(x_new, w, b)``."""
    hidden = int(x.shape[-1])
    x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, gate.dtype))
    weight_ok = weight is None or (
        weight.device == x.device
        and weight.dtype == x_out_dtype
        and weight.is_contiguous()
        and weight.shape == (hidden,)
    )
    bias_ok = bias is None or (
        bias.device == x.device and bias.dtype == x_out_dtype and bias.is_contiguous() and bias.shape == (hidden,)
    )
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(x)
        and y.device == x.device
        and y.dtype in _SUPPORTED_DTYPES
        and x.ndim == 3
        and y.shape == x.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.numel() > 0
        and 0 < hidden <= _MAX_HIDDEN_SIZE
        and num_frames > 0
        and frame_seq > 0
        and int(x.shape[1]) == int(num_frames) * int(frame_seq)
        and _frame_mod_valid(gate, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
        and weight_ok
        and bias_ok
    ):
        if exact:
            x_new = _sf_residual_gate_add_op(x, y, gate, int(num_frames), int(frame_seq))
            norm_out = F.layer_norm(x_new, (hidden,), weight=weight, bias=bias, eps=float(eps)).type_as(x_new)
            return x_new, norm_out
        return _sf_residual_gate_add_layernorm_op(x, y, gate, weight, bias, int(num_frames), int(frame_seq), float(eps))
    gate_b = gate.unsqueeze(2) if gate.ndim == 3 else gate
    x_new = x + (y.unflatten(1, (int(num_frames), int(frame_seq))) * gate_b).flatten(1, 2)
    norm_out = F.layer_norm(x_new, (hidden,), weight=weight, bias=bias, eps=float(eps)).type_as(x_new)
    return x_new, norm_out


def sf_residual_add_layernorm_adaln(
    x: torch.Tensor,
    residual_update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
    exact: bool = False,
    enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-stage super-fusion: ``x_new = x + residual_update`` AND ``LayerNorm(x_new) * (1 + scale) + shift``."""
    hidden = int(x.shape[-1])
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(x)
        and residual_update.device == x.device
        and residual_update.dtype in _SUPPORTED_DTYPES
        and x.ndim == 3
        and residual_update.shape == x.shape
        and x.is_contiguous()
        and residual_update.is_contiguous()
        and x.numel() > 0
        and 0 < hidden <= _MAX_HIDDEN_SIZE
        and num_frames > 0
        and frame_seq > 0
        and int(x.shape[1]) == int(num_frames) * int(frame_seq)
        and _frame_mod_valid(scale, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
        and _frame_mod_valid(shift, batch=int(x.shape[0]), num_frames=int(num_frames), hidden=hidden, device=x.device)
    ):
        if exact:
            x_new = x + residual_update
            normed = F.layer_norm(x_new, (hidden,), weight=None, bias=None, eps=float(eps)).type_as(x_new)
            ffn_in = _sf_scale_shift_op(normed, scale, shift, int(num_frames), int(frame_seq))
            return x_new, ffn_in
        return _sf_residual_add_layernorm_adaln_op(
            x, residual_update, scale, shift, int(num_frames), int(frame_seq), float(eps)
        )
    x_new = x + residual_update
    normed = F.layer_norm(x_new, (hidden,), weight=None, bias=None, eps=float(eps)).type_as(x_new)
    scale_b = scale.unsqueeze(2) if scale.ndim == 3 else scale
    shift_b = shift.unsqueeze(2) if shift.ndim == 3 else shift
    ffn_in = (normed.unflatten(1, (int(num_frames), int(frame_seq))) * (1 + scale_b) + shift_b).flatten(1, 2)
    return x_new, ffn_in


def _exact_rms_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute exact WanRMSNorm rstd without mutating FP32 inputs or duplicating FP32 temporaries."""
    sq = x.square() if x.dtype == torch.float32 else x.float().square_()
    return torch.rsqrt(sq.mean(dim=-1) + eps)


def sf_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    exact: bool = False,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Single-pass coalesced ``WanRMSNorm`` supporting strided last-dim-contiguous ``[B, S, C]``."""
    hidden = int(x.shape[-1])
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(x)
        and weight.device == x.device
        and weight.dtype in _SUPPORTED_DTYPES
        and weight.ndim == 1
        and int(weight.shape[0]) == hidden
        and weight.is_contiguous()
        and x.ndim == 3
        and x.numel() > 0
        and x.stride(-1) == 1
        and 0 < hidden <= _MAX_HIDDEN_SIZE
    ):
        if exact:
            rstd = _exact_rms_rstd(x, float(eps))
            return _sf_apply_rstd_weight_op(x, rstd, weight)
        return _sf_rmsnorm_op(x, weight, float(eps))
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))).type_as(x) * weight


def sf_apply_rope_bshd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Bit-exact real-valued 3D RoPE on ``[B, S, H, D]`` matching ``causal_rope_apply``."""
    seq_len = int(cos.shape[1]) if cos.ndim == 4 else int(cos.shape[0])
    target = x[:, :seq_len]
    if target.numel() == 0:
        return x.clone()
    head_dim = int(target.shape[-1])
    half_d = head_dim // 2
    cos_ok = (
        cos.device == target.device
        and sin.device == target.device
        and cos.shape == sin.shape
        and cos.dtype == sin.dtype
        and cos.dtype in {torch.float32, torch.float64}
        and cos.numel() == seq_len * half_d
    )
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(target)
        and target.ndim == 4
        and target.stride(-1) == 1
        and head_dim >= 2
        and head_dim % 2 == 0
        and half_d <= 256
        and cos_ok
    ):
        rotated = _sf_apply_rope_bshd_op(target, cos, sin)
    else:
        x_pairs = target.to(cos.dtype).reshape(int(target.shape[0]), seq_len, int(target.shape[2]), half_d, 2)
        x0, x1 = x_pairs.unbind(-1)
        cos_v = cos.reshape(1, seq_len, 1, half_d)
        sin_v = sin.reshape(1, seq_len, 1, half_d)
        rotated = (
            torch.stack((x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v), dim=-1).flatten(-2).float().to(target.dtype)
        )
    if seq_len == int(x.shape[1]):
        return rotated
    return torch.cat((rotated, x[:, seq_len:]), dim=1)


def sf_qk_rmsnorm_rope_write_kv(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    dst_start: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    rotate_k: bool = True,
    exact: bool = False,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Single-pass strided Q/K WanRMSNorm + 3D RoPE + direct K/V cache writes.

    Accepts strided views straight from ``self.qkv(x).chunk(3, dim=-1)`` as well
    as separate contiguous ``q``, ``k``, ``v`` projections without ``.contiguous()``
    copies. When ``rotate_k=True``, writes rotated ``K`` into ``k_cache[:, dst_start:dst_end]``.
    When ``rotate_k=False`` (``kv_cache_relative_rope=True``), writes un-rotated
    normalized ``K`` into ``k_cache[:, dst_start:dst_end]`` while returning rotated ``Q``.
    Always writes ``V`` directly into ``v_cache[:, dst_start:dst_end]``.
    When ``exact=True``, computes exact PyTorch ``rstd`` values and fuses the
    scaling, weighting, FP64/FP32 3D RoPE, and KV cache writes in Triton for
    0-ULP bitwise parity with eager PyTorch.
    """
    batch, seq_len, hidden = int(q_proj.shape[0]), int(q_proj.shape[1]), int(q_proj.shape[2])
    half_d = int(head_dim) // 2
    dst_end = int(dst_start) + seq_len
    norm_pair_ok = (q_weight is None and k_weight is None) or (
        q_weight is not None
        and k_weight is not None
        and q_weight.device == q_proj.device
        and k_weight.device == q_proj.device
        and q_weight.dtype in _SUPPORTED_DTYPES
        and k_weight.dtype in _SUPPORTED_DTYPES
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and q_weight.shape == (hidden,)
        and k_weight.shape == (hidden,)
    )
    if (
        sf_fast_kernels_enabled(enabled)
        and not torch.is_grad_enabled()
        and _cuda_triton_eligible(q_proj)
        and k_proj.device == q_proj.device
        and v_proj.device == q_proj.device
        and k_cache.device == q_proj.device
        and v_cache.device == q_proj.device
        and cos.device == q_proj.device
        and sin.device == q_proj.device
        and q_proj.ndim == 3
        and k_proj.shape == q_proj.shape
        and v_proj.shape == q_proj.shape
        and q_proj.dtype == k_proj.dtype == v_proj.dtype == k_cache.dtype == v_cache.dtype
        and q_proj.numel() > 0
        and q_proj.stride(-1) == 1
        and k_proj.stride(-1) == 1
        and v_proj.stride(-1) == 1
        and k_cache.ndim == 4
        and v_cache.ndim == 4
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and int(k_cache.shape[0]) == batch
        and int(v_cache.shape[0]) == batch
        and int(k_cache.shape[2]) == int(num_heads)
        and int(k_cache.shape[3]) == int(head_dim)
        and int(v_cache.shape[2]) == int(num_heads)
        and int(v_cache.shape[3]) == int(head_dim)
        and int(dst_start) >= 0
        and dst_end <= int(k_cache.shape[1])
        and dst_end <= int(v_cache.shape[1])
        and hidden == int(num_heads) * int(head_dim)
        and 0 < hidden <= _MAX_HIDDEN_SIZE
        and head_dim >= 2
        and head_dim % 2 == 0
        and cos.shape == sin.shape
        and cos.dtype == sin.dtype
        and cos.dtype in {torch.float32, torch.float64}
        and cos.numel() == seq_len * half_d
        and norm_pair_ok
    ):
        if exact and q_weight is not None and k_weight is not None:
            q_rstd = _exact_rms_rstd(q_proj, float(eps))
            k_rstd = _exact_rms_rstd(k_proj, float(eps))
            return _sf_qk_apply_rstd_rope_write_kv_op(
                q_proj,
                k_proj,
                v_proj,
                q_rstd,
                k_rstd,
                q_weight,
                k_weight,
                cos,
                sin,
                k_cache,
                v_cache,
                int(dst_start),
                int(num_heads),
                int(head_dim),
                bool(rotate_k),
            )
        return _sf_qk_rmsnorm_rope_write_kv_op(
            q_proj,
            k_proj,
            v_proj,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            int(dst_start),
            int(num_heads),
            int(head_dim),
            float(eps),
            bool(rotate_k),
        )

    # Exact PyTorch fallback
    if q_weight is not None:
        qf = q_proj.float()
        q_norm = (
            (qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + float(eps))).type_as(q_proj) * q_weight
        ).view(batch, seq_len, int(num_heads), int(head_dim))
    else:
        q_norm = q_proj.view(batch, seq_len, int(num_heads), int(head_dim))
    if k_weight is not None:
        kf = k_proj.float()
        k_norm = (
            (kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + float(eps))).type_as(k_proj) * k_weight
        ).view(batch, seq_len, int(num_heads), int(head_dim))
    else:
        k_norm = k_proj.view(batch, seq_len, int(num_heads), int(head_dim))
    v_view = v_proj.view(batch, seq_len, int(num_heads), int(head_dim))
    q_rot = sf_apply_rope_bshd(q_norm, cos, sin, enabled=False).type_as(v_view)
    if rotate_k:
        k_rot = sf_apply_rope_bshd(k_norm, cos, sin, enabled=False).type_as(v_view)
        k_cache[:, int(dst_start) : dst_end] = k_rot
    else:
        k_cache[:, int(dst_start) : dst_end] = k_norm.type_as(k_cache)
    v_cache[:, int(dst_start) : dst_end] = v_view
    return q_rot


def sf_cached_t2v_cross_attention(
    cross: Any,
    x: torch.Tensor,
    crossattn_cache: dict[str, Any],
    *,
    exact: bool = False,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Execute cached Wan T2V text cross-attention with fused ``sf_rmsnorm`` on Q."""
    from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
        _can_use_wan_fa3_bshd,
        wan_flash_attn_3_bshd,
    )
    from worldfoundry.core.attention.sequence.varlen import flash_attention

    b, s = int(x.shape[0]), int(x.shape[1])
    n, d = int(cross.num_heads), int(cross.head_dim)
    q_raw = cross.q(x)
    norm_q = cross.norm_q
    if hasattr(norm_q, "weight") and hasattr(norm_q, "eps"):
        q = sf_rmsnorm(q_raw, norm_q.weight, eps=float(norm_q.eps), exact=exact, enabled=enabled).view(b, s, n, d)
    else:
        q = norm_q(q_raw).view(b, s, n, d)
    k = crossattn_cache["k"]
    v = crossattn_cache["v"]
    if getattr(cross, "attention_backend", None) == "flash_attention_3" and _can_use_wan_fa3_bshd(q, k, v):
        out = wan_flash_attn_3_bshd(q, k, v)
    else:
        out = flash_attention(q, k, v, k_lens=None)
    return cross.o(out.flatten(2))


__all__ = [
    "set_sf_exact_norms",
    "set_sf_fast_kernels_enabled",
    "sf_apply_rope_bshd",
    "sf_cached_t2v_cross_attention",
    "sf_exact_norms_enabled",
    "sf_fast_kernels_enabled",
    "sf_layernorm_adaln",
    "sf_qk_rmsnorm_rope_write_kv",
    "sf_residual_add_layernorm_adaln",
    "sf_residual_gate_add",
    "sf_residual_gate_add_layernorm",
    "sf_rmsnorm",
]
