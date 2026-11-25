# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for shared rotary position embeddings.

Provides fused, register-safe, strict-``int64``-indexed Triton kernels behind
``@torch.library.custom_op`` + ``.register_fake`` entries for:

- :func:`triton_complex_rotary_embedding` (``worldfoundry::complex_rope_fwd``):
  in-register FP64/FP32 complex RoPE eliminating 4x FP64 HBM materialization.
- :func:`triton_apply_rotary_pos_emb` (``worldfoundry::qk_half_rope_fwd``):
  fused Q+K half-split RoPE supporting both ``unsqueeze_dim=1`` (BHSD) and
  ``unsqueeze_dim=2`` (BSHD), GQA (``Hq != Hk``), and mixed table precision.
- :func:`triton_rope_2d_indexed` (``worldfoundry::rope_2d_indexed_fwd``):
  fused 2D axial RoPE with inline ``cos``/``sin`` table gather for both signed
  (``RotaryPositionEmbedding2D``) and unsigned (``IndexedRotaryPositionEmbedding2D``)
  position layouts.
- :func:`triton_interleaved_partial_rope` (``worldfoundry::interleaved_partial_rope_fwd``):
  fused adjacent-pair partial-slice RoPE preserving ``t_left`` and ``t_right``
  in a single HBM pass.

All Triton imports are strictly deferred until first eligible CUDA launch so
CPU-only processes (even without Triton installed) import and run cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import torch

_SUPPORTED_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_COMPLEX_DTYPES = frozenset({torch.complex64, torch.complex128})
_INT_DTYPES = frozenset({torch.int64, torch.int32, torch.int16, torch.uint8})
_TORCH_BACKENDS = frozenset({"torch", "pytorch", "native", "off", "disabled"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "disabled"})

_TRITON_SUITE: _TritonRotarySuite | None = None
_TRITON_SPEC_AVAILABLE: bool | None = None
_BF16_CAP_CACHE: dict[int, bool] = {}
_TORCH_BACKENDS_BYTES = frozenset({b"torch", b"pytorch", b"native", b"off", b"disabled"})
_FALSY_VALUES_BYTES = frozenset({b"0", b"false", b"no", b"off", b"disabled"})

# Bound lazily inside _TritonRotarySuite.__init__ so Triton's AST compiler
# resolves them from module globals without importing triton at module load.
tl: Any = None
_round_bf16: Any = None
_round_fp16: Any = None
_cw_sincos_fp32: Any = None

_assume_constant_result = getattr(
    getattr(torch, "_dynamo", None),
    "assume_constant_result",
    lambda fn: fn,
)


@_assume_constant_result
def rotary_triton_ops_enabled() -> bool:
    """Return True when shared Hopper Triton rotary kernels are enabled."""
    env_data = getattr(os.environ, "_data", None)
    if isinstance(env_data, dict):
        b_raw = env_data.get(b"WORLDFOUNDRY_KERNEL_BACKEND")
        if b_raw is not None and b_raw != b"auto":
            if isinstance(b_raw, bytes):
                if b_raw.strip().lower() in _TORCH_BACKENDS_BYTES:
                    return False
            elif str(b_raw).strip().casefold() in _TORCH_BACKENDS:
                return False
        r_raw = env_data.get(b"WORLDFOUNDRY_ROTARY_TRITON_OPS")
        if r_raw is None or r_raw == b"1":
            return True
        if isinstance(r_raw, bytes):
            return r_raw.strip().lower() not in _FALSY_VALUES_BYTES
        return str(r_raw).strip().casefold() not in _FALSY_VALUES
    backend = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold()
    if backend in _TORCH_BACKENDS:
        return False
    raw = os.getenv("WORLDFOUNDRY_ROTARY_TRITON_OPS", "1").strip().casefold()
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
    """Common CUDA, backend, Triton availability, autograd, and device check."""
    if not tensors or not rotary_triton_ops_enabled():
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


def _get_triton_suite() -> _TritonRotarySuite:
    global _TRITON_SUITE
    if _TRITON_SUITE is None:
        _TRITON_SUITE = _TritonRotarySuite()
    return _TRITON_SUITE


class _TritonRotarySuite:
    """Lazily-compiled Hopper SM90 Triton kernels for shared core rotary ops."""

    def __init__(self) -> None:
        global tl, _round_bf16, _round_fp16, _cw_sincos_fp32
        try:
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="shared-rotary-triton")
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

        # Zero-spill FP32 sincos: 3-part Cody-Waite mod (pi/2) + minimax poly.
        # Avoids NVVM libdevice Payne-Hanek .local stack depot (n_spills == 0)
        # while preserving < 1 FP32 ULP accuracy across [-1e6, 1e6].
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
        _cw_sincos_fp32 = _cw_sincos_fp32_fn

        # ──────────────────────────────────────────────────────────────────
        # 1. Complex RoPE: in-register FP64/FP32 complex multiply
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _complex_rope_fwd_kernel(
            val_ptr,
            freq_re_ptr,
            freq_im_ptr,
            out_ptr,
            stride_vb: tl.int64,
            stride_vs: tl.int64,
            stride_fr_b: tl.int64,
            stride_fr_s: tl.int64,
            stride_fr_h: tl.int64,
            stride_fr_d: tl.int64,
            stride_fi_b: tl.int64,
            stride_fi_s: tl.int64,
            stride_fi_h: tl.int64,
            stride_fi_d: tl.int64,
            stride_ob: tl.int64,
            stride_os: tl.int64,
            S: tl.int64,
            H: tl.int64,
            D: tl.int64,
            D_HALF: tl.int64,
            USE_FP64: tl.constexpr,
            FREQ_PER_HEAD: tl.constexpr,
            BLOCK_H: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
            BLOCK_D_HALF: tl.constexpr,
        ):
            pid_bs = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)
            pid_d = tl.program_id(2).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            s_i64 = z64 + S
            h_i64 = z64 + H
            d_i64 = z64 + D
            d_half_i64 = z64 + D_HALF

            b_idx = pid_bs // s_i64
            s_idx = pid_bs - b_idx * s_i64

            d_base = pid_d * BLOCK_D_HALF
            d_half = d_base + tl.arange(0, BLOCK_D_HALF).to(tl.int64)
            d_half_mask = d_half < d_half_i64

            d_full = 2 * d_base + tl.arange(0, 2 * BLOCK_D_HALF).to(tl.int64)
            d_full_mask = d_full < d_i64

            val_row = val_ptr + b_idx * stride_vb + s_idx * stride_vs
            out_row = out_ptr + b_idx * stride_ob + s_idx * stride_os
            fr_base = freq_re_ptr + b_idx * stride_fr_b + s_idx * stride_fr_s
            fi_base = freq_im_ptr + b_idx * stride_fi_b + s_idx * stride_fi_s

            if not FREQ_PER_HEAD:
                f_re_1d = tl.load(fr_base + d_half * stride_fr_d, mask=d_half_mask, other=0.0)
                f_im_1d = tl.load(fi_base + d_half * stride_fi_d, mask=d_half_mask, other=0.0)
                if USE_FP64:
                    f_re_shared = f_re_1d.to(tl.float64)[None, :]
                    f_im_shared = f_im_1d.to(tl.float64)[None, :]
                else:
                    f_re_shared = f_re_1d.to(tl.float32)[None, :]
                    f_im_shared = f_im_1d.to(tl.float32)[None, :]

            h_group_base = pid_hg * (BLOCK_H * HEAD_ITERS)
            h_local = tl.arange(0, BLOCK_H).to(tl.int64)

            for h_step in range(HEAD_ITERS):
                h_idx = h_group_base + h_step * BLOCK_H + h_local
                h_mask = h_idx < h_i64

                io_off = h_idx[:, None] * d_i64 + d_full[None, :]
                io_mask = h_mask[:, None] & d_full_mask[None, :]
                x_flat = tl.load(val_row + io_off, mask=io_mask, other=0.0)
                x_pairs = tl.reshape(x_flat, (BLOCK_H, BLOCK_D_HALF, 2))
                v_re, v_im = tl.split(x_pairs)

                if FREQ_PER_HEAD:
                    f_mask = h_mask[:, None] & d_half_mask[None, :]
                    f_re_raw = tl.load(
                        fr_base + h_idx[:, None] * stride_fr_h + d_half[None, :] * stride_fr_d,
                        mask=f_mask,
                        other=0.0,
                    )
                    f_im_raw = tl.load(
                        fi_base + h_idx[:, None] * stride_fi_h + d_half[None, :] * stride_fi_d,
                        mask=f_mask,
                        other=0.0,
                    )
                    if USE_FP64:
                        f_re = f_re_raw.to(tl.float64)
                        f_im = f_im_raw.to(tl.float64)
                    else:
                        f_re = f_re_raw.to(tl.float32)
                        f_im = f_im_raw.to(tl.float32)
                else:
                    f_re = f_re_shared
                    f_im = f_im_shared

                if USE_FP64:
                    v_re_c = v_re.to(tl.float64)
                    v_im_c = v_im.to(tl.float64)
                    out_re = v_re_c * f_re - v_im_c * f_im
                    out_im = v_re_c * f_im + v_im_c * f_re
                    if x_flat.dtype == tl.bfloat16 or x_flat.dtype == tl.float16:
                        # Match PyTorch float64 -> float32 -> bf16/fp16 cast semantics bit-for-bit.
                        out_re_cast = out_re.to(tl.float32).to(x_flat.dtype)
                        out_im_cast = out_im.to(tl.float32).to(x_flat.dtype)
                    else:
                        out_re_cast = out_re.to(x_flat.dtype)
                        out_im_cast = out_im.to(x_flat.dtype)
                else:
                    v_re_c = v_re.to(tl.float32)
                    v_im_c = v_im.to(tl.float32)
                    out_re = v_re_c * f_re - v_im_c * f_im
                    out_im = v_re_c * f_im + v_im_c * f_re
                    out_re_cast = out_re.to(x_flat.dtype)
                    out_im_cast = out_im.to(x_flat.dtype)

                out_pairs = tl.join(out_re_cast, out_im_cast)
                out_flat = tl.reshape(out_pairs, (BLOCK_H, 2 * BLOCK_D_HALF))
                tl.store(out_row + io_off, out_flat, mask=io_mask)

        self.complex_rope_fwd_kernel = _complex_rope_fwd_kernel

        # ──────────────────────────────────────────────────────────────────
        # 2. QK Half-Split RoPE: single fused launch for Q + K (GQA, BHSD/BSHD)
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _qk_half_rope_fwd_kernel(
            q_ptr,
            k_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_out_ptr,
            stride_qb: tl.int64,
            stride_qh: tl.int64,
            stride_qs: tl.int64,
            stride_kb: tl.int64,
            stride_kh: tl.int64,
            stride_ks: tl.int64,
            stride_cb: tl.int64,
            stride_cs: tl.int64,
            stride_sb: tl.int64,
            stride_ss: tl.int64,
            stride_qob: tl.int64,
            stride_qoh: tl.int64,
            stride_qos: tl.int64,
            stride_kob: tl.int64,
            stride_koh: tl.int64,
            stride_kos: tl.int64,
            S: tl.int64,
            Hq: tl.int64,
            Hk: tl.int64,
            D_HALF: tl.int64,
            Q_HEAD_BLOCKS: tl.int64,
            ROUND_MODE: tl.constexpr,
            BLOCK_S: tl.constexpr,
            BLOCK_H: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
            BLOCK_D_HALF: tl.constexpr,
        ):
            pid_bs = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)
            pid_d = tl.program_id(2).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            s_i64 = z64 + S
            d_half_i64 = z64 + D_HALF
            q_head_blocks_i64 = z64 + Q_HEAD_BLOCKS
            s_blocks = (s_i64 + BLOCK_S - 1) // BLOCK_S
            b_idx = pid_bs // s_blocks
            s_blk = pid_bs - b_idx * s_blocks

            s_off = s_blk * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
            s_mask = s_off < s_i64

            d_off = pid_d * BLOCK_D_HALF + tl.arange(0, BLOCK_D_HALF).to(tl.int64)
            d_mask = d_off < d_half_i64
            sd_mask = s_mask[:, None] & d_mask[None, :]

            # Load cos/sin halves ONCE for the token tile (handles arbitrary non-duplicated cos/sin too).
            c_off0 = b_idx * stride_cb + s_off[:, None] * stride_cs + d_off[None, :]
            c_off1 = c_off0 + d_half_i64
            s_off0 = b_idx * stride_sb + s_off[:, None] * stride_ss + d_off[None, :]
            s_off1 = s_off0 + d_half_i64

            c0 = tl.load(cos_ptr + c_off0, mask=sd_mask, other=0.0).to(tl.float32)
            c1 = tl.load(cos_ptr + c_off1, mask=sd_mask, other=0.0).to(tl.float32)
            sn0 = tl.load(sin_ptr + s_off0, mask=sd_mask, other=0.0).to(tl.float32)
            sn1 = tl.load(sin_ptr + s_off1, mask=sd_mask, other=0.0).to(tl.float32)

            is_q = pid_hg < q_head_blocks_i64
            if is_q:
                x_base = q_ptr + b_idx * stride_qb
                o_base = q_out_ptr + b_idx * stride_qob
                st_h = stride_qh
                st_s = stride_qs
                st_oh = stride_qoh
                st_os = stride_qos
                H_lim = z64 + Hq
                h_group_base = pid_hg * (BLOCK_H * HEAD_ITERS)
            else:
                x_base = k_ptr + b_idx * stride_kb
                o_base = k_out_ptr + b_idx * stride_kob
                st_h = stride_kh
                st_s = stride_ks
                st_oh = stride_koh
                st_os = stride_kos
                H_lim = z64 + Hk
                h_group_base = (pid_hg - q_head_blocks_i64) * (BLOCK_H * HEAD_ITERS)

            if BLOCK_H == 1:
                for h_step in tl.static_range(HEAD_ITERS):
                    h_idx = h_group_base + h_step
                    if h_idx < H_lim:
                        in0_off = h_idx * st_h + s_off[:, None] * st_s + d_off[None, :]
                        in1_off = in0_off + d_half_i64
                        x0 = tl.load(x_base + in0_off, mask=sd_mask, other=0.0).to(tl.float32)
                        x1 = tl.load(x_base + in1_off, mask=sd_mask, other=0.0).to(tl.float32)

                        if ROUND_MODE == 1:
                            o0 = (_round_bf16(x0 * c0) + _round_bf16((-x1) * sn0)).to(tl.bfloat16)
                            o1 = (_round_bf16(x1 * c1) + _round_bf16(x0 * sn1)).to(tl.bfloat16)
                        elif ROUND_MODE == 2:
                            o0 = (_round_fp16(x0 * c0) + _round_fp16((-x1) * sn0)).to(tl.float16)
                            o1 = (_round_fp16(x1 * c1) + _round_fp16(x0 * sn1)).to(tl.float16)
                        else:
                            o0 = x0 * c0 + (-x1) * sn0
                            o1 = x1 * c1 + x0 * sn1

                        out0_off = h_idx * st_oh + s_off[:, None] * st_os + d_off[None, :]
                        out1_off = out0_off + d_half_i64
                        tl.store(o_base + out0_off, o0, mask=sd_mask)
                        tl.store(o_base + out1_off, o1, mask=sd_mask)
            else:
                # BLOCK_S == 1, coalesced across BLOCK_H heads (BSHD contiguous layout)
                s_idx = s_blk
                h_local = tl.arange(0, BLOCK_H).to(tl.int64)
                c0_b = tl.reshape(c0, (1, BLOCK_D_HALF))
                c1_b = tl.reshape(c1, (1, BLOCK_D_HALF))
                sn0_b = tl.reshape(sn0, (1, BLOCK_D_HALF))
                sn1_b = tl.reshape(sn1, (1, BLOCK_D_HALF))
                if s_idx < s_i64:
                    for h_step in tl.static_range(HEAD_ITERS):
                        h_idx = h_group_base + h_step * BLOCK_H + h_local
                        hd_mask = (h_idx[:, None] < H_lim) & d_mask[None, :]
                        in0_off = h_idx[:, None] * st_h + s_idx * st_s + d_off[None, :]
                        in1_off = in0_off + d_half_i64
                        x0 = tl.load(x_base + in0_off, mask=hd_mask, other=0.0).to(tl.float32)
                        x1 = tl.load(x_base + in1_off, mask=hd_mask, other=0.0).to(tl.float32)

                        if ROUND_MODE == 1:
                            o0 = (_round_bf16(x0 * c0_b) + _round_bf16((-x1) * sn0_b)).to(tl.bfloat16)
                            o1 = (_round_bf16(x1 * c1_b) + _round_bf16(x0 * sn1_b)).to(tl.bfloat16)
                        elif ROUND_MODE == 2:
                            o0 = (_round_fp16(x0 * c0_b) + _round_fp16((-x1) * sn0_b)).to(tl.float16)
                            o1 = (_round_fp16(x1 * c1_b) + _round_fp16(x0 * sn1_b)).to(tl.float16)
                        else:
                            o0 = x0 * c0_b + (-x1) * sn0_b
                            o1 = x1 * c1_b + x0 * sn1_b

                        out0_off = h_idx[:, None] * st_oh + s_idx * st_os + d_off[None, :]
                        out1_off = out0_off + d_half_i64
                        tl.store(o_base + out0_off, o0, mask=hd_mask)
                        tl.store(o_base + out1_off, o1, mask=hd_mask)

        self.qk_half_rope_fwd_kernel = _qk_half_rope_fwd_kernel

        # ──────────────────────────────────────────────────────────────────
        # 3. 2D Indexed Axial RoPE (signed RotaryPositionEmbedding2D + unsigned Indexed2D)
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _rope_2d_indexed_contig_kernel(
            tok_ptr,
            pos_ptr,
            cos_ptr,
            sin_ptr,
            out_ptr,
            stride_tb: tl.int64,
            stride_th: tl.int64,
            stride_tn: tl.int64,
            stride_pb: tl.int64,
            stride_pn: tl.int64,
            stride_p2: tl.int64,
            stride_cm: tl.int64,
            stride_sm: tl.int64,
            stride_ob: tl.int64,
            stride_oh: tl.int64,
            stride_on: tl.int64,
            N: tl.int64,
            H: tl.int64,
            TABLE_LEN: tl.int64,
            D_QUART: tl.constexpr,
            SIGNED_POSITIONS: tl.constexpr,
            ROUND_MODE: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
        ):
            pid_bn = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            h_i64 = z64 + H
            tlen_i64 = z64 + TABLE_LEN

            n_blocks = (n_i64 + BLOCK_N - 1) // BLOCK_N
            b_idx = pid_bn // n_blocks
            n_blk = pid_bn - b_idx * n_blocks

            n_off = n_blk * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            n_mask = n_off < n_i64

            pos_base = pos_ptr + b_idx * stride_pb + n_off * stride_pn
            py = tl.load(pos_base, mask=n_mask, other=0).to(tl.int64)
            px = tl.load(pos_base + stride_p2, mask=n_mask, other=0).to(tl.int64)

            if SIGNED_POSITIONS:
                abs_py = tl.where(py < 0, -py, py)
                abs_px = tl.where(px < 0, -px, px)
                sign_py = tl.where(py > 0, 1.0, tl.where(py < 0, -1.0, 0.0)).to(tl.float32)[:, None]
                sign_px = tl.where(px > 0, 1.0, tl.where(px < 0, -1.0, 0.0)).to(tl.float32)[:, None]
            else:
                abs_py = py
                abs_px = px

            h_off = tl.arange(0, 2 * D_QUART).to(tl.int64)
            ymask = n_mask[:, None] & (abs_py[:, None] >= 0) & (abs_py[:, None] < tlen_i64)
            xmask = n_mask[:, None] & (abs_px[:, None] >= 0) & (abs_px[:, None] < tlen_i64)

            cy = tl.load(cos_ptr + abs_py[:, None] * stride_cm + h_off[None, :], mask=ymask, other=0.0).to(tl.float32)
            sy = tl.load(sin_ptr + abs_py[:, None] * stride_sm + h_off[None, :], mask=ymask, other=0.0).to(tl.float32)
            cx = tl.load(cos_ptr + abs_px[:, None] * stride_cm + h_off[None, :], mask=xmask, other=0.0).to(tl.float32)
            sx = tl.load(sin_ptr + abs_px[:, None] * stride_sm + h_off[None, :], mask=xmask, other=0.0).to(tl.float32)
            if SIGNED_POSITIONS:
                sy = sy * sign_py
                sx = sx * sign_px

            cy0, cy1 = tl.split(tl.permute(tl.reshape(cy, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))
            sy0, sy1 = tl.split(tl.permute(tl.reshape(sy, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))
            cx0, cx1 = tl.split(tl.permute(tl.reshape(cx, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))
            sx0, sx1 = tl.split(tl.permute(tl.reshape(sx, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))

            d_full = tl.arange(0, 4 * D_QUART).to(tl.int64)
            tok_b = tok_ptr + b_idx * stride_tb
            out_b = out_ptr + b_idx * stride_ob
            h_base = pid_hg * HEAD_ITERS

            for h_step in range(HEAD_ITERS):
                h_idx = h_base + h_step
                if h_idx < h_i64:
                    row_in = tok_b + h_idx * stride_th + n_off[:, None] * stride_tn + d_full[None, :]
                    full = tl.load(row_in, mask=n_mask[:, None], other=0.0)
                    yh, xh = tl.split(tl.permute(tl.reshape(full, (BLOCK_N, 2, 2 * D_QUART)), (0, 2, 1)))
                    y0, y1 = tl.split(tl.permute(tl.reshape(yh, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))
                    x0, x1 = tl.split(tl.permute(tl.reshape(xh, (BLOCK_N, 2, D_QUART)), (0, 2, 1)))

                    y0f = y0.to(tl.float32)
                    y1f = y1.to(tl.float32)
                    x0f = x0.to(tl.float32)
                    x1f = x1.to(tl.float32)

                    if ROUND_MODE == 1:
                        oy0 = (_round_bf16(y0f * cy0) + _round_bf16((-y1f) * sy0)).to(tl.bfloat16)
                        oy1 = (_round_bf16(y1f * cy1) + _round_bf16(y0f * sy1)).to(tl.bfloat16)
                        ox0 = (_round_bf16(x0f * cx0) + _round_bf16((-x1f) * sx0)).to(tl.bfloat16)
                        ox1 = (_round_bf16(x1f * cx1) + _round_bf16(x0f * sx1)).to(tl.bfloat16)
                    elif ROUND_MODE == 2:
                        oy0 = (_round_fp16(y0f * cy0) + _round_fp16((-y1f) * sy0)).to(tl.float16)
                        oy1 = (_round_fp16(y1f * cy1) + _round_fp16(y0f * sy1)).to(tl.float16)
                        ox0 = (_round_fp16(x0f * cx0) + _round_fp16((-x1f) * sx0)).to(tl.float16)
                        ox1 = (_round_fp16(x1f * cx1) + _round_fp16(x0f * sx1)).to(tl.float16)
                    else:
                        oy0 = y0f * cy0 + (-y1f) * sy0
                        oy1 = y1f * cy1 + y0f * sy1
                        ox0 = x0f * cx0 + (-x1f) * sx0
                        ox1 = x1f * cx1 + x0f * sx1

                    oy = tl.reshape(tl.permute(tl.join(oy0, oy1), (0, 2, 1)), (BLOCK_N, 2 * D_QUART))
                    ox = tl.reshape(tl.permute(tl.join(ox0, ox1), (0, 2, 1)), (BLOCK_N, 2 * D_QUART))
                    ofull = tl.reshape(tl.permute(tl.join(oy, ox), (0, 2, 1)), (BLOCK_N, 4 * D_QUART))
                    row_out = out_b + h_idx * stride_oh + n_off[:, None] * stride_on + d_full[None, :]
                    tl.store(row_out, ofull, mask=n_mask[:, None])

        self.rope_2d_indexed_contig_kernel = _rope_2d_indexed_contig_kernel

        @triton.jit
        def _rope_2d_indexed_fwd_kernel(
            tok_ptr,
            pos_ptr,
            cos_ptr,
            sin_ptr,
            out_ptr,
            stride_tb: tl.int64,
            stride_th: tl.int64,
            stride_tn: tl.int64,
            stride_pb: tl.int64,
            stride_pn: tl.int64,
            stride_p2: tl.int64,
            stride_cm: tl.int64,
            stride_sm: tl.int64,
            stride_ob: tl.int64,
            stride_oh: tl.int64,
            stride_on: tl.int64,
            N: tl.int64,
            H: tl.int64,
            D: tl.int64,
            TABLE_LEN: tl.int64,
            SIGNED_POSITIONS: tl.constexpr,
            ROUND_MODE: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_bn = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)
            pid_d = tl.program_id(2).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            n_i64 = z64 + N
            h_i64 = z64 + H
            d_i64 = z64 + D
            dh_i64 = d_i64 // 2
            dq_i64 = d_i64 // 4
            tlen_i64 = z64 + TABLE_LEN

            n_blocks = (n_i64 + BLOCK_N - 1) // BLOCK_N
            b_idx = pid_bn // n_blocks
            n_blk = pid_bn - b_idx * n_blocks

            n_off = n_blk * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            n_mask = n_off < n_i64

            d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
            d_mask = d_off < d_i64
            nd_mask = n_mask[:, None] & d_mask[None, :]

            is_x = d_off >= dh_i64
            axis_d = tl.where(is_x, d_off - dh_i64, d_off)
            is_second = axis_d >= dq_i64
            partner_d = tl.where(is_second, d_off - dq_i64, d_off + dq_i64)

            pos_base = pos_ptr + b_idx * stride_pb + n_off * stride_pn
            py = tl.load(pos_base, mask=n_mask, other=0).to(tl.int64)
            px = tl.load(pos_base + stride_p2, mask=n_mask, other=0).to(tl.int64)

            if SIGNED_POSITIONS:
                abs_py = tl.where(py < 0, -py, py)
                abs_px = tl.where(px < 0, -px, px)
                sign_py = tl.where(py > 0, 1.0, tl.where(py < 0, -1.0, 0.0)).to(tl.float32)
                sign_px = tl.where(px > 0, 1.0, tl.where(px < 0, -1.0, 0.0)).to(tl.float32)
                pos_row = tl.where(is_x[None, :], abs_px[:, None], abs_py[:, None])
                pos_sign = tl.where(is_x[None, :], sign_px[:, None], sign_py[:, None])
            else:
                pos_row = tl.where(is_x[None, :], px[:, None], py[:, None])

            tbl_mask = nd_mask & (pos_row >= 0) & (pos_row < tlen_i64)
            c_val = tl.load(cos_ptr + pos_row * stride_cm + axis_d[None, :], mask=tbl_mask, other=0.0).to(tl.float32)
            s_val = tl.load(sin_ptr + pos_row * stride_sm + axis_d[None, :], mask=tbl_mask, other=0.0).to(tl.float32)
            if SIGNED_POSITIONS:
                s_val = s_val * pos_sign
            s_signed = tl.where(is_second[None, :], s_val, -s_val)

            tok_b = tok_ptr + b_idx * stride_tb
            out_b = out_ptr + b_idx * stride_ob
            h_base = pid_hg * HEAD_ITERS

            for h_step in range(HEAD_ITERS):
                h_idx = h_base + h_step
                if h_idx < h_i64:
                    in_hn = tok_b + h_idx * stride_th + n_off[:, None] * stride_tn
                    out_hn = out_b + h_idx * stride_oh + n_off[:, None] * stride_on

                    x = tl.load(in_hn + d_off[None, :], mask=nd_mask, other=0.0).to(tl.float32)
                    xp = tl.load(in_hn + partner_d[None, :], mask=nd_mask, other=0.0).to(tl.float32)

                    if ROUND_MODE == 1:
                        out_val = (_round_bf16(x * c_val) + _round_bf16(xp * s_signed)).to(tl.bfloat16)
                    elif ROUND_MODE == 2:
                        out_val = (_round_fp16(x * c_val) + _round_fp16(xp * s_signed)).to(tl.float16)
                    else:
                        out_val = x * c_val + xp * s_signed

                    tl.store(out_hn + d_off[None, :], out_val, mask=nd_mask)

        self.rope_2d_indexed_fwd_kernel = _rope_2d_indexed_fwd_kernel

        # ──────────────────────────────────────────────────────────────────
        # 4. Interleaved Partial-Slice RoPE: fused t_left + rot(t_mid) + t_right
        # ──────────────────────────────────────────────────────────────────
        @triton.jit
        def _interleaved_even_start_kernel(
            freqs_ptr,
            t_ptr,
            out_ptr,
            stride_fb: tl.int64,
            stride_fh: tl.int64,
            stride_fs: tl.int64,
            stride_tb: tl.int64,
            stride_th: tl.int64,
            stride_ts: tl.int64,
            stride_ob: tl.int64,
            stride_oh: tl.int64,
            stride_os: tl.int64,
            S: tl.int64,
            H: tl.int64,
            D: tl.int64,
            START_IDX: tl.int64,
            ROT_DIM: tl.int64,
            SCALE: tl.float32,
            FREQ_PER_HEAD: tl.constexpr,
            BLOCK_S: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
            BLOCK_D_HALF: tl.constexpr,
        ):
            pid_bs = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)
            pid_d = tl.program_id(2).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            s_i64 = z64 + S
            h_i64 = z64 + H
            d_i64 = z64 + D
            start_i64 = z64 + START_IDX
            rot_i64 = z64 + ROT_DIM
            start_half = start_i64 // 2
            rot_half = rot_i64 // 2
            scale_f32 = tl.zeros([], dtype=tl.float32) + SCALE

            s_blocks = (s_i64 + BLOCK_S - 1) // BLOCK_S
            b_idx = pid_bs // s_blocks
            s_blk = pid_bs - b_idx * s_blocks
            s_off = s_blk * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
            s_mask = s_off < s_i64

            d_base = pid_d * (2 * BLOCK_D_HALF)
            d_full = d_base + tl.arange(0, 2 * BLOCK_D_HALF).to(tl.int64)
            d_mask = d_full < d_i64
            sd_mask = s_mask[:, None] & d_mask[None, :]

            r_full = d_full - start_i64
            in_rot_full = (r_full >= 0) & (r_full < rot_i64)
            f_safe = tl.where(in_rot_full, r_full, 0)
            f_mask = s_mask[:, None] & in_rot_full[None, :]

            p_off = pid_d * BLOCK_D_HALF + tl.arange(0, BLOCK_D_HALF).to(tl.int64)
            in_rot_pair = (p_off >= start_half) & (p_off < start_half + rot_half)

            t_b = t_ptr + b_idx * stride_tb
            o_b = out_ptr + b_idx * stride_ob
            f_b = freqs_ptr + b_idx * stride_fb

            if not FREQ_PER_HEAD:
                f_flat = tl.load(
                    f_b + s_off[:, None] * stride_fs + f_safe[None, :],
                    mask=f_mask,
                    other=0.0,
                ).to(tl.float32)
                fe, fo = tl.split(tl.reshape(f_flat, (BLOCK_S, BLOCK_D_HALF, 2)))
                ce_shared, se_shared = _cw_sincos_fp32(fe)
                co_shared, so_shared = _cw_sincos_fp32(fo)

            h_base = pid_hg * HEAD_ITERS
            for h_step in range(HEAD_ITERS):
                h_idx = h_base + h_step
                if h_idx < h_i64:
                    in_hs = t_b + h_idx * stride_th + s_off[:, None] * stride_ts
                    out_hs = o_b + h_idx * stride_oh + s_off[:, None] * stride_os
                    x_flat = tl.load(in_hs + d_full[None, :], mask=sd_mask, other=0.0)
                    x0, x1 = tl.split(tl.reshape(x_flat, (BLOCK_S, BLOCK_D_HALF, 2)))
                    x0f = x0.to(tl.float32)
                    x1f = x1.to(tl.float32)

                    if FREQ_PER_HEAD:
                        f_flat = tl.load(
                            f_b + h_idx * stride_fh + s_off[:, None] * stride_fs + f_safe[None, :],
                            mask=f_mask,
                            other=0.0,
                        ).to(tl.float32)
                        fe, fo = tl.split(tl.reshape(f_flat, (BLOCK_S, BLOCK_D_HALF, 2)))
                        ce, se = _cw_sincos_fp32(fe)
                        co, so = _cw_sincos_fp32(fo)
                    else:
                        ce = ce_shared
                        se = se_shared
                        co = co_shared
                        so = so_shared

                    oe = ((x0f * ce) * scale_f32 + ((-x1f) * se) * scale_f32).to(x_flat.dtype)
                    oo = ((x1f * co) * scale_f32 + (x0f * so) * scale_f32).to(x_flat.dtype)
                    o0 = tl.where(in_rot_pair[None, :], oe, x0)
                    o1 = tl.where(in_rot_pair[None, :], oo, x1)
                    out_flat = tl.reshape(tl.join(o0, o1), (BLOCK_S, 2 * BLOCK_D_HALF))
                    tl.store(out_hs + d_full[None, :], out_flat, mask=sd_mask)

        self.interleaved_even_start_kernel = _interleaved_even_start_kernel

        @triton.jit
        def _interleaved_partial_rope_fwd_kernel(
            freqs_ptr,
            t_ptr,
            out_ptr,
            stride_fb: tl.int64,
            stride_fh: tl.int64,
            stride_fs: tl.int64,
            stride_tb: tl.int64,
            stride_th: tl.int64,
            stride_ts: tl.int64,
            stride_ob: tl.int64,
            stride_oh: tl.int64,
            stride_os: tl.int64,
            S: tl.int64,
            H: tl.int64,
            D: tl.int64,
            START_IDX: tl.int64,
            ROT_DIM: tl.int64,
            SCALE: tl.float32,
            FREQ_PER_HEAD: tl.constexpr,
            BLOCK_S: tl.constexpr,
            HEAD_ITERS: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_bs = tl.program_id(0).to(tl.int64)
            pid_hg = tl.program_id(1).to(tl.int64)
            pid_d = tl.program_id(2).to(tl.int64)

            z64 = tl.zeros([], dtype=tl.int64)
            s_i64 = z64 + S
            h_i64 = z64 + H
            d_i64 = z64 + D
            start_i64 = z64 + START_IDX
            rot_i64 = z64 + ROT_DIM
            scale_f32 = tl.zeros([], dtype=tl.float32) + SCALE

            s_blocks = (s_i64 + BLOCK_S - 1) // BLOCK_S
            b_idx = pid_bs // s_blocks
            s_blk = pid_bs - b_idx * s_blocks

            s_off = s_blk * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
            s_mask = s_off < s_i64

            d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
            d_mask = d_off < d_i64
            sd_mask = s_mask[:, None] & d_mask[None, :]

            r_idx = d_off - start_i64
            in_rot = (r_idx >= 0) & (r_idx < rot_i64)
            is_even = (r_idx & 1) == 0
            f_safe = tl.where(in_rot, r_idx, 0)
            partner_off = tl.where(in_rot, tl.where(is_even, d_off + 1, d_off - 1), d_off)

            t_b = t_ptr + b_idx * stride_tb
            o_b = out_ptr + b_idx * stride_ob
            f_b = freqs_ptr + b_idx * stride_fb

            if not FREQ_PER_HEAD:
                f_mask = s_mask[:, None] & in_rot[None, :]
                f_val = tl.load(
                    f_b + s_off[:, None] * stride_fs + f_safe[None, :],
                    mask=f_mask,
                    other=0.0,
                ).to(tl.float32)
                cos_shared, sin_raw = _cw_sincos_fp32(f_val)
                sin_shared = tl.where(is_even[None, :], -sin_raw, sin_raw)

            h_base = pid_hg * HEAD_ITERS
            for h_step in range(HEAD_ITERS):
                h_idx = h_base + h_step
                if h_idx < h_i64:
                    in_hs = t_b + h_idx * stride_th + s_off[:, None] * stride_ts
                    out_hs = o_b + h_idx * stride_oh + s_off[:, None] * stride_os

                    x = tl.load(in_hs + d_off[None, :], mask=sd_mask, other=0.0)
                    xp = tl.load(in_hs + partner_off[None, :], mask=sd_mask, other=0.0)

                    if FREQ_PER_HEAD:
                        f_mask = s_mask[:, None] & in_rot[None, :]
                        f_val = tl.load(
                            f_b + h_idx * stride_fh + s_off[:, None] * stride_fs + f_safe[None, :],
                            mask=f_mask,
                            other=0.0,
                        ).to(tl.float32)
                        c_v, s_raw = _cw_sincos_fp32(f_val)
                        s_v = tl.where(is_even[None, :], -s_raw, s_raw)
                    else:
                        c_v = cos_shared
                        s_v = sin_shared

                    rot_val = ((x.to(tl.float32) * c_v) * scale_f32 + (xp.to(tl.float32) * s_v) * scale_f32).to(x.dtype)
                    out_val = tl.where(in_rot[None, :], rot_val, x)
                    tl.store(out_hs + d_off[None, :], out_val, mask=sd_mask)

        self.interleaved_partial_rope_fwd_kernel = _interleaved_partial_rope_fwd_kernel


# ──────────────────────────────────────────────────────────────────────────
# Custom Ops + Fake Registrations (compile & CUDA Graph safe)
# ──────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("worldfoundry::complex_rope_fwd", mutates_args=())
def _complex_rope_fwd_op(
    value: torch.Tensor,
    freq_re: torch.Tensor,
    freq_im: torch.Tensor,
    num_heads: int,
    use_fp64: bool,
) -> torch.Tensor:
    b, s, hd = value.shape
    h = int(num_heads)
    d = hd // h
    d_half = d // 2
    out = torch.empty((b, s, hd), device=value.device, dtype=value.dtype)
    if b == 0 or s == 0:
        return out

    suite = _get_triton_suite()
    stride_fr_b = freq_re.stride(0) if freq_re.shape[0] > 1 else 0
    stride_fr_s = freq_re.stride(1) if freq_re.shape[1] > 1 else 0
    stride_fr_h = freq_re.stride(2) if freq_re.shape[2] > 1 else 0
    stride_fr_d = freq_re.stride(3)

    stride_fi_b = freq_im.stride(0) if freq_im.shape[0] > 1 else 0
    stride_fi_s = freq_im.stride(1) if freq_im.shape[1] > 1 else 0
    stride_fi_h = freq_im.stride(2) if freq_im.shape[2] > 1 else 0
    stride_fi_d = freq_im.stride(3)

    freq_per_head = (stride_fr_h != 0) or (stride_fi_h != 0)
    block_d_half = min(256, max(2, _next_pow2(d_half)))
    if block_d_half <= 128 and h >= 4:
        block_h = 4
        max_iters = _cdiv(h, block_h)
        if not freq_per_head and b * s >= 512:
            head_iters = min(max_iters, 5)
        elif not freq_per_head and b * s >= 64:
            head_iters = min(max_iters, 2)
        else:
            head_iters = 1
        num_warps = 4
    elif block_d_half <= 128 and h >= 2:
        block_h = 2
        head_iters = 1
        num_warps = 2 if not use_fp64 else 4
    else:
        block_h = 1
        head_iters = 1
        num_warps = 4 if (block_d_half >= 128 or use_fp64) else 2

    grid = (
        b * s,
        _cdiv(h, block_h * head_iters),
        _cdiv(d_half, block_d_half),
    )
    suite.complex_rope_fwd_kernel[grid](
        value,
        freq_re,
        freq_im,
        out,
        value.stride(0),
        value.stride(1),
        stride_fr_b,
        stride_fr_s,
        stride_fr_h,
        stride_fr_d,
        stride_fi_b,
        stride_fi_s,
        stride_fi_h,
        stride_fi_d,
        out.stride(0),
        out.stride(1),
        s,
        h,
        d,
        d_half,
        USE_FP64=bool(use_fp64),
        FREQ_PER_HEAD=freq_per_head,
        BLOCK_H=block_h,
        HEAD_ITERS=head_iters,
        BLOCK_D_HALF=block_d_half,
        num_warps=num_warps,
    )
    return out


@_complex_rope_fwd_op.register_fake
def _complex_rope_fwd_fake(
    value: torch.Tensor,
    freq_re: torch.Tensor,
    freq_im: torch.Tensor,
    num_heads: int,
    use_fp64: bool,
) -> torch.Tensor:
    del freq_re, freq_im, num_heads, use_fp64
    return torch.empty(value.shape, device=value.device, dtype=value.dtype)


@torch.library.custom_op("worldfoundry::qk_half_rope_fwd", mutates_args=())
def _qk_half_rope_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dtype = torch.promote_types(q.dtype, torch.promote_types(cos.dtype, sin.dtype))
    q_out = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=out_dtype)
    if q.numel() == 0 and k.numel() == 0:
        return q_out, k_out

    if unsqueeze_dim == 1:
        b, hq, s, d = q.shape
        hk = k.shape[1]
        st_qb, st_qh, st_qs = q.stride(0), q.stride(1), q.stride(2)
        st_kb, st_kh, st_ks = k.stride(0), k.stride(1), k.stride(2)
        st_qob, st_qoh, st_qos = q_out.stride(0), q_out.stride(1), q_out.stride(2)
        st_kob, st_koh, st_kos = k_out.stride(0), k_out.stride(1), k_out.stride(2)
    else:
        b, s, hq, d = q.shape
        hk = k.shape[2]
        st_qb, st_qh, st_qs = q.stride(0), q.stride(2), q.stride(1)
        st_kb, st_kh, st_ks = k.stride(0), k.stride(2), k.stride(1)
        st_qob, st_qoh, st_qos = q_out.stride(0), q_out.stride(2), q_out.stride(1)
        st_kob, st_koh, st_kos = k_out.stride(0), k_out.stride(2), k_out.stride(1)

    if b == 0 or s == 0 or (hq == 0 and hk == 0):
        return q_out, k_out

    d_half = d // 2
    stride_cb = cos.stride(0) if cos.shape[0] > 1 else 0
    stride_cs = cos.stride(1) if cos.shape[1] > 1 else 0
    stride_sb = sin.stride(0) if sin.shape[0] > 1 else 0
    stride_ss = sin.stride(1) if sin.shape[1] > 1 else 0

    block_d_half = min(256, max(2, _next_pow2(d_half)))
    round_mode = _dtype_round_code(out_dtype)

    # Pick coalesced tiling along whichever axis (S or H) is inner-contiguous.
    if st_qoh <= st_qos and st_qh <= st_qs and hq >= 2:
        block_s = 1
        min_h = min(hq, hk) if hk > 0 else hq
        if block_d_half <= 64 and min_h >= 8:
            block_h = 8
            head_iters = 2 if hq >= 16 else 1
            num_warps = 4
        elif block_d_half <= 128 and min_h >= 4:
            block_h = 4
            head_iters = 2 if hq >= 8 else 1
            num_warps = 4
        else:
            block_h = 2
            head_iters = 2 if hq >= 4 else 1
            num_warps = 2
    else:
        block_h = 1
        if block_d_half <= 64:
            block_s = 16 if s >= 16 else max(1, _next_pow2(s))
        elif block_d_half <= 128:
            block_s = 8 if s >= 8 else max(1, _next_pow2(s))
        else:
            block_s = 4 if s >= 4 else max(1, _next_pow2(s))
        head_iters = 4 if hq >= 4 else (2 if hq >= 2 else 1)
        num_warps = 4 if block_s * block_d_half >= 256 else 2

    q_head_blocks = _cdiv(hq, block_h * head_iters) if hq > 0 else 0
    k_head_blocks = _cdiv(hk, block_h * head_iters) if hk > 0 else 0
    s_blocks = _cdiv(s, block_s)

    suite = _get_triton_suite()
    grid = (
        b * s_blocks,
        q_head_blocks + k_head_blocks,
        _cdiv(d_half, block_d_half),
    )
    suite.qk_half_rope_fwd_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        st_qb,
        st_qh,
        st_qs,
        st_kb,
        st_kh,
        st_ks,
        stride_cb,
        stride_cs,
        stride_sb,
        stride_ss,
        st_qob,
        st_qoh,
        st_qos,
        st_kob,
        st_koh,
        st_kos,
        s,
        hq,
        hk,
        d_half,
        q_head_blocks,
        ROUND_MODE=round_mode,
        BLOCK_S=block_s,
        BLOCK_H=block_h,
        HEAD_ITERS=head_iters,
        BLOCK_D_HALF=block_d_half,
        num_warps=num_warps,
    )
    return q_out, k_out


@_qk_half_rope_fwd_op.register_fake
def _qk_half_rope_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del unsqueeze_dim
    out_dtype = torch.promote_types(q.dtype, torch.promote_types(cos.dtype, sin.dtype))
    return (
        torch.empty(q.shape, device=q.device, dtype=out_dtype),
        torch.empty(k.shape, device=k.device, dtype=out_dtype),
    )


@torch.library.custom_op("worldfoundry::rope_2d_indexed_fwd", mutates_args=())
def _rope_2d_indexed_fwd_op(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    signed_positions: bool,
) -> torch.Tensor:
    out_dtype = torch.promote_types(tokens.dtype, torch.promote_types(cos_table.dtype, sin_table.dtype))
    out = torch.empty(tokens.shape, device=tokens.device, dtype=out_dtype)
    if tokens.numel() == 0:
        return out

    b, h, n, d = tokens.shape
    d_quart = d // 4
    table_len = cos_table.shape[0]
    stride_pb = positions.stride(0) if positions.shape[0] > 1 else 0
    round_mode = _dtype_round_code(out_dtype)
    suite = _get_triton_suite()

    # Fast 1-load contiguous path for power-of-2 quarter widths (D in {8, 16, 32, 64, 128, 256}).
    if 2 <= d_quart <= 64 and (d_quart & (d_quart - 1)) == 0:
        if d <= 64:
            block_n = 8 if n >= 8 else max(1, _next_pow2(n))
            num_warps = 2
        elif d <= 128:
            block_n = 8 if n >= 8 else max(1, _next_pow2(n))
            num_warps = 4
        else:
            block_n = 4 if n >= 4 else max(1, _next_pow2(n))
            num_warps = 4
        head_iters = 8 if (h >= 8 and b * _cdiv(n, block_n) >= 256) else (4 if h >= 4 else (2 if h >= 2 else 1))
        n_blocks = _cdiv(n, block_n)
        grid2 = (b * n_blocks, _cdiv(h, head_iters))
        suite.rope_2d_indexed_contig_kernel[grid2](
            tokens,
            positions,
            cos_table,
            sin_table,
            out,
            tokens.stride(0),
            tokens.stride(1),
            tokens.stride(2),
            stride_pb,
            positions.stride(1),
            positions.stride(2),
            cos_table.stride(0),
            sin_table.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            n,
            h,
            table_len,
            D_QUART=d_quart,
            SIGNED_POSITIONS=bool(signed_positions),
            ROUND_MODE=round_mode,
            BLOCK_N=block_n,
            HEAD_ITERS=head_iters,
            num_warps=num_warps,
        )
        return out

    block_d = min(256, max(4, _next_pow2(d)))
    if block_d <= 64:
        block_n = 16 if n >= 16 else max(1, _next_pow2(n))
    elif block_d <= 128:
        block_n = 8 if n >= 8 else max(1, _next_pow2(n))
    else:
        block_n = 4 if n >= 4 else max(1, _next_pow2(n))

    head_iters = 4 if h >= 4 else (2 if h >= 2 else 1)
    num_warps = 4 if block_n * block_d >= 512 else 2

    n_blocks = _cdiv(n, block_n)
    grid = (
        b * n_blocks,
        _cdiv(h, head_iters),
        _cdiv(d, block_d),
    )
    suite.rope_2d_indexed_fwd_kernel[grid](
        tokens,
        positions,
        cos_table,
        sin_table,
        out,
        tokens.stride(0),
        tokens.stride(1),
        tokens.stride(2),
        stride_pb,
        positions.stride(1),
        positions.stride(2),
        cos_table.stride(0),
        sin_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n,
        h,
        d,
        table_len,
        SIGNED_POSITIONS=bool(signed_positions),
        ROUND_MODE=round_mode,
        BLOCK_N=block_n,
        HEAD_ITERS=head_iters,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


@_rope_2d_indexed_fwd_op.register_fake
def _rope_2d_indexed_fwd_fake(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    signed_positions: bool,
) -> torch.Tensor:
    del positions, signed_positions
    out_dtype = torch.promote_types(tokens.dtype, torch.promote_types(cos_table.dtype, sin_table.dtype))
    return torch.empty(tokens.shape, device=tokens.device, dtype=out_dtype)


@torch.library.custom_op("worldfoundry::interleaved_partial_rope_fwd", mutates_args=())
def _interleaved_partial_rope_fwd_op(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int,
    scale: float,
) -> torch.Tensor:
    b, h, s, d = t.shape
    out = torch.empty((b, h, s, d), device=t.device, dtype=t.dtype)
    if b == 0 or h == 0 or s == 0:
        return out

    rot_dim = freqs.shape[-1]

    stride_fb = freqs.stride(0) if freqs.shape[0] > 1 else 0
    stride_fh = freqs.stride(1) if freqs.shape[1] > 1 else 0
    stride_fs = freqs.stride(2) if freqs.shape[2] > 1 else 0
    freq_per_head = stride_fh != 0
    suite = _get_triton_suite()

    # Fast 1-load coalesced path whenever start_index is even (pairs align with full-row 2-element tiles).
    if start_index % 2 == 0:
        d_pairs = _cdiv(d, 2)
        block_d_half = min(128, max(2, _next_pow2(d_pairs)))
        if block_d_half <= 64:
            block_s = 4 if s >= 4 else max(1, _next_pow2(s))
            num_warps = 2
        else:
            block_s = 4 if s >= 4 else max(1, _next_pow2(s))
            num_warps = 4
        head_iters = (
            8
            if (not freq_per_head and h >= 8 and b * _cdiv(s, block_s) >= 256)
            else (4 if (not freq_per_head and h >= 4) else (2 if (not freq_per_head and h >= 2) else 1))
        )
        s_blocks = _cdiv(s, block_s)
        grid_even = (
            b * s_blocks,
            _cdiv(h, head_iters),
            _cdiv(d_pairs, block_d_half),
        )
        suite.interleaved_even_start_kernel[grid_even](
            freqs,
            t,
            out,
            stride_fb,
            stride_fh,
            stride_fs,
            t.stride(0),
            t.stride(1),
            t.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            s,
            h,
            d,
            int(start_index),
            rot_dim,
            float(scale),
            FREQ_PER_HEAD=freq_per_head,
            BLOCK_S=block_s,
            HEAD_ITERS=head_iters,
            BLOCK_D_HALF=block_d_half,
            num_warps=num_warps,
        )
        return out

    block_d = min(256, max(4, _next_pow2(d)))
    if block_d <= 64:
        block_s = 8 if s >= 8 else max(1, _next_pow2(s))
    elif block_d <= 128:
        block_s = 4 if s >= 4 else max(1, _next_pow2(s))
    else:
        block_s = 4 if s >= 4 else max(1, _next_pow2(s))
    head_iters = 4 if (not freq_per_head and h >= 4) else (2 if (not freq_per_head and h >= 2) else 1)
    num_warps = 4 if block_s * block_d >= 512 else 2

    s_blocks = _cdiv(s, block_s)
    grid = (
        b * s_blocks,
        _cdiv(h, head_iters),
        _cdiv(d, block_d),
    )
    suite.interleaved_partial_rope_fwd_kernel[grid](
        freqs,
        t,
        out,
        stride_fb,
        stride_fh,
        stride_fs,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        s,
        h,
        d,
        int(start_index),
        rot_dim,
        float(scale),
        FREQ_PER_HEAD=freq_per_head,
        BLOCK_S=block_s,
        HEAD_ITERS=head_iters,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


@_interleaved_partial_rope_fwd_op.register_fake
def _interleaved_partial_rope_fwd_fake(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int,
    scale: float,
) -> torch.Tensor:
    del freqs, start_index, scale
    return torch.empty(t.shape, device=t.device, dtype=t.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Public Python dispatch helpers (return None when caller must use PyTorch)
# ──────────────────────────────────────────────────────────────────────────


def triton_complex_rotary_embedding(
    value: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Apply complex RoPE via Hopper Triton when eligible, else return ``None``."""
    if (
        value.ndim != 3
        or value.dtype not in _SUPPORTED_DTYPES
        or value.stride(-1) != 1
        or frequencies.dtype not in _COMPLEX_DTYPES
        or frequencies.ndim not in (3, 4)
        or not _is_cuda_inference_ready(value, frequencies)
    ):
        return None

    h = int(num_heads)
    b, s, hd = value.shape
    if h <= 0 or hd <= 0 or hd % h != 0:
        return None
    d = hd // h
    if d <= 0 or d % 2 != 0:
        return None
    d_half = d // 2

    f4 = frequencies.unsqueeze(0) if frequencies.ndim == 3 else frequencies
    if f4.shape[0] not in (1, b) or f4.shape[1] not in (1, s) or f4.shape[2] not in (1, h) or f4.shape[3] != d_half:
        return None

    freqs_ri = torch.view_as_real(f4)
    freq_re = freqs_ri[..., 0]
    freq_im = freqs_ri[..., 1]
    return _complex_rope_fwd_op(value, freq_re, freq_im, h, compute_dtype == torch.float64)


def triton_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply fused Q+K half-split RoPE via Hopper Triton when eligible, else ``None``."""
    if (
        q.dtype != k.dtype
        or q.dtype not in _SUPPORTED_DTYPES
        or cos.dtype not in _SUPPORTED_DTYPES
        or sin.dtype not in _SUPPORTED_DTYPES
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
        or not _is_cuda_inference_ready(q, k, cos, sin)
    ):
        return None

    # Support 3D (S, H, D) with 2D (S, D) cos/sin (e.g. Qwen vision) via unsqueeze(0).
    squeeze_batch = False
    if q.ndim == 3 and k.ndim == 3 and cos.ndim == 2 and sin.ndim == 2:
        u_dim = unsqueeze_dim if unsqueeze_dim >= 0 else (3 + unsqueeze_dim)
        if u_dim != 1:
            return None
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        u_dim = 2
        squeeze_batch = True
    elif q.ndim == 4 and k.ndim == 4 and cos.ndim == 3 and sin.ndim == 3:
        u_dim = unsqueeze_dim if unsqueeze_dim >= 0 else (4 + unsqueeze_dim)
        if u_dim not in (1, 2):
            return None
    else:
        return None

    if u_dim == 1:
        b, _, s, d = q.shape
        if k.shape[0] != b or k.shape[2] != s or k.shape[3] != d:
            return None
    else:
        b, s, _, d = q.shape
        if k.shape[0] != b or k.shape[1] != s or k.shape[3] != d:
            return None

    if d <= 0 or d % 2 != 0:
        return None
    if (
        cos.shape[0] not in (1, b)
        or cos.shape[1] not in (1, s)
        or cos.shape[2] != d
        or sin.shape[0] not in (1, b)
        or sin.shape[1] not in (1, s)
        or sin.shape[2] != d
    ):
        return None

    q_out, k_out = _qk_half_rope_fwd_op(q, k, cos, sin, u_dim)
    if squeeze_batch:
        return q_out.squeeze(0), k_out.squeeze(0)
    return q_out, k_out


def triton_rope_2d_indexed(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    signed_positions: bool,
) -> torch.Tensor | None:
    """Apply fused 2D axial indexed RoPE via Hopper Triton when eligible, else ``None``."""
    if (
        tokens.ndim != 4
        or positions.ndim != 3
        or positions.shape[-1] != 2
        or cos_table.ndim != 2
        or sin_table.ndim != 2
        or tokens.dtype not in _SUPPORTED_DTYPES
        or cos_table.dtype not in _SUPPORTED_DTYPES
        or sin_table.dtype not in _SUPPORTED_DTYPES
        or positions.dtype not in _INT_DTYPES
        or tokens.stride(-1) != 1
        or cos_table.stride(-1) != 1
        or sin_table.stride(-1) != 1
        or not _is_cuda_inference_ready(tokens, positions, cos_table, sin_table)
    ):
        return None

    b, _, n, d = tokens.shape
    if d < 4 or d % 4 != 0:
        return None
    if positions.shape[0] not in (1, b) or positions.shape[1] != n:
        return None
    d_half = d // 2
    if cos_table.shape[1] != d_half or sin_table.shape != cos_table.shape:
        return None

    return _rope_2d_indexed_fwd_op(tokens, positions, cos_table, sin_table, bool(signed_positions))


def triton_interleaved_partial_rope(
    freqs: torch.Tensor,
    t: torch.Tensor,
    *,
    start_index: int = 0,
    scale: float | torch.Tensor = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor | None:
    """Apply fused adjacent-pair partial-slice RoPE via Hopper Triton when eligible, else ``None``."""
    if (
        t.ndim not in (3, 4)
        or freqs.ndim < 2
        or freqs.ndim > t.ndim
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or t.dtype not in _SUPPORTED_DTYPES
        or freqs.dtype not in _SUPPORTED_DTYPES
        or t.stride(-1) != 1
        or freqs.stride(-1) != 1
        or not _is_cuda_inference_ready(freqs, t)
    ):
        return None

    norm_seq_dim = seq_dim if seq_dim >= 0 else (t.ndim + seq_dim)
    if norm_seq_dim != t.ndim - 2:
        return None

    rot_dim = freqs.shape[-1]
    d = t.shape[-1]
    if rot_dim <= 0 or rot_dim % 2 != 0 or start_index < 0 or start_index + rot_dim > d:
        return None

    squeeze_head = False
    if t.ndim == 3:
        b, s, _ = t.shape
        t4 = t.unsqueeze(1)
        f3 = freqs.unsqueeze(0) if freqs.ndim == 2 else freqs
        if f3.shape[0] not in (1, b) or f3.shape[1] not in (1, s) or f3.shape[2] != rot_dim:
            return None
        f4 = f3.unsqueeze(1)
        squeeze_head = True
    else:
        b, h, s, _ = t.shape
        t4 = t
        f4 = freqs.view((1,) * (4 - freqs.ndim) + tuple(freqs.shape))
        if (
            f4.shape[0] not in (1, b)
            or f4.shape[1] not in (1, h)
            or f4.shape[2] not in (1, s)
            or f4.shape[3] != rot_dim
        ):
            return None

    out = _interleaved_partial_rope_fwd_op(f4, t4, int(start_index), float(scale))
    return out.squeeze(1) if squeeze_head else out


__all__ = [
    "rotary_triton_ops_enabled",
    "triton_apply_rotary_pos_emb",
    "triton_complex_rotary_embedding",
    "triton_interleaved_partial_rope",
    "triton_rope_2d_indexed",
]
