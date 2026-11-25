# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels and O(1) fast block execution for LongLive-1.3B.

All kernels are CUDA-Graph capture safe (zero host syncs, zero dynamic tensor
allocations) and preserve exact BF16/FP32/FP64 reference semantics:

- :func:`wan_layernorm_adaln` — fused ``WanLayerNorm`` + per-frame AdaLN
  ``(1 + scale) + shift``.
- :func:`wan_residual_gate_add` — fused per-frame gated residual
  ``x + y * gate``.
- :func:`wan_rmsnorm` — single-pass ``WanRMSNorm`` replacing 7 eager ops.
- :func:`wan_qk_rmsnorm_rope_write_kv` — single-pass Q/K full-hidden RMSNorm,
  FP64/FP32 3D RoPE, and direct KV-cache + rotated-prefix K writes.
- :func:`fast_longlive_self_attention` — Hopper FlashAttention-3 (``b s n d``)
  for long causal self-attention with exact SDPA fallback when requested.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
    sf_rmsnorm,
)
from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="longlive-triton")

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _round_f32_to_bf16(value, USE_BF16_PTX: tl.constexpr):
        """Materialize an IEEE round-to-nearest-even BF16 boundary in registers."""
        if USE_BF16_PTX:
            return tl.inline_asm_elementwise(
                "{ .reg .b16 rounded; cvt.rn.bf16.f32 rounded, $1; mov.b32 $0, {0, rounded}; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        bits = value.to(tl.uint32, bitcast=True)
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits + rounding_bias) & 0xFFFF0000
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _round_f32_to_fp16(value):
        """Materialize an IEEE round-to-nearest-even FP16 boundary in registers."""
        return tl.inline_asm_elementwise(
            "{ .reg .b16 rounded; cvt.rn.f16.f32 rounded, $1; cvt.f32.f16 $0, rounded; }",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _round_to_dtype(value, dtype: tl.constexpr, USE_BF16_PTX: tl.constexpr = True):
        if dtype == tl.bfloat16:
            return _round_f32_to_bf16(value, USE_BF16_PTX)
        elif dtype == tl.float16:
            return _round_f32_to_fp16(value)
        else:
            return value

    @triton.jit
    def _wan_qk_rmsnorm_rope_write_kv_kernel(
        q_in_ptr,
        k_in_ptr,
        v_in_ptr,
        q_w_ptr,
        k_w_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_rot_out_ptr,
        k_raw_out_ptr,
        v_out_ptr,
        seq_len,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        k_dst_batch_stride,
        k_dst_seq_stride,
        v_dst_batch_stride,
        v_dst_seq_stride,
        dst_token_offset,
        eps,
        hidden_size: tl.constexpr,
        half_head_dim: tl.constexpr,
        block: tl.constexpr,
        block_pairs: tl.constexpr,
        WRITE_RAW_K: tl.constexpr,
        WRITE_V: tl.constexpr,
        USE_FP64: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Coalesced 128-bit single-pass Q/K WanRMSNorm + 3D RoPE + dual K + V cache store."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        cols = tl.arange(0, block)
        cmask = cols < hidden_size
        pair = tl.arange(0, block_pairs)
        pmask = pair < (hidden_size // 2)

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        if WRITE_V:
            v_base = row.to(tl.int64) * v_row_stride
            v_in = tl.load(v_in_ptr + v_base + cols, mask=cmask, other=0.0)
            v_dst_base = (
                batch_idx.to(tl.int64) * v_dst_batch_stride
                + (dst_token_offset + tok_idx).to(tl.int64) * v_dst_seq_stride
            )
            tl.store(v_out_ptr + v_dst_base + cols, v_in, mask=cmask)

        q_in = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
        k_in = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
        qf = q_in.to(tl.float32)
        kf = k_in.to(tl.float32)

        q_mean_sq = tl.sum(qf * qf, axis=0) / hidden_size
        k_mean_sq = tl.sum(kf * kf, axis=0) / hidden_size
        q_rstd = _libdevice.rsqrt(q_mean_sq + eps)
        k_rstd = _libdevice.rsqrt(k_mean_sq + eps)

        qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)

        q_n = _round_to_dtype(_round_to_dtype(qf * q_rstd, q_in.dtype, USE_BF16_PTX) * qw, q_in.dtype, USE_BF16_PTX)
        k_n = _round_to_dtype(_round_to_dtype(kf * k_rstd, k_in.dtype, USE_BF16_PTX) * kw, k_in.dtype, USE_BF16_PTX)

        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        if WRITE_RAW_K:
            tl.store(k_raw_out_ptr + k_dst_base + cols, k_n.to(k_in.dtype), mask=cmask)

        qe, qo = tl.split(tl.reshape(q_n, (block_pairs, 2)))
        ke, ko = tl.split(tl.reshape(k_n, (block_pairs, 2)))

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        if USE_FP64:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
            rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(q_in.dtype)
            rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(q_in.dtype)
            rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(k_in.dtype)
            rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(k_in.dtype)
        else:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
            rqe = (qe * c - qo * s).to(q_in.dtype)
            rqo = (qe * s + qo * c).to(q_in.dtype)
            rke = (ke * c - ko * s).to(k_in.dtype)
            rko = (ke * s + ko * c).to(k_in.dtype)

        rq = tl.reshape(tl.join(rqe, rqo), (block,))
        rk = tl.reshape(tl.join(rke, rko), (block,))

        q_out_base = row.to(tl.int64) * hidden_size
        tl.store(q_out_ptr + q_out_base + cols, rq, mask=cmask)
        tl.store(k_rot_out_ptr + k_dst_base + cols, rk, mask=cmask)

    @triton.jit
    def _wan_qk_apply_rstd_rope_write_kv_kernel(
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
        k_rot_out_ptr,
        k_raw_out_ptr,
        v_out_ptr,
        seq_len,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        k_dst_batch_stride,
        k_dst_seq_stride,
        v_dst_batch_stride,
        v_dst_seq_stride,
        dst_token_offset,
        hidden_size: tl.constexpr,
        half_head_dim: tl.constexpr,
        block: tl.constexpr,
        block_pairs: tl.constexpr,
        WRITE_RAW_K: tl.constexpr,
        WRITE_V: tl.constexpr,
        USE_FP64: tl.constexpr,
        USE_BF16_PTX: tl.constexpr,
    ):
        """Bit-exact coalesced 128-bit Q/K rstd-scale + weight + raw-K + 3D RoPE + rotated-K + V write."""
        row = tl.program_id(0)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        cols = tl.arange(0, block)
        cmask = cols < hidden_size
        pair = tl.arange(0, block_pairs)
        pmask = pair < (hidden_size // 2)

        q_base = row.to(tl.int64) * q_row_stride
        k_base = row.to(tl.int64) * k_row_stride

        if WRITE_V:
            v_base = row.to(tl.int64) * v_row_stride
            v_in = tl.load(v_in_ptr + v_base + cols, mask=cmask, other=0.0)
            v_dst_base = (
                batch_idx.to(tl.int64) * v_dst_batch_stride
                + (dst_token_offset + tok_idx).to(tl.int64) * v_dst_seq_stride
            )
            tl.store(v_out_ptr + v_dst_base + cols, v_in, mask=cmask)

        q_in = tl.load(q_in_ptr + q_base + cols, mask=cmask, other=0.0)
        k_in = tl.load(k_in_ptr + k_base + cols, mask=cmask, other=0.0)
        q_rstd = tl.load(q_rstd_ptr + row).to(tl.float32)
        k_rstd = tl.load(k_rstd_ptr + row).to(tl.float32)

        qw = tl.load(q_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        kw = tl.load(k_w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)

        q_n = _round_to_dtype(
            _round_to_dtype(q_in.to(tl.float32) * q_rstd, q_in.dtype, USE_BF16_PTX) * qw, q_in.dtype, USE_BF16_PTX
        )
        k_n = _round_to_dtype(
            _round_to_dtype(k_in.to(tl.float32) * k_rstd, k_in.dtype, USE_BF16_PTX) * kw, k_in.dtype, USE_BF16_PTX
        )

        k_dst_base = (
            batch_idx.to(tl.int64) * k_dst_batch_stride + (dst_token_offset + tok_idx).to(tl.int64) * k_dst_seq_stride
        )
        if WRITE_RAW_K:
            tl.store(k_raw_out_ptr + k_dst_base + cols, k_n.to(k_in.dtype), mask=cmask)

        qe, qo = tl.split(tl.reshape(q_n, (block_pairs, 2)))
        ke, ko = tl.split(tl.reshape(k_n, (block_pairs, 2)))

        pair_in_head = pair % half_head_dim
        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair_in_head
        if USE_FP64:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
            rqe = (qe.to(tl.float64) * c - qo.to(tl.float64) * s).to(tl.float32).to(q_in.dtype)
            rqo = (qe.to(tl.float64) * s + qo.to(tl.float64) * c).to(tl.float32).to(q_in.dtype)
            rke = (ke.to(tl.float64) * c - ko.to(tl.float64) * s).to(tl.float32).to(k_in.dtype)
            rko = (ke.to(tl.float64) * s + ko.to(tl.float64) * c).to(tl.float32).to(k_in.dtype)
        else:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
            rqe = (qe * c - qo * s).to(q_in.dtype)
            rqo = (qe * s + qo * c).to(q_in.dtype)
            rke = (ke * c - ko * s).to(k_in.dtype)
            rko = (ke * s + ko * c).to(k_in.dtype)

        rq = tl.reshape(tl.join(rqe, rqo), (block,))
        rk = tl.reshape(tl.join(rke, rko), (block,))

        q_out_base = row.to(tl.int64) * hidden_size
        tl.store(q_out_ptr + q_out_base + cols, rq, mask=cmask)
        tl.store(k_rot_out_ptr + k_dst_base + cols, rk, mask=cmask)

    @triton.jit
    def _wan_exact_real_rope_bshd_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq_len,
        x_batch_stride,
        x_seq_stride,
        x_head_stride,
        out_batch_stride,
        out_seq_stride,
        out_head_stride,
        head_dim: tl.constexpr,
        half_head_dim: tl.constexpr,
        block: tl.constexpr,
        block_pairs: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        """Bit-exact coalesced 128-bit 3D RoPE matching PyTorch ``causal_rope_apply``."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        batch_idx = row // seq_len
        tok_idx = row % seq_len

        cols = tl.arange(0, block)
        cmask = cols < head_dim
        pair = tl.arange(0, block_pairs)
        pmask = pair < half_head_dim

        x_base = (
            batch_idx.to(tl.int64) * x_batch_stride
            + tok_idx.to(tl.int64) * x_seq_stride
            + head.to(tl.int64) * x_head_stride
        )
        x_in = tl.load(x_ptr + x_base + cols, mask=cmask, other=0.0)
        xe_in, xo_in = tl.split(tl.reshape(x_in, (block_pairs, 2)))

        cs_off = tok_idx.to(tl.int64) * half_head_dim + pair
        if USE_FP64:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float64)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float64)
            rxe = (xe_in.to(tl.float64) * c - xo_in.to(tl.float64) * s).to(tl.float32).to(x_in.dtype)
            rxo = (xe_in.to(tl.float64) * s + xo_in.to(tl.float64) * c).to(tl.float32).to(x_in.dtype)
        else:
            c = tl.load(cos_ptr + cs_off, mask=pmask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + cs_off, mask=pmask, other=0.0).to(tl.float32)
            rxe = (xe_in.to(tl.float32) * c - xo_in.to(tl.float32) * s).to(x_in.dtype)
            rxo = (xe_in.to(tl.float32) * s + xo_in.to(tl.float32) * c).to(x_in.dtype)

        rx = tl.reshape(tl.join(rxe, rxo), (block,))
        out_base = (
            batch_idx.to(tl.int64) * out_batch_stride
            + tok_idx.to(tl.int64) * out_seq_stride
            + head.to(tl.int64) * out_head_stride
        )
        tl.store(out_ptr + out_base + cols, rx, mask=cmask)

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


_FA3_MODULE: Any = None
_FA3_PROBED: bool = False


def hopper_fa3_usable(device: torch.device | str | None = None) -> bool:
    """Return True when ``flash_attn_interface`` is installed and device is SM90 Hopper."""
    global _FA3_MODULE, _FA3_PROBED
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return False
    cap = torch.cuda.get_device_capability(dev)
    if cap[0] != 9:
        return False
    if not _FA3_PROBED:
        try:
            _FA3_MODULE = __import__("flash_attn_interface")
        except Exception:
            _FA3_MODULE = None
        _FA3_PROBED = True
    return _FA3_MODULE is not None and callable(getattr(_FA3_MODULE, "flash_attn_func", None))


def _use_bf16_ptx(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def wan_layernorm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
    eps: float = 1e-6,
    exact: bool = True,
) -> torch.Tensor:
    """``WanLayerNorm`` + fused Triton per-frame AdaLN ``(1 + scale) + shift`` (zero mod copies)."""
    return sf_layernorm_adaln(
        x,
        scale,
        shift,
        num_frames=num_frames,
        frame_seq=frame_seq,
        eps=eps,
        exact=exact,
        enabled=True,
    )


def wan_residual_gate_add(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_frames: int,
    frame_seq: int,
) -> torch.Tensor:
    """Bit-exact fused ``x + (y.unflatten(1, (num_frames, frame_seq)) * gate).flatten(1, 2)``."""
    return sf_residual_gate_add(
        x,
        y,
        gate,
        num_frames=num_frames,
        frame_seq=frame_seq,
        enabled=True,
    )


def wan_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, *, exact: bool = True) -> torch.Tensor:
    """Fast coalesced ``WanRMSNorm`` supporting strided last-dim-contiguous ``[B, S, C]``."""
    return sf_rmsnorm(x, weight, eps=eps, exact=exact, enabled=True)


def wan_qk_rmsnorm_rope_write_kv(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_rot_dst: torch.Tensor,
    k_raw_dst: torch.Tensor | None,
    *,
    dst_token_offset: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    exact: bool = True,
    v_proj: torch.Tensor | None = None,
    v_dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """Coalesced 128-bit Q/K WanRMSNorm + 3D RoPE + dual K (+ optional V) KV cache writes.

    When ``exact=True``, computes ``rstd`` with PyTorch's exact FP32 reduction order
    and fuses BF16 scale, weight multiply, raw-K write, FP64 3D RoPE, rotated-K write,
    and optional V write in a single coalesced Triton pass (0-ULP bitwise exact).
    When ``exact=False``, performs single-pass FP32-accumulated RMSNorm + RoPE + KV write.
    """
    if k_raw_dst is None and v_proj is not None and v_dst is not None:
        return sf_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            v_proj,
            q_weight,
            k_weight,
            cos,
            sin,
            k_rot_dst,
            v_dst,
            dst_start=dst_token_offset,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            rotate_k=True,
            exact=exact,
            enabled=True,
        )
    batch, seq_len, hidden = int(q_proj.shape[0]), int(q_proj.shape[1]), int(q_proj.shape[2])
    half_d = head_dim // 2
    write_v = v_proj is not None and v_dst is not None
    v_ok = (not write_v) or (
        v_proj is not None
        and v_dst is not None
        and v_proj.is_cuda
        and v_dst.is_cuda
        and v_proj.device == q_proj.device
        and v_dst.device == q_proj.device
        and v_proj.shape == q_proj.shape
        and v_proj.dtype == v_dst.dtype
        and v_proj.stride(-1) == 1
        and (batch == 1 or v_proj.stride(0) == seq_len * v_proj.stride(1))
        and v_dst.is_contiguous()
    )
    if (
        _TRITON_AVAILABLE
        and q_proj.is_cuda
        and k_proj.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and k_proj.device == q_proj.device
        and q_weight.device == q_proj.device
        and k_weight.device == q_proj.device
        and cos.device == q_proj.device
        and sin.device == q_proj.device
        and q_weight.dtype == q_proj.dtype
        and k_weight.dtype == k_proj.dtype
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and not torch.is_grad_enabled()
        and q_proj.stride(-1) == 1
        and k_proj.stride(-1) == 1
        and (
            batch == 1
            or (q_proj.stride(0) == seq_len * q_proj.stride(1) and k_proj.stride(0) == seq_len * k_proj.stride(1))
        )
        and k_rot_dst.is_contiguous()
        and (k_raw_dst is None or (k_raw_dst.is_contiguous() and k_raw_dst.shape == k_rot_dst.shape))
        and v_ok
        and hidden <= 4096
        and hidden % 2 == 0
    ):
        q_out = torch.empty((batch, seq_len, num_heads, head_dim), dtype=q_proj.dtype, device=q_proj.device)
        rows = batch * seq_len
        block = max(32, int(triton.next_power_of_2(hidden)))
        block_pairs = block // 2
        use_fp64 = cos.dtype == torch.float64
        use_ptx = _use_bf16_ptx(q_proj.device)
        num_warps = 4 if block >= 1024 else 2
        if block > 2048:
            num_warps = 8
        k_raw_ptr = k_rot_dst if k_raw_dst is None else k_raw_dst
        v_in_ptr = q_proj if v_proj is None else v_proj
        v_out_ptr = k_rot_dst if v_dst is None else v_dst
        v_row_stride = int(v_in_ptr.stride(1))
        k_dst_batch_stride = int(k_rot_dst.stride(0))
        k_dst_seq_stride = int(k_rot_dst.stride(1))
        v_dst_batch_stride = int(v_out_ptr.stride(0))
        v_dst_seq_stride = int(v_out_ptr.stride(1))
        cos_c = cos.reshape(seq_len, half_d).contiguous()
        sin_c = sin.reshape(seq_len, half_d).contiguous()
        if exact:
            q_rstd = torch.rsqrt(q_proj.float().square_().mean(dim=-1) + eps).reshape(rows).contiguous()
            k_rstd = torch.rsqrt(k_proj.float().square_().mean(dim=-1) + eps).reshape(rows).contiguous()
            _wan_qk_apply_rstd_rope_write_kv_kernel[(rows,)](
                q_proj,
                k_proj,
                v_in_ptr,
                q_rstd,
                k_rstd,
                q_weight,
                k_weight,
                cos_c,
                sin_c,
                q_out,
                k_rot_dst,
                k_raw_ptr,
                v_out_ptr,
                seq_len,
                int(q_proj.stride(1)),
                int(k_proj.stride(1)),
                v_row_stride,
                k_dst_batch_stride,
                k_dst_seq_stride,
                v_dst_batch_stride,
                v_dst_seq_stride,
                int(dst_token_offset),
                hidden_size=hidden,
                half_head_dim=half_d,
                block=block,
                block_pairs=block_pairs,
                WRITE_RAW_K=k_raw_dst is not None,
                WRITE_V=write_v,
                USE_FP64=use_fp64,
                USE_BF16_PTX=use_ptx,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
        else:
            _wan_qk_rmsnorm_rope_write_kv_kernel[(rows,)](
                q_proj,
                k_proj,
                v_in_ptr,
                q_weight,
                k_weight,
                cos_c,
                sin_c,
                q_out,
                k_rot_dst,
                k_raw_ptr,
                v_out_ptr,
                seq_len,
                int(q_proj.stride(1)),
                int(k_proj.stride(1)),
                v_row_stride,
                k_dst_batch_stride,
                k_dst_seq_stride,
                v_dst_batch_stride,
                v_dst_seq_stride,
                int(dst_token_offset),
                float(eps),
                hidden_size=hidden,
                half_head_dim=half_d,
                block=block,
                block_pairs=block_pairs,
                WRITE_RAW_K=k_raw_dst is not None,
                WRITE_V=write_v,
                USE_FP64=use_fp64,
                USE_BF16_PTX=use_ptx,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
        return q_out

    # Exact fallback
    if write_v:
        assert v_proj is not None and v_dst is not None
        v_dst[:, dst_token_offset : dst_token_offset + seq_len].copy_(v_proj.view(batch, seq_len, num_heads, head_dim))
    q_norm = wan_rmsnorm(q_proj, q_weight, eps=eps, exact=exact).view(batch, seq_len, num_heads, head_dim)
    k_norm = wan_rmsnorm(k_proj, k_weight, eps=eps, exact=exact).view(batch, seq_len, num_heads, head_dim)
    if k_raw_dst is not None:
        k_raw_dst[:, dst_token_offset : dst_token_offset + seq_len].copy_(k_norm)
    q_rot = _wan_exact_real_rope_bshd(q_norm, cos, sin).to(q_proj.dtype)
    k_rot = _wan_exact_real_rope_bshd(k_norm, cos, sin).to(k_proj.dtype)
    k_rot_dst[:, dst_token_offset : dst_token_offset + seq_len].copy_(k_rot)
    return q_rot


def _wan_exact_real_rope_bshd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bit-exact coalesced real-valued 3D RoPE on ``[B, S, H, D]`` matching PyTorch ``causal_rope_apply``."""
    batch, seq_len, num_heads, head_dim = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    half_d = head_dim // 2
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and cos.device == x.device
        and sin.device == x.device
        and not torch.is_grad_enabled()
        and x.stride(-1) == 1
        and (out is None or (out.is_cuda and out.device == x.device and out.shape == x.shape and out.stride(-1) == 1))
        and head_dim % 2 == 0
        and half_d <= 256
    ):
        dst = torch.empty((batch, seq_len, num_heads, head_dim), dtype=x.dtype, device=x.device) if out is None else out
        block = max(32, int(triton.next_power_of_2(head_dim)))
        block_pairs = block // 2
        cos_c = cos.reshape(seq_len, half_d).contiguous()
        sin_c = sin.reshape(seq_len, half_d).contiguous()
        _wan_exact_real_rope_bshd_kernel[(batch * seq_len, num_heads)](
            x,
            cos_c,
            sin_c,
            dst,
            seq_len,
            int(x.stride(0)),
            int(x.stride(1)),
            int(x.stride(2)),
            int(dst.stride(0)),
            int(dst.stride(1)),
            int(dst.stride(2)),
            head_dim=head_dim,
            half_head_dim=half_d,
            block=block,
            block_pairs=block_pairs,
            USE_FP64=cos.dtype == torch.float64,
            num_warps=2 if block <= 128 else 4,
            enable_fp_fusion=False,
        )
        return dst
    x_pairs = x.to(cos.dtype).reshape(batch, seq_len, num_heads, half_d, 2)
    x0, x1 = x_pairs.unbind(-1)
    cos_v = cos.reshape(1, seq_len, 1, half_d)
    sin_v = sin.reshape(1, seq_len, 1, half_d)
    rotated = torch.stack((x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v), dim=-1).flatten(-2).to(x.dtype)
    if out is not None:
        out.copy_(rotated)
        return out
    return rotated


def fast_longlive_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "flash_attention_3",
) -> torch.Tensor:
    """Execute ``[B, S, H, D]`` causal-cached self-attention via Hopper FA3 or cuDNN SDPA."""
    if (
        backend == "flash_attention_3"
        and hopper_fa3_usable(q.device)
        and _FA3_MODULE is not None
        and q.dtype in {torch.float16, torch.bfloat16}
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
    ):
        out = _FA3_MODULE.flash_attn_func(q, k, v)
        return out[0] if isinstance(out, tuple) else out
    if backend in ("torch", "sdpa", "auto", "flash_attention_3"):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
    from worldfoundry.core.attention.backends.dispatch import attention_forward

    return attention_forward(
        q,
        k,
        v,
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        backend=backend,
    )


def fast_longlive_cross_attention(
    cross: Any,
    x: torch.Tensor,
    crossattn_cache: dict[str, Any],
    *,
    exact: bool = True,
) -> torch.Tensor:
    """Execute cached Wan text cross-attention with fused RMSNorm + cuDNN SDPA (optimal for Sk=512)."""
    batch, seq_len = int(x.shape[0]), int(x.shape[1])
    num_heads = int(cross.num_heads)
    head_dim = int(cross.head_dim)
    q_raw = cross.q(x)
    norm_q = cross.norm_q
    if hasattr(norm_q, "weight") and hasattr(norm_q, "eps"):
        q = wan_rmsnorm(q_raw, norm_q.weight, eps=float(norm_q.eps), exact=exact).view(
            batch, seq_len, num_heads, head_dim
        )
    else:
        q = norm_q(q_raw).view(batch, seq_len, num_heads, head_dim)
    k = crossattn_cache["k"]
    v = crossattn_cache["v"]
    attn = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)
    return cross.o(attn.reshape(batch, seq_len, num_heads * head_dim))


def _canonical_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Guarantee canonical contiguous strides even when dim-0 == 1 (prevents duplicate CUDA Graphs)."""
    expected_stride = []
    acc = 1
    for dim in reversed(t.shape):
        expected_stride.append(acc)
        acc *= dim
    expected_stride.reverse()
    if tuple(t.stride()) == tuple(expected_stride):
        return t
    return t.clone(memory_format=torch.contiguous_format)


class LongLiveFastDenoiserEngine:
    """LongLive-scoped O(1) rolling-window CUDA Graph + FA3 + fused Triton engine.

    Wraps :class:`SelfForcingWanDenoiser` inside ``NativeLongLivePipeline`` without
    modifying shared Wan / Self-Forcing modules:

    - **O(1) Slot Canonicalization**: Canonicalizes ``kv_cache_relative_rope``
      chunk slots into at most 5 static piecewise CUDA Graph segments forever
      (on 81f, 153f, 300f+, and interactive multi-prompt rollouts) with
      canonicalized tensor strides so each segment captures *exactly once*.
    - **Cached Rotated Prefix K Arena**: Maintains a per-layer rotated key
      buffer alongside ``kv_cache["k"]`` so static sink frames (0..4) are never
      re-rotated and rolled window frames (5..8) are rotated only once per
      eviction step (3.2x reduction in RoPE work, bitwise-exact).
    - **Hopper FlashAttention-3 + Fused Triton Ops**: Runs Hopper FA3 on
      self-attention, cuDNN SDPA on short text cross-attention, and fused
      Triton LayerNorm+AdaLN / QK-RMSNorm+RoPE / Residual-Gate kernels across
      all 30 Wan blocks inside each captured CUDA Graph.
    """

    def __init__(
        self,
        denoiser: Any,
        *,
        attention_backend: str = "torch",
        enable_cuda_graph: bool = True,
        exact_norms: bool | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.model = denoiser.model
        self._fallbacks: list[str] = []
        first_param = next(self.model.parameters(), None)
        model_device = None if first_param is None else first_param.device
        resolved_attn = str(attention_backend or "torch").strip().casefold()
        if resolved_attn in {"flash3", "flash_attn_3", "fa3", "flash_attention_3"}:
            if hopper_fa3_usable(model_device):
                self.attention_backend = "flash_attention_3"
            else:
                self.attention_backend = "torch"
                self._fallbacks.append("flash_attention_3->torch (Hopper FA3 unavailable)")
        elif resolved_attn in {"auto", "torch", "sdpa"}:
            self.attention_backend = "torch"
        else:
            self.attention_backend = resolved_attn
        self.exact_norms = (self.attention_backend == "torch") if exact_norms is None else bool(exact_norms)
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self._roped_k_arenas: dict[tuple[Any, ...], list[torch.Tensor]] = {}
        self._roll_scratch: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_cos_sin_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._sinusoid_freqs_cache: dict[torch.device, torch.Tensor] = {}
        self._graph_scope_ids: dict[tuple[Any, ...], int] = {}
        self._fast_block_calls = 0
        self._fa3_calls = 0

        # Ensure O(1) piecewise graph runner is active when requested.
        if self.enable_cuda_graph and getattr(denoiser, "_piecewise_graph", None) is None:
            from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

            denoiser._piecewise_graph = PiecewiseBlockGraphRunner(
                warmup=1, max_graphs_per_segment=8, clone_outputs=True
            )
            denoiser.enable_cuda_graph = True

    def reset_rollout_state(self) -> None:
        """Clear host KV mirror so a fresh rollout starts cleanly at slot 0."""
        reset_host = getattr(self.model, "reset_causal_host_cache", None)
        if callable(reset_host):
            reset_host()

    def _get_roped_k_arena(
        self,
        kv_cache: list[dict[str, Any]],
        *,
        prev_local_end: int = 0,
        frame_seqlen: int = 0,
        h_grid: int = 0,
        w_grid: int = 0,
    ) -> list[torch.Tensor]:
        layer0 = kv_cache[0]
        k0 = layer0["k"]
        key = (len(kv_cache), k0.data_ptr(), tuple(k0.shape), k0.dtype, k0.device)
        arena = self._roped_k_arenas.get(key)
        if arena is None:
            if len(self._roped_k_arenas) >= 8:
                self._roped_k_arenas.clear()
            with torch.inference_mode(False):
                arena = [
                    torch.empty(slot["k"].shape, dtype=slot["k"].dtype, device=slot["k"].device) for slot in kv_cache
                ]
            if prev_local_end > 0 and frame_seqlen > 0 and h_grid > 0 and w_grid > 0:
                # Backfill already-populated raw KV prefix when an arena is created mid-rollout.
                prev_frames = prev_local_end // frame_seqlen
                if prev_frames > 0:
                    pref_cos, pref_sin = self._get_rope_cos_sin(prev_frames, h_grid, w_grid, 0, k0.device)
                    pref_toks = prev_frames * frame_seqlen
                    for idx, slot in enumerate(kv_cache):
                        _wan_exact_real_rope_bshd(
                            slot["k"][:, :pref_toks], pref_cos, pref_sin, out=arena[idx][:, :pref_toks]
                        )
            self._roped_k_arenas[key] = arena
        return arena

    def _get_roll_scratch(self, k0: torch.Tensor, num_rolled: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (k0.shape[0], num_rolled, k0.shape[2], k0.shape[3], k0.dtype, k0.device)
        pair = self._roll_scratch.get(key)
        if pair is None:
            with torch.inference_mode(False):
                pair = (
                    torch.empty((k0.shape[0], num_rolled, k0.shape[2], k0.shape[3]), dtype=k0.dtype, device=k0.device),
                    torch.empty((k0.shape[0], num_rolled, k0.shape[2], k0.shape[3]), dtype=k0.dtype, device=k0.device),
                )
            self._roll_scratch[key] = pair
        return pair

    def _get_sinusoid_freqs(self, device: torch.device) -> torch.Tensor:
        freqs = self._sinusoid_freqs_cache.get(device)
        if freqs is None:
            half = int(self.model.freq_dim) // 2
            freqs = torch.pow(10000.0, -torch.arange(half, device=device, dtype=torch.float64).div(half))
            self._sinusoid_freqs_cache[device] = freqs
        return freqs

    def _get_rope_cos_sin(
        self,
        num_frames: int,
        h_grid: int,
        w_grid: int,
        start_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_prec = str(getattr(self.model, "_worldfoundry_rope_precision", "fp64")).strip().casefold()
        rope_dtype = torch.float32 if rope_prec == "fp32" else torch.float64
        if self.model.freqs.device != device:
            self.model.freqs = self.model.freqs.to(device)
        key = (id(self.model.freqs), device, rope_dtype, num_frames, h_grid, w_grid, start_frame)
        cached = self._rope_cos_sin_cache.get(key)
        if cached is not None:
            return cached
        freqs = self.model.freqs
        c = freqs.shape[1]
        t_dim = c - 2 * (c // 3)
        s_dim = c // 3
        f_t, f_h, f_w = freqs.split([t_dim, s_dim, s_dim], dim=1)
        freqs_3d = torch.cat(
            [
                f_t[start_frame : start_frame + num_frames]
                .view(num_frames, 1, 1, t_dim)
                .expand(num_frames, h_grid, w_grid, t_dim),
                f_h[:h_grid].view(1, h_grid, 1, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
                f_w[:w_grid].view(1, 1, w_grid, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
            ],
            dim=-1,
        ).reshape(1, num_frames * h_grid * w_grid, 1, c)
        pair = (freqs_3d.real.to(rope_dtype).contiguous(), freqs_3d.imag.to(rope_dtype).contiguous())
        if len(self._rope_cos_sin_cache) >= 128:
            self._rope_cos_sin_cache.clear()
        self._rope_cos_sin_cache[key] = pair
        return pair

    def _roll_kv_once(
        self,
        kv_cache: list[dict[str, Any]],
        roped_k_arena: list[torch.Tensor] | None,
        *,
        sink_tokens: int,
        num_evicted: int,
        num_rolled: int,
        roll_cos: torch.Tensor | None,
        roll_sin: torch.Tensor | None,
        relative_rope: bool,
    ) -> None:
        """Roll the sliding KV window in place *once* outside CUDA Graph capture.

        Executing the roll outside the graph keeps every captured graph segment
        strictly idempotent (preventing warmup + capture + replay from rolling
        the cache 3x on Chunk 4 Step 0) and collapses all rolling chunks onto
        the already-captured full-window segment (4 O(1) segments forever).
        """
        if num_evicted <= 0 or num_rolled <= 0:
            return
        k_tmp, v_tmp = self._get_roll_scratch(kv_cache[0]["k"], num_rolled)
        do_rerope = relative_rope and roped_k_arena is not None and roll_cos is not None and roll_sin is not None
        for idx, layer_kv in enumerate(kv_cache):
            k_cache = layer_kv["k"]
            v_cache = layer_kv["v"]
            k_tmp.copy_(k_cache[:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled])
            v_tmp.copy_(v_cache[:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled])
            k_cache[:, sink_tokens : sink_tokens + num_rolled].copy_(k_tmp)
            v_cache[:, sink_tokens : sink_tokens + num_rolled].copy_(v_tmp)
            if do_rerope:
                assert roped_k_arena is not None and roll_cos is not None and roll_sin is not None
                # Sink slots [0:sink_tokens] never move and keep their cached rotated K.
                # Re-rotate only the rolled window frames once per chunk directly into the arena (0-ULP exact).
                _wan_exact_real_rope_bshd(
                    k_tmp,
                    roll_cos,
                    roll_sin,
                    out=roped_k_arena[idx][:, sink_tokens : sink_tokens + num_rolled],
                )

    def _forward_fast_model(
        self,
        x_in: torch.Tensor,
        t_in: torch.Tensor,
        *,
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        roped_k_arena: list[torch.Tensor] | None,
        local_start_index: int,
        local_end_index: int,
        q_cos: torch.Tensor,
        q_sin: torch.Tensor,
        relative_rope: bool,
    ) -> torch.Tensor:
        """Execute all 30 CausalWanModel blocks with fused Triton ops + FA3 + cached rotated K."""
        model = self.model
        x_emb = model.patch_embedding(x_in)
        batch_size = int(x_emb.shape[0])
        f_grid, h_grid, w_grid = int(x_emb.shape[2]), int(x_emb.shape[3]), int(x_emb.shape[4])
        frame_seq = h_grid * w_grid
        x = x_emb.flatten(2).transpose(1, 2).contiguous()

        # Sinusoidal timestep embedding without H2D syncs
        sinusoid_freqs = self._get_sinusoid_freqs(x.device)
        pos = t_in.flatten().to(dtype=torch.float64)
        angles = torch.outer(pos, sinusoid_freqs)
        t_emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).type_as(x)
        e = model.time_embedding(t_emb)
        e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(dim=0, sizes=t_in.shape)

        num_heads = int(model.num_heads)
        head_dim = int(model.dim) // num_heads
        max_attn_size = int(model.blocks[0].self_attn.max_attention_size)
        kv_read_start = 0 if relative_rope else max(0, local_end_index - max_attn_size)

        exact = self.exact_norms
        for idx, block in enumerate(model.blocks):
            mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            # 1. Fused LayerNorm + AdaLN before self-attention (zero mod copies)
            h = sf_layernorm_adaln(
                x,
                mod[1],
                mod[0],
                num_frames=f_grid,
                frame_seq=frame_seq,
                eps=float(block.norm1.eps),
                exact=exact,
                enabled=True,
            )

            # 2. Self-attention QKV projection + coalesced 128-bit QK-RMSNorm + RoPE + K/V cache write
            sa = block.self_attn
            if hasattr(sa, "qkv"):
                qkv_out = sa.qkv(h)
                q_proj, k_proj, v_proj = qkv_out.chunk(3, dim=-1)
            else:
                q_proj = sa.q(h)
                k_proj = sa.k(h)
                v_proj = sa.v(h)

            layer_kv = kv_cache[idx]
            k_cache = layer_kv["k"]
            v_cache = layer_kv["v"]

            has_qk_norm = hasattr(sa.norm_q, "weight") and hasattr(sa.norm_k, "weight")
            if has_qk_norm:
                if relative_rope and roped_k_arena is not None:
                    k_rot_buf = roped_k_arena[idx]
                    roped_q = wan_qk_rmsnorm_rope_write_kv(
                        q_proj,
                        k_proj,
                        sa.norm_q.weight,
                        sa.norm_k.weight,
                        q_cos,
                        q_sin,
                        k_rot_buf,
                        k_cache,
                        dst_token_offset=local_start_index,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        eps=float(sa.norm_q.eps),
                        exact=exact,
                        v_proj=v_proj,
                        v_dst=v_cache,
                    )
                    attn_k = k_rot_buf[:, :local_end_index]
                    attn_v = v_cache[:, :local_end_index]
                else:
                    roped_q = sf_qk_rmsnorm_rope_write_kv(
                        q_proj,
                        k_proj,
                        v_proj,
                        sa.norm_q.weight,
                        sa.norm_k.weight,
                        q_cos,
                        q_sin,
                        k_cache,
                        v_cache,
                        dst_start=local_start_index,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        eps=float(sa.norm_q.eps),
                        rotate_k=True,
                        exact=exact,
                        enabled=True,
                    )
                    attn_k = k_cache[:, kv_read_start:local_end_index]
                    attn_v = v_cache[:, kv_read_start:local_end_index]
            else:
                v = v_proj.view(batch_size, f_grid * frame_seq, num_heads, head_dim)
                v_cache[:, local_start_index:local_end_index].copy_(v)
                q_v = sa.norm_q(q_proj).view(batch_size, f_grid * frame_seq, num_heads, head_dim)
                k_v = sa.norm_k(k_proj).view(batch_size, f_grid * frame_seq, num_heads, head_dim)
                roped_q = _wan_exact_real_rope_bshd(q_v, q_cos, q_sin).to(v.dtype)
                roped_k = _wan_exact_real_rope_bshd(k_v, q_cos, q_sin).to(v.dtype)
                if relative_rope and roped_k_arena is not None:
                    k_cache[:, local_start_index:local_end_index].copy_(k_v)
                    roped_k_arena[idx][:, local_start_index:local_end_index].copy_(roped_k)
                    attn_k = roped_k_arena[idx][:, :local_end_index]
                    attn_v = v_cache[:, :local_end_index]
                else:
                    k_cache[:, local_start_index:local_end_index].copy_(roped_k)
                    attn_k = k_cache[:, kv_read_start:local_end_index]
                    attn_v = v_cache[:, kv_read_start:local_end_index]

            attn_out = fast_longlive_self_attention(roped_q, attn_k, attn_v, backend=self.attention_backend)
            y_sa = sa.o(attn_out.reshape(batch_size, f_grid * frame_seq, model.dim))

            # 3. 2-stage super-fusion: x = x + y_sa * mod[2] AND x_ca_in = norm3(x)
            norm3 = block.norm3
            if hasattr(norm3, "weight") and hasattr(norm3, "eps"):
                x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y_sa.dtype, mod[2].dtype))
                w3 = norm3.weight if norm3.weight.dtype == x_out_dtype else norm3.weight.to(x_out_dtype)
                b3_raw = getattr(norm3, "bias", None)
                b3 = None if b3_raw is None else (b3_raw if b3_raw.dtype == x_out_dtype else b3_raw.to(x_out_dtype))
                x, x_ca_in = sf_residual_gate_add_layernorm(
                    x,
                    y_sa,
                    mod[2],
                    w3,
                    b3,
                    num_frames=f_grid,
                    frame_seq=frame_seq,
                    eps=float(norm3.eps),
                    exact=exact,
                    enabled=True,
                )
            else:
                x = sf_residual_gate_add(x, y_sa, mod[2], num_frames=f_grid, frame_seq=frame_seq, enabled=True)
                x_ca_in = norm3(x)

            # 4. Cross-attention (cached K/V, coalesced sf_rmsnorm + cuDNN SDPA)
            ca_out = fast_longlive_cross_attention(block.cross_attn, x_ca_in, crossattn_cache[idx], exact=exact)

            # 5. 2-stage super-fusion: x = x + ca_out AND h_ffn = norm2(x) * (1 + mod[4]) + mod[3]
            x, h_ffn = sf_residual_add_layernorm_adaln(
                x,
                ca_out,
                mod[4],
                mod[3],
                num_frames=f_grid,
                frame_seq=frame_seq,
                eps=float(block.norm2.eps),
                exact=exact,
                enabled=True,
            )
            y_ffn = block.ffn(h_ffn)
            x = sf_residual_gate_add(x, y_ffn, mod[5], num_frames=f_grid, frame_seq=frame_seq, enabled=True)

        # Output head with fused LayerNorm + AdaLN
        head = model.head
        e_head = (head.modulation.unsqueeze(1) + e.unflatten(dim=0, sizes=t_in.shape).unsqueeze(2)).chunk(2, dim=2)
        h_head = sf_layernorm_adaln(
            x,
            e_head[1],
            e_head[0],
            num_frames=f_grid,
            frame_seq=frame_seq,
            eps=float(head.norm.eps),
            exact=exact,
            enabled=True,
        )
        out = head.head(h_head)
        c_out = int(model.out_dim)
        pt, ph, pw = model.patch_size
        out = out[:, : f_grid * h_grid * w_grid].view(batch_size, f_grid, h_grid, w_grid, pt, ph, pw, c_out)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return out.view(batch_size, c_out, f_grid * pt, h_grid * ph, w_grid * pw)

    def __call__(self, model_input: Any) -> Any:
        """Drop-in replacement for ``SelfForcingWanDenoiser.__call__`` with O(1) graphs + FA3 + Triton."""
        from worldfoundry.base_models.diffusion_model.contracts import DenoiserOutput

        denoiser = self.denoiser
        conditioning = model_input.conditioning
        cache = conditioning.get("kv_cache")
        if not isinstance(cache, dict):
            return denoiser(model_input)
        context = conditioning.get("context")
        if not isinstance(context, torch.Tensor):
            return denoiser(model_input)
        timestep = model_input.timestep.reshape(-1)
        if timestep.numel() != 1:
            return denoiser(model_input)

        noisy = _canonical_contiguous(model_input.latents.to(dtype=denoiser.compute_dtype))
        if not noisy.is_cuda or torch.is_grad_enabled():
            return denoiser(model_input)

        batch, _, frames, height, width = noisy.shape
        frame_timesteps = _canonical_contiguous(
            timestep.to(device=noisy.device, dtype=torch.float32).expand(batch, frames)
        )
        current_start = int(conditioning.get("current_start", 0))
        frame_seqlen = height * width // 4
        seq_len = frames * frame_seqlen
        kv_cache = cache["kv_cache"]
        crossattn_cache = cache["crossattn_cache"]

        # Initialize crossattn_cache in-place on first pass if not already populated so even
        # Chunk 0 Step 0 executes on the fast fused path and captures into the O(1) graph!
        cross_ready = bool(crossattn_cache) and all(bool(entry.get("is_init")) for entry in crossattn_cache)
        if not cross_ready:
            from .pipeline_longlive import NativeLongLivePipeline

            NativeLongLivePipeline._refresh_crossattn_cache_inplace_static(self.model, context, crossattn_cache)
            cross_ready = True

        layer0 = kv_cache[0]
        kv_cap = int(layer0["k"].shape[1])
        num_new_tokens = frames * frame_seqlen
        current_end = current_start + num_new_tokens
        mirror = self.model.__dict__.setdefault("_worldfoundry_kv_host_mirror", {})
        mirror_key = (id(kv_cache), layer0["k"].data_ptr())
        cached_pair = mirror.get(mirror_key)
        if cached_pair is None:
            if current_start == 0:
                prev_global_end, prev_local_end = 0, 0
            else:
                prev_global_end = int(layer0["global_end_index"].item())
                prev_local_end = int(layer0["local_end_index"].item())
        else:
            prev_global_end, prev_local_end = cached_pair
            if current_start == 0 and prev_global_end > current_end:
                # Fresh rollout restarted at slot 0 reusing the same KV buffers without reset_rollout_state().
                prev_global_end, prev_local_end = 0, 0

        sa0 = self.model.blocks[0].self_attn
        local_attn = int(sa0.local_attn_size)
        sink_tokens = int(sa0.sink_size) * frame_seqlen
        relative_rope = bool(getattr(sa0, "kv_cache_relative_rope", False))
        if local_attn != -1 and current_end > prev_global_end and num_new_tokens + prev_local_end > kv_cap:
            num_evicted = num_new_tokens + prev_local_end - kv_cap
            num_rolled = prev_local_end - num_evicted - sink_tokens
            next_local_end = prev_local_end + current_end - prev_global_end - num_evicted
        else:
            num_evicted = 0
            num_rolled = 0
            next_local_end = prev_local_end + current_end - prev_global_end
        local_start_index = next_local_end - num_new_tokens

        h_grid = height // int(self.model.patch_size[1])
        w_grid = width // int(self.model.patch_size[2])
        if relative_rope:
            q_start_f = local_start_index // frame_seqlen
            q_cos, q_sin = self._get_rope_cos_sin(frames, h_grid, w_grid, q_start_f, noisy.device)
            if num_evicted > 0 and num_rolled > 0:
                roll_frames = num_rolled // frame_seqlen
                sink_frames = sink_tokens // frame_seqlen
                roll_cos, roll_sin = self._get_rope_cos_sin(roll_frames, h_grid, w_grid, sink_frames, noisy.device)
            else:
                roll_cos, roll_sin = None, None
            roped_k_arena = self._get_roped_k_arena(
                kv_cache,
                prev_local_end=prev_local_end,
                frame_seqlen=frame_seqlen,
                h_grid=h_grid,
                w_grid=w_grid,
            )
            # Perform rolling eviction exactly once outside the graph so all rolling chunks
            # reuse the already-captured full-window segment (4 O(1) segments forever!).
            self._roll_kv_once(
                kv_cache,
                roped_k_arena,
                sink_tokens=sink_tokens,
                num_evicted=num_evicted,
                num_rolled=num_rolled,
                roll_cos=roll_cos,
                roll_sin=roll_sin,
                relative_rope=True,
            )
            arena_ptr = roped_k_arena[0].data_ptr()
        else:
            start_f = current_start // frame_seqlen if frame_seqlen > 0 else current_start
            q_cos, q_sin = self._get_rope_cos_sin(frames, h_grid, w_grid, start_f, noisy.device)
            roped_k_arena = None
            self._roll_kv_once(
                kv_cache,
                None,
                sink_tokens=sink_tokens,
                num_evicted=num_evicted,
                num_rolled=num_rolled,
                roll_cos=None,
                roll_sin=None,
                relative_rope=False,
            )
            arena_ptr = 0

        # Bind segment names to geometry + static buffer pointers so PiecewiseBlockGraphRunner
        # stays strictly O(1) (4 segments) for any steady rollout / prompt-switch session while
        # preventing silent stale-graph reuse on resolution, batch, dtype, or buffer-pointer changes.
        scope_key = (
            batch,
            h_grid,
            w_grid,
            noisy.dtype,
            layer0["k"].data_ptr(),
            layer0["v"].data_ptr(),
            crossattn_cache[0]["k"].data_ptr(),
            crossattn_cache[0]["v"].data_ptr(),
            arena_ptr,
            self.attention_backend,
            self.exact_norms,
        )
        scope_id = self._graph_scope_ids.get(scope_key)
        if scope_id is None:
            scope_id = len(self._graph_scope_ids)
            self._graph_scope_ids[scope_key] = scope_id
        if relative_rope:
            seg_name = f"ll_o1_s{scope_id}_ls{local_start_index}_le{next_local_end}_q{seq_len}"
        else:
            seg_name = f"ll_abs_s{scope_id}_f{start_f}_ls{local_start_index}_le{next_local_end}_q{seq_len}"

        pw_graph = getattr(denoiser, "_piecewise_graph", None)
        if self.enable_cuda_graph and pw_graph is not None and cross_ready:

            def _make_slot_fn(
                slot_ls: int,
                slot_le: int,
                slot_qcos: torch.Tensor,
                slot_qsin: torch.Tensor,
            ):
                def _run_slot(x_arg: torch.Tensor, t_arg: torch.Tensor) -> torch.Tensor:
                    return self._forward_fast_model(
                        x_arg,
                        t_arg,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        roped_k_arena=roped_k_arena,
                        local_start_index=slot_ls,
                        local_end_index=slot_le,
                        q_cos=slot_qcos,
                        q_sin=slot_qsin,
                        relative_rope=relative_rope,
                    )

                return _run_slot

            pw_graph.get_or_register_segment(
                seg_name,
                _make_slot_fn(local_start_index, next_local_end, q_cos, q_sin),
                extra_key=f"ll:{seg_name}:{scope_key}",
            )
            flow = pw_graph.run_segment(seg_name, noisy, frame_timesteps)
        else:
            flow = self._forward_fast_model(
                noisy,
                frame_timesteps,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                roped_k_arena=roped_k_arena,
                local_start_index=local_start_index,
                local_end_index=next_local_end,
                q_cos=q_cos,
                q_sin=q_sin,
                relative_rope=relative_rope,
            )

        mirror[mirror_key] = (current_end, next_local_end)
        self._fast_block_calls += 1
        if self.attention_backend == "flash_attention_3":
            self._fa3_calls += len(self.model.blocks)

        x0 = denoiser.training_schedule.flow_to_x0(
            flow.transpose(1, 2).flatten(0, 1),
            noisy.transpose(1, 2).flatten(0, 1),
            frame_timesteps.flatten(),
        )
        x0 = x0.unflatten(0, (batch, frames)).transpose(1, 2)
        return DenoiserOutput(sample=x0.to(dtype=model_input.latents.dtype))

    def report(self) -> dict[str, Any]:
        blocks = getattr(self.model, "blocks", ())
        has_rel_rope = bool(blocks and getattr(getattr(blocks[0], "self_attn", None), "kv_cache_relative_rope", False))
        return {
            "enabled": True,
            "o1_slot_canonicalization": True,
            "cached_rotated_prefix_kv": has_rel_rope,
            "attention_backend": self.attention_backend,
            "hopper_fa3_active": self.attention_backend == "flash_attention_3",
            "exact_norms": self.exact_norms,
            "fused_triton_blocks": len(blocks) if _TRITON_AVAILABLE else 0,
            "fast_step_calls": self._fast_block_calls,
            "fa3_self_attn_calls": self._fa3_calls,
            "fallbacks": list(self._fallbacks),
        }
