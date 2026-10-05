# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Inference-side helpers for the bidirectional fused GDN path.

The Python GDN math lives in ``sana_gdn_blocks``; this module only prepares
RoPE / inv-RMS tables and dispatches the scan to ``fused_gdn_chunkwise``.

What this file contains:
  1. Config helpers (``_get_kernel_config``, ``_kcfg``, ``_precision_params``,
     ``_resolve_launch_config``) — re-used by ``fused_gdn_chunkwise``.
  2. ``prepare_rope_tables`` — complex ``(1,1,N,D//2)`` RoPE freqs → expanded
     ``(N, D)`` cos/sin tables with the interleaved-pair layout.
  3. ``_precompute_inv_rms`` — cross-head (full-channel) 1/RMS per token.
  4. Fused single-pass Q+K inverse-RMS Triton kernel + ``fused_qk_inv_rms``.
  5. ``fused_bigdn_func`` — thin bidirectional entry point that delegates to
     the chunkwise kernel in ``fused_gdn_chunkwise``.

Precision knob: env var ``FUSED_GDN_PRECISION`` or ``PRECISION_OVERRIDE``:
  0=IEEE fp32 dots, 1=TF32, 2=bf16 TC + fp32 state [default], 3=bf16 TC + bf16 state.
"""

# ruff: noqa: E501

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# =====================================================================
#  GPU-adaptive kernel config
# =====================================================================


def _get_kernel_config() -> dict:
    """Return optimal kernel parameters for the current GPU.

    STATE_FP32: use fp32 state_prev when SRAM is large enough.
      - bf16 state_prev: ~96KB total SRAM (fits GB10's 101KB).
      - fp32 state_prev: ~128KB total SRAM (needs H100's 228KB+).
    """
    if not torch.cuda.is_available():
        return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 4, "STATE_FP32": False}
    props = torch.cuda.get_device_properties(0)
    # Some supported PyTorch builds omit this optional CUDA property. Match
    # fused_gdn_chunkwise's conservative H100 fallback instead of failing
    # before kernel selection.
    smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
    state_fp32 = smem >= 150 * 1024  # H100 (228KB) yes, GB10 (101KB) no
    return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 8, "STATE_FP32": state_fp32}


_KCFG = None


def _kcfg():
    """Helper function to kcfg."""
    global _KCFG
    if _KCFG is None:
        _KCFG = _get_kernel_config()
    return _KCFG


# precision=0 → IEEE fp32 dots + fp32 state  (DOT_PRECISION=2, STATE_FP32=1)
# precision=1 → TF32  dots   + fp32 state    (DOT_PRECISION=1, STATE_FP32=1)
# precision=2 → bf16  dots   + fp32 state    (DOT_PRECISION=0, STATE_FP32=1) [default]
# precision=3 → bf16  dots   + bf16 state    (DOT_PRECISION=0, STATE_FP32=0)
def _precision_params(precision: int) -> tuple:
    """Helper function to precision params.

    Args:
        precision: The precision.

    Returns:
        The return value.
    """
    if precision == 0:
        return 2, True
    elif precision == 1:
        return 1, True
    elif precision == 3:
        return 0, False
    else:  # default
        return 0, True


_env_prec = os.environ.get("FUSED_GDN_PRECISION", None)
PRECISION_OVERRIDE: int | None = int(_env_prec) if _env_prec is not None else None


def _resolve_launch_config() -> tuple:
    """Returns (prec, dot_prec, state_fp32, num_warps).

    Uses ``PRECISION_OVERRIDE`` when set; otherwise falls back to ``_kcfg()``
    (which picks ``STATE_FP32`` based on per-GPU SRAM). ``num_warps`` is
    clamped to 4 when dots run on fp32 operands (more registers needed).
    """
    cfg = _kcfg()
    prec = PRECISION_OVERRIDE if PRECISION_OVERRIDE is not None else 2
    dot_prec, state_fp32 = _precision_params(prec)
    if PRECISION_OVERRIDE is None:
        state_fp32 = cfg["STATE_FP32"]
    nw = cfg["num_warps"]
    if dot_prec >= 1:
        nw = min(nw, 4)
    return prec, dot_prec, state_fp32, nw


def prepare_rope_tables(rotary_emb, N: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex rotary_emb `(1, 1, N, D//2)` → expanded (N, D) cos/sin tables.

    Encodes the interleaved-pair rotation
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    as  y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]
    where sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].

    Returns (cos_exp, sin_exp) both (N, D) float32, contiguous.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


def _precompute_inv_rms(qkv: torch.Tensor, idx: int, C: int, eps: float = 1e-5) -> torch.Tensor:
    """Compute 1/RMS for one component of QKV over the full C = H*D channel dim.

    Args:
      qkv:   (B, N, 3, H, D)
      idx:   0 for Q, 1 for K, 2 for V
      C:     H*D (channel count)
      eps:   RMSNorm epsilon

    Returns:
      inv_rms: (B, N) float32
    """
    raw = qkv[:, :, idx].float()  # (B, N, H, D)
    sq_sum = (raw * raw).sum(dim=(-2, -1))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps)


# =====================================================================
#  Fused single-pass Q+K inverse-RMS Triton kernel
# =====================================================================
# Single Triton launch that reads each `(b, n)` row of `qkv` once and emits
# both `q_inv_rms[b, n]` and `k_inv_rms[b, n]`. Replaces two separate PyTorch
# scans (cast→square→sum→rsqrt) over `qkv[:, :, 0]` and `qkv[:, :, 1]`.
#
# Layout assumed: `qkv` is (B, N, 3, H, D) contiguous, so the C = H*D channels
# for a given (b, n, qkv_idx) live in a contiguous memory span.


@triton.jit
def _fused_qk_inv_rms_kernel(
    qkv_ptr,  # *T_in     (B, N, 3, H, D), contiguous
    q_inv_rms_ptr,  # *float32  (B, N)
    k_inv_rms_ptr,  # *float32  (B, N)
    N: tl.constexpr,
    C: tl.constexpr,  # H * D
    eps,
    BLOCK_C: tl.constexpr,
):
    """Helper function to fused qk inv rms kernel.

    Args:
        qkv_ptr: The qkv ptr.
        q_inv_rms_ptr: The q inv rms ptr.
        k_inv_rms_ptr: The k inv rms ptr.
        N: The n.
        C: The c.
        eps: The eps.
        BLOCK_C: The block c.
    """
    bn_id = tl.program_id(0)
    qkv_row_stride = 3 * C
    row_base = bn_id * qkv_row_stride
    q_base = row_base
    k_base = row_base + C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    q_vals = tl.load(qkv_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(qkv_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    q_sq = tl.sum(q_vals * q_vals, axis=0)
    k_sq = tl.sum(k_vals * k_vals, axis=0)

    inv_c = 1.0 / C
    q_inv = tl.rsqrt(q_sq * inv_c + eps)
    k_inv = tl.rsqrt(k_sq * inv_c + eps)

    tl.store(q_inv_rms_ptr + bn_id, q_inv)
    tl.store(k_inv_rms_ptr + bn_id, k_inv)


def fused_qk_inv_rms(
    qkv: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Triton fused Q+K inverse-RMS.

    Replaces ``(_precompute_inv_rms(qkv, 0, C, eps), _precompute_inv_rms(qkv, 1, C, eps))``
    with one launch that reads each ``(b, n)`` row of ``qkv`` exactly once.

    Args:
      qkv: (B, N, 3, H, D) contiguous tensor, any fp dtype.
      eps: RMSNorm epsilon.

    Returns:
      (q_inv_rms, k_inv_rms), each (B, N) float32 contiguous.
    """
    assert qkv.is_contiguous(), "qkv must be contiguous (B, N, 3, H, D)"
    assert qkv.dim() == 5 and qkv.shape[2] == 3, f"expected (B, N, 3, H, D), got {tuple(qkv.shape)}"
    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    k_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_qk_inv_rms_kernel[(B * N,)](
        qkv,
        q_inv_rms,
        k_inv_rms,
        N=N,
        C=C,
        eps=eps,
        BLOCK_C=BLOCK_C,
    )
    return q_inv_rms, k_inv_rms


@triton.jit
def _fused_bidi_merge_kernel(
    num_fwd_ptr,
    num_bwd_ptr,
    den_fwd_ptr,
    den_bwd_ptr,
    gate_ptr,
    out_ptr,
    B,
    N,
    H,
    D,
    eps,
    snum_b,
    snum_n,
    snum_h,
    snum_d,
    sden_b,
    sden_h,
    sden_n,
    APPLY_GATE: tl.constexpr,
    PRE_SUMMED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D
    mask_nd = mask_n[:, None] & mask_d[None, :]

    num_base = b * snum_b + offs_n[:, None] * snum_n + h * snum_h + offs_d[None, :] * snum_d
    nf = tl.load(num_fwd_ptr + num_base, mask=mask_nd, other=0.0).to(tl.float32)
    den_base = b * sden_b + h * sden_h + offs_n * sden_n
    df = tl.load(den_fwd_ptr + den_base, mask=mask_n, other=0.0).to(tl.float32)

    if PRE_SUMMED:
        num_total = nf
        den_total = df + eps
    else:
        nb = tl.load(num_bwd_ptr + num_base, mask=mask_nd, other=0.0).to(tl.float32)
        db = tl.load(den_bwd_ptr + den_base, mask=mask_n, other=0.0).to(tl.float32)
        num_total = nf + nb
        den_total = df + db + eps
    out_val = num_total / den_total[:, None]

    if APPLY_GATE:
        g = tl.load(gate_ptr + num_base, mask=mask_nd, other=0.0).to(tl.float32)
        silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
        out_val = out_val * silu_g

    tl.store(out_ptr + num_base, out_val.to(tl.bfloat16), mask=mask_nd)


def fused_bidi_merge(
    num_fwd: torch.Tensor,
    num_bwd: torch.Tensor | None,
    den_fwd: torch.Tensor,
    den_bwd: torch.Tensor | None,
    eps: float,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    pre_summed = num_bwd is None
    assert (num_bwd is None) == (den_bwd is None), "num_bwd/den_bwd must both be None or both provided"
    if not pre_summed:
        assert num_fwd.shape == num_bwd.shape and den_fwd.shape == den_bwd.shape
        assert num_fwd.dtype == num_bwd.dtype and den_fwd.dtype == den_bwd.dtype
    B, N, H, D = num_fwd.shape
    out = torch.empty(
        B, N, H, D, device=num_fwd.device, dtype=(torch.float32 if num_fwd.dtype == torch.float32 else torch.bfloat16)
    )
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 64
    grid = (B * H, triton.cdiv(N, BLOCK_N))
    if gate is not None:
        assert gate.shape == (B, N, H, D), f"gate shape {gate.shape} != {(B, N, H, D)}"
        gate_arg = gate
        apply_gate = 1
    else:
        gate_arg = num_fwd
        apply_gate = 0
    num_bwd_arg = num_bwd if num_bwd is not None else num_fwd
    den_bwd_arg = den_bwd if den_bwd is not None else den_fwd
    _fused_bidi_merge_kernel[grid](
        num_fwd,
        num_bwd_arg,
        den_fwd,
        den_bwd_arg,
        gate_arg,
        out,
        B,
        N,
        H,
        D,
        float(eps),
        num_fwd.stride(0),
        num_fwd.stride(1),
        num_fwd.stride(2),
        num_fwd.stride(3),
        den_fwd.stride(0),
        den_fwd.stride(1),
        den_fwd.stride(2),
        APPLY_GATE=apply_gate,
        PRE_SUMMED=1 if pre_summed else 0,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


# =====================================================================
#  Bidirectional GDN entry point (delegates to chunkwise)
# =====================================================================


def fused_bigdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) float32
    k_inv_rms: torch.Tensor,  # (B, N) float32
    q_norm_weight: torch.Tensor,  # (C,) float32
    k_norm_weight: torch.Tensor,  # (C,) float32
    rope_cos: torch.Tensor,  # (N, D) float32
    rope_sin: torch.Tensor,  # (N, D) float32
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional fused GDN. Returns ``(B, N, H, D)``.

    Thin entry point kept for call-site stability; delegates to
    :func:`fused_bigdn_bidi_chunkwise` from ``fused_gdn_chunkwise``.
    """
    from worldfoundry.base_models.diffusion_model.models.networks.sana.ops.fused_gdn_chunkwise import (
        fused_bigdn_bidi_chunkwise,
    )

    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
    )
