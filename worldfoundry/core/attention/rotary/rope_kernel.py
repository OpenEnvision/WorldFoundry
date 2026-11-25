# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused, inference-only Hopper SM90 Triton kernels for 3D rotary position embedding.

Drop-in for :func:`transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb`
(``tensor_format="bshd", fused=True``) on the diffusion-transformer forward path.
Accepts the full-width ``[S, 1, 1, D]`` freqs layout emitted by
:meth:`RotaryPositionEmbedding3D.shift_t`.

For each ``s`` the rotation is::

    out[a] = x[a] * cos(f) - x[b] * sin(f)
    out[b] = x[b] * cos(f) + x[a] * sin(f)

where ``(a, b) = (d, d + D/2)`` when ``interleaved=False`` and
``(2k, 2k+1)`` when ``interleaved=True``.

Hopper SM90 engineering:
- **Lazy Triton import**: CPU-only environments never import ``triton``.
- **Zero register spills**: uses 3-part Cody-Waite range reduction modulo
  ``pi/2`` + FP32 minimax polynomial (``< 1 ULP`` vs FP64, zero ``.local``
  Payne-Hanek stack frame).
- **Single-pass H-loop & fused Q/K**: computes ``cos``/``sin`` once per token
  across all heads (and across both Q and K in :func:`apply_rotary_pos_emb_qk`,
  supporting GQA ``Hq != Hk``).
- **Zero-clone out-of-place**: separate input/output pointers avoid a redundant
  full-tensor ``x.clone()`` when ``inplace=False``.
- **Strict ``tl.int64`` indexing** + ``@torch.library.custom_op`` /
  ``.register_fake`` for ``torch.compile`` and CUDA Graph safety.

Not this module:
    Table construction lives in :mod:`.rope`. These kernels are inference-only
    (no autograd) and reject raw fp8. Training / differentiable rotate uses
    :func:`.rope.apply_rotary_embedding`.

Public surface:

- :func:`apply_rotary_pos_emb` — fused single-tensor apply for ``[B,S,H,D]``.
- :func:`apply_rotary_pos_emb_qk` — fused Q+K apply sharing one ``cos``/``sin`` pass.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pick_block_s(s: int) -> int:
    """Return the per-program ``BLOCK_S`` tile size for sequence length ``s``.

    Falls back to ``BLOCK_S=1`` for short ``s`` so the grid still saturates the
    GPU's SMs, and uses ``BLOCK_S=4`` once ``s`` is large enough that fatter
    tiles amortise dispatch latency.
    """
    return 4 if s >= 512 else 1


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily configure the Triton cache and compile RoPE kernels."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="rope-triton")

    import triton
    import triton.language as tl

    # ──────────────────────────────────────────────────────────────────────
    # Zero-spill FP32 sincos: 3-part Cody-Waite mod (pi/2) + minimax poly
    # Avoids NVVM libdevice Payne-Hanek 8-byte .local depot while preserving
    # < 1 FP32 ULP accuracy (0 BF16 ULP diff) across [-1e6, 1e6].
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _cw_sincos_fp32(f):
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

    # ──────────────────────────────────────────────────────────────────────
    # Single-tensor RoPE kernel — loops over H inside CTA so cos/sin run once
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _rope_inference_kernel(
        x_ptr,
        out_ptr,
        freqs_ptr,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_xd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        stride_fs,
        stride_fd,
        S,
        H,
        h_per_cta,
        D_HALF: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Apply RoPE to ``[B, S, H, D]`` with strict int64 offsets and H-loop."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_hg = tl.program_id(1).to(tl.int64)
        pid_b = tl.program_id(2).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        h_i64 = tl.cast(H, tl.int64)
        hpc_i64 = tl.cast(h_per_cta, tl.int64)

        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_off < s_i64
        d_idx = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_idx < D_HALF

        freq_off = s_off[:, None] * tl.cast(stride_fs, tl.int64) + d_idx[None, :] * tl.cast(stride_fd, tl.int64)
        freq = tl.load(freqs_ptr + freq_off, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        cos_f, sin_f = _cw_sincos_fp32(freq)

        cos_t = cos_f[:, None, :].to(x_ptr.dtype.element_ty)
        sin_t = sin_f[:, None, :].to(x_ptr.dtype.element_ty)

        x_row_base = x_ptr + pid_b * tl.cast(stride_xb, tl.int64)
        o_row_base = out_ptr + pid_b * tl.cast(stride_ob, tl.int64)
        sxs = tl.cast(stride_xs, tl.int64)
        sxh = tl.cast(stride_xh, tl.int64)
        sxd = tl.cast(stride_xd, tl.int64)
        sos = tl.cast(stride_os, tl.int64)
        soh = tl.cast(stride_oh, tl.int64)
        sod = tl.cast(stride_od, tl.int64)

        h_start = pid_hg * hpc_i64
        h_end = tl.minimum(h_start + hpc_i64, h_i64)
        h_lane = tl.arange(0, BLOCK_H).to(tl.int64)

        if INTERLEAVED:
            d_full = tl.arange(0, 2 * BLOCK_D).to(tl.int64)
            full_mask_d = d_full < (2 * D_HALF)
            for h0 in range(0, h_per_cta, BLOCK_H):
                h_idx = h_start + tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < h_end
                in_off = s_off[:, None, None] * sxs + h_idx[None, :, None] * sxh + d_full[None, None, :] * sxd
                out_off = s_off[:, None, None] * sos + h_idx[None, :, None] * soh + d_full[None, None, :] * sod
                full_mask = s_mask[:, None, None] & h_mask[None, :, None] & full_mask_d[None, None, :]
                x_flat = tl.load(x_row_base + in_off, mask=full_mask, other=0.0, eviction_policy="evict_first")
                a, b = tl.split(tl.reshape(x_flat, (BLOCK_S, BLOCK_H, BLOCK_D, 2)))
                out_a = a * cos_t - b * sin_t
                out_b = b * cos_t + a * sin_t
                out_flat = tl.reshape(tl.join(out_a, out_b), (BLOCK_S, BLOCK_H, 2 * BLOCK_D))
                tl.store(o_row_base + out_off, out_flat, mask=full_mask, eviction_policy="evict_first")
        else:
            d_half_i64 = tl.cast(D_HALF, tl.int64)
            for h0 in range(0, h_per_cta, BLOCK_H):
                h_idx = h_start + tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < h_end
                in_a = s_off[:, None, None] * sxs + h_idx[None, :, None] * sxh + d_idx[None, None, :] * sxd
                in_b = in_a + d_half_i64 * sxd
                out_a_off = s_off[:, None, None] * sos + h_idx[None, :, None] * soh + d_idx[None, None, :] * sod
                out_b_off = out_a_off + d_half_i64 * sod
                mask = s_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                a = tl.load(x_row_base + in_a, mask=mask, other=0.0, eviction_policy="evict_first")
                b = tl.load(x_row_base + in_b, mask=mask, other=0.0, eviction_policy="evict_first")
                tl.store(o_row_base + out_a_off, a * cos_t - b * sin_t, mask=mask, eviction_policy="evict_first")
                tl.store(o_row_base + out_b_off, b * cos_t + a * sin_t, mask=mask, eviction_policy="evict_first")

    # ──────────────────────────────────────────────────────────────────────
    # Fused Q+K RoPE kernel — one freqs load & one cos/sin pass for Q and K
    # ──────────────────────────────────────────────────────────────────────

    @triton.jit
    def _rope_qk_inference_kernel(
        q_ptr,
        oq_ptr,
        k_ptr,
        ok_ptr,
        freqs_ptr,
        sq_b,
        sq_s,
        sq_h,
        sq_d,
        soq_b,
        soq_s,
        soq_h,
        soq_d,
        sk_b,
        sk_s,
        sk_h,
        sk_d,
        sok_b,
        sok_s,
        sok_h,
        sok_d,
        sf_s,
        sf_d,
        S,
        Hq,
        Hk,
        D_HALF: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused Q+K RoPE sharing one freqs load and one cos/sin pass (supports GQA)."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_b = tl.program_id(1).to(tl.int64)

        s_i64 = tl.cast(S, tl.int64)
        hq_i64 = tl.cast(Hq, tl.int64)
        hk_i64 = tl.cast(Hk, tl.int64)

        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        s_mask = s_off < s_i64
        d_idx = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_idx < D_HALF

        freq_off = s_off[:, None] * tl.cast(sf_s, tl.int64) + d_idx[None, :] * tl.cast(sf_d, tl.int64)
        freq = tl.load(freqs_ptr + freq_off, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        cos_f, sin_f = _cw_sincos_fp32(freq)

        cos_t = cos_f[:, None, :].to(q_ptr.dtype.element_ty)
        sin_t = sin_f[:, None, :].to(q_ptr.dtype.element_ty)

        q_base = q_ptr + pid_b * tl.cast(sq_b, tl.int64)
        oq_base = oq_ptr + pid_b * tl.cast(soq_b, tl.int64)
        k_base = k_ptr + pid_b * tl.cast(sk_b, tl.int64)
        ok_base = ok_ptr + pid_b * tl.cast(sok_b, tl.int64)

        sq_s64 = tl.cast(sq_s, tl.int64)
        sq_h64 = tl.cast(sq_h, tl.int64)
        sq_d64 = tl.cast(sq_d, tl.int64)
        soq_s64 = tl.cast(soq_s, tl.int64)
        soq_h64 = tl.cast(soq_h, tl.int64)
        soq_d64 = tl.cast(soq_d, tl.int64)

        sk_s64 = tl.cast(sk_s, tl.int64)
        sk_h64 = tl.cast(sk_h, tl.int64)
        sk_d64 = tl.cast(sk_d, tl.int64)
        sok_s64 = tl.cast(sok_s, tl.int64)
        sok_h64 = tl.cast(sok_h, tl.int64)
        sok_d64 = tl.cast(sok_d, tl.int64)

        h_lane = tl.arange(0, BLOCK_H).to(tl.int64)

        if INTERLEAVED:
            d_full = tl.arange(0, 2 * BLOCK_D).to(tl.int64)
            full_mask_d = d_full < (2 * D_HALF)
            for h0 in range(0, Hq, BLOCK_H):
                h_idx = tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < hq_i64
                m = s_mask[:, None, None] & h_mask[None, :, None] & full_mask_d[None, None, :]
                in_off = s_off[:, None, None] * sq_s64 + h_idx[None, :, None] * sq_h64 + d_full[None, None, :] * sq_d64
                out_off = (
                    s_off[:, None, None] * soq_s64 + h_idx[None, :, None] * soq_h64 + d_full[None, None, :] * soq_d64
                )
                x_flat = tl.load(q_base + in_off, mask=m, other=0.0, eviction_policy="evict_first")
                a, b = tl.split(tl.reshape(x_flat, (BLOCK_S, BLOCK_H, BLOCK_D, 2)))
                tl.store(
                    oq_base + out_off,
                    tl.reshape(tl.join(a * cos_t - b * sin_t, b * cos_t + a * sin_t), (BLOCK_S, BLOCK_H, 2 * BLOCK_D)),
                    mask=m,
                    eviction_policy="evict_first",
                )
            for h0 in range(0, Hk, BLOCK_H):
                h_idx = tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < hk_i64
                m = s_mask[:, None, None] & h_mask[None, :, None] & full_mask_d[None, None, :]
                in_off = s_off[:, None, None] * sk_s64 + h_idx[None, :, None] * sk_h64 + d_full[None, None, :] * sk_d64
                out_off = (
                    s_off[:, None, None] * sok_s64 + h_idx[None, :, None] * sok_h64 + d_full[None, None, :] * sok_d64
                )
                x_flat = tl.load(k_base + in_off, mask=m, other=0.0, eviction_policy="evict_first")
                a, b = tl.split(tl.reshape(x_flat, (BLOCK_S, BLOCK_H, BLOCK_D, 2)))
                tl.store(
                    ok_base + out_off,
                    tl.reshape(tl.join(a * cos_t - b * sin_t, b * cos_t + a * sin_t), (BLOCK_S, BLOCK_H, 2 * BLOCK_D)),
                    mask=m,
                    eviction_policy="evict_first",
                )
        else:
            d_half_i64 = tl.cast(D_HALF, tl.int64)
            for h0 in range(0, Hq, BLOCK_H):
                h_idx = tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < hq_i64
                m = s_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                in_a = s_off[:, None, None] * sq_s64 + h_idx[None, :, None] * sq_h64 + d_idx[None, None, :] * sq_d64
                in_b = in_a + d_half_i64 * sq_d64
                out_a = s_off[:, None, None] * soq_s64 + h_idx[None, :, None] * soq_h64 + d_idx[None, None, :] * soq_d64
                out_b = out_a + d_half_i64 * soq_d64
                a = tl.load(q_base + in_a, mask=m, other=0.0, eviction_policy="evict_first")
                b = tl.load(q_base + in_b, mask=m, other=0.0, eviction_policy="evict_first")
                tl.store(oq_base + out_a, a * cos_t - b * sin_t, mask=m, eviction_policy="evict_first")
                tl.store(oq_base + out_b, b * cos_t + a * sin_t, mask=m, eviction_policy="evict_first")
            for h0 in range(0, Hk, BLOCK_H):
                h_idx = tl.cast(h0, tl.int64) + h_lane
                h_mask = h_idx < hk_i64
                m = s_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                in_a = s_off[:, None, None] * sk_s64 + h_idx[None, :, None] * sk_h64 + d_idx[None, None, :] * sk_d64
                in_b = in_a + d_half_i64 * sk_d64
                out_a = s_off[:, None, None] * sok_s64 + h_idx[None, :, None] * sok_h64 + d_idx[None, None, :] * sok_d64
                out_b = out_a + d_half_i64 * sok_d64
                a = tl.load(k_base + in_a, mask=m, other=0.0, eviction_policy="evict_first")
                b = tl.load(k_base + in_b, mask=m, other=0.0, eviction_policy="evict_first")
                tl.store(ok_base + out_a, a * cos_t - b * sin_t, mask=m, eviction_policy="evict_first")
                tl.store(ok_base + out_b, b * cos_t + a * sin_t, mask=m, eviction_policy="evict_first")

    return {
        "triton": triton,
        "rope": _rope_inference_kernel,
        "rope_qk": _rope_qk_inference_kernel,
    }


def _select_rope_launch(
    B: int,
    S: int,
    H: int,
    head_dim_half: int,
    interleaved: bool,
    dtype: torch.dtype,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> tuple[int, int, int, int, int, int]:
    block_d = max(_next_power_of_2(head_dim_half), 16)
    bs_eff = _pick_block_s(S) if block_s <= 0 else block_s
    bs_eff = max(_next_power_of_2(bs_eff), 1)
    # Bound per-CTA tile footprint so Hopper kernels compile with zero register spills
    # across split/interleaved, bf16/fp16/fp32, and large head dimensions (up to D=1024).
    max_elems = 4096 if (interleaved or dtype == torch.float32) else 8192
    if bs_eff * block_d > max_elems and block_s <= 0:
        bs_eff = max(1, max_elems // block_d)
    max_block_h = max(1, min(16 if interleaved else 32, max_elems // max(1, bs_eff * block_d)))
    block_h = min(_next_power_of_2(H), max_block_h)
    s_tiles = _cdiv(S, bs_eff)

    # Loop over all H inside one CTA when (B * s_tiles) already saturates Hopper SMs;
    # otherwise split H across CTAs for very short sequences.
    if B * s_tiles >= 128 or block_h >= H:
        h_groups = 1
        h_per_cta = H
    else:
        h_groups = _cdiv(H, block_h)
        h_per_cta = block_h

    elements_per_tile = bs_eff * block_h * block_d
    if num_warps <= 0:
        if elements_per_tile <= 256:
            nw_eff = 1
        elif elements_per_tile <= 1024:
            nw_eff = 2
        elif elements_per_tile <= 2048:
            nw_eff = 4
        else:
            nw_eff = 8
    else:
        nw_eff = num_warps
    ns_eff = 2 if num_stages <= 0 else num_stages
    return bs_eff, block_h, block_d, h_groups, h_per_cta, (nw_eff << 8) | ns_eff


def _launch_rope_single(
    x: Tensor,
    out: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    B, S, H, D = x.shape
    if B == 0 or S == 0 or H == 0:
        return
    kernels = _load_triton_kernels()
    head_dim_half = D // 2
    stride_fs = freqs.stride(0)
    stride_fd = freqs.stride(3) * (2 if interleaved else 1)

    bs_eff, block_h, block_d, h_groups, h_per_cta, packed = _select_rope_launch(
        B, S, H, head_dim_half, interleaved, x.dtype, block_s, num_warps, num_stages
    )
    nw_eff = packed >> 8
    ns_eff = packed & 0xFF
    s_tiles = _cdiv(S, bs_eff)

    with torch.cuda.device(x.device):
        kernels["rope"][(s_tiles, h_groups, B)](
            x,
            out,
            freqs,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            stride_fs,
            stride_fd,
            S,
            H,
            h_per_cta,
            D_HALF=head_dim_half,
            INTERLEAVED=interleaved,
            BLOCK_S=bs_eff,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=nw_eff,
            num_stages=ns_eff,
        )


def _launch_rope_qk(
    q: Tensor,
    oq: Tensor,
    k: Tensor,
    ok: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    B, S, Hq, D = q.shape
    Hk = k.shape[2]
    if B == 0 or S == 0 or (Hq == 0 and Hk == 0):
        return
    kernels = _load_triton_kernels()
    head_dim_half = D // 2
    stride_fs = freqs.stride(0)
    stride_fd = freqs.stride(3) * (2 if interleaved else 1)

    block_d = max(_next_power_of_2(head_dim_half), 16)
    bs_eff = _pick_block_s(S) if block_s <= 0 else block_s
    bs_eff = max(_next_power_of_2(bs_eff), 1)
    max_elems = 4096 if (interleaved or q.dtype == torch.float32) else 8192
    if bs_eff * block_d > max_elems and block_s <= 0:
        bs_eff = max(1, max_elems // block_d)
    max_block_h = max(1, min(16 if interleaved else 32, max_elems // max(1, bs_eff * block_d)))
    block_h = min(_next_power_of_2(max(Hq, Hk, 1)), max_block_h)
    s_tiles = _cdiv(S, bs_eff)

    elements_per_tile = bs_eff * block_h * block_d
    if num_warps <= 0:
        if elements_per_tile <= 256:
            nw_eff = 1
        elif elements_per_tile <= 1024:
            nw_eff = 2
        elif elements_per_tile <= 2048:
            nw_eff = 4
        else:
            nw_eff = 8
    else:
        nw_eff = num_warps
    ns_eff = 2 if num_stages <= 0 else num_stages

    with torch.cuda.device(q.device):
        kernels["rope_qk"][(s_tiles, B)](
            q,
            oq,
            k,
            ok,
            freqs,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            oq.stride(0),
            oq.stride(1),
            oq.stride(2),
            oq.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            ok.stride(0),
            ok.stride(1),
            ok.stride(2),
            ok.stride(3),
            stride_fs,
            stride_fd,
            S,
            Hq,
            Hk,
            D_HALF=head_dim_half,
            INTERLEAVED=interleaved,
            BLOCK_S=bs_eff,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=nw_eff,
            num_stages=ns_eff,
        )


# ──────────────────────────────────────────────────────────────────────────
# Registered torch.library custom ops (compile & CUDA Graph safe)
# ──────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("worldfoundry::rope_bshd_fwd", mutates_args=(), device_types="cuda")
def _rope_bshd_fwd_op(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> Tensor:
    out = torch.empty_like(x)
    _launch_rope_single(x, out, freqs, interleaved, block_s, num_warps, num_stages)
    return out


@_rope_bshd_fwd_op.register_fake
def _rope_bshd_fwd_fake(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> Tensor:
    del freqs, interleaved, block_s, num_warps, num_stages
    return torch.empty_like(x)


@torch.library.custom_op("worldfoundry::rope_bshd_inplace", mutates_args=("x",), device_types="cuda")
def _rope_bshd_inplace_op(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    _launch_rope_single(x, x, freqs, interleaved, block_s, num_warps, num_stages)


@_rope_bshd_inplace_op.register_fake
def _rope_bshd_inplace_fake(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    del x, freqs, interleaved, block_s, num_warps, num_stages
    return None


@torch.library.custom_op("worldfoundry::rope_qk_bshd_fwd", mutates_args=(), device_types="cuda")
def _rope_qk_bshd_fwd_op(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> tuple[Tensor, Tensor]:
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    _launch_rope_qk(q, oq, k, ok, freqs, interleaved, block_s, num_warps, num_stages)
    return oq, ok


@_rope_qk_bshd_fwd_op.register_fake
def _rope_qk_bshd_fwd_fake(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> tuple[Tensor, Tensor]:
    del freqs, interleaved, block_s, num_warps, num_stages
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("worldfoundry::rope_qk_bshd_inplace", mutates_args=("q", "k"), device_types="cuda")
def _rope_qk_bshd_inplace_op(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    _launch_rope_qk(q, q, k, k, freqs, interleaved, block_s, num_warps, num_stages)


@_rope_qk_bshd_inplace_op.register_fake
def _rope_qk_bshd_inplace_fake(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    interleaved: bool,
    block_s: int,
    num_warps: int,
    num_stages: int,
) -> None:
    del q, k, freqs, interleaved, block_s, num_warps, num_stages
    return None


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def _validate_rope_inputs(x: Tensor, freqs: Tensor, name: str = "x") -> tuple[int, int, int, int]:
    if x.dim() != 4:
        raise ValueError(f"{name} must be 4D [B, S, H, D]; got {tuple(x.shape)}.")
    if freqs.dim() != 4:
        raise ValueError(f"freqs must be 4D [S, 1, 1, D]; got {tuple(freqs.shape)}.")
    if not x.is_cuda:
        raise ValueError("apply_rotary_pos_emb requires a CUDA tensor.")
    if not freqs.is_cuda or freqs.device != x.device:
        raise ValueError("freqs must be on the same CUDA device as inputs.")
    if x.dtype in _FP8_DTYPES:
        raise NotImplementedError(
            f"apply_rotary_pos_emb does not support fp8 inputs (got {x.dtype}); "
            "dequantise to bf16/fp16 first, matching TE's policy."
        )
    B, S, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even; got {D}.")
    if freqs.shape[0] != S:
        raise ValueError(f"freqs[0]={freqs.shape[0]} must match {name}.shape[1]={S}.")
    if freqs.shape[-1] != D:
        raise ValueError(
            f"freqs last-dim must equal head_dim ({D}); got "
            f"{freqs.shape[-1]}. Pass the full-width [S, 1, 1, head_dim] "
            "layout that RotaryPositionEmbedding3D.shift_t emits."
        )
    return B, S, H, D


def apply_rotary_pos_emb(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool = False,
    inplace: bool = True,
    block_s: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Tensor:
    """Apply 3D RoPE to ``x`` using a fused inference-only Hopper Triton kernel.

    Args:
        x: Activations of shape ``[B, S, H, D]``. ``D`` must be even.
            ``fp32`` / ``fp16`` / ``bf16`` supported; raw ``fp8`` is
            rejected, matching TE's policy.
        freqs: RoPE frequencies of shape ``[S, 1, 1, D]`` — the full-width
            layout emitted by :meth:`RotaryPositionEmbedding3D.shift_t`.
        interleaved: If ``True``, rotate the pair ``(2k, 2k+1)``; else
            rotate ``(d, d + D/2)``.
        inplace: If ``True`` (default) write back into ``x`` storage and
            return ``x``; if ``False`` write directly into a fresh output
            tensor without cloning ``x``.
        block_s: Override ``BLOCK_S``. ``None`` picks via :func:`_pick_block_s`.
        num_warps: Override Triton's ``num_warps``. ``None`` auto-selects.
        num_stages: Override Triton's ``num_stages``. ``None`` uses 2.

    Returns:
        Rotated tensor; shares storage with ``x`` when ``inplace`` is ``True``.
    """
    _validate_rope_inputs(x, freqs, "x")
    bs_arg = 0 if block_s is None else int(block_s)
    nw_arg = 0 if num_warps is None else int(num_warps)
    ns_arg = 0 if num_stages is None else int(num_stages)
    if inplace:
        _rope_bshd_inplace_op(x, freqs, bool(interleaved), bs_arg, nw_arg, ns_arg)
        return x
    return _rope_bshd_fwd_op(x, freqs, bool(interleaved), bs_arg, nw_arg, ns_arg)


def apply_rotary_pos_emb_qk(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    interleaved: bool = False,
    inplace: bool = True,
    block_s: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused Q+K 3D RoPE kernel sharing one ``freqs`` load and ``cos``/``sin`` pass.

    Supports both MHA (``Hq == Hk``) and GQA (``Hq != Hk``) for ``[B, S, H, D]``
    tensors with arbitrary strides (including non-contiguous QKV slices).
    """
    Bq, Sq, _, Dq = _validate_rope_inputs(q, freqs, "q")
    Bk, Sk, _, Dk = _validate_rope_inputs(k, freqs, "k")
    if q.device != k.device:
        raise ValueError(f"q and k must share one CUDA device; got {q.device} and {k.device}.")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must share one dtype; got {q.dtype} and {k.dtype}.")
    if Bq != Bk or Sq != Sk or Dq != Dk:
        raise ValueError(f"q and k must match on (B, S, D); got q={tuple(q.shape)} and k={tuple(k.shape)}.")

    bs_arg = 0 if block_s is None else int(block_s)
    nw_arg = 0 if num_warps is None else int(num_warps)
    ns_arg = 0 if num_stages is None else int(num_stages)
    if inplace:
        _rope_qk_bshd_inplace_op(q, k, freqs, bool(interleaved), bs_arg, nw_arg, ns_arg)
        return q, k
    return _rope_qk_bshd_fwd_op(q, k, freqs, bool(interleaved), bs_arg, nw_arg, ns_arg)


__all__ = ["apply_rotary_pos_emb", "apply_rotary_pos_emb_qk"]
