# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton 3D neighborhood attention (NATTEN ``na3d`` semantics).

Provides a 3D-tiled Hopper SM90 WGMMA Triton implementation for BF16/FP16
volumes (tiling queries in ``(BT, BH, BW)`` and keys/values in power-of-2
``(BKT, BKH, BKW)`` blocks with zero runtime integer divisions, ``uint32``
range checks, ``exp2`` online softmax, and in-place ``tl.dot(..., acc=acc)``
accumulation), alongside the scalar-T/H fallback kernel for FP32, wide head
dimensions, or tiny boundary shapes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton requires module-level JIT globals to be constexpr instances (not annotations).
_NEG_INF = tl.constexpr(-3.0e38)
_LOG2E = 1.4426950408889634


@triton.jit
def _na3d_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)

    # (frame, batch, head) share axis 2 and H owns axis 1, rather than folding T*H into axis 1:
    # gridDim.y/z are capped at 65535 and T*H passes that at production tile sizes (392*272 =
    # 106624), which fails the launch outright with "Triton Error [CUDA]: invalid argument".
    # Same layout as joint_triton, for the same reason.
    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    # int64 throughout: every individual stride fits in int32, so Triton specializes them there,
    # but t_q * s_t reaches ~1.3e10 on a 392-frame tile and would wrap -- an illegal memory access
    # rather than a wrong answer.
    base = (pid_bn // num_heads).to(tl.int64) * s_b + (pid_bn % num_heads).to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    if causal_t:
        t_lo = tl.maximum(t_q - kt + 1, 0)
        t_hi = t_q + 1
    else:
        t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_hi = t_lo + kt
    if causal_h:
        h_lo = tl.maximum(h_q - kh + 1, 0)
        h_hi = h_q + 1
    else:
        h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_hi = h_lo + kh

    w_q = tl.where(w_valid, w_off, w_size - 1)
    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.maximum(blk_first - kw + 1, 0)
        w_hi = blk_last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_end = w_start + kw
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.minimum(tl.maximum(blk_first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(blk_last - kw // 2, 0), w_size - kw) + kw

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for tk in range(t_lo, t_hi):
        for hk in range(h_lo, h_hi):
            plane = base + tk.to(tl.int64) * s_t + hk.to(tl.int64) * s_h
            for wk0 in range(w_lo, w_hi, block_k):
                wk = wk0 + tl.arange(0, block_k)
                kmask = wk < w_hi
                kv_ptrs = plane + wk[:, None] * s_w + d_off[None, :]
                kv_mask = kmask[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    s = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
                else:
                    s = tl.dot(q_blk, tl.trans(k_blk)) * scale
                vis = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :]
                s = tl.where(vis, s, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
                m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = out_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


@triton.jit
def _na3d_3d_tiled_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale_log2e,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BKT: tl.constexpr,
    BKH: tl.constexpr,
    BKW: tl.constexpr,
    SINGLE_W: tl.constexpr,
):
    """3D-tiled Hopper WGMMA NA3D kernel (BQ = BT*BH*BW = 64, BK = BKT*BKH*BKW = 64).

    Tiles queries across ``(T, H, W)`` to share K/V halo loads across neighboring
    rows and frames, uses compile-time power-of-2 bit decomposition for ``BK``
    (zero runtime integer divisions), ``exp2`` online softmax, and in-place
    ``tl.dot(..., acc=acc)`` accumulation. All byte offsets use ``tl.int64``.
    """
    BQ: tl.constexpr = BT * BH * BW
    BK: tl.constexpr = BKT * BKH * BKW

    # Keep gridDim.y = cdiv(H, BH) and gridDim.z = cdiv(T, BT) well under 65535;
    # place (B*NH) innermost on axis 0 so concurrent CTAs share L2 cache lines
    # across heads at the same (T, H, W) tile.
    pid0 = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)
    pid_bn = pid0 % bn_count
    pid_w = pid0 // bn_count

    t0 = pid_t * BT
    h0 = pid_h * BH
    w0 = pid_w * BW

    base = (pid_bn // num_heads).to(tl.int64) * s_b + (pid_bn % num_heads).to(tl.int64) * s_n

    q_lin = tl.arange(0, BQ)
    qt_rel = q_lin // (BH * BW)
    q_rem = q_lin % (BH * BW)
    qh_rel = q_rem // BW
    qw_rel = q_rem % BW

    t_off = t0 + qt_rel
    h_off = h0 + qh_rel
    w_off = w0 + qw_rel
    q_valid = (t_off < t_size) & (h_off < h_size) & (w_off < w_size)

    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        q_ptr
        + base
        + t_off.to(tl.int64)[:, None] * s_t
        + h_off.to(tl.int64)[:, None] * s_h
        + w_off.to(tl.int64)[:, None] * s_w
        + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    t_q = tl.where(t_off < t_size, t_off, t_size - 1)
    h_q = tl.where(h_off < h_size, h_off, h_size - 1)
    w_q = tl.where(w_off < w_size, w_off, w_size - 1)

    t_first = tl.minimum(t0, t_size - 1)
    t_last = tl.minimum(t0 + BT - 1, t_size - 1)
    h_first = tl.minimum(h0, h_size - 1)
    h_last = tl.minimum(h0 + BH - 1, h_size - 1)
    w_first = tl.minimum(w0, w_size - 1)
    w_last = tl.minimum(w0 + BW - 1, w_size - 1)

    if causal_t:
        t_start = tl.maximum(t_q - kt + 1, 0)
        t_end = t_q + 1
        t_lo = tl.maximum(t_first - kt + 1, 0)
        t_hi = t_last + 1
    else:
        t_start = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_lo = tl.minimum(tl.maximum(t_first - kt // 2, 0), t_size - kt)
        t_hi = tl.minimum(tl.maximum(t_last - kt // 2, 0), t_size - kt) + kt

    if causal_h:
        h_start = tl.maximum(h_q - kh + 1, 0)
        h_end = h_q + 1
        h_lo = tl.maximum(h_first - kh + 1, 0)
        h_hi = h_last + 1
    else:
        h_start = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_lo = tl.minimum(tl.maximum(h_first - kh // 2, 0), h_size - kh)
        h_hi = tl.minimum(tl.maximum(h_last - kh // 2, 0), h_size - kh) + kh

    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        w_lo = tl.maximum(w_first - kw + 1, 0)
        w_hi = w_last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_lo = tl.minimum(tl.maximum(w_first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(w_last - kw // 2, 0), w_size - kw) + kw

    k_lin = tl.arange(0, BK)
    kt_rel = k_lin // (BKH * BKW)
    k_rem = k_lin % (BKH * BKW)
    kh_rel = k_rem // BKW
    kw_rel = k_rem % BKW

    m_i = tl.full((BQ,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BQ,), dtype=tl.float32)
    acc = tl.zeros((BQ, hd_pad), dtype=tl.float32)

    if SINGLE_W:
        wk = w_lo + kw_rel
        w_in = wk < w_hi
        wk_ptr = base + wk.to(tl.int64) * s_w
        for tk0 in range(t_lo, t_hi, BKT):
            tk = tk0 + kt_rel
            t_in = tk < t_hi
            tk_ptr = wk_ptr + tk.to(tl.int64) * s_t
            for hk0 in range(h_lo, h_hi, BKH):
                hk = hk0 + kh_rel
                h_in = hk < h_hi
                k_valid = t_in & h_in & w_in
                kv_ptrs = (tk_ptr + hk.to(tl.int64) * s_h)[:, None] + d_off[None, :]
                kv_mask = k_valid[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                s = tl.dot(q_blk, tl.trans(k_blk)) * scale_log2e
                if causal_t:
                    vt = (tk[None, :] >= t_start[:, None]) & (tk[None, :] < t_end[:, None])
                else:
                    vt = (tk[None, :] - t_start[:, None]).to(tl.uint32) < kt
                if causal_h:
                    vh = (hk[None, :] >= h_start[:, None]) & (hk[None, :] < h_end[:, None])
                else:
                    vh = (hk[None, :] - h_start[:, None]).to(tl.uint32) < kh
                if causal_w:
                    vw = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None])
                else:
                    vw = (wk[None, :] - w_start[:, None]).to(tl.uint32) < kw
                vis = k_valid[None, :] & vt & vh & vw
                s = tl.where(vis, s, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_new
    else:
        for tk0 in range(t_lo, t_hi, BKT):
            tk = tk0 + kt_rel
            t_in = tk < t_hi
            tk_ptr = base + tk.to(tl.int64) * s_t
            for hk0 in range(h_lo, h_hi, BKH):
                hk = hk0 + kh_rel
                h_in = hk < h_hi
                th_ptr = tk_ptr + hk.to(tl.int64) * s_h
                for wk0 in range(w_lo, w_hi, BKW):
                    wk = wk0 + kw_rel
                    w_in = wk < w_hi
                    k_valid = t_in & h_in & w_in
                    kv_ptrs = (th_ptr + wk.to(tl.int64) * s_w)[:, None] + d_off[None, :]
                    kv_mask = k_valid[:, None] & d_mask[None, :]
                    k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                    s = tl.dot(q_blk, tl.trans(k_blk)) * scale_log2e
                    if causal_t:
                        vt = (tk[None, :] >= t_start[:, None]) & (tk[None, :] < t_end[:, None])
                    else:
                        vt = (tk[None, :] - t_start[:, None]).to(tl.uint32) < kt
                    if causal_h:
                        vh = (hk[None, :] >= h_start[:, None]) & (hk[None, :] < h_end[:, None])
                    else:
                        vh = (hk[None, :] - h_start[:, None]).to(tl.uint32) < kh
                    if causal_w:
                        vw = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None])
                    else:
                        vw = (wk[None, :] - w_start[:, None]).to(tl.uint32) < kw
                    vis = k_valid[None, :] & vt & vh & vw
                    s = tl.where(vis, s, _NEG_INF)
                    m_new = tl.maximum(m_i, tl.max(s, 1))
                    alpha = tl.math.exp2(m_i - m_new)
                    p = tl.math.exp2(s - m_new[:, None])
                    l_i = l_i * alpha + tl.sum(p, 1)
                    v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                    acc = acc * alpha[:, None]
                    acc = tl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                    m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = (
        out_ptr
        + base
        + t_off.to(tl.int64)[:, None] * s_t
        + h_off.to(tl.int64)[:, None] * s_h
        + w_off.to(tl.int64)[:, None] * s_w
        + d_off[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=q_valid[:, None] & d_mask[None, :])


def _select_3d_tile(t: int, h: int, w: int, kt: int, kh: int, kw: int) -> tuple[int, int, int, int, int, int]:
    """Choose ``(BT, BH, BW, BKT, BKH, BKW)`` with ``BQ = BK = 64`` for Hopper WGMMA."""
    if kt >= 7 and kh >= 7 and kw >= 7:
        return (4, 4, 4, 2, 2, 16)
    if kt <= 5 and kw <= 5 and h >= 8 and w >= 4:
        return (2, 8, 4, 2, 4, 8)
    if w >= 8 and h >= 4:
        return (2, 4, 8, 2, 2, 16)
    return (4, 4, 4, 2, 2, 16)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int] | tuple[int, ...],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` tensors."""
    batch, t, h, w, nh, hd = q.shape
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kt, kh, kw = (k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, (t, h, w), strict=True))
    if scale is None:
        scale = hd**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    if out.numel() == 0:
        return out

    hd_p = max(16, triton.next_power_of_2(hd))

    use_3d_tiled = q.dtype in (torch.bfloat16, torch.float16) and hd_p <= 128 and t >= 2 and h >= 4 and w >= 4
    if use_3d_tiled:
        bt, bh, bw, bkt, bkh, bkw = _select_3d_tile(t, h, w, kt, kh, kw)
        grid_3d = (triton.cdiv(w, bw) * batch * nh, triton.cdiv(h, bh), triton.cdiv(t, bt))
        if grid_3d[1] <= 65535 and grid_3d[2] <= 65535:
            _na3d_3d_tiled_kernel[grid_3d](
                q,
                k,
                v,
                out,
                t,
                h,
                w,
                nh,
                batch * nh,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                q.stride(4),
                float(scale) * _LOG2E,
                kt=kt,
                kh=kh,
                kw=kw,
                causal_t=causal[0],
                causal_h=causal[1],
                causal_w=causal[2],
                hd=hd,
                hd_pad=hd_p,
                BT=bt,
                BH=bh,
                BW=bw,
                BKT=bkt,
                BKH=bkh,
                BKW=bkw,
                SINGLE_W=(bw + kw - 1 <= bkw),
                num_warps=4,
                num_stages=2,
            )
            return out

    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))

    grid = (triton.cdiv(w, block_q), h, t * batch * nh)
    _na3d_kernel[grid](
        q,
        k,
        v,
        out,
        t,
        h,
        w,
        nh,
        batch * nh,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=causal[0],
        causal_h=causal[1],
        causal_w=causal[2],
        hd=hd,
        hd_pad=hd_p,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    return out
