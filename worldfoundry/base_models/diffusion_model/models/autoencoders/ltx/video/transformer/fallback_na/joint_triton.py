"""Triton joint (video + keyframe) 3D neighborhood attention.
Same semantics as :mod:`joint_eager` -- see its module docstring for the window rule and
the two query kinds -- but flash-style: keeps ``m/l/acc`` in registers and never
materializes scores.

Provides a 3D-tiled Hopper SM90 WGMMA fast path for BF16/FP16 volumes
(``_joint_video_3d_tiled_kernel`` with ``BQ=BK=64`` + fast same-plane keyframe slot
folding, and 2D-tiled ``_joint_keyframe_2d_tiled_kernel`` with ``BQ=BK=64``) alongside
the scalar-T/H fallback kernels for FP32, wide head dims, or tiny shapes.

Three semantic details are load-bearing across both fast and fallback paths:
1. **Clamp-and-mask, not NATTEN's inward shift.** Boundary queries see fewer keys and
   the softmax renormalizes over valid in-bounds taps only.
2. **The slot loop shares the same ``m_i``/``l_i``/``acc``.** Cross-stream keys are
   folded into the same online softmax as the local window.
3. **The softmax weights are masked, not just the scores.** ``p = where(vis, exp2(...), 0)``
   so wholly masked rows (such as invalid keyframe planes) contribute 0 and store 0.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...keyframes import KEYFRAME_CONTEXT_SLOTS, keyframe_video_slots, video_keyframe_slots

# Triton requires module-level JIT globals to be constexpr instances (not annotations).
_NEG_INF = tl.constexpr(-3.0e38)
_LSE_FLOOR = tl.constexpr(1e-30)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fold_w_run(
    m_i,
    l_i,
    acc,
    q_blk,
    k_ptr,
    v_ptr,
    plane_base,
    row_ok,
    w_lo,
    w_hi,
    w_start,
    w_end,
    s_w,
    d_off,
    d_mask,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Fold one ``(frame|plane, h)`` row's W run into the running softmax state."""
    for wk0 in range(w_lo, w_hi, block_k):
        wk = wk0 + tl.arange(0, block_k)
        kmask = wk < w_hi
        kv_ptrs = plane_base + wk[:, None] * s_w + d_off[None, :]
        kv_mask = kmask[:, None] & d_mask[None, :] & row_ok
        k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        k_t = tl.trans(k_blk)
        s = tl.dot(q_blk, k_t, input_precision="ieee") if is_fp32 else tl.dot(q_blk, k_t)
        vis = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :] & row_ok
        s = tl.where(vis, s, _NEG_INF)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(vis, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        if is_fp32:
            acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _joint_video_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kf_k_ptr,
    kf_v_ptr,
    slot_ptr,
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
    kf_s_b,
    kf_s_p,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Video queries: local ``Kt x Kh x Kw`` window plus ``num_slots`` keyframe planes."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)

    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for d_t in range(kt):
        tk = t_q - kt // 2 + d_t
        t_ok = (tk >= 0) & (tk < t_size)
        tk_c = tl.maximum(tl.minimum(tk, t_size - 1), 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = t_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + tk_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    for slot in range(num_slots):
        plane = tl.load(slot_ptr + t_q * num_slots + slot)
        plane_ok = plane >= 0
        plane_c = tl.maximum(plane, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = plane_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                kf_k_ptr,
                kf_v_ptr,
                kf_base + plane_c.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = out_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


@triton.jit
def _joint_keyframe_kernel(
    kf_q_ptr,
    kf_k_ptr,
    kf_v_ptr,
    k_ptr,
    v_ptr,
    slot_ptr,
    valid_ptr,
    out_ptr,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kf_s_b,
    kf_s_p,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Keyframe queries: own plane only plus ``num_slots`` video frames."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_pbn = tl.program_id(2)

    p_q = pid_pbn // bn_count
    pid_bn = pid_pbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        kf_q_ptr + kf_base + p_q.to(tl.int64) * kf_s_p + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    own_ok = tl.load(valid_ptr + p_q) != 0
    for d_h in range(kh):
        hk = h_q - kh // 2 + d_h
        row_ok = own_ok & (hk >= 0) & (hk < h_size)
        hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
        m_i, l_i, acc = _fold_w_run(
            m_i,
            l_i,
            acc,
            q_blk,
            kf_k_ptr,
            kf_v_ptr,
            kf_base + p_q.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
            row_ok,
            w_lo,
            w_hi,
            w_start,
            w_end,
            s_w,
            d_off,
            d_mask,
            block_k=block_k,
            is_fp32=is_fp32,
        )

    for slot in range(num_slots):
        frame = tl.load(slot_ptr + p_q * num_slots + slot)
        frame_ok = frame >= 0
        frame_c = tl.maximum(frame, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = frame_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + frame_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr + kf_base + p_q.to(tl.int64) * kf_s_p + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


@triton.jit
def _fold_plane_2d(
    m_i,
    l_i,
    acc,
    q_blk,
    k_ptr,
    v_ptr,
    plane_base,
    q_row_mask,
    h_lo,
    h_hi,
    w_lo,
    w_hi,
    h_start,
    h_end,
    w_start,
    w_end,
    s_h,
    s_w,
    d_off,
    d_mask,
    BKH2: tl.constexpr,
    BKW2: tl.constexpr,
    SINGLE_W2: tl.constexpr,
):
    """Fold a single 2D ``[h_lo, h_hi) x [w_lo, w_hi)`` plane (video frame or keyframe) with ``BK = BKH2*BKW2``."""
    BK2: tl.constexpr = BKH2 * BKW2
    k2_lin = tl.arange(0, BK2)
    kh2_rel = k2_lin // BKW2
    kw2_rel = k2_lin % BKW2

    if SINGLE_W2:
        wk = w_lo + kw2_rel
        w_in = wk < w_hi
        wk_ptr = plane_base + wk.to(tl.int64) * s_w
        for hk0 in range(h_lo, h_hi, BKH2):
            hk = hk0 + kh2_rel
            h_in = hk < h_hi
            k_valid = h_in & w_in
            kv_ptrs = (wk_ptr + hk.to(tl.int64) * s_h)[:, None] + d_off[None, :]
            kv_mask = k_valid[:, None] & d_mask[None, :]
            k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
            s = tl.dot(q_blk, tl.trans(k_blk)) * _LOG2E
            vis = (
                q_row_mask[:, None]
                & k_valid[None, :]
                & (hk[None, :] >= h_start[:, None])
                & (hk[None, :] < h_end[:, None])
                & (wk[None, :] >= w_start[:, None])
                & (wk[None, :] < w_end[:, None])
            )
            s = tl.where(vis, s, _NEG_INF)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.where(vis, tl.math.exp2(s - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
            m_i = m_new
    else:
        for hk0 in range(h_lo, h_hi, BKH2):
            hk = hk0 + kh2_rel
            h_in = hk < h_hi
            hk_ptr = plane_base + hk.to(tl.int64) * s_h
            for wk0 in range(w_lo, w_hi, BKW2):
                wk = wk0 + kw2_rel
                w_in = wk < w_hi
                k_valid = h_in & w_in
                kv_ptrs = (hk_ptr + wk.to(tl.int64) * s_w)[:, None] + d_off[None, :]
                kv_mask = k_valid[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                s = tl.dot(q_blk, tl.trans(k_blk)) * _LOG2E
                vis = (
                    q_row_mask[:, None]
                    & k_valid[None, :]
                    & (hk[None, :] >= h_start[:, None])
                    & (hk[None, :] < h_end[:, None])
                    & (wk[None, :] >= w_start[:, None])
                    & (wk[None, :] < w_end[:, None])
                )
                s = tl.where(vis, s, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.where(vis, tl.math.exp2(s - m_new[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(p, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _joint_video_3d_tiled_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kf_k_ptr,
    kf_v_ptr,
    slot_ptr,
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
    kf_s_b,
    kf_s_p,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
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
    """3D-tiled Hopper WGMMA joint video kernel (clamp-and-mask local window + keyframe slots)."""
    BQ: tl.constexpr = BT * BH * BW
    BK: tl.constexpr = BKT * BKH * BKW

    pid0 = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)
    pid_bn = pid0 % bn_count
    pid_w = pid0 // bn_count

    t0 = pid_t * BT
    h0 = pid_h * BH
    w0 = pid_w * BW

    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

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

    t_q = tl.where(t_off < t_size, t_off, 0)
    h_q = tl.where(h_off < h_size, h_off, 0)
    w_q = tl.where(w_off < w_size, w_off, 0)

    # Exact clamp-and-mask per-query bounds and tile bounding box.
    t_start = tl.maximum(t_q - kt // 2, 0)
    t_end = tl.minimum(t_q - kt // 2 + kt, t_size)
    h_start = tl.maximum(h_q - kh // 2, 0)
    h_end = tl.minimum(h_q - kh // 2 + kh, h_size)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)

    t_last = tl.minimum(t0 + BT - 1, t_size - 1)
    h_last = tl.minimum(h0 + BH - 1, h_size - 1)
    w_last = tl.minimum(w0 + BW - 1, w_size - 1)
    t_lo = tl.maximum(t0 - kt // 2, 0)
    t_hi = tl.minimum(t_last - kt // 2 + kt, t_size)
    h_lo = tl.maximum(h0 - kh // 2, 0)
    h_hi = tl.minimum(h_last - kh // 2 + kh, h_size)
    w_lo = tl.maximum(w0 - kw // 2, 0)
    w_hi = tl.minimum(w_last - kw // 2 + kw, w_size)

    k_lin = tl.arange(0, BK)
    kt_rel = k_lin // (BKH * BKW)
    k_rem = k_lin % (BKH * BKW)
    kh_rel = k_rem // BKW
    kw_rel = k_rem % BKW

    m_i = tl.full((BQ,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BQ,), dtype=tl.float32)
    acc = tl.zeros((BQ, hd_pad), dtype=tl.float32)

    # 1. Local 3D video window
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
                s = tl.dot(q_blk, tl.trans(k_blk)) * _LOG2E
                vis = (
                    k_valid[None, :]
                    & (tk[None, :] >= t_start[:, None])
                    & (tk[None, :] < t_end[:, None])
                    & (hk[None, :] >= h_start[:, None])
                    & (hk[None, :] < h_end[:, None])
                    & (wk[None, :] >= w_start[:, None])
                    & (wk[None, :] < w_end[:, None])
                )
                s = tl.where(vis, s, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.where(vis, tl.math.exp2(s - m_new[:, None]), 0.0)
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
                    s = tl.dot(q_blk, tl.trans(k_blk)) * _LOG2E
                    vis = (
                        k_valid[None, :]
                        & (tk[None, :] >= t_start[:, None])
                        & (tk[None, :] < t_end[:, None])
                        & (hk[None, :] >= h_start[:, None])
                        & (hk[None, :] < h_end[:, None])
                        & (wk[None, :] >= w_start[:, None])
                        & (wk[None, :] < w_end[:, None])
                    )
                    s = tl.where(vis, s, _NEG_INF)
                    m_new = tl.maximum(m_i, tl.max(s, 1))
                    alpha = tl.math.exp2(m_i - m_new)
                    p = tl.where(vis, tl.math.exp2(s - m_new[:, None]), 0.0)
                    l_i = l_i * alpha + tl.sum(p, 1)
                    v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                    acc = acc * alpha[:, None]
                    acc = tl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                    m_i = m_new

    # 2. Cross-stream keyframe plane slots (folded into the exact same softmax state).
    BKH2: tl.constexpr = 4
    BKW2: tl.constexpr = 16
    SINGLE_W2: tl.constexpr = BW + kw - 1 <= BKW2
    for slot in range(num_slots):
        plane_q = tl.load(slot_ptr + t_q * num_slots + slot)
        p_min = tl.min(plane_q, axis=0)
        p_max = tl.max(plane_q, axis=0)
        if p_min == p_max:
            if p_min >= 0:
                m_i, l_i, acc = _fold_plane_2d(
                    m_i,
                    l_i,
                    acc,
                    q_blk,
                    kf_k_ptr,
                    kf_v_ptr,
                    kf_base + p_min.to(tl.int64) * kf_s_p,
                    q_valid,
                    h_lo,
                    h_hi,
                    w_lo,
                    w_hi,
                    h_start,
                    h_end,
                    w_start,
                    w_end,
                    s_h,
                    s_w,
                    d_off,
                    d_mask,
                    BKH2=BKH2,
                    BKW2=BKW2,
                    SINGLE_W2=SINGLE_W2,
                )
        else:
            for qt_i in range(BT):
                ti = t0 + qt_i
                if ti < t_size:
                    plane = tl.load(slot_ptr + ti * num_slots + slot)
                    if plane >= 0:
                        m_i, l_i, acc = _fold_plane_2d(
                            m_i,
                            l_i,
                            acc,
                            q_blk,
                            kf_k_ptr,
                            kf_v_ptr,
                            kf_base + plane.to(tl.int64) * kf_s_p,
                            q_valid & (qt_rel == qt_i),
                            h_lo,
                            h_hi,
                            w_lo,
                            w_hi,
                            h_start,
                            h_end,
                            w_start,
                            w_end,
                            s_h,
                            s_w,
                            d_off,
                            d_mask,
                            BKH2=BKH2,
                            BKW2=BKW2,
                            SINGLE_W2=SINGLE_W2,
                        )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr
        + base
        + t_off.to(tl.int64)[:, None] * s_t
        + h_off.to(tl.int64)[:, None] * s_h
        + w_off.to(tl.int64)[:, None] * s_w
        + d_off[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=q_valid[:, None] & d_mask[None, :])


@triton.jit
def _joint_keyframe_2d_tiled_kernel(
    kf_q_ptr,
    kf_k_ptr,
    kf_v_ptr,
    k_ptr,
    v_ptr,
    slot_ptr,
    valid_ptr,
    out_ptr,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kf_s_b,
    kf_s_p,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BKH2: tl.constexpr,
    BKW2: tl.constexpr,
    SINGLE_W2: tl.constexpr,
):
    """2D-tiled Hopper WGMMA joint keyframe kernel (BQ = BH*BW = 64, BK = BKH2*BKW2 = 64)."""
    BQ: tl.constexpr = BH * BW

    pid0 = tl.program_id(0)
    pid_h = tl.program_id(1)
    p_q = tl.program_id(2)
    pid_bn = pid0 % bn_count
    pid_w = pid0 // bn_count

    h0 = pid_h * BH
    w0 = pid_w * BW

    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    q_lin = tl.arange(0, BQ)
    h_off = h0 + q_lin // BW
    w_off = w0 + q_lin % BW
    q_valid = (h_off < h_size) & (w_off < w_size)

    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        kf_q_ptr
        + kf_base
        + p_q.to(tl.int64) * kf_s_p
        + h_off.to(tl.int64)[:, None] * s_h
        + w_off.to(tl.int64)[:, None] * s_w
        + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    h_q = tl.where(h_off < h_size, h_off, 0)
    w_q = tl.where(w_off < w_size, w_off, 0)
    h_start = tl.maximum(h_q - kh // 2, 0)
    h_end = tl.minimum(h_q - kh // 2 + kh, h_size)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)

    h_last = tl.minimum(h0 + BH - 1, h_size - 1)
    w_last = tl.minimum(w0 + BW - 1, w_size - 1)
    h_lo = tl.maximum(h0 - kh // 2, 0)
    h_hi = tl.minimum(h_last - kh // 2 + kh, h_size)
    w_lo = tl.maximum(w0 - kw // 2, 0)
    w_hi = tl.minimum(w_last - kw // 2 + kw, w_size)

    m_i = tl.full((BQ,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BQ,), dtype=tl.float32)
    acc = tl.zeros((BQ, hd_pad), dtype=tl.float32)

    own_ok = tl.load(valid_ptr + p_q) != 0
    if own_ok:
        m_i, l_i, acc = _fold_plane_2d(
            m_i,
            l_i,
            acc,
            q_blk,
            kf_k_ptr,
            kf_v_ptr,
            kf_base + p_q.to(tl.int64) * kf_s_p,
            q_valid,
            h_lo,
            h_hi,
            w_lo,
            w_hi,
            h_start,
            h_end,
            w_start,
            w_end,
            s_h,
            s_w,
            d_off,
            d_mask,
            BKH2=BKH2,
            BKW2=BKW2,
            SINGLE_W2=SINGLE_W2,
        )

    for slot in range(num_slots):
        frame = tl.load(slot_ptr + p_q * num_slots + slot)
        if frame >= 0:
            m_i, l_i, acc = _fold_plane_2d(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + frame.to(tl.int64) * s_t,
                q_valid,
                h_lo,
                h_hi,
                w_lo,
                w_hi,
                h_start,
                h_end,
                w_start,
                w_end,
                s_h,
                s_w,
                d_off,
                d_mask,
                BKH2=BKH2,
                BKW2=BKW2,
                SINGLE_W2=SINGLE_W2,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr
        + kf_base
        + p_q.to(tl.int64) * kf_s_p
        + h_off.to(tl.int64)[:, None] * s_h
        + w_off.to(tl.int64)[:, None] * s_w
        + d_off[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=q_valid[:, None] & d_mask[None, :])


def _check_shared_layout(video: torch.Tensor, keyframes: torch.Tensor) -> None:
    """Both streams must be contiguous and agree on H/W/NH/HD."""
    if video.shape[2:] != keyframes.shape[2:]:
        raise ValueError(
            f"video and keyframe streams must agree on H/W/NH/HD, got {tuple(video.shape)} vs {tuple(keyframes.shape)}"
        )
    if not video.is_contiguous() or not keyframes.is_contiguous():
        raise ValueError("joint Triton NA needs both streams contiguous")
    if video.stride()[2:5] != keyframes.stride()[2:5]:
        raise ValueError(
            f"H/W/NH strides must match across streams, got {video.stride()[2:5]} vs {keyframes.stride()[2:5]}"
        )


def joint_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint neighborhood attention over a video volume and a keyframe plane stack."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    keyframe_q = keyframe_q.contiguous()
    keyframe_k = keyframe_k.contiguous()
    keyframe_v = keyframe_v.contiguous()
    _check_shared_layout(q, keyframe_q)

    batch, time, height, width, heads, head_dim = q.shape
    planes = keyframe_q.shape[1]
    if keyframe_times.shape != (planes,) or keyframe_valid.shape != (planes,):
        raise ValueError(
            f"keyframe_times/keyframe_valid must be ({planes},), got "
            f"{tuple(keyframe_times.shape)} / {tuple(keyframe_valid.shape)}"
        )
    kernel_t, kernel_h, kernel_w = kernel_size

    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    video_slots = video_slots.to(device=q.device, dtype=torch.int32).contiguous()
    keyframe_slots = keyframe_slots.to(device=q.device, dtype=torch.int32).contiguous()
    valid_i32 = keyframe_valid.to(device=q.device, dtype=torch.int32).contiguous()

    video_out = torch.empty_like(q)
    keyframe_out = torch.empty_like(keyframe_q)

    hd_pad = max(16, triton.next_power_of_2(head_dim))
    strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4))
    kf_strides = (keyframe_q.stride(0), keyframe_q.stride(1))
    bn_count = batch * heads

    use_3d_tiled = (
        q.dtype in (torch.bfloat16, torch.float16)
        and hd_pad <= 128
        and time >= 2
        and height >= 4
        and width >= 4
        and planes <= 65535
    )
    if use_3d_tiled:
        if kernel_t >= 7 and kernel_h >= 7 and kernel_w >= 7:
            bt, bh, bw, bkt, bkh, bkw = 4, 4, 4, 2, 2, 16
        elif kernel_t <= 5 and kernel_w <= 5 and height >= 8:
            bt, bh, bw, bkt, bkh, bkw = 2, 8, 4, 2, 4, 8
        elif width >= 8:
            bt, bh, bw, bkt, bkh, bkw = 2, 4, 8, 2, 2, 16
        else:
            bt, bh, bw, bkt, bkh, bkw = 4, 4, 4, 2, 2, 16

        grid_v = (triton.cdiv(width, bw) * bn_count, triton.cdiv(height, bh), triton.cdiv(time, bt))
        kf_bh, kf_bw = 8, 8
        grid_k = (triton.cdiv(width, kf_bw) * bn_count, triton.cdiv(height, kf_bh), planes)
        if grid_v[1] <= 65535 and grid_v[2] <= 65535 and grid_k[1] <= 65535:
            _joint_video_3d_tiled_kernel[grid_v](
                q,
                k,
                v,
                keyframe_k,
                keyframe_v,
                video_slots,
                video_out,
                time,
                height,
                width,
                heads,
                bn_count,
                *strides,
                *kf_strides,
                kt=kernel_t,
                kh=kernel_h,
                kw=kernel_w,
                num_slots=num_slots,
                hd=head_dim,
                hd_pad=hd_pad,
                BT=bt,
                BH=bh,
                BW=bw,
                BKT=bkt,
                BKH=bkh,
                BKW=bkw,
                SINGLE_W=(bw + kernel_w - 1 <= bkw),
                num_warps=4,
                num_stages=2,
            )
            if planes > 0:
                _joint_keyframe_2d_tiled_kernel[grid_k](
                    keyframe_q,
                    keyframe_k,
                    keyframe_v,
                    k,
                    v,
                    keyframe_slots,
                    valid_i32,
                    keyframe_out,
                    height,
                    width,
                    heads,
                    bn_count,
                    *strides,
                    *kf_strides,
                    kh=kernel_h,
                    kw=kernel_w,
                    num_slots=num_slots,
                    hd=head_dim,
                    hd_pad=hd_pad,
                    BH=kf_bh,
                    BW=kf_bw,
                    BKH2=4,
                    BKW2=16,
                    SINGLE_W2=(kf_bw + kernel_w - 1 <= 16),
                    num_warps=4,
                    num_stages=2,
                )
            return video_out, keyframe_out

    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(width, block_q + kernel_w))))
    is_fp32 = q.dtype == torch.float32
    _joint_video_kernel[(triton.cdiv(width, block_q), height, time * bn_count)](
        q,
        k,
        v,
        keyframe_k,
        keyframe_v,
        video_slots,
        video_out,
        time,
        height,
        width,
        heads,
        bn_count,
        *strides,
        *kf_strides,
        kt=kernel_t,
        kh=kernel_h,
        kw=kernel_w,
        num_slots=num_slots,
        hd=head_dim,
        hd_pad=hd_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=is_fp32,
        num_warps=4,
    )
    if planes > 0:
        _joint_keyframe_kernel[(triton.cdiv(width, block_q), height, planes * bn_count)](
            keyframe_q,
            keyframe_k,
            keyframe_v,
            k,
            v,
            keyframe_slots,
            valid_i32,
            keyframe_out,
            height,
            width,
            heads,
            bn_count,
            *strides,
            *kf_strides,
            kh=kernel_h,
            kw=kernel_w,
            num_slots=num_slots,
            hd=head_dim,
            hd_pad=hd_pad,
            block_q=block_q,
            block_k=block_k,
            is_fp32=is_fp32,
            num_warps=4,
        )
    return video_out, keyframe_out
