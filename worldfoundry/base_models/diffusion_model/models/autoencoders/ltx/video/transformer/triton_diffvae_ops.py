"""Lossless Hopper SM90 Triton kernels for LTX-2.5 Diffusion VideoVAE (DiffVAE).

All CUDA ops are registered via ``@torch.library.custom_op`` + ``.register_fake``
for ``torch.compile`` and CUDA Graph compatibility, use lazy Triton imports so
CPU-only environments stay clean, use strict ``tl.int64`` pointer arithmetic for
large 5D video volumes, and fall back cleanly to exact PyTorch implementations
when CUDA/Triton/dtype/autograd constraints are not met.

Public surface:
- :func:`diffvae_rmsnorm_modulate` — fused ``rms_norm(x, w, eps) * (1 + scale) + shift``.
- :func:`diffvae_rmsnorm_modulate_into` — in-place write into a preallocated output buffer.
- :func:`diffvae_rmsnorm_3d_abs_rope` — fused per-head RMSNorm + scale + 3D absolute RoPE
  over ``[B, T, H, W, NH, HD]`` in a single HBM pass (head-grouped trig sharing).
- :func:`diffvae_w_slice_add` — 128-bit coalesced in-place W-slice residual add.
- :func:`diffvae_cut_slab` — single-pass W-slab extraction with neighbor halo + edge replicate.
- :func:`diffvae_pixel_shuffle_add_w_slice` — single-pass 3D pixel-shuffle + W-slice residual add.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from .rope_math import h_positions, rot_abs_axis_impl, t_positions

_TRITON_MOD: Any | None = None
_TRITON_KERNELS: dict[str, Any] | None = None


def _diffvae_ops_enabled() -> bool:
    if os.environ.get("WORLDFOUNDRY_KERNEL_BACKEND", "").strip().lower() == "torch":
        return False
    return os.environ.get("WORLDFOUNDRY_LTX_DIFFVAE_TRITON", "1").strip().lower() not in ("0", "false", "no", "off")


def _get_triton() -> tuple[Any, Any] | None:
    global _TRITON_MOD
    if _TRITON_MOD is None:
        try:
            import triton
            import triton.language as tl

            _TRITON_MOD = (triton, tl)
        except Exception:
            _TRITON_MOD = False
    return None if _TRITON_MOD is False else _TRITON_MOD


def triton_diffvae_ops_available() -> bool:
    return _diffvae_ops_enabled() and torch.cuda.is_available() and _get_triton() is not None


def _get_kernels() -> dict[str, Any]:
    global _TRITON_KERNELS
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS

    mod = _get_triton()
    if mod is None:
        raise RuntimeError("Triton is not available")
    triton, tl = mod

    @triton.jit
    def _rmsnorm_modulate_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        n_rows,
        eps,
        DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < DIM
        mask = row_mask[:, None] & d_mask[None, :]

        ptrs = rows[:, None] * DIM + offs_d[None, :]
        x_raw = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        x_f32 = x_raw.to(tl.float32)

        inv_dim: tl.constexpr = 1.0 / DIM
        var = tl.sum(x_f32 * x_f32, axis=1) * inv_dim
        rstd = tl.rsqrt(var + eps)[:, None]
        y_f32 = x_f32 * rstd

        if HAS_WEIGHT:
            w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)[None, :]
            y_f32 = y_f32 * w

        if OUT_BF16:
            y_f32 = y_f32.to(tl.bfloat16).to(tl.float32)
        elif OUT_FP16:
            y_f32 = y_f32.to(tl.float16).to(tl.float32)

        scale = tl.load(scale_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        shift = tl.load(shift_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        out_f32 = y_f32 * (1.0 + scale) + shift

        if OUT_BF16:
            out_val = out_f32.to(tl.bfloat16)
        elif OUT_FP16:
            out_val = out_f32.to(tl.float16)
        else:
            out_val = out_f32

        tl.store(out_ptr + ptrs, out_val, mask=mask)

    @triton.jit
    def _rmsnorm_3d_abs_rope_kernel(
        x_ptr,
        w_ptr,
        t_pos_ptr,
        w_pos_ptr,
        inv_t_ptr,
        inv_h_ptr,
        inv_w_ptr,
        out_ptr,
        t_size,
        h_size,
        w_size,
        num_heads,
        eps,
        attn_scale,
        D_T: tl.constexpr,
        D_H: tl.constexpr,
        D_W: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_NH: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        DO_RMSNORM: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_T_POS: tl.constexpr,
        HAS_W_POS: tl.constexpr,
        ROPE_COMPUTE_BF16: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        """Head-grouped single-pass per-head RMSNorm (+ scale) + 3D abs-RoPE.

        Processes ``BLOCK_W`` spatial-W positions and loops over head tiles of
        size ``BLOCK_NH`` at each ``(b, t, h)``, computing ``cos/sin`` ONCE per
        ``w`` and sharing it across all ``num_heads`` (zero integer division in
        the hot loop).
        """
        HALF_T: tl.constexpr = D_T // 2
        HALF_H: tl.constexpr = D_H // 2
        HALF_TH: tl.constexpr = HALF_T + HALF_H
        HALF_HD: tl.constexpr = HEAD_DIM // 2
        BLOCK_HALF: tl.constexpr = BLOCK_HD // 2

        pid_w = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        pid_bt = tl.program_id(2).to(tl.int64)

        t_idx = pid_bt % t_size
        h_idx = pid_h

        w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_offs < w_size

        d_offs = tl.arange(0, BLOCK_HD)
        d_mask = d_offs < HEAD_DIM
        j = tl.arange(0, BLOCK_HALF)
        is_t = j < HALF_T
        is_h = (j >= HALF_T) & (j < HALF_TH)
        is_w = (j >= HALF_TH) & (j < HALF_HD)

        inv_t_val = tl.load(inv_t_ptr + j, mask=is_t, other=0.0).to(tl.float32)
        inv_h_val = tl.load(inv_h_ptr + (j - HALF_T), mask=is_h, other=0.0).to(tl.float32)
        inv_w_val = tl.load(inv_w_ptr + (j - HALF_TH), mask=is_w, other=0.0).to(tl.float32)
        inv_val = tl.where(is_t, inv_t_val, tl.where(is_h, inv_h_val, inv_w_val))

        if HAS_T_POS:  # noqa: SIM108
            t_val = tl.load(t_pos_ptr + t_idx).to(tl.float32)
        else:
            t_val = t_idx.to(tl.float32)
        h_val = h_idx.to(tl.float32)
        if HAS_W_POS:
            w_val = tl.load(w_pos_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)
        else:
            w_val = w_offs.to(tl.float32)

        # Compute cos/sin ONCE for [BLOCK_W, BLOCK_HALF] and reuse across all heads!
        pos_val = tl.where(is_t[None, :], t_val, tl.where(is_h[None, :], h_val, w_val[:, None]))
        ang = pos_val * inv_val[None, :]
        c_w = tl.cos(ang)
        s_w = tl.sin(ang)
        if ROPE_COMPUTE_BF16:
            c_w = c_w.to(tl.bfloat16).to(tl.float32)
            s_w = s_w.to(tl.bfloat16).to(tl.float32)

        if DO_RMSNORM and HAS_WEIGHT:
            w_norm = tl.load(w_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

        nh_i64 = num_heads.to(tl.int64)
        plane_row_base = (pid_bt * h_size.to(tl.int64) + h_idx) * (w_size.to(tl.int64) * nh_i64)

        # Flatten [BLOCK_W, BLOCK_NH] into ROWS = BLOCK_W * BLOCK_NH for 128-bit coalesced loads.
        ROWS: tl.constexpr = BLOCK_W * BLOCK_NH
        r_idx = tl.arange(0, ROWS)
        r_w = r_idx // BLOCK_NH
        r_h = r_idx % BLOCK_NH
        w_for_row = pid_w * BLOCK_W + r_w
        w_valid_row = w_for_row < w_size

        # Broadcast [BLOCK_W, BLOCK_HALF] -> [ROWS, BLOCK_HALF] via [BLOCK_W, 1, BLOCK_HALF]
        c_row = tl.reshape(tl.broadcast_to(c_w[:, None, :], (BLOCK_W, BLOCK_NH, BLOCK_HALF)), (ROWS, BLOCK_HALF))
        s_row = tl.reshape(tl.broadcast_to(s_w[:, None, :], (BLOCK_W, BLOCK_NH, BLOCK_HALF)), (ROWS, BLOCK_HALF))

        inv_hd: tl.constexpr = 1.0 / HEAD_DIM
        for nh0 in range(0, num_heads, BLOCK_NH):
            nh_for_row = nh0 + r_h
            row_mask = w_valid_row & (nh_for_row < num_heads)
            load_mask = row_mask[:, None] & d_mask[None, :]

            row_id = plane_row_base + w_for_row.to(tl.int64) * nh_i64 + nh_for_row.to(tl.int64)
            ptrs = row_id[:, None] * HEAD_DIM + d_offs[None, :]
            x_raw = tl.load(x_ptr + ptrs, mask=load_mask, other=0.0)
            x_f32 = x_raw.to(tl.float32)

            if DO_RMSNORM:
                rstd = tl.rsqrt(tl.sum(x_f32 * x_f32, axis=1) * inv_hd + eps)[:, None]
                x_f32 = x_f32 * rstd
                if HAS_WEIGHT:
                    x_f32 = x_f32 * w_norm
                if OUT_BF16:
                    x_f32 = x_f32.to(tl.bfloat16).to(tl.float32)
                elif OUT_FP16:
                    x_f32 = x_f32.to(tl.float16).to(tl.float32)
                if HAS_SCALE:
                    x_f32 = x_f32 * attn_scale
                    if OUT_BF16:
                        x_f32 = x_f32.to(tl.bfloat16).to(tl.float32)
                    elif OUT_FP16:
                        x_f32 = x_f32.to(tl.float16).to(tl.float32)

            if ROPE_COMPUTE_BF16:
                x_f32 = x_f32.to(tl.bfloat16).to(tl.float32)

            x_pairs = tl.reshape(x_f32, (ROWS, BLOCK_HALF, 2))
            xe, xo = tl.split(x_pairs)
            re = xe * c_row - xo * s_row
            ro = xe * s_row + xo * c_row

            if OUT_BF16:
                re_out = re.to(tl.bfloat16)
                ro_out = ro.to(tl.bfloat16)
            elif OUT_FP16:
                re_out = re.to(tl.float16)
                ro_out = ro.to(tl.float16)
            else:
                re_out = re
                ro_out = ro

            out_vals = tl.reshape(tl.join(re_out, ro_out), (ROWS, BLOCK_HD))
            tl.store(out_ptr + ptrs, out_vals, mask=load_mask)

    @triton.jit
    def _w_slice_add_kernel(
        x_ptr,
        out_ptr,
        bth_count,
        w_x,
        w_out,
        core_start,
        halo,
        core_len,
        DIM: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        """128-bit coalesced ``x[b,t,h,core_start:core_start+core_len,:] += out[b,t,h,halo:halo+core_len,:]``."""
        pid_w = tl.program_id(0).to(tl.int64)
        pid_bth = tl.program_id(1).to(tl.int64)

        w_rel = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_rel < core_len
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < DIM
        mask = w_mask[:, None] & d_mask[None, :]

        x_base = (pid_bth * w_x.to(tl.int64) + core_start.to(tl.int64) + w_rel.to(tl.int64)) * DIM
        o_base = (pid_bth * w_out.to(tl.int64) + halo.to(tl.int64) + w_rel.to(tl.int64)) * DIM
        x_ptrs = x_base[:, None] + d_offs[None, :]
        o_ptrs = o_base[:, None] + d_offs[None, :]

        xv = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0).to(tl.float32)
        ov = tl.load(out_ptr + o_ptrs, mask=mask, other=0.0).to(tl.float32)
        res = xv + ov
        if OUT_BF16:
            res_out = res.to(tl.bfloat16)
        elif OUT_FP16:
            res_out = res.to(tl.float16)
        else:
            res_out = res
        tl.store(x_ptr + x_ptrs, res_out, mask=mask)

    @triton.jit
    def _cut_slab_kernel(
        x_ptr,
        left_halo_ptr,
        buf_ptr,
        w_x,
        extent,
        core_start,
        core_len,
        halo,
        lh_len,
        right_filled,
        HAS_LEFT_HALO: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass W-slab extraction with neighbor halo + boundary edge replication."""
        pid_w = tl.program_id(0).to(tl.int64)
        pid_bth = tl.program_id(1).to(tl.int64)

        w_dst = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_dst < extent
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < DIM
        mask = w_mask[:, None] & d_mask[None, :]

        # Determine source for each w_dst in [0, extent):
        # - If HAS_LEFT_HALO and w_dst in [halo - lh_len, halo): read left_halo[w_dst - (halo - lh_len)]
        # - Else if w_dst < halo:
        #     - If not HAS_LEFT_HALO and core_len > 0: replicate x[core_start]
        #     - Else: zero
        # - Else if w_dst < halo + core_len + right_filled: read x[core_start + (w_dst - halo)]
        # - Else if core_len > 0: replicate x[core_start + core_len - 1]
        # - Else: zero
        if HAS_LEFT_HALO:
            lh_start = halo - lh_len
            from_lh = (w_dst >= lh_start) & (w_dst < halo)
            lh_idx = tl.maximum(tl.minimum(w_dst - lh_start, lh_len - 1), 0)
            lh_ptrs = (pid_bth * lh_len.to(tl.int64) + lh_idx.to(tl.int64))[:, None] * DIM + d_offs[None, :]
            lh_val = tl.load(left_halo_ptr + lh_ptrs, mask=mask & from_lh[:, None], other=0.0)

            in_x = (w_dst >= halo) & (w_dst < halo + core_len + right_filled)
            in_rep_r = (w_dst >= halo + core_len + right_filled) & (core_len > 0)
            x_idx = tl.where(in_x, core_start + (w_dst - halo), core_start + core_len - 1)
            x_idx = tl.maximum(tl.minimum(x_idx, w_x - 1), 0)
            from_x = in_x | in_rep_r
            x_ptrs = (pid_bth * w_x.to(tl.int64) + x_idx.to(tl.int64))[:, None] * DIM + d_offs[None, :]
            x_val = tl.load(x_ptr + x_ptrs, mask=mask & from_x[:, None], other=0.0)
            val = tl.where(from_lh[:, None], lh_val, tl.where(from_x[:, None], x_val, 0.0))
        else:
            # No left halo: true left boundary replicates core_start, right tail replicates core_start + core_len - 1
            in_left_rep = (w_dst < halo) & (core_len > 0)
            in_x = (w_dst >= halo) & (w_dst < halo + core_len + right_filled)
            in_right_rep = (w_dst >= halo + core_len + right_filled) & (core_len > 0)
            x_idx = tl.where(
                in_left_rep,
                core_start,
                tl.where(in_x, core_start + (w_dst - halo), core_start + core_len - 1),
            )
            x_idx = tl.maximum(tl.minimum(x_idx, w_x - 1), 0)
            from_x = in_left_rep | in_x | in_right_rep
            x_ptrs = (pid_bth * w_x.to(tl.int64) + x_idx.to(tl.int64))[:, None] * DIM + d_offs[None, :]
            val = tl.load(x_ptr + x_ptrs, mask=mask & from_x[:, None], other=0.0)

        dst_ptrs = (pid_bth * extent.to(tl.int64) + w_dst.to(tl.int64))[:, None] * DIM + d_offs[None, :]
        tl.store(buf_ptr + dst_ptrs, val, mask=mask)

    @triton.jit
    def _pixel_shuffle_add_w_slice_kernel(
        x_ptr,
        y_ptr,
        t_out,
        h_out,
        w_x,
        lo,
        w_len,
        t_in,
        h_in,
        w_in,
        P1: tl.constexpr,
        P2: tl.constexpr,
        P3: tl.constexpr,
        DROP_FIRST_T: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        """Fused 3D pixel-shuffle from ``y[B, T_in, H_in, W_in, P1*P2*P3, DIM]`` + add into ``x[..., lo:lo+w_len, :]``."""
        S_TOT: tl.constexpr = P1 * P2 * P3
        pid_w = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        pid_bt = tl.program_id(2).to(tl.int64)

        b_idx = pid_bt // t_out
        t_o = pid_bt % t_out
        h_o = pid_h

        t_raw = t_o + (1 if DROP_FIRST_T else 0)
        ti = t_raw // P1
        i1 = t_raw % P1
        hi = h_o // P2
        i2 = h_o % P2

        w_rel = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_rel < w_len
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < DIM
        mask = w_mask[:, None] & d_mask[None, :]

        wi = w_rel // P3
        i3 = w_rel % P3
        s_idx = (i1 * P2 + i2) * P3 + i3

        # y layout: [B, T_in, H_in, W_in, S_TOT, DIM]
        y_row = (((b_idx * t_in.to(tl.int64) + ti) * h_in.to(tl.int64) + hi) * w_in.to(tl.int64) + wi) * S_TOT + s_idx
        y_ptrs = y_row[:, None] * DIM + d_offs[None, :]

        # x layout: [B, T_out, H_out, W_x, DIM]
        x_row = (pid_bt * h_out.to(tl.int64) + h_o) * w_x.to(tl.int64) + lo.to(tl.int64) + w_rel
        x_ptrs = x_row[:, None] * DIM + d_offs[None, :]

        xv = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0).to(tl.float32)
        yv = tl.load(y_ptr + y_ptrs, mask=mask, other=0.0).to(tl.float32)
        res = xv + yv
        if OUT_BF16:
            res_out = res.to(tl.bfloat16)
        elif OUT_FP16:
            res_out = res.to(tl.float16)
        else:
            res_out = res
        tl.store(x_ptr + x_ptrs, res_out, mask=mask)

    _TRITON_KERNELS = {
        "rmsnorm_modulate": _rmsnorm_modulate_kernel,
        "rmsnorm_3d_abs_rope": _rmsnorm_3d_abs_rope_kernel,
        "w_slice_add": _w_slice_add_kernel,
        "cut_slab": _cut_slab_kernel,
        "pixel_shuffle_add_w_slice": _pixel_shuffle_add_w_slice_kernel,
    }
    return _TRITON_KERNELS


def _rmsnorm_modulate_torch(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dim = x.shape[-1]
    y = F.rms_norm(x, (dim,), norm_weight, eps)
    return y * (1.0 + scale) + shift


def _rmsnorm_3d_abs_rope_torch(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    t_pos: torch.Tensor | None,
    w_pos: torch.Tensor | None,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    eps: float,
    attn_scale: float,
    do_rmsnorm: bool,
    rope_compute_bf16: bool,
) -> torch.Tensor:
    del d_w
    y = x
    if do_rmsnorm:
        y = F.rms_norm(y, (y.shape[-1],), norm_weight, eps)
        if attn_scale != 1.0:
            y = y * attn_scale
    compute_dtype = torch.bfloat16 if rope_compute_bf16 else torch.float32
    t = y.shape[1]
    h = y.shape[2]
    w = y.shape[3]
    positions_t = t_positions(t, y.device) if t_pos is None else t_pos
    positions_w = torch.arange(w, dtype=torch.float32, device=y.device) if w_pos is None else w_pos
    xt = rot_abs_axis_impl(y[..., :d_t], positions_t, inv_t, axis=1, compute_dtype=compute_dtype)
    xh = rot_abs_axis_impl(
        y[..., d_t : d_t + d_h], h_positions(h, y.device), inv_h, axis=2, compute_dtype=compute_dtype
    )
    xw = rot_abs_axis_impl(y[..., d_t + d_h :], positions_w, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _launch_rmsnorm_modulate(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    mod = _get_triton()
    assert mod is not None
    triton, _ = mod
    kernels = _get_kernels()

    dim = x.shape[-1]
    x_c = x.contiguous()
    n_rows = x_c.numel() // dim
    if n_rows == 0:
        return
    s_c = scale.reshape(-1).contiguous()
    sh_c = shift.reshape(-1).contiguous()
    has_w = norm_weight is not None
    w_c = norm_weight.contiguous() if norm_weight is not None else s_c

    block_d = triton.next_power_of_2(dim)
    if block_d <= 256:
        block_m = 16
        num_warps = 4
    elif block_d <= 512:
        block_m = 8
        num_warps = 4
    elif block_d <= 1024:
        block_m = 4
        num_warps = 4
    else:
        block_m = 2
        num_warps = 8

    grid = (triton.cdiv(n_rows, block_m),)
    with torch.cuda.device(x_c.device):
        kernels["rmsnorm_modulate"][grid](
            x_c,
            w_c,
            s_c,
            sh_c,
            out,
            n_rows,
            float(eps),
            DIM=dim,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            HAS_WEIGHT=has_w,
            OUT_BF16=x_c.dtype == torch.bfloat16,
            OUT_FP16=x_c.dtype == torch.float16,
            num_warps=num_warps,
        )


def _can_use_rmsnorm_modulate(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    if not triton_diffvae_ops_available():
        return False
    if not x.is_cuda or x.requires_grad or torch.is_grad_enabled():
        return False
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    dim = x.shape[-1]
    if dim <= 0 or dim > 4096:
        return False
    if scale.device != x.device or shift.device != x.device:
        return False
    if scale.numel() != dim or shift.numel() != dim:
        return False
    return norm_weight is None or (norm_weight.device == x.device and norm_weight.shape == (dim,))


@torch.library.custom_op("worldfoundry::ltx_diffvae_rmsnorm_modulate", mutates_args=(), device_types="cuda")
def _rmsnorm_modulate_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _launch_rmsnorm_modulate(x, norm_weight, scale, shift, out, eps)
    return out


@_rmsnorm_modulate_op.register_fake
def _rmsnorm_modulate_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del norm_weight, scale, shift, eps
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("worldfoundry::ltx_diffvae_rmsnorm_modulate_into", mutates_args=("out",), device_types="cuda")
def _rmsnorm_modulate_into_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    _launch_rmsnorm_modulate(x, norm_weight, scale, shift, out, eps)


@_rmsnorm_modulate_into_op.register_fake
def _rmsnorm_modulate_into_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    del x, norm_weight, scale, shift, out, eps


def diffvae_rmsnorm_modulate(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused ``rms_norm(x, norm_weight, eps) * (1 + scale) + shift``."""
    if _can_use_rmsnorm_modulate(x, norm_weight, scale, shift):
        return _rmsnorm_modulate_op(x, norm_weight, scale, shift, float(eps))
    return _rmsnorm_modulate_torch(x, norm_weight, scale, shift, float(eps))


def diffvae_rmsnorm_modulate_into(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> None:
    """Write ``rms_norm(x, norm_weight, eps) * (1 + scale) + shift`` into ``out``."""
    if (
        _can_use_rmsnorm_modulate(x, norm_weight, scale, shift)
        and out.is_cuda
        and out.device == x.device
        and out.dtype == x.dtype
        and out.is_contiguous()
        and out.shape == x.shape
    ):
        _rmsnorm_modulate_into_op(x, norm_weight, scale, shift, out, float(eps))
        return
    out.copy_(_rmsnorm_modulate_torch(x, norm_weight, scale, shift, float(eps)))


def _can_use_rmsnorm_3d_abs_rope(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    t_pos: torch.Tensor | None,
    w_pos: torch.Tensor | None,
) -> bool:
    if not triton_diffvae_ops_available():
        return False
    if not x.is_cuda or x.requires_grad or torch.is_grad_enabled():
        return False
    if x.ndim != 6 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    head_dim = x.shape[-1]
    if head_dim <= 0 or head_dim > 256 or head_dim % 2 != 0:
        return False
    if d_t <= 0 or d_h <= 0 or d_w <= 0 or d_t % 2 != 0 or d_h % 2 != 0 or d_w % 2 != 0:
        return False
    if d_t + d_h + d_w != head_dim:
        return False
    if inv_t.device != x.device or inv_h.device != x.device or inv_w.device != x.device:
        return False
    if inv_t.numel() != d_t // 2 or inv_h.numel() != d_h // 2 or inv_w.numel() != d_w // 2:
        return False
    if norm_weight is not None and (norm_weight.device != x.device or norm_weight.shape != (head_dim,)):
        return False
    if t_pos is not None and (t_pos.device != x.device or t_pos.shape != (x.shape[1],)):
        return False
    return w_pos is None or (w_pos.device == x.device and w_pos.shape == (x.shape[3],))


@torch.library.custom_op("worldfoundry::ltx_diffvae_rmsnorm_3d_abs_rope", mutates_args=(), device_types="cuda")
def _rmsnorm_3d_abs_rope_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    t_pos: torch.Tensor | None,
    w_pos: torch.Tensor | None,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    eps: float,
    attn_scale: float,
    do_rmsnorm: bool,
    rope_compute_bf16: bool,
) -> torch.Tensor:
    mod = _get_triton()
    assert mod is not None
    triton, _ = mod
    kernels = _get_kernels()

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    batch, t_size, h_size, w_size, num_heads, head_dim = x_c.shape
    if x_c.numel() == 0:
        return out

    inv_t_c = inv_t.contiguous()
    inv_h_c = inv_h.contiguous()
    inv_w_c = inv_w.contiguous()
    has_weight = norm_weight is not None
    w_c = norm_weight.contiguous() if norm_weight is not None else inv_t_c
    has_t_pos = t_pos is not None
    t_pos_c = t_pos.contiguous() if t_pos is not None else inv_t_c
    has_w_pos = w_pos is not None
    w_pos_c = w_pos.contiguous() if w_pos is not None else inv_t_c

    block_hd = triton.next_power_of_2(head_dim)
    block_nh = min(triton.next_power_of_2(num_heads), 8 if block_hd <= 64 else 4)
    target_rows = 32 if block_hd <= 64 else (16 if block_hd <= 128 else 8)
    block_w = max(1, target_rows // block_nh)
    grid = (triton.cdiv(w_size, block_w), h_size, batch * t_size)
    with torch.cuda.device(x_c.device):
        kernels["rmsnorm_3d_abs_rope"][grid](
            x_c,
            w_c,
            t_pos_c,
            w_pos_c,
            inv_t_c,
            inv_h_c,
            inv_w_c,
            out,
            t_size,
            h_size,
            w_size,
            num_heads,
            float(eps),
            float(attn_scale),
            D_T=d_t,
            D_H=d_h,
            D_W=d_w,
            HEAD_DIM=head_dim,
            BLOCK_W=block_w,
            BLOCK_NH=block_nh,
            BLOCK_HD=block_hd,
            DO_RMSNORM=bool(do_rmsnorm),
            HAS_WEIGHT=has_weight,
            HAS_SCALE=float(attn_scale) != 1.0,
            HAS_T_POS=has_t_pos,
            HAS_W_POS=has_w_pos,
            ROPE_COMPUTE_BF16=bool(rope_compute_bf16),
            OUT_BF16=x_c.dtype == torch.bfloat16,
            OUT_FP16=x_c.dtype == torch.float16,
            num_warps=4,
        )
    return out


@_rmsnorm_3d_abs_rope_op.register_fake
def _rmsnorm_3d_abs_rope_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    t_pos: torch.Tensor | None,
    w_pos: torch.Tensor | None,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    eps: float,
    attn_scale: float,
    do_rmsnorm: bool,
    rope_compute_bf16: bool,
) -> torch.Tensor:
    del norm_weight, t_pos, w_pos, inv_t, inv_h, inv_w, d_t, d_h, d_w, eps, attn_scale, do_rmsnorm, rope_compute_bf16
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def diffvae_rmsnorm_3d_abs_rope(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor | None = None,
    t_pos: torch.Tensor | None = None,
    eps: float = 1e-6,
    attn_scale: float = 1.0,
    do_rmsnorm: bool = True,
    rope_compute_bf16: bool = False,
) -> torch.Tensor:
    """Single-pass per-head RMSNorm (+ optional attn_scale) + 3D absolute RoPE."""
    d_t, d_h, d_w = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    if _can_use_rmsnorm_3d_abs_rope(x, norm_weight, inv_t, inv_h, inv_w, d_t, d_h, d_w, t_pos, w_pos):
        return _rmsnorm_3d_abs_rope_op(
            x,
            norm_weight,
            t_pos,
            w_pos,
            inv_t,
            inv_h,
            inv_w,
            d_t,
            d_h,
            d_w,
            float(eps),
            float(attn_scale),
            bool(do_rmsnorm),
            bool(rope_compute_bf16),
        )
    return _rmsnorm_3d_abs_rope_torch(
        x,
        norm_weight,
        t_pos,
        w_pos,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        float(eps),
        float(attn_scale),
        bool(do_rmsnorm),
        bool(rope_compute_bf16),
    )


@torch.library.custom_op("worldfoundry::ltx_diffvae_w_slice_add", mutates_args=("x",), device_types="cuda")
def _w_slice_add_op(
    x: torch.Tensor,
    out: torch.Tensor,
    core_start: int,
    halo: int,
    core_len: int,
) -> None:
    mod = _get_triton()
    assert mod is not None
    triton, _ = mod
    kernels = _get_kernels()
    if core_len <= 0 or x.numel() == 0:
        return
    b, t, h, w_x, dim = x.shape
    w_out = out.shape[3]
    out_c = out.contiguous()
    bth = b * t * h
    block_d = triton.next_power_of_2(dim)
    block_w = 8 if block_d <= 256 else (4 if block_d <= 512 else 2)
    grid = (triton.cdiv(core_len, block_w), bth)
    with torch.cuda.device(x.device):
        kernels["w_slice_add"][grid](
            x,
            out_c,
            bth,
            w_x,
            w_out,
            core_start,
            halo,
            core_len,
            DIM=dim,
            BLOCK_W=block_w,
            BLOCK_D=block_d,
            OUT_BF16=x.dtype == torch.bfloat16,
            OUT_FP16=x.dtype == torch.float16,
            num_warps=4,
        )


@_w_slice_add_op.register_fake
def _w_slice_add_fake(
    x: torch.Tensor,
    out: torch.Tensor,
    core_start: int,
    halo: int,
    core_len: int,
) -> None:
    del x, out, core_start, halo, core_len


def diffvae_w_slice_add(
    x: torch.Tensor,
    out: torch.Tensor,
    core_start: int,
    halo: int,
    core_len: int,
) -> None:
    """In-place ``x[:, :, :, core_start:core_start+core_len, :] += out[:, :, :, halo:halo+core_len, :]``."""
    if (
        triton_diffvae_ops_available()
        and x.is_cuda
        and out.is_cuda
        and x.device == out.device
        and x.dtype == out.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim == 5
        and out.ndim == 5
        and x.is_contiguous()
        and not x.requires_grad
        and not out.requires_grad
        and not torch.is_grad_enabled()
        and x.shape[-1] <= 2048
        and x.shape[0] * x.shape[1] * x.shape[2] <= 65535
    ):
        _w_slice_add_op(x, out, core_start, halo, core_len)
        return
    x[:, :, :, core_start : core_start + core_len, :].add_(out[:, :, :, halo : halo + core_len, :])


@torch.library.custom_op("worldfoundry::ltx_diffvae_cut_slab", mutates_args=(), device_types="cuda")
def _cut_slab_op(
    x: torch.Tensor,
    left_halo: torch.Tensor | None,
    core_start: int,
    core_len: int,
    halo: int,
    extent: int,
    right_filled: int,
) -> torch.Tensor:
    mod = _get_triton()
    assert mod is not None
    triton, _ = mod
    kernels = _get_kernels()
    b, t, h, w_x, dim = x.shape
    buf = torch.empty((b, t, h, extent, dim), device=x.device, dtype=x.dtype)
    if buf.numel() == 0:
        return buf
    bth = b * t * h
    has_lh = left_halo is not None
    lh_c = left_halo.contiguous() if left_halo is not None else x
    lh_len = left_halo.shape[3] if left_halo is not None else 0
    block_d = triton.next_power_of_2(dim)
    block_w = 8 if block_d <= 256 else (4 if block_d <= 512 else 2)
    grid = (triton.cdiv(extent, block_w), bth)
    with torch.cuda.device(x.device):
        kernels["cut_slab"][grid](
            x,
            lh_c,
            buf,
            w_x,
            extent,
            core_start,
            core_len,
            halo,
            lh_len,
            right_filled,
            HAS_LEFT_HALO=has_lh,
            DIM=dim,
            BLOCK_W=block_w,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return buf


@_cut_slab_op.register_fake
def _cut_slab_fake(
    x: torch.Tensor,
    left_halo: torch.Tensor | None,
    core_start: int,
    core_len: int,
    halo: int,
    extent: int,
    right_filled: int,
) -> torch.Tensor:
    del left_halo, core_start, core_len, halo, right_filled
    return torch.empty((*x.shape[:3], extent, x.shape[4]), device=x.device, dtype=x.dtype)


def diffvae_cut_slab(
    x: torch.Tensor,
    core_start: int,
    core_end: int,
    halo: int,
    extent: int,
    left_halo: torch.Tensor | None,
    *,
    has_right_neighbor: bool,
) -> torch.Tensor | None:
    """Single-pass Triton W-slab extraction; returns ``None`` if ineligible so caller uses fallback."""
    if not (
        triton_diffvae_ops_available()
        and x.is_cuda
        and x.is_contiguous()
        and x.ndim == 5
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and not x.requires_grad
        and not torch.is_grad_enabled()
        and x.shape[-1] <= 2048
        and x.shape[0] * x.shape[1] * x.shape[2] <= 65535
    ):
        return None
    if left_halo is not None and (
        left_halo.device != x.device or left_halo.dtype != x.dtype or left_halo.requires_grad
    ):
        return None
    w = x.shape[3]
    core_len = core_end - core_start
    right_filled = min(w, core_end + halo) - core_end if has_right_neighbor else 0
    return _cut_slab_op(x, left_halo, core_start, core_len, halo, extent, right_filled)


@torch.library.custom_op(
    "worldfoundry::ltx_diffvae_pixel_shuffle_add_w_slice", mutates_args=("x",), device_types="cuda"
)
def _pixel_shuffle_add_w_slice_op(
    x: torch.Tensor,
    y: torch.Tensor,
    lo: int,
    w_len: int,
    t_in: int,
    h_in: int,
    w_in: int,
    p1: int,
    p2: int,
    p3: int,
    drop_leading_frame: bool,
) -> None:
    mod = _get_triton()
    assert mod is not None
    triton, _ = mod
    kernels = _get_kernels()
    if w_len <= 0 or x.numel() == 0:
        return
    b, t_out, h_out, w_x, dim = x.shape
    y_c = y.contiguous()
    block_d = triton.next_power_of_2(dim)
    block_w = 8 if block_d <= 256 else (4 if block_d <= 512 else 2)
    grid = (triton.cdiv(w_len, block_w), h_out, b * t_out)
    with torch.cuda.device(x.device):
        kernels["pixel_shuffle_add_w_slice"][grid](
            x,
            y_c,
            t_out,
            h_out,
            w_x,
            lo,
            w_len,
            t_in,
            h_in,
            w_in,
            P1=p1,
            P2=p2,
            P3=p3,
            DROP_FIRST_T=(p1 == 2 and drop_leading_frame),
            DIM=dim,
            BLOCK_W=block_w,
            BLOCK_D=block_d,
            OUT_BF16=x.dtype == torch.bfloat16,
            OUT_FP16=x.dtype == torch.float16,
            num_warps=4,
        )


@_pixel_shuffle_add_w_slice_op.register_fake
def _pixel_shuffle_add_w_slice_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    lo: int,
    w_len: int,
    t_in: int,
    h_in: int,
    w_in: int,
    p1: int,
    p2: int,
    p3: int,
    drop_leading_frame: bool,
) -> None:
    del x, y, lo, w_len, t_in, h_in, w_in, p1, p2, p3, drop_leading_frame


def diffvae_pixel_shuffle_add_w_slice(
    x: torch.Tensor,
    y: torch.Tensor,
    lo: int,
    w_len: int,
    t_in: int,
    h_in: int,
    w_in: int,
    stride: tuple[int, int, int],
    *,
    drop_leading_frame: bool,
) -> bool:
    """Fused 3D pixel-shuffle of ``y[B, T_in, H_in, W_in, P1*P2*P3*C]`` + add into ``x[..., lo:lo+w_len, :]``.

    Returns ``True`` if executed via Triton, else ``False``.
    """
    p1, p2, p3 = stride
    if not (
        triton_diffvae_ops_available()
        and x.is_cuda
        and y.is_cuda
        and x.device == y.device
        and x.dtype == y.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim == 5
        and x.is_contiguous()
        and not x.requires_grad
        and not y.requires_grad
        and not torch.is_grad_enabled()
        and x.shape[-1] <= 2048
        and x.shape[2] <= 65535
        and x.shape[0] * x.shape[1] <= 65535
        and p1 >= 1
        and p2 >= 1
        and p3 >= 1
    ):
        return False
    _pixel_shuffle_add_w_slice_op(x, y, lo, w_len, t_in, h_in, w_in, p1, p2, p3, bool(drop_leading_frame))
    return True


__all__ = [
    "diffvae_cut_slab",
    "diffvae_pixel_shuffle_add_w_slice",
    "diffvae_rmsnorm_3d_abs_rope",
    "diffvae_rmsnorm_modulate",
    "diffvae_rmsnorm_modulate_into",
    "diffvae_w_slice_add",
    "triton_diffvae_ops_available",
]
