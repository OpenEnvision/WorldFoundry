# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for the HunyuanVideo-1.5 / HY-WorldPlay VideoVAE.

Replaces multi-pass eager PyTorch 5-D ops and unvectorized ``Conv3d`` bias
additions in :mod:`.h15` with single-pass, strictly ``int64``-indexed,
layout-preserving Triton kernels supporting BOTH ``NCDHW`` contiguous and
``channels_last_3d`` (``NDHWC``) tensors:

- :func:`triton_h15_rms_norm_silu_pad_3d` — super-fused ``(optional pre_bias) +
  RMS_norm + SiLU + 3-D replicate/constant/reflect pad`` feeding
  ``CausalConv3d.conv`` directly (also eliminates preceding ``conv1.bias`` pass).
- :func:`triton_h15_pad_3d_align_c` — fused 3-D pad + channel zero-padding to
  an 8-channel Tensor Core boundary in ``channels_last_3d`` (avoids cuDNN's
  18x slowdown on ``in_channels < 8`` ``channels_last_3d`` 3-D convolutions).
- :func:`triton_h15_upsample_add` — single-pass fused 3-D depth-to-space for
  ``h`` (+ optional ``h_bias``, with causal first-frame slicing) + shortcut
  depth-to-space + ``repeat_interleave`` + residual add.
- :func:`triton_h15_downsample_add` — single-pass fused 3-D space-to-depth for
  ``h`` (+ optional ``h_bias``, with causal first-frame channel duplication) +
  shortcut space-to-depth + group channel mean + residual add.
- :func:`triton_h15_repeat_add` — single-pass fused ``h (+ h_bias) +
  z.repeat_interleave`` for ``Decoder.conv_in``.
- :func:`triton_h15_residual_bias_add` — single-pass coalesced
  ``x + (h + bias)`` (or ``h + bias`` when ``x is None``) eliminating PyTorch's
  unvectorized 5-D ``Conv3d`` bias broadcast pass.

All CUDA ops are registered via ``torch.library.custom_op`` + ``register_fake``
for ``torch.compile`` and CUDA Graph capture safety. Importing this module does
not import Triton or initialize CUDA; CPU, autograd, and unsupported shapes or
dtypes fall back transparently to exact PyTorch implementations.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MODE_TO_CODE = {"constant": 0, "replicate": 1, "reflect": 2}


def _is_channels_last_3d(x: torch.Tensor) -> bool:
    """Return True when 5-D *x* uses ``channels_last_3d`` (``NDHWC``) memory order."""
    if x.ndim != 5:
        return False
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        if not x.is_contiguous():
            return True
        return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]
    return False


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_3d_torch_ref(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
) -> torch.Tensor:
    t_mode = spatial_mode if temporal_mode is None else temporal_mode
    if all(p == 0 for p in pad):
        return x
    if spatial_mode == t_mode:
        return F.pad(x, pad, mode=spatial_mode)
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    out = x
    if pw_l or pw_r or ph_l or ph_r:
        out = F.pad(out, (pw_l, pw_r, ph_l, ph_r, 0, 0), mode=spatial_mode)
    if pt_l or pt_r:
        out = F.pad(out, (0, 0, 0, 0, pt_l, pt_r), mode=t_mode)
    return out


def _rms_norm_silu_pad_3d_torch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pad: tuple[int, int, int, int, int, int],
    *,
    scale: float,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
    apply_silu: bool = True,
    pre_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    inp = x + pre_bias.view(1, -1, 1, 1, 1) if pre_bias is not None else x
    y = F.normalize(inp, dim=1) * scale * gamma
    if bias is not None:
        y = y + bias
    if apply_silu:
        y = F.silu(y, inplace=True)
    return _pad_3d_torch_ref(y, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)


def _upsample_add_torch(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_upsample: bool,
    repeats: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    h_eff = h + h_bias.view(1, -1, 1, 1, 1) if h_bias is not None else h
    r1 = 2 if add_temporal_upsample else 1
    if add_temporal_upsample:
        h_first = h_eff[:, :, :1, :, :]
        h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
        h_first = h_first[:, : h_first.shape[1] // 2]
        h_next = h_eff[:, :, 1:, :, :]
        h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        h_up = torch.cat([h_first, h_next], dim=2)

        x_first = x[:, :, :1, :, :]
        x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
        x_first = x_first.repeat_interleave(repeats=repeats // 2, dim=1)

        x_next = x[:, :, 1:, :, :]
        x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        x_next = x_next.repeat_interleave(repeats=repeats, dim=1)
        shortcut = torch.cat([x_first, x_next], dim=2)
    else:
        h_up = rearrange(h_eff, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
    return h_up + shortcut


def _downsample_add_torch(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_downsample: bool,
    group_size: int,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    h_eff = h + h_bias.view(1, -1, 1, 1, 1) if h_bias is not None else h
    r1 = 2 if add_temporal_downsample else 1
    if add_temporal_downsample:
        h_first = h_eff[:, :, :1, :, :]
        h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
        h_first = torch.cat([h_first, h_first], dim=1)
        h_next = h_eff[:, :, 1:, :, :]
        h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        h_dn = torch.cat([h_first, h_next], dim=2)

        x_first = x[:, :, :1, :, :]
        x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
        b_f, _, t_f, h_f, w_f = x_first.shape
        x_first = x_first.view(b_f, h_dn.shape[1], group_size // 2, t_f, h_f, w_f).mean(dim=2)

        x_next = x[:, :, 1:, :, :]
        x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        b_n, _, t_n, h_n, w_n = x_next.shape
        x_next = x_next.view(b_n, h_dn.shape[1], group_size, t_n, h_n, w_n).mean(dim=2)
        shortcut = torch.cat([x_first, x_next], dim=2)
    else:
        h_dn = rearrange(h_eff, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        b_s, _, t_s, h_s, w_s = shortcut.shape
        shortcut = shortcut.view(b_s, h_dn.shape[1], group_size, t_s, h_s, w_s).mean(dim=2)
    return h_dn + shortcut


def _repeat_add_torch(
    h: torch.Tensor,
    z: torch.Tensor,
    repeats: int,
    *,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    h_eff = h + h_bias.view(1, -1, 1, 1, 1) if h_bias is not None else h
    return h_eff + z.repeat_interleave(repeats=repeats, dim=1)


def _residual_bias_add_torch(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    h_eff = h + bias.view(1, -1, 1, 1, 1) if bias is not None else h
    return h_eff if x is None else x + h_eff


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Hopper SM90 Triton kernels for H15 VideoVAE."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="h15-vae-triton")

    import triton
    import triton.language as tl

    @triton.jit
    def _map_coord_i64(idx, pad_left, size, mode: tl.constexpr):
        """Map int64 output coordinate ``idx`` to int64 input coordinate and validity."""
        raw = idx - pad_left
        if mode == 0:  # constant zero
            valid = (raw >= 0) & (raw < size)
            coord = tl.maximum(tl.minimum(raw, size - 1), 0)
            return coord, valid
        elif mode == 1:  # replicate
            coord = tl.maximum(tl.minimum(raw, size - 1), 0)
            return coord, raw == raw
        else:  # reflect (guaranteed pad_left < size and pad_right < size)
            coord = tl.where(raw < 0, -raw, raw)
            coord = tl.where(coord >= size, 2 * (size - 1) - coord, coord)
            return coord, raw == raw

    # =========================================================================
    # 1. Super-fused (pre_bias +) RMS_norm + SiLU + 3-D Pad (NDHWC & NCDHW)
    # =========================================================================

    @triton.jit
    def _h15_rms_norm_silu_pad_3d_ndhwc_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        out_ptr,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        scale,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        apply_silu: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """Single-pass ``channels_last_3d`` ``(pre_bias +) RMS_norm + SiLU + 3D pad``."""
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_in_bounds = w_offs < w_out_i64
        c_offs = tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < channels
        out_mask = w_in_bounds[:, None] & c_mask[None, :]

        out_base = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64
        out_ptrs = out_base + w_offs[:, None] * so_w_i64 + c_offs[None, :]

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, pad_h_l_i64, h_in_i64, s_mode)

        if (not d_valid) or (not h_valid):
            tl.store(
                out_ptr + out_ptrs,
                tl.zeros((block_w, block_c), dtype=x_ptr.dtype.element_ty),
                mask=out_mask,
                eviction_policy="evict_first",
            )
            return

        w_i, w_valid = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)
        load_mask = (w_in_bounds & w_valid)[:, None] & c_mask[None, :]

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)

        in_base = b_idx * sx_b_i64 + d_i * sx_d_i64 + h_i * sx_h_i64
        in_ptrs = in_base + w_i[:, None] * sx_w_i64 + c_offs[None, :]

        x_in = tl.load(x_ptr + in_ptrs, mask=load_mask, other=0.0)
        x_f = x_in.to(tl.float32)
        if has_pre_bias:
            pre_b = tl.load(pre_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            x_f = (x_f + pre_b).to(x_in.dtype).to(tl.float32)

        norm = tl.maximum(tl.sqrt(tl.sum(x_f * x_f, axis=1)), 1e-12)
        inv_norm = (scale / norm)[:, None]

        gamma = tl.load(gamma_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)[None, :]
        y = x_f * inv_norm * gamma
        if has_bias:
            bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            y = y + bias
        if apply_silu:
            y = y * tl.sigmoid(y)
        y_out = y.to(x_in.dtype)
        if s_mode == 0:
            zero_v = tl.zeros((1, 1), dtype=x_in.dtype)
            y_out = tl.where(w_valid[:, None], y_out, zero_v)
        tl.store(out_ptr + out_ptrs, y_out, mask=out_mask, eviction_policy="evict_first")

    @triton.jit
    def _h15_rms_norm_silu_pad_3d_ncdhw_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        pre_bias_ptr,
        out_ptr,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
        scale,
        channels: tl.constexpr,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        has_bias: tl.constexpr,
        has_pre_bias: tl.constexpr,
        apply_silu: tl.constexpr,
        block_w: tl.constexpr,
    ):
        """Coalesced ``NCDHW`` ``(pre_bias +) RMS_norm + SiLU + 3D pad``."""
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, pad_h_l_i64, h_in_i64, s_mode)

        in_base = b_idx * sx_b_i64 + d_i * sx_d_i64 + h_i * sx_h_i64
        out_base = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64

        num_interior_w = (w_in_i64 + block_w - 1) // block_w
        if w_blk < num_interior_w:
            w_in_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
            w_mask = w_in_offs < w_in_i64
            w_out_offs = pad_w_l_i64 + w_in_offs

            if (not d_valid) or (not h_valid):
                zeros = tl.zeros((block_w,), dtype=x_ptr.dtype.element_ty)
                for c in range(channels):
                    tl.store(
                        out_ptr + out_base + c * so_c_i64 + w_out_offs,
                        zeros,
                        mask=w_mask,
                        eviction_policy="evict_first",
                    )
                return

            sq_sum = tl.zeros((block_w,), dtype=tl.float32)
            for c in range(channels):
                v_raw = tl.load(
                    x_ptr + in_base + c * sx_c_i64 + w_in_offs,
                    mask=w_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                v = v_raw.to(tl.float32)
                if has_pre_bias:
                    pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                    v = (v + pb).to(v_raw.dtype).to(tl.float32)
                sq_sum += v * v
            inv_norm = scale / tl.maximum(tl.sqrt(sq_sum), 1e-12)

            for c in range(channels):
                x_in = tl.load(
                    x_ptr + in_base + c * sx_c_i64 + w_in_offs,
                    mask=w_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                x_f = x_in.to(tl.float32)
                if has_pre_bias:
                    pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                    x_f = (x_f + pb).to(x_in.dtype).to(tl.float32)
                g = tl.load(gamma_ptr + c).to(tl.float32)
                y = x_f * inv_norm * g
                if has_bias:
                    y = y + tl.load(bias_ptr + c).to(tl.float32)
                if apply_silu:
                    y = y * tl.sigmoid(y)
                tl.store(
                    out_ptr + out_base + c * so_c_i64 + w_out_offs,
                    y.to(x_in.dtype),
                    mask=w_mask,
                    eviction_policy="evict_first",
                )
        else:
            pad_w_r_i64 = w_out_i64 - pad_w_l_i64 - w_in_i64
            total_border = pad_w_l_i64 + pad_w_r_i64
            for b_start in range(0, total_border, block_w):
                b_idx_w = b_start + tl.arange(0, block_w).to(tl.int64)
                b_in_bounds = b_idx_w < total_border
                w_offs = tl.where(b_idx_w < pad_w_l_i64, b_idx_w, pad_w_l_i64 + w_in_i64 + (b_idx_w - pad_w_l_i64))

                if (not d_valid) or (not h_valid) or (s_mode == 0):
                    zeros = tl.zeros((block_w,), dtype=x_ptr.dtype.element_ty)
                    for c in range(channels):
                        tl.store(out_ptr + out_base + c * so_c_i64 + w_offs, zeros, mask=b_in_bounds)
                else:
                    w_i, _ = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)
                    sq_sum = tl.zeros((block_w,), dtype=tl.float32)
                    for c in range(channels):
                        v_raw = tl.load(x_ptr + in_base + c * sx_c_i64 + w_i, mask=b_in_bounds, other=0.0)
                        v = v_raw.to(tl.float32)
                        if has_pre_bias:
                            pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                            v = (v + pb).to(v_raw.dtype).to(tl.float32)
                        sq_sum += v * v
                    inv_norm = scale / tl.maximum(tl.sqrt(sq_sum), 1e-12)

                    for c in range(channels):
                        x_in = tl.load(x_ptr + in_base + c * sx_c_i64 + w_i, mask=b_in_bounds, other=0.0)
                        x_f = x_in.to(tl.float32)
                        if has_pre_bias:
                            pb = tl.load(pre_bias_ptr + c).to(tl.float32)
                            x_f = (x_f + pb).to(x_in.dtype).to(tl.float32)
                        g = tl.load(gamma_ptr + c).to(tl.float32)
                        y = x_f * inv_norm * g
                        if has_bias:
                            y = y + tl.load(bias_ptr + c).to(tl.float32)
                        if apply_silu:
                            y = y * tl.sigmoid(y)
                        tl.store(out_ptr + out_base + c * so_c_i64 + w_offs, y.to(x_in.dtype), mask=b_in_bounds)

    # =========================================================================
    # 2. Fused 3-D Pad + Channel Zero-Pad Alignment to NDHWC (for cin < 8)
    # =========================================================================

    @triton.jit
    def _h15_pad_3d_align_c_kernel(
        x_ptr,
        out_ptr,
        c_in,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        t_mode: tl.constexpr,
        s_mode: tl.constexpr,
        c_align: tl.constexpr,
        block_w: tl.constexpr,
    ):
        """Pad (D, H, W) and zero-pad C -> ``c_align`` in ``channels_last_3d``."""
        bdh = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)

        d_in_i64 = tl.cast(d_in, tl.int64)
        h_in_i64 = tl.cast(h_in, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        d_out_i64 = tl.cast(d_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        pad_t_l_i64 = tl.cast(pad_t_l, tl.int64)
        pad_h_l_i64 = tl.cast(pad_h_l, tl.int64)
        pad_w_l_i64 = tl.cast(pad_w_l, tl.int64)

        dh_out = d_out_i64 * h_out_i64
        b_idx = bdh // dh_out
        rem = bdh - b_idx * dh_out
        d_o = rem // h_out_i64
        h_o = rem - d_o * h_out_i64

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_in_bounds = w_offs < w_out_i64
        c_offs = tl.arange(0, c_align).to(tl.int64)
        c_valid = c_offs < c_in

        d_i, d_valid = _map_coord_i64(d_o, pad_t_l_i64, d_in_i64, t_mode)
        h_i, h_valid = _map_coord_i64(h_o, pad_h_l_i64, h_in_i64, s_mode)
        w_i, w_valid = _map_coord_i64(w_offs, pad_w_l_i64, w_in_i64, s_mode)

        load_mask = (d_valid & h_valid & w_in_bounds & w_valid)[:, None] & c_valid[None, :]

        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)

        in_ptrs = (
            b_idx * sx_b_i64 + d_i * sx_d_i64 + h_i * sx_h_i64 + w_i[:, None] * sx_w_i64 + c_offs[None, :] * sx_c_i64
        )
        vals = tl.load(x_ptr + in_ptrs, mask=load_mask, other=0.0)

        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)
        out_ptrs = b_idx * so_b_i64 + d_o * so_d_i64 + h_o * so_h_i64 + w_offs[:, None] * so_w_i64 + c_offs[None, :]
        tl.store(out_ptr + out_ptrs, vals, mask=w_in_bounds[:, None])

    # =========================================================================
    # 3. Fused Upsample Depth-to-Space (+ h_bias) + Shortcut Repeat + Add
    # =========================================================================

    @triton.jit
    def _h15_upsample_add_ndhwc_fast_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        w_in,
        t_out,
        h_out,
        c_in,
        c_out,
        sh_b,
        sh_d,
        sh_h,
        sh_w,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        add_temporal: tl.constexpr,
        has_h_bias: tl.constexpr,
        repeats: tl.constexpr,
        block_w_in: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """100% coalesced 128-bit ``channels_last_3d`` ``Upsample`` (+ ``h_bias``) + add."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        w_in_i64 = tl.cast(w_in, tl.int64)
        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)

        th_out = t_out_i64 * h_out_i64
        b_idx = bth // th_out
        rem = bth - b_idx * th_out
        t_o = rem // h_out_i64
        h_o = rem - t_o * h_out_i64

        h_i = h_o >> 1
        rh = h_o & 1

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        wi_mask = wi_offs < w_in_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = wi_mask[:, None] & c_mask[None, :]

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sh_w_i64 = tl.cast(sh_w, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        out_base = b_idx * so_b_i64 + t_o * so_d_i64 + h_o * so_h_i64
        out_ptrs0 = out_base + (2 * wi_offs)[:, None] * so_w_i64 + c_offs[None, :]
        out_ptrs1 = out_base + (2 * wi_offs + 1)[:, None] * so_w_i64 + c_offs[None, :]

        if add_temporal:
            if t_o == 0:
                rep_f: tl.constexpr = repeats // 2
                cx_len_f: tl.constexpr = block_c // rep_f
                sp0_f = 2 * rh
                sp1_f = sp0_f + 1
                ch_h0_f = sp0_f * (2 * c_out_i64) + c_offs
                ch_h1_f = sp1_f * (2 * c_out_i64) + c_offs
                h_base_f = b_idx * sh_b_i64 + h_i * sh_h_i64 + wi_offs[:, None] * sh_w_i64
                h0_f = tl.load(h_ptr + h_base_f + ch_h0_f[None, :], mask=mask, other=0.0)
                h1_f = tl.load(h_ptr + h_base_f + ch_h1_f[None, :], mask=mask, other=0.0)
                if has_h_bias:
                    hb0_f = tl.load(h_bias_ptr + ch_h0_f, mask=c_mask, other=0.0).to(tl.float32)[None, :]
                    hb1_f = tl.load(h_bias_ptr + ch_h1_f, mask=c_mask, other=0.0).to(tl.float32)[None, :]
                    h0_f = (h0_f.to(tl.float32) + hb0_f).to(out_ty)
                    h1_f = (h1_f.to(tl.float32) + hb1_f).to(out_ty)

                cx_base_f = c_blk * cx_len_f + tl.arange(0, cx_len_f).to(tl.int64)
                cx_mask_f = wi_mask[:, None] & (cx_base_f < (c_in_i64 // 4))[None, :]
                x_base_f = b_idx * sx_b_i64 + h_i * sx_h_i64 + wi_offs[:, None] * sx_w_i64
                xs0_f = tl.load(
                    x_ptr + x_base_f + sp0_f * (c_in_i64 // 4) + cx_base_f[None, :], mask=cx_mask_f, other=0.0
                )
                xs1_f = tl.load(
                    x_ptr + x_base_f + sp1_f * (c_in_i64 // 4) + cx_base_f[None, :], mask=cx_mask_f, other=0.0
                )
                x0_f = tl.reshape(
                    tl.broadcast_to(xs0_f[:, :, None], (block_w_in, cx_len_f, rep_f)), (block_w_in, block_c)
                )
                x1_f = tl.reshape(
                    tl.broadcast_to(xs1_f[:, :, None], (block_w_in, cx_len_f, rep_f)), (block_w_in, block_c)
                )

                out0_f = (h0_f.to(tl.float32) + x0_f.to(tl.float32)).to(out_ty)
                out1_f = (h1_f.to(tl.float32) + x1_f.to(tl.float32)).to(out_ty)
                tl.store(out_ptr + out_ptrs0, out0_f, mask=mask, eviction_policy="evict_first")
                tl.store(out_ptr + out_ptrs1, out1_f, mask=mask, eviction_policy="evict_first")
                return

            t_next = t_o - 1
            t_i = 1 + (t_next >> 1)
            rt = t_next & 1
            sp0 = 4 * rt + 2 * rh
            sp1 = sp0 + 1
            cx_stride = c_in_i64 // 8
        else:
            t_i = t_o
            sp0 = 2 * rh
            sp1 = sp0 + 1
            cx_stride = c_out_i64 // repeats

        cx_len: tl.constexpr = block_c // repeats
        ch_h0 = sp0 * c_out_i64 + c_offs
        ch_h1 = sp1 * c_out_i64 + c_offs
        h_base = b_idx * sh_b_i64 + t_i * sh_d_i64 + h_i * sh_h_i64 + wi_offs[:, None] * sh_w_i64
        h0 = tl.load(h_ptr + h_base + ch_h0[None, :], mask=mask, other=0.0, eviction_policy="evict_first")
        h1 = tl.load(h_ptr + h_base + ch_h1[None, :], mask=mask, other=0.0, eviction_policy="evict_first")
        if has_h_bias:
            hb0 = tl.load(h_bias_ptr + ch_h0, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            hb1 = tl.load(h_bias_ptr + ch_h1, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            h0 = (h0.to(tl.float32) + hb0).to(out_ty)
            h1 = (h1.to(tl.float32) + hb1).to(out_ty)

        cx_base = c_blk * cx_len + tl.arange(0, cx_len).to(tl.int64)
        cx_mask = wi_mask[:, None] & (cx_base < cx_stride)[None, :]
        x_base = b_idx * sx_b_i64 + t_i * sx_d_i64 + h_i * sx_h_i64 + wi_offs[:, None] * sx_w_i64
        xs0 = tl.load(x_ptr + x_base + sp0 * cx_stride + cx_base[None, :], mask=cx_mask, other=0.0)
        xs1 = tl.load(x_ptr + x_base + sp1 * cx_stride + cx_base[None, :], mask=cx_mask, other=0.0)
        x0 = tl.reshape(tl.broadcast_to(xs0[:, :, None], (block_w_in, cx_len, repeats)), (block_w_in, block_c))
        x1 = tl.reshape(tl.broadcast_to(xs1[:, :, None], (block_w_in, cx_len, repeats)), (block_w_in, block_c))

        out0 = (h0.to(tl.float32) + x0.to(tl.float32)).to(out_ty)
        out1 = (h1.to(tl.float32) + x1.to(tl.float32)).to(out_ty)
        tl.store(out_ptr + out_ptrs0, out0, mask=mask, eviction_policy="evict_first")
        tl.store(out_ptr + out_ptrs1, out1, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _h15_upsample_add_ndhwc_general_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_out,
        h_out,
        w_out,
        c_in,
        c_out,
        sh_b,
        sh_d,
        sh_h,
        sh_w,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        add_temporal: tl.constexpr,
        has_h_bias: tl.constexpr,
        repeats: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """General ``channels_last_3d`` ``Upsample`` (+ ``h_bias``) + add for arbitrary ``repeats``."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        c_blk = tl.program_id(2).to(tl.int64)

        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)

        th_out = t_out_i64 * h_out_i64
        b_idx = bth // th_out
        rem = bth - b_idx * th_out
        t_o = rem // h_out_i64
        h_o = rem - t_o * h_out_i64

        h_i = h_o >> 1
        rh = h_o & 1

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = w_offs < w_out_i64
        w_i = w_offs >> 1
        rw = w_offs & 1

        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = w_mask[:, None] & c_mask[None, :]

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sh_w_i64 = tl.cast(sh_w, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        if add_temporal:
            if t_o == 0:
                t_i = tl.cast(0, tl.int64)
                sp = 2 * rh + rw
                h_ch = sp[:, None] * (2 * c_out_i64) + c_offs[None, :]
                x_ch = sp[:, None] * (c_in_i64 // 4) + (c_offs // (repeats // 2))[None, :]
            else:
                t_next = t_o - 1
                t_i = 1 + (t_next >> 1)
                rt = t_next & 1
                sp = 4 * rt + 2 * rh + rw
                h_ch = sp[:, None] * c_out_i64 + c_offs[None, :]
                x_ch = sp[:, None] * (c_in_i64 // 8) + (c_offs // repeats)[None, :]
        else:
            t_i = t_o
            sp = 2 * rh + rw
            h_ch = sp[:, None] * c_out_i64 + c_offs[None, :]
            x_ch = (sp[:, None] * c_out_i64 + c_offs[None, :]) // repeats

        h_ptrs = b_idx * sh_b_i64 + t_i * sh_d_i64 + h_i * sh_h_i64 + w_i[:, None] * sh_w_i64 + h_ch
        x_ptrs = b_idx * sx_b_i64 + t_i * sx_d_i64 + h_i * sx_h_i64 + w_i[:, None] * sx_w_i64 + x_ch

        h_val = tl.load(h_ptr + h_ptrs, mask=mask, other=0.0)
        if has_h_bias:
            hb = tl.load(h_bias_ptr + h_ch, mask=mask, other=0.0).to(tl.float32)
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)
        x_val = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
        out_val = (h_val.to(tl.float32) + x_val.to(tl.float32)).to(out_ty)

        out_ptrs = b_idx * so_b_i64 + t_o * so_d_i64 + h_o * so_h_i64 + w_offs[:, None] * so_w_i64 + c_offs[None, :]
        tl.store(out_ptr + out_ptrs, out_val, mask=mask)

    @triton.jit
    def _h15_upsample_add_ncdhw_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        h_out,
        w_in,
        w_out,
        t_out,
        c_in,
        c_out,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
        add_temporal: tl.constexpr,
        has_h_bias: tl.constexpr,
        repeats: tl.constexpr,
        rows_per_cta: tl.constexpr,
        block_w_in: tl.constexpr,
    ):
        """Bit-exact coalesced ``NCDHW`` ``Upsample`` (+ ``h_bias``) + add."""
        bct = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)

        t_o = bct % t_out_i64
        bc = bct // t_out_i64
        c_o = bc % c_out_i64
        b_idx = bc // c_out_i64

        wi_offs = w_blk * block_w_in + tl.arange(0, block_w_in).to(tl.int64)
        wi_mask = wi_offs < w_in_i64
        wo_offs = w_blk * (2 * block_w_in) + tl.arange(0, 2 * block_w_in).to(tl.int64)
        wo_mask = wo_offs < w_out_i64

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_c_i64 = tl.cast(sh_c, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        if add_temporal:
            if t_o == 0:
                t_i = tl.cast(0, tl.int64)
                rt = tl.cast(0, tl.int64)
            else:
                t_next = t_o - 1
                t_i = 1 + (t_next >> 1)
                rt = t_next & 1
        else:
            t_i = t_o
            rt = tl.cast(0, tl.int64)

        h_bt = b_idx * sh_b_i64 + t_i * sh_d_i64 + wi_offs
        x_bt = b_idx * sx_b_i64 + t_i * sx_d_i64 + wi_offs
        out_bct = b_idx * so_b_i64 + c_o * so_c_i64 + t_o * so_d_i64 + wo_offs

        h_o_base = h_blk * rows_per_cta
        for r in tl.static_range(0, rows_per_cta):
            h_o = h_o_base + r
            if h_o < h_out_i64:
                h_i = h_o >> 1
                rh = h_o & 1
                if add_temporal:
                    if t_o == 0:
                        sp0 = 2 * rh
                        sp1 = sp0 + 1
                        ch_h0 = sp0 * (2 * c_out_i64) + c_o
                        ch_h1 = sp1 * (2 * c_out_i64) + c_o
                        ch_x0 = sp0 * (c_in_i64 // 4) + (c_o // (repeats // 2))
                        ch_x1 = sp1 * (c_in_i64 // 4) + (c_o // (repeats // 2))
                    else:
                        sp0 = 4 * rt + 2 * rh
                        sp1 = sp0 + 1
                        ch_h0 = sp0 * c_out_i64 + c_o
                        ch_h1 = sp1 * c_out_i64 + c_o
                        ch_x0 = sp0 * (c_in_i64 // 8) + (c_o // repeats)
                        ch_x1 = sp1 * (c_in_i64 // 8) + (c_o // repeats)
                else:
                    sp0 = 2 * rh
                    sp1 = sp0 + 1
                    ch_h0 = sp0 * c_out_i64 + c_o
                    ch_h1 = sp1 * c_out_i64 + c_o
                    ch_x0 = (sp0 * c_out_i64 + c_o) // repeats
                    ch_x1 = (sp1 * c_out_i64 + c_o) // repeats

                h_row = h_bt + h_i * sh_h_i64
                x_row = x_bt + h_i * sx_h_i64
                h0 = tl.load(h_ptr + h_row + ch_h0 * sh_c_i64, mask=wi_mask, other=0.0, eviction_policy="evict_first")
                h1 = tl.load(h_ptr + h_row + ch_h1 * sh_c_i64, mask=wi_mask, other=0.0, eviction_policy="evict_first")
                if has_h_bias:
                    hb0 = tl.load(h_bias_ptr + ch_h0).to(tl.float32)
                    hb1 = tl.load(h_bias_ptr + ch_h1).to(tl.float32)
                    h0 = (h0.to(tl.float32) + hb0).to(out_ty)
                    h1 = (h1.to(tl.float32) + hb1).to(out_ty)
                x0 = tl.load(x_ptr + x_row + ch_x0 * sx_c_i64, mask=wi_mask, other=0.0)
                x1 = tl.load(x_ptr + x_row + ch_x1 * sx_c_i64, mask=wi_mask, other=0.0)

                out0 = (h0.to(tl.float32) + x0.to(tl.float32)).to(out_ty)
                out1 = (h1.to(tl.float32) + x1.to(tl.float32)).to(out_ty)
                packed = tl.reshape(tl.join(out0, out1), (2 * block_w_in,))
                tl.store(out_ptr + out_bct + h_o * so_h_i64, packed, mask=wo_mask, eviction_policy="evict_first")

    # =========================================================================
    # 4. Fused Downsample Space-to-Depth (+ h_bias) + Shortcut Group Mean + Add
    # =========================================================================

    @triton.jit
    def _h15_downsample_add_ndhwc_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        t_out,
        h_out,
        w_out,
        c_in,
        c_h,
        sh_b,
        sh_d,
        sh_h,
        sh_w,
        sx_b,
        sx_d,
        sx_h,
        sx_w,
        so_b,
        so_d,
        so_h,
        so_w,
        add_temporal: tl.constexpr,
        has_h_bias: tl.constexpr,
        group_size: tl.constexpr,
        is_pow2_g: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
        c_tiles_per_sp: tl.constexpr,
    ):
        """Coalesced single-pass ``channels_last_3d`` ``Downsample`` (+ ``h_bias``) + add."""
        bth = tl.program_id(0).to(tl.int64)
        w_blk = tl.program_id(1).to(tl.int64)
        sp_c_blk = tl.program_id(2).to(tl.int64)

        sp = sp_c_blk // c_tiles_per_sp
        c_blk = sp_c_blk - sp * c_tiles_per_sp

        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        c_h_i64 = tl.cast(c_h, tl.int64)

        th_out = t_out_i64 * h_out_i64
        b_idx = bth // th_out
        rem = bth - b_idx * th_out
        t_o = rem // h_out_i64
        h_o = rem - t_o * h_out_i64

        w_offs = w_blk * block_w + tl.arange(0, block_w).to(tl.int64)
        w_mask = w_offs < w_out_i64
        ch_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        ch_mask = ch_offs < c_h_i64
        mask = w_mask[:, None] & ch_mask[None, :]

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sh_w_i64 = tl.cast(sh_w, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        sx_w_i64 = tl.cast(sx_w, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        so_w_i64 = tl.cast(so_w, tl.int64)

        out_ty = out_ptr.dtype.element_ty

        rh_h = (sp >> 1) & 1
        rw_h = sp & 1
        if add_temporal:
            if t_o == 0:
                t_ih = tl.cast(0, tl.int64)
            else:
                rt_h = sp >> 2
                t_ih = 2 * t_o - 1 + rt_h
        else:
            t_ih = t_o

        h_ptrs = (
            b_idx * sh_b_i64
            + t_ih * sh_d_i64
            + (2 * h_o + rh_h) * sh_h_i64
            + (2 * w_offs + rw_h)[:, None] * sh_w_i64
            + ch_offs[None, :]
        )
        h_val = tl.load(h_ptr + h_ptrs, mask=mask, other=0.0)
        if has_h_bias:
            hb = tl.load(h_bias_ptr + ch_offs, mask=ch_mask, other=0.0).to(tl.float32)[None, :]
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)

        co_offs = sp * c_h_i64 + ch_offs
        out_ptrs = b_idx * so_b_i64 + t_o * so_d_i64 + h_o * so_h_i64 + w_offs[:, None] * so_w_i64 + co_offs[None, :]

        if add_temporal and (t_o == 0):
            g_first: tl.constexpr = group_size // 2
            sp_x = sp >> 1
            rh_x = sp_x >> 1
            rw_x = sp_x & 1
            cx_half_offset = (sp & 1) * (c_in_i64 // 2)
            x_base_f = b_idx * sx_b_i64 + (2 * h_o + rh_x) * sx_h_i64 + (2 * w_offs + rw_x)[:, None] * sx_w_i64
            if is_pow2_g:
                cx_rel_f = c_blk * (block_c * g_first) + tl.arange(0, block_c * g_first).to(tl.int64)
                cx_offs_f = cx_half_offset + cx_rel_f
                cx_mask_f = cx_rel_f < (c_h_i64 * g_first)
                x_vals_f = tl.load(
                    x_ptr + x_base_f + cx_offs_f[None, :],
                    mask=w_mask[:, None] & cx_mask_f[None, :],
                    other=0.0,
                ).to(tl.float32)
                x_mean_f = tl.sum(tl.reshape(x_vals_f, (block_w, block_c, g_first)), axis=2) * (1.0 / g_first)
            else:
                acc_f = tl.zeros((block_w, block_c), dtype=tl.float32)
                for j in tl.static_range(0, g_first):
                    cx_f = cx_half_offset + ch_offs * g_first + j
                    acc_f += tl.load(x_ptr + x_base_f + cx_f[None, :], mask=mask, other=0.0).to(tl.float32)
                x_mean_f = acc_f * (1.0 / g_first)
            out_val_f = (h_val.to(tl.float32) + x_mean_f.to(out_ty).to(tl.float32)).to(out_ty)
            tl.store(out_ptr + out_ptrs, out_val_f, mask=mask, eviction_policy="evict_first")
            return

        x_base = (
            b_idx * sx_b_i64 + t_ih * sx_d_i64 + (2 * h_o + rh_h) * sx_h_i64 + (2 * w_offs + rw_h)[:, None] * sx_w_i64
        )
        if is_pow2_g:
            cx_offs = c_blk * (block_c * group_size) + tl.arange(0, block_c * group_size).to(tl.int64)
            cx_mask = cx_offs < c_in_i64
            x_vals = tl.load(
                x_ptr + x_base + cx_offs[None, :],
                mask=w_mask[:, None] & cx_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_mean = tl.sum(tl.reshape(x_vals, (block_w, block_c, group_size)), axis=2) * (1.0 / group_size)
        else:
            acc = tl.zeros((block_w, block_c), dtype=tl.float32)
            for j in tl.static_range(0, group_size):
                cx = ch_offs * group_size + j
                acc += tl.load(x_ptr + x_base + cx[None, :], mask=mask, other=0.0).to(tl.float32)
            x_mean = acc * (1.0 / group_size)

        out_val = (h_val.to(tl.float32) + x_mean.to(out_ty).to(tl.float32)).to(out_ty)
        tl.store(out_ptr + out_ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _h15_downsample_add_ncdhw_kernel(
        h_ptr,
        x_ptr,
        h_bias_ptr,
        out_ptr,
        h_out,
        w_in,
        w_out,
        t_out,
        c_in,
        c_h,
        sh_b,
        sh_c,
        sh_d,
        sh_h,
        sx_b,
        sx_c,
        sx_d,
        sx_h,
        so_b,
        so_c,
        so_d,
        so_h,
        add_temporal: tl.constexpr,
        has_h_bias: tl.constexpr,
        group_size: tl.constexpr,
        num_pairs: tl.constexpr,
        rows_per_cta: tl.constexpr,
        block_w_out: tl.constexpr,
    ):
        """Coalesced ``NCDHW`` ``Downsample`` (+ ``h_bias``) + group-mean add."""
        bpct = tl.program_id(0).to(tl.int64)
        h_blk = tl.program_id(1).to(tl.int64)
        w_blk = tl.program_id(2).to(tl.int64)

        t_out_i64 = tl.cast(t_out, tl.int64)
        h_out_i64 = tl.cast(h_out, tl.int64)
        w_in_i64 = tl.cast(w_in, tl.int64)
        w_out_i64 = tl.cast(w_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        c_h_i64 = tl.cast(c_h, tl.int64)

        t_o = bpct % t_out_i64
        rem0 = bpct // t_out_i64
        c_h_idx = rem0 % c_h_i64
        rem1 = rem0 // c_h_i64
        q = rem1 % num_pairs
        b_idx = rem1 // num_pairs

        sp0 = 2 * q
        sp1 = sp0 + 1
        co0 = sp0 * c_h_i64 + c_h_idx
        co1 = sp1 * c_h_i64 + c_h_idx

        wo_offs = w_blk * block_w_out + tl.arange(0, block_w_out).to(tl.int64)
        wo_mask = wo_offs < w_out_i64
        wi_offs = w_blk * (2 * block_w_out) + tl.arange(0, 2 * block_w_out).to(tl.int64)
        wi_mask = wi_offs < w_in_i64

        sh_b_i64 = tl.cast(sh_b, tl.int64)
        sh_c_i64 = tl.cast(sh_c, tl.int64)
        sh_d_i64 = tl.cast(sh_d, tl.int64)
        sh_h_i64 = tl.cast(sh_h, tl.int64)
        sx_b_i64 = tl.cast(sx_b, tl.int64)
        sx_c_i64 = tl.cast(sx_c, tl.int64)
        sx_d_i64 = tl.cast(sx_d, tl.int64)
        sx_h_i64 = tl.cast(sx_h, tl.int64)
        so_b_i64 = tl.cast(so_b, tl.int64)
        so_c_i64 = tl.cast(so_c, tl.int64)
        so_d_i64 = tl.cast(so_d, tl.int64)
        so_h_i64 = tl.cast(so_h, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        hb_scalar = tl.load(h_bias_ptr + c_h_idx).to(tl.float32) if has_h_bias else 0.0

        rh_h = q & 1
        if add_temporal:
            if t_o == 0:
                t_ih = tl.cast(0, tl.int64)
            else:
                rt_h = q >> 1
                t_ih = 2 * t_o - 1 + rt_h
        else:
            t_ih = t_o

        h_bct = b_idx * sh_b_i64 + c_h_idx * sh_c_i64 + t_ih * sh_d_i64 + wi_offs
        out_bt0 = b_idx * so_b_i64 + co0 * so_c_i64 + t_o * so_d_i64 + wo_offs
        out_bt1 = b_idx * so_b_i64 + co1 * so_c_i64 + t_o * so_d_i64 + wo_offs

        h_o_base = h_blk * rows_per_cta
        if add_temporal and (t_o == 0):
            g_first: tl.constexpr = group_size // 2
            rh_x = q >> 1
            rw_x = q & 1
            cx0_base = c_h_idx * g_first
            cx1_base = (c_in_i64 // 2) + cx0_base
            x_bw_f = b_idx * sx_b_i64 + (2 * wo_offs + rw_x)
            for r in tl.static_range(0, rows_per_cta):
                h_o = h_o_base + r
                if h_o < h_out_i64:
                    h_pair_f = tl.load(h_ptr + h_bct + (2 * h_o + rh_h) * sh_h_i64, mask=wi_mask, other=0.0)
                    h_even_f, h_odd_f = tl.split(tl.reshape(h_pair_f, (block_w_out, 2)))
                    if has_h_bias:
                        h_even_f = (h_even_f.to(tl.float32) + hb_scalar).to(out_ty)
                        h_odd_f = (h_odd_f.to(tl.float32) + hb_scalar).to(out_ty)
                    x_row_f = x_bw_f + (2 * h_o + rh_x) * sx_h_i64
                    acc0_f = tl.zeros((block_w_out,), dtype=tl.float32)
                    acc1_f = tl.zeros((block_w_out,), dtype=tl.float32)
                    for j in tl.static_range(0, g_first):
                        acc0_f += tl.load(x_ptr + x_row_f + (cx0_base + j) * sx_c_i64, mask=wo_mask, other=0.0).to(
                            tl.float32
                        )
                        acc1_f += tl.load(x_ptr + x_row_f + (cx1_base + j) * sx_c_i64, mask=wo_mask, other=0.0).to(
                            tl.float32
                        )
                    m0_f = (acc0_f * (1.0 / g_first)).to(out_ty).to(tl.float32)
                    m1_f = (acc1_f * (1.0 / g_first)).to(out_ty).to(tl.float32)
                    tl.store(
                        out_ptr + out_bt0 + h_o * so_h_i64,
                        (h_even_f.to(tl.float32) + m0_f).to(out_ty),
                        mask=wo_mask,
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        out_ptr + out_bt1 + h_o * so_h_i64,
                        (h_odd_f.to(tl.float32) + m1_f).to(out_ty),
                        mask=wo_mask,
                        eviction_policy="evict_first",
                    )
            return

        x_bt = b_idx * sx_b_i64 + t_ih * sx_d_i64 + wi_offs
        cx_base = c_h_idx * group_size
        for r in tl.static_range(0, rows_per_cta):
            h_o = h_o_base + r
            if h_o < h_out_i64:
                row_in = 2 * h_o + rh_h
                h_pair = tl.load(
                    h_ptr + h_bct + row_in * sh_h_i64, mask=wi_mask, other=0.0, eviction_policy="evict_first"
                )
                h_even, h_odd = tl.split(tl.reshape(h_pair, (block_w_out, 2)))
                if has_h_bias:
                    h_even = (h_even.to(tl.float32) + hb_scalar).to(out_ty)
                    h_odd = (h_odd.to(tl.float32) + hb_scalar).to(out_ty)
                x_row = x_bt + row_in * sx_h_i64
                acc0 = tl.zeros((block_w_out,), dtype=tl.float32)
                acc1 = tl.zeros((block_w_out,), dtype=tl.float32)
                for j in tl.static_range(0, group_size):
                    x_pair = tl.load(
                        x_ptr + x_row + (cx_base + j) * sx_c_i64,
                        mask=wi_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    x_even, x_odd = tl.split(tl.reshape(x_pair, (block_w_out, 2)))
                    acc0 += x_even.to(tl.float32)
                    acc1 += x_odd.to(tl.float32)
                m0 = (acc0 * (1.0 / group_size)).to(out_ty).to(tl.float32)
                m1 = (acc1 * (1.0 / group_size)).to(out_ty).to(tl.float32)
                tl.store(
                    out_ptr + out_bt0 + h_o * so_h_i64,
                    (h_even.to(tl.float32) + m0).to(out_ty),
                    mask=wo_mask,
                    eviction_policy="evict_first",
                )
                tl.store(
                    out_ptr + out_bt1 + h_o * so_h_i64,
                    (h_odd.to(tl.float32) + m1).to(out_ty),
                    mask=wo_mask,
                    eviction_policy="evict_first",
                )

    # =========================================================================
    # 5. Fused Decoder conv_in Residual Repeat Add: h (+ h_bias) + repeat(z)
    # =========================================================================

    @triton.jit
    def _h15_repeat_add_ndhwc_kernel(
        h_ptr,
        z_ptr,
        h_bias_ptr,
        out_ptr,
        total_voxels,
        c_out,
        c_in,
        has_h_bias: tl.constexpr,
        repeats: tl.constexpr,
        block_v: tl.constexpr,
        block_c: tl.constexpr,
    ):
        v_blk = tl.program_id(0).to(tl.int64)
        c_blk = tl.program_id(1).to(tl.int64)

        total_v_i64 = tl.cast(total_voxels, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        v_offs = v_blk * block_v + tl.arange(0, block_v).to(tl.int64)
        v_mask = v_offs < total_v_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_out_i64
        mask = v_mask[:, None] & c_mask[None, :]

        h_ptrs = v_offs[:, None] * c_out_i64 + c_offs[None, :]
        z_ptrs = v_offs[:, None] * c_in_i64 + (c_offs // repeats)[None, :]

        h_val = tl.load(h_ptr + h_ptrs, mask=mask, other=0.0)
        if has_h_bias:
            hb = tl.load(h_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)
        z_val = tl.load(z_ptr + z_ptrs, mask=mask, other=0.0)
        out_val = (h_val.to(tl.float32) + z_val.to(tl.float32)).to(out_ty)
        tl.store(out_ptr + h_ptrs, out_val, mask=mask)

    @triton.jit
    def _h15_repeat_add_ncdhw_kernel(
        h_ptr,
        z_ptr,
        h_bias_ptr,
        out_ptr,
        spatial_size,
        c_out,
        c_in,
        has_h_bias: tl.constexpr,
        repeats: tl.constexpr,
        block_s: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        s_blk = tl.program_id(1).to(tl.int64)

        spatial_i64 = tl.cast(spatial_size, tl.int64)
        c_out_i64 = tl.cast(c_out, tl.int64)
        c_in_i64 = tl.cast(c_in, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        c_o = bc % c_out_i64
        b_idx = bc // c_out_i64
        c_i = c_o // repeats

        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64

        h_ptrs = (b_idx * c_out_i64 + c_o) * spatial_i64 + s_offs
        z_ptrs = (b_idx * c_in_i64 + c_i) * spatial_i64 + s_offs

        h_val = tl.load(h_ptr + h_ptrs, mask=s_mask, other=0.0)
        if has_h_bias:
            hb = tl.load(h_bias_ptr + c_o).to(tl.float32)
            h_val = (h_val.to(tl.float32) + hb).to(out_ty)
        z_val = tl.load(z_ptr + z_ptrs, mask=s_mask, other=0.0)
        out_val = (h_val.to(tl.float32) + z_val.to(tl.float32)).to(out_ty)
        tl.store(out_ptr + h_ptrs, out_val, mask=s_mask)

    # =========================================================================
    # 6. Fused Coalesced Residual + Conv3d Bias Add: (x +) (h + bias)
    # =========================================================================

    @triton.jit
    def _h15_residual_bias_add_ndhwc_kernel(
        x_ptr,
        h_ptr,
        bias_ptr,
        out_ptr,
        total_voxels,
        channels,
        has_x: tl.constexpr,
        has_bias: tl.constexpr,
        block_v: tl.constexpr,
        block_c: tl.constexpr,
    ):
        v_blk = tl.program_id(0).to(tl.int64)
        c_blk = tl.program_id(1).to(tl.int64)

        total_v_i64 = tl.cast(total_voxels, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        v_offs = v_blk * block_v + tl.arange(0, block_v).to(tl.int64)
        v_mask = v_offs < total_v_i64
        c_offs = c_blk * block_c + tl.arange(0, block_c).to(tl.int64)
        c_mask = c_offs < c_i64
        mask = v_mask[:, None] & c_mask[None, :]

        ptrs = v_offs[:, None] * c_i64 + c_offs[None, :]
        h_val = tl.load(h_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
        if has_bias:
            b_val = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)[None, :]
            h_val = (h_val.to(tl.float32) + b_val).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=mask, eviction_policy="evict_first")

    @triton.jit
    def _h15_residual_bias_add_ncdhw_kernel(
        x_ptr,
        h_ptr,
        bias_ptr,
        out_ptr,
        spatial_size,
        channels,
        has_x: tl.constexpr,
        has_bias: tl.constexpr,
        block_s: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        s_blk = tl.program_id(1).to(tl.int64)

        spatial_i64 = tl.cast(spatial_size, tl.int64)
        c_i64 = tl.cast(channels, tl.int64)
        out_ty = out_ptr.dtype.element_ty

        c_idx = bc % c_i64
        s_offs = s_blk * block_s + tl.arange(0, block_s).to(tl.int64)
        s_mask = s_offs < spatial_i64

        ptrs = bc * spatial_i64 + s_offs
        h_val = tl.load(h_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
        if has_bias:
            b_val = tl.load(bias_ptr + c_idx).to(tl.float32)
            h_val = (h_val.to(tl.float32) + b_val).to(out_ty)
        if has_x:
            x_val = tl.load(x_ptr + ptrs, mask=s_mask, other=0.0, eviction_policy="evict_first")
            out_val = (x_val.to(tl.float32) + h_val.to(tl.float32)).to(out_ty)
        else:
            out_val = h_val
        tl.store(out_ptr + ptrs, out_val, mask=s_mask, eviction_policy="evict_first")

    return {
        "triton": triton,
        "rms_pad_ndhwc": _h15_rms_norm_silu_pad_3d_ndhwc_kernel,
        "rms_pad_ncdhw": _h15_rms_norm_silu_pad_3d_ncdhw_kernel,
        "pad_align_c": _h15_pad_3d_align_c_kernel,
        "up_ndhwc_fast": _h15_upsample_add_ndhwc_fast_kernel,
        "up_ndhwc_gen": _h15_upsample_add_ndhwc_general_kernel,
        "up_ncdhw": _h15_upsample_add_ncdhw_kernel,
        "down_ndhwc": _h15_downsample_add_ndhwc_kernel,
        "down_ncdhw": _h15_downsample_add_ncdhw_kernel,
        "rep_ndhwc": _h15_repeat_add_ndhwc_kernel,
        "rep_ncdhw": _h15_repeat_add_ncdhw_kernel,
        "res_bias_ndhwc": _h15_residual_bias_add_ndhwc_kernel,
        "res_bias_ncdhw": _h15_residual_bias_add_ncdhw_kernel,
    }


# =============================================================================
# Custom Op 1: worldfoundry::h15_rms_norm_silu_pad_3d
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_rms_norm_silu_pad_3d", mutates_args=(), device_types="cuda")
def _h15_rms_norm_silu_pad_3d_op(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c, d_in, h_in, w_in = x.shape
    d_out = d_in + pad_t_l + pad_t_r
    h_out = h_in + pad_h_l + pad_h_r
    w_out = w_in + pad_w_l + pad_w_r

    has_bias = bias is not None
    has_pre_bias = pre_bias is not None
    gamma_1d = gamma.reshape(c).contiguous()
    bias_1d = bias.reshape(c).contiguous() if bias is not None else gamma_1d
    pre_bias_1d = pre_bias.reshape(c).contiguous() if pre_bias is not None else gamma_1d

    if _is_channels_last_3d(x):
        out = torch.empty(
            (b, c, d_out, h_out, w_out),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
        if out.numel() == 0:
            return out
        block_c = _next_power_of_2(c)
        block_w = 32 if block_c <= 64 else (16 if block_c <= 256 else (8 if block_c <= 512 else 4))
        num_warps = 4 if block_w * block_c <= 2048 else 8
        grid = (b * d_out * h_out, _cdiv(w_out, block_w))
        kernels["rms_pad_ndhwc"][grid](
            x,
            gamma_1d,
            bias_1d,
            pre_bias_1d,
            out,
            d_in,
            h_in,
            w_in,
            d_out,
            h_out,
            w_out,
            pad_t_l,
            pad_h_l,
            pad_w_l,
            x.stride(0),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            float(scale),
            channels=c,
            t_mode=t_mode_code,
            s_mode=s_mode_code,
            has_bias=has_bias,
            has_pre_bias=has_pre_bias,
            apply_silu=bool(apply_silu),
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty((b, c, d_out, h_out, w_out), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    block_w = 256 if (w_in >= 256 and c <= 256) else (128 if w_in >= 128 else _next_power_of_2(max(w_in, 16)))
    num_warps = 8 if block_w >= 256 else 4
    num_w_tiles = _cdiv(w_in, block_w) + (1 if (pad_w_l > 0 or pad_w_r > 0) else 0)
    grid = (b * d_out * h_out, num_w_tiles)
    kernels["rms_pad_ncdhw"][grid](
        x_c,
        gamma_1d,
        bias_1d,
        pre_bias_1d,
        out,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        x_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        float(scale),
        channels=c,
        t_mode=t_mode_code,
        s_mode=s_mode_code,
        has_bias=has_bias,
        has_pre_bias=has_pre_bias,
        apply_silu=bool(apply_silu),
        block_w=block_w,
        num_warps=num_warps,
    )
    return out


@_h15_rms_norm_silu_pad_3d_op.register_fake
def _h15_rms_norm_silu_pad_3d_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pre_bias: torch.Tensor | None,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    scale: float,
    apply_silu: bool,
) -> torch.Tensor:
    del gamma, bias, pre_bias, s_mode_code, t_mode_code, scale, apply_silu
    b, c, d_in, h_in, w_in = x.shape
    shape = (b, c, d_in + pad_t_l + pad_t_r, h_in + pad_h_l + pad_h_r, w_in + pad_w_l + pad_w_r)
    if _is_channels_last_3d(x):
        return torch.empty(shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    return x.new_empty(shape)


def can_use_triton_h15_rms_norm_silu_pad_3d(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
) -> bool:
    """Return True when the super-fused H15 ``RMS_norm + SiLU + 3D pad`` kernel is eligible."""
    t_mode = spatial_mode if temporal_mode is None else temporal_mode
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    if x.ndim != 5 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if spatial_mode not in _MODE_TO_CODE or t_mode not in _MODE_TO_CODE:
        return False
    if len(pad) != 6 or any(p < 0 for p in pad):
        return False
    _, c, d_in, h_in, w_in = x.shape
    if not (0 < c <= 1024) or d_in <= 0 or h_in <= 0 or w_in <= 0:
        return False
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    if spatial_mode == "reflect" and (pw_l >= w_in or pw_r >= w_in or ph_l >= h_in or ph_r >= h_in):
        return False
    return not (t_mode == "reflect" and (pt_l >= d_in or pt_r >= d_in))


def triton_h15_rms_norm_silu_pad_3d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    pad: tuple[int, int, int, int, int, int],
    *,
    scale: float | None = None,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
    apply_silu: bool = True,
    pre_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass ``(pre_bias +) RMS_norm (+ SiLU) + 3D pad`` preserving layout."""
    eff_scale = math.sqrt(float(x.shape[1])) if scale is None else float(scale)
    t_mode = spatial_mode if temporal_mode is None else temporal_mode
    if (
        can_use_triton_h15_rms_norm_silu_pad_3d(x, pad, spatial_mode=spatial_mode, temporal_mode=t_mode)
        and gamma.device == x.device
        and (bias is None or bias.device == x.device)
        and (pre_bias is None or pre_bias.device == x.device)
    ):
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = (int(p) for p in pad)
        return _h15_rms_norm_silu_pad_3d_op(
            x,
            gamma,
            bias,
            pre_bias,
            pw_l,
            pw_r,
            ph_l,
            ph_r,
            pt_l,
            pt_r,
            _MODE_TO_CODE[spatial_mode],
            _MODE_TO_CODE[t_mode],
            eff_scale,
            bool(apply_silu),
        )
    return _rms_norm_silu_pad_3d_torch(
        x,
        gamma,
        bias,
        pad,
        scale=eff_scale,
        spatial_mode=spatial_mode,
        temporal_mode=t_mode,
        apply_silu=apply_silu,
        pre_bias=pre_bias,
    )


# =============================================================================
# Custom Op 2: worldfoundry::h15_pad_3d_align_c (fast CL3D path for cin < 8)
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_pad_3d_align_c", mutates_args=(), device_types="cuda")
def _h15_pad_3d_align_c_op(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    align_c: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_in, d_in, h_in, w_in = x.shape
    d_out = d_in + pad_t_l + pad_t_r
    h_out = h_in + pad_h_l + pad_h_r
    w_out = w_in + pad_w_l + pad_w_r

    out = torch.empty(
        (b, align_c, d_out, h_out, w_out),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )
    if out.numel() == 0:
        return out
    block_w = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
    grid = (b * d_out * h_out, _cdiv(w_out, block_w))
    kernels["pad_align_c"][grid](
        x,
        out,
        c_in,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_t_l,
        pad_h_l,
        pad_w_l,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        out.stride(0),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        t_mode=t_mode_code,
        s_mode=s_mode_code,
        c_align=align_c,
        block_w=block_w,
        num_warps=4,
    )
    return out


@_h15_pad_3d_align_c_op.register_fake
def _h15_pad_3d_align_c_fake(
    x: torch.Tensor,
    pad_w_l: int,
    pad_w_r: int,
    pad_h_l: int,
    pad_h_r: int,
    pad_t_l: int,
    pad_t_r: int,
    s_mode_code: int,
    t_mode_code: int,
    align_c: int,
) -> torch.Tensor:
    del s_mode_code, t_mode_code
    b, _, d_in, h_in, w_in = x.shape
    shape = (b, align_c, d_in + pad_t_l + pad_t_r, h_in + pad_h_l + pad_h_r, w_in + pad_w_l + pad_w_r)
    return torch.empty(shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)


def can_use_triton_h15_pad_3d_align_c(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
    align_c: int = 8,
) -> bool:
    """Return True when fused 3-D pad + channel zero-padding to ``align_c`` is eligible."""
    t_mode = spatial_mode if temporal_mode is None else temporal_mode
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    if x.ndim != 5 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if spatial_mode not in _MODE_TO_CODE or t_mode not in _MODE_TO_CODE:
        return False
    if len(pad) != 6 or any(p < 0 for p in pad):
        return False
    if align_c <= 0 or (align_c & (align_c - 1)) != 0:
        return False
    _, c_in, d_in, h_in, w_in = x.shape
    if not (0 < c_in < align_c) or d_in <= 0 or h_in <= 0 or w_in <= 0:
        return False
    pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = pad
    if spatial_mode == "reflect" and (pw_l >= w_in or pw_r >= w_in or ph_l >= h_in or ph_r >= h_in):
        return False
    return not (t_mode == "reflect" and (pt_l >= d_in or pt_r >= d_in))


def triton_h15_pad_3d_align_c(
    x: torch.Tensor,
    pad: tuple[int, int, int, int, int, int],
    *,
    spatial_mode: str = "replicate",
    temporal_mode: str | None = None,
    align_c: int = 8,
) -> torch.Tensor:
    """Fuse 3-D pad + channel zero-padding to ``align_c`` in ``channels_last_3d``."""
    t_mode = spatial_mode if temporal_mode is None else temporal_mode
    if can_use_triton_h15_pad_3d_align_c(x, pad, spatial_mode=spatial_mode, temporal_mode=t_mode, align_c=align_c):
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = (int(p) for p in pad)
        return _h15_pad_3d_align_c_op(
            x,
            pw_l,
            pw_r,
            ph_l,
            ph_r,
            pt_l,
            pt_r,
            _MODE_TO_CODE[spatial_mode],
            _MODE_TO_CODE[t_mode],
            int(align_c),
        )
    padded = _pad_3d_torch_ref(x, pad, spatial_mode=spatial_mode, temporal_mode=t_mode)
    b, c_in, d_out, h_out, w_out = padded.shape
    out = torch.empty(
        (b, align_c, d_out, h_out, w_out),
        device=padded.device,
        dtype=padded.dtype,
        memory_format=torch.channels_last_3d,
    ).zero_()
    out[:, :c_in] = padded
    return out


# =============================================================================
# Custom Op 3: worldfoundry::h15_upsample_add
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_upsample_add", mutates_args=(), device_types="cuda")
def _h15_upsample_add_op(
    h: torch.Tensor,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    add_temporal_upsample: bool,
    repeats: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_h, t_in, h_in, w_in = h.shape
    c_in = x.shape[1]
    factor = 8 if add_temporal_upsample else 4
    c_out = c_h // factor
    t_out = (2 * t_in - 1) if add_temporal_upsample else t_in
    h_out = 2 * h_in
    w_out = 2 * w_in

    has_h_bias = h_bias is not None
    hb_1d = h_bias.reshape(c_h).contiguous() if h_bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or _is_channels_last_3d(x)
    if use_cl3d:
        h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
        x_cl = x if _is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d)
        out = torch.empty(
            (b, c_out, t_out, h_out, w_out),
            device=h.device,
            dtype=h.dtype,
            memory_format=torch.channels_last_3d,
        )
        if out.numel() == 0:
            return out
        block_c = 256 if c_out >= 256 else (128 if c_out >= 128 else _next_power_of_2(max(c_out, 16)))
        is_pow2_rep = (repeats & (repeats - 1)) == 0
        can_fast = (
            is_pow2_rep
            and block_c >= repeats
            and (add_temporal_upsample or (c_out % repeats == 0))
            and (not add_temporal_upsample or (repeats >= 2))
        )
        if can_fast:
            block_w_in = 8 if block_c >= 256 else 16
            num_warps = 8 if block_w_in * block_c >= 2048 else 4
            grid = (b * t_out * h_out, _cdiv(w_in, block_w_in), _cdiv(c_out, block_c))
            kernels["up_ndhwc_fast"][grid](
                h_cl,
                x_cl,
                hb_1d,
                out,
                w_in,
                t_out,
                h_out,
                c_in,
                c_out,
                h_cl.stride(0),
                h_cl.stride(2),
                h_cl.stride(3),
                h_cl.stride(4),
                x_cl.stride(0),
                x_cl.stride(2),
                x_cl.stride(3),
                x_cl.stride(4),
                out.stride(0),
                out.stride(2),
                out.stride(3),
                out.stride(4),
                add_temporal=bool(add_temporal_upsample),
                has_h_bias=has_h_bias,
                repeats=int(repeats),
                block_w_in=block_w_in,
                block_c=block_c,
                num_warps=num_warps,
            )
            return out

        block_w = 8 if block_c >= 256 else 16
        num_warps = 8 if block_w * block_c >= 2048 else 4
        grid = (b * t_out * h_out, _cdiv(w_out, block_w), _cdiv(c_out, block_c))
        kernels["up_ndhwc_gen"][grid](
            h_cl,
            x_cl,
            hb_1d,
            out,
            t_out,
            h_out,
            w_out,
            c_in,
            c_out,
            h_cl.stride(0),
            h_cl.stride(2),
            h_cl.stride(3),
            h_cl.stride(4),
            x_cl.stride(0),
            x_cl.stride(2),
            x_cl.stride(3),
            x_cl.stride(4),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            add_temporal=bool(add_temporal_upsample),
            has_h_bias=has_h_bias,
            repeats=int(repeats),
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    h_c = h if h.is_contiguous() else h.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty((b, c_out, t_out, h_out, w_out), device=h.device, dtype=h.dtype)
    if out.numel() == 0:
        return out
    block_w_in = 128 if w_in >= 128 else _next_power_of_2(max(w_in, 16))
    rows_per_cta = 8 if h_out >= 8 else _next_power_of_2(max(h_out, 1))
    grid = (b * c_out * t_out, _cdiv(h_out, rows_per_cta), _cdiv(w_in, block_w_in))
    kernels["up_ncdhw"][grid](
        h_c,
        x_c,
        hb_1d,
        out,
        h_out,
        w_in,
        w_out,
        t_out,
        c_in,
        c_out,
        h_c.stride(0),
        h_c.stride(1),
        h_c.stride(2),
        h_c.stride(3),
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        x_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        add_temporal=bool(add_temporal_upsample),
        has_h_bias=has_h_bias,
        repeats=int(repeats),
        rows_per_cta=rows_per_cta,
        block_w_in=block_w_in,
        num_warps=4,
    )
    return out


@_h15_upsample_add_op.register_fake
def _h15_upsample_add_fake(
    h: torch.Tensor,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    add_temporal_upsample: bool,
    repeats: int,
) -> torch.Tensor:
    del h_bias, repeats
    b, c_h, t_in, h_in, w_in = h.shape
    factor = 8 if add_temporal_upsample else 4
    c_out = c_h // factor
    t_out = (2 * t_in - 1) if add_temporal_upsample else t_in
    shape = (b, c_out, t_out, 2 * h_in, 2 * w_in)
    if _is_channels_last_3d(h) or _is_channels_last_3d(x):
        return torch.empty(shape, device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return h.new_empty(shape)


def can_use_triton_h15_upsample_add(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_upsample: bool,
    repeats: int,
    h_bias: torch.Tensor | None = None,
) -> bool:
    """Return True when the fused H15 ``Upsample`` + shortcut residual add kernel is eligible."""
    if not h.is_cuda or not x.is_cuda or h.device != x.device or torch.is_grad_enabled():
        return False
    if h.ndim != 5 or x.ndim != 5 or h.dtype != x.dtype or h.dtype not in _SUPPORTED_DTYPES:
        return False
    if h_bias is not None and (h_bias.device != h.device or h_bias.numel() != h.shape[1]):
        return False
    b, c_h, t_in, h_in, w_in = h.shape
    bx, c_in, tx, hx, wx = x.shape
    if b != bx or t_in != tx or h_in != hx or w_in != wx:
        return False
    if t_in <= 0 or h_in <= 0 or w_in <= 0:
        return False
    factor = 8 if add_temporal_upsample else 4
    if c_h <= 0 or c_h % factor != 0 or repeats <= 0:
        return False
    c_out = c_h // factor
    if add_temporal_upsample:
        if c_in % 8 != 0 or repeats < 2 or repeats % 2 != 0:
            return False
        return (c_in // 8) * repeats == c_out and (c_in // 4) * (repeats // 2) == c_out
    return c_in * repeats == 4 * c_out


def triton_h15_upsample_add(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_upsample: bool = True,
    repeats: int | None = None,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass fused ``Upsample`` 3-D depth-to-space (+ ``h_bias``) + shortcut repeat + add."""
    eff_repeats = (h.shape[1] // x.shape[1]) if repeats is None else int(repeats)
    if can_use_triton_h15_upsample_add(
        h, x, add_temporal_upsample=add_temporal_upsample, repeats=eff_repeats, h_bias=h_bias
    ):
        return _h15_upsample_add_op(h, x, h_bias, bool(add_temporal_upsample), eff_repeats)
    return _upsample_add_torch(h, x, add_temporal_upsample=add_temporal_upsample, repeats=eff_repeats, h_bias=h_bias)


# =============================================================================
# Custom Op 4: worldfoundry::h15_downsample_add
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_downsample_add", mutates_args=(), device_types="cuda")
def _h15_downsample_add_op(
    h: torch.Tensor,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    add_temporal_downsample: bool,
    group_size: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_h, t_in, h_in, w_in = h.shape
    c_in = x.shape[1]
    factor = 8 if add_temporal_downsample else 4
    c_out = c_h * factor
    t_out = ((t_in + 1) // 2) if add_temporal_downsample else t_in
    h_out = h_in // 2
    w_out = w_in // 2

    has_h_bias = h_bias is not None
    hb_1d = h_bias.reshape(c_h).contiguous() if h_bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or _is_channels_last_3d(x)
    if use_cl3d:
        h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
        x_cl = x if _is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d)
        out = torch.empty(
            (b, c_out, t_out, h_out, w_out),
            device=h.device,
            dtype=h.dtype,
            memory_format=torch.channels_last_3d,
        )
        if out.numel() == 0:
            return out
        is_pow2_g = (group_size & (group_size - 1)) == 0
        max_bc = max(16, 512 // max(group_size, 1)) if is_pow2_g else 128
        block_c = min(_next_power_of_2(max(c_h, 16)), max_bc)
        if block_c > 128:
            block_c = 128
        eff_row_elems = block_c * (group_size if is_pow2_g else 1)
        block_w = 16 if eff_row_elems <= 128 else (8 if eff_row_elems <= 256 else 4)
        num_warps = 8 if block_w * eff_row_elems >= 2048 else 4
        c_tiles_per_sp = _cdiv(c_h, block_c)
        grid = (b * t_out * h_out, _cdiv(w_out, block_w), factor * c_tiles_per_sp)
        kernels["down_ndhwc"][grid](
            h_cl,
            x_cl,
            hb_1d,
            out,
            t_out,
            h_out,
            w_out,
            c_in,
            c_h,
            h_cl.stride(0),
            h_cl.stride(2),
            h_cl.stride(3),
            h_cl.stride(4),
            x_cl.stride(0),
            x_cl.stride(2),
            x_cl.stride(3),
            x_cl.stride(4),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            add_temporal=bool(add_temporal_downsample),
            has_h_bias=has_h_bias,
            group_size=int(group_size),
            is_pow2_g=is_pow2_g,
            block_w=block_w,
            block_c=block_c,
            c_tiles_per_sp=c_tiles_per_sp,
            num_warps=num_warps,
        )
        return out

    h_c = h if h.is_contiguous() else h.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty((b, c_out, t_out, h_out, w_out), device=h.device, dtype=h.dtype)
    if out.numel() == 0:
        return out
    block_w_out = 128 if w_out >= 128 else _next_power_of_2(max(w_out, 16))
    rows_per_cta = 8 if h_out >= 8 else _next_power_of_2(max(h_out, 1))
    num_pairs = factor // 2
    grid = (b * num_pairs * c_h * t_out, _cdiv(h_out, rows_per_cta), _cdiv(w_out, block_w_out))
    kernels["down_ncdhw"][grid](
        h_c,
        x_c,
        hb_1d,
        out,
        h_out,
        w_in,
        w_out,
        t_out,
        c_in,
        c_h,
        h_c.stride(0),
        h_c.stride(1),
        h_c.stride(2),
        h_c.stride(3),
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        x_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        add_temporal=bool(add_temporal_downsample),
        has_h_bias=has_h_bias,
        group_size=int(group_size),
        num_pairs=num_pairs,
        rows_per_cta=rows_per_cta,
        block_w_out=block_w_out,
        num_warps=4,
    )
    return out


@_h15_downsample_add_op.register_fake
def _h15_downsample_add_fake(
    h: torch.Tensor,
    x: torch.Tensor,
    h_bias: torch.Tensor | None,
    add_temporal_downsample: bool,
    group_size: int,
) -> torch.Tensor:
    del h_bias, group_size
    b, c_h, t_in, h_in, w_in = h.shape
    factor = 8 if add_temporal_downsample else 4
    t_out = ((t_in + 1) // 2) if add_temporal_downsample else t_in
    shape = (b, c_h * factor, t_out, h_in // 2, w_in // 2)
    if _is_channels_last_3d(h) or _is_channels_last_3d(x):
        return torch.empty(shape, device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return h.new_empty(shape)


def can_use_triton_h15_downsample_add(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_downsample: bool,
    group_size: int,
    h_bias: torch.Tensor | None = None,
) -> bool:
    """Return True when the fused H15 ``Downsample`` + group-mean residual add kernel is eligible."""
    if not h.is_cuda or not x.is_cuda or h.device != x.device or torch.is_grad_enabled():
        return False
    if h.ndim != 5 or x.ndim != 5 or h.dtype != x.dtype or h.dtype not in _SUPPORTED_DTYPES:
        return False
    if h_bias is not None and (h_bias.device != h.device or h_bias.numel() != h.shape[1]):
        return False
    b, c_h, t_in, h_in, w_in = h.shape
    bx, c_in, tx, hx, wx = x.shape
    if b != bx or t_in != tx or h_in != hx or w_in != wx:
        return False
    if t_in <= 0 or h_in <= 0 or w_in <= 0 or h_in % 2 != 0 or w_in % 2 != 0:
        return False
    if c_h <= 0 or group_size <= 0 or group_size > 32:
        return False
    if add_temporal_downsample and ((t_in - 1) % 2 != 0 or group_size < 2 or group_size % 2 != 0):
        return False
    return c_h * group_size == c_in


def triton_h15_downsample_add(
    h: torch.Tensor,
    x: torch.Tensor,
    *,
    add_temporal_downsample: bool = True,
    group_size: int | None = None,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass fused ``Downsample`` 3-D space-to-depth (+ ``h_bias``) + shortcut group mean + add."""
    eff_group = (x.shape[1] // h.shape[1]) if group_size is None else int(group_size)
    if can_use_triton_h15_downsample_add(
        h, x, add_temporal_downsample=add_temporal_downsample, group_size=eff_group, h_bias=h_bias
    ):
        return _h15_downsample_add_op(h, x, h_bias, bool(add_temporal_downsample), eff_group)
    return _downsample_add_torch(
        h, x, add_temporal_downsample=add_temporal_downsample, group_size=eff_group, h_bias=h_bias
    )


# =============================================================================
# Custom Op 5: worldfoundry::h15_repeat_add (Decoder conv_in shortcut)
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_repeat_add", mutates_args=(), device_types="cuda")
def _h15_repeat_add_op(
    h: torch.Tensor,
    z: torch.Tensor,
    h_bias: torch.Tensor | None,
    repeats: int,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c_out, t, h_dim, w_dim = h.shape
    c_in = z.shape[1]
    has_h_bias = h_bias is not None
    hb_1d = h_bias.reshape(c_out).contiguous() if h_bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or _is_channels_last_3d(z)
    if use_cl3d:
        h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
        z_cl = z if _is_channels_last_3d(z) else z.contiguous(memory_format=torch.channels_last_3d)
        out = torch.empty(
            (b, c_out, t, h_dim, w_dim),
            device=h.device,
            dtype=h.dtype,
            memory_format=torch.channels_last_3d,
        )
        total_voxels = b * t * h_dim * w_dim
        if total_voxels == 0 or c_out == 0:
            return out
        block_c = 256 if c_out >= 256 else _next_power_of_2(max(c_out, 16))
        block_v = 16 if block_c <= 256 else 8
        num_warps = 8 if block_v * block_c >= 2048 else 4
        kernels["rep_ndhwc"][(_cdiv(total_voxels, block_v), _cdiv(c_out, block_c))](
            h_cl,
            z_cl,
            hb_1d,
            out,
            total_voxels,
            c_out,
            c_in,
            has_h_bias=has_h_bias,
            repeats=int(repeats),
            block_v=block_v,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    h_c = h if h.is_contiguous() else h.contiguous()
    z_c = z if z.is_contiguous() else z.contiguous()
    out = torch.empty_like(h_c)
    spatial_size = t * h_dim * w_dim
    if spatial_size == 0 or b == 0 or c_out == 0:
        return out
    block_s = 256 if spatial_size >= 256 else _next_power_of_2(max(spatial_size, 16))
    kernels["rep_ncdhw"][(b * c_out, _cdiv(spatial_size, block_s))](
        h_c,
        z_c,
        hb_1d,
        out,
        spatial_size,
        c_out,
        c_in,
        has_h_bias=has_h_bias,
        repeats=int(repeats),
        block_s=block_s,
        num_warps=4,
    )
    return out


@_h15_repeat_add_op.register_fake
def _h15_repeat_add_fake(
    h: torch.Tensor,
    z: torch.Tensor,
    h_bias: torch.Tensor | None,
    repeats: int,
) -> torch.Tensor:
    del h_bias, repeats
    if _is_channels_last_3d(h) or _is_channels_last_3d(z):
        return torch.empty(h.shape, device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return torch.empty_like(h)


def can_use_triton_h15_repeat_add(
    h: torch.Tensor,
    z: torch.Tensor,
    repeats: int,
    *,
    h_bias: torch.Tensor | None = None,
) -> bool:
    """Return True when fused ``h (+ h_bias) + z.repeat_interleave(repeats, dim=1)`` is eligible."""
    if not h.is_cuda or not z.is_cuda or h.device != z.device or torch.is_grad_enabled():
        return False
    if h.ndim != 5 or z.ndim != 5 or h.dtype != z.dtype or h.dtype not in _SUPPORTED_DTYPES:
        return False
    if h_bias is not None and (h_bias.device != h.device or h_bias.numel() != h.shape[1]):
        return False
    if repeats <= 0 or z.shape[1] * repeats != h.shape[1]:
        return False
    return h.shape[0] == z.shape[0] and h.shape[2:] == z.shape[2:]


def triton_h15_repeat_add(
    h: torch.Tensor,
    z: torch.Tensor,
    repeats: int,
    *,
    h_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-pass ``h (+ h_bias) + z.repeat_interleave(repeats, dim=1)`` preserving layout."""
    if can_use_triton_h15_repeat_add(h, z, repeats, h_bias=h_bias):
        return _h15_repeat_add_op(h, z, h_bias, int(repeats))
    return _repeat_add_torch(h, z, repeats, h_bias=h_bias)


# =============================================================================
# Custom Op 6: worldfoundry::h15_residual_bias_add (ResnetBlock & Conv3d bias)
# =============================================================================


@torch.library.custom_op("worldfoundry::h15_residual_bias_add", mutates_args=(), device_types="cuda")
def _h15_residual_bias_add_op(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    kernels = _load_triton_kernels()

    b, c, t, h_dim, w_dim = h.shape
    has_x = x is not None
    has_bias = bias is not None
    b_1d = bias.reshape(c).contiguous() if bias is not None else h

    use_cl3d = _is_channels_last_3d(h) or (x is not None and _is_channels_last_3d(x))
    if use_cl3d:
        h_cl = h if _is_channels_last_3d(h) else h.contiguous(memory_format=torch.channels_last_3d)
        x_cl = (
            (x if _is_channels_last_3d(x) else x.contiguous(memory_format=torch.channels_last_3d))
            if x is not None
            else h_cl
        )
        out = torch.empty(
            (b, c, t, h_dim, w_dim),
            device=h.device,
            dtype=h.dtype,
            memory_format=torch.channels_last_3d,
        )
        total_voxels = b * t * h_dim * w_dim
        if total_voxels == 0 or c == 0:
            return out
        block_c = 256 if c >= 256 else (128 if c >= 128 else _next_power_of_2(max(c, 16)))
        block_v = 16 if block_c <= 128 else 8
        num_warps = 8 if block_v * block_c >= 2048 else 4
        kernels["res_bias_ndhwc"][(_cdiv(total_voxels, block_v), _cdiv(c, block_c))](
            x_cl,
            h_cl,
            b_1d,
            out,
            total_voxels,
            c,
            has_x=has_x,
            has_bias=has_bias,
            block_v=block_v,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    h_c = h if h.is_contiguous() else h.contiguous()
    x_c = (x if x.is_contiguous() else x.contiguous()) if x is not None else h_c
    out = torch.empty_like(h_c)
    spatial_size = t * h_dim * w_dim
    if spatial_size == 0 or b == 0 or c == 0:
        return out
    block_s = 1024 if spatial_size >= 1024 else _next_power_of_2(max(spatial_size, 32))
    num_warps = 8 if block_s >= 1024 else 4
    kernels["res_bias_ncdhw"][(b * c, _cdiv(spatial_size, block_s))](
        x_c,
        h_c,
        b_1d,
        out,
        spatial_size,
        c,
        has_x=has_x,
        has_bias=has_bias,
        block_s=block_s,
        num_warps=num_warps,
    )
    return out


@_h15_residual_bias_add_op.register_fake
def _h15_residual_bias_add_fake(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    del bias
    if _is_channels_last_3d(h) or (x is not None and _is_channels_last_3d(x)):
        return torch.empty(h.shape, device=h.device, dtype=h.dtype, memory_format=torch.channels_last_3d)
    return torch.empty_like(h)


def can_use_triton_h15_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Return True when fused ``(x +) (h + bias)`` is eligible."""
    if not h.is_cuda or torch.is_grad_enabled():
        return False
    if h.ndim != 5 or h.dtype not in _SUPPORTED_DTYPES:
        return False
    if x is not None and (not x.is_cuda or x.device != h.device or x.shape != h.shape or x.dtype != h.dtype):
        return False
    return not (bias is not None and (bias.device != h.device or bias.numel() != h.shape[1]))


def triton_h15_residual_bias_add(
    x: torch.Tensor | None,
    h: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Single-pass coalesced ``x + (h + bias)`` (or ``h + bias`` if ``x is None``)."""
    if can_use_triton_h15_residual_bias_add(x, h, bias):
        return _h15_residual_bias_add_op(x, h, bias)
    return _residual_bias_add_torch(x, h, bias)


__all__ = [
    "_is_channels_last_3d",
    "can_use_triton_h15_downsample_add",
    "can_use_triton_h15_pad_3d_align_c",
    "can_use_triton_h15_repeat_add",
    "can_use_triton_h15_residual_bias_add",
    "can_use_triton_h15_rms_norm_silu_pad_3d",
    "can_use_triton_h15_upsample_add",
    "triton_h15_downsample_add",
    "triton_h15_pad_3d_align_c",
    "triton_h15_repeat_add",
    "triton_h15_residual_bias_add",
    "triton_h15_rms_norm_silu_pad_3d",
    "triton_h15_upsample_add",
]
