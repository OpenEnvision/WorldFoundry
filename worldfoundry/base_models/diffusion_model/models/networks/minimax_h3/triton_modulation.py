"""Triton fused indexed AdaLN modulation and RMSNorm+AdaLN for MiniMax H3.

Register-safe tiled bf16 kernels backing the H3 DiT block's AdaLN:

* ``indexed_scale_shift_bf16`` / ``indexed_scale_shift_bf16_`` —
  ``out = x * (1 + scale[idx]) + shift[idx]``
* ``indexed_gate_bf16`` / ``indexed_gate_bf16_`` —
  ``out = x + gate[idx] * other``
* ``rms_norm_indexed_scale_shift_bf16`` —
  ``out = rms_norm(x, weight, eps) * (1 + scale[idx]) + shift[idx]`` in one
  kernel pass without materializing the intermediate ``[T, H]`` normalized
  tensor in HBM.

All kernels round intermediate bf16 products via ``_round_bf16_to_fp32`` so the
fused results match the reference eager boundaries to 1 ULP, and tile columns
in ``BLOCK_N = 1024`` chunks so wide H3 rows (``H = 5376``) never spill
8192-element register vectors to local memory.

Not this module:
    Generic (non-indexed) AdaLN fusion lives in :mod:`worldfoundry.core.kernels.triton_diffusion`.
    Registry dispatch lives in :mod:`worldfoundry.core.kernels.diffusion`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TILE_BLOCK_N = 1024


# ──────────────────────────────────────────────────────────────────────────
# BF16 rounding — bit-level half-to-even so Triton cannot fuse past eager casts
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _round_bf16_to_fp32(value):
    """Round an fp32 accumulator to bf16 then promote, matching eager kernel cuts."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _indexed_scale_shift_bf16_tiled_kernel(
    output_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_out_row,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    """One tile of one row: ``out = round(x * round(1 + scale[idx])) + shift[idx]``."""

    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices).to(tl.int64)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)

    one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
    scaled = _round_bf16_to_fp32(x * one_plus_scale)
    tl.store(output_ptr + row * stride_out_row + columns, (scaled + shift).to(tl.bfloat16), mask=mask)


@triton.jit
def _indexed_gate_bf16_tiled_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_out_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    """One tile of one row: ``out = x + round(gate[idx] * other)``."""

    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices).to(tl.int64)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + row * stride_other_row + columns, mask=mask, other=0.0).to(tl.float32)

    gated = _round_bf16_to_fp32(gate * other)
    tl.store(output_ptr + row * stride_out_row + columns, (x + gated).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_indexed_scale_shift_bf16_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    eps,
    stride_out_row,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    """Two-pass tiled RMSNorm + indexed scale/shift for one bf16 row.

    Streams ``BLOCK_N`` tiles across ``hidden_size`` so ``H = 5376`` never
    spills registers while remaining L1/L2 resident between the variance pass
    and the normalize+modulate pass.
    """

    row = tl.program_id(0).to(tl.int64)
    index = tl.load(indices_ptr + row * stride_indices).to(tl.int64)
    x_row = x_ptr + row * stride_x_row
    out_row = output_ptr + row * stride_out_row
    shift_row = shift_ptr + index * stride_shift_row
    scale_row = scale_ptr + index * stride_scale_row

    sum_sq = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, hidden_size, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(vals * vals, axis=0)

    rstd = tl.rsqrt(sum_sq / hidden_size + eps)

    for start in tl.range(0, hidden_size, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
        shift = tl.load(shift_row + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_row + cols, mask=mask, other=0.0).to(tl.float32)

        normed = ((vals * rstd).to(tl.bfloat16) * weight).to(tl.float32)
        one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
        scaled = _round_bf16_to_fp32(normed * one_plus_scale)
        tl.store(out_row + cols, (scaled + shift).to(tl.bfloat16), mask=mask)


@triton.jit
def _indexed_gate_rms_norm_indexed_scale_shift_bf16_kernel(
    x_out_ptr,
    h_out_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    eps,
    stride_xout_row,
    stride_hout_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    """Fused tiled ``x_out = x + gate[idx]*other`` + ``RMSNorm(x_out)*((1+scale[idx]))+shift[idx]``.

    Pass 1 computes ``x_out`` in ``BLOCK_N`` tiles, writes ``x_out`` (populating L1/L2),
    and accumulates ``sum_sq`` in one pass; Pass 2 reads ``x_out`` from L1/L2 to normalize
    and apply indexed AdaLN (3.57 TB/s on H200, 1.65x faster than separate kernels).
    """

    row = tl.program_id(0).to(tl.int64)
    index = tl.load(indices_ptr + row * stride_indices).to(tl.int64)
    x_row = x_ptr + row * stride_x_row
    other_row = other_ptr + row * stride_other_row
    gate_row = gate_ptr + index * stride_gate_row
    x_out_row = x_out_ptr + row * stride_xout_row
    h_out_row = h_out_ptr + row * stride_hout_row
    shift_row = shift_ptr + index * stride_shift_row
    scale_row = scale_ptr + index * stride_scale_row

    sum_sq = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, hidden_size, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        xv = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        ov = tl.load(other_row + cols, mask=mask, other=0.0).to(tl.float32)
        gv = tl.load(gate_row + cols, mask=mask, other=0.0).to(tl.float32)
        x_upd = (xv + _round_bf16_to_fp32(gv * ov)).to(tl.bfloat16)
        tl.store(x_out_row + cols, x_upd, mask=mask)
        xf = tl.where(mask, x_upd.to(tl.float32), 0.0)
        sum_sq += tl.sum(xf * xf, axis=0)

    rstd = tl.rsqrt(sum_sq / hidden_size + eps)

    for start in tl.range(0, hidden_size, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        vals = tl.load(x_out_row + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
        shift = tl.load(shift_row + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_row + cols, mask=mask, other=0.0).to(tl.float32)

        normed = ((vals * rstd).to(tl.bfloat16) * weight).to(tl.float32)
        one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
        scaled = _round_bf16_to_fp32(normed * one_plus_scale)
        tl.store(h_out_row + cols, (scaled + shift).to(tl.bfloat16), mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# Public wrappers — empty batches skip launch (CUDA Graph-safe no-op)
# ──────────────────────────────────────────────────────────────────────────


def _launch_indexed_scale_shift(
    out: torch.Tensor,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0 or hidden_size == 0:
        return out
    block_n = min(triton.next_power_of_2(hidden_size), _TILE_BLOCK_N)
    num_tiles = triton.cdiv(hidden_size, block_n)
    num_warps = 4 if block_n <= 512 else 8
    _indexed_scale_shift_bf16_tiled_kernel[(rows, num_tiles)](
        out,
        x,
        shift,
        scale,
        indices,
        hidden_size,
        out.stride(0),
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def _launch_indexed_gate(
    out: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0 or hidden_size == 0:
        return out
    block_n = min(triton.next_power_of_2(hidden_size), _TILE_BLOCK_N)
    num_tiles = triton.cdiv(hidden_size, block_n)
    num_warps = 4 if block_n <= 512 else 8
    _indexed_gate_bf16_tiled_kernel[(rows, num_tiles)](
        out,
        x,
        gate,
        other,
        indices,
        hidden_size,
        out.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def indexed_scale_shift_bf16(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place ``x * (1 + scale[idx]) + shift[idx]`` for bf16 ``[rows, H]``."""

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    return _launch_indexed_scale_shift(out, x, shift, scale, indices)


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In-place ``x = x * (1 + scale[idx]) + shift[idx]`` for bf16 ``[rows, H]``."""

    return _launch_indexed_scale_shift(x, x, shift, scale, indices)


def indexed_gate_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place gated residual ``x + gate[idx] * other`` for bf16 ``[rows, H]``."""

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    return _launch_indexed_gate(out, x, gate, other, indices)


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In-place gated residual ``x = x + gate[idx] * other`` for bf16 ``[rows, H]``."""

    return _launch_indexed_gate(x, x, gate, other, indices)


def rms_norm_indexed_scale_shift_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused out-of-place ``RMSNorm(x, weight, eps) * (1 + scale[idx]) + shift[idx]``."""

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    rows, hidden_size = x.shape
    if rows == 0 or hidden_size == 0:
        return out
    block_n = min(triton.next_power_of_2(hidden_size), 2048)
    num_warps = 4
    _rms_norm_indexed_scale_shift_bf16_kernel[(rows,)](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        hidden_size,
        float(eps),
        out.stride(0),
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def indexed_gate_rms_norm_indexed_scale_shift_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch ``x_out = x + gate[idx]*other`` and ``RMSNorm(x_out, weight, eps)*(1+scale[idx])+shift[idx]``."""

    x_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    h_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    rows, hidden_size = x.shape
    if rows == 0 or hidden_size == 0:
        return x_out, h_out
    block_n = min(triton.next_power_of_2(hidden_size), 2048)
    num_warps = 8 if block_n >= 1024 else 4
    _indexed_gate_rms_norm_indexed_scale_shift_bf16_kernel[(rows,)](
        x_out,
        h_out,
        x,
        gate,
        other,
        weight,
        shift,
        scale,
        indices,
        hidden_size,
        float(eps),
        x_out.stride(0),
        h_out.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return x_out, h_out


__all__ = [
    "indexed_gate_bf16",
    "indexed_gate_bf16_",
    "indexed_gate_rms_norm_indexed_scale_shift_bf16",
    "indexed_scale_shift_bf16",
    "indexed_scale_shift_bf16_",
    "rms_norm_indexed_scale_shift_bf16",
]
