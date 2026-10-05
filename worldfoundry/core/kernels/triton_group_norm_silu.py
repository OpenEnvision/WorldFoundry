# SPDX-License-Identifier: Apache-2.0
#
# Standalone in-tree Triton implementation with a registry-friendly launcher.

"""In-tree fused GroupNorm + SiLU Triton kernel.

Entry point is selected by :func:`worldfoundry.core.kernels.diffusion.group_norm_silu`.
Do not import this file from DiT blocks — it skips capability gates.

Not this module:
    Eligibility, size thresholds, and PyTorch fallback live in :mod:`.diffusion`.

Public surface:

- :func:`group_norm_silu` — one-pass for small groups, chunked reduce for large.
"""

from __future__ import annotations

import math

import torch

from worldfoundry.core.execution.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="group-norm-triton")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_LARGE_GROUP_THRESHOLD = 1 << 18
_BLOCK_SIZE = 4096
_BLOCKS_PER_PROGRAM = 2
_CHUNK_SIZE = _BLOCK_SIZE * _BLOCKS_PER_PROGRAM


# ──────────────────────────────────────────────────────────────────────────
# One-pass — group fits in a few 4096-wide tiles; mean/var stay in registers
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _group_norm_silu_contiguous_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels,
    spatial_size,
    channels_per_group,
    group_size,
    eps,
    block_size: tl.constexpr,
):
    """Exact GroupNorm + affine + SiLU when the whole group is scanned twice.

    Grid is ``(num_groups, batch)``. Variance uses the two-pass
    ``E[x^2] - mean^2`` form so a third reduction is not required.
    """

    group_id = tl.program_id(0).to(tl.int64)
    batch_id = tl.program_id(1).to(tl.int64)
    group_base = batch_id * channels * spatial_size + group_id * group_size
    offsets = tl.arange(0, block_size)

    sum_value = tl.zeros((), dtype=tl.float32)
    sum_square = tl.zeros((), dtype=tl.float32)
    for start in range(0, group_size, block_size):
        indices = start + offsets
        mask = indices < group_size
        value = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        sum_value += tl.sum(value, axis=0)
        sum_square += tl.sum(value * value, axis=0)

    inverse_group = 1.0 / group_size
    mean = sum_value * inverse_group
    variance = sum_square * inverse_group - mean * mean
    reciprocal_std = tl.rsqrt(variance + eps)
    weight_group_offset = group_id * channels_per_group

    for start in range(0, group_size, block_size):
        indices = start + offsets
        mask = indices < group_size
        value = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        channel_offsets = weight_group_offset + indices // spatial_size
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        output = (value - mean) * reciprocal_std
        output = output * weight + bias
        output = output * tl.sigmoid(output)
        tl.store(output_ptr + group_base + indices, output, mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# Chunked — groups above 2^18 elements cannot keep the reduction in one CTA
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _group_norm_stats_kernel(
    input_ptr,
    partial_sum_ptr,
    partial_square_ptr,
    channels,
    spatial_size,
    num_groups,
    channels_per_group,
    group_size,
    chunks_per_row,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Partial sum / sum-of-squares for one ``(batch, group, chunk)`` tile.

    Grid is ``(batch * num_groups, chunks_per_row)``. Each program covers
    ``blocks_per_program`` blocks of ``block_size`` (4096 × 2).
    """

    row = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1).to(tl.int64)
    batch_id = row // num_groups
    group_id = row - batch_id * num_groups
    chunk_start = chunk_id * block_size * blocks_per_program
    group_base = batch_id * channels * spatial_size + group_id * group_size

    sum_value = tl.zeros((), dtype=tl.float32)
    sum_square = tl.zeros((), dtype=tl.float32)
    offsets = tl.arange(0, block_size)
    for block_id in range(blocks_per_program):
        indices = chunk_start + block_id * block_size + offsets
        mask = indices < group_size
        value = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        sum_value += tl.sum(value, axis=0)
        sum_square += tl.sum(value * value, axis=0)

    partial_index = row * chunks_per_row + chunk_id
    tl.store(partial_sum_ptr + partial_index, sum_value)
    tl.store(partial_square_ptr + partial_index, sum_square)


@triton.jit
def _group_norm_finalize_stats_kernel(
    partial_sum_ptr,
    partial_square_ptr,
    stats_ptr,
    chunks_per_row,
    group_size,
    eps,
    block_size: tl.constexpr,
):
    """Reduce partial stats to ``(mean, rsqrt(var + eps))`` stored as a pair."""

    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    sum_value = tl.zeros((), dtype=tl.float32)
    sum_square = tl.zeros((), dtype=tl.float32)
    base = row * chunks_per_row
    for start in range(0, chunks_per_row, block_size):
        indices = start + offsets
        mask = indices < chunks_per_row
        sum_value += tl.sum(tl.load(partial_sum_ptr + base + indices, mask=mask, other=0.0), axis=0)
        sum_square += tl.sum(
            tl.load(partial_square_ptr + base + indices, mask=mask, other=0.0), axis=0
        )

    inverse_group = 1.0 / group_size
    mean = sum_value * inverse_group
    variance = sum_square * inverse_group - mean * mean
    reciprocal_std = tl.rsqrt(variance + eps)
    tl.store(stats_ptr + row * 2, mean)
    tl.store(stats_ptr + row * 2 + 1, reciprocal_std)


@triton.jit
def _group_norm_apply_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stats_ptr,
    channels,
    spatial_size,
    num_groups,
    channels_per_group,
    group_size,
    chunks_per_row,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Apply finalized stats with per-element affine (channel may change inside a chunk)."""

    row = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1).to(tl.int64)
    batch_id = row // num_groups
    group_id = row - batch_id * num_groups
    chunk_start = chunk_id * block_size * blocks_per_program
    group_base = batch_id * channels * spatial_size + group_id * group_size
    weight_group_offset = group_id * channels_per_group
    mean = tl.load(stats_ptr + row * 2)
    reciprocal_std = tl.load(stats_ptr + row * 2 + 1)
    offsets = tl.arange(0, block_size)

    for block_id in range(blocks_per_program):
        indices = chunk_start + block_id * block_size + offsets
        mask = indices < group_size
        value = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        channel_offsets = weight_group_offset + indices // spatial_size
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        output = (value - mean) * reciprocal_std
        output = output * weight + bias
        output = output * tl.sigmoid(output)
        tl.store(output_ptr + group_base + indices, output, mask=mask)


@triton.jit
def _group_norm_apply_scalar_affine_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stats_ptr,
    channels,
    spatial_size,
    num_groups,
    channels_per_group,
    group_size,
    chunks_per_row,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Apply finalized stats when each chunk lies inside one channel (scalar affine)."""

    row = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1).to(tl.int64)
    batch_id = row // num_groups
    group_id = row - batch_id * num_groups
    chunk_start = chunk_id * block_size * blocks_per_program
    group_base = batch_id * channels * spatial_size + group_id * group_size
    channel_id = chunk_start // spatial_size
    affine_offset = group_id * channels_per_group + channel_id
    weight = tl.load(weight_ptr + affine_offset).to(tl.float32)
    bias = tl.load(bias_ptr + affine_offset).to(tl.float32)
    mean = tl.load(stats_ptr + row * 2)
    reciprocal_std = tl.load(stats_ptr + row * 2 + 1)
    offsets = tl.arange(0, block_size)

    for block_id in range(blocks_per_program):
        indices = chunk_start + block_id * block_size + offsets
        mask = indices < group_size
        value = tl.load(input_ptr + group_base + indices, mask=mask, other=0.0).to(tl.float32)
        output = (value - mean) * reciprocal_std
        output = output * weight + bias
        output = output * tl.sigmoid(output)
        tl.store(output_ptr + group_base + indices, output, mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# Launchers — bind CUDA device; block_size is min(4096, next_pow2(group))
# ──────────────────────────────────────────────────────────────────────────


def _launch_one_pass(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Launch the contiguous one-pass kernel; *input* must already be NCHW-contiguous."""

    batch_size, channels = input.shape[:2]
    spatial_size = math.prod(input.shape[2:]) if input.ndim > 2 else 1
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial_size
    input_flat = input.reshape(batch_size, channels, spatial_size, 1)
    output_flat = torch.empty_like(input_flat)
    block_size = min(4096, triton.next_power_of_2(max(1, min(group_size, 4096))))
    _group_norm_silu_contiguous_kernel[(num_groups, batch_size)](
        input_flat,
        weight,
        bias,
        output_flat,
        channels,
        spatial_size,
        channels_per_group,
        group_size,
        eps,
        block_size=block_size,
    )
    return output_flat.reshape_as(input)


def _launch_chunked(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Three-kernel path: partial stats, finalize, then apply (scalar affine when safe)."""

    batch_size, channels = input.shape[:2]
    spatial_size = math.prod(input.shape[2:]) if input.ndim > 2 else 1
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial_size
    rows = batch_size * num_groups
    chunks_per_row = triton.cdiv(group_size, _CHUNK_SIZE)
    input_flat = input.reshape(-1)
    output = torch.empty_like(input)
    output_flat = output.reshape(-1)
    partial_sum = torch.empty((rows, chunks_per_row), device=input.device, dtype=torch.float32)
    partial_square = torch.empty_like(partial_sum)
    stats = torch.empty((rows, 2), device=input.device, dtype=torch.float32)

    _group_norm_stats_kernel[(rows, chunks_per_row)](
        input_flat,
        partial_sum,
        partial_square,
        channels,
        spatial_size,
        num_groups,
        channels_per_group,
        group_size,
        chunks_per_row,
        block_size=_BLOCK_SIZE,
        blocks_per_program=_BLOCKS_PER_PROGRAM,
        num_warps=8,
        num_stages=3,
    )
    reduce_block = min(1024, triton.next_power_of_2(max(1, chunks_per_row)))
    _group_norm_finalize_stats_kernel[(rows,)](
        partial_sum,
        partial_square,
        stats,
        chunks_per_row,
        group_size,
        eps,
        block_size=reduce_block,
        num_warps=4,
        num_stages=2,
    )

    apply_kernel = (
        _group_norm_apply_scalar_affine_kernel
        if spatial_size % _CHUNK_SIZE == 0 and chunks_per_row >= 64
        else _group_norm_apply_kernel
    )
    apply_kernel[(rows, chunks_per_row)](
        input_flat,
        weight,
        bias,
        output_flat,
        stats,
        channels,
        spatial_size,
        num_groups,
        channels_per_group,
        group_size,
        chunks_per_row,
        block_size=_BLOCK_SIZE,
        blocks_per_program=_BLOCKS_PER_PROGRAM,
        num_warps=4 if apply_kernel is _group_norm_apply_scalar_affine_kernel else 8,
        num_stages=3,
    )
    return output


def group_norm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Launch the in-tree exact GroupNorm + SiLU Triton kernel."""

    spatial_size = math.prod(input.shape[2:]) if input.ndim > 2 else 1
    group_size = (input.shape[1] // int(num_groups)) * spatial_size
    with torch.cuda.device(input.device):
        if group_size >= _LARGE_GROUP_THRESHOLD:
            return _launch_chunked(input, weight, bias, int(num_groups), float(eps))
        return _launch_one_pass(input, weight, bias, int(num_groups), float(eps))


__all__ = ["group_norm_silu"]
