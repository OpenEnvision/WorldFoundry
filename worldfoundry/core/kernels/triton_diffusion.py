"""Portable Triton kernels used by diffusion and world-model blocks.

These kernels are authored in-tree and target regular CUDA tensors. They
do not require an external attention or serving checkout.

Not this module:
    Capability gates, size thresholds, and PyTorch fallbacks live in
    :mod:`.diffusion`. GroupNorm+SiLU lives in :mod:`.triton_group_norm_silu`.
    MiniMax H3 indexed AdaLN and Q/K RoPE kernels live with the MiniMax H3
    network in ``base_models/diffusion_model/models/networks/minimax_h3``.

Public surface:

- :func:`silu_mul` / :func:`silu_and_mul` — pointwise gated activations.
- :func:`residual_gate` / :func:`scale_shift` — AdaLN residual / modulation.
- :func:`layer_norm_scale_shift` / :func:`rms_norm_scale_shift` — fused norm+AdaLN.
- :func:`qk_rmsnorm_rope` / :func:`hidden_qk_rmsnorm_rope_3d` — Q/K RoPE fusions.
"""

from __future__ import annotations

import torch

from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="diffusion-triton")

import triton  # noqa: E402 - Follows the Triton compile-cache configuration.
import triton.language as tl  # noqa: E402 - Follows the Triton compile-cache configuration.

# ──────────────────────────────────────────────────────────────────────────
# Pointwise SiLU — BLOCK=1024; fp32 sigmoid then store in the output dtype
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    value_ptr,
    out_ptr,
    elements,
    block: tl.constexpr,
):
    """Elementwise ``silu(gate) * value``; *block* tiles a flat contiguous buffer."""

    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block).to(tl.int64)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = gate * tl.sigmoid(gate) * value
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _packed_silu_mul_kernel(
    input_ptr,
    out_ptr,
    output_elements,
    half_features: tl.constexpr,
    block: tl.constexpr,
):
    """Packed last-dim split: first half is gate, second half is value (strict int64)."""

    output_offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block).to(tl.int64)
    mask = output_offsets < output_elements
    row = output_offsets // half_features
    feature = output_offsets - row * half_features
    input_base = row * (2 * half_features) + feature
    gate = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(input_ptr + input_base + half_features, mask=mask, other=0.0).to(tl.float32)
    output = gate * tl.sigmoid(gate) * value
    tl.store(out_ptr + output_offsets, output, mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# Broadcast AdaLN — pad rank to 5 so one kernel covers 2D–5D activations
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _broadcast_row_offset(row, d0, d1, d2, d3, s0, s1, s2, s3):
    """Decode a flattened row index into a 4-D leading-axis pointer offset."""

    # Per-token modulation can span more than 2**31 elements (e.g. LingBot
    # at 720p). Widen before stride multiplication, not after it overflows.
    row = row.to(tl.int64)
    i3 = row % d3
    row = row // d3
    i2 = row % d2
    row = row // d2
    i1 = row % d1
    i0 = row // d1
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _round_float32_to_bfloat16(value):
    """Materialize an IEEE round-to-nearest-even BF16 boundary in registers.

    LLVM can legally eliminate a float32 -> BF16 -> float32 cast pair when no
    memory observes the intermediate value. The eager Wan expression does
    materialize ``update * gate`` as BF16, so encode that rounding explicitly
    in the float32 bit representation before the residual add.
    """

    bits = value.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _round_float32_to_float16(value):
    """Keep eager FP16 product rounding even if LLVM removes cast pairs."""

    return tl.inline_asm_elementwise(
        "{ .reg .b16 rounded; cvt.rn.f16.f32 rounded, $1; cvt.f32.f16 $0, rounded; }",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_float32_to_bfloat16_ptx(value):
    """Single-cycle SM80+ PTX IEEE round-to-nearest-even BF16 boundary."""

    return tl.inline_asm_elementwise(
        "{ .reg .b16 r; .reg .b32 b; cvt.rn.bf16.f32 r, $1; cvt.u32.u16 b, r; shl.b32 $0, b, 16; }",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_to_dtype(value, dtype: tl.constexpr):
    """Materialize a PyTorch low-precision expression boundary in registers."""

    if dtype == tl.bfloat16:
        return _round_float32_to_bfloat16(value)
    elif dtype == tl.float16:
        return _round_float32_to_float16(value)
    else:
        return value


@triton.jit
def _round_to_dtype_ptx(value, dtype: tl.constexpr):
    """Single-instruction SM80+ IEEE rounding boundary for BF16/FP16."""

    if dtype == tl.bfloat16:
        return _round_float32_to_bfloat16_ptx(value)
    elif dtype == tl.float16:
        return _round_float32_to_float16(value)
    else:
        return value


@triton.jit
def _fused_layer_norm_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Single-pass affine-free LayerNorm + AdaLN scale/shift with exact IEEE BF16/FP16 boundaries."""

    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols

    x_input = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f = x_input.to(tl.float32)
    mean = tl.sum(x_f, axis=0) / features
    diff = tl.where(mask, x_f - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / features
    rstd = tl.rsqrt(var + eps)
    normed = _round_to_dtype_ptx(diff * rstd, x_input.dtype)

    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    scale_factor = _round_to_dtype_ptx(1.0 + scale_input.to(tl.float32), scale_input.dtype)
    product = normed * scale_factor
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)
    tl.store(out_ptr + offsets, product + shift_input.to(tl.float32), mask=mask)


@triton.jit
def _residual_gate_layer_norm_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    x_out_ptr,
    norm_out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    gs4: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    round_x_bf16: tl.constexpr,
    round_x_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Single-pass ``x_out = residual + update * gate`` and ``norm_out = LayerNorm(x_out, w, b)``."""

    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols

    gate_base = _broadcast_row_offset(row, d0, d1, d2, d3, gs0, gs1, gs2, gs3)
    residual_f = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update_f = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_f = tl.load(gate_ptr + gate_base + cols * gs4, mask=mask, other=0.0).to(tl.float32)

    product = update_f * gate_f
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)

    x_new = residual_f + product
    if round_x_bf16:
        x_new = _round_float32_to_bfloat16_ptx(x_new)
    elif round_x_fp16:
        x_new = _round_float32_to_float16(x_new)
    tl.store(x_out_ptr + offsets, x_new, mask=mask)

    x_masked = tl.where(mask, x_new, 0.0)
    mean = tl.sum(x_masked, axis=0) / features
    diff = tl.where(mask, x_masked - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / features
    rstd = tl.rsqrt(var + eps)
    normed = diff * rstd
    if has_weight:
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w
    if has_bias:
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b
    tl.store(norm_out_ptr + offsets, normed, mask=mask)


@triton.jit
def _residual_add_layer_norm_scale_shift_kernel(
    residual_ptr,
    update_ptr,
    scale_ptr,
    shift_ptr,
    x_out_ptr,
    norm_out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    round_x_bf16: tl.constexpr,
    round_x_fp16: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Single-pass ``x_out = residual + update`` and ``norm_out = LayerNorm(x_out)*(1+scale)+shift``."""

    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols

    residual_f = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update_f = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_new = residual_f + update_f
    if round_x_bf16:
        x_new = _round_float32_to_bfloat16_ptx(x_new)
    elif round_x_fp16:
        x_new = _round_float32_to_float16(x_new)
    tl.store(x_out_ptr + offsets, x_new, mask=mask)

    x_masked = tl.where(mask, x_new, 0.0)
    mean = tl.sum(x_masked, axis=0) / features
    diff = tl.where(mask, x_masked - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / features
    rstd = tl.rsqrt(var + eps)
    normed = diff * rstd
    if round_x_bf16:
        normed = _round_float32_to_bfloat16_ptx(normed)
    elif round_x_fp16:
        normed = _round_float32_to_float16(normed)

    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    scale_factor = _round_to_dtype_ptx(1.0 + scale_input.to(tl.float32), scale_input.dtype)
    product = normed * scale_factor
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)
    tl.store(norm_out_ptr + offsets, product + shift_input.to(tl.float32), mask=mask)


@triton.jit
def _residual_gate_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    out_ptr,
    rows,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    gs4: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Fused ``residual + update * gate`` with broadcast gate strides.

    One program per row; ``block`` is ``next_power_of_2(features)`` and must
    cover the last dimension (eligibility caps features at 8192).
    """

    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols
    gate_base = _broadcast_row_offset(row, d0, d1, d2, d3, gs0, gs1, gs2, gs3)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update_input = tl.load(update_ptr + offsets, mask=mask, other=0.0)
    gate_input = tl.load(gate_ptr + gate_base + cols * gs4, mask=mask, other=0.0)
    product = update_input.to(tl.float32) * gate_input.to(tl.float32)
    # Eager ``update * gate`` materializes in its promoted dtype before the
    # following add.  Restore that BF16/FP16 rounding boundary while retaining
    # a single launch.
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)
    tl.store(out_ptr + offsets, residual + product.to(tl.float32), mask=mask)


@triton.jit
def _scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    elements,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Fused ``x * (1 + scale) + shift`` (strict int64, single-cycle SM80+ PTX rounding)."""

    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block).to(tl.int64)
    mask = offsets < elements
    row = offsets // features
    cols = offsets - row * features
    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    x_input = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    # Match ``x * (1 + scale) + shift`` expression boundaries via single-cycle PTX rounding.
    scale_factor = _round_to_dtype_ptx(1.0 + scale_input.to(tl.float32), scale_input.dtype)
    product = x_input.to(tl.float32) * scale_factor
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)
    tl.store(out_ptr + offsets, product + shift_input.to(tl.float32), mask=mask)


@triton.jit
def _rms_norm_scale_shift_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    has_weight: tl.constexpr,
    round_modulation: tl.constexpr,
    block: tl.constexpr,
):
    """RMSNorm (optional weight) then AdaLN; weight is applied before the dtype cut."""

    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block).to(tl.int64)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols
    x_input = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x_input.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / features
    normed = x * tl.rsqrt(variance + eps)
    if has_weight:
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed *= weight
    # torch.rms_norm applies its learned weight in the accumulation dtype and
    # rounds the completed norm once. Rounding before the weight can introduce
    # 0.1-level BF16 errors after AdaLN modulation on common DiT activations.
    normed = normed.to(x_input.dtype)

    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    scale_factor = (1.0 + scale_input.to(tl.float32)).to(scale_input.dtype)
    modulated = normed.to(tl.float32) * scale_factor.to(tl.float32)
    if round_modulation:
        modulated = modulated.to(x_input.dtype)
    output = modulated.to(tl.float32) + shift_input.to(tl.float32)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    q_stride_s,
    q_stride_h,
    k_stride_s,
    k_stride_h,
    freq_stride_s,
    freq_stride_d,
    sequence,
    heads: tl.constexpr,
    eps,
    rope_fp32: tl.constexpr,
    half_dim: tl.constexpr,
    block_half: tl.constexpr,
    block_h: tl.constexpr,
):
    """Head-tiled per-head RMSNorm + non-interleaved RoPE; ``grid = (B*S, 2)``.

    Splits Q (pid(1) == 0) and K (pid(1) == 1) across CTAs and tiles ``block_h``
    heads per iteration so ``cos``/``sin`` and RMSNorm weights are loaded once
    per token while keeping register pressure low (zero local-memory spills).
    """

    tok = tl.program_id(0).to(tl.int64)
    is_k = tl.program_id(1)
    batch = tok // sequence
    seq = tok % sequence

    offs = tl.arange(0, block_half)
    d_mask = offs < half_dim
    freq = tl.load(freqs_ptr + seq * freq_stride_s + offs * freq_stride_d, mask=d_mask, other=0.0).to(tl.float32)
    cosine = tl.cos(freq)[None, :]
    sine = tl.sin(freq)[None, :]

    if is_k == 0:
        in_ptr = q_ptr
        out_ptr = q_out_ptr
        stride_s = q_stride_s
        stride_h = q_stride_h
        w0 = tl.load(q_weight_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        w1 = tl.load(q_weight_ptr + half_dim + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
    else:
        in_ptr = k_ptr
        out_ptr = k_out_ptr
        stride_s = k_stride_s
        stride_h = k_stride_h
        w0 = tl.load(k_weight_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]
        w1 = tl.load(k_weight_ptr + half_dim + offs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

    tok_base = (batch * sequence + seq) * stride_s
    head_dim: tl.constexpr = 2 * half_dim
    for h_start in tl.static_range(0, heads, block_h):
        hs = h_start + tl.arange(0, block_h)
        mask = (hs < heads)[:, None] & d_mask[None, :]
        row_base = tok_base + hs[:, None] * stride_h
        x0_in = tl.load(in_ptr + row_base + offs[None, :], mask=mask, other=0.0)
        x1_in = tl.load(in_ptr + row_base + half_dim + offs[None, :], mask=mask, other=0.0)
        x0_f = x0_in.to(tl.float32)
        x1_f = x1_in.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x0_f * x0_f + x1_f * x1_f, axis=1) / head_dim + eps)[:, None]
        # Match Wan/Gamma RMSNorm contract: round normalized activation to input
        # dtype before multiplying by the learned weight.
        xa = ((x0_f * rstd).to(x0_in.dtype).to(tl.float32) * w0).to(x0_in.dtype)
        xb = ((x1_f * rstd).to(x1_in.dtype).to(tl.float32) * w1).to(x1_in.dtype)
        if rope_fp32:
            xa = xa.to(tl.float32)
            xb = xb.to(tl.float32)
        tl.store(out_ptr + row_base + offs[None, :], xa * cosine - xb * sine, mask=mask)
        tl.store(out_ptr + row_base + half_dim + offs[None, :], xb * cosine + xa * sine, mask=mask)


@triton.jit
def _hidden_qk_rmsnorm_rope_3d_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    sequence,
    q_batch_stride,
    q_seq_stride,
    k_batch_stride,
    k_seq_stride,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    pairs_per_head: tl.constexpr,
    valid_tokens,
    sequence_offset,
    start_frame,
    height,
    width,
    temporal_pairs: tl.constexpr,
    height_pairs: tl.constexpr,
    store_feature_start: tl.constexpr,
    store_feature_end: tl.constexpr,
    eps,
    freq_stride_position,
    freq_stride_pair,
    freq_stride_component,
    BLOCK_N: tl.constexpr,
    FAST_HEAD_FREQ: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """Split-Q/K register-safe full-hidden RMSNorm + packed interleaved 3D RoPE.

    Uses ``grid = (B*S, 2)`` (Q on pid(1)==0, K on pid(1)==1), 128-bit contiguous
    loads/stores with in-register ``reshape``/``split``/``join`` de-interleaving,
    and loads the per-head 3D frequency table once per CTA when ``pairs_per_head``
    is a power of two.
    """

    row = tl.program_id(0).to(tl.int64)
    is_k = tl.program_id(1)
    batch_idx = row // sequence
    local_token = row % sequence
    global_token = sequence_offset + local_token

    if is_k == 0:
        in_base = q_ptr + batch_idx * q_batch_stride + local_token * q_seq_stride
        out_base = q_out_ptr + row * hidden_size
        w_ptr = q_weight_ptr
    else:
        in_base = k_ptr + batch_idx * k_batch_stride + local_token * k_seq_stride
        out_base = k_out_ptr + row * hidden_size
        w_ptr = k_weight_ptr

    BLOCK_PAIRS: tl.constexpr = BLOCK_N // 2
    spatial_plane = height * width
    temporal_position = start_frame + global_token // spatial_plane
    spatial_position = global_token % spatial_plane
    height_position = spatial_position // width
    width_position = spatial_position % width
    valid = global_token < valid_tokens

    if SINGLE_PASS:
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
        x_f = x_raw.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=0) / hidden_size + eps)
        w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # WanRMSNorm rounds the normalized value to the projection dtype before
        # multiplying by the learned weight.
        normed = ((x_f * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
        xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
        p_idx = tl.arange(0, BLOCK_PAIRS)
        even_col = p_idx * 2
        odd_col = even_col + 1
        rot_mask = (even_col >= store_feature_start) & (odd_col < store_feature_end)
        if FAST_HEAD_FREQ:
            HEADS_TILE: tl.constexpr = BLOCK_PAIRS // pairs_per_head
            p_head = tl.arange(0, pairs_per_head)
            rope_pos_h = tl.where(
                p_head < temporal_pairs,
                temporal_position,
                tl.where(p_head < temporal_pairs + height_pairs, height_position, width_position),
            )
            fb_h = rope_pos_h * freq_stride_position + p_head * freq_stride_pair
            if USE_FP64:
                c_h = tl.load(freqs_ptr + fb_h, mask=valid, other=1.0).to(tl.float64)[None, :]
                s_h = tl.load(freqs_ptr + fb_h + freq_stride_component, mask=valid, other=0.0).to(tl.float64)[None, :]
            else:
                c_h = tl.load(freqs_ptr + fb_h, mask=valid, other=1.0).to(tl.float32)[None, :]
                s_h = tl.load(freqs_ptr + fb_h + freq_stride_component, mask=valid, other=0.0).to(tl.float32)[None, :]
            xe2 = tl.reshape(xe, (HEADS_TILE, pairs_per_head))
            xo2 = tl.reshape(xo, (HEADS_TILE, pairs_per_head))
            rot2 = tl.reshape(rot_mask, (HEADS_TILE, pairs_per_head))
            if USE_FP64:
                re = (xe2.to(tl.float64) * c_h - xo2.to(tl.float64) * s_h).to(x_raw.dtype)
                ro = (xo2.to(tl.float64) * c_h + xe2.to(tl.float64) * s_h).to(x_raw.dtype)
            else:
                re = (xe2.to(tl.float32) * c_h - xo2.to(tl.float32) * s_h).to(x_raw.dtype)
                ro = (xo2.to(tl.float32) * c_h + xe2.to(tl.float32) * s_h).to(x_raw.dtype)
            out_e = tl.where(rot2, re, xe2)
            out_o = tl.where(rot2, ro, xo2)
            out_vec = tl.reshape(tl.join(out_e, out_o), (BLOCK_N,))
        else:
            p_mask = odd_col < hidden_size
            p_in = p_idx % pairs_per_head
            rope_pos = tl.where(
                p_in < temporal_pairs,
                temporal_position,
                tl.where(p_in < temporal_pairs + height_pairs, height_position, width_position),
            )
            fb = rope_pos * freq_stride_position + p_in * freq_stride_pair
            if USE_FP64:
                c = tl.load(freqs_ptr + fb, mask=p_mask & valid, other=1.0).to(tl.float64)
                s = tl.load(freqs_ptr + fb + freq_stride_component, mask=p_mask & valid, other=0.0).to(tl.float64)
                re = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(x_raw.dtype)
                ro = (xo.to(tl.float64) * c + xe.to(tl.float64) * s).to(x_raw.dtype)
            else:
                c = tl.load(freqs_ptr + fb, mask=p_mask & valid, other=1.0).to(tl.float32)
                s = tl.load(freqs_ptr + fb + freq_stride_component, mask=p_mask & valid, other=0.0).to(tl.float32)
                re = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                ro = (xo.to(tl.float32) * c + xe.to(tl.float32) * s).to(x_raw.dtype)
            out_vec = tl.reshape(tl.join(tl.where(rot_mask, re, xe), tl.where(rot_mask, ro, xo)), (BLOCK_N,))
        tl.store(out_base + cols, out_vec, mask=mask)
    else:
        sq_acc = tl.zeros((), dtype=tl.float32)
        for start in tl.static_range(0, hidden_size, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < hidden_size
            x_f = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
            sq_acc += tl.sum(x_f * x_f, axis=0)
        rstd = tl.rsqrt(sq_acc / hidden_size + eps)

        if FAST_HEAD_FREQ:
            HEADS_TILE: tl.constexpr = BLOCK_PAIRS // pairs_per_head
            p_head = tl.arange(0, pairs_per_head)
            rope_pos_h = tl.where(
                p_head < temporal_pairs,
                temporal_position,
                tl.where(p_head < temporal_pairs + height_pairs, height_position, width_position),
            )
            fb_h = rope_pos_h * freq_stride_position + p_head * freq_stride_pair
            if USE_FP64:
                c_h = tl.load(freqs_ptr + fb_h, mask=valid, other=1.0).to(tl.float64)[None, :]
                s_h = tl.load(freqs_ptr + fb_h + freq_stride_component, mask=valid, other=0.0).to(tl.float64)[None, :]
            else:
                c_h = tl.load(freqs_ptr + fb_h, mask=valid, other=1.0).to(tl.float32)[None, :]
                s_h = tl.load(freqs_ptr + fb_h + freq_stride_component, mask=valid, other=0.0).to(tl.float32)[None, :]
            for start in tl.static_range(0, hidden_size, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < hidden_size
                x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
                w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = ((x_raw.to(tl.float32) * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
                xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
                p_idx = (start // 2) + tl.arange(0, BLOCK_PAIRS)
                even_col = p_idx * 2
                odd_col = even_col + 1
                rot_mask = (even_col >= store_feature_start) & (odd_col < store_feature_end)
                xe2 = tl.reshape(xe, (HEADS_TILE, pairs_per_head))
                xo2 = tl.reshape(xo, (HEADS_TILE, pairs_per_head))
                rot2 = tl.reshape(rot_mask, (HEADS_TILE, pairs_per_head))
                if USE_FP64:
                    re = (xe2.to(tl.float64) * c_h - xo2.to(tl.float64) * s_h).to(x_raw.dtype)
                    ro = (xo2.to(tl.float64) * c_h + xe2.to(tl.float64) * s_h).to(x_raw.dtype)
                else:
                    re = (xe2.to(tl.float32) * c_h - xo2.to(tl.float32) * s_h).to(x_raw.dtype)
                    ro = (xo2.to(tl.float32) * c_h + xe2.to(tl.float32) * s_h).to(x_raw.dtype)
                out_vec = tl.reshape(tl.join(tl.where(rot2, re, xe2), tl.where(rot2, ro, xo2)), (BLOCK_N,))
                tl.store(out_base + cols, out_vec, mask=mask)
        else:
            for start in tl.static_range(0, hidden_size, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < hidden_size
                x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
                w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = ((x_raw.to(tl.float32) * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
                xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
                p_idx = (start // 2) + tl.arange(0, BLOCK_PAIRS)
                even_col = p_idx * 2
                odd_col = even_col + 1
                rot_mask = (even_col >= store_feature_start) & (odd_col < store_feature_end)
                p_mask = odd_col < hidden_size
                p_in = p_idx % pairs_per_head
                rope_pos = tl.where(
                    p_in < temporal_pairs,
                    temporal_position,
                    tl.where(p_in < temporal_pairs + height_pairs, height_position, width_position),
                )
                fb = rope_pos * freq_stride_position + p_in * freq_stride_pair
                if USE_FP64:
                    c = tl.load(freqs_ptr + fb, mask=p_mask & valid, other=1.0).to(tl.float64)
                    s = tl.load(freqs_ptr + fb + freq_stride_component, mask=p_mask & valid, other=0.0).to(tl.float64)
                    re = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(x_raw.dtype)
                    ro = (xo.to(tl.float64) * c + xe.to(tl.float64) * s).to(x_raw.dtype)
                else:
                    c = tl.load(freqs_ptr + fb, mask=p_mask & valid, other=1.0).to(tl.float32)
                    s = tl.load(freqs_ptr + fb + freq_stride_component, mask=p_mask & valid, other=0.0).to(tl.float32)
                    re = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                    ro = (xo.to(tl.float32) * c + xe.to(tl.float32) * s).to(x_raw.dtype)
                out_vec = tl.reshape(tl.join(tl.where(rot_mask, re, xe), tl.where(rot_mask, ro, xo)), (BLOCK_N,))
                tl.store(out_base + cols, out_vec, mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# Launch helpers — pad rank to 5; warps grow with next_pow2 feature width
# ──────────────────────────────────────────────────────────────────────────


def _padded_outer_shape_and_strides(
    x: torch.Tensor, broadcast: torch.Tensor
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return 4-D leading shape plus 5-D strides after expanding *broadcast* to *x*."""

    expanded = broadcast.expand_as(x)
    outer_shape = (1,) * (5 - x.ndim) + tuple(int(dim) for dim in x.shape)
    strides = (0,) * (5 - expanded.ndim) + tuple(int(stride) for stride in expanded.stride())
    return outer_shape[:-1], strides


def _num_warps(block: int) -> int:
    """Use 8 warps once the feature tile exceeds 256 so wide AdaLN rows stay occupied."""

    if block <= 256:
        return 4
    return 8


def _norm_num_warps(features: int, block: int) -> int:
    """Hopper-calibrated warp count for row-wise norm+AdaLN kernels.

    Avoids excess cross-warp reduction sync on medium/padded widths (1536, 2048,
    3072, 5120) while keeping 8 warps for exact 4096-wide rows.
    """

    if block <= 256:
        return 4
    if features <= 1536:
        return 2
    if features <= 3072 or features > 4096:
        return 4
    return 8


def silu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * value`` in one regular pointwise launch."""

    output = torch.empty_like(gate)
    block = 1024
    _silu_mul_kernel[(triton.cdiv(gate.numel(), block),)](
        gate,
        value,
        output,
        gate.numel(),
        block=block,
        num_warps=4,
    )
    return output


def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Split the last dimension and compute ``silu(first) * second``."""

    half_features = int(input.shape[-1] // 2)
    output = torch.empty((*input.shape[:-1], half_features), dtype=input.dtype, device=input.device)
    block = 1024
    _packed_silu_mul_kernel[(triton.cdiv(output.numel(), block),)](
        input,
        output,
        output.numel(),
        half_features=half_features,
        block=block,
        num_warps=4,
    )
    return output


# ──────────────────────────────────────────────────────────────────────────
# Public wrappers — promote output dtype; BLOCK is next_pow2(features) or 1024
# ──────────────────────────────────────────────────────────────────────────


def residual_gate(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Launch fused residual gating; output dtype is the three-way promotion."""

    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    dims, gate_strides = _padded_outer_shape_and_strides(residual, gate)
    block = triton.next_power_of_2(features)
    output_dtype = torch.promote_types(residual.dtype, torch.promote_types(update.dtype, gate.dtype))
    product_dtype = torch.promote_types(update.dtype, gate.dtype)
    output = torch.empty_like(residual, dtype=output_dtype)
    _residual_gate_kernel[(rows,)](
        residual,
        update,
        gate,
        output,
        rows,
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        gs0=gate_strides[0],
        gs1=gate_strides[1],
        gs2=gate_strides[2],
        gs3=gate_strides[3],
        gs4=gate_strides[4],
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        block=block,
        num_warps=_num_warps(block),
        enable_fp_fusion=False,
    )
    return output


def _launch_residual_gate_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Launch fused residual gating with ``residual`` as the output buffer."""

    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    dims, gate_strides = _padded_outer_shape_and_strides(residual, gate)
    block = triton.next_power_of_2(features)
    product_dtype = torch.promote_types(update.dtype, gate.dtype)
    _residual_gate_kernel[(rows,)](
        residual,
        update,
        gate,
        residual,
        rows,
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        gs0=gate_strides[0],
        gs1=gate_strides[1],
        gs2=gate_strides[2],
        gs3=gate_strides[3],
        gs4=gate_strides[4],
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        block=block,
        num_warps=_num_warps(block),
    )


@torch.library.custom_op(
    "worldfoundry::residual_gate_inplace",
    mutates_args=("residual",),
    device_types="cuda",
)
def _residual_gate_inplace_custom(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Tell PyTorch that the opaque Triton launch mutates ``residual``."""

    _launch_residual_gate_inplace(residual, update, gate)


@_residual_gate_inplace_custom.register_fake
def _residual_gate_inplace_custom_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Mutation-only fake implementation used by export/FakeTensorMode."""

    del residual, update, gate


def residual_gate_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Run the mutation-aware custom op and return its aliased input."""

    _residual_gate_inplace_custom(residual, update, gate)
    return residual


def scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Fuse ``x * (1 + scale) + shift`` while preserving eager rounding."""

    features = int(x.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(x, scale)
    _, shift_strides = _padded_outer_shape_and_strides(x, shift)
    output_dtype = torch.promote_types(x.dtype, torch.promote_types(scale.dtype, shift.dtype))
    product_dtype = torch.promote_types(x.dtype, scale.dtype)
    output = torch.empty_like(x, dtype=output_dtype)
    block = 1024
    _scale_shift_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        scale,
        shift,
        output,
        x.numel(),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        block=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    *,
    upcast: bool = False,
) -> torch.Tensor:
    """Preserve vendor LayerNorm bits, then launch one fused AdaLN kernel."""

    # A custom reduction can change even low-precision results near a rounding
    # boundary. Keep the same vendor reduction, dtype, and optional upcast as
    # the eager expression; only combine its following pointwise operations.
    normalized = torch.nn.functional.layer_norm(
        x.float() if upcast else x,
        (x.shape[-1],),
        weight=None,
        bias=None,
        eps=eps,
    )
    if upcast:
        normalized = normalized.to(x.dtype)
    return scale_shift(normalized, scale, shift)


def rms_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Launch fused RMSNorm+AdaLN; a missing weight reuses *x* as a dummy pointer."""

    rows = x.numel() // x.shape[-1]
    features = int(x.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(x, scale)
    _, shift_strides = _padded_outer_shape_and_strides(x, shift)
    block = triton.next_power_of_2(features)
    modulation_dtype = torch.promote_types(x.dtype, scale.dtype)
    output_dtype = torch.promote_types(x.dtype, torch.promote_types(scale.dtype, shift.dtype))
    output = torch.empty_like(x, dtype=output_dtype)
    weight_pointer = x if weight is None else weight
    _rms_norm_scale_shift_kernel[(rows,)](
        x,
        weight_pointer,
        scale,
        shift,
        output,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        has_weight=weight is not None,
        round_modulation=modulation_dtype in {torch.float16, torch.bfloat16},
        block=block,
        num_warps=_norm_num_warps(features, block),
    )
    return output


def qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    rope_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch head-tiled per-head RMSNorm+RoPE; *q* and *k* must already be ``[B, S, H, D]``."""

    batch, sequence, heads, head_dim = q.shape
    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    max_block_h = 4 if head_dim >= 256 else 8
    block_h = min(triton.next_power_of_2(heads), max_block_h)
    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)
    _qk_rmsnorm_rope_kernel[(batch * sequence, 2)](
        q,
        k,
        q_output,
        k_output,
        q_weight,
        k_weight,
        freqs,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        freqs.stride(0),
        freqs.stride(3),
        sequence,
        heads=heads,
        eps=float(eps),
        rope_fp32=bool(rope_fp32),
        half_dim=half_dim,
        block_half=block_half,
        block_h=block_h,
        num_warps=4,
    )
    return q_output, k_output


def hidden_qk_rmsnorm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    *,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float,
    sequence_offset: int,
    start_frame: int,
    valid_tokens: int,
    head_start: int,
    head_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch packed 3D RoPE; temporal pair count uses Wan's ``pairs - 2*(pairs//3)`` split."""

    hidden_size = int(q.shape[-1])
    head_dim = hidden_size // int(num_heads)
    _, height, width = (int(value) for value in grid_size)
    complex_dim = head_dim // 2
    temporal_pairs = complex_dim - 2 * (complex_dim // 3)
    height_pairs = complex_dim // 3
    real_freqs = torch.view_as_real(freqs) if freqs.is_complex() else freqs
    use_fp64 = real_freqs.dtype == torch.float64
    single_pass = hidden_size <= 2048
    block_n = max(2, triton.next_power_of_2(hidden_size)) if single_pass else 2048
    block_pairs = block_n // 2
    fast_head_freq = complex_dim > 0 and (complex_dim & (complex_dim - 1)) == 0 and block_pairs >= complex_dim
    q_output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_output = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    warps = 2 if (use_fp64 or block_n <= 2048) else 4
    _hidden_qk_rmsnorm_rope_3d_kernel[(q.shape[0] * q.shape[1], 2)](
        q,
        k,
        q_output,
        k_output,
        q_weight,
        k_weight,
        real_freqs,
        q.shape[1],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        hidden_size=hidden_size,
        head_dim=head_dim,
        pairs_per_head=complex_dim,
        valid_tokens=int(valid_tokens),
        sequence_offset=int(sequence_offset),
        start_frame=int(start_frame),
        height=height,
        width=width,
        temporal_pairs=temporal_pairs,
        height_pairs=height_pairs,
        store_feature_start=int(head_start) * head_dim,
        store_feature_end=int(head_end) * head_dim,
        eps=float(eps),
        freq_stride_position=real_freqs.stride(0),
        freq_stride_pair=real_freqs.stride(1),
        freq_stride_component=real_freqs.stride(2),
        BLOCK_N=block_n,
        FAST_HEAD_FREQ=fast_head_freq,
        SINGLE_PASS=single_pass,
        USE_FP64=use_fp64,
        num_warps=warps,
    )
    return q_output, k_output


@triton.jit
def _real_rope_bshd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    seq_len,
    num_heads,
    x_stride_b,
    x_stride_s,
    x_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    cos_stride_s,
    sin_stride_s,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Head-tiled real-valued RoPE on ``[B, S, H, D]`` with 128-bit contiguous loads/stores."""
    token_row = tl.program_id(0).to(tl.int64)
    head_block = tl.program_id(1).to(tl.int64)
    batch_idx = token_row // seq_len
    seq_idx = token_row % seq_len

    hs = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = hs < num_heads
    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < HEAD_DIM
    mask = h_mask[:, None] & d_mask[None, :]
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2
    p_offs = tl.arange(0, BLOCK_PAIRS)
    p_mask = p_offs < PAIRS

    x_base = x_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s + hs[:, None] * x_stride_h
    out_base = out_ptr + batch_idx * out_stride_b + seq_idx * out_stride_s + hs[:, None] * out_stride_h
    x_raw = tl.load(x_base + cols[None, :], mask=mask, other=0.0)
    x_even_raw, x_odd_raw = tl.split(tl.reshape(x_raw, (BLOCK_H, BLOCK_PAIRS, 2)))
    if USE_FP64:
        x_even = x_even_raw.to(tl.float64)
        x_odd = x_odd_raw.to(tl.float64)
        c = tl.load(cos_ptr + seq_idx * cos_stride_s + p_offs, mask=p_mask, other=1.0).to(tl.float64)[None, :]
        s = tl.load(sin_ptr + seq_idx * sin_stride_s + p_offs, mask=p_mask, other=0.0).to(tl.float64)[None, :]
    else:
        x_even = x_even_raw.to(tl.float32)
        x_odd = x_odd_raw.to(tl.float32)
        c = tl.load(cos_ptr + seq_idx * cos_stride_s + p_offs, mask=p_mask, other=1.0).to(tl.float32)[None, :]
        s = tl.load(sin_ptr + seq_idx * sin_stride_s + p_offs, mask=p_mask, other=0.0).to(tl.float32)[None, :]

    rot_even = (x_even * c - x_odd * s).to(x_raw.dtype)
    rot_odd = (x_even * s + x_odd * c).to(x_raw.dtype)
    tl.store(out_base + cols[None, :], tl.reshape(tl.join(rot_even, rot_odd), (BLOCK_H, BLOCK_D)), mask=mask)


@triton.jit
def _real_rope_bshd_1cta_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    seq_len,
    NUM_HEADS: tl.constexpr,
    x_stride_b,
    x_stride_s,
    x_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    cos_stride_s,
    sin_stride_s,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Single-CTA-per-token real RoPE amortizing FP64/FP32 cos/sin across all heads."""
    token_row = tl.program_id(0).to(tl.int64)
    batch_idx = token_row // seq_len
    seq_idx = token_row % seq_len

    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < HEAD_DIM
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2
    p_offs = tl.arange(0, BLOCK_PAIRS)
    p_mask = p_offs < PAIRS
    if USE_FP64:
        c = tl.load(cos_ptr + seq_idx * cos_stride_s + p_offs, mask=p_mask, other=1.0).to(tl.float64)[None, :]
        s = tl.load(sin_ptr + seq_idx * sin_stride_s + p_offs, mask=p_mask, other=0.0).to(tl.float64)[None, :]
    else:
        c = tl.load(cos_ptr + seq_idx * cos_stride_s + p_offs, mask=p_mask, other=1.0).to(tl.float32)[None, :]
        s = tl.load(sin_ptr + seq_idx * sin_stride_s + p_offs, mask=p_mask, other=0.0).to(tl.float32)[None, :]

    x_tok = x_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
    out_tok = out_ptr + batch_idx * out_stride_b + seq_idx * out_stride_s
    for h_start in tl.static_range(0, NUM_HEADS, BLOCK_H):
        hs = h_start + tl.arange(0, BLOCK_H)
        mask = (hs < NUM_HEADS)[:, None] & d_mask[None, :]
        x_raw = tl.load(x_tok + hs[:, None] * x_stride_h + cols[None, :], mask=mask, other=0.0)
        x_even_raw, x_odd_raw = tl.split(tl.reshape(x_raw, (BLOCK_H, BLOCK_PAIRS, 2)))
        if USE_FP64:
            rot_even = (x_even_raw.to(tl.float64) * c - x_odd_raw.to(tl.float64) * s).to(x_raw.dtype)
            rot_odd = (x_even_raw.to(tl.float64) * s + x_odd_raw.to(tl.float64) * c).to(x_raw.dtype)
        else:
            rot_even = (x_even_raw.to(tl.float32) * c - x_odd_raw.to(tl.float32) * s).to(x_raw.dtype)
            rot_odd = (x_even_raw.to(tl.float32) * s + x_odd_raw.to(tl.float32) * c).to(x_raw.dtype)
        tl.store(
            out_tok + hs[:, None] * out_stride_h + cols[None, :],
            tl.reshape(tl.join(rot_even, rot_odd), (BLOCK_H, BLOCK_D)),
            mask=mask,
        )


def apply_real_rope_bshd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply real-valued ``(cos, sin)`` RoPE in one Triton pass with register-resident FP64/FP32."""
    b, s, h, d = x.shape
    pairs = d // 2
    block_d = max(2, triton.next_power_of_2(d))
    cos_2d = cos.reshape(s, pairs).contiguous()
    sin_2d = sin.reshape(s, pairs).contiguous()
    out = torch.empty((b, s, h, d), device=x.device, dtype=x.dtype)
    use_fp64 = cos.dtype == torch.float64
    total_tokens = b * s
    if total_tokens >= 512:
        if d <= 128 and h <= 32 and (h & (h - 1)) == 0:
            block_h = max(1, h)
        else:
            block_h = 16 if d <= 128 else 8
            block_h = min(block_h, triton.next_power_of_2(h))
        _real_rope_bshd_1cta_kernel[(total_tokens,)](
            x,
            cos_2d,
            sin_2d,
            out,
            s,
            NUM_HEADS=h,
            x_stride_b=x.stride(0),
            x_stride_s=x.stride(1),
            x_stride_h=x.stride(2),
            out_stride_b=out.stride(0),
            out_stride_s=out.stride(1),
            out_stride_h=out.stride(2),
            cos_stride_s=cos_2d.stride(0),
            sin_stride_s=sin_2d.stride(0),
            HEAD_DIM=d,
            PAIRS=pairs,
            BLOCK_D=block_d,
            BLOCK_H=block_h,
            USE_FP64=use_fp64,
            num_warps=4,
        )
    else:
        block_h = min(8 if d <= 128 else 4, triton.next_power_of_2(h))
        _real_rope_bshd_kernel[(total_tokens, triton.cdiv(h, block_h))](
            x,
            cos_2d,
            sin_2d,
            out,
            s,
            h,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            cos_2d.stride(0),
            sin_2d.stride(0),
            HEAD_DIM=d,
            PAIRS=pairs,
            BLOCK_D=block_d,
            BLOCK_H=block_h,
            USE_FP64=use_fp64,
            num_warps=4,
        )
    return out


def fused_layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Single-pass affine-free LayerNorm + AdaLN scale/shift with exact IEEE BF16/FP16 rounding."""

    rows = x.numel() // x.shape[-1]
    features = int(x.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(x, scale)
    _, shift_strides = _padded_outer_shape_and_strides(x, shift)
    block = triton.next_power_of_2(features)
    product_dtype = torch.promote_types(x.dtype, scale.dtype)
    output_dtype = torch.promote_types(product_dtype, shift.dtype)
    output = torch.empty_like(x, dtype=output_dtype)
    _fused_layer_norm_scale_shift_kernel[(rows,)](
        x,
        scale,
        shift,
        output,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        block=block,
        num_warps=_norm_num_warps(features, block),
        enable_fp_fusion=False,
    )
    return output


def _launch_residual_gate_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_out: torch.Tensor,
    norm_out: torch.Tensor,
    *,
    has_weight: bool,
    has_bias: bool,
    eps: float,
) -> None:
    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    dims, gate_strides = _padded_outer_shape_and_strides(residual, gate)
    block = triton.next_power_of_2(features)
    product_dtype = torch.promote_types(update.dtype, gate.dtype)
    x_dtype = x_out.dtype
    _residual_gate_layer_norm_kernel[(rows,)](
        residual,
        update,
        gate,
        weight,
        bias,
        x_out,
        norm_out,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        gs0=gate_strides[0],
        gs1=gate_strides[1],
        gs2=gate_strides[2],
        gs3=gate_strides[3],
        gs4=gate_strides[4],
        has_weight=has_weight,
        has_bias=has_bias,
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        round_x_bf16=x_dtype == torch.bfloat16,
        round_x_fp16=x_dtype == torch.float16,
        block=block,
        num_warps=_num_warps(block),
        enable_fp_fusion=False,
    )


def residual_gate_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass ``x_out = residual + update * gate`` and ``LayerNorm(x_out, w, b)``."""

    product_dtype = torch.promote_types(update.dtype, gate.dtype)
    x_out_dtype = torch.promote_types(residual.dtype, product_dtype)
    x_out = torch.empty_like(residual, dtype=x_out_dtype)
    norm_out = torch.empty_like(residual, dtype=x_out_dtype)
    _launch_residual_gate_layer_norm(
        residual,
        update,
        gate,
        residual if weight is None else weight,
        residual if bias is None else bias,
        x_out,
        norm_out,
        has_weight=weight is not None,
        has_bias=bias is not None,
        eps=eps,
    )
    return x_out, norm_out


@torch.library.custom_op(
    "worldfoundry::residual_gate_layer_norm_inplace",
    mutates_args=("residual",),
    device_types="cuda",
)
def _residual_gate_layer_norm_inplace_custom(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    norm_out = torch.empty_like(residual)
    _launch_residual_gate_layer_norm(
        residual,
        update,
        gate,
        residual if weight is None else weight,
        residual if bias is None else bias,
        residual,
        norm_out,
        has_weight=weight is not None,
        has_bias=bias is not None,
        eps=eps,
    )
    return norm_out


@_residual_gate_layer_norm_inplace_custom.register_fake
def _residual_gate_layer_norm_inplace_custom_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    del update, gate, weight, bias, eps
    return torch.empty_like(residual)


def residual_gate_layer_norm_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate ``residual`` in place and return ``(residual, LayerNorm(residual, w, b))``."""

    norm_out = _residual_gate_layer_norm_inplace_custom(
        residual,
        update,
        gate,
        weight,
        bias,
        float(eps),
    )
    return residual, norm_out


def _launch_residual_add_layer_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    x_out: torch.Tensor,
    norm_out: torch.Tensor,
    *,
    eps: float,
) -> None:
    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(residual, scale)
    _, shift_strides = _padded_outer_shape_and_strides(residual, shift)
    block = triton.next_power_of_2(features)
    x_dtype = x_out.dtype
    product_dtype = torch.promote_types(x_dtype, scale.dtype)
    _residual_add_layer_norm_scale_shift_kernel[(rows,)](
        residual,
        update,
        scale,
        shift,
        x_out,
        norm_out,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        round_x_bf16=x_dtype == torch.bfloat16,
        round_x_fp16=x_dtype == torch.float16,
        round_product_bf16=product_dtype == torch.bfloat16,
        round_product_fp16=product_dtype == torch.float16,
        block=block,
        num_warps=_num_warps(block),
        enable_fp_fusion=False,
    )


def residual_add_layer_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass ``x_out = residual + update`` and ``LayerNorm(x_out)*(1+scale)+shift``."""

    x_out_dtype = torch.promote_types(residual.dtype, update.dtype)
    product_dtype = torch.promote_types(x_out_dtype, scale.dtype)
    norm_out_dtype = torch.promote_types(product_dtype, shift.dtype)
    x_out = torch.empty_like(residual, dtype=x_out_dtype)
    norm_out = torch.empty_like(residual, dtype=norm_out_dtype)
    _launch_residual_add_layer_norm_scale_shift(residual, update, scale, shift, x_out, norm_out, eps=eps)
    return x_out, norm_out


@torch.library.custom_op(
    "worldfoundry::residual_add_layer_norm_scale_shift_inplace",
    mutates_args=("residual",),
    device_types="cuda",
)
def _residual_add_layer_norm_scale_shift_inplace_custom(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    product_dtype = torch.promote_types(residual.dtype, scale.dtype)
    norm_out_dtype = torch.promote_types(product_dtype, shift.dtype)
    norm_out = torch.empty_like(residual, dtype=norm_out_dtype)
    _launch_residual_add_layer_norm_scale_shift(residual, update, scale, shift, residual, norm_out, eps=eps)
    return norm_out


@_residual_add_layer_norm_scale_shift_inplace_custom.register_fake
def _residual_add_layer_norm_scale_shift_inplace_custom_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del update, eps
    product_dtype = torch.promote_types(residual.dtype, scale.dtype)
    norm_out_dtype = torch.promote_types(product_dtype, shift.dtype)
    return torch.empty_like(residual, dtype=norm_out_dtype)


def residual_add_layer_norm_scale_shift_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate ``residual`` in place and return ``(residual, LayerNorm(residual)*(1+scale)+shift)``."""

    norm_out = _residual_add_layer_norm_scale_shift_inplace_custom(residual, update, scale, shift, float(eps))
    return residual, norm_out


@triton.jit
def _hidden_qk_rmsnorm_rope_freqs_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    sequence,
    q_batch_stride,
    q_seq_stride,
    k_batch_stride,
    k_seq_stride,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    pairs_per_head: tl.constexpr,
    eps,
    freq_stride_pos,
    freq_stride_pair,
    freq_stride_comp,
    BLOCK_N: tl.constexpr,
    FAST_HEAD_FREQ: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    USE_FP64: tl.constexpr = True,
):
    """Split-Q/K register-safe full-hidden RMSNorm + pre-indexed interleaved RoPE.

    Splits Q (pid(1) == 0) and K (pid(1) == 1) into independent CTAs, uses
    128-bit contiguous loads/stores with in-register ``reshape``/``split``/``join``
    de-interleaving, tiles columns at ``BLOCK_N`` for wide hidden dims (zero spills),
    and broadcasts per-head ``(cos, sin)`` loaded once per CTA.
    """

    row = tl.program_id(0).to(tl.int64)
    is_k = tl.program_id(1)
    batch_idx = row // sequence
    local_token = row % sequence

    if is_k == 0:
        in_base = q_ptr + batch_idx * q_batch_stride + local_token * q_seq_stride
        out_base = q_out_ptr + row * hidden_size
        w_ptr = q_weight_ptr
    else:
        in_base = k_ptr + batch_idx * k_batch_stride + local_token * k_seq_stride
        out_base = k_out_ptr + row * hidden_size
        w_ptr = k_weight_ptr

    BLOCK_PAIRS: tl.constexpr = BLOCK_N // 2

    if SINGLE_PASS:
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden_size
        x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
        x_f = x_raw.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x_f * x_f, axis=0) / hidden_size + eps)
        w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = ((x_f * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
        xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
        if FAST_HEAD_FREQ:
            HEADS_TILE: tl.constexpr = BLOCK_PAIRS // pairs_per_head
            f_idx = tl.arange(0, pairs_per_head * 2)
            f_raw = tl.load(freqs_ptr + local_token * freq_stride_pos + f_idx)
            c_raw, s_raw = tl.split(tl.reshape(f_raw, (pairs_per_head, 2)))
            xe2 = tl.reshape(xe, (HEADS_TILE, pairs_per_head))
            xo2 = tl.reshape(xo, (HEADS_TILE, pairs_per_head))
            if USE_FP64:
                c_b = c_raw.to(tl.float64)[None, :]
                s_b = s_raw.to(tl.float64)[None, :]
                re = (xe2.to(tl.float64) * c_b - xo2.to(tl.float64) * s_b).to(x_raw.dtype)
                ro = (xo2.to(tl.float64) * c_b + xe2.to(tl.float64) * s_b).to(x_raw.dtype)
            else:
                c_b = c_raw.to(tl.float32)[None, :]
                s_b = s_raw.to(tl.float32)[None, :]
                re = (xe2.to(tl.float32) * c_b - xo2.to(tl.float32) * s_b).to(x_raw.dtype)
                ro = (xo2.to(tl.float32) * c_b + xe2.to(tl.float32) * s_b).to(x_raw.dtype)
            out_vec = tl.reshape(tl.join(re, ro), (BLOCK_N,))
        else:
            p_idx = tl.arange(0, BLOCK_PAIRS)
            p_mask = p_idx < (hidden_size // 2)
            p_in = p_idx % pairs_per_head
            fb = local_token * freq_stride_pos + p_in * freq_stride_pair
            if USE_FP64:
                c = tl.load(freqs_ptr + fb, mask=p_mask, other=1.0).to(tl.float64)
                s = tl.load(freqs_ptr + fb + freq_stride_comp, mask=p_mask, other=0.0).to(tl.float64)
                re = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(x_raw.dtype)
                ro = (xo.to(tl.float64) * c + xe.to(tl.float64) * s).to(x_raw.dtype)
            else:
                c = tl.load(freqs_ptr + fb, mask=p_mask, other=1.0).to(tl.float32)
                s = tl.load(freqs_ptr + fb + freq_stride_comp, mask=p_mask, other=0.0).to(tl.float32)
                re = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                ro = (xo.to(tl.float32) * c + xe.to(tl.float32) * s).to(x_raw.dtype)
            out_vec = tl.reshape(tl.join(re, ro), (BLOCK_N,))
        tl.store(out_base + cols, out_vec, mask=mask)
    else:
        sq_acc = tl.zeros((), dtype=tl.float32)
        for start in tl.static_range(0, hidden_size, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < hidden_size
            x_f = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
            sq_acc += tl.sum(x_f * x_f, axis=0)
        rstd = tl.rsqrt(sq_acc / hidden_size + eps)

        if FAST_HEAD_FREQ:
            HEADS_TILE: tl.constexpr = BLOCK_PAIRS // pairs_per_head
            f_idx = tl.arange(0, pairs_per_head * 2)
            f_raw = tl.load(freqs_ptr + local_token * freq_stride_pos + f_idx)
            c_raw, s_raw = tl.split(tl.reshape(f_raw, (pairs_per_head, 2)))
            if USE_FP64:
                c_b = c_raw.to(tl.float64)[None, :]
                s_b = s_raw.to(tl.float64)[None, :]
            else:
                c_b = c_raw.to(tl.float32)[None, :]
                s_b = s_raw.to(tl.float32)[None, :]
            for start in tl.static_range(0, hidden_size, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < hidden_size
                x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
                w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = ((x_raw.to(tl.float32) * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
                xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
                xe2 = tl.reshape(xe, (HEADS_TILE, pairs_per_head))
                xo2 = tl.reshape(xo, (HEADS_TILE, pairs_per_head))
                if USE_FP64:
                    re = (xe2.to(tl.float64) * c_b - xo2.to(tl.float64) * s_b).to(x_raw.dtype)
                    ro = (xo2.to(tl.float64) * c_b + xe2.to(tl.float64) * s_b).to(x_raw.dtype)
                else:
                    re = (xe2.to(tl.float32) * c_b - xo2.to(tl.float32) * s_b).to(x_raw.dtype)
                    ro = (xo2.to(tl.float32) * c_b + xe2.to(tl.float32) * s_b).to(x_raw.dtype)
                out_vec = tl.reshape(tl.join(re, ro), (BLOCK_N,))
                tl.store(out_base + cols, out_vec, mask=mask)
        else:
            for start in tl.static_range(0, hidden_size, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < hidden_size
                x_raw = tl.load(in_base + cols, mask=mask, other=0.0)
                w_f = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = ((x_raw.to(tl.float32) * rstd).to(x_raw.dtype).to(tl.float32) * w_f).to(x_raw.dtype)
                xe, xo = tl.split(tl.reshape(normed, (BLOCK_PAIRS, 2)))
                p_idx = (start // 2) + tl.arange(0, BLOCK_PAIRS)
                p_mask = p_idx < (hidden_size // 2)
                p_in = p_idx % pairs_per_head
                fb = local_token * freq_stride_pos + p_in * freq_stride_pair
                if USE_FP64:
                    c = tl.load(freqs_ptr + fb, mask=p_mask, other=1.0).to(tl.float64)
                    s = tl.load(freqs_ptr + fb + freq_stride_comp, mask=p_mask, other=0.0).to(tl.float64)
                    re = (xe.to(tl.float64) * c - xo.to(tl.float64) * s).to(x_raw.dtype)
                    ro = (xo.to(tl.float64) * c + xe.to(tl.float64) * s).to(x_raw.dtype)
                else:
                    c = tl.load(freqs_ptr + fb, mask=p_mask, other=1.0).to(tl.float32)
                    s = tl.load(freqs_ptr + fb + freq_stride_comp, mask=p_mask, other=0.0).to(tl.float32)
                    re = (xe.to(tl.float32) * c - xo.to(tl.float32) * s).to(x_raw.dtype)
                    ro = (xo.to(tl.float32) * c + xe.to(tl.float32) * s).to(x_raw.dtype)
                out_vec = tl.reshape(tl.join(re, ro), (BLOCK_N,))
                tl.store(out_base + cols, out_vec, mask=mask)


def hidden_qk_rmsnorm_rope_freqs(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    *,
    num_heads: int,
    eps: float,
    precision: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused Q/K RMSNorm + pre-indexed per-token interleaved RoPE."""

    hidden_size = int(q.shape[-1])
    head_dim = hidden_size // int(num_heads)
    pairs_per_head = head_dim // 2
    if freqs.is_complex():
        freqs_2d = freqs.squeeze(1) if freqs.ndim == 3 and freqs.shape[1] == 1 else freqs
        real_freqs = torch.view_as_real(freqs_2d)
    else:
        real_freqs = freqs.squeeze(1) if freqs.ndim == 4 and freqs.shape[1] == 1 else freqs
    if precision is None:
        use_fp64 = real_freqs.dtype == torch.float64
    else:
        normalized_precision = precision.strip().casefold()
        if normalized_precision not in {"fp32", "fp64"}:
            raise ValueError(f"unsupported RoPE precision: {precision!r}")
        use_fp64 = normalized_precision == "fp64"
    single_pass = hidden_size <= 2048
    block_n = max(2, triton.next_power_of_2(hidden_size)) if single_pass else 2048
    block_pairs = block_n // 2
    fast_head_freq = (
        pairs_per_head > 0
        and (pairs_per_head & (pairs_per_head - 1)) == 0
        and block_pairs >= pairs_per_head
        and real_freqs.stride(2) == 1
        and real_freqs.stride(1) == 2
    )
    q_output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_output = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    warps = 2 if use_fp64 else (2 if hidden_size <= 3072 else 4)
    _hidden_qk_rmsnorm_rope_freqs_kernel[(q.shape[0] * q.shape[1], 2)](
        q,
        k,
        q_output,
        k_output,
        q_weight,
        k_weight,
        real_freqs,
        q.shape[1],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        hidden_size=hidden_size,
        head_dim=head_dim,
        pairs_per_head=pairs_per_head,
        eps=float(eps),
        freq_stride_pos=real_freqs.stride(0),
        freq_stride_pair=real_freqs.stride(1),
        freq_stride_comp=real_freqs.stride(2),
        BLOCK_N=block_n,
        FAST_HEAD_FREQ=fast_head_freq,
        SINGLE_PASS=single_pass,
        USE_FP64=use_fp64,
        num_warps=warps,
    )
    return q_output, k_output


@triton.jit
def _residual_gate_add_rms_norm_kernel(
    residual_ptr,
    value_ptr,
    gate_ptr,
    weight_ptr,
    x_out_ptr,
    norm_out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    gs4: tl.constexpr,
    has_weight: tl.constexpr,
    round_product_bf16: tl.constexpr,
    round_product_fp16: tl.constexpr,
    round_x_bf16: tl.constexpr,
    round_x_fp16: tl.constexpr,
    block: tl.constexpr,
):
    """Single-pass ``x_out = residual + value * gate`` and ``norm_out = rms_norm(x_out, weight, eps)``."""

    row = tl.program_id(0).to(tl.int64)
    row_off = row * features
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)

    res_row = residual_ptr + row_off
    val_row = value_ptr + row_off
    xout_row = x_out_ptr + row_off
    nout_row = norm_out_ptr + row_off
    gate_row = gate_ptr + _broadcast_row_offset(row, d0, d1, d2, d3, gs0, gs1, gs2, gs3)

    res_f = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
    val_f = tl.load(val_row + cols, mask=mask, other=0.0).to(tl.float32)
    gate_f = tl.load(gate_row + cols * gs4, mask=mask, other=0.0).to(tl.float32)

    product = val_f * gate_f
    if round_product_bf16:
        product = _round_float32_to_bfloat16_ptx(product)
    elif round_product_fp16:
        product = _round_float32_to_float16(product)

    x_new = res_f + product
    if round_x_bf16:
        x_new = _round_float32_to_bfloat16_ptx(x_new)
    elif round_x_fp16:
        x_new = _round_float32_to_float16(x_new)
    tl.store(xout_row + cols, x_new, mask=mask)

    x_masked = tl.where(mask, x_new, 0.0)
    rstd = tl.rsqrt(tl.sum(x_masked * x_masked, axis=0) / features + eps)
    normed = x_masked * rstd
    if has_weight:
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = normed * weight
    tl.store(nout_row + cols, normed, mask=mask)


def residual_gate_add_rms_norm(
    residual: torch.Tensor,
    value: torch.Tensor,
    norm_weights: torch.Tensor | None,
    eps: float,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Hopper Triton ``output = residual + value * gate`` and ``rms_norm(output, norm_weights, eps)``."""

    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    gate_for_broadcast = gate.reshape(1) if gate.ndim == 0 else gate
    dims, gate_strides = _padded_outer_shape_and_strides(residual, gate_for_broadcast)
    block = triton.next_power_of_2(features)
    product_dtype = torch.promote_types(value.dtype, gate.dtype)
    out_dtype = torch.promote_types(residual.dtype, product_dtype)
    x_out = torch.empty_like(residual, dtype=out_dtype)
    norm_out = torch.empty_like(residual, dtype=out_dtype)
    weight_ptr = residual if norm_weights is None else norm_weights
    num_warps = 4 if features <= 3072 else 8
    with torch.cuda.device(residual.device):
        _residual_gate_add_rms_norm_kernel[(rows,)](
            residual,
            value,
            gate_for_broadcast,
            weight_ptr,
            x_out,
            norm_out,
            rows,
            float(eps),
            features=features,
            d0=dims[0],
            d1=dims[1],
            d2=dims[2],
            d3=dims[3],
            gs0=gate_strides[0],
            gs1=gate_strides[1],
            gs2=gate_strides[2],
            gs3=gate_strides[3],
            gs4=gate_strides[4],
            has_weight=norm_weights is not None,
            round_product_bf16=product_dtype == torch.bfloat16,
            round_product_fp16=product_dtype == torch.float16,
            round_x_bf16=out_dtype == torch.bfloat16,
            round_x_fp16=out_dtype == torch.float16,
            block=block,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return x_out, norm_out


@triton.jit
def _head_sigmoid_gate_kernel(
    attn_ptr,
    logits_ptr,
    out_ptr,
    total_heads,
    dim_head: tl.constexpr,
    round_gate_bf16: tl.constexpr,
    round_gate_fp16: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-pass per-head ``2.0 * sigmoid(gate_logits) * attention_output`` with 128-bit coalesced I/O."""

    h_start = tl.program_id(0).to(tl.int64) * BLOCK_H
    hs_i32 = tl.arange(0, BLOCK_H)
    h_mask = (h_start + hs_i32) < total_heads
    ds_i32 = tl.arange(0, BLOCK_D)
    d_mask = ds_i32 < dim_head
    mask = h_mask[:, None] & d_mask[None, :]

    g_logit = tl.load((logits_ptr + h_start) + hs_i32, mask=h_mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.math.exp2(-g_logit * 1.4426950408889634))
    if round_gate_bf16:
        sig = _round_float32_to_bfloat16_ptx(sig)
    elif round_gate_fp16:
        sig = _round_float32_to_float16(sig)
    gate = (2.0 * sig)[:, None]

    base_off = h_start * dim_head
    tile_offs = hs_i32[:, None] * dim_head + ds_i32[None, :]
    attn = tl.load((attn_ptr + base_off) + tile_offs, mask=mask, other=0.0).to(tl.float32)
    tl.store((out_ptr + base_off) + tile_offs, attn * gate, mask=mask)


def head_sigmoid_gate(
    attention_output: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    dim_head: int,
) -> torch.Tensor:
    """Apply fused ``2.0 * sigmoid(gate_logits)`` per head to ``attention_output``."""

    del heads
    total_heads = int(gate_logits.numel())
    out_dtype = torch.promote_types(attention_output.dtype, gate_logits.dtype)
    output = torch.empty_like(attention_output, dtype=out_dtype)
    if total_heads == 0 or dim_head == 0:
        return output
    block_d = triton.next_power_of_2(dim_head)
    block_h = 16 if block_d <= 64 else (8 if block_d <= 256 else 4)
    num_warps = 4 if (block_h * block_d) <= 1024 else 8
    grid = (triton.cdiv(total_heads, block_h),)
    with torch.cuda.device(attention_output.device):
        _head_sigmoid_gate_kernel[grid](
            attention_output,
            gate_logits,
            output,
            total_heads,
            dim_head=int(dim_head),
            round_gate_bf16=gate_logits.dtype == torch.bfloat16,
            round_gate_fp16=gate_logits.dtype == torch.float16,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
    return output


__all__ = [
    "apply_real_rope_bshd",
    "fused_layer_norm_scale_shift",
    "head_sigmoid_gate",
    "hidden_qk_rmsnorm_rope_3d",
    "hidden_qk_rmsnorm_rope_freqs",
    "layer_norm_scale_shift",
    "qk_rmsnorm_rope",
    "residual_add_layer_norm_scale_shift",
    "residual_add_layer_norm_scale_shift_inplace",
    "residual_gate",
    "residual_gate_add_rms_norm",
    "residual_gate_inplace",
    "residual_gate_layer_norm",
    "residual_gate_layer_norm_inplace",
    "rms_norm_scale_shift",
    "scale_shift",
    "silu_and_mul",
    "silu_mul",
]
