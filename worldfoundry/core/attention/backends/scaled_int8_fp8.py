# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-aware INT8 QK / FP8 PV attention for unmasked inference.

Q/K scales are per token block and head; V scales are per head and channel.
K centering is softmax-invariant. Online softmax and accumulation remain FP32.
This explicit approximate provider never substitutes a dense attention kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_qk(
    x,
    mean,
    out,
    scales,
    length: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block: tl.constexpr,
    centered: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    count = tl.cdiv(length, block)
    batch = row // (heads * count)
    token = ((row // heads) % count) * block + tl.arange(0, block)
    feature = tl.arange(0, dim)
    mask = token[:, None] < length
    offset = ((batch * length + token[:, None]) * heads + head) * dim + feature[None, :]
    values = tl.load(x + offset, mask=mask, other=0).to(tl.float32)
    if centered:
        values -= tl.load(mean + (batch * heads + head) * dim + feature)[None, :]
        values = tl.where(mask, values, 0)
    scale = tl.maximum(tl.max(tl.max(tl.abs(values), axis=1), axis=0) / 127.0, 1.1754943508222875e-38)
    quantized = tl.extra.cuda.libdevice.rint(values / scale)
    tl.store(out + offset, tl.minimum(127.0, tl.maximum(-127.0, quantized)), mask=mask)
    tl.store(scales + row, scale)


@triton.jit
def _attention(
    q,
    k,
    v,
    qs,
    ks,
    vs,
    out,
    length: tl.constexpr,
    context: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    scale: tl.constexpr,
    qblock: tl.constexpr,
    kblock: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    query_tile = tl.program_id(0)
    plane = tl.program_id(1)
    batch, head = plane // heads, plane % heads
    query_rows = query_tile * M + tl.arange(0, M)
    features = tl.arange(0, dim)
    query = tl.load(
        q + ((batch * length + query_rows[:, None]) * heads + head) * dim + features[None, :],
        mask=query_rows[:, None] < length,
        other=0,
    )
    query_scale = tl.load(
        qs + (batch * tl.cdiv(length, qblock) + query_rows // qblock) * heads + head, mask=query_rows < length, other=0
    )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    accumulator = tl.zeros((M, dim), tl.float32)
    for start in tl.range(0, context, N):
        rows = start + tl.arange(0, N)
        offsets = ((batch * context + rows[:, None]) * heads + head) * dim + features[None, :]
        keys = tl.load(k + offsets, mask=rows[:, None] < context, other=0)
        key_scale = tl.load(
            ks + (batch * tl.cdiv(context, kblock) + rows // kblock) * heads + head, mask=rows < context, other=0
        )
        scores = tl.dot(query, tl.trans(keys)).to(tl.float32)
        scores *= query_scale[:, None] * key_scale[None, :] * (scale * 1.4426950408889634)
        scores = tl.where(rows[None, :] < context, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        correction = tl.exp2(maximum - next_maximum)
        probability = tl.exp2(scores - next_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)
        values = tl.load(v + offsets, mask=rows[:, None] < context, other=0.0)
        accumulator *= correction[:, None]
        accumulator = tl.dot((probability * 128.0).to(tl.float8e4nv), values, accumulator)
        maximum = next_maximum
    value_scale = tl.load(vs + (batch * heads + head) * dim + features)
    result = accumulator / denominator[:, None] * (value_scale[None, :] / 128.0)
    tl.store(
        out + ((batch * length + query_rows[:, None]) * heads + head) * dim + features[None, :],
        result,
        mask=query_rows[:, None] < length,
    )


def _quantize(input, mean, block):
    batch, length, heads, dim = input.shape
    quantized = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((batch, triton.cdiv(length, block), heads), device=input.device, dtype=torch.float32)
    _quantize_qk[(scales.numel(),)](
        input,
        input if mean is None else mean,
        quantized,
        scales,
        length,
        heads,
        dim,
        block,
        mean is not None,
        num_warps=4,
    )
    return quantized, scales


@torch.compiler.disable
def scaled_int8_fp8_attention(query, key, value, *, scale=None, key_mean=None, qk_quantization_blocks=(128, 64)):
    """Compute approximate attention on CUDA FP16/BF16 ``[B,L,H,D]`` inputs.

    ``key_mean`` may be a request-owned FP16/BF16 ``[B,1,H,D]`` tensor to reuse
    finalized history reductions in a streaming model. Masks, GQA, backward,
    and CUDA Graph capture are not part of this provider's contract.
    """
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("scaled_int8_fp8 requires [B,L,H,D] tensors")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("scaled_int8_fp8 requires matching batch/head/dim and K/V shapes")
    if any(tensor.device != query.device or tensor.dtype != query.dtype for tensor in (key, value)):
        raise ValueError("scaled_int8_fp8 requires uniform placement and dtype")
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("scaled_int8_fp8 requires CUDA FP16/BF16 tensors")
    if torch.version.hip or torch.cuda.get_device_capability(query.device)[0] < 9:
        raise ValueError("scaled_int8_fp8 requires NVIDIA SM90 or newer")
    if torch.is_grad_enabled() or any(tensor.requires_grad for tensor in (query, key, value)):
        raise ValueError("scaled_int8_fp8 is inference-only")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("scaled_int8_fp8 CUDA Graph capture is unvalidated")
    dim = query.shape[-1]
    if dim not in (64, 128, 256) or min(*query.shape, key.shape[1]) < 1:
        raise ValueError("scaled_int8_fp8 requires nonempty inputs and head dim 64/128/256")
    if len(qk_quantization_blocks) != 2 or any(
        type(n) is not int or n < 1 or n > 128 or n & (n - 1) for n in qk_quantization_blocks
    ):
        raise ValueError("Q/K quantization blocks must be powers of two in [1,128]")
    scale = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be positive and finite")
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    if key_mean is not None and (
        key_mean.shape != (key.shape[0], 1, *key.shape[2:])
        or key_mean.dtype != key.dtype
        or key_mean.device != key.device
    ):
        raise ValueError("key_mean must match key batch, heads, dtype and device")
    mean = key.mean(dim=1, keepdim=True) if key_mean is None else key_mean.contiguous()
    qblock, kblock = qk_quantization_blocks
    q, qs = _quantize(query, None, qblock)
    k, ks = _quantize(key, mean, kblock)
    value_scale = value.float().abs().amax(dim=1, keepdim=True).div(448.0).clamp_min(torch.finfo(torch.float32).tiny)
    v = (value.float() / value_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    output = torch.empty_like(query)
    _attention[(triton.cdiv(query.shape[1], 64), query.shape[0] * query.shape[2])](
        q,
        k,
        v,
        qs,
        ks,
        value_scale,
        output,
        query.shape[1],
        key.shape[1],
        query.shape[2],
        dim,
        scale,
        qblock,
        kblock,
        M=64,
        N=64,
        num_warps=4,
        num_stages=3,
    )
    return output


__all__ = ["scaled_int8_fp8_attention"]
