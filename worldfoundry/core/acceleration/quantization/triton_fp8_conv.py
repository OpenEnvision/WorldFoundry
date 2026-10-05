"""Calibrated FP8 implicit-GEMM convolutions, with FP32 accumulation/output.

Channel scales are folded into the exported weights. Input patches are
quantized inside the GEMM; no im2col buffer or dense dequantized weight exists.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(
    X,
    W,
    ScaleX,
    ScaleW,
    Bias,
    Out,
    Clipped,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DT: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    total_m: tl.constexpr = B * OT * OH * OW
    total_k: tl.constexpr = C * KT * KH * KW
    batch = m // (OT * OH * OW)
    tt, hh, ww = (m // (OH * OW)) % OT, (m // OW) % OH, m % OW
    accumulator = tl.full((BM, BN), 0, tl.float32)
    clipped = tl.full((BM, BK), 0, tl.int32)
    for block in range(tl.cdiv(total_k, BK)):
        k = block * BK + tl.arange(0, BK)
        channel = k // (KT * KH * KW)
        it = tt[:, None] * ST + ((k // (KH * KW)) % KT)[None, :] * DT - PT
        ih = hh[:, None] * SH + ((k // KW) % KH)[None, :] * DH - PH
        iw = ww[:, None] * SW + (k % KW)[None, :] * DW - PW
        valid = (
            (m[:, None] < total_m)
            & (k[None, :] < total_k)
            & (it >= 0)
            & (it < T)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < WIDTH)
        )
        offsets = batch[:, None] * XS0 + channel[None, :] * XS1 + it * XS2 + ih * XS3 + iw * XS4
        x = tl.load(X + offsets, valid, 0).to(tl.float32)
        scale_x = tl.load(ScaleX + channel, channel < C, 1)
        x = tl.div_rn(x, scale_x[None, :])
        clipped += (tl.abs(x) > 448).to(tl.int32)
        x = tl.minimum(tl.maximum(x, -448), 448).to(tl.float8e4nv)
        weight = tl.load(W + n[None, :] * total_k + k[:, None], (n[None, :] < N) & (k[:, None] < total_k), 0.0)
        accumulator += tl.dot(x, weight)
    scale_w = tl.load(ScaleW + n, n < N, 0)
    accumulator *= scale_w[None, :]
    if HAS_BIAS:
        accumulator += tl.load(Bias + n, n < N, 0)[None, :]
    # The output is NCTHW, retaining the codec's canonical layout.
    offsets = (batch[:, None] * N + n[None, :]) * (OT * OH * OW) + (m % (OT * OH * OW))[:, None]
    tl.store(Out + offsets, accumulator, (m[:, None] < total_m) & (n[None, :] < N))
    # Count input operands once, independent of the number of output tiles.
    if tl.program_id(1) == 0:
        tl.atomic_add(Clipped, tl.sum(tl.sum(clipped.to(tl.int64), 0), 0))


def fp8_convolution(input, weight, input_scale, weight_scale, bias, *, stride, padding, dilation, clipped):
    is_2d = input.ndim == 4
    x = input.unsqueeze(2) if is_2d else input
    kernel = (1, *weight.shape[2:]) if is_2d else weight.shape[2:]
    strides = (1, *stride) if is_2d else stride
    pads = (0, *padding) if is_2d else padding
    dilations = (1, *dilation) if is_2d else dilation
    b, c, t, h, w = x.shape
    n = weight.shape[0]
    geometry = tuple(
        (dim + 2 * pad - dil * (size - 1) - 1) // step + 1
        for dim, size, step, pad, dil in zip((t, h, w), kernel, strides, pads, dilations)
    )
    if min(geometry) <= 0:
        raise ValueError("FP8 convolution has an empty output geometry")
    output = torch.empty((b, n, *geometry), device=x.device, dtype=x.dtype)
    count = b * geometry[0] * geometry[1] * geometry[2]
    _conv[(triton.cdiv(count, 32), triton.cdiv(n, 32))](
        x,
        weight,
        input_scale,
        weight_scale,
        bias,
        output,
        clipped,
        b,
        c,
        t,
        h,
        w,
        n,
        *kernel,
        *geometry,
        *strides,
        *pads,
        *dilations,
        *x.stride(),
        bias is not None,
        num_warps=4,
    )
    return output.squeeze(2) if is_2d else output
