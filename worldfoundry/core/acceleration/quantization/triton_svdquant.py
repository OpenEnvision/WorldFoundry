"""Packed signed INT4 GEMM with group scales and a fused low-rank epilogue.

The packed nibbles are unpacked into integer registers for tensor-core INT8
dot products. No dense floating-point weight matrix is materialized. This is
an independent SVDQuant provider, not Nunchaku's binary weight layout.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Smooth, Packed, Scales, M: tl.constexpr, K: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    half = tl.arange(0, G // 2)
    col = group * G + half * 2
    even = tl.div_rn(tl.load(X + row * K + col).to(tl.float32), tl.load(Smooth + col))
    odd = tl.div_rn(tl.load(X + row * K + col + 1).to(tl.float32), tl.load(Smooth + col + 1))
    scale = tl.maximum(tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), 0) / 7.0, 1.0e-8)
    low = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(tl.div_rn(even, scale)), -7), 7).to(tl.int32) & 15
    high = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(tl.div_rn(odd, scale)), -7), 7).to(tl.int32) & 15
    tl.store(Packed + row * (K // 2) + group * (G // 2) + half, low | (high << 4))
    tl.store(Scales + row * (K // G) + group, scale)


@triton.jit
def _gemm(
    A,
    AS,
    W,
    WS,
    Low,
    Up,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    G: tl.constexpr = 64,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    within = tl.arange(0, G)
    total = tl.full((BM, BN), 0, tl.float32)
    for group in range(K // G):
        kk = group * G + within
        a = tl.load(A + rows[:, None] * (K // 2) + kk[None, :] // 2, rows[:, None] < M, 0).to(tl.int32)
        w = tl.load(W + cols[None, :] * (K // 2) + kk[:, None] // 2, cols[None, :] < N, 0).to(tl.int32)
        a = (a >> ((kk[None, :] % 2) * 4)) & 15
        w = (w >> ((kk[:, None] % 2) * 4)) & 15
        a = tl.where(a >= 8, a - 16, a).to(tl.int8)
        w = tl.where(w >= 8, w - 16, w).to(tl.int8)
        product = tl.dot(a, w).to(tl.float32)
        sa = tl.load(AS + rows * (K // G) + group, rows < M, 0)
        sw = tl.load(WS + cols * (K // G) + group, cols < N, 0)
        total += product * sa[:, None] * sw[None, :]
    rr = tl.arange(0, triton.next_power_of_2(R))
    low = tl.load(Low + rows[:, None] * R + rr[None, :], (rows[:, None] < M) & (rr[None, :] < R), 0)
    up = tl.load(Up + cols[None, :] * R + rr[:, None], (cols[None, :] < N) & (rr[:, None] < R), 0)
    total += tl.dot(low, up).to(tl.float32)
    if HAS_BIAS:
        total += tl.load(Bias + cols, cols < N, 0)[None, :]
    tl.store(Out + rows[:, None] * N + cols[None, :], total, (rows[:, None] < M) & (cols[None, :] < N))


def packed_svdquant(input, smooth, qweight, scales, down, up, bias):
    shape = input.shape
    x = input.reshape(-1, shape[-1]).contiguous()
    m, k = x.shape
    n, rank = up.shape
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    activation_scales = torch.empty((m, k // 64), dtype=torch.float32, device=x.device)
    _quantize[(m, k // 64)](x, smooth, packed, activation_scales, m, k, 64, num_warps=4, enable_fp_fusion=False)
    # Down projection is expressed against the original input; the offline
    # export has already divided its columns by smoothing factors.
    low = torch.nn.functional.linear(x, down)
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        packed,
        activation_scales,
        qweight,
        scales,
        low,
        up,
        bias,
        output,
        m,
        n,
        k,
        rank,
        bias is not None,
        num_warps=4,
    )
    return output.reshape(*shape[:-1], n)
