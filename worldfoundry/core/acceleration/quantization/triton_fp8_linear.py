"""Masked FP8 tensor-core GEMM with row scales and optional FP32 bias epilogue."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, W, AS, WS, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    acc = tl.full((32, 64), 0, tl.float32)
    for block in range(tl.cdiv(K, 64)):
        kk = block * 64 + tl.arange(0, 64)
        a = tl.load(A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0.0)
        w = tl.load(W + cols[None, :] * K + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), 0.0)
        acc = tl.dot(a, w, acc)
    sa = tl.load(AS + rows, rows < M, 0)
    sw = tl.load(WS + cols, cols < N, 0)
    acc = (acc * sa[:, None]) * sw[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, 0)[None, :].to(tl.float32)
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def fp8_linear(input, weight, input_scale, weight_scale, bias, output_dtype):
    m, k = input.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=input.device, dtype=output_dtype)
    _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        input, weight, input_scale, weight_scale, bias, output, m, n, k, bias is not None, num_warps=4
    )
    return output
