# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless Hopper SM90 Triton kernels for Cosmos3 AVAE audio decoder.

Fuses the 8-kernel ``Snake1d`` periodic activation::

    y = x + (exp(beta) + 1e-9)^(-1) * sin(exp(alpha) * x)^2

into a single coalesced HBM pass with channel-scalar ``exp(alpha)`` and
``1 / (exp(beta) + 1e-9)`` hoisted into registers per CTA. All ops use strict
``int64`` pointer arithmetic, register ``@torch.library.custom_op`` +
``.register_fake`` entries for ``torch.compile`` / CUDA Graphs, and import
Triton lazily so CPU-only environments stay fast and clean.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@lru_cache(maxsize=1)
def _load_triton_kernels() -> dict[str, Any]:
    """Lazily compile Triton kernels on first eligible CUDA call."""
    from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

    configure_persistent_compile_cache(namespace="cosmos3-audio-triton")

    import triton
    import triton.language as tl

    @triton.jit
    def _cosmos3_snake1d_kernel(
        x_ptr,
        alpha_ptr,
        beta_ptr,
        out_ptr,
        num_channels,
        length,
        stride_b,
        stride_c,
        out_stride_b,
        out_stride_c,
        block_l: tl.constexpr,
    ):
        """Single-pass coalesced ``x + inv(exp(beta) + 1e-9) * sin(exp(alpha) * x)^2``."""
        pid_l = tl.program_id(0).to(tl.int64)
        bc = tl.program_id(1).to(tl.int64)
        b = bc // num_channels
        c = bc % num_channels

        alpha_val = tl.exp(tl.load(alpha_ptr + c).to(tl.float32))
        beta_val = tl.exp(tl.load(beta_ptr + c).to(tl.float32))
        inv_beta = 1.0 / (beta_val + 1e-9)

        offs_l = pid_l * block_l + tl.arange(0, block_l)
        mask = offs_l < length
        offs_l64 = offs_l.to(tl.int64)

        in_base = b * stride_b + c * stride_c
        out_base = b * out_stride_b + c * out_stride_c
        x = tl.load(x_ptr + in_base + offs_l64, mask=mask, other=0.0).to(tl.float32)
        s = tl.sin(alpha_val * x)
        y = (x + inv_beta * (s * s)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + out_base + offs_l64, y, mask=mask)

    return {
        "triton": triton,
        "snake1d": _cosmos3_snake1d_kernel,
    }


@torch.library.custom_op("worldfoundry::cosmos3_snake1d_fwd", mutates_args=(), device_types="cuda")
def _cosmos3_snake1d_fwd_op(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    kernels = _load_triton_kernels()
    triton = kernels["triton"]

    b, c, length = x.shape
    out = torch.empty((b, c, length), device=x.device, dtype=x.dtype)
    if b == 0 or c == 0 or length == 0:
        return out

    x_c = x if x.stride(-1) == 1 else x.contiguous()
    alpha_flat = alpha.reshape(c).contiguous()
    beta_flat = beta.reshape(c).contiguous()

    block_l = 1024 if length >= 1024 else max(64, triton.next_power_of_2(length))
    num_warps = 8 if block_l >= 1024 else 4
    grid = (triton.cdiv(length, block_l), b * c)
    kernels["snake1d"][grid](
        x_c,
        alpha_flat,
        beta_flat,
        out,
        c,
        length,
        x_c.stride(0),
        x_c.stride(1),
        out.stride(0),
        out.stride(1),
        block_l=block_l,
        num_warps=num_warps,
    )
    return out


@_cosmos3_snake1d_fwd_op.register_fake
def _cosmos3_snake1d_fwd_fake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    del alpha, beta
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def can_use_cosmos3_snake1d(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> bool:
    """Return True when the fused Triton ``Snake1d`` op is eligible."""
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    if x.ndim != 3 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    c = int(x.shape[1])
    if c <= 0:
        return False
    return (
        alpha.device == x.device
        and beta.device == x.device
        and alpha.numel() == c
        and beta.numel() == c
        and alpha.dtype in _SUPPORTED_DTYPES
        and beta.dtype in _SUPPORTED_DTYPES
    )


def fused_cosmos3_snake1d(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Run lossless single-pass Hopper Triton ``Snake1d`` activation."""
    return _cosmos3_snake1d_fwd_op(x, alpha, beta)


__all__ = ["can_use_cosmos3_snake1d", "fused_cosmos3_snake1d"]
