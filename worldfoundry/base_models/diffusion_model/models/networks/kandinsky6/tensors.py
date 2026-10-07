from __future__ import annotations

import math

import torch
from torch import Tensor


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Return inverse frequencies for rotary position embeddings.

    Args:
        dim (`int`): Number of frequency values to generate.
        max_period (`float`, *optional*, defaults to 10000.0): Maximum period
            used by the frequency schedule.

    Returns:
        `torch.Tensor`: Frequency values in float32.
    """
    return torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )


def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """AdaLN-style affine in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Residual gate in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)
