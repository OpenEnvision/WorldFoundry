"""Stateless normalization helpers for tensor-like arrays.

These are *functions*, not ``nn.Module``s: no registered parameters, so they
can run on numpy or torch and sit next to fused kernels that already own
weights. :func:`rms_norm` prefers ``F.rms_norm`` when the value is a Tensor
(keeps AMP/kernel selection) and otherwise uses a portable mean-square path.
:func:`layer_scale` is the affine ``x * γ + β`` used after residual blocks
without constructing a :class:`~worldfoundry.core.nn.blocks.layers.LayerScale`.
"""

from __future__ import annotations

from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Stateless ops — no registered parameters; pair with fused kernels that own γ
# ──────────────────────────────────────────────────────────────────────────


def rms_norm(value: Any, weight: Any = None, *, eps: float = 1e-6) -> Any:
    """Apply RMS normalization over the last dimension."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return torch.nn.functional.rms_norm(
                value,
                (value.shape[-1],),
                weight=weight,
                eps=float(eps),
            )
    except ImportError:
        pass

    squared_mean = (value * value).mean(axis=-1, keepdims=True)
    normalized = value / ((squared_mean + float(eps)) ** 0.5)
    return normalized if weight is None else normalized * weight


def layer_scale(value: Any, scale: Any, *, bias: Any = None) -> Any:
    """Apply a stateless elementwise layer-scale transform."""

    output = value * scale
    return output if bias is None else output + bias


__all__ = ["layer_scale", "rms_norm"]
