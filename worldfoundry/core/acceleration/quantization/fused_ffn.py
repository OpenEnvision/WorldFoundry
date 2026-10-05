"""Opt-in FP8 Linear/GELU/Linear with fused activation and row quantization.

Preserves Sequential checkpoint keys and calibrated dense dispatch. The second
GEMM consumes FP8 activations directly; its BF16 GELU intermediate is omitted.
Like FP8 Linear itself this path needs a checkpoint quality gate before use.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .linear import Float8Linear, _fp8_linear_eligible, _is_compiling


class FusedFP8GELUFeedForward(nn.Sequential):
    def __init__(self, original: nn.Sequential) -> None:
        super().__init__(OrderedDict(original._modules.items()))
        self.fused_calls = 0
        self.fallback_calls = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        up, activation, down = self[0], self[1], self[2]
        hidden = up(input)
        eligible = (
            down.low_precision_enabled
            and down.scaling == "rowwise"
            and _fp8_linear_eligible(
                hidden, down.out_features, down._hardware_eligible, min_gemm_work=0 if down.weight is None else None
            )
        )
        if not eligible:
            if not _is_compiling():
                self.fallback_calls += 1
            return down(activation(hidden))
        # Import lazily so CPU/docs environments do not need Triton.
        try:
            from .triton_fp8 import quantize_rowwise_fp8_triton
        except ImportError:
            if not _is_compiling():
                self.fallback_calls += 1
            return down(activation(hidden))
        mode = "gelu_tanh" if activation.approximate == "tanh" else "gelu"
        codes, scales = quantize_rowwise_fp8_triton(
            hidden.reshape(-1, down.in_features), down.fp8_dtype, activation=mode
        )
        result = down._forward_quantized(codes, scales, hidden.shape, hidden.dtype)
        if not _is_compiling():
            self.fused_calls += 1
        return result


def fuse_fp8_gelu_feed_forwards(model: nn.Module) -> int:
    """Fuse exact three-layer patterns after FP8 weight conversion, once."""
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, FusedFP8GELUFeedForward):
            continue
        if (
            type(child) is nn.Sequential
            and len(child) == 3
            and isinstance(child[0], Float8Linear)
            and isinstance(child[1], nn.GELU)
            and isinstance(child[2], Float8Linear)
            and child[0].out_features == child[2].in_features
            and child[2].scaling == "rowwise"
        ):
            setattr(model, name, FusedFP8GELUFeedForward(child))
            count += 1
        else:
            count += fuse_fp8_gelu_feed_forwards(child)
    return count


__all__ = ["FusedFP8GELUFeedForward", "fuse_fp8_gelu_feed_forwards"]
