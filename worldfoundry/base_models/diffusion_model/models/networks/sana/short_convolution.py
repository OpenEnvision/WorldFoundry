"""Native causal depthwise short convolution for SANA GDN blocks.

GDN applies a small causal conv on Q/K/V before the delta-rule scan (the
FLA ``ShortConvolution``).  This module is the inference-only equivalent
that loads the checkpoint ``weight`` of shape ``(C, 1, K)`` without pulling
the full Flash Linear Attention package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Module):
    """Inference-only equivalent of FLA's checkpoint-facing convolution.

    Sana checkpoints only persist ``weight`` with shape ``[C, 1, K]``.  The
    implementation therefore keeps that exact state-dict surface while using
    ordinary PyTorch depthwise convolution and explicit causal padding.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        activation=None,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or kernel_size <= 0:
            raise ValueError("hidden_size and kernel_size must be positive")
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None
        self.activation = activation
        nn.init.zeros_(self.weight)
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor, cache=None):
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                "ShortConvolution expects [batch, time, channels] with "
                f"channels={self.hidden_size}, got {tuple(x.shape)}"
            )
        temporal = x.transpose(1, 2)
        if cache is not None:
            if not isinstance(cache, torch.Tensor):
                raise TypeError("short-convolution cache must be a tensor")
            temporal = torch.cat((cache.to(temporal), temporal), dim=-1)
        context = temporal[..., -(self.kernel_size - 1) :].detach() if self.kernel_size > 1 else None
        temporal = F.pad(temporal, (self.kernel_size - 1, 0))
        output = F.conv1d(
            temporal,
            self.weight.to(dtype=temporal.dtype),
            self.bias.to(dtype=temporal.dtype) if self.bias is not None else None,
            groups=self.hidden_size,
        )
        output = output[..., -x.shape[1] :].transpose(1, 2)
        if self.activation is not None:
            output = self.activation(output)
        return output, context


__all__ = ["ShortConvolution"]
