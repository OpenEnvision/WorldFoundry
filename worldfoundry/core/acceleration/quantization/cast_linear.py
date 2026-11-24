# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Unscaled FP8 weight storage with dense computation in the input dtype.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Both weight and bias use deterministic E4M3
rounding, without activation quantization or scale calibration.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class Float8CastLinear(nn.Module):
    """Store ``[out, in]`` weights and bias in FP8; cast before ``F.linear``."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self._worldfoundry_quantization_layer = True
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.register_buffer("weight", weight.detach().to(torch.float8_e4m3fn))
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.float8_e4m3fn))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run dense GEMM on the deterministically rounded weight and bias."""
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, self.weight.to(dtype=input.dtype), bias)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Float8CastLinear:
        """Follow device moves while retaining the FP8 storage dtype."""
        weight, bias = self.weight, self.bias
        super()._apply(fn, recurse=recurse)
        device = self.weight.device
        self.weight = weight.to(device=device)
        self.bias = None if bias is None else bias.to(device=device)
        return self
