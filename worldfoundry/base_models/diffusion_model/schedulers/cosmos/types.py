# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Cosmos Predict1 prediction and condition dataclasses.

:class:`DenoisePrediction` holds the residual-sampler outputs
(``x0``, optional ``eps`` and ``logvar``).  :class:`LabelImageCondition`
is the class-conditional label tensor plus a zeroed CFG twin.

These types are consumed by :mod:`.res_sampler` and family adapters,
not by the unified NativeDiffusionRunner path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LabelImageCondition:
    """Label image condition implementation."""

    label: torch.Tensor

    def get_classifier_free_guidance_condition(self) -> "LabelImageCondition":
        """Get classifier free guidance condition.

        Returns:
            The return value.
        """
        return LabelImageCondition(torch.zeros_like(self.label))


@dataclass
class DenoisePrediction:
    """Denoise prediction implementation."""

    x0: torch.Tensor
    eps: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
