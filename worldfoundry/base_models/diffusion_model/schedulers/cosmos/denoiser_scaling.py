# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EDM and rectified-flow scaling coefficients for Cosmos denoisers.

:class:`EDMScaling` returns the classic Karras ``(c_skip, c_out, c_in,
c_noise)`` tuple from a per-sample sigma, plus EDM loss weights.
:class:`RectifiedFlowScaling` maps sigma to the RF / flow-matching
coefficients (requires ``sigma_data == 1``).

Consumed by Predict1 residual samplers, not by the unified
DiffusionScheduler path.  Cosmos3 uses UniPC in ``schedulers/cosmos3``
instead.
"""

from typing import Tuple

import torch


class EDMScaling:
    """Edm scaling implementation."""

    def __init__(self, sigma_data: float = 0.5):
        """Init.

        Args:
            sigma_data: The sigma data.
        """
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call.

        Args:
            sigma: The sigma.

        Returns:
            The return value.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM preconditioning weights ``(σ² + σ_data²) / (σ σ_data)²``."""
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class RectifiedFlowScaling:
    """Scaling coefficients for rectified-flow inference."""

    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1.0):
        if abs(sigma_data - 1.0) >= 1e-6:
            raise ValueError("sigma_data must be 1.0 for RectifiedFlowScaling")
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = sigma / (sigma + 1)
        return 1.0 - t, -t, 1.0 - t, t * self.t_scaling_factor

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """Rectified-flow loss weights ``1 / σ²``."""
        return 1.0 / sigma**2
