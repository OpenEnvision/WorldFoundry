# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Coefficient placement follows the Diffusers/Wan UniPC implementation (Apache-2.0).
"""UniPC variant that constructs and solves coefficients on the sample device."""

from __future__ import annotations

import torch

from .flow_unipc import FlowUniPCMultistepScheduler


class DeviceCoefficientFlowUniPCMultistepScheduler(FlowUniPCMultistepScheduler):
    """Keep device-side coefficients while sharing the native UniPC update math."""

    def _coefficient_ratios(self, rks: list[torch.Tensor], h: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.tensor([*rks, 1.0], device=device)

    def _coefficient_rhs(self, values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, device=device)

    def _solve_coefficients(
        self, matrix: torch.Tensor, rhs: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.linalg.solve(matrix, rhs).to(device).to(dtype)
