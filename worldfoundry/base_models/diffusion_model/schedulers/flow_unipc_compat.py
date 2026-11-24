# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Diffusers serialization for the native device-coefficient UniPC variant."""

from __future__ import annotations

from inspect import unwrap
from typing import ClassVar

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

from .flow_unipc_device import DeviceCoefficientFlowUniPCMultistepScheduler


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin, DeviceCoefficientFlowUniPCMultistepScheduler):
    """Load Diffusers scheduler configs without duplicating the native solver."""

    _compatibles: ClassVar[list[str]] = [scheduler.name for scheduler in KarrasDiffusionSchedulers]

    # Use Diffusers config capture with the native parameter/state initializer.
    __init__ = register_to_config(unwrap(DeviceCoefficientFlowUniPCMultistepScheduler.__init__))

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        previous = super().step(model_output, timestep, sample, return_dict=False, generator=generator)[0]
        return SchedulerOutput(prev_sample=previous) if return_dict else (previous,)
