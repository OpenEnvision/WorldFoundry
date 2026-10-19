# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Few-step TrigFlow sampling with fixed original-noise renoising.

Adapted from NVIDIA/DreamDojo at 02f119b759d5c7f84a399fdeea3c6e82e7ed6cff,
cosmos_predict2/_src/predict2/interactive/models/action_video2world_self_forcing.py.
"""

from __future__ import annotations

import math

import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep

SAMPLING_TIMES = (math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3))


class TrigFlowScheduler:
    """Use the selected schedule prefix and the same noise at every hop.

    ``step`` receives the original frame noise as ``latents`` and a clean x0
    prediction as ``model_output``. This is not an Euler velocity update.
    """

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[SchedulerStep, ...]:
        """Return 1–4 released student times rounded to BF16, as upstream does."""
        del dtype
        count = sampling.num_inference_steps
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(SAMPLING_TIMES):
            raise ValueError("DreamDojo student num_inference_steps must be an integer from 1 to 4")
        if sampling.guidance_scale != 1.0 or sampling.scheduler_options:
            raise ValueError("DreamDojo student uses distilled guidance and a fixed sampling schedule")
        times = torch.tensor((*SAMPLING_TIMES[:count], 0.0), device=device, dtype=torch.bfloat16)
        return tuple(
            SchedulerStep(index=index, timestep=times[index], next_timestep=times[index + 1]) for index in range(count)
        )

    def scale_model_input(self, latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        """Preconditioning belongs to the denoiser; preserve the sampler state."""
        del step
        return latents

    def step(
        self,
        model_output: torch.Tensor,
        step: SchedulerStep,
        latents: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Renoise with the original noise; the last hop returns x0 unchanged."""
        del generator
        return step.next_timestep.cos() * model_output + step.next_timestep.sin() * latents


def build_trigflow_scheduler(context: ComponentBuildContext) -> TrigFlowScheduler:
    """Build the stateless released four-time student schedule."""
    del context
    return TrigFlowScheduler()
