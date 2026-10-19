# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Stateless DDIM and unscaled Euler updates for SD2/Marigold.

Schedules use Diffusers' numerical schedule primitive, not its pipeline or
stateful sampling loop. Each native SchedulerStep carries its own interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep
from ..loaders import NativeCheckpointResolver, checkpoint_json_config

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler


class SDDiscreteScheduler:
    """Deterministic DDIM or WonderWorld's unscaled Marigold Euler scheme."""

    def __init__(self, config: Mapping[str, object], *, algorithm: str) -> None:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

        if algorithm not in {"ddim", "euler-unscaled"}:
            raise ValueError("unsupported SD discrete scheduler")
        self.algorithm = algorithm
        self.config = dict(config)
        self.scheduler_class = DDIMScheduler if algorithm == "ddim" else EulerDiscreteScheduler
        numerical = cast("DDIMScheduler | EulerDiscreteScheduler", self.scheduler_class.from_config(self.config))
        self.alphas = numerical.alphas_cumprod
        self.training_sigmas = ((1 - self.alphas) / self.alphas).sqrt().numpy()
        self.prediction_type = str(numerical.config["prediction_type"])
        self.final_alpha = torch.tensor(1.0) if self.config.get("set_alpha_to_one", True) else self.alphas[0]
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("SD discrete scheduler requires epsilon or v_prediction")
        if self.config.get("clip_sample", False) or self.config.get("thresholding", False):
            raise ValueError("SD2/Marigold checkpoints require unclipped latent predictions")

    def schedule(
        self, sampling: SamplingConfig, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[SchedulerStep, ...]:
        """Construct an independent trajectory without retaining a step index."""
        del dtype
        if sampling.scheduler_options:
            raise ValueError("SD discrete sampling does not accept scheduler overrides")
        if sampling.num_inference_steps > len(self.alphas):
            raise ValueError("inference steps cannot exceed training timesteps")
        numerical = cast("DDIMScheduler | EulerDiscreteScheduler", self.scheduler_class.from_config(self.config))
        numerical.set_timesteps(sampling.num_inference_steps, device=device)
        times = numerical.timesteps
        if self.algorithm == "ddim" and int(times[0]) >= len(self.alphas):
            raise ValueError("DDIM schedule exceeds training timesteps; reduce num_inference_steps")
        return tuple(
            SchedulerStep(i, t, times[i + 1] if i + 1 < len(times) else t.new_tensor(-1)) for i, t in enumerate(times)
        )

    @staticmethod
    def scale_model_input(latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        """WonderWorld does not apply Euler input scaling to Marigold latents."""
        del step
        return latents

    def _sigma(self, timestep: torch.Tensor) -> float:
        value = float(timestep)
        if value < 0:
            return 0.0
        return float(np.float32(np.interp(value, np.arange(len(self.training_sigmas)), self.training_sigmas)))

    def step(
        self, model_output: torch.Tensor, step: SchedulerStep, latents: torch.Tensor, *, generator: torch.Generator
    ) -> torch.Tensor:
        """Update this interval only; repeated guidance probes cannot advance it."""
        del generator
        if self.algorithm == "euler-unscaled":
            sample = latents.float()
            sigma = torch.tensor(self._sigma(step.timestep))
            following = torch.tensor(self._sigma(step.next_timestep))
            if self.prediction_type == "epsilon":
                clean = sample - sigma * model_output
            else:
                clean = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + sample / (sigma**2 + 1)
            derivative = (sample - clean) / sigma
            return (sample + derivative * (following - sigma)).to(dtype=model_output.dtype)
        # Released schedules retain FP32 CPU scalars, including for CUDA
        # latents; scalar promotion is part of the BF16 sampling contract.
        alpha = self.alphas[int(step.timestep)]
        next_alpha = self.alphas[int(step.next_timestep)] if int(step.next_timestep) >= 0 else self.final_alpha
        beta = 1 - alpha
        if self.prediction_type == "epsilon":
            clean = (latents - beta**0.5 * model_output) / alpha**0.5
            epsilon = model_output
        else:
            clean = alpha**0.5 * latents - beta**0.5 * model_output
            epsilon = alpha**0.5 * model_output + beta**0.5 * latents
        return next_alpha**0.5 * clean + (1 - next_alpha) ** 0.5 * epsilon


def build_sd_discrete_scheduler(context: ComponentBuildContext) -> SDDiscreteScheduler:
    """Read the component's explicit numerical scheduler resource."""
    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint())
    config = checkpoint_json_config(checkpoint, "scheduler/scheduler_config.json")
    return SDDiscreteScheduler(config, algorithm=str(context.component_options["algorithm"]))
