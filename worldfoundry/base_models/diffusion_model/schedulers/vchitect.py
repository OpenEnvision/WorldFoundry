"""The released Vchitect-2 FlowMatchEuler timetable."""

from __future__ import annotations

import numpy as np
import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep
from .wan import WanFlowMatchEulerScheduler, shift_flow_sigmas


class VchitectFlowMatchEulerScheduler(WanFlowMatchEulerScheduler):
    """Match the released Diffusers scheduler, including its shifted endpoint."""

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        if sampling.scheduler_options:
            raise ValueError("Vchitect does not support scheduler_options overrides")
        count = sampling.num_inference_steps
        if count <= 0:
            raise ValueError("num_inference_steps must be positive")

        # Diffusers initializes sigma_min from a shifted training timestep of 1,
        # then shifts the inference linspace again in set_timesteps().
        minimum = shift_flow_sigmas(
            torch.tensor(1.0 / self.num_train_timesteps, dtype=torch.float32), self.shift
        ).item()
        # Diffusers uses NumPy's float64 linspace, then shifts before the float32 cast.
        base = np.linspace(
            float(self.num_train_timesteps), minimum * self.num_train_timesteps, count
        ) / self.num_train_timesteps
        shifted = self.shift * base / (1.0 + (self.shift - 1.0) * base)
        sigmas = torch.from_numpy(shifted).to(device=device, dtype=torch.float32)
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        timesteps = sigmas * self.num_train_timesteps
        return tuple(
            SchedulerStep(index=index, timestep=timesteps[index], next_timestep=timesteps[index + 1])
            for index in range(count)
        )

    def step(
        self,
        model_output: torch.Tensor,
        step: SchedulerStep,
        latents: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Follow Diffusers' float32 Euler update and output dtype cast."""
        del generator
        delta = (step.next_timestep - step.timestep) / self.num_train_timesteps
        return (latents.float() + delta * model_output).to(model_output.dtype)


def build_vchitect_flow_match_euler_scheduler(
    context: ComponentBuildContext,
) -> VchitectFlowMatchEulerScheduler:
    options = context.component_options
    return VchitectFlowMatchEulerScheduler(
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        shift=float(options.get("shift", 3.0)),
    )


__all__ = ["VchitectFlowMatchEulerScheduler", "build_vchitect_flow_match_euler_scheduler"]
