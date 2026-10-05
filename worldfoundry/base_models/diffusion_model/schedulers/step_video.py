"""Stateless native StepVideo flow scheduler.

:class:`StepVideoFlowScheduler` builds a shifted linear sigma path and
integrates flow with Euler.  Only ``time_shift`` is accepted in
``scheduler_options``; other keys raise :exc:`ValueError`.  ``time_shift``
must be positive.
"""

from __future__ import annotations

import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep


class StepVideoFlowScheduler:
    """Euler flow integrator with StepVideo's rational time-shift."""
    def __init__(self, *, time_shift: float = 13.0, reverse: bool = False) -> None:
        self.time_shift = float(time_shift)
        self.reverse = bool(reverse)

    def schedule(self, sampling: SamplingConfig, *, device: torch.device, dtype: torch.dtype):
        del dtype
        shift = float(sampling.scheduler_options.get("time_shift", self.time_shift))
        if shift <= 0:
            raise ValueError("StepVideo time_shift must be positive")
        unsupported = set(sampling.scheduler_options) - {"time_shift"}
        if unsupported:
            raise ValueError(f"unsupported StepVideo scheduler options: {sorted(unsupported)}")
        sigmas = torch.linspace(1.0, 0.0, sampling.num_inference_steps + 1, device=device)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        if not self.reverse:
            sigmas = 1.0 - sigmas
        return tuple(
            SchedulerStep(index=index, timestep=sigmas[index], next_timestep=sigmas[index + 1])
            for index in range(sampling.num_inference_steps)
        )

    @staticmethod
    def scale_model_input(latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        del step
        return latents

    @staticmethod
    def step(model_output, step, latents, *, generator):
        del generator
        delta = (step.next_timestep - step.timestep).to(latents)
        return (latents.float() + model_output.float() * delta).to(latents.dtype)


def build_step_video_flow_scheduler(context: ComponentBuildContext) -> StepVideoFlowScheduler:
    return StepVideoFlowScheduler(
        time_shift=float(context.component_options.get("time_shift", 13.0)),
        reverse=bool(context.component_options.get("reverse", False)),
    )


__all__ = ["StepVideoFlowScheduler", "build_step_video_flow_scheduler"]
