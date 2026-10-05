"""Stateless Euler integration for rectified-flow and flow-matching models.

Generic :class:`~..contracts.DiffusionScheduler` used when a family does not
need Wan's rational shift.  ``schedule`` emits a descending linear sigma
path of length ``N`` (plus a terminal point).  ``step`` is
``latents + (σ_{next} - σ_t) * model_output``.  The instance holds no
per-run state and may be reused across requests.

Fails when ``num_inference_steps <= 0``, ``sigma_max <= sigma_min``, or
``scheduler_options`` contains keys other than ``sigma_max`` / ``sigma_min``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..contracts import SamplingConfig, SchedulerStep


class FlowMatchEulerScheduler:
    """Build and integrate a descending flow-matching trajectory.

    The scheduler owns no per-run mutable state.  A runner may therefore reuse
    one scheduler instance safely across sequential requests.
    """

    def __init__(
        self,
        *,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        timestep_dtype: torch.dtype = torch.float32,
    ) -> None:
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if not timestep_dtype.is_floating_point:
            raise ValueError("timestep_dtype must be floating point")
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.timestep_dtype = timestep_dtype

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        num_inference_steps = sampling.num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        unsupported_options = set(sampling.scheduler_options) - {
            "sigma_max",
            "sigma_min",
        }
        if unsupported_options:
            raise ValueError(f"unsupported FlowMatchEulerScheduler options: {sorted(unsupported_options)}")
        sigma_max = float(sampling.scheduler_options.get("sigma_max", self.sigma_max))
        sigma_min = float(sampling.scheduler_options.get("sigma_min", self.sigma_min))
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        timesteps = torch.linspace(
            sigma_max,
            sigma_min,
            num_inference_steps + 1,
            device=device,
            dtype=self.timestep_dtype,
        )
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timesteps[index],
                next_timestep=timesteps[index + 1],
            )
            for index in range(num_inference_steps)
        )

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        del step
        return latents

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        del generator
        delta = (step.next_timestep - step.timestep).to(latents)
        return latents + delta * model_output


__all__ = ["FlowMatchEulerScheduler"]
