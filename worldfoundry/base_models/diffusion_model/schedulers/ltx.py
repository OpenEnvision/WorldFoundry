"""Native LTX rectified-flow schedules.

:class:`LTXFixedEulerScheduler` consumes an explicit distilled sigma list
(not a generated path).  ``schedule`` fails if
``num_inference_steps != len(sigmas) - 1`` or if sigmas are not
monotonically non-increasing.  Used by the LTX joint / video denoisers
through the standard runner loop.
"""

from __future__ import annotations

import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep


class LTXFixedEulerScheduler:
    """Euler sampler over an explicit distilled sigma trajectory."""

    def __init__(self, sigmas: tuple[float, ...]) -> None:
        if len(sigmas) < 2:
            raise ValueError("LTX sigma schedule requires at least two values")
        if any(left < right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("LTX sigmas must be monotonically non-increasing")
        self.sigmas = tuple(float(value) for value in sigmas)

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        if sampling.num_inference_steps != len(self.sigmas) - 1:
            raise ValueError(
                f"LTX fixed schedule contains {len(self.sigmas) - 1} steps, got {sampling.num_inference_steps}"
            )
        values = torch.tensor(self.sigmas, device=device, dtype=torch.float32)
        return tuple(
            SchedulerStep(index=index, timestep=values[index], next_timestep=values[index + 1])
            for index in range(len(self.sigmas) - 1)
        )

    def scale_model_input(self, latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        return latents

    def step(
        self,
        model_output: torch.Tensor,
        step: SchedulerStep,
        latents: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator
        sigma = step.timestep.to(device=latents.device, dtype=torch.float32)
        sigma_next = step.next_timestep.to(device=latents.device, dtype=torch.float32)
        if float(sigma.item()) == 0.0:
            return model_output.to(latents.dtype)
        velocity = (latents.float() - model_output.float()) / sigma
        return (latents.float() + velocity * (sigma_next - sigma)).to(latents.dtype)


def build_ltx_fixed_euler_scheduler(context: ComponentBuildContext) -> LTXFixedEulerScheduler:
    raw_sigmas = context.component_options.get("sigmas")
    if not isinstance(raw_sigmas, (tuple, list)):
        raise TypeError("LTX scheduler component requires a sigmas sequence")
    return LTXFixedEulerScheduler(tuple(float(value) for value in raw_sigmas))


__all__ = ["LTXFixedEulerScheduler", "build_ltx_fixed_euler_scheduler"]
