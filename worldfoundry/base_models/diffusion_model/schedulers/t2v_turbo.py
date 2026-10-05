"""Native latent-consistency scheduler used by T2V-Turbo.

Implements :class:`~..contracts.DiffusionScheduler` for 4–8 LCM steps
without a Diffusers dependency.  Paired with
:class:`~..models.denoisers.t2v_turbo.T2VTurboDenoiser`.
"""

from __future__ import annotations

import math

import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep


class T2VTurboLCMScheduler:
    """Four-to-eight-step LCM update without a Diffusers scheduler dependency."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        sigma_data: float = 0.5,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.sigma_data = float(sigma_data)
        betas = torch.linspace(
            math.sqrt(float(linear_start)),
            math.sqrt(float(linear_end)),
            self.num_train_timesteps,
            dtype=torch.float32,
        ).square()
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        origin_steps = int(sampling.scheduler_options.get("lcm_origin_steps", 200))
        if origin_steps <= 0 or origin_steps > self.num_train_timesteps:
            raise ValueError("lcm_origin_steps must be in [1, num_train_timesteps]")
        if sampling.num_inference_steps > origin_steps:
            raise ValueError("T2V-Turbo inference steps cannot exceed lcm_origin_steps")
        multiplier = self.num_train_timesteps // origin_steps
        origin = torch.arange(1, origin_steps + 1, dtype=torch.long) * multiplier - 1
        stride = max(origin_steps // sampling.num_inference_steps, 1)
        timesteps = origin.flip(0)[::stride][: sampling.num_inference_steps].to(device=device)
        if len(timesteps) != sampling.num_inference_steps:
            raise ValueError("could not construct the requested T2V-Turbo schedule")
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timestep,
                next_timestep=timesteps[index + 1] if index + 1 < len(timesteps) else timestep,
            )
            for index, timestep in enumerate(timesteps)
        )

    @staticmethod
    def scale_model_input(latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        del step
        return latents

    def _boundary_scalings(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = timestep.to(dtype=torch.float32) / 0.1
        denominator = value.square() + self.sigma_data**2
        return self.sigma_data**2 / denominator, value / denominator.sqrt()

    def step(
        self,
        model_output: torch.Tensor,
        step: SchedulerStep,
        latents: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        timestep = int(step.timestep.item())
        next_timestep = int(step.next_timestep.item())
        alphas = self.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        alpha_t = alphas[timestep]
        beta_t = 1.0 - alpha_t
        pred_x0 = (latents - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        c_skip, c_out = self._boundary_scalings(step.timestep)
        denoised = c_out.to(latents) * pred_x0 + c_skip.to(latents) * latents

        # The standard runner decodes its final latent. Upstream T2V-Turbo
        # separately returned ``denoised`` on the last iteration, so preserve
        # that behavior explicitly here.
        if next_timestep == timestep:
            return denoised
        alpha_next = alphas[next_timestep]
        noise = torch.randn(
            denoised.shape,
            generator=generator,
            device=denoised.device,
            dtype=denoised.dtype,
        )
        return alpha_next.sqrt() * denoised + (1.0 - alpha_next).sqrt() * noise


def build_t2v_turbo_lcm_scheduler(context: ComponentBuildContext) -> T2VTurboLCMScheduler:
    return T2VTurboLCMScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        linear_start=float(context.component_options.get("linear_start", 0.00085)),
        linear_end=float(context.component_options.get("linear_end", 0.012)),
    )


__all__ = ["T2VTurboLCMScheduler", "build_t2v_turbo_lcm_scheduler"]
