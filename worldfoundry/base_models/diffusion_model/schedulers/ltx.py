"""Native LTX rectified-flow schedules.

:class:`LTXFixedEulerScheduler` consumes an explicit distilled sigma list
(not a generated path).  ``schedule`` fails if
``num_inference_steps != len(sigmas) - 1`` or if sigmas are not
monotonically non-increasing.  Used by the LTX joint / video denoisers
through the standard runner loop.

:class:`LTXProjectedEulerScheduler` and
:class:`LTXProjectedAncestralEulerScheduler` follow the ``ltx_pipelines``
denoising loops: the clean-sample prediction is projected onto the
conditioning latent (``denoise_mask``) before the solver step, and the
ancestral step re-projects after injecting noise.  They implement
``step_state`` so the joint multi-stage runner leaves that projection to them.
"""

from __future__ import annotations

import itertools

import torch

from worldfoundry.core.nn.transformer.dit import denoised_to_velocity

from ..components import ComponentBuildContext
from ..contracts import ModalityState, SamplingConfig, SchedulerStep


class LTXFixedEulerScheduler:
    """Euler sampler over an explicit distilled sigma trajectory."""

    def __init__(self, sigmas: tuple[float, ...]) -> None:
        if len(sigmas) < 2:
            raise ValueError("LTX sigma schedule requires at least two values")
        if any(left < right for left, right in itertools.pairwise(sigmas)):
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


def project_ltx_prediction(denoised: torch.Tensor, state: ModalityState) -> torch.Tensor:
    """Blend a clean-sample prediction with the conditioning latent by ``denoise_mask``."""

    mask = state.denoise_mask
    return (denoised * mask + state.clean_latent.float() * (1 - mask)).to(denoised.dtype)


class LTXProjectedEulerScheduler(LTXFixedEulerScheduler):
    """Deterministic Euler step on the mask-projected clean-sample prediction."""

    def step_state(
        self,
        denoised: torch.Tensor,
        step: SchedulerStep,
        state: ModalityState,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator
        denoised = project_ltx_prediction(denoised, state)
        sample = state.latent
        velocity = denoised_to_velocity(sample, step.timestep, denoised)
        delta = step.next_timestep - step.timestep
        return (sample.to(torch.float32) + velocity.to(torch.float32) * delta).to(sample.dtype)


class LTXProjectedAncestralEulerScheduler(LTXFixedEulerScheduler):
    """Rectified-flow ancestral (SDE) Euler step on the mask-projected prediction.

    Each step moves deterministically to ``sigma_down <= sigma_next`` and renoises back to
    ``sigma_next`` with the variance-preserving ``alpha = 1 - sigma`` rescale; ``eta``
    interpolates between a plain Euler step (0) and a fully ancestral step (1). A terminal
    zero sigma returns the projected prediction.
    """

    def __init__(self, sigmas: tuple[float, ...], *, eta: float = 1.0, s_noise: float = 1.0) -> None:
        super().__init__(sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)

    def step_state(
        self,
        denoised: torch.Tensor,
        step: SchedulerStep,
        state: ModalityState,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        sample = state.latent
        denoised = project_ltx_prediction(denoised.float(), state)
        sigma = step.timestep.to(torch.float32)
        sigma_next = step.next_timestep.to(torch.float32)
        if sigma_next == 0:
            return denoised.to(sample.dtype)
        noise = (
            torch.randn(sample.shape, generator=generator, dtype=sample.dtype, device=sample.device)
            if self.eta > 0
            else None
        )
        x = sample.float()
        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * self.eta
        sigma_down = sigma_next * downstep_ratio
        sigma_down_ratio = sigma_down / sigma
        x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised
        if noise is not None:
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
            x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * self.s_noise * renoise_coeff
            x_next = project_ltx_prediction(x_next, state)
        return x_next.to(sample.dtype)


def _ltx_sigmas(context: ComponentBuildContext) -> tuple[float, ...]:
    raw_sigmas = context.component_options.get("sigmas")
    if not isinstance(raw_sigmas, (tuple, list)):
        raise TypeError("LTX scheduler component requires a sigmas sequence")
    return tuple(float(value) for value in raw_sigmas)


def build_ltx_fixed_euler_scheduler(context: ComponentBuildContext) -> LTXFixedEulerScheduler:
    return LTXFixedEulerScheduler(_ltx_sigmas(context))


def build_ltx_projected_euler_scheduler(context: ComponentBuildContext) -> LTXProjectedEulerScheduler:
    return LTXProjectedEulerScheduler(_ltx_sigmas(context))


def build_ltx_projected_ancestral_euler_scheduler(
    context: ComponentBuildContext,
) -> LTXProjectedAncestralEulerScheduler:
    return LTXProjectedAncestralEulerScheduler(
        _ltx_sigmas(context),
        eta=float(context.component_options.get("eta", 1.0)),
        s_noise=float(context.component_options.get("s_noise", 1.0)),
    )


__all__ = [
    "LTXFixedEulerScheduler",
    "LTXProjectedAncestralEulerScheduler",
    "LTXProjectedEulerScheduler",
    "build_ltx_fixed_euler_scheduler",
    "build_ltx_projected_ancestral_euler_scheduler",
    "build_ltx_projected_euler_scheduler",
    "project_ltx_prediction",
]
