"""Framework-owned Karras schedule with x0-form Adams-Bashforth updates.

Used when a denoiser predicts clean ``x0`` (Cosmos EDM-style) rather than
flow.  Implements :class:`~..contracts.DiffusionScheduler`.  The AB2
instance stores the previous prediction across steps of one run.
"""

from __future__ import annotations

import torch

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep


def _batch_mul(scale: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return scale.reshape(-1, *([1] * (value.ndim - 1))) * value


def _phi1(value: torch.Tensor) -> torch.Tensor:
    return torch.expm1(value) / value


def _phi2(value: torch.Tensor) -> torch.Tensor:
    return (_phi1(value) - 1.0) / value


class KarrasX0AB2Scheduler:
    """Second-order residual/AB solver for denoisers that predict clean x0."""

    def __init__(self, *, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self._previous_x0: torch.Tensor | None = None
        self._previous_sigma: torch.Tensor | None = None
        self._final_sigma: torch.Tensor | None = None
        self._num_steps = 0

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        supported = {"sigma_min", "sigma_max", "rho"}
        unsupported = set(sampling.scheduler_options) - supported
        if unsupported:
            raise ValueError(f"unsupported Karras x0 AB2 scheduler options: {sorted(unsupported)}")
        options = sampling.scheduler_options
        sigma_min = float(options.get("sigma_min", self.sigma_min))
        sigma_max = float(options.get("sigma_max", self.sigma_max))
        rho = float(options.get("rho", self.rho))
        if sigma_min <= 0 or sigma_max <= sigma_min or rho <= 0:
            raise ValueError("Karras x0 AB2 requires 0 < sigma_min < sigma_max and rho > 0")
        ramp = torch.linspace(0, 1, sampling.num_inference_steps + 1, device=device, dtype=torch.float64)
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        # Network precision must not quantize the integration grid. NVIDIA's
        # rectified-flow sampler keeps sigma and its persistent sample in fp32.
        dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self._previous_x0 = None
        self._previous_sigma = None
        self._final_sigma = sigmas[-1].to(dtype=dtype)
        self._num_steps = sampling.num_inference_steps
        return tuple(
            SchedulerStep(
                index=index,
                timestep=sigmas[index].to(dtype=dtype),
                next_timestep=sigmas[index + 1].to(dtype=dtype),
            )
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
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
        del generator
        work = torch.float64
        current = latents.to(work)
        x0 = model_output.to(work)
        sigma = step.timestep.to(device=latents.device, dtype=work).reshape(1)
        next_sigma = step.next_timestep.to(device=latents.device, dtype=work).reshape(1)
        if self._previous_x0 is None or self._previous_sigma is None:
            result = _batch_mul((sigma - next_sigma) / sigma, x0) + _batch_mul(
                next_sigma / sigma, current
            )
        else:
            previous_x0 = self._previous_x0.to(device=latents.device, dtype=work)
            previous_sigma = self._previous_sigma.to(device=latents.device, dtype=work).reshape(1)
            log_s = -torch.log(sigma)
            log_t = -torch.log(next_sigma)
            log_previous = -torch.log(previous_sigma)
            delta = log_t - log_s
            c2 = (log_previous - log_s) / delta
            phi1 = _phi1(-delta)
            phi2 = _phi2(-delta)
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            result = _batch_mul(torch.exp(-delta), current) + _batch_mul(
                delta, _batch_mul(b1, x0) + _batch_mul(b2, previous_x0)
            )
        self._previous_x0 = model_output.detach()
        self._previous_sigma = step.timestep.detach()
        return result.to(dtype=latents.dtype)

    def final_denoise_step(self) -> SchedulerStep | None:
        """Request the common terminal clean prediction at ``sigma_min``."""

        if self._final_sigma is None:
            return None
        return SchedulerStep(
            index=self._num_steps,
            timestep=self._final_sigma,
            next_timestep=self._final_sigma.new_zeros(()),
        )


class KarrasX0EulerScheduler:
    """Karras EDM Euler updates for denoisers that return clean ``x0``.

    Diffusers' :class:`EDMEulerScheduler`, used by Cosmos Predict1 and GEN3C,
    evaluates exactly ``num_inference_steps`` sigma values and appends a final
    zero sigma.  The model wrapper already applies EDM input/output
    preconditioning and therefore exposes clean ``x0`` to this scheduler.
    """

    def __init__(self, *, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        supported = {"sigma_min", "sigma_max", "rho"}
        unsupported = set(sampling.scheduler_options) - supported
        if unsupported:
            raise ValueError(f"unsupported Karras x0 Euler scheduler options: {sorted(unsupported)}")
        options = sampling.scheduler_options
        sigma_min = float(options.get("sigma_min", self.sigma_min))
        sigma_max = float(options.get("sigma_max", self.sigma_max))
        rho = float(options.get("rho", self.rho))
        if sigma_min <= 0 or sigma_max <= sigma_min or rho <= 0:
            raise ValueError("Karras x0 Euler requires 0 < sigma_min < sigma_max and rho > 0")
        ramp = torch.linspace(0, 1, sampling.num_inference_steps, device=device, dtype=torch.float64)
        nonzero = (
            sigma_max ** (1 / rho)
            + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        sigmas = torch.cat((nonzero, nonzero.new_zeros(1)))
        return tuple(
            SchedulerStep(
                index=index,
                timestep=sigmas[index].to(dtype=dtype),
                next_timestep=sigmas[index + 1].to(dtype=dtype),
            )
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
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
        del generator
        current = latents.float()
        x0 = model_output.float()
        sigma = step.timestep.to(device=latents.device, dtype=torch.float32).reshape(1)
        next_sigma = step.next_timestep.to(device=latents.device, dtype=torch.float32).reshape(1)
        result = current + _batch_mul((next_sigma - sigma) / sigma, current - x0)
        return result.to(dtype=latents.dtype)


def build_karras_x0_ab2_scheduler(context: ComponentBuildContext) -> KarrasX0AB2Scheduler:
    options = context.component_options
    return KarrasX0AB2Scheduler(
        sigma_min=float(options.get("sigma_min", 0.002)),
        sigma_max=float(options.get("sigma_max", 80.0)),
        rho=float(options.get("rho", 7.0)),
    )


def build_karras_x0_euler_scheduler(context: ComponentBuildContext) -> KarrasX0EulerScheduler:
    options = context.component_options
    return KarrasX0EulerScheduler(
        sigma_min=float(options.get("sigma_min", 0.002)),
        sigma_max=float(options.get("sigma_max", 80.0)),
        rho=float(options.get("rho", 7.0)),
    )


__all__ = [
    "KarrasX0AB2Scheduler",
    "KarrasX0EulerScheduler",
    "build_karras_x0_ab2_scheduler",
    "build_karras_x0_euler_scheduler",
]
