"""Canonical flow-matching schedules used by Wan inference families.

Implements the runner :class:`~..contracts.DiffusionScheduler` Protocol for
Wan recipes.  :class:`WanFlowMatchEulerScheduler` is the stateless default:
``schedule`` builds a shifted sigma path and ``step`` does
``latents + (σ_{t+1} - σ_t) * flow``.  :class:`WanFlowUniPCScheduler` wraps
the shared UniPC solver.  :class:`InferenceFlowMatchScheduler` is a
transitional stateful API for causal wrappers.

Unsupported keys in ``SamplingConfig.scheduler_options`` raise
:exc:`ValueError`.  Shift / sigma / step counts must be positive and
``sigma_max > sigma_min``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep

if TYPE_CHECKING:
    from .flow_unipc import FlowUniPCMultistepScheduler


def _validate_schedule_values(
    *,
    num_inference_steps: int,
    shift: float,
    sigma_max: float,
    sigma_min: float,
    denoising_strength: float,
) -> None:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")
    if not 0 <= denoising_strength <= 1:
        raise ValueError("denoising_strength must be in [0, 1]")


def shift_flow_sigmas(sigmas: Tensor, shift: float) -> Tensor:
    """Apply Wan's rational flow-shift transform without changing dtype/device."""

    shift = float(shift)
    if shift <= 0:
        raise ValueError("shift must be positive")
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def build_wan_sigmas(
    num_inference_steps: int,
    *,
    shift: float = 3.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
    denoising_strength: float = 1.0,
    include_terminal: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build Wan's descending shifted sigma trajectory.

    By default the returned tensor contains one sigma per model evaluation.
    ``include_terminal=True`` appends the final integration point, producing
    ``num_inference_steps + 1`` values for a stateless native runner.
    """

    num_inference_steps = int(num_inference_steps)
    shift = float(shift)
    sigma_max = float(sigma_max)
    sigma_min = float(sigma_min)
    denoising_strength = float(denoising_strength)
    _validate_schedule_values(
        num_inference_steps=num_inference_steps,
        shift=shift,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        denoising_strength=denoising_strength,
    )
    if not dtype.is_floating_point:
        raise ValueError("Wan sigma dtype must be floating point")

    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    path = torch.linspace(
        sigma_start,
        sigma_min,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    shifted = shift_flow_sigmas(path, shift)
    return shifted if include_terminal else shifted[:-1]


def _broadcast_sigma(sigma: Tensor, reference: Tensor) -> Tensor:
    values = sigma.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if values.numel() == 1:
        return values.reshape((1,) * reference.ndim)
    if not reference.ndim or values.numel() != reference.shape[0]:
        raise ValueError(
            "sigma must be scalar or have one value per sample: "
            f"got {values.numel()} values for shape {tuple(reference.shape)}"
        )
    return values.reshape(values.shape[0], *((1,) * (reference.ndim - 1)))


def add_flow_noise(original: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Interpolate clean samples and noise at a flow-matching sigma."""

    if original.shape != noise.shape:
        raise ValueError(f"original and noise shapes must match: {original.shape} != {noise.shape}")
    weight = _broadcast_sigma(sigma, noise)
    return ((1.0 - weight) * original + weight * noise).type_as(noise)


def flow_prediction_to_x0(flow: Tensor, noisy: Tensor, sigma: Tensor) -> Tensor:
    """Convert a Wan flow prediction into its clean-sample prediction."""

    if flow.shape != noisy.shape:
        raise ValueError(f"flow and noisy shapes must match: {flow.shape} != {noisy.shape}")
    original_dtype = flow.dtype
    flow64 = flow.double()
    noisy64 = noisy.double()
    weight = _broadcast_sigma(sigma, flow64)
    return (noisy64 - weight * flow64).to(original_dtype)


class WanFlowMatchEulerScheduler:
    """Stateless native Euler scheduler matching Wan's shifted flow trajectory."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        timestep_dtype: torch.dtype = torch.float32,
    ) -> None:
        if int(num_train_timesteps) <= 0:
            raise ValueError("num_train_timesteps must be positive")
        _validate_schedule_values(
            num_inference_steps=1,
            shift=float(shift),
            sigma_max=float(sigma_max),
            sigma_min=float(sigma_min),
            denoising_strength=1.0,
        )
        if not timestep_dtype.is_floating_point:
            raise ValueError("timestep_dtype must be floating point")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
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
        supported_options = {
            "denoising_strength",
            "shift",
            "sigma_max",
            "sigma_min",
        }
        unsupported = set(sampling.scheduler_options) - supported_options
        if unsupported:
            raise ValueError(f"unsupported WanFlowMatchEulerScheduler options: {sorted(unsupported)}")
        options = sampling.scheduler_options
        sigmas = build_wan_sigmas(
            sampling.num_inference_steps,
            shift=float(options.get("shift", self.shift)),
            sigma_max=float(options.get("sigma_max", self.sigma_max)),
            sigma_min=float(options.get("sigma_min", self.sigma_min)),
            denoising_strength=float(options.get("denoising_strength", 1.0)),
            include_terminal=True,
            device=device,
            dtype=self.timestep_dtype,
        )
        timesteps = sigmas * self.num_train_timesteps
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timesteps[index],
                next_timestep=timesteps[index + 1],
            )
            for index in range(sampling.num_inference_steps)
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
        timestep_scale = float(self.num_train_timesteps)
        sigma = step.timestep / timestep_scale
        next_sigma = step.next_timestep / timestep_scale
        delta = (next_sigma - sigma).to(device=latents.device, dtype=latents.dtype)
        return latents + delta * model_output.to(latents.dtype)


def build_wan_flow_match_euler_scheduler(
    context: ComponentBuildContext,
) -> WanFlowMatchEulerScheduler:
    """Build the shared Wan Euler scheduler from component options."""

    options = context.component_options
    return WanFlowMatchEulerScheduler(
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        shift=float(options.get("shift", 3.0)),
        sigma_max=float(options.get("sigma_max", 1.0)),
        sigma_min=float(options.get("sigma_min", 0.0)),
    )


@dataclass(frozen=True, slots=True)
class _CausalWanSchedulerStep(SchedulerStep):
    """Carry the request's shifted boundary without mutable scheduler state."""

    boundary_sigma: Tensor


class FastVideoCausalWanSelfForcingScheduler:
    """Fixed eight-step self-forcing sampler released with CausalWan2.2.

    The configured values are raw training indices. High-noise expert outputs
    predict the boundary latent; low-noise outputs predict x0. The boundary
    sigma is the nearest point on FastVideo's 1000-step training grid.
    """

    DEFAULT_RAW_TIMESTEPS = (1000, 850, 700, 550, 350, 275, 200, 125)

    def __init__(
        self,
        *,
        raw_timesteps: tuple[int, ...] = DEFAULT_RAW_TIMESTEPS,
        shift: float = 12.0,
        boundary_ratio: float = 0.875,
        num_train_timesteps: int = 1000,
        timestep_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.raw_timesteps = tuple(int(value) for value in raw_timesteps)
        self.shift = float(shift)
        self.boundary_ratio = float(boundary_ratio)
        self.num_train_timesteps = int(num_train_timesteps)
        self.timestep_dtype = timestep_dtype
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if self.shift <= 0:
            raise ValueError("shift must be positive")
        if not 0.0 < self.boundary_ratio < 1.0:
            raise ValueError("boundary_ratio must be in (0, 1)")
        if not self.timestep_dtype.is_floating_point:
            raise ValueError("timestep_dtype must be floating point")
        if not self.raw_timesteps:
            raise ValueError("FastVideo CausalWan raw_timesteps cannot be empty")
        if any(value <= 0 or value > self.num_train_timesteps for value in self.raw_timesteps):
            raise ValueError("FastVideo CausalWan raw timesteps must be in (0, num_train_timesteps]")
        if any(left <= right for left, right in zip(self.raw_timesteps, self.raw_timesteps[1:])):
            raise ValueError("FastVideo CausalWan raw timesteps must be strictly descending")

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(
                "unsupported FastVideoCausalWanSelfForcingScheduler options: "
                f"{sorted(unsupported)}"
            )
        if sampling.num_inference_steps != len(self.raw_timesteps):
            raise ValueError(
                "FastVideo CausalWan requires exactly "
                f"{len(self.raw_timesteps)} inference steps"
            )
        shift = float(sampling.scheduler_options.get("shift", self.shift))
        raw = torch.tensor(
            (*self.raw_timesteps, 0),
            device=device,
            dtype=self.timestep_dtype,
        )
        sigmas = shift_flow_sigmas(raw / float(self.num_train_timesteps), shift)
        model_timesteps = sigmas * float(self.num_train_timesteps)
        training_sigmas = shift_flow_sigmas(
            torch.linspace(
                1.0,
                0.0,
                self.num_train_timesteps + 1,
                device=device,
                dtype=self.timestep_dtype,
            )[:-1],
            shift,
        )
        boundary_index = torch.argmin(
            (training_sigmas * self.num_train_timesteps
             - self.boundary_ratio * self.num_train_timesteps).abs()
        )
        boundary_sigma = training_sigmas[boundary_index]
        return tuple(
            _CausalWanSchedulerStep(
                index=index,
                timestep=model_timesteps[index],
                next_timestep=model_timesteps[index + 1],
                boundary_sigma=boundary_sigma,
            )
            for index in range(len(self.raw_timesteps))
        )

    @staticmethod
    def scale_model_input(latents: Tensor, step: SchedulerStep) -> Tensor:
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
        if not isinstance(step, _CausalWanSchedulerStep):
            raise TypeError("FastVideo CausalWan requires its own scheduler step")
        sigma = step.timestep.to(device=latents.device, dtype=torch.float64)
        sigma = sigma / float(self.num_train_timesteps)
        boundary_sigma = step.boundary_sigma.to(device=latents.device, dtype=torch.float64)
        boundary_timestep = self.boundary_ratio * self.num_train_timesteps
        high_noise = bool(step.timestep >= boundary_timestep)
        clean = flow_prediction_to_x0(
            model_output,
            latents,
            sigma - boundary_sigma if high_noise else sigma,
        )
        next_sigma = step.next_timestep.to(device=latents.device, dtype=torch.float64)
        next_sigma = next_sigma / float(self.num_train_timesteps)
        if float(next_sigma.detach().cpu()) == 0.0:
            return clean
        noise = torch.randn(
            clean.shape,
            generator=generator,
            device=generator.device,
            dtype=clean.dtype,
        ).to(clean.device)
        if high_noise and bool(step.next_timestep < boundary_timestep):
            # Upstream draws and discards this noise on the expert switch.
            return clean
        next_sigma32 = next_sigma.to(torch.float32)
        if high_noise:
            boundary32 = boundary_sigma.to(torch.float32)
            alpha = (1.0 - next_sigma32) / (1.0 - boundary32)
            beta = torch.sqrt(next_sigma32.square() - (alpha * boundary32).square())
            return (alpha * clean + beta * noise).to(noise.dtype)
        return ((1.0 - next_sigma32) * clean + next_sigma32 * noise).to(noise.dtype)


def build_fastvideo_causal_wan_self_forcing_scheduler(
    context: ComponentBuildContext,
) -> FastVideoCausalWanSelfForcingScheduler:
    options = context.component_options
    return FastVideoCausalWanSelfForcingScheduler(
        raw_timesteps=tuple(
            int(value)
            for value in options.get(
                "raw_timesteps",
                FastVideoCausalWanSelfForcingScheduler.DEFAULT_RAW_TIMESTEPS,
            )
        ),
        shift=float(options.get("shift", 12.0)),
        boundary_ratio=float(options.get("boundary_ratio", 0.875)),
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
    )


@dataclass(frozen=True, slots=True)
class _WanUniPCSchedulerStep(SchedulerStep):
    """Retain the multistep history owned by this sampling request."""

    solver: FlowUniPCMultistepScheduler


class WanFlowUniPCScheduler:
    """Native scheduler-contract adapter for Wan's canonical UniPC solver."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        use_karras_sigma: bool = False,
        karras_steps_are_intervals: bool = False,
    ) -> None:
        from .flow_unipc import FlowUniPCMultistepScheduler

        self.shift = float(shift)
        self.use_karras_sigma = bool(use_karras_sigma)
        self.karras_steps_are_intervals = bool(karras_steps_are_intervals)
        self.solver = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        from .flow_unipc import FlowUniPCMultistepScheduler

        del dtype
        unsupported = set(sampling.scheduler_options) - {"shift", "use_karras_sigma"}
        if unsupported:
            raise ValueError(f"unsupported WanFlowUniPCScheduler options: {sorted(unsupported)}")
        use_karras_sigma = bool(sampling.scheduler_options.get("use_karras_sigma", self.use_karras_sigma))
        # The NVIDIA solver's Karras branch interprets ``N`` as intervals and
        # returns N+1 model-evaluation points. Most native callers define N as
        # evaluations, so translate by default. The post-trained Cosmos2.5-2B
        # recipe opts into NVIDIA's original interval count for exact parity.
        # A one-evaluation canonical schedule uses the equivalent single flow
        # step because passing zero Karras intervals would divide by zero.
        effective_karras_sigma = use_karras_sigma and (
            self.karras_steps_are_intervals or sampling.num_inference_steps > 1
        )
        solver_steps = (
            sampling.num_inference_steps - 1
            if effective_karras_sigma and not self.karras_steps_are_intervals
            else sampling.num_inference_steps
        )
        solver = FlowUniPCMultistepScheduler(**dict(self.solver.config))
        solver.set_timesteps(
            solver_steps,
            device=device,
            shift=float(sampling.scheduler_options.get("shift", self.shift)),
            use_kerras_sigma=effective_karras_sigma,
        )
        timesteps = solver.timesteps
        terminal = torch.zeros((), device=device, dtype=timesteps.dtype)
        return tuple(
            _WanUniPCSchedulerStep(
                index=index,
                timestep=timestep,
                next_timestep=(timesteps[index + 1] if index + 1 < len(timesteps) else terminal),
                solver=solver,
            )
            for index, timestep in enumerate(timesteps)
        )

    def expected_step_count(self, sampling: SamplingConfig) -> int:
        """Account for NVIDIA's 2B convention: N Karras intervals yield N+1 updates."""

        use_karras_sigma = bool(sampling.scheduler_options.get("use_karras_sigma", self.use_karras_sigma))
        return sampling.num_inference_steps + int(self.karras_steps_are_intervals and use_karras_sigma)

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
        if not isinstance(step, _WanUniPCSchedulerStep):
            raise TypeError("Wan UniPC requires a step returned by its schedule")
        return step.solver.step(
            model_output,
            step.timestep,
            latents,
            return_dict=False,
            generator=generator,
        )[0]


def build_wan_flow_unipc_scheduler(context: ComponentBuildContext) -> WanFlowUniPCScheduler:
    """Build the shared Wan UniPC solver from component options."""

    return WanFlowUniPCScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 5.0)),
        use_karras_sigma=bool(context.component_options.get("use_karras_sigma", False)),
        karras_steps_are_intervals=bool(context.component_options.get("karras_steps_are_intervals", False)),
    )


class InferenceFlowMatchScheduler:
    """Transitional stateful API for causal Wan runtimes.

    New native recipes should use :class:`WanFlowMatchEulerScheduler`. This
    compatibility surface preserves the exact API used by existing causal
    model wrappers while sharing the canonical schedule and conversion code.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        *,
        num_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ) -> None:
        if int(num_timesteps) <= 0:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = int(num_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
    ) -> None:
        num_inference_steps = int(num_inference_steps)
        if self.extra_one_step:
            self.sigmas = build_wan_sigmas(
                num_inference_steps,
                shift=self.shift,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                denoising_strength=denoising_strength,
            )
        else:
            _validate_schedule_values(
                num_inference_steps=num_inference_steps,
                shift=self.shift,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                denoising_strength=float(denoising_strength),
            )
            sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)
            unshifted = torch.linspace(
                sigma_start,
                self.sigma_min,
                num_inference_steps,
                dtype=torch.float32,
            )
            self.sigmas = shift_flow_sigmas(unshifted, self.shift)
        self.timesteps = self.sigmas * self.num_timesteps

    def _indices(self, timestep: Tensor, device: torch.device) -> Tensor:
        timesteps = self.timesteps.to(device=device, dtype=torch.float64)
        values = timestep.reshape(-1).to(device=device, dtype=torch.float64)
        return torch.argmin(
            (timesteps.unsqueeze(0) - values.unsqueeze(1)).abs(),
            dim=1,
        )

    def sigma_at(self, timestep: Tensor, reference: Tensor) -> Tensor:
        indices = self._indices(timestep, reference.device)
        values = self.sigmas.to(device=reference.device, dtype=reference.dtype)[indices]
        return _broadcast_sigma(values, reference)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        return add_flow_noise(
            original_samples,
            noise,
            self.sigma_at(timestep, noise),
        )

    def flow_to_x0(self, flow: Tensor, noisy: Tensor, timestep: Tensor) -> Tensor:
        sigma = self.sigma_at(timestep, flow.double())
        return flow_prediction_to_x0(flow, noisy, sigma)

    def flow_step(
        self,
        flow: Tensor,
        noisy: Tensor,
        timestep: Tensor,
        next_timestep: Tensor,
    ) -> Tensor:
        sigma = self.sigma_at(timestep, noisy)
        next_sigma = self.sigma_at(next_timestep, noisy)
        return noisy + flow.to(noisy.dtype) * (next_sigma - sigma)


__all__ = [
    "FastVideoCausalWanSelfForcingScheduler",
    "InferenceFlowMatchScheduler",
    "WanFlowMatchEulerScheduler",
    "WanFlowUniPCScheduler",
    "add_flow_noise",
    "build_fastvideo_causal_wan_self_forcing_scheduler",
    "build_wan_flow_match_euler_scheduler",
    "build_wan_flow_unipc_scheduler",
    "build_wan_sigmas",
    "flow_prediction_to_x0",
    "shift_flow_sigmas",
]
