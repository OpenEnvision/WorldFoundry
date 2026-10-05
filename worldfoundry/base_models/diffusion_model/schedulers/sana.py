"""Stateless schedulers for Sana recipes.

Factories return objects that satisfy :class:`~..contracts.DiffusionScheduler`.
Flow-match reuses Wan Euler; DPM / SCM / streaming Euler are Sana-specific
adapters.  Unknown ``scheduler_options`` keys raise :exc:`ValueError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep
from .flow_dpm import FlowDPMSolverMultistepScheduler
from .wan import WanFlowMatchEulerScheduler


def build_sana_flow_match_scheduler(context: ComponentBuildContext) -> WanFlowMatchEulerScheduler:
    """Reuse the canonical shifted flow Euler trajectory for Sana."""

    return WanFlowMatchEulerScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 3.0)),
    )


@dataclass(frozen=True, slots=True)
class _SanaDPMSchedulerStep(SchedulerStep):
    """Retain the multistep history owned by this sampling request."""

    solver: FlowDPMSolverMultistepScheduler


class SanaFlowDPMScheduler:
    """DPM-Solver++ order-2 adapter for released bidirectional Sana graphs."""

    def __init__(self, *, shift: float = 8.0, num_train_timesteps: int = 1000) -> None:
        self.shift = float(shift)
        self.solver = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            solver_order=2,
            prediction_type="flow_prediction",
            shift=1.0,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            final_sigmas_type="zero",
        )

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
            raise ValueError(f"unsupported SanaFlowDPMScheduler options: {sorted(unsupported)}")
        solver = FlowDPMSolverMultistepScheduler(**dict(self.solver.config))
        solver.set_timesteps(
            sampling.num_inference_steps,
            device=device,
            shift=float(sampling.scheduler_options.get("shift", self.shift)),
        )
        timesteps = solver.timesteps
        terminal = torch.zeros((), device=device, dtype=timesteps.dtype)
        return tuple(
            _SanaDPMSchedulerStep(
                index=index,
                timestep=timestep,
                next_timestep=(timesteps[index + 1] if index + 1 < len(timesteps) else terminal),
                solver=solver,
            )
            for index, timestep in enumerate(timesteps)
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
        if not isinstance(step, _SanaDPMSchedulerStep):
            raise TypeError("Sana DPM requires a step returned by its schedule")
        return step.solver.step(
            model_output,
            step.timestep,
            latents,
            generator=generator,
            return_dict=False,
        )[0]


def build_sana_flow_dpm_scheduler(context: ComponentBuildContext) -> SanaFlowDPMScheduler:
    return SanaFlowDPMScheduler(
        shift=float(context.component_options.get("shift", 8.0)),
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
    )


@dataclass(frozen=True, slots=True)
class _LongLiveSchedulerStep(SchedulerStep):
    """Retain the training sigma table owned by this sampling request."""

    sigma_table: Tensor


class SanaLongLiveScheduler:
    """Released four-step flow x0 prediction with fresh noise between steps.

    LongLive looks up sigmas in a shifted 1000-entry training table. Model
    timesteps follow the execution dtype (including BF16 rounding), while
    re-noising looks up the original integer next timestep.
    """

    TIMESTEPS = (1000, 960, 889, 727)

    def __init__(self, *, shift: float = 7.0) -> None:
        self.shift = float(shift)

    def schedule(self, sampling, *, device, dtype):
        if sampling.num_inference_steps != len(self.TIMESTEPS):
            raise ValueError("LongSANA's released LongLive checkpoint requires 4 inference steps")
        if sampling.guidance_scale != 1.0:
            raise ValueError("LongSANA's released LongLive checkpoint uses guidance_scale=1")
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(f"unsupported LongLive scheduler options: {sorted(unsupported)}")
        shift = float(sampling.scheduler_options.get("shift", self.shift))
        if shift <= 0:
            raise ValueError("LongLive shift must be positive")
        sigma = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigma_table = (shift * sigma / (1.0 + (shift - 1.0) * sigma)).to(device)
        return tuple(
            _LongLiveSchedulerStep(
                index=index,
                timestep=torch.tensor(value, device=device, dtype=dtype),
                next_timestep=torch.tensor(
                    self.TIMESTEPS[index + 1] if index + 1 < len(self.TIMESTEPS) else 0,
                    device=device, dtype=torch.float32,
                ),
                sigma_table=sigma_table,
            )
            for index, value in enumerate(self.TIMESTEPS)
        )

    @staticmethod
    def scale_model_input(latents, step):
        return latents

    def step(self, model_output, step, latents, *, generator):
        if not isinstance(step, _LongLiveSchedulerStep):
            raise TypeError("LongLive requires a step returned by its schedule")
        table = step.sigma_table
        times = table * 1000.0
        index = (times.double() - step.timestep.double()).abs().argmin()
        clean = (latents.double() - table[index].double() * model_output.double()).to(model_output.dtype)
        if step.index == len(self.TIMESTEPS) - 1:
            return clean
        index = (times - step.next_timestep).abs().argmin()
        sigma = table[index]
        batch, channels, frames, height, width = clean.shape
        flat = clean.permute(0, 2, 1, 3, 4).flatten(0, 1)
        noise = torch.empty_like(flat).normal_(generator=generator)
        noise = noise.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        calculation_dtype = torch.promote_types(clean.dtype, torch.float32)
        return (
            (1 - sigma) * clean.to(calculation_dtype) + sigma * noise.to(calculation_dtype)
        ).to(noise.dtype)


def build_sana_longlive_scheduler(context: ComponentBuildContext) -> SanaLongLiveScheduler:
    return SanaLongLiveScheduler(shift=float(context.component_options.get("shift", 7.0)))


class SanaStreamingEulerScheduler(WanFlowMatchEulerScheduler):
    """Euler flow schedule including released 2/4-step streaming timesteps."""

    _FIXED_TIMESTEPS = {
        2: (1000.0, 743.0),
        4: (1000.0, 961.0, 893.0, 743.0),
    }

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        fixed = self._FIXED_TIMESTEPS.get(sampling.num_inference_steps)
        if fixed is None:
            return super().schedule(sampling, device=device, dtype=dtype)
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(
                f"unsupported SanaStreamingEulerScheduler options: {sorted(unsupported)}"
            )
        timesteps = torch.tensor(
            (*fixed, 0.0),
            device=device,
            dtype=self.timestep_dtype,
        )
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timesteps[index],
                next_timestep=timesteps[index + 1],
            )
            for index in range(sampling.num_inference_steps)
        )


def build_sana_streaming_euler_scheduler(
    context: ComponentBuildContext,
) -> SanaStreamingEulerScheduler:
    return SanaStreamingEulerScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 8.0)),
    )


class SanaWMStreamingEulerScheduler:
    """Fixed distilled Stage-1 schedule for SANA-WM streaming.

    ``(1000, 960, 889, 727, 0)`` already contains the released schedule
    shift, so the runtime-facing ``shift`` option is accepted as provenance
    only and is deliberately not applied a second time.
    """

    TIMESTEPS = (1000.0, 960.0, 889.0, 727.0, 0.0)

    def __init__(self, *, num_train_timesteps: int = 1000) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")

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
                f"unsupported SanaWMStreamingEulerScheduler options: {sorted(unsupported)}"
            )
        required_steps = len(self.TIMESTEPS) - 1
        if sampling.num_inference_steps != required_steps:
            raise ValueError(
                f"SANA-WM streaming requires exactly {required_steps} inference steps"
            )
        values = torch.tensor(self.TIMESTEPS, device=device, dtype=torch.float32)
        return tuple(
            SchedulerStep(index=index, timestep=values[index], next_timestep=values[index + 1])
            for index in range(required_steps)
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
        del generator
        current = step.timestep.to(device=latents.device, dtype=latents.dtype)
        following = step.next_timestep.to(device=latents.device, dtype=latents.dtype)
        delta = (following - current) / float(self.num_train_timesteps)
        return latents + delta * model_output.to(dtype=latents.dtype)


def build_sana_wm_streaming_euler_scheduler(
    context: ComponentBuildContext,
) -> SanaWMStreamingEulerScheduler:
    return SanaWMStreamingEulerScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000))
    )


class SanaSCMScheduler:
    """Native TrigFlow/sCM sampler used by Sana Sprint checkpoints."""

    def __init__(self, *, sigma_data: float = 0.5) -> None:
        self.sigma_data = float(sigma_data)
        if self.sigma_data <= 0:
            raise ValueError("Sana Sprint sigma_data must be positive")

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"timesteps"}
        if unsupported:
            raise ValueError(f"unsupported SanaSCMScheduler options: {sorted(unsupported)}")
        explicit = sampling.scheduler_options.get("timesteps")
        if explicit is not None:
            values = tuple(float(value) for value in explicit)  # type: ignore[arg-type]
            if len(values) != sampling.num_inference_steps + 1:
                raise ValueError("Sana Sprint explicit timesteps must contain num_inference_steps + 1 values")
            timesteps = torch.tensor(values, device=device, dtype=torch.float32)
        elif sampling.num_inference_steps == 2:
            timesteps = torch.tensor((math.pi / 2, 1.3, 0.0), device=device, dtype=torch.float32)
        else:
            timesteps = torch.linspace(
                math.pi / 2,
                0.0,
                sampling.num_inference_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        return tuple(
            SchedulerStep(index=index, timestep=timesteps[index], next_timestep=timesteps[index + 1])
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        del step
        return latents / self.sigma_data

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        current = step.timestep.to(device=latents.device, dtype=latents.dtype)
        following = step.next_timestep.to(device=latents.device, dtype=latents.dtype)
        pred_x0 = torch.cos(current) * latents - torch.sin(current) * model_output.to(latents.dtype)
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        ) * self.sigma_data
        return torch.cos(following) * pred_x0 + torch.sin(following) * noise


def build_sana_scm_scheduler(context: ComponentBuildContext) -> SanaSCMScheduler:
    return SanaSCMScheduler(sigma_data=float(context.component_options.get("sigma_data", 0.5)))


__all__ = [
    "SanaSCMScheduler",
    "SanaFlowDPMScheduler",
    "SanaStreamingEulerScheduler",
    "SanaWMStreamingEulerScheduler",
    "build_sana_flow_dpm_scheduler",
    "build_sana_longlive_scheduler",
    "build_sana_flow_match_scheduler",
    "build_sana_scm_scheduler",
    "build_sana_streaming_euler_scheduler",
    "build_sana_wm_streaming_euler_scheduler",
]
