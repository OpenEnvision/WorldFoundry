"""Cosmos3's Karras flow UniPC schedule on the shared native solver.

Adapter that reads the published Cosmos3 scheduler JSON (via
:func:`~..loaders.metadata.checkpoint_json_config`) and drives
:class:`~.flow_unipc.FlowUniPCMultistepScheduler`.  The runner only sees
``SchedulerStep`` timesteps; UniPC state stays inside this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep
from ..loaders import NativeCheckpointResolver, checkpoint_json_config
from .flow_unipc import FlowUniPCMultistepScheduler


class Cosmos3FlowUniPCScheduler:
    """Scheduler-contract adapter matching the published Cosmos3 scheduler config."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        self.sigma_min = float(self.config.get("sigma_min", 0.147))
        self.sigma_max = float(self.config.get("sigma_max", 200.0))
        self.use_karras_sigmas = bool(self.config.get("use_karras_sigmas", True))
        self.flow_shift = float(self.config.get("flow_shift", 1.0))
        self.solver = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            solver_order=int(self.config.get("solver_order", 2)),
            prediction_type=str(self.config.get("prediction_type", "flow_prediction")),
            shift=1.0,
            thresholding=bool(self.config.get("thresholding", False)),
            dynamic_thresholding_ratio=float(self.config.get("dynamic_thresholding_ratio", 0.995)),
            sample_max_value=float(self.config.get("sample_max_value", 1.0)),
            predict_x0=bool(self.config.get("predict_x0", True)),
            solver_type=str(self.config.get("solver_type", "bh2")),
            lower_order_final=bool(self.config.get("lower_order_final", True)),
            disable_corrector=list(self.config.get("disable_corrector", ())),
            final_sigmas_type=str(self.config.get("final_sigmas_type", "zero")),
        )

    @staticmethod
    def _karras_sigmas(count: int, sigma_min: float, sigma_max: float) -> np.ndarray:
        rho = 7.0
        ramp = np.linspace(0, 1, count)
        minimum = sigma_min ** (1 / rho)
        maximum = sigma_max ** (1 / rho)
        return (maximum + ramp * (minimum - maximum)) ** rho

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        supported = {"flow_shift", "sigma_min", "sigma_max", "use_karras_sigmas"}
        unsupported = set(sampling.scheduler_options) - supported
        if unsupported:
            raise ValueError(f"unsupported Cosmos3 scheduler options: {sorted(unsupported)}")
        options = sampling.scheduler_options
        sigma_min = float(options.get("sigma_min", self.sigma_min))
        sigma_max = float(options.get("sigma_max", self.sigma_max))
        use_karras = bool(options.get("use_karras_sigmas", self.use_karras_sigmas))
        flow_shift = float(options.get("flow_shift", self.flow_shift))
        if sigma_min <= 0 or sigma_max <= sigma_min or flow_shift <= 0:
            raise ValueError("Cosmos3 scheduler requires 0 < sigma_min < sigma_max and flow_shift > 0")
        if use_karras:
            sigmas = self._karras_sigmas(sampling.num_inference_steps, sigma_min, sigma_max)
            sigmas = sigmas / (sigmas + 1.0)
        else:
            sigmas = np.linspace(1, 1 / self.num_train_timesteps, sampling.num_inference_steps + 1)[:-1]
            sigmas = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)
            if abs(float(sigmas[0]) - 1.0) < 1e-6:
                sigmas[0] -= 1e-6
        self.solver.set_timesteps(
            sampling.num_inference_steps,
            device=device,
            sigmas=sigmas,
            shift=1.0,
        )
        timesteps = self.solver.timesteps
        terminal = torch.zeros((), device=device, dtype=timesteps.dtype)
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timestep,
                next_timestep=timesteps[index + 1] if index + 1 < len(timesteps) else terminal,
            )
            for index, timestep in enumerate(timesteps)
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
        add_batch = latents.ndim == 2
        if add_batch:
            latents = latents.unsqueeze(0)
            model_output = model_output.unsqueeze(0)
        result = self.solver.step(
            model_output,
            step.timestep,
            latents,
            return_dict=False,
            generator=generator,
        )[0]
        return result.squeeze(0) if add_batch else result


def build_cosmos3_flow_unipc_scheduler(context: ComponentBuildContext) -> Cosmos3FlowUniPCScheduler:
    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("config"))
    config = checkpoint_json_config(checkpoint, "scheduler/scheduler_config.json")
    return Cosmos3FlowUniPCScheduler(config)


__all__ = ["Cosmos3FlowUniPCScheduler", "build_cosmos3_flow_unipc_scheduler"]
