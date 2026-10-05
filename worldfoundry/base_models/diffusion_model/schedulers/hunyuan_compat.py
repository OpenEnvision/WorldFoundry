"""Diffusers-facing adapter for the shared Hunyuan Euler implementation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from ..contracts import SamplingConfig
from .hunyuan_video import HunyuanVideoFlowMatchEulerScheduler


class HunyuanVideoFlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Expose the legacy descending Euler API without duplicating its math."""

    order = 1
    init_noise_sigma = 1.0

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=1.0, reverse=True, solver="euler"):
        if not reverse or solver != "euler":
            raise ValueError("Only descending Euler inference is supported.")
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        # Keep native steps in sigma units to avoid a multiply/divide round trip.
        self.native = HunyuanVideoFlowMatchEulerScheduler(num_train_timesteps=1, shift=shift)
        self.set_timesteps(num_train_timesteps)

    def set_timesteps(self, num_inference_steps, device=None, n_tokens=None):
        del n_tokens
        self.num_inference_steps = int(num_inference_steps)
        self._steps = self.native.schedule(
            SamplingConfig(num_inference_steps=self.num_inference_steps),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.sigmas = torch.stack([step.timestep for step in self._steps] + [self._steps[-1].next_timestep])
        self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(device=device)
        self._step_index = None

    @property
    def step_index(self):
        return self._step_index

    def scale_model_input(self, sample, timestep=None):
        return sample

    def step(self, model_output, timestep, sample, return_dict=True):
        if not torch.is_floating_point(torch.as_tensor(timestep)):
            raise ValueError("Pass a scheduler timestep, not an integer step index.")
        if self._step_index is None:
            indices = (self.timesteps == timestep).nonzero().flatten()
            if not len(indices):
                raise ValueError("Timestep is not in the current schedule.")
            self._step_index = int(indices[1 if len(indices) > 1 else 0])
        previous = self.native.step(
            model_output.float(), self._steps[self._step_index], sample.float(), generator=None
        )
        self._step_index += 1
        return SimpleNamespace(prev_sample=previous) if return_dict else (previous,)
