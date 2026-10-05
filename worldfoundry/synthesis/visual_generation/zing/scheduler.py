# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.zing.config import InferenceConfig


class DmdScheduler:
    def __init__(self, config: InferenceConfig):
        raw_sigmas = torch.linspace(1.0, 0.0, config.num_timesteps + 1)[:-1]
        sigmas = config.timestep_shift * raw_sigmas / (1.0 + (config.timestep_shift - 1.0) * raw_sigmas)
        grid = sigmas * config.num_timesteps
        steps = torch.tensor(config.denoising_steps, dtype=torch.long)
        warped_steps = torch.cat((grid, torch.tensor([0.0])))[config.num_timesteps - steps].float()
        if warped_steps[-1] == 0:
            warped_steps = warped_steps[:-1]
        indices = torch.argmin((grid[None, :] - warped_steps[:, None]).abs(), dim=1)
        self.timesteps = warped_steps
        self.sigmas = sigmas[indices]
        self.index = 0

    def to(self, device: torch.device) -> DmdScheduler:
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def step(self, model_output: torch.Tensor, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigmas[self.index]
        x0 = sample.float() - sigma.float() * model_output.float()
        x0_model = x0.to(sample.dtype)
        self.index += 1
        if self.index < len(self.timesteps):
            next_sigma = self.sigmas[self.index]
            noise = torch.randn_like(x0_model)
            output = ((1.0 - next_sigma.float()) * x0_model.float() + next_sigma.float() * noise.float()).to(
                sample.dtype
            )
        else:
            output = x0_model
        return output, x0
