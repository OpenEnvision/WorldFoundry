"""Stochastic rectified-flow updates from clean-sample predictions."""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.schedulers.ltx import LTXFixedEulerScheduler


class CleanSampleNoiseScheduler(LTXFixedEulerScheduler):
    """Re-noise each clean prediction at the next explicitly scheduled sigma."""

    def step(self, model_output, step, latents, *, generator):
        sigma_next = step.next_timestep.to(device=latents.device, dtype=torch.float32)
        if float(sigma_next) == 0:
            return model_output.to(latents.dtype)
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        return (model_output.float() + sigma_next * (noise.float() - model_output.float())).to(latents.dtype)


def build_clean_sample_noise_scheduler(context):
    return CleanSampleNoiseScheduler(tuple(context.component_options["sigmas"]))
