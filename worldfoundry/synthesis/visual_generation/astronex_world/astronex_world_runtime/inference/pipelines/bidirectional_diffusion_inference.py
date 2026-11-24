# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1,
# inference/pipelines/bidirectional_diffusion_inference.py.
"""Bidirectional multi-step sampler: UniPC over the whole clip with classifier-free guidance."""

from __future__ import annotations

import torch
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler

from ...models.components import AstronexTextEncoder, AstronexVAE
from ...models.denoiser import AstronexDenoiser


class BidirectionalDiffusionInferencePipeline(torch.nn.Module):
    """Denoises every latent frame at once; an image reference replaces frame 0 at ``t = 0``."""

    def __init__(self, args):
        super().__init__()
        model_kwargs = dict(args.model_kwargs)
        self.generator = AstronexDenoiser(**model_kwargs, is_causal=False)
        self.text_encoder = AstronexTextEncoder(model_kwargs["model_root"])
        self.vae = AstronexVAE(model_kwargs["model_root"])
        self.vae_device = None
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = int(args.get("sampling_steps", 50))
        self.shift = 5.0
        self.args = args

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        return_latents: bool = False,
        initial_latent: torch.Tensor | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        event_prompts: str | None = None,
        event_start_frame: int = 0,
    ):
        """Sample ``noise`` (``[B, F, C, H, W]``); returns the decoded video in ``[0, 1]`` (and latents)."""
        if event_start_frame:
            raise ValueError("event_start_frame is only supported by the causal pipeline")
        if event_prompts:
            raise ValueError("the released bidirectional model has no event-prompt branch")
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        latents = noise
        if initial_latent is not None:
            first_frame_mask = torch.ones(
                latents.shape[0], latents.shape[1], 1, 1, 1, device=initial_latent.device, dtype=initial_latent.dtype
            )
            first_frame_mask[:, :1] = 0

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(self.sampling_steps, device=noise.device, shift=self.shift)
        for t in tqdm(list(scheduler.timesteps)):
            if initial_latent is not None:
                latent_model_input = (1 - first_frame_mask) * initial_latent + first_frame_mask * latents
            else:
                latent_model_input = latents
            timestep = t * torch.ones([latents.shape[0], latents.shape[1]], device=noise.device, dtype=torch.float32)
            if initial_latent is not None:
                timestep[:, :1] = 0
            flow_pred_cond = self.generator(latent_model_input, conditional_dict, timestep, viewmats=viewmats, Ks=Ks)
            flow_pred_uncond = self.generator(
                latent_model_input, unconditional_dict, timestep, viewmats=viewmats, Ks=Ks
            )
            flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
            latents = scheduler.step(flow_pred, t, latents, return_dict=False)[0]

        if initial_latent is not None:
            latents = (1 - first_frame_mask) * initial_latent + first_frame_mask * latents

        video = self.vae.decode_to_pixel(latents.to(self.vae_device) if self.vae_device is not None else latents)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, latents
        return video
