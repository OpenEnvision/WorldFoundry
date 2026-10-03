# SPDX-License-Identifier: Apache-2.0
"""Shared InSpatio World block sampler, with injected native Wan models.

Extracted from the existing InSpatio 1.0 runtime. Version 1.5 uses the same
reference/previous-block context and four-step causal sampling equations.
"""
from contextlib import nullcontext
from typing import List, Optional

import torch


def sample_noise_like(value, generator=None):
    """Preserve randn_like layout and RNG draws with an optional request RNG."""
    if generator is None:
        return torch.randn_like(value)
    return torch.empty_like(value).normal_(generator=generator)


def denoise_block(
    generator,
    scheduler,
    noisy_input,
    conditional_dict,
    kv_cache,
    *,
    context_frames=None,
    context_no_grad=True,
    context_freqs_offset=0,
    context_kv_size_0=0,
    render_block=None,
    denoising_kv_size=0,
    denoising_kv_size_0=0,
    denoising_steps=None,
    noise_generator=None,
):
    """
    Shared block-based diffusion core: optional context encoding pass + denoising.

    Returns (denoised_pred, noise_before_last_step).
    """
    B, F = noisy_input.shape[:2]
    device = noisy_input.device
    noise_before_last_step = None

    if context_frames is not None:
        times_zero = torch.zeros([B, F], device=device, dtype=torch.int64)
        ctx = torch.no_grad() if context_no_grad else nullcontext()
        with ctx:
            generator(
                noisy_image_or_video=context_frames,
                conditional_dict=conditional_dict,
                timestep=times_zero,
                kv_cache=kv_cache,
                render_latent_input=render_block,
                kv_size=(context_kv_size_0, -1),
                freqs_offset=context_freqs_offset,
            )

    for index, current_timestep in enumerate(denoising_steps):
        is_last_step = (index == len(denoising_steps) - 1)
        timestep = torch.ones([B, F], device=device, dtype=torch.int64) * current_timestep

        ctx = torch.no_grad() if not is_last_step else nullcontext()
        with ctx:
            _, denoised_pred = generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                kv_size=(denoising_kv_size_0, denoising_kv_size),
                render_latent_input=render_block,
                freqs_offset=6,
            )

        if is_last_step:
            noise_before_last_step = noisy_input.clone()
        else:
            next_t = denoising_steps[index + 1]
            noisy_input = scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                sample_noise_like(denoised_pred.flatten(0, 1), noise_generator),
                next_t * torch.ones([B * F], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

    return denoised_pred, noise_before_last_step


class InspatioCausalRollout(torch.nn.Module):
    def __init__(self, args, *, generator, text_encoder, vae=None):
        super().__init__()
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            print("warping denoising step list")
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.max_num_context_frames = 6

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        ref_latent: Optional[torch.Tensor] = None,
        render_latent: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
        decode: bool = True,
        noise_generator=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            decode (bool): If True (default), decode latents to pixel space via VAE.
                If False, return denoised latents directly (e.g. for external TAE decoder).
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                When decode=True, normalized to [0, 1]. When decode=False, raw latents.
        """
        if noise.ndim != 5 or noise.shape[1] < 1:
            raise ValueError("Noise must be non-empty [B,F,C,H,W]")
        if (ref_latent is None or render_latent is None or mask_latent is None
                or ref_latent.shape != noise.shape or render_latent.shape != noise.shape
                or mask_latent.shape != (*noise.shape[:2], 4, *noise.shape[-2:])):
            raise ValueError("Source, render, and four-channel mask latent shapes must match noise")
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0, f"num_frames {num_frames} is not a multiple of num_frame_per_block {self.num_frame_per_block}"
        num_blocks = num_frames // self.num_frame_per_block

        num_output_frames = num_frames   # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )


        self._initialize_kv_cache(batch_size, noise.dtype, noise.device)

        # Step 3: Temporal denoising loop
        print(f"Generating {num_blocks} blocks...")
        all_num_frames = [self.num_frame_per_block] * num_blocks

        start_index = 0
        last_pred = None
        for num_block_frame in all_num_frames:
            noisy_input = noise[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            ref_block = ref_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = render_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            mask_block = mask_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = torch.cat([mask_block, render_block], dim=2)

            kv_size = self.frame_seq_length * self.num_frame_per_block

            # Prepare context
            context_frames = None
            zero_latents = torch.zeros_like(ref_block)
            ref_block = torch.cat([ref_block, zero_latents[:, :, :4], zero_latents], dim=2)
            if start_index == 0:
                context_frames = ref_block
            else:
                # Prepare context similar to training pipeline
                zero_latents = torch.zeros_like(last_pred)
                last_pred_padded = torch.cat([last_pred, zero_latents[:, :, :4], zero_latents], dim=2)
                context_frames = torch.cat([ref_block, last_pred_padded], dim=1)
                kv_size = kv_size + self.frame_seq_length * self.num_frame_per_block

            denoised_pred, _ = denoise_block(
                self.generator, self.scheduler, noisy_input, conditional_dict,
                self.kv_cache1,
                context_frames=context_frames,
                context_no_grad=True,
                context_freqs_offset=0,
                render_block=render_block,
                denoising_kv_size=kv_size,
                denoising_steps=self.denoising_step_list,
                noise_generator=noise_generator,
            )


            # Step 3.2: record the model's output
            output[:, start_index:start_index + num_block_frame] = denoised_pred
            last_pred = denoised_pred.clone().detach()

            # Step 3.4: update the start and end frame indices
            start_index += num_block_frame


        # Step 4: Decode the output
        if not decode:
            return output

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize or reuse KV cache for the Wan model.
        Uses detach() + zero_() to safely reuse cache without gradient issues.
        Cache is allocated only once; subsequent calls only zero the existing tensors.
        """
        if self.kv_cache1 is not None and len(self.kv_cache1) == self.num_transformer_blocks \
                and self.kv_cache1[0]["k"].shape[0] == batch_size \
                and self.kv_cache1[0]["k"].dtype == dtype \
                and self.kv_cache1[0]["k"].device == device:
            for block_cache in self.kv_cache1:
                block_cache["k"].detach_().zero_()
                block_cache["v"].detach_().zero_()
            return

        kv_cache_size = self.frame_seq_length * 6
        num_heads = self.generator.model.config.num_heads
        dim = self.generator.model.config.dim

        print(f"Initializing kv cache with size: {kv_cache_size}")
        self.kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            self.kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
            })
