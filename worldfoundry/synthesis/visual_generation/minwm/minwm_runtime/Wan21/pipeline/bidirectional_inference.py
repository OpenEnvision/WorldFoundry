import time

import torch

from ..wan_utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=False)
            if generator is None
            else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Latency to first denoised latent (excludes VAE decode), set per-call.
        self.last_chunk0_latency = None

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        return_latents=False,
        initial_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        # Start chunk0 latency timer AFTER text encoder, BEFORE VAE decode
        # — matches HY15 latency definition (excludes both text encoder and decode).
        torch.cuda.synchronize()
        _chunk0_t0 = time.perf_counter()
        self.last_chunk0_latency = None

        # initial point
        noisy_image_or_video = noise

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list[:-1]):
            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
            )  # [B, F, C, H, W]

            next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2], dtype=torch.long, device=noise.device
            )

            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1),
            ).unflatten(0, noise.shape[:2])

        # Stop chunk0 timer once the first (and only) denoised latent is ready,
        # before VAE decode — matches HY15 latency definition.
        torch.cuda.synchronize()
        self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, pred_image_or_video
        return video
