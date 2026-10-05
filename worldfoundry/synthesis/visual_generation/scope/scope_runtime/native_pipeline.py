"""Minimal native inference pipeline for the SCOPE Wan2.2 variant."""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import WanVideoVAE38
from worldfoundry.base_models.diffusion_model.models.encoders.wan import WanPrompter, WanTextEncoder
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import ScopeActionWanModel
from worldfoundry.base_models.diffusion_model.runners.staged import StagedDiffusionPipeline
from worldfoundry.core.nn import FlowMatchScheduler


class ScopeVideoPipeline(StagedDiffusionPipeline):
    """SCOPE-specific conditioning over the shared Wan inference primitives."""

    def __init__(
        self,
        *,
        dit: ScopeActionWanModel,
        text_encoder: WanTextEncoder,
        vae: WanVideoVAE38,
        tokenizer_path: str,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=vae.upsampling_factor * 2,
            width_division_factor=vae.upsampling_factor * 2,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.dit = dit
        self.text_encoder = text_encoder
        self.vae = vae
        self.model_names = ("text_encoder", "dit", "vae")
        self.scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.prompter = WanPrompter(tokenizer_path)
        self.prompter.fetch_models(text_encoder)

    def _encode_prompt(self, prompt: str, *, positive: bool) -> torch.Tensor:
        self.load_models_to_device(("text_encoder",))
        return self.prompter.encode_prompt(prompt, positive=positive, device=self.device).to(
            dtype=self.torch_dtype,
            device=self.device,
        )

    def _encode_first_frame(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        self.load_models_to_device(("vae",))
        image_tensor = self.preprocess_image(image.resize((width, height))).transpose(0, 1)
        return self.vae.encode(
            [image_tensor],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.torch_dtype, device=self.device)

    @staticmethod
    def _token_timesteps(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = latents.shape
        tokens_per_frame = (height // 2) * (width // 2)
        token_timestep = timestep.reshape(batch, 1).expand(batch, frames * tokens_per_frame).clone()
        token_timestep[:, :tokens_per_frame] = 0
        return token_timestep

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image,
        mouse_action: torch.Tensor,
        keyboard_action: torch.Tensor,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 30,
        sigma_shift: float = 5.0,
        cfg_scale: float = 5.0,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd: Callable = tqdm,
    ) -> list[Image.Image]:
        if input_image is None:
            raise ValueError("SCOPE requires input_image")
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        latent_shape = (
            1,
            self.vae.z_dim,
            (num_frames - 1) // 4 + 1,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        latents = self.generate_noise(
            latent_shape,
            seed=seed,
            rand_device=rand_device,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        first_frame = self._encode_first_frame(
            input_image,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, :1] = first_frame

        context_positive = self._encode_prompt(prompt, positive=True)
        context_negative = None
        if cfg_scale != 1.0:
            context_negative = self._encode_prompt(negative_prompt, positive=False)
        mouse_action = mouse_action.to(dtype=self.torch_dtype, device=self.device)
        keyboard_action = keyboard_action.to(dtype=self.torch_dtype, device=self.device)

        self.load_models_to_device(("dit",))
        for timestep in progress_bar_cmd(self.scheduler.timesteps):
            scalar_timestep = timestep.reshape(1).to(dtype=self.torch_dtype, device=self.device)
            token_timestep = self._token_timesteps(latents, scalar_timestep)
            positive = self.dit(
                x=latents,
                timestep=token_timestep,
                context=context_positive,
                mouse_action=mouse_action,
                keyboard_action=keyboard_action,
            )
            if context_negative is None:
                prediction = positive
            else:
                negative = self.dit(
                    x=latents,
                    timestep=token_timestep,
                    context=context_negative,
                    mouse_action=mouse_action,
                    keyboard_action=keyboard_action,
                )
                prediction = negative + cfg_scale * (positive - negative)
            latents = self.scheduler.step(prediction, timestep, latents)
            latents[:, :, :1] = first_frame

        self.load_models_to_device(("vae",))
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.load_models_to_device(())
        return self.vae_output_to_video(video)


__all__ = ["ScopeVideoPipeline"]
