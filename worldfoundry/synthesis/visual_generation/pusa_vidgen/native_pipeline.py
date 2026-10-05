"""Native PUSA inference on the shared Wan model and staged runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.loaders import (
    WanConditioningComponents,
    load_wan_conditioning_components,
    load_wan_transformer_checkpoint,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WAN21_T2V_14B_CONFIG
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import PusaWanModel
from worldfoundry.base_models.diffusion_model.runners.wan_staged import WanStagedPipeline
from worldfoundry.core.model_loading import (
    GeneralLoRALoader,
    load_state_dict,
    merge_rank_scaled_lora_,
)


class PusaWanPipeline(WanStagedPipeline):
    """PUSA's dual-DiT inference policy over canonical Wan roles."""

    def __init__(
        self,
        *,
        high_noise_dit: PusaWanModel,
        low_noise_dit: PusaWanModel,
        conditioning: WanConditioningComponents,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_path=str(conditioning.tokenizer_path),
        )
        self.dit = high_noise_dit
        self.dit2 = low_noise_dit
        self.text_encoder = conditioning.text_encoder
        self.vae = conditioning.vae
        self.image_encoder = None
        self.prompter.fetch_models(self.text_encoder)
        self.model_names = ["text_encoder", "dit", "dit2", "vae"]

    def enable_vram_management(self, num_persistent_param_in_dit=None) -> None:
        """Enable role-level offload for the two mutually exclusive DiTs."""

        del num_persistent_param_in_dit
        self.enable_cpu_offload()

    def _encode_conditioning_image(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        tensor = self.preprocess_image(image, pattern="C H W").unsqueeze(1)
        return self.vae.encode(
            [tensor],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @staticmethod
    def _frame_step(
        scheduler,
        *,
        progress_id: int,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        frame_multipliers: torch.Tensor,
    ) -> torch.Tensor:
        sigma = scheduler.sigmas[progress_id].to(sample)
        if progress_id + 1 < len(scheduler.sigmas):
            next_sigma = scheduler.sigmas[progress_id + 1].to(sample)
        else:
            next_sigma = torch.zeros((), dtype=sample.dtype, device=sample.device)
        delta = (next_sigma - sigma) * frame_multipliers.to(sample)
        return sample + model_output * delta[:, None, :, None, None]

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        multi_frame_images: Mapping[int, tuple[Image.Image, float]] | None = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        cfg_scale: float = 3.0,
        num_inference_steps: int = 4,
        sigma_shift: float = 5.0,
        switch_DiT_boundary: float = 0.875,
        seed: int | None = None,
        rand_device: str | torch.device = "cpu",
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
    ) -> list[Image.Image]:
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        latent_frames = (num_frames - 1) // 4 + 1
        latent_shape = (1, 16, latent_frames, height // 8, width // 8)
        noise = self.generate_noise(
            latent_shape,
            seed=seed,
            rand_device=rand_device,
            dtype=torch.float32,
        ).to(dtype=self.torch_dtype, device=self.device)
        latents = noise.clone()

        frame_multipliers = torch.ones((1, latent_frames), dtype=self.torch_dtype, device=self.device)
        if multi_frame_images:
            self.load_models_to_device(["vae"])
            for latent_index, (image, multiplier) in multi_frame_images.items():
                if not 0 <= int(latent_index) < latent_frames:
                    raise IndexError(
                        f"PUSA conditioning latent index {latent_index} is outside [0, {latent_frames})"
                    )
                frame_multipliers[:, int(latent_index)] = float(multiplier)
                encoded = self._encode_conditioning_image(
                    image,
                    height=height,
                    width=width,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(latents)
                sigma = self.scheduler.sigmas[0].to(latents) * float(multiplier)
                latents[:, :, int(latent_index) : int(latent_index) + 1] = (
                    (1 - sigma) * encoded + sigma * noise[:, :, int(latent_index) : int(latent_index) + 1]
                )

        self.load_models_to_device(["text_encoder"])
        context_positive = self.encode_prompt(prompt, positive=True)["context"]
        context_negative = None
        if cfg_scale != 1.0:
            context_negative = self.encode_prompt(negative_prompt, positive=False)["context"]

        active_name = None
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            use_low_noise = float(timestep) < switch_DiT_boundary * self.scheduler.num_train_timesteps
            model_name = "dit2" if use_low_noise else "dit"
            if active_name != model_name:
                self.load_models_to_device([model_name])
                active_name = model_name
            dit = self.dit2 if use_low_noise else self.dit
            frame_timestep = timestep.to(device=self.device, dtype=self.torch_dtype).reshape(1, 1)
            frame_timestep = frame_timestep * frame_multipliers
            positive = dit(x=latents, timestep=frame_timestep, context=context_positive)
            if context_negative is None:
                noise_pred = positive
            else:
                negative = dit(x=latents, timestep=frame_timestep, context=context_negative)
                noise_pred = negative + cfg_scale * (positive - negative)
            latents = self._frame_step(
                self.scheduler,
                progress_id=progress_id,
                model_output=noise_pred,
                sample=latents,
                frame_multipliers=frame_multipliers,
            )

        self.load_models_to_device(["vae"])
        frames = self.vae.decode(
            latents.to(dtype=next(self.vae.parameters()).dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.load_models_to_device([])
        return self.tensor2video(frames[0])


def _merge_lora(
    model: torch.nn.Module,
    path: str | Path,
    *,
    alpha: float,
    rank_scaled: bool,
) -> None:
    state_dict = load_state_dict(str(path), torch_dtype=torch.float32, device="cpu")
    if rank_scaled:
        updated = merge_rank_scaled_lora_(model, state_dict, alpha=alpha)
    else:
        updated = GeneralLoRALoader(device="cpu", torch_dtype=torch.float32).load(
            model,
            state_dict,
            alpha=alpha,
        )
    if not updated:
        raise ValueError(f"PUSA LoRA did not match the native Wan graph: {path}")


def load_pusa_pipeline(
    *,
    base_model_root: str | Path,
    high_model_dir: str | Path,
    low_model_dir: str | Path,
    high_lora_path: str | Path,
    low_lora_path: str | Path,
    high_lora_alpha: float = 1.5,
    low_lora_alpha: float = 1.4,
    lightx2v_high_path: str | Path | None = None,
    lightx2v_low_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> PusaWanPipeline:
    """Assemble PUSA from explicit native Wan roles and checkpoint paths."""

    config = {**WAN21_T2V_14B_CONFIG, "per_token_timestep": True}
    high_noise_dit = load_wan_transformer_checkpoint(
        high_model_dir,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_class=PusaWanModel,
        transformer_config=config,
    )
    low_noise_dit = load_wan_transformer_checkpoint(
        low_model_dir,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_class=PusaWanModel,
        transformer_config=config,
    )
    conditioning = load_wan_conditioning_components(
        base_model_root,
        image_conditioned=False,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    if lightx2v_high_path is not None:
        _merge_lora(high_noise_dit, lightx2v_high_path, alpha=1.0, rank_scaled=True)
    if lightx2v_low_path is not None:
        _merge_lora(low_noise_dit, lightx2v_low_path, alpha=1.0, rank_scaled=True)
    _merge_lora(high_noise_dit, high_lora_path, alpha=high_lora_alpha, rank_scaled=False)
    _merge_lora(low_noise_dit, low_lora_path, alpha=low_lora_alpha, rank_scaled=False)
    return PusaWanPipeline(
        high_noise_dit=high_noise_dit,
        low_noise_dit=low_noise_dit,
        conditioning=conditioning,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["PusaWanPipeline", "load_pusa_pipeline"]
