"""Native SAMA inference built from shared Wan roles."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.loaders import (
    WanInferenceComponents,
    load_wan_inference_components,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WAN21_T2V_14B_CONFIG
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import SamaWanModel
from worldfoundry.base_models.diffusion_model.runners.wan_staged import WanStagedPipeline
from worldfoundry.core.model_loading import GeneralLoRALoader, load_state_dict


class SamaWanPipeline(WanStagedPipeline):
    """Joint video/semantic diffusion with source-video token conditioning."""

    def __init__(
        self,
        *,
        components: WanInferenceComponents,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_path=str(components.tokenizer_path),
        )
        self.dit = components.dit
        self.text_encoder = components.text_encoder
        self.vae = components.vae
        self.image_encoder = None
        self.prompter.fetch_models(self.text_encoder)
        self.model_names = ["text_encoder", "dit", "vae"]

    def enable_vram_management(self, num_persistent_param_in_dit=None) -> None:
        del num_persistent_param_in_dit
        self.enable_cpu_offload()

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        source_video: list[Image.Image],
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        semantic_token_count: int = 65,
        semantic_dim: int = 1152,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        seed: int | None = None,
        rand_device: str | torch.device = "cpu",
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
    ) -> list[Image.Image]:
        if not source_video:
            raise ValueError("SAMA requires a non-empty source_video")
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
        frames = list(source_video[:num_frames])
        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        frames = [frame.convert("RGB").resize((width, height), Image.Resampling.LANCZOS) for frame in frames]

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        latent_frames = (num_frames - 1) // 4 + 1
        latent_shape = (1, 16, latent_frames, height // 8, width // 8)
        latents = self.generate_noise(
            latent_shape,
            seed=seed,
            rand_device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        semantic_seed = None if seed is None else int(seed) + 1
        semantic_latents = self.generate_noise(
            (1, semantic_token_count, semantic_dim),
            seed=semantic_seed,
            rand_device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["vae"])
        source_tensor = self.preprocess_video(
            frames,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        source_latents = self.vae.encode(
            source_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["text_encoder"])
        context_positive = self.encode_prompt(prompt, positive=True)["context"]
        context_negative = None
        if cfg_scale != 1.0:
            context_negative = self.encode_prompt(negative_prompt, positive=False)["context"]

        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_batch = timestep.to(device=self.device, dtype=self.torch_dtype).reshape(1)
            positive, semantic_positive = self.dit(
                x=latents,
                timestep=timestep_batch,
                context=context_positive,
                source_latents=source_latents,
                semantic_latents=semantic_latents,
            )
            if context_negative is None:
                noise_pred = positive
                semantic_pred = semantic_positive
            else:
                negative, semantic_negative = self.dit(
                    x=latents,
                    timestep=timestep_batch,
                    context=context_negative,
                    source_latents=source_latents,
                    semantic_latents=semantic_latents,
                )
                noise_pred = negative + cfg_scale * (positive - negative)
                semantic_pred = semantic_negative + cfg_scale * (semantic_positive - semantic_negative)
            latents = self.scheduler.step(noise_pred, timestep, latents)
            semantic_latents = self.scheduler.step(semantic_pred, timestep, semantic_latents)

        self.load_models_to_device(["vae"])
        decoded = self.vae.decode(
            latents.to(dtype=next(self.vae.parameters()).dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.load_models_to_device([])
        return self.tensor2video(decoded[0])


def load_sama_pipeline(
    model_root: str | Path,
    *,
    state_dict_path: str | Path | None = None,
    lora_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> SamaWanPipeline:
    """Load SAMA's native Wan variant and optional released adaptation."""

    components = load_wan_inference_components(
        model_root,
        image_conditioned=False,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_class=SamaWanModel,
        transformer_config=WAN21_T2V_14B_CONFIG,
        transformer_strict=False,
    )
    if state_dict_path:
        state_dict = load_state_dict(str(state_dict_path), torch_dtype=torch_dtype, device="cpu")
        incompatible = components.dit.load_state_dict(state_dict, strict=False)
        matched = len(state_dict) - len(incompatible.unexpected_keys)
        if matched <= 0:
            raise ValueError(f"SAMA state dict did not match the native Wan variant: {state_dict_path}")
    if lora_path:
        state_dict = load_state_dict(str(lora_path), torch_dtype=torch.float32, device="cpu")
        updated = GeneralLoRALoader(device="cpu", torch_dtype=torch.float32).load(
            components.dit,
            state_dict,
            alpha=1.0,
        )
        if not updated:
            raise ValueError(f"SAMA LoRA did not match the native Wan variant: {lora_path}")
    return SamaWanPipeline(components=components, device=device, torch_dtype=torch_dtype)


__all__ = ["SamaWanPipeline", "load_sama_pipeline"]
