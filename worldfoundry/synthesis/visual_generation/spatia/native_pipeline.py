"""Native Spatia sampling on shared Wan and VACE roles."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.loaders import load_wan_inference_components
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WAN22_TI2V_5B_CONFIG
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.spatia import (
    SpatiaWanModel,
    VaceWanModel,
)
from worldfoundry.base_models.diffusion_model.runners.wan_staged import WanStagedPipeline
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.core.model_loading import GeneralLoRALoader, load_model, load_state_dict


class SpatiaWanPipeline(WanStagedPipeline):
    """Long-horizon Spatia round sampler with explicit native model roles."""

    def __init__(
        self,
        *,
        components,
        vace: VaceWanModel,
        lora_state_dict: dict | None = None,
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
        self.vace = vace
        self.prompter.fetch_models(self.text_encoder)
        self.model_names = ["text_encoder", "dit", "vae", "vace"]
        self.height_division_factor = self.vae.upsampling_factor * self.dit.patch_size[1]
        self.width_division_factor = self.vae.upsampling_factor * self.dit.patch_size[2]
        self.lora_state_dict = lora_state_dict
        self._lora_loaded = False

    def enable_vram_management(self, num_persistent_param_in_dit=None) -> None:
        del num_persistent_param_in_dit
        self.enable_cpu_offload()

    def load_lora(self, module=None, lora_path=None, lora_state_dict=None, alpha=1.0) -> None:
        if self._lora_loaded:
            return
        state_dict = lora_state_dict or self.lora_state_dict
        if state_dict is None and lora_path:
            state_dict = load_state_dict(str(lora_path), torch_dtype=torch.float32, device="cpu")
        if state_dict is None:
            return
        updated = GeneralLoRALoader(device="cpu", torch_dtype=torch.float32).load(
            module or self.dit,
            state_dict,
            alpha=alpha,
        )
        if not updated:
            raise ValueError("Spatia LoRA did not match the native Wan graph")
        self._lora_loaded = True

    def unload_lora(self, module=None, lora_path=None, lora_state_dict=None, alpha=1.0) -> None:
        if not self._lora_loaded:
            return
        state_dict = lora_state_dict or self.lora_state_dict
        if state_dict is None and lora_path:
            state_dict = load_state_dict(str(lora_path), torch_dtype=torch.float32, device="cpu")
        if state_dict is None:
            return
        GeneralLoRALoader(device="cpu", torch_dtype=torch.float32).load(
            module or self.dit,
            state_dict,
            alpha=-float(alpha),
        )
        self._lora_loaded = False

    def _encode_frames(
        self,
        frames: Sequence[Image.Image | np.ndarray],
        *,
        height: int,
        width: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
        min_value: float = -1,
        max_value: float = 1,
    ) -> torch.Tensor:
        normalized = []
        for frame in frames:
            image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame).astype(np.uint8))
            normalized.append(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
        tensor = self.preprocess_video(
            normalized,
            torch_dtype=self.torch_dtype,
            device=self.device,
            min_value=min_value,
            max_value=max_value,
        )
        return self.vae.encode(
            tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(device=self.device, dtype=self.torch_dtype)

    def _encode_reference_images(
        self,
        images: Sequence[Image.Image],
        *,
        height: int,
        width: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor | None:
        if not images:
            return None
        values = []
        for image in images:
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            values.append(self.preprocess_image(image, pattern="C H W").unsqueeze(1))
        encoded = self.vae.encode(
            values,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return encoded.squeeze(2).transpose(0, 1).unsqueeze(0).contiguous().to(
            device=self.device,
            dtype=self.torch_dtype,
        )

    @torch.no_grad()
    def call_latent_inference(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        input_video: Sequence[Image.Image] | None = None,
        ar_hist_latents_num: int | None = None,
        ref_images: Sequence[Image.Image] | None = None,
        control_video: Sequence[Image.Image | np.ndarray] | None = None,
        control_score: Sequence[Image.Image | np.ndarray] | None = None,
        vace_scale: float = 1.0,
        seed: int | None = None,
        rand_device: str | torch.device = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
        return_latents: bool = False,
        sampler: str = "uni_pc",
        verbose: bool = True,
        **kwargs,
    ):
        del kwargs
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
        latent_frames = (num_frames - 1) // 4 + 1
        latent_channels = int(self.vae.model.z_dim)
        latent_height = height // self.vae.upsampling_factor
        latent_width = width // self.vae.upsampling_factor
        latents = self.generate_noise(
            (1, latent_channels, latent_frames, latent_height, latent_width),
            seed=seed,
            rand_device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["vae"])
        history_latents = None
        if input_video:
            history_latents = self._encode_frames(
                input_video,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            count = min(history_latents.shape[2], latents.shape[2])
            latents[:, :, :count] = history_latents[:, :, :count]
        first_frame_latent = None
        if input_image is not None:
            first_frame_latent = self._encode_frames(
                [input_image],
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            latents[:, :, :1] = first_frame_latent
        reference_latents = self._encode_reference_images(
            list(ref_images or ()),
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        num_ref_frames = 0 if reference_latents is None else reference_latents.shape[2]
        if reference_latents is not None:
            latents = torch.cat([reference_latents, latents], dim=2)

        vace_context = None
        if control_video is not None or control_score is not None:
            if control_video is None:
                control_video = [Image.new("RGB", (width, height)) for _ in range(num_frames)]
            if control_score is None:
                control_score = [np.full((height, width, 3), 255, dtype=np.uint8) for _ in range(num_frames)]
            control_latents = self._encode_frames(
                control_video,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            score_latents = self._encode_frames(
                control_score,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                min_value=0,
                max_value=1,
            )
            vace_context = torch.cat([control_latents, score_latents], dim=1)

        self.load_models_to_device(["text_encoder"])
        context_positive = self.encode_prompt(prompt, positive=True)["context"]
        context_negative = None
        if cfg_scale != 1.0:
            context_negative = self.encode_prompt(negative_prompt, positive=False)["context"]

        if sampler == "uni_pc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(num_inference_steps, device=self.device, shift=sigma_shift)
        elif sampler == "euler":
            scheduler = self.scheduler
            scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        else:
            raise ValueError(f"Unsupported Spatia sampler: {sampler}")

        history_count = min(int(ar_hist_latents_num or 0), latent_frames)
        frozen_count = num_ref_frames + history_count
        if first_frame_latent is not None:
            frozen_count = max(frozen_count, num_ref_frames + 1)
        frozen_prefix = latents[:, :, :frozen_count].clone() if frozen_count else None
        spatial_tokens = (latents.shape[3] // self.dit.patch_size[1]) * (
            latents.shape[4] // self.dit.patch_size[2]
        )
        generator = torch.Generator(device=self.device).manual_seed(int(seed or 0))

        self.load_models_to_device(["dit", "vace"])
        for progress_id, timestep in enumerate(progress_bar_cmd(scheduler.timesteps, disable=not verbose)):
            frame_timestep = torch.full(
                (1, latents.shape[2]),
                float(timestep),
                device=self.device,
                dtype=self.torch_dtype,
            )
            if frozen_count:
                frame_timestep[:, :frozen_count] = 0
            token_timestep = frame_timestep.repeat_interleave(spatial_tokens, dim=1)
            positive = self.dit(
                x=latents,
                timestep=token_timestep,
                context=context_positive,
                num_ref_frames=num_ref_frames,
                vace=self.vace,
                vace_context=vace_context,
                vace_scale=vace_scale,
            )
            if context_negative is None:
                noise_pred = positive
            else:
                negative = self.dit(
                    x=latents,
                    timestep=token_timestep,
                    context=context_negative,
                    num_ref_frames=num_ref_frames,
                    vace=self.vace,
                    vace_context=vace_context,
                    vace_scale=vace_scale,
                )
                noise_pred = negative + cfg_scale * (positive - negative)
            if sampler == "uni_pc":
                latents = scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]
            else:
                latents = scheduler.step(noise_pred, scheduler.timesteps[progress_id], latents)
            if frozen_prefix is not None:
                latents[:, :, :frozen_count] = frozen_prefix

        if num_ref_frames:
            latents = latents[:, :, num_ref_frames:]
        self.load_models_to_device(["vae"])
        decoded = self.vae.decode(
            latents.to(dtype=next(self.vae.parameters()).dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.load_models_to_device([])
        video = self.tensor2video(decoded[0])
        return (video, latents) if return_latents else video


def load_spatia_pipeline(
    *,
    model_root: str | Path,
    vace_path: str | Path,
    lora_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> SpatiaWanPipeline:
    """Assemble Spatia from the shared Wan2.2 roles and its VACE checkpoint."""

    config = {
        **WAN22_TI2V_5B_CONFIG,
        "seperated_timestep": True,
        "fuse_vae_embedding_in_latents": True,
    }
    components = load_wan_inference_components(
        model_root,
        image_conditioned=False,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_class=SpatiaWanModel,
        transformer_config=config,
    )
    vace_config = {
        "vace_layers": (0, 4, 8, 12, 16, 20, 24, 28),
        "vace_in_dim": 96,
        "dim": components.dit.dim,
        "patch_size": components.dit.patch_size,
        "num_heads": components.dit.blocks[0].num_heads,
        "ffn_dim": components.dit.blocks[0].ffn_dim,
        "has_image_input": False,
    }
    vace = load_model(
        VaceWanModel,
        str(vace_path),
        config=vace_config,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    lora_state_dict = None
    if lora_path:
        raw = load_state_dict(str(lora_path), torch_dtype=torch.float32, device="cpu")
        lora_state_dict = {
            key.removeprefix("pipe.dit."): value
            for key, value in raw.items()
            if key.startswith("pipe.dit.")
        }
        if not lora_state_dict:
            lora_state_dict = raw
    return SpatiaWanPipeline(
        components=components,
        vace=vace,
        lora_state_dict=lora_state_dict,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["SpatiaWanPipeline", "load_spatia_pipeline"]
