"""HyDRA video pipeline built on the native Wan staged runner."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from worldfoundry.core.model_loading import load_state_dict

from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
from worldfoundry.base_models.diffusion_model.runners import TeaCache, WanStagedPipeline
from worldfoundry.core.nn import sinusoidal_embedding_1d

from .hydra_model import HyDRAAttentionConfig, configure_hydra_model


class HyDRAPipeline(WanStagedPipeline):
    """Generate a target view from a context video and two camera tracks."""

    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, tokenizer_path=None) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.hydra_config: HyDRAAttentionConfig | None = None

    @classmethod
    def from_components(
        cls,
        *,
        text_encoder,
        dit,
        vae,
        tokenizer_path: str,
        torch_dtype: torch.dtype,
        device: str | torch.device,
    ) -> "HyDRAPipeline":
        """Create a HyDRA pipeline from explicitly loaded native components."""

        pipe = cls(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        pipe.text_encoder = text_encoder
        pipe.dit = dit
        pipe.vae = vae
        pipe.prompter.fetch_models(text_encoder)
        return pipe

    def configure_hydra(
        self,
        *,
        enabled: bool = True,
        config: HyDRAAttentionConfig | None = None,
    ) -> "HyDRAPipeline":
        """Attach HyDRA modules to the loaded shared Wan DiT."""

        if self.dit is None:
            raise RuntimeError("Load a Wan video DiT before configuring HyDRA")
        self.hydra_config = config or HyDRAAttentionConfig()
        configure_hydra_model(self.dit, enabled=enabled, config=self.hydra_config)
        return self

    def load_hydra_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        enabled: bool = True,
        config: HyDRAAttentionConfig | None = None,
    ) -> "HyDRAPipeline":
        """Configure the shared Wan DiT and strictly load a HyDRA checkpoint."""

        self.configure_hydra(enabled=enabled, config=config)
        state_dict = load_state_dict(checkpoint_path)
        self.dit.load_state_dict(state_dict, strict=True)
        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        source_video: torch.Tensor | None = None,
        target_camera: torch.Tensor | None = None,
        ref_camera: torch.Tensor | None = None,
        ref_input: torch.Tensor | None = None,
        frame_ids=None,
        input_image=None,
        input_video=None,
        denoising_strength: float = 1.0,
        seed: int | None = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 77,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        tea_cache_l1_thresh: float | None = None,
        tea_cache_model_id: str = "",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        """Run HyDRA inference.

        Returns:
            A pair ``(generated_frames, context_frames)``.  The second item is
            retained for compatibility with the original HyDRA inference API.
        """

        del frame_ids, input_image
        if source_video is None:
            raise ValueError("HyDRA requires source_video")
        if target_camera is None or ref_camera is None:
            raise ValueError("HyDRA requires target_camera and ref_camera")
        if self.hydra_config is None:
            self.configure_hydra()

        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only num_frames % 4 == 1 is supported; rounded to {num_frames}.")
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            prepared_video = self.preprocess_images(input_video)
            prepared_video = torch.stack(prepared_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(prepared_video, **tiler_kwargs).to(
                dtype=self.torch_dtype,
                device=self.device,
            )
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        self.load_models_to_device(["vae"])
        source_video = source_video.to(dtype=self.torch_dtype, device=self.device)
        source_latents = self.encode_video(source_video.unsqueeze(0), **tiler_kwargs).to(
            dtype=self.torch_dtype,
            device=self.device,
        )
        if ref_input is not None:
            ref_input = ref_input.to(dtype=self.torch_dtype, device=self.device)
            ref_latents = self.encode_video(ref_input.unsqueeze(0), **tiler_kwargs).to(
                dtype=self.torch_dtype,
                device=self.device,
            )
        else:
            ref_latents = None

        cam_emb_tgt = target_camera.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        cam_emb_con = ref_camera.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False) if cfg_scale != 1.0 else None
        extra_input = self.prepare_extra_input(latents)

        tea_cache_posi = (
            TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        )
        tea_cache_nega = (
            TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        )

        self.load_models_to_device(["dit"])
        target_offset = source_latents.shape[2] + (ref_latents.shape[2] if ref_latents is not None else 0)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            model_timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            latent_parts = [source_latents, latents]
            if ref_latents is not None:
                latent_parts.insert(0, ref_latents)
            latents_input = torch.cat(latent_parts, dim=2)
            noise_pred_posi = model_fn_hydra(
                self.dit,
                latents_input,
                timestep=model_timestep,
                cam_emb_tgt=cam_emb_tgt,
                cam_emb_con=cam_emb_con,
                tea_cache=tea_cache_posi,
                **prompt_emb_posi,
                **extra_input,
            )
            if prompt_emb_nega is not None:
                noise_pred_nega = model_fn_hydra(
                    self.dit,
                    latents_input,
                    timestep=model_timestep,
                    cam_emb_tgt=cam_emb_tgt,
                    cam_emb_con=cam_emb_con,
                    tea_cache=tea_cache_nega,
                    **prompt_emb_nega,
                    **extra_input,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            latents = self.scheduler.step(
                noise_pred[:, :, target_offset:],
                self.scheduler.timesteps[progress_id],
                latents,
            )
            if progress_bar_st is not None:
                progress_bar_st.progress((progress_id + 1) / len(self.scheduler.timesteps))

        self.load_models_to_device(["vae"])
        generated_frames = self.tensor2video(self.decode_video(latents, **tiler_kwargs)[0])
        context_frames = self.tensor2video(source_video)
        self.load_models_to_device([])
        return generated_frames, context_frames


def model_fn_hydra(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb_tgt: torch.Tensor,
    cam_emb_con: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache | None = None,
    **kwargs,
) -> torch.Tensor:
    """Run one HyDRA-conditioned Wan DiT denoising pass."""

    del kwargs
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)

    x, (frames, height, width) = dit.patchify(x)
    config: HyDRAAttentionConfig = dit.hydra_config
    actual_grid = (frames, height, width)
    expected_grid = (config.num_frames, config.frame_height, config.frame_width)
    if actual_grid != expected_grid:
        raise ValueError(f"HyDRA checkpoint expects token grid {expected_grid}, got {actual_grid}")
    freqs = torch.cat(
        [
            dit.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            dit.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            dit.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1).to(x.device)

    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block in dit.blocks:
            x = block(
                x,
                context,
                t_mod,
                freqs,
                cam_emb_tgt=cam_emb_tgt,
                cam_emb_con=cam_emb_con,
            )
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    return dit.unpatchify(x, (frames, height, width))


__all__ = ["HyDRAPipeline", "model_fn_hydra"]
