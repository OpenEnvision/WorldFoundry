# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Contract-driven sampler with separate image- and text-to-video routing."""

from typing import Any, Callable
import math
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from ..models.networks.worldcrafter.fast.contract import resolve_dmd_inference_trace
from ..models.networks.worldcrafter.fast.camera import build_ucpe_attention_kwargs_sequential_pyramid, pyramid_token_grids
from .helios import XLA_AVAILABLE


def sample_fast(
    self,
    latents: torch.Tensor = None,
    pyramid_num_stages: int = None,
    pyramid_num_inference_steps_list: list[int] = None,
    prompt_embeds: torch.Tensor = None,
    guidance_scale: float | None = 1.0,
    indices_hidden_states: torch.Tensor = None,
    indices_latents_history_short: torch.Tensor = None,
    indices_latents_history_mid: torch.Tensor = None,
    indices_latents_history_long: torch.Tensor = None,
    latents_history_short: torch.Tensor = None,
    latents_history_mid: torch.Tensor = None,
    latents_history_long: torch.Tensor = None,
    attention_kwargs: dict | None = None,
    camera_trajectory: dict[str, Any] | None = None,
    num_latent_frames_per_chunk: int | None = None,
    chunk_index: int | None = None,
    camera_restart_each_chunk: bool = True,
    camera_translation_scale: float = 1.0,
    ucpe_pixel_center: bool = False,
    device: torch.device | None = None,
    transformer_dtype: torch.dtype = None,
    generator: torch.Generator | None = None,
    callback_on_step_end: (
        Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None
    ) = None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    progress_bar=None,
):
    """Run the checkpoint's three-stage DMD schedule with CFG=1."""
    if not self.config.is_distilled or guidance_scale != 1.0:
        raise ValueError("Fast sampling requires a distilled model and CFG=1")
    if pyramid_num_inference_steps_list is not None:
        raise ValueError("Fast step counts are defined by the checkpoint")
    controller = self.resident_branches
    stage_models = self.stage_transformers
    if len(stage_models) != 3:
        raise ValueError("Fast sampling requires three stage models")
    dmd_trace = resolve_dmd_inference_trace(
        self.dmd_timestep_contract,
        latent_shape=latents.shape[1:],
        history_tensors=(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ),
        num_stages=pyramid_num_stages,
    )
    batch_size, num_channel, num_frames, height, width = latents.shape
    patch_size = self.transformer.config.patch_size
    ucpe_attention_kwargs_per_stage = None
    if (
        camera_trajectory is not None
        and num_latent_frames_per_chunk is not None
        and (chunk_index is not None)
    ):
        ucpe_attention_kwargs_per_stage = (
            build_ucpe_attention_kwargs_sequential_pyramid(
                self.transformer,
                camera_trajectory,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                chunk_index=chunk_index,
                token_grids=pyramid_token_grids(
                    height // patch_size[1],
                    width // patch_size[2],
                    pyramid_num_stages,
                    low_to_high=True,
                ),
                vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                restart_each_chunk=camera_restart_each_chunk,
                translation_scale=camera_translation_scale,
                pixel_center=ucpe_pixel_center,
            )
        )
    latents = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, num_channel, height, width
    )
    for _ in range(pyramid_num_stages - 1):
        height //= 2
        width //= 2
        latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
    latents = latents.reshape(
        batch_size, num_frames, num_channel, height, width
    ).permute(0, 2, 1, 3, 4)
    batch_size = latents.shape[0]
    start_point_list = [latents]
    i = 0
    for i_s in range(pyramid_num_stages):
        stage_contract = self.dmd_timestep_contract.stage_tensors(
            i_s, empty_history=dmd_trace.empty_history, device=device
        )
        self.scheduler.timesteps = stage_contract["model_timestep"]
        self.scheduler.sigmas = stage_contract["current_sigma"]
        timesteps = self.scheduler.timesteps
        if i_s > 0:
            height *= 2
            width *= 2
            num_frames = latents.shape[2]
            latents = latents.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frames, num_channel, height // 2, width // 2
            )
            latents = F.interpolate(latents, size=(height, width), mode="nearest")
            latents = latents.reshape(
                batch_size, num_frames, num_channel, height, width
            ).permute(0, 2, 1, 3, 4)
            ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
            gamma = self.scheduler.config.gamma
            alpha = 1 / (math.sqrt(1 + 1 / gamma) * (1 - ori_sigma) + ori_sigma)
            beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
            batch_size, channel, num_frames, height, width = latents.shape
            noise = self.sample_block_noise(
                batch_size,
                channel,
                num_frames,
                height,
                width,
                patch_size,
                device,
                generator,
            )
            noise = noise.to(device=device, dtype=transformer_dtype)
            latents = alpha * latents + beta * noise
            start_point_list.append(latents)
        stage_attention_kwargs = dict(attention_kwargs or {})
        if ucpe_attention_kwargs_per_stage is not None:
            stage_attention_kwargs.update(ucpe_attention_kwargs_per_stage[i_s])
        for idx, t in enumerate(timesteps):
            use_low_noise = (
                i_s >= 1
                if getattr(self, "fast_inference_mode", "i2v") == "t2v"
                else i_s == 2 and idx >= len(timesteps) // 2
            )
            branch = "old" if use_low_noise else "equal"
            controller.switch(branch)
            stage_transformer = stage_models[2 if branch == "old" else 0]
            self.stage_forward_context = (
                int(chunk_index),
                int(i_s),
                int(idx),
                int(len(timesteps)),
            )
            timestep = self.dmd_timestep_contract.student_condition(
                t, latents.shape[0], device=latents.device
            )
            with stage_transformer.cache_context("cond"):
                noise_pred = stage_transformer(
                    hidden_states=latents.to(transformer_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=stage_attention_kwargs,
                    return_dict=False,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                )[0]
            current_sigma = stage_contract["current_sigma"][idx].float()
            next_sigma = stage_contract["next_sigma"][idx].float()
            pred_image_or_video = latents.float() - current_sigma * noise_pred.float()
            latents = self.dmd_timestep_contract.renoise_x0(
                pred_image_or_video, start_point_list[i_s].float(), next_sigma
            )
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            progress_bar.update()
            if XLA_AVAILABLE:
                from .helios import xm

                xm.mark_step()
            i += 1
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Fast denoising produced NaN or Inf")
    return latents
