"""Model adapter used by causal autoregressive inference."""

from __future__ import annotations

from dataclasses import replace

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig
from worldfoundry.base_models.diffusion_model.models.representations.ltx.echo_wm import EchoWMModality as Modality

from ..ltx_core.utils import to_denoised
from .cache import CausalCacheConfig


class CausalModelWrapper(torch.nn.Module):
    """Adapt the release LTX velocity model to causal x0 prediction."""

    def __init__(self, model: torch.nn.Module, patches_per_frame: int, cache: CausalCacheConfig):
        super().__init__()
        cache.validate()
        self.model = model
        self.patches_per_frame = patches_per_frame
        self.cache = cache

    def forward(
        self,
        video: Modality,
        audio: Modality | None,
        action_cond: dict[str, torch.Tensor] | None,
        kv_caches: list[dict],
        video_start_frame: int,
        audio_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video = replace(video, sigma=torch.ones_like(video.sigma))
        if audio is not None:
            audio = replace(audio, sigma=torch.ones_like(audio.sigma))
        velocity_video, velocity_audio = self.model(
            video=video,
            audio=audio,
            perturbations=BatchedPerturbationConfig.empty(video.latent.shape[0], num_blocks=self.model.num_blocks),
            action_cond=action_cond,
            kv_caches=kv_caches,
            current_video_token_start=video_start_frame * self.patches_per_frame,
            current_audio_token_start=audio_start_frame,
        )
        video_sigma = video.timesteps.unsqueeze(-1) if video.timesteps.ndim == 2 else video.timesteps
        video_x0 = to_denoised(video.latent, velocity_video, video_sigma)
        audio_x0 = None
        if audio is not None and velocity_audio is not None:
            audio_sigma = audio.timesteps.unsqueeze(-1) if audio.timesteps.ndim == 2 else audio.timesteps
            audio_x0 = to_denoised(audio.latent, velocity_audio, audio_sigma)
        return video_x0, audio_x0

    def init_caches(
        self,
        *,
        batch_size: int,
        video_frames: int,
        audio_frames: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict]:
        patches_per_frame = self.patches_per_frame
        return self.model.init_av_kv_caches(
            batch_size=batch_size,
            max_video_tokens=video_frames * patches_per_frame,
            max_audio_tokens=audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_local_attn_tokens=self.cache.video_local_attn_size * patches_per_frame,
            video_sink_tokens=self.cache.video_sink_size * patches_per_frame,
            video_ucpe_local_attn_tokens=self.cache.video_local_attn_size * patches_per_frame,
            video_ucpe_sink_tokens=self.cache.video_sink_size * patches_per_frame,
            audio_local_attn_tokens=self.cache.audio_local_attn_size,
            audio_sink_tokens=self.cache.audio_sink_size,
        )
