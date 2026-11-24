# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo-WM's camera/cache interfaces over the shared LTX graph.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from typing import Any, cast

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LegacyX0Model, X0Model
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import TransformerConfig
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.echo_wm import EchoWMControl, EchoWMModality
from worldfoundry.core.nn import velocity_to_denoised as to_denoised

from .model import JoyAIEchoModel
from .transformer import ActionBlockConfig, EchoWMTransformerBlock
from .transformer_args import EchoWMArgsPreprocessor, EchoWMAVArgsPreprocessor


class EchoWMModel(JoyAIEchoModel):
    """Add Echo-WM controls without retaining request state on the shared backbone."""

    TRANSFORMER_BLOCK_CLS = EchoWMTransformerBlock

    def _init_preprocessors(self, cross_pe_max_pos: int | None = None) -> None:
        super()._init_preprocessors(cross_pe_max_pos)
        for name in ("video_args_preprocessor", "audio_args_preprocessor"):
            preprocessor = getattr(self, name, None)
            if preprocessor is None:
                continue
            if isinstance(preprocessor, MultiModalTransformerArgsPreprocessor):
                preprocessor.__class__ = EchoWMAVArgsPreprocessor
                preprocessor.simple_preprocessor.__class__ = EchoWMArgsPreprocessor
            else:
                preprocessor.__class__ = EchoWMArgsPreprocessor

    def _init_transformer_blocks(self, *args: Any, **kwargs: Any) -> None:
        super()._init_transformer_blocks(*args, **kwargs)
        for index, block in enumerate(self.transformer_blocks):
            block.idx = index
            block.num_layers = self.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
        action_cond: dict[str, torch.Tensor] | None = None,
        kv_caches: list[dict[str, Any]] | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        control = EchoWMControl(action_cond, kv_caches, current_video_token_start, current_audio_token_start)
        if video is not None:
            video_fields: dict[str, Any] = vars(video) | {"control": control}
            video = EchoWMModality(**video_fields)
        if audio is not None:
            audio_fields: dict[str, Any] = vars(audio) | {"control": control}
            audio = EchoWMModality(**audio_fields)
        return super().forward(video, audio, perturbations)

    def enable_action_conditioning(self, action_config: ActionBlockConfig) -> None:
        """Attach the optional pure-UCPE branch after base weights are loaded."""
        if not isinstance(action_config, ActionBlockConfig):
            raise TypeError(f"action_config must be ActionBlockConfig, got {type(action_config)}")
        if not self.model_type.is_video_enabled():
            raise RuntimeError("UCPE conditioning requires a video-enabled model")
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.inner_dim // self.num_attention_heads,
            context_dim=0,
        )
        for block in self.transformer_blocks:
            block._init_action_params(video_config, action_config)

    def init_av_kv_caches(
        self,
        batch_size: int,
        max_video_tokens: int,
        max_audio_tokens: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        video_local_attn_tokens: int = -1,
        video_sink_tokens: int = 0,
        video_ucpe_local_attn_tokens: int | None = None,
        video_ucpe_sink_tokens: int | None = None,
        audio_local_attn_tokens: int = -1,
        audio_sink_tokens: int = 0,
    ) -> list[dict]:
        """Allocate inference-only per-layer AV caches."""

        def allocate(max_tokens: int, dim: int, local: int, sink: int) -> dict:
            capacity = max_tokens if local < 0 else min(max_tokens, local)
            if capacity <= 0:
                raise ValueError("KV cache capacity must be positive")
            return {
                "k": torch.zeros(batch_size, capacity, dim, device=device, dtype=dtype),
                "v": torch.zeros(batch_size, capacity, dim, device=device, dtype=dtype),
                "positions": torch.full((capacity,), -1, device=device, dtype=torch.long),
                "length": 0,
                "local_attn_size": local,
                "sink_tokens": sink,
            }

        def allocate_text(dim: int) -> dict:
            return {
                "k": torch.zeros(batch_size, text_seq_len, dim, device=device, dtype=dtype),
                "v": torch.zeros(batch_size, text_seq_len, dim, device=device, dtype=dtype),
                "length": 0,
                "is_init": False,
            }

        ucpe_local = video_local_attn_tokens if video_ucpe_local_attn_tokens is None else video_ucpe_local_attn_tokens
        ucpe_sink = video_sink_tokens if video_ucpe_sink_tokens is None else video_ucpe_sink_tokens
        caches: list[dict] = []
        for block in self.transformer_blocks:
            layer: dict = {}
            if self.model_type.is_video_enabled():
                layer["video_self"] = allocate(
                    max_video_tokens, self.inner_dim, video_local_attn_tokens, video_sink_tokens
                )
                layer["video_text"] = allocate_text(self.inner_dim)
                if block.action_ucpe_enabled:
                    layer["video_ucpe"] = allocate(
                        max_video_tokens,
                        block.ucpe_num_heads * block.ucpe_head_dim,
                        ucpe_local,
                        ucpe_sink,
                    )
            if self.model_type.is_audio_enabled() and max_audio_tokens > 0:
                layer["audio_self"] = allocate(
                    max_audio_tokens, self.audio_inner_dim, audio_local_attn_tokens, audio_sink_tokens
                )
                layer["audio_text"] = allocate_text(self.audio_inner_dim)
            if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled() and max_audio_tokens > 0:
                layer["a2v"] = allocate(
                    max_audio_tokens, self.audio_inner_dim, audio_local_attn_tokens, audio_sink_tokens
                )
                layer["v2a"] = allocate(
                    max_video_tokens, self.audio_inner_dim, video_local_attn_tokens, video_sink_tokens
                )
            caches.append(layer)
        return caches


class EchoWMLegacyX0Model(LegacyX0Model):
    """Convert Echo-WM velocity while forwarding its camera condition."""

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
        action_cond: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations, action_cond=action_cond)
        denoised_video = to_denoised(cast(Modality, video).latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(cast(Modality, audio).latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class EchoWMX0Model(X0Model):
    """Convert Echo-WM velocity while forwarding its camera condition."""

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        action_cond: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations, action_cond=action_cond)
        denoised_video = (
            to_denoised(cast(Modality, video).latent, vx, cast(Modality, video).timesteps) if vx is not None else None
        )
        denoised_audio = (
            to_denoised(cast(Modality, audio).latent, ax, cast(Modality, audio).timesteps) if ax is not None else None
        )
        return denoised_video, denoised_audio
