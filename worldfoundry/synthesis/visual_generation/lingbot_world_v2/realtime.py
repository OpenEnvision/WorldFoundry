"""Persistent low-latency rollout for LingBot-World-V2.

V2 shares the in-tree causal Wan session machinery with LingBot-World Fast,
but advertises its own four-latent cadence, attention window, denoising
schedule, and camera normalization.  No subprocess, MP4, or checkpoint reload
is allowed after the session has been prepared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import (
    LingBotRealtimeSession,
    _is_fsdp_module,
)

if TYPE_CHECKING:
    import torch

    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.conditioning import WanZeroConditionCache
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.streaming import StreamingWanDecoder
    from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import StaticConditionCache


class LingBotWorldV2RealtimeSession(LingBotRealtimeSession):
    """One persistent four-latent LingBot-World-V2 causal rollout."""

    chunk_latent_frames = 4
    first_output_frames = 13
    steady_output_frames = 16
    temporal_window_frames = 18
    sink_frames = 6
    cache_frames = 18
    timestep_indices = (0, 250, 500, 750)
    camera_move_speed_per_second = 1.6
    camera_rotate_speed_degrees_per_second = 3.2
    normalize_camera_translation = True
    compile_namespace = "lingbot-world-v2"
    compile_env = "WORLDFOUNDRY_LINGBOT_V2_REALTIME_COMPILE"
    condition_prefetch_chunks = 8
    condition_replenish_chunks = 4


class LingBotWorldV2SmallRealtimeSession(LingBotWorldV2RealtimeSession):
    """1.3B reference sampler with four latent frames and request-owned FP32 noise."""

    sample_shift = 5.0
    reference_sampling = True
    reference_noise_chunks = 20
    compile_namespace = "lingbot-world-v2-1.3b"

    def _new_text_condition(self, context: list[torch.Tensor]) -> StaticConditionCache | None:
        if self.sp_size != 1 or _is_fsdp_module(self.model) or hasattr(self.model, "_orig_mod"):
            return None
        from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import StaticConditionCache

        return StaticConditionCache(context[0].unsqueeze(0))

    def _prepare_rotary_frequencies(
        self,
        latent: torch.Tensor,
        *,
        current_start: int,
    ) -> tuple[torch.Tensor, ...] | None:
        if self.sp_size != 1 or _is_fsdp_module(self.model) or hasattr(self.model, "_orig_mod"):
            return None
        return self.raw_model.prepare_rotary_frequencies(
            [latent], current_start=current_start, frame_seqlen=self.frame_sequence_length,
        )

    def _new_condition_cache(self) -> WanZeroConditionCache | None:
        if self.sp_size != 1:
            return None
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.conditioning import WanZeroConditionCache

        return WanZeroConditionCache(self.core.vae.model.encoder)

    def _load_decoder(self):
        if self._decoder is not None:
            return
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.streaming import StreamingWanDecoder
        from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

        convert_convolution_weight_layouts(self.core.vae.model, conv2d=True, conv3d=True)
        self._decoder_model = self.core.vae.model.decoder
        self._decoder = StreamingWanDecoder(self.core.vae, channels_last_3d=True,
                                            compile_decoder=self.core.vae_streaming_compile)

    def _decode(self, latent):
        import torch

        decoder = cast("StreamingWanDecoder", self._decoder)
        video = decoder.decode(latent.unsqueeze(0))[0]
        if video.shape[1] != self.next_output_frames():
            raise RuntimeError(f"Wan decoder produced {video.shape[1]} frames; expected {self.next_output_frames()}")
        return video.permute(1, 2, 3, 0).add_(1.).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


__all__ = ["LingBotWorldV2RealtimeSession", "LingBotWorldV2SmallRealtimeSession"]
