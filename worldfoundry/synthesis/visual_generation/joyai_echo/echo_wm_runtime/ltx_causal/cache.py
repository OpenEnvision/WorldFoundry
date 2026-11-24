"""Bounded causal cache configuration for audio-video inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .scheduling import (
    CAUSAL_VIDEO_CHUNK_SIZE,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
)

if TYPE_CHECKING:
    from .causal_wrapper import CausalModelWrapper


@dataclass(frozen=True)
class CausalCacheConfig:
    """Video cache sizes in latent-frame units, with aligned audio sizes."""

    video_local_attn_size: int = 19
    video_sink_size: int = 7
    video_chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE

    @property
    def audio_local_attn_size(self) -> int:
        return causal_audio_frames(self.video_local_attn_size, self.video_chunk_size)

    @property
    def audio_sink_size(self) -> int:
        return causal_audio_frames(self.video_sink_size, self.video_chunk_size)

    def validate(self) -> None:
        if self.video_chunk_size != CAUSAL_VIDEO_CHUNK_SIZE:
            raise ValueError(
                f"Echo-WM Flash requires video_chunk_size={CAUSAL_VIDEO_CHUNK_SIZE}, got {self.video_chunk_size}"
            )
        if not 0 < self.video_sink_size < self.video_local_attn_size:
            raise ValueError("expected 0 < video_sink_size < video_local_attn_size")
        if self.video_chunk_size > self.video_local_attn_size - self.video_sink_size:
            raise ValueError("video_chunk_size must fit in the FIFO portion of the cache")
        for name, size in (
            ("video_local_attn_size", self.video_local_attn_size),
            ("video_sink_size", self.video_sink_size),
        ):
            if (size - 1) % self.video_chunk_size:
                raise ValueError(f"{name} must be 1 + n * video_chunk_size for audio alignment")


def _position_preprocessor(value):
    return getattr(value, "simple_preprocessor", value)


def _make_rope(preprocessor, positions: torch.Tensor, dtype: torch.dtype, *, inner_dim=None, max_pos=None):
    return preprocessor._prepare_positional_embeddings(
        positions=positions,
        inner_dim=preprocessor.inner_dim if inner_dim is None else inner_dim,
        max_pos=preprocessor.max_pos if max_pos is None else max_pos,
        use_middle_indices_grid=preprocessor.use_middle_indices_grid,
        num_attention_heads=preprocessor.num_attention_heads,
        x_dtype=dtype,
    )


def configure_bounded_caches(
    wrapper: CausalModelWrapper,
    caches: list[dict],
    video_positions: torch.Tensor,
    audio_positions: torch.Tensor,
    action_cond: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> None:
    """Configure bounded sink-plus-FIFO RoPE and anchor translation."""
    cfg, ppf, model = wrapper.cache, wrapper.patches_per_frame, wrapper.model
    video_raw = model.video_args_preprocessor
    audio_raw = model.audio_args_preprocessor
    video_pre = _position_preprocessor(video_raw)
    audio_pre = _position_preprocessor(audio_raw)
    video_tokens = cfg.video_local_attn_size * ppf
    video_rope = _make_rope(video_pre, video_positions[:, :, :video_tokens], dtype)
    audio_rope = _make_rope(audio_pre, audio_positions[:, :, : cfg.audio_local_attn_size], dtype)

    video_cross_rope = _make_rope(
        video_pre,
        video_positions[:, 0:1, :video_tokens],
        dtype,
        inner_dim=video_raw.audio_cross_attention_dim,
        max_pos=[video_raw.cross_pe_max_pos],
    )
    audio_cross_rope = _make_rope(
        audio_pre,
        audio_positions[:, 0:1, : cfg.audio_local_attn_size],
        dtype,
        inner_dim=audio_raw.audio_cross_attention_dim,
        max_pos=[audio_raw.cross_pe_max_pos],
    )

    video_frames = video_positions.shape[2] // ppf
    video_blocks = causal_video_blocks(video_frames, cfg.video_chunk_size)
    audio_blocks = causal_audio_blocks(video_frames, cfg.video_chunk_size)
    audio_to_video_slices: dict[tuple[int, int], tuple[int, int]] = {}
    video_to_audio_slices: dict[tuple[int, int], tuple[int, int]] = {}
    for (video_start, video_end), (audio_start, audio_end) in zip(video_blocks, audio_blocks, strict=True):
        video_query_end = min(video_end, cfg.video_local_attn_size) * ppf
        audio_to_video_slices[(audio_start, audio_end)] = (
            video_query_end - (video_end - video_start) * ppf,
            video_query_end,
        )
        audio_query_end = min(audio_end, cfg.audio_local_attn_size)
        video_to_audio_slices[(video_start * ppf, video_end * ppf)] = (
            audio_query_end - (audio_end - audio_start),
            audio_query_end,
        )

    for layer in caches:
        layer["video_self"]["local_rope_pe"] = video_rope
        layer["audio_self"]["local_rope_pe"] = audio_rope
        layer["a2v"].update(
            local_cross_q_rope_pe=video_cross_rope,
            local_cross_k_rope_pe=audio_cross_rope,
            local_cross_q_slices=audio_to_video_slices,
        )
        layer["v2a"].update(
            local_cross_q_rope_pe=audio_cross_rope,
            local_cross_k_rope_pe=video_cross_rope,
            local_cross_q_slices=video_to_audio_slices,
        )
        ucpe = layer.get("video_ucpe")
        if ucpe is not None:
            ucpe.update(
                bounded_anchor_translation=True,
                full_ucpe_viewmats=action_cond["ucpe_viewmats"],
                full_ucpe_Ks=action_cond["ucpe_Ks"],
                patches_per_frame=ppf,
            )
