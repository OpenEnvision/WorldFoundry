"""Inference-only AHA-WAM policy with history-aware video KV prefill."""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from .ahawam_chunk_base import AHAWAMChunkBase


class AHAWAM(AHAWAMChunkBase):
    ACTION_VIDEO_READ_MODES = ("current_only", "history_current")

    def configure_chunk_history(
        self,
        *,
        num_history_frames: int = 0,
        action_video_read_mode: str = "current_only",
        video_rope_frame_stride: int = 1,
    ) -> None:
        self.num_history_frames = int(num_history_frames)
        if self.num_history_frames < 0:
            raise ValueError(f"`num_history_frames` must be >= 0, got {num_history_frames}")
        self.action_video_read_mode = self._normalize_action_video_read_mode(action_video_read_mode)
        self.video_rope_frame_stride = int(video_rope_frame_stride)
        if self.video_rope_frame_stride <= 0:
            raise ValueError(f"`video_rope_frame_stride` must be positive, got {video_rope_frame_stride}.")

    def _configured_num_history_frames(self) -> int:
        return int(getattr(self, "num_history_frames", 0))

    def _normalize_action_video_read_mode(self, mode: str) -> str:
        normalized = str(mode)
        if normalized not in self.ACTION_VIDEO_READ_MODES:
            raise ValueError(f"`action_video_read_mode` must be one of {self.ACTION_VIDEO_READ_MODES}, got {mode!r}.")
        return normalized

    def _configured_video_rope_frame_stride(self) -> int:
        stride = int(getattr(self, "video_rope_frame_stride", 1))
        if stride <= 0:
            raise ValueError(f"`video_rope_frame_stride` must be positive, got {stride}.")
        return stride

    @override
    def _should_update_action_history(self) -> bool:
        return False

    @override
    def _prepare_inference_cross_attn_kv_cache(
        self,
        *,
        inference_state: dict[str, Any],
        context: torch.Tensor,
        obs_context: torch.Tensor | None,
    ) -> list[dict[str, torch.Tensor]] | None:
        del inference_state, context, obs_context
        return None

    @override
    def _prepare_inference_chunk_conditioning(
        self,
        *,
        chunk_obs_image: torch.Tensor,
        chunk_proprio: torch.Tensor | None,
        chunk_index: int,
        inference_state: dict[str, Any],
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare single-chunk local obs/proprio conditioning for inference."""
        if self.proprio_encoder is None:
            raise ValueError(f"{type(self).__name__} requires `proprio_encoder` for chunk inference conditioning.")
        if chunk_proprio is None:
            raise ValueError("`chunk_proprio` is required for chunk inference conditioning.")
        chunk_obs_image = chunk_obs_image.to(device=self.device, dtype=self.torch_dtype)
        if chunk_obs_image.ndim == 3:
            chunk_obs_image = chunk_obs_image.unsqueeze(0)
        self.proprio_encoder = self.proprio_encoder.to(
            device=chunk_obs_image.device,
            dtype=self.torch_dtype,
        )
        single_chunk_images = chunk_obs_image.unsqueeze(1)
        obs_context, obs_context_mask = self._build_chunk_aligned_obs_context_from_images(
            chunk_obs_images=single_chunk_images,
            tiled=tiled,
        )
        proprio_for_chunk = chunk_proprio.unsqueeze(1)
        obs_context, obs_context_mask = self._append_proprio_to_obs_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            chunk_start_proprio=proprio_for_chunk,
        )
        (
            visual_obs_context,
            visual_obs_mask,
            proprio_context,
            proprio_mask,
        ) = self._split_visual_obs_and_proprio_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )
        chunk_queries = self._build_chunk_kv_queries(
            obs_context=visual_obs_context,
            obs_context_mask=visual_obs_mask,
        )
        inference_state["_chunk_video_kv_cache"] = self.mot.build_chunk_updated_video_kv_cache(
            video_kv_cache=inference_state["video_kv_cache"],
            chunk_queries=chunk_queries,
            video_tokens_per_frame=int(inference_state["video_tokens_per_frame"]),
            chunk_index=0,
        )
        return proprio_context, proprio_mask, chunk_index

    def _concat_history_kv_entries(
        self,
        entries: list[list[dict[str, torch.Tensor]]],
    ) -> list[dict[str, torch.Tensor]]:
        """Concatenate multiple per-layer KV cache entries into one."""
        num_layers = len(entries[0])
        result: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_parts = [e[layer_idx]["k"] for e in entries]
            v_parts = [e[layer_idx]["v"] for e in entries]
            result.append(
                {
                    "k": torch.cat(k_parts, dim=1),
                    "v": torch.cat(v_parts, dim=1),
                }
            )
        return result

    @override
    def prefill_video(
        self,
        *,
        prompt: str | None = None,
        input_image: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        video_frame_index: int | None = None,
    ) -> dict[str, Any]:
        self.eval()
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        video_mask_mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if video_mask_mode not in {"first_frame_causal", "per_frame_causal"}:
            raise ValueError(
                "Two-phase inference requires `video_attention_mask_mode` to be "
                "'first_frame_causal' or 'per_frame_causal'."
            )

        latents_action, batch_size = self._prepare_action_start_latents(
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=None,
            seed=seed,
            rand_device=rand_device,
        )
        context, context_mask = self._prepare_action_context(
            prompt=prompt,
            batch_size=batch_size,
            proprio=None,
            context=context,
            context_mask=context_mask,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if not fuse_flag:
            raise ValueError("AHAWAMChunkBase requires `fuse_vae_embedding_in_latents=True`.")

        video_rope_frame_stride = self._configured_video_rope_frame_stride()
        current_frame_index = (
            int(video_frame_index) if video_frame_index is not None else getattr(self, "_observed_frame_index", 0)
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            temporal_position_ids=torch.full(
                (1,),
                int(current_frame_index),
                dtype=torch.long,
                device=first_frame_latents.device,
            ),
            clean_prefix_frames=1,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        num_history = self._configured_num_history_frames()
        prior_entries = (getattr(self, "_history_kv_entries", None) or []) if num_history > 0 else []

        if prior_entries:
            prefix_cache = self._concat_history_kv_entries(prior_entries)
            prefix_seq_len = int(prefix_cache[0]["k"].shape[1])
            video_attention_mask = torch.ones(
                (video_seq_len, prefix_seq_len + video_seq_len),
                dtype=torch.bool,
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache_with_prefix(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                prefix_video_kv_cache=prefix_cache,
                prefix_video_seq_len=prefix_seq_len,
                video_attention_mask=video_attention_mask,
            )
        else:
            video_kv_cache = self._prefill_action_video_cache(
                video_pre=video_pre,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
            )

        self._observed_frame_index = current_frame_index + action_horizon // video_rope_frame_stride

        if num_history > 0:
            prior_entries.append(video_kv_cache)
            if len(prior_entries) > num_history:
                prior_entries = prior_entries[-num_history:]
            self._history_kv_entries = prior_entries

        state = {
            "start_latents": latents_action,
            "context": context,
            "context_mask": context_mask,
            "batch_size": batch_size,
            "video_pre": video_pre,
            "video_seq_len": video_seq_len,
            "video_tokens_per_frame": video_tokens_per_frame,
            "video_kv_cache": video_kv_cache,
            "action_history_kv_cache": None,
            "action_history_seq_len": 0,
        }
        return state

    def reset_history(self) -> None:
        """Clear accumulated history KV cache entries and frame counter."""
        self._history_kv_entries = []
        self._observed_frame_index = 0

    @override
    def _build_prefilled_action_attention_mask(
        self,
        *,
        current_action_seq_len: int,
        action_history_seq_len: int,
        chunk_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        del action_history_seq_len, chunk_start
        return self.action_expert.build_single_branch_chunk_causal_mask(
            seq_len=current_action_seq_len,
            chunk_size=self.action_chunk_size,
            device=device,
        )
