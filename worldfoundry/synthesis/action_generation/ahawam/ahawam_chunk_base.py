"""Inference-only action chunking and cache orchestration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_wam import BaseWAM


class MultiQueryChunkObsEncoder(nn.Module):
    """Encode multiple obs-conditioned latent queries per action chunk."""

    def __init__(self, *, text_dim: int, num_queries: int):
        super().__init__()
        self.text_dim = int(text_dim)
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}")
        self.base_queries = nn.Parameter(
            torch.randn(1, 1, self.num_queries, self.text_dim) / (float(self.text_dim) ** 0.5)
        )
        self.obs_key_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self.obs_value_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )

    def forward(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if obs_context.ndim != 4:
            raise ValueError(f"`obs_context` must be [B, N, L, D], got shape {tuple(obs_context.shape)}")
        if obs_context_mask.ndim != 3:
            raise ValueError(f"`obs_context_mask` must be [B, N, L], got shape {tuple(obs_context_mask.shape)}")
        batch_size, num_chunks, _, _ = obs_context.shape
        queries = self.base_queries.expand(batch_size, num_chunks, -1, -1)
        keys = self.obs_key_proj(obs_context)
        values = self.obs_value_proj(obs_context)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (float(self.text_dim) ** 0.5)
        scores = scores.masked_fill(~obs_context_mask.unsqueeze(2), float("-inf"))
        obs_weights = torch.softmax(scores, dim=-1)
        obs_guided_queries = torch.matmul(obs_weights, values)
        return self.query_proj(obs_guided_queries)


class AHAWAMChunkBase(BaseWAM):
    obs_context_causal: bool = False

    def configure_action_chunking(
        self,
        *,
        action_horizon: int,
        action_chunk_size: int,
    ) -> None:
        """Set checkpoint-compatible chunk-action inference dimensions."""
        self.action_horizon = int(action_horizon)
        self.action_chunk_size = int(action_chunk_size)
        self._validate_action_chunking_configuration()

    def _validate_action_chunking_configuration(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be > 0, got {self.action_horizon}")
        if self.action_chunk_size <= 0:
            raise ValueError(f"`action_chunk_size` must be > 0, got {self.action_chunk_size}")
        if self.action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        expert_chunk_size = int(getattr(self.action_expert, "action_chunk_size", self.action_chunk_size))
        if expert_chunk_size != self.action_chunk_size:
            raise ValueError(
                f"`action_expert.action_chunk_size` ({expert_chunk_size}) must match "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        if not bool(getattr(self.action_expert, "autoregressive_teacher_forcing", False)):
            raise ValueError("Chunked inference requires `action_expert.autoregressive_teacher_forcing=true`.")

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the joint video/action mask with chunk-action visibility."""
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        if action_self_attn_mask is None:
            raise ValueError("`action_self_attn_mask` is required for AHAWAMChunkBase.")
        if action_self_attn_mask.ndim != 2:
            raise ValueError(
                f"`action_self_attn_mask` must be 2D [S, S], got shape {tuple(action_self_attn_mask.shape)}"
            )
        if action_self_attn_mask.shape != (action_seq_len, action_seq_len):
            raise ValueError(
                "`action_self_attn_mask` shape mismatch: "
                f"got {tuple(action_self_attn_mask.shape)} vs expected {(action_seq_len, action_seq_len)}"
            )
        mask[video_seq_len:, video_seq_len:] = action_self_attn_mask.to(device=device)

        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _build_prefilled_action_attention_mask(
        self,
        *,
        current_action_seq_len: int,
        action_history_seq_len: int,
        chunk_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the action-only mask used when earlier chunks already live in KV cache."""
        if action_history_seq_len == 0:
            return self.action_expert.build_single_branch_chunk_causal_mask(
                seq_len=current_action_seq_len,
                chunk_size=self.action_chunk_size,
                device=device,
            )
        return self.action_expert.build_action_self_attention_mask(
            noisy_seq_len=current_action_seq_len,
            chunk_size=self.action_chunk_size,
            device=device,
            clean_seq_len=action_history_seq_len,
            noisy_position_offset=chunk_start,
        )

    def configure_chunk_obs_context(
        self,
        *,
        action_obs_downsample_factor: int,
        chunk_kv_editor_num_queries: int,
        chunk_kv_delta_gate: bool,
        chunk_kv_gate_init: float,
    ) -> None:
        self.action_obs_downsample_factor = int(action_obs_downsample_factor)
        if self.action_obs_downsample_factor <= 0:
            raise ValueError(f"`action_obs_downsample_factor` must be > 0, got {self.action_obs_downsample_factor}")

        vae_latent_channels = getattr(self.vae, "latent_dim", None)
        if vae_latent_channels is None:
            vae_latent_channels = getattr(self.vae, "z_dim", None)
        if vae_latent_channels is None:
            raise ValueError("Could not infer VAE latent channels from `self.vae`.")

        self.action_obs_visual_proj = nn.Sequential(
            nn.LayerNorm(int(vae_latent_channels)),
            nn.Linear(int(vae_latent_channels), self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.chunk_obs_query_encoder = MultiQueryChunkObsEncoder(
            text_dim=self.text_dim,
            num_queries=int(chunk_kv_editor_num_queries),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.mot.configure_chunk_kv_cache_editor(
            query_dim=self.text_dim,
            use_delta_gate=bool(chunk_kv_delta_gate),
            gate_init=float(chunk_kv_gate_init),
        )

    def _build_chunk_kv_queries(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.chunk_obs_query_encoder(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )

    def _split_visual_obs_and_proprio_context(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()
        if proprio_tokens_per_chunk != 1:
            raise ValueError(
                f"Chunk KV editor currently expects one proprio token per chunk, got {proprio_tokens_per_chunk}."
            )
        if obs_context.shape[2] <= proprio_tokens_per_chunk:
            raise ValueError("`obs_context` must contain visual obs tokens plus proprio tokens.")
        visual_context = obs_context[:, :, :-proprio_tokens_per_chunk]
        visual_mask = obs_context_mask[:, :, :-proprio_tokens_per_chunk]
        proprio_context = obs_context[:, :, -proprio_tokens_per_chunk:]
        proprio_mask = obs_context_mask[:, :, -proprio_tokens_per_chunk:]
        return visual_context, visual_mask, proprio_context, proprio_mask

    def _build_action_obs_context(
        self,
        obs_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError("`action_obs_visual_proj` must be initialized for required obs context.")
        if obs_latents is None:
            raise ValueError("`obs_latents` is required to build mandatory obs context.")
        normalized_obs_latents = obs_latents

        if normalized_obs_latents.ndim == 4:
            normalized_obs_latents = normalized_obs_latents.unsqueeze(0)
        if normalized_obs_latents.ndim == 6:
            batch_size, num_chunks = (
                int(normalized_obs_latents.shape[0]),
                int(normalized_obs_latents.shape[1]),
            )
            flat_latents = normalized_obs_latents.reshape(batch_size * num_chunks, *normalized_obs_latents.shape[2:])
            flat_tokens, flat_mask = self._build_action_obs_context(flat_latents)
            return (
                flat_tokens.reshape(batch_size, num_chunks, flat_tokens.shape[1], flat_tokens.shape[2]),
                flat_mask.reshape(batch_size, num_chunks, flat_mask.shape[1]),
            )
        if normalized_obs_latents.ndim != 5:
            raise ValueError(
                "`obs_latents` must be [C,1,H,W], [B,C,1,H,W], or [B,N,C,1,H,W], "
                f"got shape {tuple(normalized_obs_latents.shape)}"
            )
        if normalized_obs_latents.shape[2] != 1:
            raise ValueError(
                f"`obs_latents` time dim (dim 2) must be 1, got shape {tuple(normalized_obs_latents.shape)}"
            )

        spatial_latents = normalized_obs_latents[:, :, 0].to(device=self.device, dtype=self.torch_dtype)
        if self.action_obs_downsample_factor > 1:
            spatial_latents = F.avg_pool2d(
                spatial_latents,
                kernel_size=self.action_obs_downsample_factor,
                stride=self.action_obs_downsample_factor,
            )
        obs_tokens = spatial_latents.flatten(2).transpose(1, 2).contiguous()
        obs_tokens = action_obs_visual_proj(obs_tokens)
        obs_mask = torch.ones(
            (obs_tokens.shape[0], obs_tokens.shape[1]),
            dtype=torch.bool,
            device=obs_tokens.device,
        )
        return obs_tokens, obs_mask

    def _build_chunk_aligned_obs_context_from_images(
        self,
        *,
        chunk_obs_images: torch.Tensor,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError("`action_obs_visual_proj` must be initialized for required obs context.")
        batch_size, num_chunks, channels, height, width = chunk_obs_images.shape
        if channels != 3:
            raise ValueError(
                f"`chunk_obs_images` channel dimension must be 3, got shape {tuple(chunk_obs_images.shape)}"
            )
        flat_images = chunk_obs_images.reshape(batch_size * num_chunks, channels, height, width)
        obs_latents = self._encode_input_image_latents_tensor(
            input_image=flat_images,
            tiled=tiled,
        )
        if obs_latents.ndim != 5 or obs_latents.shape[2] != 1:
            raise ValueError(
                f"Encoded chunk observation latents must have shape [B,C,1,H,W], got {tuple(obs_latents.shape)}"
            )
        obs_latents = obs_latents.reshape(batch_size, num_chunks, *obs_latents.shape[1:])
        return self._build_action_obs_context(obs_latents)

    def _append_proprio_to_obs_context(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
        chunk_start_proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append chunk-start proprio tokens to obs_context, one per chunk."""
        if self.proprio_encoder is None:
            raise ValueError("Cannot append proprio to obs_context when `proprio_encoder` is not configured.")
        if chunk_start_proprio.ndim != 3:
            raise ValueError(f"`chunk_start_proprio` must be [B, N, D], got shape {tuple(chunk_start_proprio.shape)}")
        encoder_device = next(self.proprio_encoder.parameters()).device
        proprio_tokens = self.proprio_encoder(chunk_start_proprio.to(device=encoder_device, dtype=self.torch_dtype)).to(
            device=obs_context.device, dtype=obs_context.dtype
        )
        proprio_tokens = proprio_tokens.unsqueeze(2)
        new_obs_context = torch.cat([obs_context, proprio_tokens], dim=2)
        num_chunks = chunk_start_proprio.shape[1]
        proprio_mask = torch.ones(
            (obs_context_mask.shape[0], num_chunks, 1),
            dtype=torch.bool,
            device=obs_context_mask.device,
        )
        new_obs_mask = torch.cat([obs_context_mask, proprio_mask], dim=2)
        return new_obs_context, new_obs_mask

    def _should_update_action_history(self) -> bool:
        return True

    def _require_obs_proprio_tokens_per_chunk(self) -> int:
        if self.proprio_encoder is None:
            raise ValueError(f"{type(self).__name__} requires `proprio_encoder` for chunk obs conditioning.")
        return 1

    @torch.no_grad()
    def infer_action_chunk(
        self,
        *,
        inference_state: dict[str, Any],
        chunk_obs_image: torch.Tensor,
        chunk_proprio: torch.Tensor | None = None,
        chunk_index: int,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()

        chunk_start = chunk_index * self.action_chunk_size
        chunk_end = chunk_start + self.action_chunk_size
        action_horizon = int(inference_state["start_latents"].shape[1])
        total_chunks = action_horizon // self.action_chunk_size

        if chunk_index < 0 or chunk_index >= total_chunks:
            raise ValueError(f"`chunk_index` must be in [0, {total_chunks}), got {chunk_index}.")

        context = inference_state["context"]
        context_mask = inference_state["context_mask"]
        video_kv_cache = inference_state["video_kv_cache"]
        video_seq_len = inference_state["video_seq_len"]
        video_tokens_per_frame = inference_state["video_tokens_per_frame"]
        action_history_kv_cache = inference_state["action_history_kv_cache"]
        action_history_seq_len = inference_state["action_history_seq_len"]

        batch_size = int(inference_state["batch_size"])
        chunk_obs_image = chunk_obs_image.to(device=self.device, dtype=self.torch_dtype)
        if chunk_obs_image.ndim == 3:
            chunk_obs_image = chunk_obs_image.unsqueeze(0)
        if chunk_obs_image.ndim != 4 or chunk_obs_image.shape[0] != batch_size:
            raise ValueError(
                f"`chunk_obs_image` must be [3,H,W] or [{batch_size},3,H,W], got shape {tuple(chunk_obs_image.shape)}"
            )
        if self.proprio_encoder is None:
            raise ValueError(f"{type(self).__name__} requires `proprio_encoder` for chunk inference.")
        if chunk_proprio is None:
            raise ValueError(f"`chunk_proprio` is required for `infer_action_chunk`. (chunk_index={chunk_index})")
        (
            chunk_conditioning_context,
            chunk_conditioning_mask,
            chunk_conditioning_offset,
        ) = self._prepare_inference_chunk_conditioning(
            chunk_obs_image=chunk_obs_image,
            chunk_proprio=chunk_proprio,
            chunk_index=chunk_index,
            inference_state=inference_state,
            tiled=tiled,
        )
        video_kv_cache = inference_state["_chunk_video_kv_cache"]
        video_seq_len = inference_state["video_seq_len"]
        cross_attn_kv_cache = self._prepare_inference_cross_attn_kv_cache(
            inference_state=inference_state,
            context=context,
            obs_context=chunk_conditioning_context,
        )
        obs_proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()

        current_latents = inference_state["start_latents"]
        noisy_chunk = current_latents[:, chunk_start:chunk_end].clone()

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=noisy_chunk.dtype,
            shift_override=sigma_shift,
        )

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action, strict=False):
            timestep_action = step_t_action.expand(noisy_chunk.shape[0]).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            pred_action = self._predict_action_chunk_with_clean_cache(
                noisy_action_chunk=noisy_chunk,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                obs_context=chunk_conditioning_context,
                obs_context_mask=chunk_conditioning_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                action_history_kv_cache=action_history_kv_cache,
                action_history_seq_len=action_history_seq_len,
                cross_attn_kv_cache=cross_attn_kv_cache,
                chunk_start=chunk_start,
                obs_chunk_offset=chunk_conditioning_offset,
                obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            )
            noisy_chunk = self.infer_action_scheduler.step(pred_action, step_delta_action, noisy_chunk)

        current_latents[:, chunk_start:chunk_end] = noisy_chunk

        if self._should_update_action_history():
            action_history_kv_cache = self._prefill_clean_action_chunk_cache(
                clean_action_chunk=noisy_chunk,
                context=context,
                context_mask=context_mask,
                obs_context=chunk_conditioning_context,
                obs_context_mask=chunk_conditioning_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                action_history_kv_cache=action_history_kv_cache,
                action_history_seq_len=action_history_seq_len,
                cross_attn_kv_cache=cross_attn_kv_cache,
                chunk_start=chunk_start,
                obs_chunk_offset=chunk_conditioning_offset,
                obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            )
            action_history_seq_len += self.action_chunk_size

        updated_state = {
            **inference_state,
            "start_latents": current_latents,
            "action_history_kv_cache": action_history_kv_cache,
            "action_history_seq_len": action_history_seq_len,
            "cross_attn_kv_cache": cross_attn_kv_cache,
        }

        return {
            "action_chunk": noisy_chunk[0].detach().to(device="cpu", dtype=torch.float32),
            "final_latents_chunk": noisy_chunk.detach().clone(),
            "chunk_index": int(chunk_index),
            "inference_state": updated_state,
        }

    @torch.no_grad()
    def _build_prefilled_action_pre_and_mask(
        self,
        *,
        action_chunk: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: torch.Tensor | None = None,
        obs_context_mask: torch.Tensor | None = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_chunk,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=chunk_start,
            single_branch_chunk_causal=True,
            obs_chunk_offset=obs_chunk_offset,
            obs_context_causal=self.obs_context_causal,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            skip_context_embedding=cross_attn_kv_cache is not None,
        )
        total_action_seq_len = action_history_seq_len + action_pre["tokens"].shape[1]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=total_action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=action_pre["tokens"].device,
            action_self_attn_mask=self._build_prefilled_action_attention_mask(
                current_action_seq_len=action_pre["tokens"].shape[1],
                action_history_seq_len=action_history_seq_len,
                chunk_start=chunk_start,
                device=action_pre["tokens"].device,
            ),
        )
        return action_pre, attention_mask

    @torch.no_grad()
    def _predict_action_chunk_with_clean_cache(
        self,
        *,
        noisy_action_chunk: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_kv_cache: list[dict[str, torch.Tensor]] | None,
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: torch.Tensor | None = None,
        obs_context_mask: torch.Tensor | None = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        action_pre, attention_mask = self._build_prefilled_action_pre_and_mask(
            action_chunk=noisy_action_chunk,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_history_seq_len=action_history_seq_len,
            chunk_start=chunk_start,
            obs_chunk_offset=obs_chunk_offset,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        prior_embed = self.mot.action_branch_embedding[1].to(
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        action_tokens = action_pre["tokens"] + prior_embed.view(1, 1, -1)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_history_kv_cache=action_history_kv_cache,
            action_history_seq_len=action_history_seq_len,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _prefill_clean_action_chunk_cache(
        self,
        *,
        clean_action_chunk: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_kv_cache: list[dict[str, torch.Tensor]] | None,
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: torch.Tensor | None = None,
        obs_context_mask: torch.Tensor | None = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        clean_timestep = torch.zeros(
            (clean_action_chunk.shape[0], clean_action_chunk.shape[1]),
            device=self.device,
            dtype=clean_action_chunk.dtype,
        )
        action_pre, attention_mask = self._build_prefilled_action_pre_and_mask(
            action_chunk=clean_action_chunk,
            timestep_action=clean_timestep,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_history_seq_len=action_history_seq_len,
            chunk_start=chunk_start,
            obs_chunk_offset=obs_chunk_offset,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        prior_embed = self.mot.action_branch_embedding[1].to(
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        action_tokens = action_pre["tokens"] + prior_embed.view(1, 1, -1)
        delta_kv_cache = self.mot.prefill_action_history_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_history_kv_cache=action_history_kv_cache,
            action_history_seq_len=action_history_seq_len,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        return self.mot.append_kv_cache(action_history_kv_cache, delta_kv_cache)

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image: torch.Tensor | None = None,
        action_horizon: int | None = None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        chunk_obs_image: torch.Tensor | None = None,
        chunk_proprio: torch.Tensor | None = None,
        video_frame_index: int | None = None,
        phase: str = "video",
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        if phase == "video":
            if input_image is None:
                raise ValueError("`input_image` is required for phase='video'.")
            if action_horizon is None:
                action_horizon = self.action_horizon

            prefill_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "input_image": input_image,
                "action_horizon": action_horizon,
                "context": context,
                "context_mask": context_mask,
                "seed": seed,
                "rand_device": rand_device,
                "tiled": tiled,
            }
            if video_frame_index is not None:
                prefill_kwargs["video_frame_index"] = video_frame_index
            self._inference_state = self.prefill_video(**prefill_kwargs)
            self._inference_state["next_chunk_index"] = 0
            return {"phase": "video", "chunk_index": 0}

        if phase == "action":
            if not hasattr(self, "_inference_state") or self._inference_state is None:
                raise RuntimeError("Must call `infer_action(phase='video')` before `phase='action'`.")
            if chunk_obs_image is None:
                raise ValueError("`chunk_obs_image` is required for phase='action'.")

            state = self._inference_state
            chunk_index = state["next_chunk_index"]

            if self.proprio_encoder is not None and chunk_proprio is None:
                raise ValueError(
                    f"`chunk_proprio` is required for phase='action' when `proprio_dim` is enabled. "
                    f"Pass the current chunk's proprio obtained from the environment. "
                    f"(chunk_index={chunk_index})"
                )

            result = self.infer_action_chunk(
                inference_state=state,
                chunk_obs_image=chunk_obs_image,
                chunk_proprio=chunk_proprio,
                chunk_index=chunk_index,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
            )

            self._inference_state = result["inference_state"]
            self._inference_state["next_chunk_index"] = chunk_index + 1

            return {
                "action_chunk": result["action_chunk"],
                "chunk_index": int(chunk_index),
                "phase": "action",
            }

        raise ValueError(f"`phase` must be 'video' or 'action', got '{phase}'.")
