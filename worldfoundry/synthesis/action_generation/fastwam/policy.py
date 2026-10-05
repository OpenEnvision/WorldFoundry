# SPDX-License-Identifier: MIT
"""Inference-only FastWAM policy wrapper around the official MoT modules."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .scheduler import WanContinuousFlowMatchScheduler


class FastWAMPolicy(nn.Module):
    """Action-only FastWAM inference with first-frame KV prefill."""

    def __init__(
        self,
        *,
        video_expert: nn.Module,
        action_expert: nn.Module,
        mot: nn.Module,
        vae: nn.Module,
        proprio_encoder: nn.Module,
        device: str,
        dtype: torch.dtype,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        video_infer_shift: float = 5.0,
        vae_tile_size: tuple[int, int] = (30, 52),
        vae_tile_stride: tuple[int, int] = (15, 26),
    ) -> None:
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vae = vae
        self.proprio_encoder = proprio_encoder
        self.proprio_dim = int(proprio_encoder.in_features)
        self.device = torch.device(device)
        self.torch_dtype = dtype
        self.vae_tile_size = vae_tile_size
        self.vae_tile_stride = vae_tile_stride
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=video_infer_shift,
        )

    def _append_proprio(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape != (context.shape[0], self.proprio_dim):
            raise ValueError(
                f"FastWAM proprio must have shape [{context.shape[0]}, {self.proprio_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=self.device)
        return torch.cat([context, token], dim=1), torch.cat([context_mask, mask], dim=1)

    @torch.inference_mode()
    def encode_first_frame(self, image: torch.Tensor, *, tiled: bool = False) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
            raise ValueError(f"FastWAM image must be [1, 3, H, W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("FastWAM image height and width must be divisible by 16")
        video = image.to(device=self.device, dtype=self.torch_dtype)[0].unsqueeze(1)
        latent = self.vae.encode(
            [video],
            device=self.device,
            tiled=tiled,
            tile_size=self.vae_tile_size,
            tile_stride=self.vae_tile_stride,
        )
        if isinstance(latent, list):
            latent = latent[0].unsqueeze(0)
        return latent

    def _attention_mask(
        self,
        *,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, : min(video_tokens_per_frame, video_seq_len)] = True
        return mask

    @torch.inference_mode()
    def infer_action(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = 0,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> torch.Tensor:
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("FastWAM action inference requires first_frame_causal video attention")
        if action_horizon <= 0:
            raise ValueError("FastWAM action_horizon must be positive")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != 4096:
            raise ValueError("FastWAM context must have shape [batch, tokens, 4096]")
        if context_mask.shape != context.shape[:2]:
            raise ValueError("FastWAM context_mask must match the context token dimensions")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        context, context_mask = self._append_proprio(context, context_mask, proprio)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        first_frame = self.encode_first_frame(input_image, tiled=tiled)
        timestep_video = torch.zeros(
            (first_frame.shape[0],),
            dtype=first_frame.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=action,
                timestep=timestep.unsqueeze(0).to(dtype=action.dtype, device=self.device),
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            prediction = self.action_expert.post_dit(action_tokens, action_pre)
            action = self.infer_action_scheduler.step(prediction, delta, action)
        return action[0].detach().to(device="cpu", dtype=torch.float32)

    @torch.inference_mode()
    def infer_action_idm(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = 0,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> torch.Tensor:
        """Optional IDM: denoise future video, then condition actions on all its frames."""
        if num_video_frames < 5 or num_video_frames % 4 != 1:
            raise ValueError("Optional IDM num_video_frames must be at least 5 and satisfy T % 4 == 1")
        if action_horizon <= 0:
            raise ValueError("FastWAM action_horizon must be positive")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != 4096 or context_mask.shape != context.shape[:2]:
            raise ValueError("FastWAM context/mask must have shapes [batch, tokens, 4096]/[batch, tokens]")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        context, context_mask = self._append_proprio(context, context_mask, proprio)

        first_frame = self.encode_first_frame(input_image, tiled=tiled)
        latent_frames = (num_video_frames - 1) // 4 + 1
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        video = torch.randn(
            (1, first_frame.shape[1], latent_frames, *first_frame.shape[-2:]),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        video[:, :, :1] = first_frame
        video_steps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=video.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(video_steps, video_deltas):
            prediction = self.video_expert(
                x=video,
                timestep=timestep.unsqueeze(0).to(device=self.device, dtype=video.dtype),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
                ),
            )
            video = self.infer_video_scheduler.step(prediction, delta, video)
            video[:, :, :1] = first_frame

        video_pre = self.video_expert.pre_dit(
            x=video,
            timestep=torch.zeros((1,), device=self.device, dtype=video.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=self.device,
        )
        # IDM action queries see the entire generated video, unlike first-frame mode.
        attention_mask[video_seq_len:, :video_seq_len] = True
        video_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_steps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(action_steps, action_deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=action,
                timestep=timestep.unsqueeze(0).to(device=self.device, dtype=action.dtype),
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action = self.infer_action_scheduler.step(
                self.action_expert.post_dit(action_tokens, action_pre), delta, action
            )
        return action[0].detach().to(device="cpu", dtype=torch.float32)


__all__ = ["FastWAMPolicy"]
