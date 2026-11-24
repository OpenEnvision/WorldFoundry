"""Inference-only base for the AHA-WAM two-phase policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .action_dit import ActionDiT
from .mot import MoT
from .scheduler import WanContinuousFlowMatchScheduler


class BaseWAM(torch.nn.Module):
    """Wan world-action model state and inference helpers."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_infer_shift: float = 5.0,
        video_num_timesteps: int = 1000,
        action_infer_shift: float = 5.0,
        action_num_timesteps: int = 1000,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_timesteps=video_num_timesteps,
            shift=video_infer_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_timesteps=action_num_timesteps,
            shift=action_infer_shift,
        )
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        self.to(self.device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: str | Sequence[str]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        encoder_parameter = next(self.text_encoder.parameters())
        encoder_device = encoder_parameter.device
        ids = ids.to(encoder_device)
        mask = mask.to(encoder_device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (
            prompt_emb.to(device=self.device, dtype=self.torch_dtype),
            mask.to(device=self.device),
        )

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        encoder_device = next(self.proprio_encoder.parameters()).device
        proprio_token = self.proprio_encoder(proprio.to(device=encoder_device, dtype=self.torch_dtype).unsqueeze(1)).to(
            device=context.device, dtype=context.dtype
        )
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        vae_parameter = next(self.vae.parameters())
        vae_device = vae_parameter.device
        vae_dtype = vae_parameter.dtype
        input_image = input_image.to(device=vae_device, dtype=vae_dtype)
        image_list = [image.unsqueeze(1) for image in input_image]
        z = self.vae.encode(
            image_list,
            device=vae_device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = torch.stack(z, dim=0)
        return z.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _prepare_action_start_latents(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        start_latents: torch.Tensor | None,
        seed: int | None,
        rand_device: str,
    ) -> tuple[torch.Tensor, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        batch_size = int(input_image.shape[0])
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        if start_latents is None:
            generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            latents_action = torch.randn(
                (batch_size, action_horizon, self.action_expert.action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if start_latents.ndim != 3:
                raise ValueError(
                    f"`start_latents` must have shape [B, T, action_dim], got {tuple(start_latents.shape)}"
                )
            if start_latents.shape[0] != batch_size:
                raise ValueError(
                    f"`start_latents.shape[0]` must equal input_image batch={batch_size}, got {start_latents.shape[0]}"
                )
            if start_latents.shape[1] != action_horizon:
                raise ValueError(
                    f"`start_latents.shape[1]` must equal action_horizon={action_horizon}, got {start_latents.shape[1]}"
                )
            if start_latents.shape[2] != self.action_expert.action_dim:
                raise ValueError(
                    f"`start_latents.shape[2]` must equal action_dim={self.action_expert.action_dim}, "
                    f"got {start_latents.shape[2]}"
                )
            latents_action = start_latents.to(device=self.device, dtype=self.torch_dtype)
        return latents_action, batch_size

    @torch.no_grad()
    def _prepare_action_context(
        self,
        *,
        prompt: str | None,
        batch_size: int,
        proprio: torch.Tensor | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2:
                raise ValueError(f"`proprio` must be [D] or [B,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            if proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` batch dim must match input_image batch={batch_size}, got {proprio.shape[0]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_context = context is not None or context_mask is not None
        if use_context:
            prompt = None
        use_prompt = prompt is not None
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            if isinstance(prompt, (list, tuple)) and len(prompt) != batch_size:
                raise ValueError(f"`prompt` batch size must match input_image batch={batch_size}, got {len(prompt)}")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
                raise ValueError(
                    "`context/context_mask` batch dim must match input_image batch: "
                    f"{context.shape[0]} / {context_mask.shape[0]} vs {batch_size}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    @torch.no_grad()
    def _prefill_action_video_cache(
        self,
        *,
        video_pre: dict[str, Any],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> list[dict[str, torch.Tensor]]:
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        return self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )
