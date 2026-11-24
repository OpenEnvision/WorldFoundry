# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors and The MiniMax and HuggingFace Teams.
# Adapted from HyperFlow 1dd2f342 and Diffusers v0.40.0 MiniMax-H3 inference.
"""Pack canonical H3 inputs and drive HyperFlow's eight coupled video/audio steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.schedulers.hyperflow import HyperFlowSchedule
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_audio_latent_t,
    minimax_h3_video_latent_t,
)

from .media import PreparedMedia


@dataclass
class HyperFlowLatents:
    video: torch.Tensor
    audio: torch.Tensor
    layout: dict[str, torch.Tensor]
    condition_video_rows: int
    condition_audio_rows: int
    video_shape: tuple[int, int, int, int]
    audio_length: int


def prepare_latents(
    media: PreparedMedia,
    text_tags: torch.Tensor,
    condition_latents: list[torch.Tensor],
    audio_condition_latents: list[torch.Tensor],
    *,
    generator: torch.Generator,
    device: torch.device,
    video_channels: int = 24,
    audio_channels: int = 32,
) -> HyperFlowLatents:
    """Preserve CPU-generator draws: conditioning, latent-layout video, row-layout audio."""
    from worldfoundry.pipelines.minimax._minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )
    from worldfoundry.pipelines.minimax._minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

    latent_t = minimax_h3_video_latent_t(media.num_frames)
    latent_h, latent_w = media.height // 16, media.width // 16
    audio_t = minimax_h3_audio_latent_t(media.num_frames / 24)
    options = {
        "text_len": len(text_tags),
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "audio_t": audio_t,
    }
    if media.workflow == "ref2va":
        visual_geometry = iter(latent.shape[2:] for latent in condition_latents)
        audio_lengths = iter(rows.shape[0] // 2 for rows in audio_condition_latents)
        blocks: list[dict[str, Any]] = []
        for reference in media.references:
            block: dict[str, Any] = {"kind": reference.kind}
            if reference.kind in ("image", "video"):
                frames, height, width = next(visual_geometry)
                block.update(latent_t=frames, latent_h=height, latent_w=width)
            if reference.kind in ("audio", "video"):
                block["ref_audio_t"] = next(audio_lengths) if reference.has_audio else 0
            blocks.append(block)
        plan = minimax_h3_packed_sequence_ref2va_blocks(**options, ref_blocks=blocks)
    else:
        plan = minimax_h3_packed_sequence(
            **options,
            include_keyframe_cond=bool(media.keyframe_indices),
            keyframe_frame_indices=media.keyframe_indices or None,
            frame_count=media.num_frames,
        )
    used = len(plan["img_pos"]) + len(plan["audio_pos"]) + len(text_tags)
    plan["token_tags"][: len(text_tags)] = text_tags.cpu()
    layout = {
        "position_ids": plan["img_position_ids"][:used].to(device),
        "token_tags": plan["token_tags"][:used].to(device),
        "video_indices": plan["img_pos"].to(device),
        "audio_indices": plan["audio_pos"].to(device),
        "text_indices": plan["text_pos"].to(device),
    }
    condition_video_rows = len(plan["img_pos"]) - latent_t * (latent_h // 2) * (latent_w // 2)
    condition_audio_rows = len(plan["audio_pos"]) - audio_t * 2

    def noise(shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)

    packed_conditions = []
    for condition in condition_latents:
        random = noise(condition.shape)
        condition = condition.to(device)
        time = condition.new_tensor(0.999)
        noised = time * condition + (1.0 - time) * random
        packed_conditions.append(minimax_h3_patchify_video_latent(noised, patch_size=(1, 2, 2)))
    video = noise((1, video_channels, latent_t, latent_h, latent_w))
    video = minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2))
    audio = noise((audio_t * 2, audio_channels))
    video = torch.cat([*packed_conditions, video])
    audio = torch.cat([*(rows.to(device) for rows in audio_condition_latents), audio])
    return HyperFlowLatents(
        video,
        audio,
        layout,
        condition_video_rows,
        condition_audio_rows,
        (latent_t, latent_h // 2, latent_w // 2, video_channels),
        audio_t,
    )


@torch.no_grad()
def denoise(
    transformer: nn.Module, latents: HyperFlowLatents, prompt_embeds: torch.Tensor, schedule: HyperFlowSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one CFG-free forward per interval; never update the reference-row prefix."""
    from worldfoundry.pipelines.minimax._minimax_h3.packed_tokens import (
        minimax_h3_unpack_audio_tokens,
        minimax_h3_unpatchify_video_tokens,
    )

    plan = schedule.row_plan(
        video_indices=latents.layout["video_indices"],
        audio_indices=latents.layout["audio_indices"],
        num_condition_video_rows=latents.condition_video_rows,
        num_condition_audio_rows=latents.condition_audio_rows,
        num_text_tokens=prompt_embeds.shape[1],
        keyframe_noise_aug=0.999,
        device=latents.video.device,
    )
    video_start, audio_start = latents.condition_video_rows, latents.condition_audio_rows
    for step, (timestep, endpoint, indices) in enumerate(plan):
        video, audio = transformer(
            hidden_states=latents.video[None],
            audio_hidden_states=latents.audio[None],
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            endpoint=endpoint,
            timestep_indices=indices,
            step=step,
            **latents.layout,
        )
        latents.video[video_start:] = schedule.step(video[0, video_start:].float(), latents.video[video_start:], step)
        latents.audio[audio_start:] = schedule.step(
            audio[0, audio_start:].float(), latents.audio[audio_start:], step, audio=True
        )
    video = minimax_h3_unpatchify_video_tokens(
        latents.video[video_start:], latent_shape=latents.video_shape, patch_size=(1, 2, 2)
    )
    audio = minimax_h3_unpack_audio_tokens(
        latents.audio[audio_start:], audio_t=latents.audio_length * 2, audio_channel=2
    )
    return video, audio
