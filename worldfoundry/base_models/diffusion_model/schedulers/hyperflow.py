# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
# Copyright 2025-2026 The MiniMax authors and The HuggingFace Team.
# Adapted from Video-Rebirth/hyperflow at 1dd2f342aba5ab51da02b62885939655e8e268da.
"""HyperFlow's fixed eight intervals and pinned conditioning-row endpoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .minimax_h3 import minimax_h3_euler_eta0_step, minimax_h3_rf_v_to_x0, minimax_h3_sigma_schedule

DEFAULT_SIGMAS_8STEP = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)


def build_row_time_pairs(
    *,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    video_endpoint: float,
    audio_timestep: float,
    audio_endpoint: float,
    condition_video_timestep: float,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return distinct (start, endpoint) pairs and each packed row's pair index."""
    video_indices, audio_indices = video_indices.cpu(), audio_indices.cpu()
    sequence_length = video_indices.numel() + audio_indices.numel() + num_text_tokens
    timestep = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32)
    endpoint = torch.full((sequence_length,), float(video_endpoint), dtype=torch.float32)
    condition_video = video_indices[:num_condition_video_rows]
    timestep[condition_video] = float(condition_video_timestep)
    endpoint[condition_video] = float(condition_video_timestep)
    generated_audio = audio_indices[num_condition_audio_rows:]
    timestep[generated_audio] = float(audio_timestep)
    endpoint[generated_audio] = float(audio_endpoint)
    condition_audio = audio_indices[:num_condition_audio_rows]
    timestep[condition_audio] = float(condition_audio_timestep)
    endpoint[condition_audio] = float(condition_audio_timestep)
    pairs = torch.stack((timestep, endpoint), dim=-1)
    unique, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


class HyperFlowSchedule:
    """The adapter grid shifted independently for video and audio, with H3 Euler math."""

    def __init__(
        self,
        sigmas: Sequence[float] = DEFAULT_SIGMAS_8STEP,
        *,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
    ) -> None:
        if len(sigmas) != 9:
            raise ValueError("HyperFlow v1.0 requires nine sigma points (eight forwards).")
        self.video_sigmas = torch.tensor(minimax_h3_sigma_schedule(sigmas, shift_scale=video_shift))
        self.audio_sigmas = torch.tensor(minimax_h3_sigma_schedule(sigmas, shift_scale=audio_shift))
        self.video_timesteps = 1.0 - self.video_sigmas[:-1]
        self.audio_timesteps = 1.0 - self.audio_sigmas[:-1]

    def row_plan(
        self,
        *,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
        num_text_tokens: int,
        keyframe_noise_aug: float,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        plan = []
        for step, timestep in enumerate(self.video_timesteps):
            pairs = build_row_time_pairs(
                video_indices=video_indices,
                audio_indices=audio_indices,
                num_condition_video_rows=num_condition_video_rows,
                num_condition_audio_rows=num_condition_audio_rows,
                num_text_tokens=num_text_tokens,
                video_timestep=float(timestep),
                video_endpoint=float(1.0 - self.video_sigmas[step + 1]),
                audio_timestep=float(self.audio_timesteps[step]),
                audio_endpoint=float(1.0 - self.audio_sigmas[step + 1]),
                condition_video_timestep=max(float(timestep), keyframe_noise_aug),
            )
            plan.append(tuple(value.to(device) for value in pairs))
        return plan

    def step(self, velocity: torch.Tensor, sample: torch.Tensor, index: int, *, audio: bool = False) -> torch.Tensor:
        sigmas = self.audio_sigmas if audio else self.video_sigmas
        timesteps = self.audio_timesteps if audio else self.video_timesteps
        denoised = minimax_h3_rf_v_to_x0(sample, velocity, timesteps[index])
        return minimax_h3_euler_eta0_step(
            sample, denoised, sigma_curr=float(sigmas[index]), sigma_next=float(sigmas[index + 1])
        )
