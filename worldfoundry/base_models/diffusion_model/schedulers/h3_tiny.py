# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from fuyi25/H3-Tiny at a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c (Apache-2.0).
"""H3-Tiny shifted sigma schedules, MeanFlow per-row (t, r) pairs, and Euler step."""

from __future__ import annotations

import math

import torch

H3_TINY_VIDEO_SHIFT = 12.0
H3_TINY_AUDIO_SHIFT = 3.0
H3_TINY_REFERENCE_TIMESTEP = 0.999
H3_TINY_AUDIO_CONDITION_TIMESTEP = 1.0
H3_TINY_FEWSTEP_DEFAULT_STEPS = 6
H3_TINY_FEWSTEP_MIN_STEPS = 4
H3_TINY_FULLSTEP_DEFAULT_STEPS = 50


def h3_tiny_shifted_sigmas(num_steps: int, shift: float) -> torch.Tensor:
    """Return ``num_steps + 1`` shifted sigmas from 1.0 to 0.0 in FP32.

    Computes ``grid = linspace(1.0, 0.0, num_steps + 1, dtype=float64)`` and applies
    ``shift * grid / (1 + (shift - 1) * grid)`` before casting to ``float32``.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}")
    shift_value = float(shift)
    if not math.isfinite(shift_value) or shift_value <= 0.0:
        raise ValueError(f"shift must be finite and positive, got {shift!r}")
    grid = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return (shift_value * grid / (1.0 + (shift_value - 1.0) * grid)).to(torch.float32)


def h3_tiny_row_time_pairs(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_time: tuple[float, float],
    audio_time: tuple[float, float],
    condition_video_timestep: float,
    condition_audio_timestep: float = H3_TINY_AUDIO_CONDITION_TIMESTEP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign unique ``(t, r)`` pairs across packed rows and return inverse indices.

    Generated video rows and text rows receive ``video_time = (t_v, r_v)``; generated
    audio rows receive ``audio_time = (t_a, r_a)``; visual and audio conditioning rows
    are pinned with ``r = t``. Deduplication is performed on 2-D ``(t, r)`` pairs so
    rows sharing ``t = 0`` at step 0 retain distinct video/audio endpoints.
    """
    video_idx = video_indices.cpu()
    audio_idx = audio_indices.cpu()
    n = int(video_idx.numel() + audio_idx.numel() + num_text_tokens)
    pairs = torch.tensor(video_time, dtype=torch.float32).repeat(n, 1)
    if num_condition_video_rows > 0:
        pairs[video_idx[:num_condition_video_rows]] = float(condition_video_timestep)
    if num_condition_audio_rows < audio_idx.numel():
        pairs[audio_idx[num_condition_audio_rows:]] = torch.tensor(audio_time, dtype=torch.float32)
    if num_condition_audio_rows > 0:
        pairs[audio_idx[:num_condition_audio_rows]] = float(condition_audio_timestep)
    unique, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


def h3_tiny_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """Advance one rectified-flow / MeanFlow Euler step: ``x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * (-v)``.

    MiniMax-H3 and H3-Tiny predict ``v = x0 - noise`` (pointing noise -> data along
    ``t = 1 - sigma``), so stepping from ``sigma_curr`` down to ``sigma_next`` adds
    ``(sigma_next - sigma_curr) * (-v) = (sigma_curr - sigma_next) * v``.
    """
    if sample.shape != velocity.shape:
        raise ValueError(f"sample and velocity shapes must match, got {sample.shape} vs {velocity.shape}")
    s_curr = float(sigma_curr)
    s_next = float(sigma_next)
    if not (math.isfinite(s_curr) and math.isfinite(s_next)) or s_curr < 0.0 or s_next < 0.0:
        raise ValueError(f"sigmas must be finite and non-negative, got sigma_curr={s_curr}, sigma_next={s_next}")
    dt = s_next - s_curr
    updated = sample.to(torch.float32) + dt * (-velocity.to(torch.float32))
    return updated.to(dtype=sample.dtype)


class H3TinySchedule:
    """Per-modality shifted sigma grids, packed ``(t, r)`` row plans, and Euler update."""

    def __init__(
        self,
        num_steps: int = H3_TINY_FEWSTEP_DEFAULT_STEPS,
        *,
        video_shift: float = H3_TINY_VIDEO_SHIFT,
        audio_shift: float = H3_TINY_AUDIO_SHIFT,
    ) -> None:
        self.num_steps = int(num_steps)
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        self.video_sigmas = h3_tiny_shifted_sigmas(self.num_steps, self.video_shift)
        self.audio_sigmas = h3_tiny_shifted_sigmas(self.num_steps, self.audio_shift)
        self.video_timesteps = 1.0 - self.video_sigmas[:-1]
        self.audio_timesteps = 1.0 - self.audio_sigmas[:-1]
        self.video_endpoints = 1.0 - self.video_sigmas[1:]
        self.audio_endpoints = 1.0 - self.audio_sigmas[1:]

    def row_plan(
        self,
        *,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
        num_text_tokens: int,
        keyframe_noise_aug: float = H3_TINY_REFERENCE_TIMESTEP,
        condition_audio_timestep: float = H3_TINY_AUDIO_CONDITION_TIMESTEP,
        device: str | torch.device = "cpu",
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Build per-step ``(timestep, endpoint_timestep, timestep_indices)`` triples."""
        target_device = torch.device(device)
        plan: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(self.num_steps):
            t_v = float(self.video_timesteps[i])
            r_v = float(self.video_endpoints[i])
            t_a = float(self.audio_timesteps[i])
            r_a = float(self.audio_endpoints[i])
            triple = h3_tiny_row_time_pairs(
                video_indices=video_indices,
                audio_indices=audio_indices,
                num_condition_video_rows=num_condition_video_rows,
                num_condition_audio_rows=num_condition_audio_rows,
                num_text_tokens=num_text_tokens,
                video_time=(t_v, r_v),
                audio_time=(t_a, r_a),
                condition_video_timestep=max(t_v, float(keyframe_noise_aug)),
                condition_audio_timestep=float(condition_audio_timestep),
            )
            plan.append((triple[0].to(target_device), triple[1].to(target_device), triple[2].to(target_device)))
        return plan

    def step(
        self,
        velocity: torch.Tensor,
        sample: torch.Tensor,
        index: int,
        *,
        audio: bool = False,
    ) -> torch.Tensor:
        """Apply one Euler step on video or audio target rows."""
        sigmas = self.audio_sigmas if audio else self.video_sigmas
        return h3_tiny_euler_step(
            sample,
            velocity,
            sigma_curr=float(sigmas[index]),
            sigma_next=float(sigmas[index + 1]),
        )


__all__ = [
    "H3_TINY_AUDIO_CONDITION_TIMESTEP",
    "H3_TINY_AUDIO_SHIFT",
    "H3_TINY_FEWSTEP_DEFAULT_STEPS",
    "H3_TINY_FEWSTEP_MIN_STEPS",
    "H3_TINY_FULLSTEP_DEFAULT_STEPS",
    "H3_TINY_REFERENCE_TIMESTEP",
    "H3_TINY_VIDEO_SHIFT",
    "H3TinySchedule",
    "h3_tiny_euler_step",
    "h3_tiny_row_time_pairs",
    "h3_tiny_shifted_sigmas",
]
