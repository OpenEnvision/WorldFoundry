"""Causal denoising schedule and aligned audio-video block layout."""

from __future__ import annotations

from itertools import pairwise

from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import LTX2Scheduler

DEFAULT_CAUSAL_TIMESTEPS = (1000, 750, 500, 250)
CAUSAL_VIDEO_CHUNK_SIZE = 3
AUDIO_PREFIX_FRAMES = 2
AUDIO_FRAMES_PER_VIDEO_BLOCK = 25


def resolve_causal_sigmas(
    timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
    *,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Map distilled student timesteps to model sigmas without appending zero."""
    if not timesteps:
        raise ValueError("at least one causal timestep is required")
    schedule = LTX2Scheduler().execute(steps=num_train_timesteps)
    result = []
    for timestep in timesteps:
        index = num_train_timesteps - int(timestep)
        if not 0 <= index < len(schedule):
            raise ValueError(f"causal timestep {timestep} is outside [0, {num_train_timesteps}]")
        result.append(float(schedule[index]))
    if any(current <= following for current, following in pairwise(result)):
        raise ValueError(f"causal sigmas must be strictly descending, got {result}")
    return result


def _causal_block_count(video_frames: int, chunk_size: int) -> int:
    if chunk_size != CAUSAL_VIDEO_CHUNK_SIZE:
        raise ValueError(f"Echo-WM Flash requires video_chunk_size={CAUSAL_VIDEO_CHUNK_SIZE}, got {chunk_size}")
    if video_frames < 1 or (video_frames - 1) % chunk_size:
        raise ValueError(f"latent video length must be 1 + n * chunk_size, got {video_frames}")
    return (video_frames - 1) // chunk_size


def causal_video_blocks(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """Return ``[0, 1]`` followed by fixed-size causal generation blocks."""
    _causal_block_count(video_frames, chunk_size)
    return [(0, 1), *[(start, start + chunk_size) for start in range(1, video_frames, chunk_size)]]


def causal_audio_frames(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> int:
    """Map video latent frames to the aligned audio layout."""
    block_count = _causal_block_count(video_frames, chunk_size)
    return AUDIO_PREFIX_FRAMES + block_count * AUDIO_FRAMES_PER_VIDEO_BLOCK


def causal_audio_blocks(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """Return the audio blocks paired with :func:`causal_video_blocks`."""
    total = causal_audio_frames(video_frames, chunk_size)
    return [
        (0, AUDIO_PREFIX_FRAMES),
        *[
            (start, min(start + AUDIO_FRAMES_PER_VIDEO_BLOCK, total))
            for start in range(AUDIO_PREFIX_FRAMES, total, AUDIO_FRAMES_PER_VIDEO_BLOCK)
        ],
    ]
