"""Executable Wan2.2 TI2V-5B Stage0.5 flow-matching runtime."""

from __future__ import annotations

from typing import Any


# This public namespace is part of the deterministic sampling contract: changing
# it changes which samples drop the first-frame image condition.
def expand_timesteps_to_tokens(timestep: Any, frame_sequence_length: int) -> Any:
    batch, frames = timestep.shape
    return (
        timestep.unsqueeze(-1)
        .expand(batch, frames, int(frame_sequence_length))
        .reshape(batch, frames * int(frame_sequence_length))
    )


__all__ = ["expand_timesteps_to_tokens"]
