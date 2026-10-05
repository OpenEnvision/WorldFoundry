"""The batch container the models read: video frames plus their aligned actions.

The models consume only ``video`` ``(B, T, C, H, W)`` uint8 and ``actions`` (an
:class:`~mira.world_model.actions_config.ActionTensors`). There is no audio: the codec
and world model are video-only here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mira.world_model.actions_config import ActionTensors


# A dataclass (not Pydantic) because Pydantic + torch.compile() has issues: Pydantic's validation
# can't be compiled, and model_construct() leads to an infinite loop during compilation.
@dataclass
class VideoActionBatch:
    """A batch of video frames and the aligned actions that produced them.

    Metadata should not live here — only what the models actually read.
    """

    # Plain ``torch.Tensor`` fields whose torch methods propagate generically. ``actions`` is an
    # ``ActionTensors`` container handled explicitly in ``to``/``pin_memory``/``clone`` below (it
    # implements the same methods), so device moves and copies still reach it.
    _TENSOR_ATTRIBUTES = ["video"]

    video: torch.Tensor  # (B, T, C, H, W) uint8
    actions: ActionTensors

    def to(self, *args: Any, **kwargs: Any) -> VideoActionBatch:
        """Propagate ``torch.Tensor.to`` to the video and the actions."""
        return VideoActionBatch(
            video=self.video.to(*args, **kwargs), actions=self.actions.to(*args, **kwargs)
        )

    def pin_memory(self) -> VideoActionBatch:
        """Propagate ``torch.Tensor.pin_memory`` to the video and the actions (called by the loader)."""
        return VideoActionBatch(video=self.video.pin_memory(), actions=self.actions.pin_memory())

    def clone(self) -> VideoActionBatch:
        """Deep-copy the batch, propagating ``torch.Tensor.clone`` to the video and the actions."""
        return VideoActionBatch(video=self.video.clone(), actions=self.actions.clone())

    def __len__(self) -> int:
        assert self.video.shape[0] == self.actions.batch_size
        return self.video.shape[0]

    def slice_time(self, start: int | None, end: int | None, *, fps: int) -> VideoActionBatch:
        """Slice the batch in time to ``[start, end)``. Views the tensors, does not copy.

        ``fps`` is accepted so callers can slice by a single time convention; video and actions are
        both frame-indexed here, so it is not needed to resolve the indices.
        """
        return VideoActionBatch(
            video=self.video[:, start:end],
            actions=self.actions.slice_time(start, end),
        )

    def cat_time(self, other: VideoActionBatch) -> VideoActionBatch:
        """Concatenate with another batch along the time dimension."""
        assert len(self) == len(other), f"Batch sizes must match: {len(self)} != {len(other)}"
        return VideoActionBatch(
            video=torch.cat([self.video, other.video.to(self.video.device)], dim=1),
            actions=self.actions.cat_time(other.actions.to(self.actions.key_presses.device)),
        )
