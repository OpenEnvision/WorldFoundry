"""Stateful rollout memory used by native Echo-Memory pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from worldfoundry.core.memory import BaseMemory


def replay_context_indices(frame_count: int, context_frames: int) -> list[int]:
    """Return Echo's newest-first plus uniform-history frame indices."""

    frame_count = int(frame_count)
    context_frames = int(context_frames)
    if frame_count <= 0 or context_frames <= 0:
        return []
    selected = min(frame_count, context_frames)
    if selected == 1:
        return [frame_count - 1]
    historical = selected - 1
    if historical == 1:
        return [frame_count - 1, 0]
    indices = [int(round(index * (frame_count - 2) / (historical - 1))) for index in range(historical)]
    return [frame_count - 1, *indices]


class EchoRolloutMemory(BaseMemory):
    """Select replay-style visual context from the latest generated chunk."""

    def __init__(self, *, context_frames: int, capacity: int = 2, **kwargs: Any) -> None:
        super().__init__(capacity=capacity, **kwargs)
        if int(context_frames) <= 0:
            raise ValueError("context_frames must be positive")
        self.context_frames = int(context_frames)

    def record(
        self,
        data: Sequence[Any],
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        frames = list(data)
        if not frames:
            raise ValueError("cannot record an empty Echo rollout chunk")
        return self.append_record(frames, kind="video", metadata=metadata)

    def select(self, context_query: Any = None, **kwargs: Any) -> list[Any]:
        requested = kwargs.get("context_frames")
        if isinstance(context_query, Mapping):
            requested = context_query.get("context_frames", requested)
        elif isinstance(context_query, int):
            requested = context_query
        count = int(requested if requested is not None else self.context_frames)
        latest = self.latest_record(prefer_type="video")
        if latest is None:
            return []
        frames = list(latest["content"])
        return [frames[index] for index in replay_context_indices(len(frames), count)]

    def compress(self, memory_items: Sequence[Any], **kwargs: Any) -> list[Any]:
        del kwargs
        return list(memory_items)

    def process(
        self,
        refined_data: Sequence[Any],
        target_format: str = "frames",
        **kwargs: Any,
    ) -> list[Any]:
        del kwargs
        if target_format != "frames":
            raise ValueError(f"EchoRolloutMemory only emits frames, got {target_format!r}")
        return list(refined_data)

    def manage(self, **kwargs: Any) -> None:
        del kwargs
        # Capacity eviction is atomic inside MemoryStore.append().
        return None


__all__ = ["EchoRolloutMemory", "replay_context_indices"]
