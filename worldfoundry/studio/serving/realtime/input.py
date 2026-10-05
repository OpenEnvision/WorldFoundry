"""Timestamped control input primitives for realtime world-model serving."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

SUPPORTED_CONTROL_KEYS = frozenset({"w", "a", "s", "d", "i", "j", "k", "l"})
KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
}


def normalize_control_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class RealtimeControlState:
    """Pressed-key state with last-pressed-wins conflict resolution."""

    pressed: set[str] = field(default_factory=set)
    _order: dict[str, int] = field(default_factory=dict)
    _sequence: int = 0

    def apply(self, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS:
            return False
        event = str(event or "").strip().lower()
        if event == "keydown":
            self.pressed.add(normalized)
            self._sequence += 1
            self._order[normalized] = self._sequence
            return True
        if event == "keyup":
            self.pressed.discard(normalized)
            self._order.pop(normalized, None)
            return True
        return False

    def _latest(self, keys: tuple[str, ...]) -> str | None:
        active = [key for key in keys if key in self.pressed]
        return max(active, key=lambda key: self._order.get(key, -1), default=None)

    def effective(self) -> frozenset[str]:
        return frozenset(
            key
            for key in (
                self._latest(("w", "s")),
                self._latest(("a", "d")),
                self._latest(("i", "k")),
                self._latest(("j", "l")),
            )
            if key is not None
        )


ControlSegment = tuple[float, float, frozenset[str]]


class RealtimeControlResampler:
    """Resample timestamped input edges into a model chunk timeline."""

    def __init__(self, *, fps: float, start_time: float = 0.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.next_chunk_start = float(start_time)
        self._events: deque[tuple[float, str, str]] = deque()
        self._state = RealtimeControlState()

    def on_edge(self, *, arrival_time: float, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS or event not in {"keydown", "keyup"}:
            return False
        self._events.append((float(arrival_time), event, normalized))
        return True

    def sample_chunk(self, num_frames: int, *, wall_time: float) -> list[ControlSegment]:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        duration = num_frames * self.dt
        if self.next_chunk_start <= 0.0 or wall_time - self.next_chunk_start > duration:
            self.next_chunk_start = float(wall_time)
        start = self.next_chunk_start
        end = start + duration

        while self._events and self._events[0][0] < start:
            _, event, key = self._events.popleft()
            self._state.apply(event, key)

        segments: list[ControlSegment] = []
        cursor = start
        effective = self._state.effective()
        while self._events and self._events[0][0] <= end:
            event_time, event, key = self._events.popleft()
            if event_time > cursor:
                segments.append((cursor, event_time, effective))
            self._state.apply(event, key)
            effective = self._state.effective()
            cursor = max(cursor, event_time)
        if cursor < end or not segments:
            segments.append((cursor, end, effective))
        self.next_chunk_start = end
        return segments

    def reset(self, *, start_time: float) -> None:
        self._events.clear()
        self._state = RealtimeControlState()
        self.next_chunk_start = float(start_time)

    @property
    def effective_keys(self) -> frozenset[str]:
        # Apply edges that have already arrived so key release can stop the
        # producer immediately after the in-flight chunk finishes.
        now = time.monotonic()
        while self._events and self._events[0][0] <= now:
            _, event, key = self._events.popleft()
            self._state.apply(event, key)
        return self._state.effective()


def interactions_from_keys(keys: frozenset[str]) -> list[str]:
    tokens: list[str] = []
    for key, token in (
        ("w", "forward"),
        ("s", "backward"),
        ("a", "left"),
        ("d", "right"),
        ("i", "camera_up"),
        ("k", "camera_down"),
        ("j", "camera_l"),
        ("l", "camera_r"),
    ):
        if key in keys:
            tokens.append(token)
    return tokens


def interactions_from_segments(segments: list[ControlSegment]) -> list[str]:
    """Return the most recent non-idle control state in a sampled chunk."""

    for _, _, keys in reversed(segments):
        interactions = interactions_from_keys(keys)
        if interactions:
            return interactions
    return []


__all__ = [
    "ControlSegment",
    "KEY_ALIASES",
    "RealtimeControlResampler",
    "RealtimeControlState",
    "SUPPORTED_CONTROL_KEYS",
    "interactions_from_keys",
    "interactions_from_segments",
    "normalize_control_key",
]
