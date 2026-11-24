"""Timestamped control input primitives for realtime world-model serving."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from worldfoundry.core.execution.realtime.contracts import DEFAULT_REALTIME_KEYS, RealtimeAxis, RealtimeKey

SUPPORTED_CONTROL_KEYS = frozenset(binding.key for binding in DEFAULT_REALTIME_KEYS)
KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
}


def normalize_control_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


def _key_groups(realtime_keys: tuple[RealtimeKey, ...]) -> tuple[tuple[str, ...], ...]:
    """Conflict groups in declaration order; an ungrouped key forms its own group."""

    groups: dict[tuple[str, str], list[str]] = {}
    for binding in realtime_keys:
        groups.setdefault((binding.group, "") if binding.group else ("", binding.key), []).append(binding.key)
    return tuple(tuple(keys) for keys in groups.values())


@dataclass(slots=True)
class RealtimeControlState:
    """Pressed-key state with last-pressed-wins conflict resolution inside each key group."""

    realtime_keys: tuple[RealtimeKey, ...] = DEFAULT_REALTIME_KEYS
    pressed: set[str] = field(default_factory=set)
    _order: dict[str, int] = field(default_factory=dict)
    _sequence: int = 0

    def apply(self, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if all(binding.key != normalized for binding in self.realtime_keys):
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
            key for key in (self._latest(group) for group in _key_groups(self.realtime_keys)) if key is not None
        )


ControlSegment = tuple[float, float, frozenset[str]]
_MESSAGE_AXIS_SOURCES = {
    "mouse": (("dx", "mouse_x"), ("dy", "mouse_y")),
    "gamepad": (("lx", "gamepad_lx"), ("ly", "gamepad_ly"), ("rx", "gamepad_rx"), ("ry", "gamepad_ry")),
}


def axis_samples_from_message(payload: Mapping[str, Any]) -> dict[str, float]:
    """Map a ``{"type": "axis"}`` message to finite samples keyed by axis source.

    Mouse messages carry ``dx``/``dy`` deltas and gamepad messages carry
    ``lx``/``ly``/``rx``/``ry`` stick levels; missing fields are ignored.
    """

    samples: dict[str, float] = {}
    for field_name, source in _MESSAGE_AXIS_SOURCES.get(str(payload.get("source") or "").lower(), ()):
        value = payload.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            samples[source] = float(value)
    return samples


def _axis_value(axis: RealtimeAxis, raw: float) -> float:
    """Apply the deadzone (level axes only), scale, then the per-frame limit."""

    if axis.mode == "level" and abs(raw) < axis.deadzone:
        return 0.0
    value = raw * axis.scale
    if axis.limit is not None:
        value = min(max(value, -axis.limit), axis.limit)
    return value


@dataclass(frozen=True, slots=True)
class RealtimeInputChunk:
    """Key segments, per-frame axes, and the IDs of axis inputs used by this chunk."""

    segments: list[ControlSegment]
    axes: dict[str, list[float]]
    axis_control_ids: tuple[str, ...] = ()

    @property
    def has_axis_input(self) -> bool:
        """Whether any declared axis is non-zero in at least one frame."""
        return any(value != 0.0 for values in self.axes.values() for value in values)


class RealtimeControlResampler:
    """Resample timestamped key edges and axis samples into a model chunk timeline."""

    def __init__(
        self,
        *,
        fps: float,
        start_time: float = 0.0,
        realtime_keys: tuple[RealtimeKey, ...] = DEFAULT_REALTIME_KEYS,
        axes: tuple[RealtimeAxis, ...] = (),
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        unreachable = [binding.key for binding in realtime_keys if normalize_control_key(binding.key) != binding.key]
        if unreachable:
            raise ValueError(f"Realtime keys must use normalized key names; cannot match {unreachable!r}")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.next_chunk_start = float(start_time)
        self.realtime_keys = realtime_keys
        self.axes = axes
        self._accepted_keys = frozenset(binding.key for binding in realtime_keys)
        self._axis_sources = frozenset(axis.source for axis in axes)
        self._events: deque[tuple[float, str, str]] = deque()
        self._axis_events: deque[tuple[float, str, float, str | None]] = deque()
        self._source_control_ids: dict[str, str | None] = dict.fromkeys(self._axis_sources)
        self._state = RealtimeControlState(realtime_keys)
        self._live_state = RealtimeControlState(realtime_keys)
        self._source_levels = dict.fromkeys(self._axis_sources, 0.0)
        self._live_source_levels = dict.fromkeys(self._axis_sources, 0.0)
        self._event_delay = 0.0
        self._minimum_state_end = float(start_time)

    def on_edge(self, *, arrival_time: float, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in self._accepted_keys or event not in {"keydown", "keyup"}:
            return False
        self._live_state.apply(event, normalized)
        self._events.append((float(arrival_time) + self._event_delay, event, normalized))
        return True

    def on_axis(self, *, arrival_time: float, samples: Mapping[str, float], control_id: str | None = None) -> bool:
        """Queue source samples for declared axes; return whether any was accepted.

        Each sample is a delta for ``delta`` axes and the current level for
        ``level`` axes of the same source.
        """

        stamp = float(arrival_time) + self._event_delay
        accepted = False
        for source, value in samples.items():
            if source not in self._axis_sources:
                continue
            self._live_source_levels[source] = float(value)
            self._axis_events.append((stamp, source, float(value), control_id))
            accepted = True
        return accepted

    def _delay_events(self, delay: float) -> None:
        first_delayed = self._events[0][0]
        self._events = deque((stamp + delay, event, key) for stamp, event, key in self._events)
        # Axis samples received after the postponed key edge keep their order behind it.
        self._axis_events = deque(
            (stamp + delay if stamp >= first_delayed else stamp, source, value, control_id)
            for stamp, source, value, control_id in self._axis_events
        )
        self._event_delay += delay

    def sample_chunk(self, num_frames: int, *, wall_time: float) -> list[ControlSegment]:
        """Consume controls for one chunk and return its key segments."""

        return self.sample_input_chunk(num_frames, wall_time=wall_time).segments

    def sample_input_chunk(self, num_frames: int, *, wall_time: float) -> RealtimeInputChunk:
        """Consume controls, retaining each new state for a native sampling interval.

        Late releases of consumed holds apply at the chunk boundary. Unsampled
        taps move onto the next available timeline, together with subsequent
        edges, so their order and longer durations survive inference latency.
        Axis samples that arrived before the chunk start apply to its first
        frame; samples stamped at or after the chunk end stay queued for a
        later chunk.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        duration = num_frames * self.dt
        interval = self.dt
        if self.next_chunk_start <= 0.0 or wall_time - self.next_chunk_start > duration:
            self.next_chunk_start = float(wall_time)
        start = self.next_chunk_start
        end = start + duration
        segments: list[ControlSegment] = []
        cursor = start
        effective = self._state.effective()
        while self._events:
            stamp = self._events[0][0]
            candidate = replace(self._state, pressed=set(self._state.pressed), _order=dict(self._state._order))
            count = 0
            for event_time, event, key in self._events:
                if event_time != stamp:
                    break
                candidate.apply(event, key)
                count += 1
            next_effective = candidate.effective()
            event_time = max(stamp, start)
            if next_effective != effective:
                event_time = max(event_time, self._minimum_state_end)
                # A new active state needs a complete interval: both frame-end
                # and frame-midpoint consumers must see it before its release.
                if next_effective and event_time + interval > end + 1e-8:
                    event_time = max(event_time, end)
                if event_time > stamp and (next_effective or event_time > start):
                    self._delay_events(event_time - stamp)
            if event_time >= end:
                break
            for _ in range(count):
                self._events.popleft()
            if next_effective != effective:
                if event_time > cursor:
                    segments.append((cursor, event_time, effective))
                cursor = event_time
                self._minimum_state_end = event_time + interval
            self._state = candidate
            effective = next_effective
        if cursor < end:
            segments.append((cursor, end, effective))
        axes, control_ids = self._sample_axes(num_frames, start=start, end=end)
        self.next_chunk_start = end
        if not self._events and not self._axis_events:
            self._event_delay = 0.0
        return RealtimeInputChunk(segments=segments, axes=axes, axis_control_ids=control_ids)

    def _sample_axes(
        self, num_frames: int, *, start: float, end: float
    ) -> tuple[dict[str, list[float]], tuple[str, ...]]:
        """Bin axis samples and retain only IDs used by the sampled frame values."""

        if not self.axes:
            return {}, ()
        deltas = {source: [0.0] * num_frames for source in self._axis_sources}
        delta_controls: dict[str, list[str]] = {source: [] for source in self._axis_sources}
        last_levels: dict[str, list[float | None]] = {source: [None] * num_frames for source in self._axis_sources}
        last_controls: dict[str, list[str | None]] = {source: [None] * num_frames for source in self._axis_sources}
        while self._axis_events and self._axis_events[0][0] < end:
            stamp, source, value, control_id = self._axis_events.popleft()
            # The small offset keeps a sample stamped on a frame boundary in
            # the frame that starts there despite float rounding.
            index = min(max(int((stamp - start) * self.fps + 1e-6), 0), num_frames - 1)
            deltas[source][index] += value
            if control_id is not None:
                delta_controls[source].append(control_id)
            last_levels[source][index] = value
            last_controls[source][index] = control_id
        levels: dict[str, list[float]] = {}
        level_controls: dict[str, list[str | None]] = {}
        for source, samples in last_levels.items():
            held = self._source_levels[source]
            held_control = self._source_control_ids[source]
            frames: list[float] = []
            controls: list[str | None] = []
            for index, sample in enumerate(samples):
                if sample is not None:
                    held = sample
                    held_control = last_controls[source][index]
                frames.append(held)
                controls.append(held_control)
            self._source_levels[source] = held
            self._source_control_ids[source] = held_control
            levels[source] = frames
            level_controls[source] = controls
        axes: dict[str, list[float]] = {}
        control_ids: list[str | None] = []
        for axis in self.axes:
            values = [_axis_value(axis, raw) for raw in (deltas if axis.mode == "delta" else levels)[axis.source]]
            axes[axis.name] = values
            if axis.mode == "delta":
                control_ids.extend(delta_controls[axis.source])
            else:
                # A held nonzero level still conditions later chunks. Neutral
                # samples belong to their own chunk, not every later idle frame.
                control_ids.extend(
                    control_id
                    for index, control_id in enumerate(level_controls[axis.source])
                    if last_levels[axis.source][index] is not None or values[index] != 0.0
                )
        return axes, tuple(dict.fromkeys(value for value in control_ids if value is not None))

    def reset(self, *, start_time: float) -> None:
        self._events.clear()
        self._axis_events.clear()
        self._source_control_ids = dict.fromkeys(self._axis_sources)
        self._state = RealtimeControlState(self.realtime_keys)
        self._live_state = RealtimeControlState(self.realtime_keys)
        self._source_levels = dict.fromkeys(self._axis_sources, 0.0)
        self._live_source_levels = dict.fromkeys(self._axis_sources, 0.0)
        self._event_delay = 0.0
        self._minimum_state_end = float(start_time)
        self.next_chunk_start = float(start_time)

    @property
    def effective_keys(self) -> frozenset[str]:
        """Current received key state; observing it never consumes model input."""
        return self._live_state.effective()

    @property
    def has_pending_input(self) -> bool:
        """Whether held controls, unsampled samples, or held axis levels still need a chunk boundary."""
        return bool(
            self._events
            or self.effective_keys
            or self._axis_events
            or any(
                _axis_value(axis, self._live_source_levels[axis.source]) != 0.0
                for axis in self.axes
                if axis.mode == "level"
            )
        )


def interactions_from_keys(
    keys: frozenset[str],
    realtime_keys: tuple[RealtimeKey, ...] = DEFAULT_REALTIME_KEYS,
) -> list[str]:
    """Return the actions of the active keys in declaration order."""

    return [binding.action for binding in realtime_keys if binding.key in keys]


def interactions_from_segments(
    segments: list[ControlSegment],
    realtime_keys: tuple[RealtimeKey, ...] = DEFAULT_REALTIME_KEYS,
) -> list[str]:
    """Return the most recent non-idle control state in a sampled chunk."""

    for _, _, keys in reversed(segments):
        interactions = interactions_from_keys(keys, realtime_keys)
        if interactions:
            return interactions
    return []


__all__ = [
    "KEY_ALIASES",
    "SUPPORTED_CONTROL_KEYS",
    "ControlSegment",
    "RealtimeControlResampler",
    "RealtimeControlState",
    "RealtimeInputChunk",
    "axis_samples_from_message",
    "interactions_from_keys",
    "interactions_from_segments",
    "normalize_control_key",
]
