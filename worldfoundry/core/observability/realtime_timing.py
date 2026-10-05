# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral timing records for realtime generation sessions.

Per-chunk records fold into an interval window. Chunk metrics and JSONL
payloads stay process-local (monotonic clocks are not exported).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass


def _percentile_sorted(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile on a pre-sorted sample; empty input is a caller bug."""
    if not values:
        raise ValueError("Cannot compute a percentile for an empty sample.")
    if len(values) == 1:
        return values[0]
    position = min(max(float(percentile), 0.0), 1.0) * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _finite_values(values: Mapping[str, float]) -> dict[str, float]:
    """Drop NaN/inf gauges so a single bad CUDA timer cannot poison the JSONL window."""
    return {str(name): float(value) for name, value in values.items() if math.isfinite(float(value))}


@dataclass(frozen=True, slots=True)
class TimingDistribution:
    """Distribution summary for one timing stage, in milliseconds."""

    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float

    @classmethod
    def from_values(cls, values: list[float]) -> "TimingDistribution":
        """Summarize finite samples only; an all-NaN list is treated as empty and rejected."""
        ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
        if not ordered:
            raise ValueError("Cannot summarize an empty timing sample.")
        return cls(
            count=len(ordered),
            mean_ms=sum(ordered) / len(ordered),
            min_ms=ordered[0],
            max_ms=ordered[-1],
            p50_ms=_percentile_sorted(ordered, 0.5),
            p90_ms=_percentile_sorted(ordered, 0.9),
        )

    def to_payload(self) -> dict[str, int | float]:
        """Flat JSON object for one stage; field names stay stable for Studio dashboards."""
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
        }


@dataclass(frozen=True, slots=True)
class RealtimeChunkTiming:
    """One server-side chunk timing record.

    ``started_at_s`` and ``completed_at_s`` use the event loop's monotonic
    clock. They are retained for interval throughput calculations but are not
    exported as timestamps because their epoch is process-local.
    """

    session_id: str
    chunk_index: int
    transport: str
    started_at_s: float
    completed_at_s: float
    output_frames: int
    queue_depth: int
    dropped_frames: int
    warmup: bool
    stage_ms: Mapping[str, float]
    gauges: Mapping[str, float]

    def __post_init__(self) -> None:
        """Strip non-finite stage/gauge values before the record can enter a window."""
        object.__setattr__(self, "stage_ms", _finite_values(self.stage_ms))
        object.__setattr__(self, "gauges", _finite_values(self.gauges))

    def to_payload(self) -> dict[str, object]:
        """JSONL row; monotonic clocks stay local — only the derived ``server_chunk_ms`` is exported."""
        return {
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "transport": self.transport,
            "output_frames": self.output_frames,
            "queue_depth": self.queue_depth,
            "dropped_frames": self.dropped_frames,
            "warmup": self.warmup,
            "server_chunk_ms": max(
                (self.completed_at_s - self.started_at_s) * 1000.0,
                0.0,
            ),
            "stage_ms": dict(self.stage_ms),
            "gauges": dict(self.gauges),
        }


@dataclass(frozen=True, slots=True)
class RealtimeTimingSummary:
    """Measured, non-warmup samples from one reporting interval."""

    chunk_count: int
    frame_count: int
    first_chunk_index: int
    last_chunk_index: int
    interval_s: float
    throughput_fps: float
    dropped_frames: int
    stages: Mapping[str, TimingDistribution]
    gauges: Mapping[str, float]

    def to_payload(self) -> dict[str, object]:
        """Serialize one interval; stage keys are sorted so diffs stay stable."""
        return {
            "chunk_count": self.chunk_count,
            "frame_count": self.frame_count,
            "first_chunk_index": self.first_chunk_index,
            "last_chunk_index": self.last_chunk_index,
            "interval_s": self.interval_s,
            "throughput_fps": self.throughput_fps,
            "dropped_frames": self.dropped_frames,
            "stages": {name: distribution.to_payload() for name, distribution in sorted(self.stages.items())},
            "gauges": dict(self.gauges),
        }


class RealtimeTimingWindow:
    """Collect chunk timings and summarize the current logging interval."""

    def __init__(self) -> None:
        """Empty window; drop counters start at zero until the first measured chunk."""
        self._samples: list[RealtimeChunkTiming] = []
        self._observed_chunks = 0
        self._dropped_at_start = 0
        self._last_dropped_frames = 0

    @property
    def observed_chunks(self) -> int:
        """All chunks observed in this interval, including warmup chunks."""

        return self._observed_chunks

    @property
    def measured_chunks(self) -> int:
        """Non-warmup samples that will appear in the next :meth:`summary`."""
        return len(self._samples)

    def record(self, timing: RealtimeChunkTiming) -> None:
        """Accept one chunk; warmup rows increment ``observed_chunks`` but stay out of the summary."""
        previous_dropped_frames = self._last_dropped_frames
        self._last_dropped_frames = max(int(timing.dropped_frames), 0)
        self._observed_chunks += 1
        if timing.warmup:
            return
        if not self._samples:
            self._dropped_at_start = previous_dropped_frames
        self._samples.append(timing)

    def summary(self) -> RealtimeTimingSummary | None:
        """Fold measured samples into one interval; ``None`` when only warmup (or nothing) arrived."""
        if not self._samples:
            return None
        grouped: dict[str, list[float]] = defaultdict(list)
        for sample in self._samples:
            for stage_name, duration_ms in sample.stage_ms.items():
                grouped[stage_name].append(duration_ms)
        first = self._samples[0]
        last = self._samples[-1]
        interval_s = max(last.completed_at_s - first.started_at_s, 0.0)
        frame_count = sum(sample.output_frames for sample in self._samples)
        return RealtimeTimingSummary(
            chunk_count=len(self._samples),
            frame_count=frame_count,
            first_chunk_index=first.chunk_index,
            last_chunk_index=last.chunk_index,
            interval_s=interval_s,
            throughput_fps=(frame_count / interval_s if interval_s > 1.0e-9 else 0.0),
            dropped_frames=max(self._last_dropped_frames - self._dropped_at_start, 0),
            stages={name: TimingDistribution.from_values(values) for name, values in sorted(grouped.items())},
            gauges=dict(last.gauges),
        )

    def reset_interval(self) -> None:
        """Start a new report interval while retaining cumulative drop state."""

        self._samples.clear()
        self._observed_chunks = 0
        self._dropped_at_start = self._last_dropped_frames


__all__ = [
    "RealtimeChunkTiming",
    "RealtimeTimingSummary",
    "RealtimeTimingWindow",
    "TimingDistribution",
]
