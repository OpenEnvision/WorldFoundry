"""Manifest-driven regression checks for Studio realtime timing traces."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from worldfoundry.core.observability.realtime_timing import TimingDistribution
from worldfoundry.runtime.performance import (
    OptimizationSnapshot,
    PerformanceManifest,
    PerformanceMetrics,
    RuntimeFingerprint,
)

REALTIME_REGRESSION_SCHEMA_VERSION = "worldfoundry-realtime-regression-v1"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    return parsed


def _non_negative_float(value: Any, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    return None if value is None else _non_negative_float(value, name=name)


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _threshold_map(value: Any, *, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object keyed by metric name.")
    return {str(metric): _non_negative_float(limit, name=f"{name}.{metric}") for metric, limit in value.items()}


@dataclass(frozen=True, slots=True)
class RealtimeRegressionThresholds:
    """Absolute performance and congestion limits for one realtime case."""

    min_chunks: int = 1
    min_output_frames: int = 1
    min_throughput_fps: float | None = None
    max_dropped_frames: int | None = None
    max_queue_depth: int | None = None
    max_stage_p50_ms: Mapping[str, float] = field(default_factory=dict)
    max_stage_p90_ms: Mapping[str, float] = field(default_factory=dict)
    max_stage_max_ms: Mapping[str, float] = field(default_factory=dict)
    min_gauges: Mapping[str, float] = field(default_factory=dict)
    max_gauges: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_chunks", _non_negative_int(self.min_chunks, name="min_chunks"))
        object.__setattr__(
            self,
            "min_output_frames",
            _non_negative_int(self.min_output_frames, name="min_output_frames"),
        )
        object.__setattr__(
            self,
            "min_throughput_fps",
            _optional_non_negative_float(self.min_throughput_fps, name="min_throughput_fps"),
        )
        for name in ("max_dropped_frames", "max_queue_depth"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _non_negative_int(value, name=name))
        for name in (
            "max_stage_p50_ms",
            "max_stage_p90_ms",
            "max_stage_max_ms",
            "min_gauges",
            "max_gauges",
        ):
            object.__setattr__(self, name, _threshold_map(getattr(self, name), name=name))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealtimeRegressionThresholds":
        return cls(
            min_chunks=data.get("min_chunks", 1),
            min_output_frames=data.get("min_output_frames", 1),
            min_throughput_fps=data.get("min_throughput_fps"),
            max_dropped_frames=data.get("max_dropped_frames"),
            max_queue_depth=data.get("max_queue_depth"),
            max_stage_p50_ms=data.get("max_stage_p50_ms", {}),
            max_stage_p90_ms=data.get("max_stage_p90_ms", {}),
            max_stage_max_ms=data.get("max_stage_max_ms", {}),
            min_gauges=data.get("min_gauges", {}),
            max_gauges=data.get("max_gauges", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_chunks": self.min_chunks,
            "min_output_frames": self.min_output_frames,
            "min_throughput_fps": self.min_throughput_fps,
            "max_dropped_frames": self.max_dropped_frames,
            "max_queue_depth": self.max_queue_depth,
            "max_stage_p50_ms": dict(self.max_stage_p50_ms),
            "max_stage_p90_ms": dict(self.max_stage_p90_ms),
            "max_stage_max_ms": dict(self.max_stage_max_ms),
            "min_gauges": dict(self.min_gauges),
            "max_gauges": dict(self.max_gauges),
        }


@dataclass(frozen=True, slots=True)
class RealtimeRegressionCase:
    """Trace selector and thresholds for one model/transport profile."""

    name: str
    thresholds: RealtimeRegressionThresholds = field(default_factory=RealtimeRegressionThresholds)
    model_id: str | None = None
    transport: str | None = None
    session_id: str | None = None
    exclude_warmup_chunks: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Realtime regression case name must not be empty.")
        object.__setattr__(
            self,
            "exclude_warmup_chunks",
            _non_negative_int(self.exclude_warmup_chunks, name="exclude_warmup_chunks"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealtimeRegressionCase":
        selectors = data.get("selectors")
        selectors = selectors if isinstance(selectors, Mapping) else {}
        thresholds = data.get("thresholds")
        return cls(
            name=str(data.get("name") or ""),
            model_id=_optional_text(selectors.get("model_id", data.get("model_id"))),
            transport=_optional_text(selectors.get("transport", data.get("transport"))),
            session_id=_optional_text(selectors.get("session_id", data.get("session_id"))),
            exclude_warmup_chunks=data.get("exclude_warmup_chunks", 0),
            thresholds=RealtimeRegressionThresholds.from_dict(thresholds if isinstance(thresholds, Mapping) else {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selectors": {
                "model_id": self.model_id,
                "transport": self.transport,
                "session_id": self.session_id,
            },
            "exclude_warmup_chunks": self.exclude_warmup_chunks,
            "thresholds": self.thresholds.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RealtimeRegressionManifest:
    """Serializable collection of realtime regression cases."""

    cases: tuple[RealtimeRegressionCase, ...]
    schema_version: str = REALTIME_REGRESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("Realtime regression manifest must define at least one case.")
        names = [case.name for case in self.cases]
        if len(set(names)) != len(names):
            raise ValueError("Realtime regression case names must be unique.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealtimeRegressionManifest":
        raw_cases = data.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes, bytearray)):
            raise TypeError("Realtime regression manifest 'cases' must be a JSON list.")
        cases = tuple(RealtimeRegressionCase.from_dict(item) for item in raw_cases if isinstance(item, Mapping))
        if len(cases) != len(raw_cases):
            raise TypeError("Every realtime regression case must be a JSON object.")
        return cls(
            cases=cases,
            schema_version=str(data.get("schema_version") or REALTIME_REGRESSION_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RealtimeRegressionManifest":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise TypeError("Realtime regression manifest JSON must contain an object.")
        return cls.from_dict(payload)

    @classmethod
    def read_json(cls, path: str | Path) -> "RealtimeRegressionManifest":
        return cls.from_json(Path(path).read_bytes())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cases": [case.to_dict() for case in self.cases],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        return destination


@dataclass(frozen=True, slots=True)
class RealtimeTraceSample:
    """Normalized ``realtime.chunk_timing`` JSONL record."""

    session_id: str
    chunk_index: int
    transport: str
    model_id: str | None
    output_frames: int
    queue_depth: int
    dropped_frames: int
    warmup: bool
    server_chunk_ms: float
    stage_ms: Mapping[str, float]
    gauges: Mapping[str, float]

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "RealtimeTraceSample":
        fields = event.get("fields")
        fields = fields if isinstance(fields, Mapping) else event
        stage_ms = fields.get("stage_ms")
        gauges = fields.get("gauges")
        if not isinstance(stage_ms, Mapping):
            stage_ms = {}
        if not isinstance(gauges, Mapping):
            gauges = {}
        server_chunk_ms = fields.get("server_chunk_ms", stage_ms.get("server_chunk_ms", 0.0))
        session_id = str(fields.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("realtime timing event is missing session_id")
        return cls(
            session_id=session_id,
            chunk_index=_non_negative_int(fields.get("chunk_index", 0), name="chunk_index"),
            transport=str(fields.get("transport") or "unknown"),
            model_id=_optional_text(event.get("model_id") or fields.get("model_id")),
            output_frames=_non_negative_int(fields.get("output_frames", 0), name="output_frames"),
            queue_depth=_non_negative_int(fields.get("queue_depth", 0), name="queue_depth"),
            dropped_frames=_non_negative_int(fields.get("dropped_frames", 0), name="dropped_frames"),
            warmup=_boolean(fields.get("warmup", False), name="warmup"),
            server_chunk_ms=_non_negative_float(server_chunk_ms, name="server_chunk_ms"),
            stage_ms={
                str(name): _non_negative_float(value, name=f"stage_ms.{name}") for name, value in stage_ms.items()
            },
            gauges={str(name): _finite_float(value, name=f"gauges.{name}") for name, value in gauges.items()},
        )


def read_realtime_trace(path: str | Path) -> list[RealtimeTraceSample]:
    """Read canonical log JSONL and ignore unrelated lifecycle events."""

    samples: list[RealtimeTraceSample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on realtime trace line {line_number}.") from exc
        if not isinstance(event, Mapping):
            raise TypeError(f"Realtime trace line {line_number} must contain a JSON object.")
        if event.get("event") not in (None, "realtime.chunk_timing"):
            continue
        fields = event.get("fields")
        candidate = fields if isinstance(fields, Mapping) else event
        if "session_id" not in candidate or "chunk_index" not in candidate:
            continue
        try:
            samples.append(RealtimeTraceSample.from_event(event))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid realtime timing event on line {line_number}: {exc}") from exc
    return samples


@dataclass(frozen=True, slots=True)
class RealtimeTraceSummary:
    session_id: str
    chunk_count: int
    frame_count: int
    server_time_ms: float
    throughput_fps: float
    dropped_frames: int
    max_queue_depth: int
    stages: Mapping[str, TimingDistribution]
    gauges: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chunk_count": self.chunk_count,
            "frame_count": self.frame_count,
            "server_time_ms": self.server_time_ms,
            "throughput_fps": self.throughput_fps,
            "dropped_frames": self.dropped_frames,
            "max_queue_depth": self.max_queue_depth,
            "stages": {name: value.to_payload() for name, value in sorted(self.stages.items())},
            "gauges": dict(self.gauges),
        }


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    metric: str
    comparator: Literal[">=", "<="]
    limit: float | int
    actual: float | int | None

    @property
    def passed(self) -> bool:
        if self.actual is None:
            return False
        if self.comparator == ">=":
            return self.actual >= self.limit
        return self.actual <= self.limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "comparator": self.comparator,
            "limit": self.limit,
            "actual": self.actual,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class RealtimeRegressionResult:
    case: RealtimeRegressionCase
    checks: tuple[RegressionCheck, ...]
    summary: RealtimeTraceSummary | None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[RegressionCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "passed": self.passed,
            "summary": None if self.summary is None else self.summary.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_performance_manifest(
        self,
        *,
        fingerprint: RuntimeFingerprint | None = None,
        optimization: OptimizationSnapshot | None = None,
    ) -> PerformanceManifest:
        summary = self.summary
        timings = (
            {}
            if summary is None
            else {name: distribution.to_payload() for name, distribution in summary.stages.items()}
        )
        return PerformanceManifest(
            model={"model_id": self.case.model_id or "unknown"},
            workload={
                "kind": "realtime",
                "case": self.case.name,
                "transport": self.case.transport,
                "session_id": None if summary is None else summary.session_id,
            },
            fingerprint=fingerprint or RuntimeFingerprint(),
            optimization=optimization or OptimizationSnapshot(),
            metrics=PerformanceMetrics(
                timings_ms=timings,
                throughput={
                    "frames_per_second": 0.0 if summary is None else summary.throughput_fps,
                },
                batch_counters={
                    "chunks": 0 if summary is None else summary.chunk_count,
                    "frames": 0 if summary is None else summary.frame_count,
                    "dropped_frames": 0 if summary is None else summary.dropped_frames,
                    "max_queue_depth": 0 if summary is None else summary.max_queue_depth,
                },
            ),
            extensions={
                "realtime_regression": {
                    "passed": self.passed,
                    "checks": [check.to_dict() for check in self.checks],
                }
            },
        )


@dataclass(frozen=True, slots=True)
class RealtimeRegressionRun:
    results: tuple[RealtimeRegressionResult, ...]
    schema_version: str = REALTIME_REGRESSION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_realtime_case(
    case: RealtimeRegressionCase,
    samples: Sequence[RealtimeTraceSample],
) -> RealtimeRegressionResult:
    """Select the latest matching session and evaluate every case threshold."""

    selected = [sample for sample in samples if _sample_matches(case, sample)]
    if case.session_id is None and selected:
        latest_session = selected[-1].session_id
        selected = [sample for sample in selected if sample.session_id == latest_session]
    measured = [sample for sample in selected if not sample.warmup]
    measured = measured[case.exclude_warmup_chunks :]
    summary = _summarize(selected, measured) if measured else None
    checks = _checks(case.thresholds, summary)
    return RealtimeRegressionResult(case=case, checks=tuple(checks), summary=summary)


def evaluate_realtime_manifest(
    manifest: RealtimeRegressionManifest,
    samples: Sequence[RealtimeTraceSample],
) -> RealtimeRegressionRun:
    return RealtimeRegressionRun(
        results=tuple(evaluate_realtime_case(case, samples) for case in manifest.cases),
        schema_version=manifest.schema_version,
    )


def _sample_matches(case: RealtimeRegressionCase, sample: RealtimeTraceSample) -> bool:
    if case.session_id is not None and sample.session_id != case.session_id:
        return False
    if case.transport is not None and sample.transport != case.transport:
        return False
    # Older timing traces did not always bind model_id into the log context.
    if case.model_id is not None and sample.model_id is not None and sample.model_id != case.model_id:
        return False
    return True


def _summarize(
    selected: Sequence[RealtimeTraceSample],
    measured: Sequence[RealtimeTraceSample],
) -> RealtimeTraceSummary:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in measured:
        for name, value in sample.stage_ms.items():
            grouped[name].append(value)
    first_index = next(index for index, sample in enumerate(selected) if sample is measured[0])
    dropped_before = selected[first_index - 1].dropped_frames if first_index else 0
    dropped_frames = 0
    previous_drops = dropped_before
    for sample in measured:
        dropped_frames += max(sample.dropped_frames - previous_drops, 0)
        previous_drops = sample.dropped_frames
    server_time_ms = sum(sample.server_chunk_ms for sample in measured)
    frame_count = sum(sample.output_frames for sample in measured)
    return RealtimeTraceSummary(
        session_id=measured[0].session_id,
        chunk_count=len(measured),
        frame_count=frame_count,
        server_time_ms=server_time_ms,
        throughput_fps=frame_count * 1000.0 / server_time_ms if server_time_ms > 0.0 else 0.0,
        dropped_frames=dropped_frames,
        max_queue_depth=max(sample.queue_depth for sample in measured),
        stages={name: TimingDistribution.from_values(values) for name, values in grouped.items()},
        gauges=dict(measured[-1].gauges),
    )


def _checks(
    thresholds: RealtimeRegressionThresholds,
    summary: RealtimeTraceSummary | None,
) -> list[RegressionCheck]:
    checks = [
        RegressionCheck(
            "chunks",
            ">=",
            thresholds.min_chunks,
            None if summary is None else summary.chunk_count,
        ),
        RegressionCheck(
            "output_frames",
            ">=",
            thresholds.min_output_frames,
            None if summary is None else summary.frame_count,
        ),
    ]
    if thresholds.min_throughput_fps is not None:
        checks.append(
            RegressionCheck(
                "throughput_fps",
                ">=",
                thresholds.min_throughput_fps,
                None if summary is None else summary.throughput_fps,
            )
        )
    if thresholds.max_dropped_frames is not None:
        checks.append(
            RegressionCheck(
                "dropped_frames",
                "<=",
                thresholds.max_dropped_frames,
                None if summary is None else summary.dropped_frames,
            )
        )
    if thresholds.max_queue_depth is not None:
        checks.append(
            RegressionCheck(
                "max_queue_depth",
                "<=",
                thresholds.max_queue_depth,
                None if summary is None else summary.max_queue_depth,
            )
        )
    for limits, field_name in (
        (thresholds.max_stage_p50_ms, "p50_ms"),
        (thresholds.max_stage_p90_ms, "p90_ms"),
        (thresholds.max_stage_max_ms, "max_ms"),
    ):
        for stage, limit in sorted(limits.items()):
            distribution = None if summary is None else summary.stages.get(stage)
            complete = distribution is not None and summary is not None and distribution.count == summary.chunk_count
            checks.append(
                RegressionCheck(
                    f"stages.{stage}.{field_name}",
                    "<=",
                    limit,
                    getattr(distribution, field_name) if complete else None,
                )
            )
    for limits, comparator in (
        (thresholds.min_gauges, ">="),
        (thresholds.max_gauges, "<="),
    ):
        for gauge, limit in sorted(limits.items()):
            checks.append(
                RegressionCheck(
                    f"gauges.{gauge}",
                    comparator,
                    limit,
                    None if summary is None else summary.gauges.get(gauge),
                )
            )
    return checks


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a JSON boolean.")
    return value


__all__ = [
    "REALTIME_REGRESSION_SCHEMA_VERSION",
    "RealtimeRegressionCase",
    "RealtimeRegressionManifest",
    "RealtimeRegressionResult",
    "RealtimeRegressionRun",
    "RealtimeRegressionThresholds",
    "RealtimeTraceSample",
    "RealtimeTraceSummary",
    "RegressionCheck",
    "evaluate_realtime_case",
    "evaluate_realtime_manifest",
    "read_realtime_trace",
]
