from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.runtime.realtime_regression import (
    RealtimeRegressionManifest,
    evaluate_realtime_manifest,
    read_realtime_trace,
)


def _event(
    *,
    session: str,
    chunk: int,
    warmup: bool,
    model_ms: float,
    dropped: int,
    queue_depth: int = 1,
) -> dict[str, object]:
    return {
        "event": "realtime.chunk_timing",
        "model_id": "lingbot-world",
        "fields": {
            "session_id": session,
            "chunk_index": chunk,
            "transport": "webrtc",
            "output_frames": 3,
            "queue_depth": queue_depth,
            "dropped_frames": dropped,
            "warmup": warmup,
            "server_chunk_ms": 100.0,
            "stage_ms": {
                "generation_ms": 80.0,
                "model_ms": model_ms,
            },
            "gauges": {"cache_frames": 8.0},
        },
    }


def _write_trace(path: Path) -> None:
    rows = [
        {"event": "unrelated", "fields": {}},
        _event(session="old", chunk=1, warmup=False, model_ms=200.0, dropped=0),
        _event(session="latest", chunk=1, warmup=True, model_ms=100.0, dropped=2),
        _event(session="latest", chunk=2, warmup=False, model_ms=10.0, dropped=2),
        _event(
            session="latest",
            chunk=3,
            warmup=False,
            model_ms=20.0,
            dropped=3,
            queue_depth=4,
        ),
        _event(session="latest", chunk=4, warmup=False, model_ms=30.0, dropped=3),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_realtime_regression_selects_latest_session_and_excludes_warmup(tmp_path: Path) -> None:
    trace = tmp_path / "timing.jsonl"
    _write_trace(trace)
    manifest = RealtimeRegressionManifest.from_dict(
        {
            "cases": [
                {
                    "name": "lingbot-webrtc",
                    "selectors": {
                        "model_id": "lingbot-world",
                        "transport": "webrtc",
                    },
                    "thresholds": {
                        "min_chunks": 3,
                        "min_output_frames": 9,
                        "min_throughput_fps": 30,
                        "max_dropped_frames": 1,
                        "max_queue_depth": 4,
                        "max_stage_p50_ms": {"model_ms": 20},
                        "max_stage_p90_ms": {"model_ms": 28},
                        "max_stage_max_ms": {"model_ms": 30},
                        "min_gauges": {"cache_frames": 8},
                    },
                }
            ]
        }
    )

    run = evaluate_realtime_manifest(manifest, read_realtime_trace(trace))

    assert run.passed is True
    result = run.results[0]
    assert result.summary is not None
    assert result.summary.session_id == "latest"
    assert result.summary.chunk_count == 3
    assert result.summary.frame_count == 9
    assert result.summary.throughput_fps == pytest.approx(30.0)
    assert result.summary.dropped_frames == 1
    assert result.summary.stages["model_ms"].p90_ms == pytest.approx(28.0)
    assert result.failures == ()


def test_realtime_regression_reports_missing_and_over_limit_metrics(tmp_path: Path) -> None:
    trace = tmp_path / "timing.jsonl"
    _write_trace(trace)
    manifest = RealtimeRegressionManifest.from_dict(
        {
            "cases": [
                {
                    "name": "strict",
                    "session_id": "latest",
                    "thresholds": {
                        "max_stage_p90_ms": {
                            "model_ms": 27,
                            "decode_ms": 10,
                        }
                    },
                }
            ]
        }
    )

    [result] = evaluate_realtime_manifest(manifest, read_realtime_trace(trace)).results

    assert result.passed is False
    failures = {check.metric: check for check in result.failures}
    assert failures["stages.model_ms.p90_ms"].actual == pytest.approx(28.0)
    assert failures["stages.decode_ms.p90_ms"].actual is None


def test_realtime_regression_requires_stage_metric_on_every_measured_chunk(tmp_path: Path) -> None:
    trace = tmp_path / "partial.jsonl"
    rows = [
        _event(session="partial", chunk=1, warmup=False, model_ms=10.0, dropped=0),
        _event(session="partial", chunk=2, warmup=False, model_ms=20.0, dropped=0),
    ]
    del rows[1]["fields"]["stage_ms"]["model_ms"]  # type: ignore[index]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = RealtimeRegressionManifest.from_dict(
        {
            "cases": [
                {
                    "name": "complete-stage",
                    "thresholds": {"max_stage_p90_ms": {"model_ms": 100}},
                }
            ]
        }
    )

    [result] = evaluate_realtime_manifest(manifest, read_realtime_trace(trace)).results

    assert result.passed is False
    assert result.failures[0].metric == "stages.model_ms.p90_ms"
    assert result.failures[0].actual is None


def test_realtime_regression_roundtrips_manifest_and_exports_performance_manifest(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "timing.jsonl"
    _write_trace(trace)
    manifest = RealtimeRegressionManifest.from_dict(
        {
            "cases": [
                {
                    "name": "export",
                    "model_id": "lingbot-world",
                    "exclude_warmup_chunks": 1,
                    "thresholds": {"min_chunks": 2},
                }
            ]
        }
    )
    path = manifest.write_json(tmp_path / "manifest.json")

    restored = RealtimeRegressionManifest.read_json(path)
    [result] = evaluate_realtime_manifest(restored, read_realtime_trace(trace)).results
    performance = result.to_performance_manifest()

    assert restored.to_dict() == manifest.to_dict()
    assert result.summary is not None and result.summary.chunk_count == 2
    assert performance.model["model_id"] == "lingbot-world"
    assert performance.metrics.throughput["frames_per_second"] == pytest.approx(30.0)
    assert performance.extensions["realtime_regression"]["passed"] is True  # type: ignore[index]


def test_realtime_trace_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n{bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_realtime_trace(path)
