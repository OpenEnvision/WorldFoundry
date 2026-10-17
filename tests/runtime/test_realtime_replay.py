# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "realtime_replay", Path(__file__).parents[1] / "manual/realtime_replay.py"
)
assert _SPEC is not None and _SPEC.loader is not None
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)
pytestmark = pytest.mark.unit


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    profile = {
        "model_id": "matrix-game-2",
        "warmup_steps": 1,
        "steps": 2,
        "repeats": 1,
        "networks": [{"name": "clean"}],
        "thresholds": {"min_chunks": 2},
        "delivery_thresholds": {"max_e2e_p90_ms": 200, "max_reset_ms": 1000},
    }
    sent: list[dict[str, Any]] = [
        {"type": "action", "action": {"event": "step"}, "control_id": key} for key in ("warm", "a", "b")
    ]
    sent.extend(
        {"type": "frame_presented", "control_id": key, "control_to_display_ms": value}
        for key, value in (("warm", 9999), ("a", 100), ("b", 200))
    )
    chunks = [{"type": "chunk_done", "chunk_index": index, "frames": 12} for index in range(1, 4)]
    browser = {
        "browser_version": "test",
        "sessions": [
            {
                "name": "clean-1",
                "network": "clean",
                "repeat": 0,
                "spec": {"fps": 12},
                "errors": [],
                "link": {"errors": []},
                "reset_clean": True,
                "reset_ms": 100,
                "observations": {
                    "sent": sent,
                    "received": chunks,
                    "frames": [],
                    "callbacks": [100, 200, 900],
                    "measuredStart": 100,
                    "measuredEnd": 1000,
                },
            }
        ],
    }
    events = []
    for index, key in enumerate(("warm", "a", "b"), 1):
        events.append(
            {
                "event": "realtime.chunk_timing",
                "model_id": "matrix-game-2",
                "fields": {
                    "session_id": "session",
                    "transport": "webrtc",
                    "chunk_index": index,
                    "warmup": index == 1,
                    "output_frames": 12,
                    "queue_depth": 1,
                    "dropped_frames": 0,
                    "server_chunk_ms": 100,
                    "stage_ms": {"generation_ms": 100},
                },
            }
        )
        events.append({"event": "realtime.frame_presented", "fields": {"session_id": "session", "control_id": key}})
    path = tmp_path / "timing.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in events))
    return profile, browser, path


@pytest.mark.parametrize(
    "failure",
    [None, "negative-rate", "loss-range", "duplicate-name", "late-outage", "unknown-threshold", "negative-step"],
)
def test_replay_validates_profile_before_loading(failure: str | None) -> None:
    profile = json.loads((Path(__file__).parents[1] / "manual/realtime_replay.example.json").read_text())
    if failure == "negative-rate":
        profile["networks"][0]["downlink"]["rate_bps"] = -1
    elif failure == "loss-range":
        profile["networks"][0]["uplink"]["loss"] = 2
    elif failure == "duplicate-name":
        profile["networks"].append(profile["networks"][0])
    elif failure == "late-outage":
        profile["networks"][-1]["outage"]["after_steps"] = profile["steps"]
    elif failure == "unknown-threshold":
        profile["delivery_thresholds"]["max_e2e_p900_ms"] = 1000
    elif failure == "negative-step":
        profile["steps"] = -1
    if failure is None:
        replay.validate_profile(profile)
    else:
        with pytest.raises(ValueError):
            replay.validate_profile(profile)


def test_replay_reuses_chunk_gates_and_excludes_warmup_controls(tmp_path: Path) -> None:
    profile, browser, trace = _fixture(tmp_path)
    report = replay.build_report(profile, browser, trace, [])
    assert report["passed"]
    result = report["results"][0]
    assert result["server"]["summary"]["chunk_count"] == 2
    assert result["delivery"]["control_to_display_ms"]["p50_ms"] == 150
    assert result["delivery"]["presentation_stalls"] == 1
    assert result["delivery"]["generated_frames"] == 24
    assert "clean-1" in replay.markdown_report(report)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-display",
        "missing-trace",
        "wrong-session",
        "wrong-model",
        "reset",
        "browser-error",
        "lost-run",
        "duplicate-run",
        "latency",
    ],
)
def test_replay_does_not_promote_incomplete_or_failing_evidence(tmp_path: Path, failure: str) -> None:
    profile, browser, trace = _fixture(tmp_path)
    session = browser["sessions"][0]
    if failure == "missing-display":
        session["observations"]["sent"].pop()
    elif failure == "missing-trace":
        trace.write_text("\n".join(line for line in trace.read_text().splitlines() if "frame_presented" not in line))
    elif failure == "wrong-session":
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        events[-1]["fields"]["session_id"] = "foreign"
        trace.write_text("\n".join(json.dumps(item) for item in events))
    elif failure == "wrong-model":
        profile["model_id"] = "different-model"
    elif failure == "reset":
        session["reset_clean"] = False
    elif failure == "browser-error":
        session["errors"].append("WebSocket fallback")
    elif failure == "lost-run":
        browser["sessions"] = []
    elif failure == "duplicate-run":
        browser["sessions"].append(session)
    elif failure == "latency":
        profile["delivery_thresholds"]["max_e2e_p90_ms"] = 100
    report = replay.build_report(profile, browser, trace, [])
    assert report["passed"] is False


def test_process_cpu_is_per_core_and_samples_match_measured_window(tmp_path: Path) -> None:
    profile, browser, trace = _fixture(tmp_path)
    session = browser["sessions"][0]
    session.update(measured_start_wall_ms=1000, measured_end_wall_ms=3000)
    samples = [
        {"wall_ms": 0, "cpu_s": 0, "rss_bytes": 900},
        {"wall_ms": 1000, "cpu_s": 1, "rss_bytes": 50},
        {"wall_ms": 3000, "cpu_s": 5, "rss_bytes": 80},
    ]
    result = replay.build_report(profile, browser, trace, samples)["results"][0]
    assert result["resources"]["cpu_percent_one_core"] == 200
    assert result["resources"]["peak_rss_bytes"] == 80
    assert result["resources"]["peak_gpu_process_bytes"] is None
