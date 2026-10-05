"""Regression test for SA-9 (B023) in ``StudioManager.list_recent_runs``.

The ``persisted_preview`` closure captured the loop variables ``payload`` and
``recovered_previews``.  It is consumed synchronously today, but if RunRecord
construction ever became deferred every record would silently read the last
iteration's manifest.  The closure now binds both values as keyword defaults;
this test pins the per-run preview resolution behaviour.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

from worldfoundry.studio.inference.execution import StudioManager


def _write_run(runs_root: Path, name: str, marker: str) -> Path:
    run_dir = runs_root / name
    run_dir.mkdir()
    video = run_dir / f"video_{marker}.mp4"
    video.write_bytes(b"\x00")
    manifest = {
        "run_id": name,
        "model_id": f"model_{marker}",
        "display_name": f"Model {marker}",
        "mode": "video",
        "status": "succeeded",
        "output_dir": str(run_dir),
        "preview_video": str(video),
        "artifacts": [str(video)],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return video


def test_list_recent_runs_previews_are_bound_per_run(tmp_path: Path) -> None:
    video_a = _write_run(tmp_path, "run_a", "a")
    video_b = _write_run(tmp_path, "run_b", "b")

    manager = types.SimpleNamespace(runs_root=str(tmp_path))
    records = StudioManager.list_recent_runs(manager)

    by_id = {record.run_id: record for record in records}
    assert set(by_id) == {"run_a", "run_b"}
    assert by_id["run_a"].preview_video == str(video_a)
    assert by_id["run_b"].preview_video == str(video_b)
