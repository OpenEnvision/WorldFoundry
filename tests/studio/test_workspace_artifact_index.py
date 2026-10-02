from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from worldfoundry.studio.serving import workspace as workspace_app
from worldfoundry.studio.inference.execution import RunRecord


class _FakeJobs:
    def list(self):
        return []


class _FakeManager:
    def __init__(self, records):
        self.records = records

    def list_recent_runs(self, limit=100):
        return list(self.records[:limit])


def _record(tmp_path: Path, artifacts: list[str]) -> RunRecord:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return RunRecord(
        run_id="run-a",
        model_id="model",
        display_name="Model",
        mode="run",
        status="succeeded",
        output_dir=str(run_dir),
        manifest_path=str(manifest),
        artifacts=artifacts,
    )


def test_registered_artifact_index_reuses_and_invalidates_exact_paths(tmp_path: Path, monkeypatch) -> None:
    artifact_a = tmp_path / "a.mp4"
    artifact_b = tmp_path / "b.mp4"
    artifact_a.write_bytes(b"a")
    artifact_b.write_bytes(b"b")
    record = _record(tmp_path, [str(artifact_a)])
    monkeypatch.setattr(workspace_app, "JOBS", _FakeJobs())
    monkeypatch.setattr(workspace_app, "MANAGER", _FakeManager([record]))

    first = workspace_app._registered_artifact_paths()
    cached = workspace_app._REGISTERED_ARTIFACT_CACHE_PATHS
    second = workspace_app._registered_artifact_paths()
    assert first == second
    assert workspace_app._REGISTERED_ARTIFACT_CACHE_PATHS is cached
    assert artifact_a.resolve() in first
    assert artifact_b.resolve() not in first

    record.artifacts.append(str(artifact_b))
    updated = workspace_app._registered_artifact_paths()
    assert workspace_app._REGISTERED_ARTIFACT_CACHE_PATHS is not cached
    assert artifact_b.resolve() in updated


def test_fastapi_shutdown_uses_process_global_workspace_teardown() -> None:
    app = workspace_app.create_app()
    assert workspace_app._shutdown_workspace in app.router.on_shutdown


def test_workspace_teardown_is_idempotent(monkeypatch) -> None:
    calls: list[str] = []

    class _ShutdownJobs:
        def shutdown(self):
            calls.append("jobs")

    monkeypatch.setattr(workspace_app, "JOBS", _ShutdownJobs())
    monkeypatch.setattr(workspace_app, "_stop_all_visualizers", lambda: calls.append("visualizers"))
    monkeypatch.setattr(workspace_app, "_WORKSPACE_SHUTDOWN_COMPLETE", False)

    workspace_app._shutdown_workspace()
    workspace_app._shutdown_workspace()
    assert calls == ["jobs", "visualizers"]
