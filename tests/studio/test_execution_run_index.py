from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from worldfoundry.studio.inference import execution as execution
from worldfoundry.studio.inference.execution import StudioManager


def _write_run(runs_root: Path, run_id: str, artifact_name: str = "preview.mp4") -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir()
    artifact = run_dir / artifact_name
    artifact.write_bytes(b"video")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_id": "model",
                "display_name": "Model",
                "mode": "run",
                "status": "succeeded",
                "output_dir": str(run_dir),
                "preview_video": str(artifact),
                "artifacts": [str(artifact)],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_recent_run_index_is_reused_and_invalidated_by_new_run(tmp_path: Path, monkeypatch) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    runs_root = Path(manager.runs_root)
    _write_run(runs_root, "run-a")
    _write_run(runs_root, "run-b")

    calls = 0
    original = execution._run_record_from_manifest

    def counted(manifest_path: Path, *, run_id: str | None = None):
        nonlocal calls
        calls += 1
        return original(manifest_path, run_id=run_id)

    monkeypatch.setattr(execution, "_run_record_from_manifest", counted)
    assert len(manager.list_recent_runs(limit=100)) == 2
    assert len(manager.list_recent_runs(limit=100)) == 2
    assert calls == 2

    _write_run(runs_root, "run-c")
    root_stat = runs_root.stat()
    changed_ns = max(time.time_ns(), root_stat.st_mtime_ns + 1)
    os.utime(runs_root, ns=(root_stat.st_atime_ns, changed_ns))
    assert {record.run_id for record in manager.list_recent_runs(limit=100)} == {"run-a", "run-b", "run-c"}
    assert calls == 5


def test_load_run_reads_direct_manifest_and_rejects_traversal(tmp_path: Path, monkeypatch) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    runs_root = Path(manager.runs_root)
    artifact = _write_run(runs_root, "run-a")
    monkeypatch.setattr(
        manager,
        "list_recent_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run index scanned")),
    )

    record = manager.load_run("run-a")
    assert record.run_id == "run-a"
    assert record.preview_video == str(artifact)

    for value in ("../run-a", "run-a/../run-a", str((runs_root / "run-a").resolve()), "..\\run-a"):
        with pytest.raises(KeyError, match="Unknown Studio run id"):
            manager.load_run(value)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    (runs_root / "linked-run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(KeyError, match="Unknown Studio run id"):
        manager.load_run("linked-run")


def test_manager_manifest_rewrite_explicitly_invalidates_index(tmp_path: Path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    record = manager.make_message_record(
        SimpleNamespace(model_id="model", display_name="Model"),
        "ready",
    )
    assert manager.list_recent_runs(limit=10)
    assert manager._recent_runs_root_signature is not None

    manager._persist_performance_metadata(record, {"total_ms": 1.25})
    assert manager._recent_runs_root_signature is None
    refreshed = manager.list_recent_runs(limit=10)
    assert refreshed[0].metadata["studio_performance"]["total_ms"] == 1.25


def test_studio_machine_json_writes_use_atomic_replace(tmp_path: Path, monkeypatch) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    calls: list[bool] = []
    original = execution._core_write_json

    def recording_write(path, payload, *, atomic=True):
        calls.append(atomic)
        return original(path, payload, atomic=atomic)

    monkeypatch.setattr(execution, "_core_write_json", recording_write)
    record = manager.make_message_record(
        SimpleNamespace(model_id="model", display_name="Model"),
        "ready",
        extra_metadata={
            "request": {
                "load_kwargs": {"api_key": "load-secret", "max_new_tokens": 16},
                "call_kwargs": {"auth_token": "call-secret"},
            }
        },
    )

    assert calls == [True]
    payload = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
    assert payload["message"] == "ready"
    request = payload["metadata"]["request"]
    assert request["load_kwargs"] == {"api_key": "<redacted>", "max_new_tokens": 16}
    assert request["call_kwargs"] == {"auth_token": "<redacted>"}
    assert "atomic=False" not in inspect.getsource(execution)
