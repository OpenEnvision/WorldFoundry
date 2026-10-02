from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from worldfoundry.studio.serving import workspace as workspace_app
from worldfoundry.studio.inference.execution import RunRecord, StudioManager
from worldfoundry.studio.serving.jobs import StudioJob, StudioJobStore
from worldfoundry.studio.serving.workspace import create_app


def _write_run(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir()
    artifact = run_dir / "preview.png"
    artifact.write_bytes(b"image")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_id": "sana",
                "display_name": "Sana",
                "mode": "run",
                "status": "succeeded",
                "output_dir": str(run_dir),
                "preview_image": str(artifact),
                "artifacts": [str(artifact)],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _completed_job(job_id: str, run_id: str = "") -> StudioJob:
    job = StudioJob(
        job_id=job_id,
        title="Sana demo",
        model_id="sana",
        display_name="Sana",
        action="inference",
        status="completed",
    )
    if run_id:
        job.result = RunRecord(
            run_id=run_id,
            model_id="sana",
            display_name="Sana",
            mode="run",
            status="succeeded",
            output_dir="",
            manifest_path="",
            preview_image="/tmp/preview.png",
        )
    return job


def test_job_store_deletes_finished_jobs_only() -> None:
    store = StudioJobStore()
    finished = _completed_job("studio-00001")
    running = StudioJob(
        job_id="studio-00002",
        title="running",
        model_id="sana",
        display_name="Sana",
        action="inference",
        status="running",
    )
    with store._lock:
        store._jobs[finished.job_id] = finished
        store._jobs[running.job_id] = running

    assert store.delete(finished.job_id) is True
    assert store.get(finished.job_id) is None
    with pytest.raises(ValueError, match="cannot delete a running job"):
        store.delete(running.job_id)
    assert store.get(running.job_id) is running
    assert store.delete("missing") is False


def test_manager_delete_run_removes_directory_and_index(tmp_path: Path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    run_dir = _write_run(Path(manager.runs_root), "run-a")
    assert manager.list_recent_runs(limit=10)
    assert manager._recent_runs_root_signature is not None

    manager.delete_run("run-a")

    assert not run_dir.exists()
    assert manager.list_recent_runs(limit=10) == []
    for value in ("../run-a", "run-a/../run-a", "..\\run-a"):
        with pytest.raises(KeyError, match="Unknown Studio run id"):
            manager.delete_run(value)


def test_gallery_api_deletes_persisted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    _write_run(Path(manager.runs_root), "run-a")
    store = StudioJobStore()
    monkeypatch.setattr(workspace_app, "MANAGER", manager)
    monkeypatch.setattr(workspace_app, "JOBS", store)
    workspace_app._invalidate_registered_artifact_cache()

    client = TestClient(create_app())
    listed = client.get("/api/gallery").json()
    assert [row["run_id"] for row in listed] == ["run-a"]

    response = client.delete("/api/gallery", params={"run_id": "run-a"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "job_id": "", "run_id": "run-a"}
    assert client.get("/api/gallery").json() == []
    assert not (Path(manager.runs_root) / "run-a").exists()


def test_gallery_api_deletes_job_and_linked_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    _write_run(Path(manager.runs_root), "run-b")
    store = StudioJobStore()
    job = _completed_job("studio-00009", "run-b")
    with store._lock:
        store._jobs[job.job_id] = job
    monkeypatch.setattr(workspace_app, "MANAGER", manager)
    monkeypatch.setattr(workspace_app, "JOBS", store)
    workspace_app._invalidate_registered_artifact_cache()

    client = TestClient(create_app())
    response = client.delete("/api/gallery", params={"job_id": "studio-00009", "run_id": "run-b"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "job_id": "studio-00009", "run_id": "run-b"}
    assert store.get("studio-00009") is None
    assert client.get("/api/gallery").json() == []


def test_gallery_api_rejects_empty_and_running_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StudioJobStore()
    running = StudioJob(
        job_id="studio-00003",
        title="running",
        model_id="sana",
        display_name="Sana",
        action="inference",
        status="running",
    )
    with store._lock:
        store._jobs[running.job_id] = running
    monkeypatch.setattr(workspace_app, "JOBS", store)
    monkeypatch.setattr(workspace_app, "MANAGER", StudioManager(workspace_root=str(tmp_path)))

    client = TestClient(create_app())
    assert client.delete("/api/gallery").status_code == 400
    assert client.delete("/api/gallery", params={"job_id": "studio-00003"}).status_code == 409
    assert client.delete("/api/gallery", params={"run_id": "missing"}).status_code == 404
