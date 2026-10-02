from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from worldfoundry.studio.serving import workspace as workspace_app
from worldfoundry.studio.inference.execution import (
    RunRecord,
    StudioManager,
    bind_run_preview_image,
    ensure_run_preview_image,
)
from worldfoundry.studio.serving.jobs import StudioJob, StudioJobStore
from worldfoundry.studio.serving.workspace import create_app


def _write_png(path: Path, color: tuple[int, int, int] = (220, 40, 40)) -> Path:
    image = Image.new("RGB", (8, 6), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _record(
    tmp_path: Path,
    *,
    run_id: str = "run-video",
    preview_video: str | None = "out.mp4",
    preview_image: str | None = None,
    gallery: list[str] | None = None,
) -> RunRecord:
    output_dir = tmp_path / run_id
    output_dir.mkdir(exist_ok=True)
    video_path = None
    if preview_video:
        video_file = output_dir / preview_video
        video_file.write_bytes(b"not-a-real-video")
        video_path = str(video_file)
    return RunRecord(
        run_id=run_id,
        model_id="wan",
        display_name="Wan",
        mode="run",
        status="succeeded",
        output_dir=str(output_dir),
        manifest_path=str(output_dir / "manifest.json"),
        preview_video=video_path,
        preview_image=preview_image,
        gallery=list(gallery or []),
    )


def test_ensure_run_preview_image_prefers_existing_first_frame(tmp_path: Path) -> None:
    record = _record(tmp_path)
    first_frame = _write_png(Path(record.output_dir) / "first_frame.png")
    assert ensure_run_preview_image(record) == str(first_frame)


def test_ensure_run_preview_image_skips_input_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record(tmp_path)
    input_image = _write_png(Path(record.output_dir) / "inputs" / "ref.png", (10, 10, 10))
    record.preview_image = str(input_image)
    extracted = Path(record.output_dir) / "first_frame.png"

    def fake_extract(video_path, output_dir, *, frame_position="first", output_name="preview.png"):
        assert video_path == record.preview_video
        assert frame_position == "first"
        assert output_name == "first_frame.png"
        return str(_write_png(Path(output_dir) / output_name, (30, 180, 80)))

    monkeypatch.setattr(
        "worldfoundry.studio.inference.execution.maybe_extract_video_preview_image",
        fake_extract,
    )
    assert bind_run_preview_image(record) == str(extracted)
    assert record.preview_image == str(extracted)
    assert Image.open(extracted).getpixel((0, 0)) == (30, 180, 80)


def test_gallery_row_advertises_poster_for_video_without_preview_image(tmp_path: Path) -> None:
    record = _record(tmp_path, run_id="run-no-still")
    row = workspace_app._gallery_row_from_run(record)
    assert row["video_url"] == "/api/runs/run-no-still/video"
    assert row["image_url"] == "/api/runs/run-no-still/image"
    assert row["poster_url"] == "/api/runs/run-no-still/image"


def test_gallery_and_image_api_serve_extracted_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    run_dir = Path(manager.runs_root) / "run-poster"
    run_dir.mkdir(parents=True)
    video = run_dir / "out.mp4"
    video.write_bytes(b"not-a-real-video")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-poster",
                "model_id": "wan",
                "display_name": "Wan",
                "mode": "run",
                "status": "succeeded",
                "output_dir": str(run_dir),
                "preview_video": str(video),
                "artifacts": [str(video)],
            }
        ),
        encoding="utf-8",
    )

    def fake_extract(video_path, output_dir, *, frame_position="first", output_name="preview.png"):
        assert Path(video_path) == video
        return str(_write_png(Path(output_dir) / output_name, (255, 200, 0)))

    monkeypatch.setattr(
        "worldfoundry.studio.inference.execution.maybe_extract_video_preview_image",
        fake_extract,
    )
    monkeypatch.setattr(workspace_app, "MANAGER", manager)
    monkeypatch.setattr(workspace_app, "JOBS", StudioJobStore())
    workspace_app._invalidate_registered_artifact_cache()

    client = TestClient(create_app())
    listed = client.get("/api/gallery").json()
    assert listed[0]["run_id"] == "run-poster"
    assert listed[0]["image_url"] == "/api/runs/run-poster/image"
    assert not (run_dir / "first_frame.png").exists()

    response = client.get("/api/runs/run-poster/image")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    served = Image.open(BytesIO(response.content))
    assert served.size == (8, 6)
    assert served.getpixel((0, 0)) == (255, 200, 0)
    assert (run_dir / "first_frame.png").is_file()


def test_job_image_api_extracts_first_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record(tmp_path, run_id="job-run")
    job = StudioJob(
        job_id="studio-00021",
        title="Wan demo",
        model_id="wan",
        display_name="Wan",
        action="inference",
        status="completed",
    )
    job.result = record
    store = StudioJobStore()
    with store._lock:
        store._jobs[job.job_id] = job

    def fake_extract(video_path, output_dir, *, frame_position="first", output_name="preview.png"):
        return str(_write_png(Path(output_dir) / output_name, (12, 34, 56)))

    monkeypatch.setattr(
        "worldfoundry.studio.inference.execution.maybe_extract_video_preview_image",
        fake_extract,
    )
    monkeypatch.setattr(workspace_app, "JOBS", store)
    monkeypatch.setattr(workspace_app, "MANAGER", StudioManager(workspace_root=str(tmp_path)))

    client = TestClient(create_app())
    row = workspace_app._gallery_row_from_job(job)
    assert row is not None
    assert row["poster_url"] == "/api/jobs/studio-00021/image"
    response = client.get("/api/jobs/studio-00021/image")
    assert response.status_code == 200
    assert Image.open(BytesIO(response.content)).getpixel((0, 0)) == (12, 34, 56)
