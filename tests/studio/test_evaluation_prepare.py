
import pytest

# This test module imports worldfoundry code that requires the optional
# "fastapi" dependency at import time; skip when it is unavailable.
pytest.importorskip("fastapi")
from pathlib import Path

from fastapi.testclient import TestClient

from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest
from worldfoundry.studio.serving import workspace as workspace_app
from worldfoundry.studio.serving.jobs import StudioJob
from worldfoundry.studio.serving.workspace import WORKSPACE_HTML, create_app


def test_workspace_prepares_custom_artifact_scoring(tmp_path: Path) -> None:
    artifacts = tmp_path / "videos"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")
    client = TestClient(create_app())

    response = client.post(
        "/api/evaluation/prepare",
        json={
            "job_type": "evaluation",
            "eval_mode": "score-artifacts",
            "benchmark_id": "vbench",
            "dataset_root": str(artifacts),
            "output_dir": str(tmp_path / "scores"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["classification"] == "custom_dataset_metric_evaluation"
    assert payload["leaderboard_candidate"] is False
    assert 'value="score-artifacts"' in WORKSPACE_HTML
    assert 'value="model-benchmark"' in WORKSPACE_HTML
    assert 'value="generate-and-score"' in WORKSPACE_HTML


def test_workspace_submits_prepared_artifact_scoring_job(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "videos"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")

    def fake_submit_run(**kwargs):
        return StudioJob(
            job_id="studio-test",
            title=kwargs["title"],
            model_id=kwargs["model_id"],
            display_name=kwargs["display_name"],
            action=kwargs["action"],
            job_type=kwargs["job_type"],
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(workspace_app.JOBS, "submit_run", fake_submit_run)
    client = TestClient(create_app())
    response = client.post(
        "/api/jobs",
        json={
            "job_type": "evaluation",
            "eval_mode": "score-artifacts",
            "benchmark_id": "vbench",
            "dataset_root": str(artifacts),
            "output_dir": str(tmp_path / "scores"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "studio-test"
    assert payload["metadata"]["eval_mode"] == "score-artifacts"


def test_workspace_prepares_generate_and_score_dataset(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"sample_id":"one","prompt":"A ball falls."}\n', encoding="utf-8")
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(
        build_dataset_manifest(samples_path=samples, dataset_id="custom-prompts"),
        manifest_path,
    )
    client = TestClient(create_app())

    response = client.post(
        "/api/evaluation/prepare",
        json={
            "job_type": "evaluation",
            "eval_mode": "generate-and-score",
            "model_id": "hailuo-2p3",
            "dataset_manifest": str(manifest_path),
            "metrics": ["artifact_count"],
            "output_dir": str(tmp_path / "output"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["intent_kind"] == "generate_and_score"
    assert payload["execution"]["kind"] == "evaluate"
    assert payload["execution"]["request"]["dataset_id"] == "custom-prompts"
