from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_metrics import compute_mirabench_metrics
from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_mirabench_generation_requests,
    unique_prompt_records,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mirabench_prompt_materialization_uses_video_idx_ids() -> None:
    requests = materialize_mirabench_generation_requests(limit=2)
    assert [request.sample_id for request in requests] == ["1", "2"]
    assert requests[0].inputs["official_video_name"] == "1.mp4"


def test_mirabench_example_prompt_count_from_upstream_repo() -> None:
    from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_prompts import load_prompt_records

    records = unique_prompt_records(load_prompt_records())
    assert len(records) == 2
    assert CANONICAL_PROMPT_COUNT == 150


def test_mirabench_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in (
        "dynamic_degree",
        "tracking_strength",
        "dino_temporal_consistency",
        "clip_temporal_consistency",
        "temporal_motion_smoothness",
        "mean_absolute_error",
        "root_mean_square_error",
        "aesthetic_quality",
        "imaging_quality",
        "camera_alignment",
        "main_object_alignment",
        "background_alignment",
        "style_alignment",
        "overall_alignment",
        "fvd",
        "fid",
        "kid",
        "mirabench_average",
    )]
    metrics = compute_mirabench_metrics(rows=rows)["metrics"]
    assert metrics["dynamic_degree"] == 0.75
    assert metrics["mirabench_average"] == 0.75


def test_mirabench_metrics_from_upstream_average_json() -> None:
    payload = {
        "temporal_dino_consistency": 0.8,
        "temporal_clip_consistency": 0.7,
        "dynamic_degree": 0.6,
        "3D_consistency_mean_err": 0.1,
        "overall_consistency": 0.9,
    }
    metrics = compute_mirabench_metrics(rows=[payload])["metrics"]
    assert metrics["dino_temporal_consistency"] == 0.8
    assert metrics["clip_temporal_consistency"] == 0.7
    assert metrics["mean_absolute_error"] == 0.1
    assert metrics["overall_alignment"] == 0.9


def test_mirabench_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("mirabench")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/mirabench/run_mirabench_official_runner.py",
            "--official-results-path",
            str(sample_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 18


def test_mirabench_official_run_with_mock_backend_writes_scorecard(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "1.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_MIRABENCH_SCORER_BACKEND"] = "mock"
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/mirabench/run_mirabench_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert (output_dir / "mirabench_average_score.json").is_file()
    assert scorecard["evaluation"]["kind"] == "mirabench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 18
