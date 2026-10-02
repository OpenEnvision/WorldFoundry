from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_metrics import compute_phygenbench_metrics
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_phygenbench_generation_requests,
    unique_generation_records,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phygenbench_prompt_materialization_uses_one_based_ids() -> None:
    requests = materialize_phygenbench_generation_requests(limit=3)
    assert [request.sample_id for request in requests] == ["1", "2", "3"]
    assert requests[0].inputs["official_video_name"] == "output_video_1.mp4"


def test_phygenbench_canonical_prompt_count_from_upstream_repo() -> None:
    from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import load_prompt_records

    records = unique_generation_records(load_prompt_records())
    assert len(records) == CANONICAL_PROMPT_COUNT


def test_phygenbench_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in (
        "physical_commonsense",
        "physical_law_adherence",
        "semantic_adherence",
        "phygenbench_average",
    )]
    metrics = compute_phygenbench_metrics(rows=rows)["metrics"]
    assert metrics["physical_commonsense"] == 0.75
    assert metrics["phygenbench_average"] == 0.75


def test_phygenbench_metrics_from_per_sample_json() -> None:
    rows = [
        {
            "caption": "A ball falls.",
            "single": 3,
            "multi_gpt": 2,
            "video_gpt": 1,
            "semantic_score": 4,
            "worldfoundry_average": 2,
        }
    ]
    metrics = compute_phygenbench_metrics(rows=rows)["metrics"]
    assert metrics["physical_law_adherence"] == (3 + 2) / 2 / 3
    assert metrics["physical_commonsense"] == (3 + 2 + 1) / 3 / 3
    assert metrics["semantic_adherence"] == 4 / 5
    assert metrics["phygenbench_average"] == 2 / 3


def test_phygenbench_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("phygenbench")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phygenbench/run_phygenbench_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 4


def test_phygenbench_official_run_with_mock_backend_writes_scorecard(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "output_video_1.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_PHYGENBENCH_JUDGE_BACKEND"] = "mock"
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phygenbench/run_phygenbench_official_runner.py",
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
    assert (output_dir / "phygenbench_results.json").is_file()
    assert scorecard["evaluation"]["kind"] == "phygenbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 4
