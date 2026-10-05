from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_metrics import (
    METRIC_ORDER,
    compute_iworldbench_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
    CANONICAL_PROMPT_COUNT,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPO_ROOT.parent / "github_repos" / "iWorld-Bench"


def test_iworldbench_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in METRIC_ORDER]
    metrics = compute_iworldbench_metrics(rows=rows)["metrics"]
    assert metrics["image_quality"] == 0.75
    assert metrics["iworldbench_average"] == 0.75


def test_iworldbench_canonical_prompt_count() -> None:
    assert CANONICAL_PROMPT_COUNT == 4900


def test_iworldbench_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("iworld-bench")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/iworldbench/run_iworldbench_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 10


def test_iworldbench_official_run_with_mock_backend_writes_scorecard(tmp_path: Path) -> None:
    if not UPSTREAM_ROOT.is_dir():
        return
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "sample_001.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_IWORLD_BENCH_RUNTIME_BACKEND"] = "mock"
    env["WORLDFOUNDRY_IWORLD_BENCH_ROOT"] = str(UPSTREAM_ROOT)
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/iworldbench/run_iworldbench_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--iworld-root",
            str(UPSTREAM_ROOT),
            "--metric",
            "memory",
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
    assert (output_dir / "reports" / "iworldbench_mock_summary.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "iworldbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 10
