from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_metrics import (
    METRIC_ORDER,
    compute_ipv_bench_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_ipv_bench_generation_requests,
    unique_prompt_records,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ipv_bench_prompt_materialization_uses_prompt_ids() -> None:
    requests = materialize_ipv_bench_generation_requests(limit=2)
    assert len(requests) == 2
    assert requests[0].inputs["official_video_name"].endswith(".mp4")


def test_ipv_bench_canonical_prompt_count_from_bundled_assets() -> None:
    from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_prompts import load_prompt_records

    records = unique_prompt_records(load_prompt_records())
    assert len(records) == 260
    assert CANONICAL_PROMPT_COUNT == 260


def test_ipv_bench_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in METRIC_ORDER]
    metrics = compute_ipv_bench_metrics(rows=rows)["metrics"]
    assert metrics["visual_quality"] == 0.75
    assert metrics["ipv_bench_average"] == 0.75


def test_ipv_bench_metrics_from_generation_annotations() -> None:
    rows = [
        {"prompt_id": "0", "visual_quality": 4.5, "prompt_following": 4.2},
        {"prompt_id": "1", "visual_quality": 3.0, "prompt_following": 4.8},
    ]
    metrics = compute_ipv_bench_metrics(rows=rows)["metrics"]
    assert metrics["impossible_video_score"] == 0.5


def test_ipv_bench_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("ipv-bench")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/ipv_bench/run_ipv_bench_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 7


def test_ipv_bench_official_run_with_artifact_results_writes_scorecard(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("ipv-bench")
    assert sample_path is not None
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    first_request = materialize_ipv_bench_generation_requests(limit=1)[0]
    (generated_dir / first_request.inputs["official_video_name"]).write_bytes(b"fake")
    (generated_dir / "ipv_results.csv").write_bytes(sample_path.read_bytes())
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/ipv_bench/run_ipv_bench_official_runner.py",
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
    assert (output_dir / "ipv_bench_results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "ipv_bench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 7
