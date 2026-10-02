from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_metrics import (
    METRIC_ORDER,
    compute_world_in_world_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
    CANONICAL_AEQA_PROMPT_COUNT,
    materialize_world_in_world_generation_requests,
    unique_prompt_records,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_world_in_world_prompt_materialization_uses_question_ids() -> None:
    requests = materialize_world_in_world_generation_requests(limit=2, task="AEQA")
    assert len(requests) == 2
    assert requests[0].inputs["task"] == "AEQA"
    assert requests[0].inputs["official_video_name"].endswith(".mp4")


def test_world_in_world_canonical_prompt_count_from_bundled_assets() -> None:
    from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
        load_prompt_records,
    )

    records = unique_prompt_records(load_prompt_records(task="AEQA"))
    assert len(records) == CANONICAL_AEQA_PROMPT_COUNT
    assert CANONICAL_AEQA_PROMPT_COUNT == 184


def test_world_in_world_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in METRIC_ORDER]
    metrics = compute_world_in_world_metrics(rows=rows)["metrics"]
    assert metrics["active_recognition_success_rate"] == 0.75
    assert metrics["world_in_world_average"] == 0.75


def test_world_in_world_metrics_from_evaluator_summary_json() -> None:
    payload = {
        "task": "IGNav",
        "summary": {
            "total_size": 10,
            "sr": 0.62,
            "spl": 0.48,
            "mean_traj_len": 11.2,
        },
        "video_metrics": {
            "ssim": 0.81,
            "psnr": 24.5,
            "lpips": 0.12,
        },
    }
    metrics = compute_world_in_world_metrics(rows=[payload], task="IGNav")["metrics"]
    assert metrics["image_goal_navigation_success_rate"] == 0.62
    assert metrics["image_goal_navigation_spl"] == 0.48
    assert metrics["ssim"] == 0.81


def test_world_in_world_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("world-in-world")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/world_in_world/run_world_in_world_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 8


def test_world_in_world_official_run_with_artifact_backend_writes_scorecard(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    first_request = materialize_world_in_world_generation_requests(limit=1, task="AEQA")[0]
    (generated_dir / first_request.inputs["official_video_name"]).write_bytes(b"fake")
    (generated_dir / "metrics.json").write_text(
        json.dumps(
            {
                "active_recognition_success_rate": 0.61,
                "image_goal_navigation_success_rate": 0.62,
                "image_goal_navigation_spl": 0.52,
                "active_embodied_qa_score": 0.71,
                "active_embodied_qa_spl": 0.63,
                "robotic_manipulation_success_rate": 0.58,
                "interaction_trace_consistency": 0.84,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_WORLD_IN_WORLD_RUNTIME_BACKEND"] = "artifact"
    env["WORLDFOUNDRY_WORLD_IN_WORLD_TASK"] = "AEQA"
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/world_in_world/run_world_in_world_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--task",
            "AEQA",
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
    assert (output_dir / "world_in_world_metrics.json").is_file()
    assert scorecard["evaluation"]["kind"] == "world_in_world_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 8
