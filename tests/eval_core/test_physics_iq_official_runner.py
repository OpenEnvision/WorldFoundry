from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_metrics import (
    metric_values_from_scores,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
    materialize_physics_iq_generation_requests,
    unique_generation_records,
    load_description_rows,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.protocols import ORIGINAL


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_physics_iq_prompt_materialization_uses_generated_video_names(tmp_path: Path) -> None:
    descriptions = tmp_path / "descriptions.csv"
    descriptions.write_text(
        "scenario,description,category,generated_video_name\n"
        "0001_perspective-left_take-1_trimmed-ball.mp4,Left view,Solid Mechanics,0001_perspective-left_trimmed-ball.mp4\n"
        "0002_perspective-center_take-1_trimmed-ball.mp4,Center view,Solid Mechanics,0002_perspective-center_trimmed-ball.mp4\n"
        "0001_perspective-left_take-2_trimmed-ball.mp4,Take two,Solid Mechanics,0001_perspective-left_trimmed-ball-t2.mp4\n",
        encoding="utf-8",
    )
    requests = materialize_physics_iq_generation_requests(descriptions_path=descriptions)
    assert [request.sample_id for request in requests] == [
        "0001_perspective-left_trimmed-ball",
        "0002_perspective-center_trimmed-ball",
    ]


def test_physics_iq_metrics_map_official_score_keys() -> None:
    metrics = metric_values_from_scores(
        {
            "final_score_orig": 0.75,
            "final_score_stable": 0.74,
            "score_spatiotemporal": 0.78,
            "score_spatial": 0.76,
            "score_weighted_spatial": 0.77,
            "score_mse": 0.02,
        },
        ORIGINAL,
    )
    assert metrics["physics_iq_score"] == 0.75
    assert metrics["physics_iq_stable_score"] == 0.74


def test_physics_iq_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("physics-iq")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/physics_iq/run_physics_iq_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 6


def test_physics_iq_official_run_requires_reference_dataset(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("physics-iq")
    assert sample_path is not None
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0001_perspective-left_trimmed-ball-and-block-fall.mp4").write_bytes(b"fake")
    (generated_dir / "physics_iq_results.csv").write_bytes(sample_path.read_bytes())
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/physics_iq/run_physics_iq_official_runner.py",
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
    assert completed.returncode == 1
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["official_benchmark_verified"] is False
    assert "needs the official dataset assets" in scorecard["run"]["error"]


def test_physics_iq_unique_generation_records_respects_take_one() -> None:
    records = unique_generation_records(load_description_rows())
    assert len(records) == 198
