from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_metrics import compute_phyground_metrics
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_prompts import (
    materialize_phyground_generation_requests,
    unique_generation_records,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phyground_prompt_materialization_uses_video_ids(tmp_path: Path) -> None:
    prompts_json = tmp_path / "phyground.json"
    prompts_json.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "video": "ball_fall_0001",
                        "prompt": "A ball falls under gravity.",
                        "physical_laws": ["gravity"],
                        "first_frame_image": "ball_fall_0001.png",
                    },
                    {
                        "video": "water_pour_0002",
                        "prompt": "Water pours into a glass.",
                        "physical_laws": ["flow_dynamics"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    requests = materialize_phyground_generation_requests(prompts_json_path=prompts_json)
    assert [request.sample_id for request in requests] == ["ball_fall_0001", "water_pour_0002"]


def test_phyground_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in (
        "semantic_adherence",
        "physical_temporal_validity",
        "persistence",
        "solid_body_score",
        "fluid_dynamics_score",
        "optics_score",
        "phyground_overall",
    )]
    metrics = compute_phyground_metrics(rows=rows)["metrics"]
    assert metrics["semantic_adherence"] == 0.75
    assert metrics["phyground_overall"] == 0.75


def test_phyground_metrics_from_scores_json() -> None:
    rows = [
        {
            "video": "sample-1",
            "SA": 4,
            "PTV": 5,
            "persistence": 3,
            "physical": {
                "laws": {
                    "gravity": {"score": 5, "status": "scored"},
                    "flow_dynamics": {"score": 4, "status": "scored"},
                    "reflection": {"score": 3, "status": "scored"},
                }
            },
        }
    ]
    metrics = compute_phyground_metrics(rows=rows)["metrics"]
    assert metrics["semantic_adherence"] == 0.8
    assert metrics["physical_temporal_validity"] == 1.0
    assert metrics["solid_body_score"] == 1.0
    assert metrics["fluid_dynamics_score"] == 0.8
    assert metrics["optics_score"] == 0.6


def test_phyground_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("phyground")
    assert sample_path is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyground/run_phyground_official_runner.py",
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


def test_phyground_official_run_with_artifact_scores_writes_scorecard(tmp_path: Path) -> None:
    prompts_json = tmp_path / "prompts" / "phyground.json"
    prompts_json.parent.mkdir(parents=True)
    prompts_json.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "video": "ball_fall_0001",
                        "prompt": "A ball falls.",
                        "physical_laws": ["gravity"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "ball_fall_0001.mp4").write_bytes(b"fake")
    (generated_dir / "scores.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "video": "ball_fall_0001",
                        "SA": 4,
                        "PTV": 5,
                        "persistence": 3,
                        "physical": {
                            "laws": {
                                "gravity": {"score": 5, "status": "scored"},
                                "flow_dynamics": {"score": 4, "status": "scored"},
                                "reflection": {"score": 3, "status": "scored"},
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_PHYGROUND_JUDGE_BACKEND"] = "artifact"
    env["WORLDFOUNDRY_PHYGROUND_PROMPT_MANIFEST"] = str(prompts_json)
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyground/run_phyground_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--prompt-manifest",
            str(prompts_json),
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
    assert (output_dir / "scores.json").is_file()
    assert scorecard["evaluation"]["kind"] == "phyground_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 7


def test_phyground_unique_generation_records_deduplicates() -> None:
    records = unique_generation_records(
        [
            {"prompt_id": "a", "prompt": "one"},
            {"prompt_id": "a", "prompt": "duplicate"},
            {"prompt_id": "b", "prompt": "two"},
        ]
    )
    assert [record["prompt_id"] for record in records] == ["a", "b"]
