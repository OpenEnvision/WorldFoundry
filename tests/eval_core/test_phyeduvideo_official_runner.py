from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.phyeduvideo_metrics import compute_phyeduvideo_metrics
from worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.phyeduvideo_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_phyeduvideo_generation_requests,
    unique_prompt_records,
    load_prompt_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _official_repo_dir(name: str) -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_OFFICIAL_REPOS_DIR", REPO_ROOT.parent / "github_repos")) / name


def test_phyeduvideo_prompt_materialization_uses_official_video_names(tmp_path: Path) -> None:
    prompts = tmp_path / "Prompts.json"
    prompts.write_text(
        json.dumps(
            [
                {
                    "concept": "Inertia",
                    "teaching_points": {
                        "T01": {
                            "teaching_point": "Rest stays at rest.",
                            "prompt": "A toy car sits still on a floor.",
                        }
                    },
                    "Id": 1,
                    "category": "Mechanics",
                },
                {
                    "concept": "Inertia",
                    "teaching_points": {
                        "T05": {
                            "teaching_point": "Force causes acceleration.",
                            "prompt": "A person pushes a shopping cart.",
                        }
                    },
                    "Id": 1,
                    "category": "Mechanics",
                },
            ]
        ),
        encoding="utf-8",
    )
    requests = materialize_phyeduvideo_generation_requests(prompts_path=prompts)
    assert [request.sample_id for request in requests] == ["Id1_T01", "Id1_T05"]


def test_phyeduvideo_metrics_from_summary_csv() -> None:
    rows = [{"metric_id": metric_id, "score": 0.75} for metric_id in (
        "semantic_adherence",
        "physics_commonsense",
        "motion_smoothness",
        "temporal_flickering",
        "phyeduvideo_average",
    )]
    metrics = compute_phyeduvideo_metrics(rows=rows)["metrics"]
    assert metrics["semantic_adherence"] == 0.75
    assert metrics["phyeduvideo_average"] == 0.75


def test_phyeduvideo_official_runner_normalizes_sample_results(tmp_path: Path) -> None:
    from worldfoundry.evaluation.utils import benchmark_task_sample_path

    sample_path = benchmark_task_sample_path("phyeduvideo")
    assert sample_path is not None
    repo_root = _official_repo_dir("PhyEduVideo")
    env = os.environ.copy()
    if repo_root.is_dir():
        env["WORLDFOUNDRY_PHYEDUVIDEO_ROOT"] = str(repo_root)
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyeduvideo/run_phyeduvideo_official_runner.py",
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
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 5


def test_phyeduvideo_official_run_with_artifact_results_writes_scorecard(tmp_path: Path) -> None:
    from worldfoundry.evaluation.utils import benchmark_task_sample_path

    sample_path = benchmark_task_sample_path("phyeduvideo")
    assert sample_path is not None
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "Id1_T01.mp4").write_bytes(b"fake")
    (generated_dir / "phyeduvideo_results.csv").write_bytes(sample_path.read_bytes())
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/phyeduvideo/run_phyeduvideo_official_runner.py",
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
    assert (output_dir / "phyeduvideo_results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "phyeduvideo_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 5


def test_phyeduvideo_unique_prompt_records_match_bundled_count() -> None:
    records = unique_prompt_records(load_prompt_records())
    assert len(records) == CANONICAL_PROMPT_COUNT
