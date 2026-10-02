from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]

PHYSICS_VIDEO_BENCHMARK_RUNNERS = (
    ("phygenbench", "worldfoundry/evaluation/tasks/execution/runners/phygenbench/run_phygenbench_official_runner.py"),
    ("videophy", "worldfoundry/evaluation/tasks/execution/runners/videophy/run_videophy_official_runner.py"),
    ("videophy2", "worldfoundry/evaluation/tasks/execution/runners/videophy2/run_videophy2_official_runner.py"),
)


@pytest.mark.parametrize(("benchmark_id", "runner_script"), PHYSICS_VIDEO_BENCHMARK_RUNNERS)
def test_physics_video_benchmark_runner_normalizes_sample_results(
    tmp_path: Path,
    benchmark_id: str,
    runner_script: str,
) -> None:
    sample_path = benchmark_task_sample_path(benchmark_id)
    assert sample_path is not None, f"missing sample results for {benchmark_id}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / benchmark_id
    completed = subprocess.run(
        [
            sys.executable,
            runner_script,
            "--from-upstream-results",
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
    assert scorecard["benchmark"]["benchmark_id"] == benchmark_id
    assert scorecard["normalization_ok"] is True
    assert scorecard["metrics"]["summary"]["official_available_count"] >= 1


@pytest.mark.parametrize(("benchmark_id", "runner_script"), PHYSICS_VIDEO_BENCHMARK_RUNNERS)
def test_physics_video_benchmark_runner_run_fixture(
    tmp_path: Path,
    benchmark_id: str,
    runner_script: str,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    output_dir = tmp_path / f"{benchmark_id}-fixture"
    completed = subprocess.run(
        [
            sys.executable,
            runner_script,
            "--run-fixture",
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
    assert scorecard["benchmark"]["benchmark_id"] == benchmark_id
    assert scorecard["normalization_ok"] is True
