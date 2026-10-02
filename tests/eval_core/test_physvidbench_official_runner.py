from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_metrics import compute_physvidbench_metrics
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_prompts import (
    materialize_physvidbench_generation_requests,
    unique_prompt_records,
    load_prompt_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _official_repo_dir(name: str) -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_OFFICIAL_REPOS_DIR", REPO_ROOT.parent / "github_repos")) / name


def test_physvidbench_prompt_materialization_uses_prompt_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts_questions.csv"
    manifest.write_text(
        "PromptID,Upsampled,Difficulty,Prompt,Question,Types\n"
        '0,True,hard,"A dog runs.","Does a dog run?","Action & Procedural Understanding"\n'
        '0,True,hard,"A dog runs.","Is the dog visible?","Object Properties & Affordances"\n'
        '1,True,easy,"A cat jumps.","Does a cat jump?","Temporal Dynamics"\n',
        encoding="utf-8",
    )
    requests = materialize_physvidbench_generation_requests(prompt_manifest_path=manifest)
    assert [request.sample_id for request in requests] == ["0", "1"]


def test_physvidbench_metrics_from_qa_rows() -> None:
    rows = [
        {"Types": "Object Properties & Affordances", "Match": True},
        {"Types": "Action & Procedural Understanding", "Match": False},
        {"Types": "Temporal Dynamics", "Match": True},
    ]
    metrics = compute_physvidbench_metrics(qa_rows=rows)["metrics"]
    assert metrics["physical_commonsense_accuracy"] == 2 / 3
    assert metrics["affordance_understanding"] == 1.0
    assert metrics["tool_use_consistency"] == 0.0
    assert metrics["temporal_dynamics_consistency"] == 1.0


def test_physvidbench_official_run_with_mock_backend_writes_scorecard(tmp_path: Path) -> None:
    repo_root = _official_repo_dir("PhysVidBenchCode")
    if not repo_root.is_dir():
        return
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0000.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "official-run"
    env = os.environ.copy()
    env["WORLDFOUNDRY_PHYSVIDBENCH_JUDGE_BACKEND"] = "mock"
    env["WORLDFOUNDRY_PHYSVIDBENCH_ROOT"] = str(repo_root)
    env["PYTHONPATH"] = str(REPO_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "worldfoundry/evaluation/tasks/execution/runners/physvidbench/run_physvidbench_official_runner.py",
            "--run-official",
            "--generated-artifact-dir",
            str(generated_dir),
            "--physvidbench-root",
            str(repo_root),
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
    assert (output_dir / "output.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "physvidbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 6
