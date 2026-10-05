from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.cli import main


def test_existing_results_scorecard_public_fields_are_stable(tmp_path: Path) -> None:
    output_dir = tmp_path / "scorecard_snapshot"
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-0001",
                "task_name": "scorecard_snapshot",
                "status": "succeeded",
                "artifacts": {"generated_text": {"uri": "outputs/sample-0001.txt", "kind": "text"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert main([
        "evaluate",
        "--results-path",
        str(results_path),
        "--output-dir",
        str(output_dir),
        "--benchmark-id",
        "scorecard-snapshot-benchmark",
        "--model-id",
        "scorecard-snapshot-model",
        "--dataset-id",
        "scorecard-snapshot-dataset",
        "--metric",
        "artifact_count",
        "--run-id",
        "scorecard-snapshot-run",
        "--json",
    ]) == 0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["run"]["run_id"] == "scorecard-snapshot-run"
    assert scorecard["benchmark"]["benchmark_name"] == "scorecard-snapshot-benchmark"
    assert scorecard["model"]["model_id"] == "scorecard-snapshot-model"
    assert scorecard["dataset"]["dataset_id"] == "scorecard-snapshot-dataset"
    assert scorecard["evaluation"]["kind"] == "existing_results"
    assert scorecard["evaluation"]["num_results"] == 1
    assert scorecard["eligibility"]["score_valid"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert "official/full-suite" in scorecard["eligibility"]["leaderboard_reason"]
    assert scorecard["metrics"]["summary"]["sample_count"] == 1
    assert scorecard["artifacts"]["scorecard"].endswith("scorecard.json")
