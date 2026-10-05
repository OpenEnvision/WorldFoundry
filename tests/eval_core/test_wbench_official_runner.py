from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.wbench import run_wbench_official_runner


def test_wbench_official_runner_normalizes_report_json(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "model": "fixture",
                "n_cases": 2,
                "n_navi": 1,
                "full": {
                    "aesthetic_quality": {"mean": 0.8, "n": 2},
                    "imaging_quality": {"mean": 0.6, "n": 2},
                    "scene_adherence": {"mean": 0.7, "n": 1},
                    "navigation_trajectory": {"mean": 0.5, "n": 1},
                    "background_consistency": {"mean": 0.9, "n": 2},
                    "visual_plausibility": {"mean": 0.4, "n": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = run_wbench_official_runner.main(
        [
            "--official-results-path",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "wbench_result_normalizer"
    assert scorecard["metrics"]["leaderboard"]["quality_score"] == pytest.approx(0.7)
    assert scorecard["metrics"]["leaderboard"]["wbench_average"] == pytest.approx(0.64)
    assert raw_rows[-1]["metric_id"] == "wbench_average"
