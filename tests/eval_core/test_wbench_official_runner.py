# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.wbench import run_wbench_official_runner

pytestmark = pytest.mark.unit


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
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "wbench_result_normalizer"
    assert scorecard["metrics"]["leaderboard"] == pytest.approx(
        {
            "aesthetic_quality": 0.8,
            "imaging_quality": 0.6,
            "scene_adherence": 0.7,
            "navigation_trajectory": 0.5,
            "background_consistency": 0.9,
            "visual_plausibility": 0.4,
        }
    )
    assert scorecard["metrics"]["per_metric"]["quality_score"]["raw_score"] is None
    assert scorecard["metrics"]["per_metric"]["wbench_average"]["raw_score"] is None
    assert len(scorecard["evaluation"]["missing_component_metrics"]) == 16
    assert not scorecard["evaluation"]["benchmark_comparable"]
    assert not scorecard["leaderboard_valid"]
    assert raw_rows[-1]["metric_id"] == "wbench_average"
    assert raw_rows[-1]["raw_score"] is None
