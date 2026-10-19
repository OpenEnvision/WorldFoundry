from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.worldbench import run_worldbench_official_runner


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_worldbench_official_runner_normalizes_summary_and_sample_rows(tmp_path: Path) -> None:
    upstream_results = tmp_path / "worldbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "summary": {
                    "foreground_miou": "62.5%",
                    "text_based_accuracy": 0.75,
                    "multiple_choice_accuracy": 0.6,
                    "binary_accuracy": "80%",
                },
                "per_sample_scores": [
                    {"scene_id": "scene-001", "component": "video_based", "correct": True},
                    {"question_id": "q-001", "question_type": "multiple_choice", "prediction": "A", "answer": "B"},
                    {"question_id": "q-002", "question_type": "binary", "score": "100%"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = _jsonl(output_dir / "raw_metric_table.jsonl")
    sample_rows = _jsonl(output_dir / "per_sample_scores.jsonl")
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "official_worldbench_result_normalizer"
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["per_metric"]["foreground_miou"]["raw_score"] == 62.5
    assert scorecard["metrics"]["per_metric"]["foreground_miou"]["normalized_score"] == 0.625
    assert scorecard["metrics"]["per_metric"]["binary_accuracy"]["normalized_score"] == 0.8
    assert "worldbench_average" not in scorecard["metrics"]["leaderboard"]
    assert {row["metric_id"] for row in raw_rows} == {
        "foreground_miou",
        "foreground_dice",
        "background_rmse",
        "text_based_accuracy",
        "multiple_choice_accuracy",
        "binary_accuracy",
    }
    assert len(sample_rows) == 3
    assert sample_rows[1]["correct"] is False


def test_worldbench_official_runner_reads_results_path_env_and_computes_sample_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_results = tmp_path / "worldbench_samples.csv"
    upstream_results.write_text(
        "\n".join(
            [
                "sample_id,component,question_type,correct",
                "v-001,video_based,,true",
                "v-002,video_based,,false",
                "q-001,text_based,multiple_choice,true",
                "q-002,text_based,multiple_choice,false",
                "q-003,text_based,binary,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDFOUNDRY_WORLDBENCH_RESULTS_PATH", str(upstream_results))
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(["--output-dir", str(output_dir), "--json"])

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    sample_rows = _jsonl(output_dir / "per_sample_scores.jsonl")
    assert exit_code == 0
    assert scorecard["validation"]["official_result_shape"]["formats"] == ["csv"]
    assert scorecard["metrics"]["per_metric"]["foreground_miou"]["normalized_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["multiple_choice_accuracy"]["normalized_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["binary_accuracy"]["normalized_score"] == 1.0
    assert scorecard["metrics"]["per_metric"]["text_based_accuracy"]["normalized_score"] == pytest.approx(2 / 3)
    assert "worldbench_average" not in scorecard["metrics"]["leaderboard"]
    assert scorecard["integration_evidence"] is False
    assert len(sample_rows) == 5


def test_worldbench_artifact_score_import_is_normalizer_only(tmp_path: Path) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "metrics.json").write_text(
        json.dumps(
            {
                "foreground_miou": 0.7,
                "multiple_choice_accuracy": 0.8,
                "binary_accuracy": 0.6,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldbench-out"

    exit_code = run_worldbench_official_runner.main(
        [
            "--from-artifact-scores",
            "--artifact-score-dir",
            str(score_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "worldfoundry_artifact_score_normalizer"
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["validation"]["official_runtime_executed"] is False
    assert scorecard["validation"]["artifact_score_imported"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["official_results_imported"] is False
    assert scorecard["metrics"]["per_metric"]["foreground_miou"]["normalized_score"] == pytest.approx(0.7)
    assert scorecard["metrics"]["per_metric"]["multiple_choice_accuracy"]["normalized_score"] == pytest.approx(0.8)
    assert scorecard["metrics"]["per_metric"]["binary_accuracy"]["normalized_score"] == pytest.approx(0.6)
    assert "worldbench_average" not in scorecard["metrics"]["leaderboard"]
