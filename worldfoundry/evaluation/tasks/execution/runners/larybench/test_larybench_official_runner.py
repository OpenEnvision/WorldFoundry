from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import benchmark_runtime_profiles_by_id
from worldfoundry.evaluation.tasks.execution.runners.larybench.cli import create_parser
from worldfoundry.evaluation.tasks.execution.runners.larybench.run_larybench_official_runner import (
    _build_runtime_command,
    _resolved_layout,
    main,
    parse_args,
)


def test_extract_command_preserves_upstream_perspective_argument(tmp_path: Path) -> None:
    cli_args = create_parser().parse_args(
        [
            "extract",
            "--model",
            "dinov2",
            "--dataset",
            "robot_1st",
            "--perspective",
            "3rd",
        ]
    )
    assert cli_args.perspective == "3rd"

    args = parse_args(
        [
            "--run-official",
            "--stage",
            "extract",
            "--model",
            "dinov2",
            "--dataset",
            "robot_1st",
            "--perspective",
            "3rd",
            "--output-dir",
            str(tmp_path),
        ]
    )
    command = _build_runtime_command(args, _resolved_layout(args))
    assert command[command.index("--perspective") + 1] == "3rd"


def test_classification_result_normalizer_writes_scorecard(tmp_path: Path) -> None:
    source = tmp_path / "classification_summary.json"
    source.write_text(
        json.dumps(
            {
                "accuracy": 0.75,
                "macro_precision": 0.7,
                "macro_recall": 0.68,
                "macro_f1": 0.69,
                "weighted_f1": 0.73,
                "sample_count": 8,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized"

    assert main(["--official-results-path", str(source), "--stage", "classify", "--output-dir", str(output)]) == 0

    scorecard = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["normalizer_only"] is True
    assert scorecard["integration_evidence"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["metrics"]["summary"]["primary_metric_id"] == "top1_accuracy"
    assert scorecard["metrics"]["per_metric"]["top1_accuracy"]["score"] == pytest.approx(0.75)
    assert (output / "benchmark_contract.json").is_file()
    assert len((output / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_extraction_result_normalizer_reports_populated_coverage(tmp_path: Path) -> None:
    source = tmp_path / "train_la_calvin_5_dinov2.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src_img", "tgt_img", "la_path"])
        writer.writeheader()
        writer.writerows(
            [
                {"src_img": "a.png", "tgt_img": "b.png", "la_path": "calvin/train/dinov2/one.npz"},
                {"src_img": "c.png", "tgt_img": "d.png", "la_path": "nan"},
                {"src_img": "e.png", "tgt_img": "f.png", "la_path": ""},
            ]
        )
    output = tmp_path / "normalized"

    assert main(["--official-results-path", str(source), "--stage", "extract", "--output-dir", str(output)]) == 0

    scorecard = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    metrics = scorecard["metrics"]["per_metric"]
    assert metrics["extraction_coverage"]["score"] == pytest.approx(1 / 3)
    assert metrics["extracted_samples"]["score"] == 1
    assert metrics["input_samples"]["score"] == 3


def test_regression_result_normalizer_preserves_lower_is_better_metrics(tmp_path: Path) -> None:
    source = tmp_path / "best_result.json"
    source.write_text(
        json.dumps(
            {
                "best_epoch": 4,
                "train_loss": 0.12,
                "val_seen_loss": 0.2,
                "val_seen_mse": 0.18,
                "val_unseen_mse": 0.31,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized"

    assert main(["--official-results-path", str(source), "--stage", "regress", "--output-dir", str(output)]) == 0

    scorecard = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    metric = scorecard["metrics"]["per_metric"]["val_seen_mse"]
    assert scorecard["metrics"]["summary"]["primary_metric_id"] == "val_seen_mse"
    assert metric["score"] == pytest.approx(0.18)
    assert metric["higher_is_better"] is False


def test_runtime_profile_is_discoverable_and_fail_closed() -> None:
    profile = benchmark_runtime_profiles_by_id()["larybench"]
    assert profile["environment_id"] == "worldfoundry-larybench-cu128"
    assert profile["status"] == "in_tree_runtime_ready_external_assets_pending"
    assert profile["requires_cuda_visibility"] is False
    assert profile["bounded_fixture_validation"]["leaderboard_valid"] is False
