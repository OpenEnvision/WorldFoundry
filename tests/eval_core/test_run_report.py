from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.reporting import (
    RUN_SUMMARY_SCHEMA_VERSION,
    build_run_summary,
    write_run_report_artifacts,
)


def _scorecard() -> dict:
    return {
        "schema_version": "worldfoundry-scorecard",
        "run": {
            "run_id": "report-run",
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "worldfoundry_version": "0.0.test",
            "run_fingerprint": "abc123",
        },
        "benchmark": {
            "suite": "unit",
            "benchmark_id": "report-benchmark",
            "benchmark_name": "report-benchmark",
            "benchmark_revision": "benchmark-rev",
            "task_type": "report-task",
            "evaluation_protocol": "report-benchmark:official-run",
            "protocol_revision": "benchmark-rev",
            "protocol_config_hash": "official-config",
        },
        "model": {"model_id": "report-model", "model_name": "Report Model", "model_type": "fake"},
        "dataset": {
            "dataset_id": "report-dataset",
            "name": "Report Dataset",
            "dataset_revision": "dataset-rev",
            "dataset_hash": "dataset-hash",
            "split": "test",
            "sample_count": 2,
        },
        "generation": {"num_requests": 2, "successful": 2, "failed": 0, "error_sample_ids": []},
        "metrics": {
            "leaderboard": {"quality": 0.75},
            "per_metric": {"quality": {"mean": 0.75, "sample_count": 2}},
            "summary": {
                "sample_count": 2,
                "successful_samples": 2,
                "failed_samples": 0,
                "failed_sample_ids": [],
                "metric_revision": "metric-rev",
                "metric_config_hash": "metric-config",
            },
        },
        "evaluation": {"kind": "existing_results"},
        "provenance": {
            "producer": "catalog_model",
            "fidelity": {"generation": "custom", "data": "official", "evaluation": "official"},
        },
        "eligibility": {
            "score_valid": True,
            "leaderboard_valid": False,
            "leaderboard_eligible": False,
            "reasons": ["missing evidence"],
            "blocking_reasons": ["missing evidence"],
        },
        "artifacts": {
            "results": "/tmp/results.jsonl",
            "scorecard": "/tmp/scorecard.json",
        },
    }


def test_build_run_summary_compacts_scorecard_fields() -> None:
    summary = build_run_summary(_scorecard())

    assert summary["schema_version"] == RUN_SUMMARY_SCHEMA_VERSION
    assert summary["run"]["run_id"] == "report-run"
    assert summary["benchmark"]["benchmark_name"] == "report-benchmark"
    assert summary["model"]["model_id"] == "report-model"
    assert summary["dataset"]["dataset_id"] == "report-dataset"
    assert summary["counts"]["sample_count"] == 2
    assert summary["leaderboard"]["quality"] == 0.75
    assert summary["eligibility"]["leaderboard_valid"] is False
    assert summary["provenance"]["producer"] == "catalog_model"
    assert summary["evaluation"]["mode"] == "new_model_evaluation"
    assert summary["comparison_identity"]["status"] == "complete"


def test_write_run_report_artifacts_writes_json_and_markdown(tmp_path: Path) -> None:
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(json.dumps(_scorecard()), encoding="utf-8")

    paths = write_run_report_artifacts(output_dir=tmp_path, scorecard_path=scorecard_path)

    assert paths["summary"] == (tmp_path / "summary.json").resolve()
    assert paths["report"] == (tmp_path / "report.md").resolve()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert summary["schema_version"] == RUN_SUMMARY_SCHEMA_VERSION
    assert summary["leaderboard"]["quality"] == 0.75
    assert "# WorldFoundry Run Report" in report
    assert "Evaluation mode: new_model_evaluation" in report
    assert "Protocol fidelity: official" in report
    assert "| quality | 0.75 |" in report
    assert "`/tmp/results.jsonl`" in report
