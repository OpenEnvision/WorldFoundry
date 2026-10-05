from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.cli import main
from worldfoundry.evaluation.reporting import validate_contract_file, validate_contract_paths


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _summary(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-run-summary",
            "run": {"run_id": "run-a", "status": "succeeded"},
            "benchmark": {"benchmark_name": "bench-a", "task_type": "task-a"},
            "model": {"model_id": "model-a", "model_name": "model-a"},
            "dataset": {"dataset_id": "dataset-a", "sample_count": 1},
            "counts": {"sample_count": 1, "successful_samples": 1, "failed_samples": 0},
            "metrics": {
                "leaderboard": {"quality": 1.0},
                "per_metric": {"quality": {"mean": 1.0, "higher_is_better": True}},
                "summary": {"sample_count": 1, "successful_samples": 1, "failed_samples": 0},
            },
            "leaderboard": {"quality": 1.0},
            "eligibility": {"score_valid": True, "leaderboard_valid": False},
            "artifacts": {"summary": str(path.resolve())},
        },
    )


def _scorecard(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-scorecard",
            "run": {"run_id": "run-a", "status": "succeeded"},
            "benchmark": {"benchmark_name": "bench-a"},
            "model": {"model_id": "model-a"},
            "dataset": {"dataset_id": "dataset-a"},
            "generation": {"successful": 1, "failed": 0},
            "metrics": {
                "leaderboard": {"quality": 1.0},
                "per_metric": {"quality": {"mean": 1.0}},
                "summary": {"sample_count": 1, "successful_samples": 1, "failed_samples": 0},
            },
            "eligibility": {"score_valid": True, "leaderboard_valid": False},
            "artifacts": {"scorecard": str(path.resolve())},
        },
    )


def _runner_scorecard(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-scorecard",
            "run": {"status": "contract_fixture"},
            "benchmark": {"benchmark_id": "vbench"},
            "dataset": {"generated_file_count": 1},
            "evaluation": {"available": False, "kind": "contract_fixture"},
            "metrics": {
                "per_metric": {"quality": {"available": False}},
                "summary": {"skipped": 1},
            },
            "artifacts": {"scorecard": str(path.resolve())},
        },
    )


def _comparison(path: Path, rows: list[dict] | object | None = None) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-run-comparison",
            "run_count": 1,
            "metric_ids": ["quality"],
            "available_metric_ids": ["quality"],
            "rows": rows if rows is not None else [{"label": "run-a", "metrics": {"quality": 1.0}}],
            "runs": rows if rows is not None else [{"label": "run-a", "metrics": {"quality": 1.0}}],
            "metrics": {"quality": {"values": {"run-a": 1.0}, "deltas": {}}},
            "artifacts": {"comparison_json": str(path.resolve())},
            "issues": [],
        },
    )


def _index(path: Path) -> Path:
    row = {
        "label": "run-a",
        "source_path": str((path.parent / "run-a" / "summary.json").resolve()),
        "metrics": {"quality": 1.0},
        "artifacts": {},
    }
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-run-index",
            "run_count": 1,
            "rows": [row],
            "runs": [row],
            "artifacts": {"index_json": str(path.resolve())},
            "issues": [],
        },
    )


def _index_jsonl(path: Path) -> Path:
    row = {"label": "run-a", "source_path": str(path.resolve()), "metrics": {"quality": 1.0}}
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _suite_manifest(path: Path, *, total: int = 1) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "worldfoundry-model-benchmark-suite",
            "status": "succeeded",
            "exit_code": 0,
            "summary": {"total": total, "succeeded": 1, "failed": 0, "skipped": 0},
            "cells": [
                {
                    "model_id": "model-a",
                    "benchmark_id": "bench-a",
                    "status": "succeeded",
                    "run_manifest_path": str((path.parent / "run_manifest.json").resolve()),
                    "run_summary_path": str((path.parent / "summary.json").resolve()),
                }
            ],
        },
    )


def test_validate_contract_paths_accepts_supported_artifacts(tmp_path: Path) -> None:
    paths = [
        _summary(tmp_path / "summary.json"),
        _scorecard(tmp_path / "scorecard.json"),
        _comparison(tmp_path / "comparison.json"),
        _index(tmp_path / "index.json"),
        _suite_manifest(tmp_path / "suite_manifest.json"),
    ]

    report = validate_contract_paths(paths)
    runner_result = validate_contract_file(_runner_scorecard(tmp_path / "runner-scorecard.json"))

    assert report["schema_version"] == "worldfoundry-contract-validation"
    assert report["ok"] is True
    assert report["valid_count"] == 5
    assert {item["kind"] for item in report["results"]} == {
        "run_summary",
        "scorecard",
        "run_comparison",
        "run_index",
        "model_benchmark_suite",
    }
    assert runner_result["ok"] is True
    assert runner_result["warning_count"] >= 1


def test_validate_contract_file_handles_jsonl_index_and_reports_bad_shapes(tmp_path: Path) -> None:
    jsonl_result = validate_contract_file(_index_jsonl(tmp_path / "index.jsonl"))
    bad_comparison = validate_contract_file(_comparison(tmp_path / "bad-comparison.json", rows={"not": "a-list"}))
    bad_suite = validate_contract_file(_suite_manifest(tmp_path / "bad-suite.json", total=2))
    source_scorecard = validate_contract_file(
        _write_json(tmp_path / "source-scorecard.json", {"schema_version": "worldfoundry-unknown-scorecard"})
    )

    assert jsonl_result["ok"] is True
    assert jsonl_result["kind"] == "run_index"
    assert bad_comparison["ok"] is False
    assert "missing or invalid list: rows" in bad_comparison["errors"]
    assert bad_suite["ok"] is False
    assert "summary.total mismatch" in "\n".join(bad_suite["errors"])
    assert source_scorecard["ok"] is False
    assert "unsupported schema_version" in "\n".join(source_scorecard["errors"])


def test_validate_artifact_cli_reports_counts_and_kind_mismatches(tmp_path: Path, capsys) -> None:
    summary = _summary(tmp_path / "summary.json")
    scorecard = _scorecard(tmp_path / "scorecard.json")

    exit_code = main(["validate-artifact", str(summary), str(scorecard), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["path_count"] == 2

    mismatch_code = main(["validate-artifact", str(scorecard), "--kind", "run-summary", "--json"])
    mismatch_payload = json.loads(capsys.readouterr().out)

    assert mismatch_code == 1
    assert mismatch_payload["ok"] is False
    assert "expected schema_version" in "\n".join(mismatch_payload["results"][0]["errors"])


def test_validate_artifact_cli_handles_malformed_json_without_traceback(tmp_path: Path, capsys) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad json", encoding="utf-8")

    exit_code = main(["validate-artifact", str(bad_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["invalid_count"] == 1
    assert payload["results"][0]["errors"]
