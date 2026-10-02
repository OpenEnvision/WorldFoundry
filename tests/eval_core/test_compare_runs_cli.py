from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.cli import main
from worldfoundry.evaluation.reporting import write_run_index


def _summary(*, run_id: str, model_id: str, quality: float) -> dict:
    return {
        "schema_version": "worldfoundry-run-summary",
        "run": {"run_id": run_id, "status": "succeeded"},
        "benchmark": {"benchmark_name": "cli-benchmark", "task_type": "cli-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "cli-dataset", "sample_count": 1},
        "counts": {
            "sample_count": 1,
            "successful_samples": 1,
            "failed_samples": 0,
            "failed_sample_ids": [],
        },
        "metrics": {
            "leaderboard": {"quality": quality, "artifact_count": 1.0},
            "per_metric": {
                "quality": {"mean": quality, "higher_is_better": True},
                "artifact_count": {"mean": 1.0, "higher_is_better": True},
            },
        },
        "leaderboard": {"quality": quality, "artifact_count": 1.0},
        "eligibility": {"score_valid": True, "leaderboard_valid": False},
        "artifacts": {"summary": f"/tmp/{run_id}/summary.json"},
    }


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_compare_runs_cli_reads_summaries_and_writes_outputs(tmp_path: Path, capsys) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    output_json = tmp_path / "comparison.json"
    output_md = tmp_path / "comparison.md"
    _write_summary(run_a, _summary(run_id="run-a", model_id="model-a", quality=0.5))
    _write_summary(run_b, _summary(run_id="run-b", model_id="model-b", quality=0.8))

    exit_code = main(
        [
            "compare-runs",
            str(run_b),
            "--baseline",
            str(run_a),
            "--baseline-label",
            "base",
            "--label",
            "candidate",
            "--metric",
            "quality",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-comparison"
    assert payload["baseline"] == "base"
    assert payload["metric_ids"] == ["quality"]
    assert payload["runs"][1]["label"] == "candidate"
    assert payload["metrics"]["quality"]["deltas"]["candidate"] == pytest.approx(0.3)
    assert output_json.is_file()
    assert output_md.is_file()
    assert "quality delta" in output_md.read_text(encoding="utf-8")


def test_compare_runs_cli_reports_unknown_metric_issue(tmp_path: Path, capsys) -> None:
    run_a = tmp_path / "run-a"
    _write_summary(run_a, _summary(run_id="run-a", model_id="model-a", quality=0.5))

    exit_code = main(["compare-runs", str(run_a), "--metric", "missing", "--fail-on-issue", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["issues"] == ["metric not found in any run: missing"]


def test_compare_runs_cli_selects_runs_from_index(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    index_path = tmp_path / "index.json"
    _write_summary(root / "run-a", _summary(run_id="run-a", model_id="model-a", quality=0.5))
    _write_summary(root / "run-b", _summary(run_id="run-b", model_id="model-b", quality=0.8))
    _write_summary(root / "run-c", _summary(run_id="run-c", model_id="model-c", quality=0.2))
    write_run_index(root, output_json=index_path)

    exit_code = main(
        [
            "compare-runs",
            "--index",
            str(index_path),
            "--index-model",
            "model-a",
            "--index-model",
            "model-b",
            "--require-score-valid",
            "--require-metric",
            "quality",
            "--baseline-run",
            "run-a",
            "--metric",
            "quality",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["run_count"] == 2
    assert payload["baseline"] == "run-a"
    assert [row["label"] for row in payload["rows"]] == ["run-a", "run-b"]
    assert payload["metrics"]["quality"]["deltas"]["run-b"] == pytest.approx(0.3)


def test_compare_runs_cli_reports_empty_index_selection(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    index_path = tmp_path / "index.json"
    _write_summary(root / "run-a", _summary(run_id="run-a", model_id="model-a", quality=0.5))
    write_run_index(root, output_json=index_path)

    exit_code = main(["compare-runs", "--index", str(index_path), "--index-model", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "run index selection matched no comparable runs" in captured.err


def test_compare_runs_cli_refuses_incompatible_benchmarks(tmp_path: Path, capsys) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    left = _summary(run_id="run-a", model_id="model-a", quality=0.5)
    right = _summary(run_id="run-b", model_id="model-b", quality=0.8)
    right["benchmark"]["benchmark_name"] = "different-benchmark"
    _write_summary(run_a, left)
    _write_summary(run_b, right)

    exit_code = main(["compare-runs", str(run_a), str(run_b)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "runs are not comparable" in captured.err
    assert "benchmark_id" in captured.err
