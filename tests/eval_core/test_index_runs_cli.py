from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.cli import main


def _summary(*, run_id: str, model_id: str, quality: float) -> dict:
    return {
        "schema_version": "worldfoundry-run-summary",
        "run": {"run_id": run_id, "status": "succeeded"},
        "benchmark": {"benchmark_name": "cli-index-benchmark", "task_type": "cli-index-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "cli-index-dataset", "sample_count": 1},
        "counts": {
            "sample_count": 1,
            "successful_samples": 1,
            "failed_samples": 0,
            "failed_sample_ids": [],
        },
        "metrics": {
            "leaderboard": {"quality": quality},
            "per_metric": {"quality": {"mean": quality, "higher_is_better": True}},
        },
        "leaderboard": {"quality": quality},
        "eligibility": {"score_valid": True, "leaderboard_valid": False},
        "artifacts": {"summary": f"/tmp/{run_id}/summary.json"},
    }


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_root_help_lists_index_runs(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "index-runs" in captured.out


def test_index_runs_cli_writes_outputs_and_prints_json(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    output_dir = tmp_path / "history"
    _write_summary(root / "run-a", _summary(run_id="run-a", model_id="model-a", quality=0.5))
    _write_summary(root / "run-b", _summary(run_id="run-b", model_id="model-b", quality=0.8))

    exit_code = main(["index-runs", str(root), "--output-dir", str(output_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-run-index"
    assert payload["run_count"] == 2
    assert payload["metric_ids"] == ["quality"]
    assert payload["artifacts"] == {
        "index_json": str((output_dir / "index.json").resolve()),
        "index_jsonl": str((output_dir / "index.jsonl").resolve()),
        "index_html": str((output_dir / "index.html").resolve()),
    }
    assert (output_dir / "index.json").is_file()
    assert (output_dir / "index.jsonl").is_file()
    assert "WorldFoundry Runs" in (output_dir / "index.html").read_text(encoding="utf-8")
    assert len((output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_index_runs_cli_can_fail_on_issues(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    _write_summary(root / "run-a", _summary(run_id="duplicate", model_id="model-a", quality=0.5))
    _write_summary(root / "run-b", _summary(run_id="duplicate", model_id="model-b", quality=0.8))

    exit_code = main(["index-runs", str(root), "--fail-on-issue", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "duplicate run_id: duplicate" in "\n".join(payload["issues"])
