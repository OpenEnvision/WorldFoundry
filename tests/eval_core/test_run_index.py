from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.reporting import (
    RUN_INDEX_SCHEMA_VERSION,
    build_run_browser_html,
    build_run_index,
    discover_run_summaries,
    load_run_index,
    run_paths_from_index,
    select_run_index_rows,
    write_run_index,
)


def _summary(*, run_id: str, model_id: str, quality: float, failed_samples: int = 0) -> dict:
    sample_count = 3
    successful_samples = sample_count - failed_samples
    return {
        "schema_version": "worldfoundry-run-summary",
        "run": {
            "run_id": run_id,
            "status": "succeeded" if failed_samples == 0 else "completed_with_failures",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
            "worldfoundry_version": "test",
            "run_fingerprint": f"fingerprint-{run_id}",
        },
        "benchmark": {"benchmark_name": "index-benchmark", "task_type": "index-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "index-dataset", "sample_count": sample_count},
        "counts": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": ["sample-2"] if failed_samples else [],
        },
        "metrics": {
            "leaderboard": {"quality": quality},
            "per_metric": {"quality": {"mean": quality, "higher_is_better": True}},
        },
        "leaderboard": {"quality": quality},
        "eligibility": {"score_valid": failed_samples == 0, "leaderboard_valid": False},
        "artifacts": {"summary": f"/tmp/{run_id}/summary.json"},
    }


def _scorecard(*, run_id: str, model_id: str, quality: float) -> dict:
    return {
        "schema_version": "worldfoundry-scorecard",
        "run": {"run_id": run_id, "status": "succeeded"},
        "benchmark": {"benchmark_name": "index-benchmark", "task_type": "index-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "index-dataset", "sample_count": 3},
        "generation": {"num_requests": 3, "successful": 3, "failed": 0},
        "metrics": {
            "leaderboard": {"quality": quality},
            "per_metric": {"quality": {"mean": quality, "higher_is_better": True}},
            "summary": {
                "sample_count": 3,
                "successful_samples": 3,
                "failed_samples": 0,
                "failed_sample_ids": [],
            },
        },
        "eligibility": {"score_valid": True, "leaderboard_valid": False},
        "artifacts": {"scorecard": f"/tmp/{run_id}/scorecard.json"},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_discover_run_summaries_prefers_root_summaries_and_prunes_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.7))
    _write_json(root / "run-a" / "metrics" / "summary.json", _summary(run_id="nested", model_id="bad", quality=0.1))
    _write_json(root / "run-b" / "scorecard.json", _scorecard(run_id="run-b", model_id="model-b", quality=0.8))
    _write_json(root / "run-c" / "summary.json", _summary(run_id="run-c", model_id="model-c", quality=0.9))
    _write_json(root / "run-c" / "scorecard.json", _scorecard(run_id="run-c-scorecard", model_id="model-c", quality=0.1))

    discovered = [path.relative_to(root).as_posix() for path in discover_run_summaries(root)]

    assert discovered == ["run-a/summary.json", "run-b/scorecard.json", "run-c/summary.json"]


def test_build_run_index_indexes_summaries_and_scorecard_fallback(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.7))
    _write_json(root / "run-b" / "scorecard.json", _scorecard(run_id="run-b", model_id="model-b", quality=0.8))

    index = build_run_index(root)

    assert index["schema_version"] == RUN_INDEX_SCHEMA_VERSION
    assert index["run_count"] == 2
    assert index["benchmarks"] == ["index-benchmark"]
    assert index["datasets"] == ["index-dataset"]
    assert index["metric_ids"] == ["quality"]
    assert index["comparison_identity_statuses"] == {"incomplete": 2}
    assert len(index["comparison_keys"]) == 1
    assert [row["run_id"] for row in index["rows"]] == ["run-a", "run-b"]
    assert index["rows"][0]["metric_ids"] == ["quality"]
    assert index["rows"][0]["artifacts"]["summary"] == "/tmp/run-a/summary.json"
    assert index["rows"][1]["model_id"] == "model-b"
    assert index["issues"] == []


def test_build_run_index_reports_invalid_rows_and_duplicate_run_ids(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="duplicate", model_id="model-a", quality=0.7))
    _write_json(root / "run-b" / "summary.json", _summary(run_id="duplicate", model_id="model-b", quality=0.8))
    _write_json(root / "broken" / "summary.json", {"schema_version": "unknown"})

    index = build_run_index(root, include_invalid=True)

    assert index["run_count"] == 3
    assert any(issue.startswith("skipped ") for issue in index["issues"])
    assert "duplicate run_id: duplicate" in "\n".join(index["issues"])
    assert index["rows"][0]["status"] == "invalid"


def test_write_run_index_outputs_json_and_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    output_json = tmp_path / "index.json"
    output_jsonl = tmp_path / "index.jsonl"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.7))
    _write_json(root / "run-b" / "summary.json", _summary(run_id="run-b", model_id="model-b", quality=0.8))

    index = write_run_index(root, output_json=output_json, output_jsonl=output_jsonl)

    assert index["artifacts"] == {
        "index_json": str(output_json.resolve()),
        "index_jsonl": str(output_jsonl.resolve()),
    }
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["schema_version"] == RUN_INDEX_SCHEMA_VERSION
    jsonl_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["run_id"] for row in jsonl_rows] == ["run-a", "run-b"]


def test_run_browser_html_embeds_filterable_index(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    output_html = tmp_path / "index.html"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.7))

    index = write_run_index(root, output_html=output_html)
    html = output_html.read_text(encoding="utf-8")

    assert index["artifacts"]["index_html"] == str(output_html.resolve())
    assert "WorldFoundry Runs" in html
    assert "worldfoundry-index" in html
    assert "model-a" in build_run_browser_html(index)


def test_load_run_index_jsonl_and_select_comparable_rows(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    output_jsonl = tmp_path / "index.jsonl"
    _write_json(root / "run-a" / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.7))
    _write_json(root / "run-b" / "summary.json", _summary(run_id="run-b", model_id="model-b", quality=0.8))
    _write_json(
        root / "run-c" / "summary.json",
        _summary(run_id="run-c", model_id="model-c", quality=0.9, failed_samples=1),
    )
    write_run_index(root, output_jsonl=output_jsonl)

    index = load_run_index(output_jsonl)
    selected = select_run_index_rows(
        index,
        models=["model-b", "model-c"],
        require_score_valid=True,
        required_metrics=["quality"],
    )
    paths = run_paths_from_index(output_jsonl, models=["model-b"], required_metrics=["quality"])

    assert index["schema_version"] == RUN_INDEX_SCHEMA_VERSION
    assert index["run_count"] == 3
    assert [row["run_id"] for row in selected] == ["run-b"]
    assert paths == [{"path": str((root / "run-b" / "summary.json").resolve()), "label": "run-b"}]
