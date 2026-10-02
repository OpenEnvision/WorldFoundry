from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.reporting import (
    RUN_COMPARISON_SCHEMA_VERSION,
    build_comparison_identity,
    build_markdown_comparison,
    build_run_comparison,
    write_run_comparison,
)


def _summary(*, run_id: str, model_id: str, quality: float, failed_samples: int = 0) -> dict:
    sample_count = 2
    successful_samples = sample_count - failed_samples
    return {
        "schema_version": "worldfoundry-run-summary",
        "run": {"run_id": run_id, "status": "succeeded" if not failed_samples else "completed_with_failures"},
        "benchmark": {"benchmark_name": "compare-benchmark", "task_type": "compare-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "compare-dataset", "sample_count": sample_count},
        "counts": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": ["failed-sample"] if failed_samples else [],
        },
        "metrics": {
            "leaderboard": {"quality": quality},
            "per_metric": {"quality": {"mean": quality, "higher_is_better": True}},
        },
        "leaderboard": {"quality": quality},
        "eligibility": {"score_valid": failed_samples == 0, "leaderboard_valid": False},
        "artifacts": {"scorecard": f"/tmp/{run_id}/scorecard.json"},
    }


def _scorecard(*, run_id: str, model_id: str, quality: float) -> dict:
    return {
        "schema_version": "worldfoundry-scorecard",
        "run": {"run_id": run_id, "status": "succeeded"},
        "benchmark": {"benchmark_name": "compare-benchmark", "task_type": "compare-task"},
        "model": {"model_id": model_id, "model_name": model_id},
        "dataset": {"dataset_id": "compare-dataset", "sample_count": 2},
        "generation": {"num_requests": 2, "successful": 2, "failed": 0},
        "metrics": {
            "leaderboard": {"quality": quality},
            "per_metric": {"quality": {"mean": quality, "higher_is_better": True}},
            "summary": {
                "sample_count": 2,
                "successful_samples": 2,
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


def _official_summary(*, run_id: str, model_id: str, quality: float, generation: str) -> dict:
    payload = _summary(run_id=run_id, model_id=model_id, quality=quality)
    payload["benchmark"].update(
        {
            "benchmark_id": "compare-benchmark",
            "benchmark_revision": "benchmark-rev",
            "evaluation_protocol": "compare-benchmark:official-run",
            "protocol_revision": "benchmark-rev",
            "protocol_config_hash": "official-config",
        }
    )
    payload["dataset"].update(
        {"dataset_revision": "dataset-rev", "dataset_hash": "dataset-hash", "split": "test"}
    )
    payload["metrics"].update(
        {"metric_revision": "metric-rev", "metric_config_hash": "metric-config"}
    )
    payload["provenance"] = {
        "fidelity": {"generation": generation, "data": "official", "evaluation": "official"}
    }
    payload["evaluation"] = {"kind": "benchmark_model"}
    payload["comparison_identity"] = build_comparison_identity(
        benchmark=payload["benchmark"],
        dataset=payload["dataset"],
        metrics=payload["metrics"],
        provenance=payload["provenance"],
        evaluation_kind="benchmark_model",
    )
    payload["evaluation"]["mode"] = payload["comparison_identity"]["evaluation_mode"]
    return payload


def test_build_run_comparison_reads_summary_dirs_and_scorecard_fallback(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_json(run_a / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.70))
    _write_json(run_b / "scorecard.json", _scorecard(run_id="run-b", model_id="model-b", quality=0.85))

    comparison = build_run_comparison(
        [run_a, run_b],
        labels=["baseline", "candidate"],
        baseline=0,
        metric_ids=["quality", "missing"],
    )

    assert comparison["schema_version"] == RUN_COMPARISON_SCHEMA_VERSION
    assert comparison["run_count"] == 2
    assert comparison["baseline"] == "baseline"
    assert comparison["metric_ids"] == ["quality", "missing"]
    assert comparison["runs"][0]["model_id"] == "model-a"
    assert comparison["runs"][1]["model_id"] == "model-b"
    assert comparison["metrics"]["quality"]["values"] == {"baseline": 0.70, "candidate": 0.85}
    assert comparison["metrics"]["quality"]["deltas"]["candidate"] == pytest.approx(0.15)
    assert comparison["rows"][1]["delta_from_baseline"]["quality"] == pytest.approx(0.15)
    assert comparison["rows"][0]["artifacts"]["scorecard"] == "/tmp/run-a/scorecard.json"
    assert comparison["best_by_metric"]["quality"]["label"] == "candidate"
    assert comparison["issues"] == ["metric not found in any run: missing"]
    assert comparison["compatibility"]["status"] == "compatible_with_incomplete_identity"


def test_build_run_comparison_accepts_reproduction_and_new_model_under_same_official_protocol(
    tmp_path: Path,
) -> None:
    reproduction = tmp_path / "reproduction"
    new_model = tmp_path / "new-model"
    _write_json(
        reproduction / "summary.json",
        _official_summary(run_id="published-cell", model_id="official-model", quality=0.7, generation="pinned"),
    )
    _write_json(
        new_model / "summary.json",
        _official_summary(run_id="new-cell", model_id="new-model", quality=0.8, generation="custom"),
    )

    comparison = build_run_comparison([reproduction, new_model])

    assert comparison["compatibility"]["status"] == "compatible"
    assert [row["evaluation_mode"] for row in comparison["rows"]] == [
        "reproduction",
        "new_model_evaluation",
    ]
    assert comparison["best_by_metric"]["quality"]["model_id"] == "new-model"


def test_build_run_comparison_rejects_different_data_or_protocol(tmp_path: Path) -> None:
    official = tmp_path / "official"
    adapted = tmp_path / "adapted"
    official_payload = _official_summary(
        run_id="official", model_id="model-a", quality=0.7, generation="custom"
    )
    adapted_payload = _official_summary(
        run_id="adapted", model_id="model-b", quality=0.8, generation="custom"
    )
    adapted_payload["dataset"]["dataset_hash"] = "different-data"
    adapted_payload["provenance"]["fidelity"]["evaluation"] = "modified"
    adapted_payload["comparison_identity"] = build_comparison_identity(
        benchmark=adapted_payload["benchmark"],
        dataset=adapted_payload["dataset"],
        metrics=adapted_payload["metrics"],
        provenance=adapted_payload["provenance"],
        evaluation_kind="benchmark_model",
    )
    _write_json(official / "summary.json", official_payload)
    _write_json(adapted / "summary.json", adapted_payload)

    with pytest.raises(ValueError, match="runs are not comparable") as exc_info:
        build_run_comparison([official, adapted])

    assert "protocol_fidelity" in str(exc_info.value)
    assert "dataset_hash" in str(exc_info.value)


def test_write_run_comparison_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    output_json = tmp_path / "comparison.json"
    output_md = tmp_path / "comparison.md"
    _write_json(run_a / "summary.json", _summary(run_id="run-a", model_id="model-a", quality=0.5))
    _write_json(run_b / "summary.json", _summary(run_id="run-b", model_id="model-b", quality=0.75))

    comparison = write_run_comparison(
        [run_a, run_b],
        labels=["model-a", "model-b"],
        baseline="model-a",
        output_json=output_json,
        output_md=output_md,
    )

    assert comparison["metrics"]["quality"]["deltas"]["model-b"] == pytest.approx(0.25)
    assert comparison["artifacts"] == {
        "comparison_json": str(output_json.resolve()),
        "comparison_markdown": str(output_md.resolve()),
    }
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["schema_version"] == RUN_COMPARISON_SCHEMA_VERSION
    assert written["artifacts"]["comparison_json"] == str(output_json.resolve())
    markdown = output_md.read_text(encoding="utf-8")
    assert "# WorldFoundry Run Comparison" in markdown
    assert "quality delta" in markdown
    assert "| model-b | succeeded | compare-benchmark | model-b |  | 2 | 0 | true | false | 0.75 | 0.25 |" in markdown


def test_build_markdown_comparison_handles_no_metrics(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    payload = _summary(run_id="run-a", model_id="model-a", quality=0.5)
    payload["leaderboard"] = {}
    payload["metrics"]["leaderboard"] = {}
    _write_json(run_a / "summary.json", payload)

    comparison = build_run_comparison([run_a])
    markdown = build_markdown_comparison(comparison)

    assert comparison["metric_ids"] == []
    assert "| Run | Status | Benchmark | Model | Backend | Samples | Failed | Score Valid | Leaderboard Valid |" in markdown
