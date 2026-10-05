from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.framework.official_runner import (
    BenchRunnerConfig,
    build_scorecard,
)


def _config() -> BenchRunnerConfig:
    metric_specs = {
        metric_id: {"name": metric_id, "higher_is_better": True, "group": "test"}
        for metric_id in ("metric_a", "metric_b", "test_average")
    }
    return BenchRunnerConfig(
        benchmark_id="test-benchmark",
        display_name="Test Benchmark",
        root_env="TEST_ROOT",
        results_path_env="TEST_RESULTS",
        default_repo_subdir="",
        metric_order=("metric_a", "metric_b", "test_average"),
        metric_specs=metric_specs,
        metric_aliases={},
        average_metric_id="test_average",
        official_entry="test.module",
    )


def _metric(metric_id: str, score: float) -> dict[str, object]:
    return {
        "metric_id": metric_id,
        "raw_score": score,
        "normalized_score": score,
        "source": "official_runtime",
        "sample_count": 1,
    }


def test_successful_shared_runtime_is_integration_evidence_not_full_suite(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")
    scorecard = build_scorecard(
        config=_config(),
        output_dir=tmp_path / "output",
        results_path=results,
        extracted={"metric_a": _metric("metric_a", 0.5)},
        command=["python", "official_eval.py"],
        duration_seconds=1.0,
        returncode=0,
    )

    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["run"]["status"] == "official_bounded"
    assert scorecard["validation"]["full_suite_complete"] is False
    assert scorecard["metrics"]["per_metric"]["test_average"]["available"] is False


def test_average_requires_every_declared_component(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")
    scorecard = build_scorecard(
        config=_config(),
        output_dir=tmp_path / "output",
        results_path=results,
        extracted={
            "metric_a": _metric("metric_a", 0.4),
            "metric_b": _metric("metric_b", 0.8),
        },
        command=["python", "official_eval.py"],
        duration_seconds=1.0,
        returncode=0,
    )

    average = scorecard["metrics"]["per_metric"]["test_average"]
    assert average["normalized_score"] == pytest.approx(0.6)
    assert average["source"] == "mean_complete_component_metrics"
