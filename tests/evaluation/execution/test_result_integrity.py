# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free regressions for result identity, finite scores, and generation progress."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.evaluation.api import AggregateResult, ArtifactRef, GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.api.metrics import aggregate_mean
from worldfoundry.evaluation.tasks.execution.orchestration.contract import run_contract
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import run_evaluate
from worldfoundry.evaluation.tasks.execution.orchestration.existing_results import run_existing_results
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark_suite import _SuiteGenerationMemoRunner
from worldfoundry.evaluation.tasks.execution.orchestration.run_session import coerce_generation_result

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def fixed_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    version = {"available": False, "commit": None, "dirty": None, "dirty_diff_sha256": None}
    monkeypatch.setattr("worldfoundry.evaluation.utils.git_metadata", lambda *args: version)
    monkeypatch.setattr("worldfoundry.evaluation.reporting.run_manifest.git_metadata", lambda *args: version)


def requests() -> list[GenerationRequest]:
    return [
        GenerationRequest(sample_id=key, task_name="integrity", generation_kwargs={"seed": index + 1})
        for index, key in enumerate(("a", "b"))
    ]


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class BatchRunner:
    model_id = "integrity"
    version = "1"
    parameters: dict[str, Any] = {}

    def __init__(self, outputs: Sequence[Any]) -> None:
        self.outputs = outputs
        self.batches: list[list[str]] = []
        self.cleaned = False

    def generate(self, rows: Sequence[GenerationRequest]) -> Sequence[Any]:
        self.batches.append([row.sample_id for row in rows])
        return self.outputs

    def cleanup(self) -> None:
        self.cleaned = True


def online(kind: str, root: Path, rows: Sequence[GenerationRequest], runner: Any, **kwargs: Any) -> Any:
    if kind == "contract":
        return run_contract(output_dir=root, requests=rows, runner=runner, **kwargs)
    return run_evaluate(output_dir=root, mode="model", requests=rows, runner=runner, **kwargs)


@pytest.mark.parametrize("kind", ["offline", "contract", "model"])
def test_duplicate_requests_fail_before_generation(tmp_path: Path, kind: str) -> None:
    rows = [requests()[0], requests()[0]]
    runner = BatchRunner([])
    with pytest.raises(ValueError, match="duplicate sample_id 'a' in requests: rows 1 and 2"):
        if kind == "offline":
            run_existing_results(output_dir=tmp_path, requests=rows, results=[])
        else:
            online(kind, tmp_path, rows, runner)
    assert runner.batches == []
    assert not (tmp_path / "scorecard.json").exists()


@pytest.mark.parametrize("kind", ["offline", "contract", "model"])
def test_duplicate_results_cannot_overwrite_scores(tmp_path: Path, kind: str) -> None:
    outputs = [{"sample_id": "a", "status": "succeeded", "scores": {"quality": value}} for value in (1, 9)]
    with pytest.raises(ValueError, match="duplicate sample_id 'a'.*rows 1 and 2"):
        if kind == "offline":
            run_evaluate(output_dir=tmp_path, requests=requests(), results=outputs, metrics=["numeric:quality"])
        else:
            online(kind, tmp_path, requests(), BatchRunner(outputs))
    assert not (tmp_path / "scorecard.json").exists()


def test_multi_seed_samples_remain_distinct(tmp_path: Path) -> None:
    rows = [
        GenerationRequest(
            sample_id=f"prompt-seed-{seed}",
            task_name="integrity",
            inputs={"prompt": "cube", "prompt_id": "prompt"},
            generation_kwargs={"seed": seed},
        )
        for seed in (1, 2)
    ]
    outputs = [
        GenerationResult(sample_id=row.sample_id, metadata={"prompt_id": "prompt", "scores": {"quality": value}})
        for row, value in zip(rows, (1, 9))
    ]
    result = run_evaluate(output_dir=tmp_path, requests=rows, results=outputs, metrics=["numeric:quality"])
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert result.successful_sample_count == 2
    assert summary["per_metric"]["quality"]["mean"] == 5
    assert summary["per_metric"]["quality"]["n_valid"] == 2


@pytest.mark.parametrize("status", [None, "", "unknown", "succeeded"])
def test_imported_status_matches_generation_contract(tmp_path: Path, status: str | None) -> None:
    output = {"sample_id": "a", "scores": {"quality": 9}}
    if status is not None:
        output["status"] = status
    expected = GenerationResult.from_dict(output).status
    assert coerce_generation_result(output, "a").status == expected
    result = run_evaluate(output_dir=tmp_path, requests=requests()[:1], results=[output], metrics=["numeric:quality"])
    assert result.successful_sample_count == int(status == "succeeded")
    scorecard = json.loads((tmp_path / "scorecard.json").read_text())
    assert scorecard["eligibility"]["score_valid"] is (status == "succeeded")


@pytest.mark.parametrize("kind", ["contract", "model"])
def test_in_process_mapping_can_report_completion_without_status(tmp_path: Path, kind: str) -> None:
    outputs = [{"sample_id": row.sample_id, "scores": {"quality": 1}} for row in requests()]
    result = online(kind, tmp_path, requests(), BatchRunner(outputs))
    assert result.successful_sample_count == 2
    assert {row["status"] for row in jsonl(tmp_path / "results.jsonl")} == {"succeeded"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "field", ["raw_value", "normalized_value", "components", "coverage", "diagnostics", "judge_trace"]
)
def test_nonfinite_metric_fields_are_invalid_and_serializable(value: float, field: str) -> None:
    payload: dict[str, Any] = {"raw_value": 0.5}
    payload[field] = {"nested": [1, value]} if field in {"components", "diagnostics", "judge_trace"} else value
    row = MetricResult(sample_id="a", metric_id="quality", **payload)
    assert not row.valid
    assert row.coverage == 0
    assert row.skip_reason == "nonfinite_metric_output"
    assert any(path.startswith(field) for path in row.diagnostics["nonfinite_values"])
    json.dumps(row.to_dict(), allow_nan=False)
    aggregate = aggregate_mean("quality", [row, MetricResult(sample_id="b", metric_id="quality", raw_value=0.9)])
    assert aggregate.valid
    assert aggregate.n_valid == 1
    assert aggregate.raw_stats["mean"] == 0.9


@pytest.mark.parametrize("kind", ["numpy_scalar", "numpy_array", "torch_scalar", "torch_array"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_runtime_numbers_are_normalized_before_serialization(kind: str, value: float) -> None:
    if kind.startswith("numpy"):
        import numpy as np

        number = np.float32(value) if kind.endswith("scalar") else np.array([0.5, value])
    else:
        import torch

        number = torch.tensor(value if kind.endswith("scalar") else [0.5, value])
    row = MetricResult(sample_id="a", metric_id="quality", raw_value=number)
    assert not row.valid
    assert row.skip_reason == "nonfinite_metric_output"
    json.dumps(row.to_dict(), allow_nan=False)


class ValueMetric:
    version = "1"
    parameters: dict[str, Any] = {}
    higher_is_better = True

    def __init__(self, name: str, values: dict[str, float]) -> None:
        self.name = name
        self.values = values
        self.calls: list[str] = []

    def compute_batch(
        self, rows: Sequence[GenerationRequest], outputs: Sequence[GenerationResult]
    ) -> list[dict[str, Any]]:
        self.calls.extend(row.sample_id for row in rows)
        return [{"metric_id": self.name, "raw_value": self.values[row.sample_id]} for row in rows]

    def compute_sample(self, row: GenerationRequest, output: GenerationResult) -> dict[str, Any]:
        self.calls.append(row.sample_id)
        return {self.name: self.values[row.sample_id]}

    def aggregate(self, rows: Sequence[MetricResult]) -> AggregateResult:
        return aggregate_mean(self.name, rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_preserve_healthy_metrics_and_checkpoints(tmp_path: Path, value: float) -> None:
    outputs = [GenerationResult(sample_id=row.sample_id) for row in requests()]
    bad = ValueMetric("bad", {"a": value, "b": 0.9})
    good = ValueMetric("good", {"a": 0.7, "b": 0.8})
    result = run_evaluate(
        output_dir=tmp_path, requests=requests(), results=outputs, metrics=[bad, good], metric_batch_size=2
    )
    assert result.scorecard_path.is_file()
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert summary["per_metric"]["bad"]["n_valid"] == 1
    assert summary["per_metric"]["bad"]["mean"] == 0.9
    assert summary["per_metric"]["good"]["mean"] == pytest.approx(0.75)
    assert summary["per_metric"]["bad"]["skip_breakdown"] == {"nonfinite_metric_output": 1}
    per_sample = jsonl(tmp_path / "metrics/per_sample.jsonl")
    assert per_sample[0]["status"] == "failed"
    assert per_sample[0]["metrics"] == {"good": 0.7}
    assert "nonfinite" in per_sample[0]["error"]
    bad.calls.clear()
    good.calls.clear()
    run_evaluate(output_dir=tmp_path, requests=requests(), results=outputs, metrics=[bad, good], resume=True)
    assert good.calls == []
    assert bad.calls == ["a"]


def test_nonfinite_custom_aggregate_is_invalid_without_losing_report(tmp_path: Path) -> None:
    class BadAggregate(ValueMetric):
        def aggregate(self, rows: Sequence[MetricResult]) -> AggregateResult:
            return AggregateResult(
                metric_id=self.name,
                n_total=len(rows),
                n_valid=len(rows),
                raw_stats={"mean": float("nan")},
                stderr=float("inf"),
            )

    bad = BadAggregate("bad", {"a": 0.5, "b": 0.9})
    good = ValueMetric("good", {"a": 0.7, "b": 0.8})
    run_evaluate(
        output_dir=tmp_path,
        requests=requests(),
        results=[GenerationResult(sample_id=q.sample_id) for q in requests()],
        metrics=[bad, good],
    )
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert not summary["per_metric"]["bad"]["valid"]
    assert summary["per_metric"]["bad"]["raw_stats"]["mean"] is None
    assert summary["leaderboard"] == {"good": pytest.approx(0.75)}
    assert not json.loads((tmp_path / "scorecard.json").read_text())["eligibility"]["score_valid"]


class StreamingRunner(BatchRunner):
    def __init__(self, root: Path, failure: type[BaseException] | None = None) -> None:
        super().__init__([])
        self.root = root
        self.failure = failure

    def generate_iter(self, rows: Sequence[GenerationRequest]) -> Iterator[GenerationResult]:
        self.batches.append([row.sample_id for row in rows])
        for index, row in enumerate(rows):
            if index == 1 and self.failure:
                # The first sample must already be flushed, before the next unit starts.
                assert len(jsonl(self.root / "results.jsonl")) == 1
                raise self.failure("interrupted generation")
            artifact = self.root / f"{row.sample_id}.txt"
            artifact.write_text("complete rollout", encoding="utf-8")
            yield GenerationResult(
                sample_id=row.sample_id,
                model_id=self.model_id,
                artifacts={"text": ArtifactRef(uri=str(artifact), kind="text")},
            )


@pytest.mark.parametrize("kind", ["contract", "model"])
@pytest.mark.parametrize("cache_mode", ["off", "read-write"])
def test_generation_interrupt_preserves_completed_unit_and_resumes(tmp_path: Path, kind: str, cache_mode: str) -> None:
    root = tmp_path / "run"
    first = StreamingRunner(root, KeyboardInterrupt)
    options = {"generation_cache_dir": tmp_path / "cache", "generation_cache_mode": cache_mode}
    with pytest.raises(KeyboardInterrupt):
        online(kind, root, requests(), first, **options)
    assert first.batches == [["a", "b"]]
    assert first.cleaned
    assert [row["sample_id"] for row in jsonl(root / "results.jsonl")] == ["a"]
    assert json.loads((root / "run_manifest.json").read_text())["status"] == "interrupted"
    if cache_mode != "off":
        with sqlite3.connect(tmp_path / "cache/cache.db") as connection:
            assert connection.execute("SELECT count(*) FROM generation_results").fetchone()[0] == 1
    second = StreamingRunner(root)
    result = online(kind, root, requests(), second, resume=True, **options)
    assert second.batches == [["b"]]
    assert result.successful_sample_count == 2
    assert [row["sample_id"] for row in jsonl(root / "results.jsonl")] == ["a", "b"]


@pytest.mark.parametrize("kind", ["contract", "model"])
def test_streaming_runner_error_retains_completed_sample(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "run"
    result = online(kind, root, requests(), StreamingRunner(root, RuntimeError))
    assert result.successful_sample_count == 1
    assert result.failed_sample_count == 1
    assert [row["status"] for row in jsonl(root / "results.jsonl")] == ["succeeded", "failed"]


def test_suite_memo_forwards_progress_without_caching_partial_batch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    runner = StreamingRunner(root, KeyboardInterrupt)
    memo = _SuiteGenerationMemoRunner(runner)
    with pytest.raises(KeyboardInterrupt):
        online("model", root, requests(), memo)
    assert memo._batches == {}
    assert len(jsonl(root / "results.jsonl")) == 1
    runner.failure = None
    list(memo.generate_iter(requests()))
    list(memo.generate_iter(requests()))
    assert memo.stats()["batch_hits"] == 1
    assert runner.batches == [["a", "b"], ["a", "b"]]


def test_pipeline_runner_checkpoints_one_complete_unit_per_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner

    root = tmp_path / "run"
    runner = WorldFoundryPipelineRunner("integrity", object(), pipeline_target="tests:Pipeline")
    lifecycle = object()
    lifecycles: list[object] = []
    monkeypatch.setattr(runner, "_runtime_profile", lambda: object())
    monkeypatch.setattr(runner, "_lifecycle", lambda profile: lifecycles.append(lifecycle) or lifecycle)

    def generate_one(row: GenerationRequest, current: object) -> GenerationResult:
        assert current is lifecycle
        if row.sample_id == "b":
            assert jsonl(root / "results.jsonl")[0]["sample_id"] == "a"
            raise KeyboardInterrupt("interrupted generation")
        return GenerationResult(sample_id=row.sample_id)

    monkeypatch.setattr(runner, "_generate_one", generate_one)
    with pytest.raises(KeyboardInterrupt):
        online("model", root, requests(), runner)
    assert len(lifecycles) == 1
    assert runner.cleaned
    assert len(jsonl(root / "results.jsonl")) == 1


def test_duplicate_streamed_results_fail_without_scorecard(tmp_path: Path) -> None:
    class DuplicateRunner(BatchRunner):
        def generate_iter(self, rows: Sequence[GenerationRequest]) -> Iterator[GenerationResult]:
            yield GenerationResult(sample_id="a")
            yield GenerationResult(sample_id="a")

    with pytest.raises(ValueError, match="duplicate sample_id 'a'.*rows 1 and 2"):
        online("model", tmp_path, requests(), DuplicateRunner([]))
    assert not (tmp_path / "scorecard.json").exists()


def test_imported_missing_status_with_error_remains_unknown() -> None:
    output = {"sample_id": "a", "error": "incomplete record"}
    assert coerce_generation_result(output, "a").status == GenerationResult.from_dict(output).status == "unknown"
