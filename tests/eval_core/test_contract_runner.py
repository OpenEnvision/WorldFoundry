from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.api import (
    AggregateResult,
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    MetricResult,
)
from worldfoundry.evaluation.runner import ContractRunRequest, execute_contract_run


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeRunner:
    model_id = "local-contract-model"
    capabilities = {"text-to-video"}

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.cleaned = False

    def generate(self, requests):
        results = []
        for request in requests:
            artifact = self.output_dir / f"{request.sample_id}.mp4"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"video-{request.sample_id}".encode("utf-8"))
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    model_id=self.model_id,
                    artifacts={"generated_video": ArtifactRef(uri=artifact.name, kind="video")},
                )
            )
        return results

    def cleanup(self) -> None:
        self.cleaned = True


class RecordingRunner:
    model_id = "recording-contract-model"
    capabilities = {"text-to-video"}

    def __init__(self, output_dir: Path, failing_sample_ids=()) -> None:
        self.output_dir = output_dir
        self.failing_sample_ids = set(failing_sample_ids)
        self.calls = []
        self.cleaned = False

    def generate(self, requests):
        request_rows = list(requests)
        self.calls.append([request.sample_id for request in request_rows])
        results = []
        for request in request_rows:
            if request.sample_id in self.failing_sample_ids:
                results.append(
                    GenerationResult(
                        sample_id=request.sample_id,
                        request_id=request.request_id,
                        model_id=self.model_id,
                        status="failed",
                        error="planned failure",
                    )
                )
                continue
            artifact = self.output_dir / f"{request.sample_id}.mp4"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"video-{request.sample_id}".encode("utf-8"))
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    model_id=self.model_id,
                    artifacts={"generated_video": ArtifactRef(uri=artifact.name, kind="video")},
                )
            )
        return results

    def cleanup(self) -> None:
        self.cleaned = True


class QualityMetric:
    name = "quality"
    version = "1"
    required_artifacts = ("generated_video",)
    higher_is_better = False

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> MetricResult:
        score = 0.8 if request.sample_id == "sample-a" else 0.4
        return MetricResult(
            sample_id=request.sample_id,
            metric_id=self.name,
            raw_value=score * 10,
            normalized_value=score,
        )

    def aggregate(self, results) -> AggregateResult:
        values = [float(result.normalized_value) for result in results if result.normalized_value is not None]
        raw_values = [float(result.raw_value) for result in results if isinstance(result.raw_value, (int, float))]
        return AggregateResult(
            metric_id=self.name,
            n_total=len(results),
            n_valid=len(values),
            n_skipped=len(results) - len(values),
            normalized_stats={"mean": sum(values) / len(values)},
            raw_stats={"mean": sum(raw_values) / len(raw_values)},
        )


def _contract_requests() -> list[GenerationRequest]:
    return [
        GenerationRequest(sample_id="sample-a", task_name="contract_t2v"),
        GenerationRequest(sample_id="sample-b", task_name="contract_t2v"),
    ]


def test_contract_runner_allows_dataclass_request_overrides(tmp_path: Path) -> None:
    original_dir = tmp_path / "original"
    override_dir = tmp_path / "override"

    result = execute_contract_run(
        ContractRunRequest(
            output_dir=original_dir,
            requests=[GenerationRequest(sample_id="sample-a", task_name="unit")],
            runner=FakeRunner(override_dir),
        ),
        output_dir=override_dir,
    )

    assert result.output_dir == override_dir
    assert result.status == "succeeded"
    assert not original_dir.exists()
    assert (override_dir / "run_manifest.json").is_file()


def _assert_summary_scorecard_and_ledger_consistent(output_dir: Path) -> None:
    ledger = _read_jsonl(output_dir / "sample_ledger.jsonl")
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    succeeded = [row["sample_id"] for row in ledger if row["status"] == "succeeded"]
    failed = [row["sample_id"] for row in ledger if row["status"] != "succeeded"]

    assert summary["sample_count"] == len(ledger)
    assert summary["successful_samples"] == len(succeeded)
    assert summary["failed_samples"] == len(failed)
    assert sorted(summary["failed_sample_ids"]) == sorted(failed)
    assert scorecard["evaluation"]["num_results"] == len(ledger)
    assert scorecard["evaluation"]["successful_samples"] == len(succeeded)
    assert scorecard["evaluation"]["errored_samples"] == len(failed)
    assert scorecard["metrics"]["summary"]["successful_samples"] == len(succeeded)


def test_contract_runner_executes_model_metric_and_writes_scorecard(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    runner = FakeRunner(output_dir)

    result = execute_contract_run(
        ContractRunRequest(
            output_dir=output_dir,
            requests=[
                GenerationRequest(sample_id="sample-a", task_name="contract_t2v"),
                GenerationRequest(sample_id="sample-b", task_name="contract_t2v"),
            ],
            runner=runner,
            metrics=[QualityMetric()],
            benchmark={"benchmark_name": "contract-unit"},
        )
    )

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.sample_count == 2
    assert runner.cleaned is True

    for relative_path in [
        "run_manifest.json",
        "environment.json",
        "env_requirements.json",
        "execution_plan.json",
        "requests.jsonl",
        "results.jsonl",
        "artifacts.jsonl",
        "sample_ledger.jsonl",
        "metrics/per_sample.jsonl",
        "metrics/summary.json",
        "summary.json",
        "report.md",
        "scorecard.json",
    ]:
        assert (output_dir / relative_path).exists()

    results_rows = _read_jsonl(output_dir / "results.jsonl")
    assert results_rows[0]["schema_version"] == "worldfoundry-generation-result"

    artifact_row = _read_jsonl(output_dir / "artifacts.jsonl")[0]
    assert artifact_row["schema_version"] == "worldfoundry-artifact-ref"
    assert artifact_row["name"] == "generated_video"
    assert artifact_row["kind"] == "video"
    assert artifact_row["size_bytes"] == len(b"video-sample-a")
    assert artifact_row["sha256"]

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["leaderboard"]["quality"] == pytest.approx(0.6)
    assert summary["per_metric"]["quality"]["higher_is_better"] is False
    assert summary["per_metric"]["quality"]["normalized_stats"]["mean"] == pytest.approx(0.6)
    assert summary["per_metric"]["quality"]["raw_stats"]["mean"] == 6.0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["evaluation"]["kind"] == "contract_runner"
    assert scorecard["metrics"]["leaderboard"]["quality"] == pytest.approx(0.6)
    run_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["schema_version"] == "worldfoundry-run-summary"
    assert run_summary["leaderboard"]["quality"] == pytest.approx(0.6)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["schema_version"] == "worldfoundry-environment"
    assert manifest["env_requirements"]["schema_version"] == "worldfoundry-env-requirements"
    assert "preflight" not in manifest
    assert manifest["artifacts"]["environment"].endswith("environment.json")
    assert manifest["artifacts"]["env_requirements"].endswith("env_requirements.json")
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "# WorldFoundry Run Report" in report
    assert "| quality | 0.6 |" in report


def test_contract_runner_default_run_does_not_reuse_successful_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    first_runner = RecordingRunner(output_dir)
    execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=first_runner,
        metrics=[QualityMetric()],
    )

    second_runner = RecordingRunner(output_dir)
    execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=second_runner,
        metrics=[QualityMetric()],
    )

    assert first_runner.calls == [["sample-a", "sample-b"]]
    assert second_runner.calls == [["sample-a", "sample-b"]]
    ledger = _read_jsonl(output_dir / "sample_ledger.jsonl")
    assert all("cached" not in row for row in ledger)


def test_contract_runner_resume_reuses_successful_samples_without_runner_call(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=RecordingRunner(output_dir),
        metrics=[QualityMetric()],
    )

    runner = RecordingRunner(output_dir)
    result = execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=runner,
        metrics=[QualityMetric()],
        resume=True,
    )

    assert result.status == "succeeded"
    assert runner.calls == []
    ledger = _read_jsonl(output_dir / "sample_ledger.jsonl")
    assert [row["sample_id"] for row in ledger] == ["sample-a", "sample-b"]
    assert all(row["cached"] is True for row in ledger)
    _assert_summary_scorecard_and_ledger_consistent(output_dir)


def test_contract_runner_resume_reruns_when_metric_contract_changes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=RecordingRunner(output_dir),
        metrics=[QualityMetric()],
    )

    runner = RecordingRunner(output_dir)
    result = execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=runner,
        metrics=[],
        resume=True,
    )

    assert result.status == "succeeded"
    assert runner.calls == [["sample-a", "sample-b"]]
    ledger = _read_jsonl(output_dir / "sample_ledger.jsonl")
    assert all("cached" not in row for row in ledger)
    per_sample = _read_jsonl(output_dir / "metrics" / "per_sample.jsonl")
    assert {row["status"] for row in per_sample} == {"not_run"}
    _assert_summary_scorecard_and_ledger_consistent(output_dir)


def test_contract_runner_resume_reruns_failed_samples_and_reuses_successes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    first_result = execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=RecordingRunner(output_dir, failing_sample_ids={"sample-b"}),
        metrics=[QualityMetric()],
    )
    assert first_result.status == "completed_with_failures"

    runner = RecordingRunner(output_dir)
    result = execute_contract_run(
        output_dir=output_dir,
        requests=_contract_requests(),
        runner=runner,
        metrics=[QualityMetric()],
        resume=True,
    )

    assert result.status == "succeeded"
    assert result.successful_sample_count == 2
    assert result.failed_sample_count == 0
    assert runner.calls == [["sample-b"]]

    ledger = _read_jsonl(output_dir / "sample_ledger.jsonl")
    by_sample_id = {row["sample_id"]: row for row in ledger}
    assert by_sample_id["sample-a"]["cached"] is True
    assert "cached" not in by_sample_id["sample-b"]
    assert by_sample_id["sample-b"]["status"] == "succeeded"
    _assert_summary_scorecard_and_ledger_consistent(output_dir)


def test_contract_runner_generation_cache_reuses_outputs_across_run_dirs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "generation-cache"
    first_output_dir = tmp_path / "first-run"
    second_output_dir = tmp_path / "second-run"

    first_runner = RecordingRunner(first_output_dir)
    first_result = execute_contract_run(
        output_dir=first_output_dir,
        requests=_contract_requests(),
        runner=first_runner,
        metrics=[QualityMetric()],
        generation_cache_dir=cache_dir,
        generation_cache_mode="read-write",
    )
    assert first_result.status == "succeeded"
    assert first_runner.calls == [["sample-a", "sample-b"]]

    second_runner = RecordingRunner(second_output_dir)
    second_result = execute_contract_run(
        output_dir=second_output_dir,
        requests=_contract_requests(),
        runner=second_runner,
        metrics=[QualityMetric()],
        generation_cache_dir=cache_dir,
        generation_cache_mode="read-write",
    )

    assert second_result.status == "succeeded"
    assert second_runner.calls == []
    manifest = json.loads((second_output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_cache"]["hits"] == 2
    assert manifest["cache_paths"]["generation_result_cache"].endswith("cache.db")
    ledger = _read_jsonl(second_output_dir / "sample_ledger.jsonl")
    assert {row["cache_source"] for row in ledger} == {"generation_result_cache"}
    artifacts = _read_jsonl(second_output_dir / "artifacts.jsonl")
    assert all(Path(row["uri"]).is_file() for row in artifacts)


class FailureRunner:
    model_id = "failure-model"
    capabilities = {"text-to-video"}

    def generate(self, requests):
        return [
            GenerationResult(sample_id="sample-ok", model_id=self.model_id),
            GenerationResult(sample_id="sample-generation-failed", status="failed", error="generation exploded"),
            GenerationResult(sample_id="sample-metric-failed", model_id=self.model_id),
        ]

    def cleanup(self) -> None:
        return None


class FailingMetric:
    name = "quality"
    version = "1"
    required_artifacts = ()
    higher_is_better = True

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> MetricResult:
        if request.sample_id == "sample-metric-failed":
            raise RuntimeError("metric exploded")
        return MetricResult(sample_id=request.sample_id, metric_id=self.name, normalized_value=1.0)

    def aggregate(self, results) -> AggregateResult:
        values = [float(result.normalized_value) for result in results if result.normalized_value is not None]
        return AggregateResult(
            metric_id=self.name,
            n_total=len(results),
            n_valid=len(values),
            normalized_stats={"mean": sum(values) / len(values)} if values else {},
            valid=bool(values),
        )


class RequestPlanRunner:
    model_id = "request-plan-model"
    capabilities = {"text-to-video"}

    def generate(self, requests):
        return [
            GenerationResult(
                sample_id=request.sample_id,
                model_id=self.model_id,
                status="request_plan",
                error="request plan only",
                metadata={"runtime_status": "request_plan"},
            )
            for request in requests
        ]

    def cleanup(self) -> None:
        return None


class RecordingMetric:
    name = "quality"
    version = "1"
    required_artifacts = ()
    higher_is_better = True

    def __init__(self) -> None:
        self.calls = []

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> MetricResult:
        self.calls.append((request.sample_id, result.status))
        return MetricResult(sample_id=request.sample_id, metric_id=self.name, normalized_value=1.0)

    def aggregate(self, results) -> AggregateResult:
        values = [float(result.normalized_value) for result in results if result.normalized_value is not None]
        return AggregateResult(
            metric_id=self.name,
            n_total=len(results),
            n_valid=len(values),
            normalized_stats={"mean": sum(values) / len(values)} if values else {},
            valid=bool(values),
        )


def test_contract_runner_records_generation_and_metric_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    result = execute_contract_run(
        output_dir=output_dir,
        requests=[
            GenerationRequest(sample_id="sample-ok", task_name="contract_t2v"),
            GenerationRequest(sample_id="sample-generation-failed", task_name="contract_t2v"),
            GenerationRequest(sample_id="sample-metric-failed", task_name="contract_t2v"),
        ],
        runner=FailureRunner(),
        metrics=[FailingMetric()],
    )

    assert result.status == "completed_with_failures"
    assert result.exit_code == 0
    assert result.successful_sample_count == 1
    assert result.failed_sample_count == 2

    ledger = {row["sample_id"]: row for row in _read_jsonl(output_dir / "sample_ledger.jsonl")}
    assert ledger["sample-ok"]["status"] == "succeeded"
    assert ledger["sample-generation-failed"]["generation_status"] == "failed"
    assert ledger["sample-generation-failed"]["metrics_status"] == "skipped"
    assert ledger["sample-metric-failed"]["generation_status"] == "succeeded"
    assert ledger["sample-metric-failed"]["metrics_status"] == "failed"

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["failed_sample_ids"] == ["sample-generation-failed"]
    assert summary["metrics"]["failed_sample_ids"] == ["sample-metric-failed"]
    assert summary["metrics"]["skipped_sample_ids"] == ["sample-generation-failed"]
    assert summary["leaderboard"]["quality"] == 1.0


def test_contract_runner_does_not_score_request_plan_results(tmp_path: Path) -> None:
    metric = RecordingMetric()
    output_dir = tmp_path / "run"

    result = execute_contract_run(
        output_dir=output_dir,
        requests=[GenerationRequest(sample_id="sample-plan", task_name="contract_t2v")],
        runner=RequestPlanRunner(),
        metrics=[metric],
    )

    ledger = {row["sample_id"]: row for row in _read_jsonl(output_dir / "sample_ledger.jsonl")}
    results = {row["sample_id"]: row for row in _read_jsonl(output_dir / "results.jsonl")}
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))

    assert result.status == "completed_with_failures"
    assert result.successful_sample_count == 0
    assert result.failed_sample_count == 1
    assert metric.calls == []
    assert results["sample-plan"]["status"] == "request_plan"
    assert ledger["sample-plan"]["generation_status"] == "failed"
    assert ledger["sample-plan"]["metrics_status"] == "skipped"
    assert summary["generation"]["failed_sample_ids"] == ["sample-plan"]
