from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.runner import ExistingResultsRunRequest, execute_existing_results


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_existing_results_runner_allows_dataclass_request_overrides(tmp_path: Path) -> None:
    original_dir = tmp_path / "original"
    override_dir = tmp_path / "override"
    request = GenerationRequest(sample_id="sample-a", task_name="unit")
    result_row = GenerationResult(sample_id="sample-a", artifacts={"video": ArtifactRef(uri="a.mp4", kind="video")})

    result = execute_existing_results(
        ExistingResultsRunRequest(
            output_dir=original_dir,
            requests=[request],
            results=[result_row],
        ),
        output_dir=override_dir,
    )

    assert result.output_dir == override_dir
    assert result.status == "succeeded"
    assert not original_dir.exists()
    assert (override_dir / "run_manifest.json").is_file()


def test_existing_results_runner_scores_api_contract_objects(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    requests = [
        GenerationRequest(sample_id="sample-a", task_name="sample_static_i2v"),
        GenerationRequest(sample_id="sample-b", task_name="sample_static_i2v"),
    ]
    results = [
        GenerationResult(
            sample_id="sample-a",
            model_id="cached-model",
            artifacts={"video": ArtifactRef(uri="outputs/a.mp4", kind="video")},
        ),
        GenerationResult(sample_id="sample-b", model_id="cached-model"),
    ]

    def metric(request: GenerationRequest, result: GenerationResult) -> MetricResult | dict:
        if request.sample_id == "sample-a":
            return MetricResult(sample_id=request.sample_id, metric_id="quality", normalized_value=0.5)
        return {"metrics": {"quality": 1.0, "consistency": 0.25}}

    result = execute_existing_results(
        ExistingResultsRunRequest(
            output_dir=output_dir,
            requests=requests,
            results=results,
            metric=metric,
            benchmark={"benchmark_name": "unit"},
            model={"model_name": "cached-model"},
        )
    )

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.sample_count == 2

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
    assert results_rows[0]["artifacts"]["video"]["uri"] == "outputs/a.mp4"

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["failed"] == 0
    assert summary["metrics"]["enabled"] is True
    assert summary["leaderboard"]["quality"] == 0.75
    assert summary["leaderboard"]["consistency"] == 0.25

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["run"]["status"] == "succeeded"
    assert scorecard["metrics"]["leaderboard"]["quality"] == 0.75
    run_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["schema_version"] == "worldfoundry-run-summary"
    assert run_summary["leaderboard"]["quality"] == 0.75
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["schema_version"] == "worldfoundry-environment"
    assert manifest["env_requirements"]["schema_version"] == "worldfoundry-env-requirements"
    assert "preflight" not in manifest
    assert manifest["artifacts"]["environment"].endswith("environment.json")
    assert manifest["artifacts"]["env_requirements"].endswith("env_requirements.json")
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "| quality | 0.75 |" in report


def test_existing_results_runner_loads_jsonl_and_isolates_metric_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    results_path = tmp_path / "existing_results.jsonl"
    rows = [
        {"sample_id": "sample-ok", "model_id": "cached-model", "outputs": {"text": "ok"}},
        {"sample_id": "sample-bad-metric", "model_id": "cached-model"},
        {"sample_id": "sample-generation-failed", "status": "failed", "error": "cached generation failed"},
    ]
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    requests = [
        GenerationRequest(sample_id="sample-ok", task_name="sample_static_i2v"),
        GenerationRequest(sample_id="sample-bad-metric", task_name="sample_static_i2v"),
        GenerationRequest(sample_id="sample-generation-failed", task_name="sample_static_i2v"),
    ]

    def metric(request: GenerationRequest, result: GenerationResult) -> dict:
        if request.sample_id == "sample-bad-metric":
            raise RuntimeError("metric exploded")
        return {"score": 1.0}

    result = execute_existing_results(output_dir=output_dir, requests=requests, results=results_path, metric=metric)

    assert result.status == "completed_with_failures"
    assert result.exit_code == 0
    assert result.successful_sample_count == 1
    assert result.failed_sample_count == 2

    ledger_rows = _read_jsonl(output_dir / "sample_ledger.jsonl")
    ledger = {row["sample_id"]: row for row in ledger_rows}
    assert ledger["sample-ok"]["status"] == "succeeded"
    assert ledger["sample-bad-metric"]["metrics_status"] == "failed"
    assert ledger["sample-bad-metric"]["errors"][0]["message"] == "RuntimeError: metric exploded"
    assert ledger["sample-generation-failed"]["generation_status"] == "failed"
    assert ledger["sample-generation-failed"]["metrics_status"] == "skipped"

    per_sample = {row["sample_id"]: row for row in _read_jsonl(output_dir / "metrics" / "per_sample.jsonl")}
    assert per_sample["sample-bad-metric"]["status"] == "failed"
    assert per_sample["sample-generation-failed"]["status"] == "skipped"

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["failed_sample_ids"] == ["sample-generation-failed"]
    assert summary["metrics"]["failed_sample_ids"] == ["sample-bad-metric"]
    assert summary["metrics"]["skipped_sample_ids"] == ["sample-generation-failed"]
    assert summary["leaderboard"]["score"] == 1.0


def test_existing_results_runner_without_metric_writes_generation_summary_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    requests = [
        GenerationRequest(sample_id="sample-a", task_name="sample_static_i2v"),
        GenerationRequest(sample_id="sample-b", task_name="sample_static_i2v"),
    ]
    results = [
        {"sample_id": "sample-a", "model_id": "cached-model"},
        {"sample_id": "sample-b", "status": "failed", "error": "cached failure"},
    ]

    result = execute_existing_results(output_dir=output_dir, requests=requests, results=results)

    assert result.status == "completed_with_failures"
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["enabled"] is False
    assert summary["leaderboard"] == {}
    assert summary["per_metric"] == {}
    assert summary["generation"]["successful"] == 1
    assert summary["generation"]["failed"] == 1

    per_sample = _read_jsonl(output_dir / "metrics" / "per_sample.jsonl")
    assert per_sample[0]["status"] == "not_run"
    assert per_sample[1]["status"] == "skipped"
