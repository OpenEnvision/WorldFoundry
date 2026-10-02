from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET
from worldfoundry.evaluation.runner import (
    EVALUATE_RUN_RESULT_SCHEMA_VERSION,
    EvaluateRunRequest,
    execute_evaluate_run,
    materialize_generation_requests,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_evaluate_runner_scores_materialized_results_without_requests(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "run"
    results = [
        {
            "sample_id": "sample-a",
            "task_name": "offline_t2v",
            "model_id": "offline-video-model",
            "artifacts": {"video": {"uri": "outputs/a.mp4", "kind": "video"}},
            "metrics": {"quality": 0.8},
        },
        {
            "sample_id": "sample-b",
            "task_name": "offline_t2v",
            "model_id": "offline-video-model",
            "metrics": {"quality": 0.4},
        },
    ]
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            results_path=results_path,
            metrics=("artifact_count", "has_artifact:video", "numeric:quality"),
            required_artifacts=("video",),
            benchmark_id="offline-vbench-smoke",
            model_id="offline-video-model",
            dataset_id="offline-fixture",
            run_id="offline-run",
        )
    )

    assert result.schema_version == EVALUATE_RUN_RESULT_SCHEMA_VERSION
    assert result.mode == "existing-results"
    assert result.delegate_runner == "ExistingResultsRunner"
    assert result.status == "succeeded"
    assert result.sample_count == 2
    assert result.artifact_count == 1

    for relative_path in [
        "run_manifest.json",
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
        assert (output_dir / relative_path).is_file()

    requests = _read_jsonl(output_dir / "requests.jsonl")
    assert [row["sample_id"] for row in requests] == ["sample-a", "sample-b"]
    assert [row["task_name"] for row in requests] == ["offline_t2v", "offline_t2v"]

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["leaderboard"]["artifact_count"] == 0.5
    assert summary["leaderboard"]["has_artifact:video"] == 0.5
    assert summary["leaderboard"]["required_artifacts_present"] == 0.5
    assert summary["leaderboard"]["quality"] == pytest.approx(0.6)
    assert summary["leaderboard"]["generation_success"] == 1.0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["run"]["run_id"] == "offline-run"
    assert scorecard["benchmark"]["benchmark_name"] == "offline-vbench-smoke"
    assert scorecard["model"]["model_id"] == "offline-video-model"
    assert scorecard["dataset"]["dataset_id"] == "offline-fixture"
    assert scorecard["metrics"]["leaderboard"]["quality"] == pytest.approx(0.6)
    run_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["leaderboard"]["quality"] == pytest.approx(0.6)
    assert "# WorldFoundry Run Report" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_evaluate_runner_model_scores_contract_runner(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            mode="model",
            model_id=CONTRACT_FIXTURE_MODEL_ID,
            model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
            requests=[
                {
                    "sample_id": "sample-a",
                    "task_name": "dry_t2v",
                    "inputs": {"prompt": "x"},
                    "output_schema": {"generated_video": {"kind": "generated_video"}},
                }
            ],
            metrics=("artifact_count",),
            benchmark_id="dry-benchmark",
        )
    )

    assert result.mode == "model"
    assert result.delegate_runner == "ResolvedWorldModelRunner+ExistingResultsRunner"
    assert result.status == "succeeded"
    assert result.sample_count == 1

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runner"] == "existing_results_runner"
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["benchmark"]["benchmark_name"] == "dry-benchmark"
    assert scorecard["model"]["model_id"] == CONTRACT_FIXTURE_MODEL_ID


def test_evaluate_runner_existing_results_defaults_to_artifact_count(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    result = execute_evaluate_run(
        output_dir=output_dir,
        results=[
            {
                "sample_id": "sample-a",
                "artifacts": {"video": {"uri": "memory://sample-a.mp4", "kind": "video"}},
            },
            {"sample_id": "sample-b"},
        ],
    )

    assert result.status == "succeeded"
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["leaderboard"]["artifact_count"] == 0.5


def test_evaluate_runner_model_mode_resolves_contract_runner_and_scores_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    requests = materialize_generation_requests(
        [
            {
                "sample_id": "sample-a",
                "initial_context": {"generation_text": "Move through a small room."},
                "expected_outputs": {"generated_video": ""},
            }
        ],
        task_name="offline_i2v",
    )

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            mode="model",
            requests=requests,
            model_id=CONTRACT_FIXTURE_MODEL_ID,
            model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
            model_parameters={"artifact_uri_template": "memory://{sample_id}.mp4"},
            metrics=("artifact_count", "has_artifact:generated_video"),
            run_id="model-mode-run",
        )
    )

    assert result.mode == "model"
    assert result.delegate_runner == "ResolvedWorldModelRunner+ExistingResultsRunner"
    assert result.status == "succeeded"
    assert result.artifact_count == 1

    results = _read_jsonl(output_dir / "results.jsonl")
    assert results[0]["model_id"] == CONTRACT_FIXTURE_MODEL_ID
    assert results[0]["artifacts"]["generated_video"]["uri"] == "memory://sample-a.mp4"

    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["leaderboard"]["artifact_count"] == 1.0
    assert summary["leaderboard"]["has_artifact:generated_video"] == 1.0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["run"]["run_id"] == "model-mode-run"
    assert scorecard["model"]["model_id"] == CONTRACT_FIXTURE_MODEL_ID
    assert scorecard["model"]["resolver"]["source"] == "runner_target"
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.md").is_file()


def test_evaluate_runner_model_mode_resolves_model_zoo_runner(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "zoo-model",
                        "source": {"status": "open_source"},
                        "integration_status": "integrated",
                        "runner_target": "tests.eval_core.contract_fixture:ContractFixtureRunner",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            mode="model",
            requests=[
                {
                    "sample_id": "sample-a",
                    "task_name": "zoo_t2v",
                    "inputs": {"prompt": "A camera moves through a kitchen."},
                    "output_schema": {"generated_video": {"type": "video"}},
                }
            ],
            model_id="zoo-model",
            model_zoo_manifest_dir=manifest_dir,
            model_parameters={"artifact_uri_template": "memory://{sample_id}.webm"},
            metrics=("artifact_count", "has_artifact:generated_video"),
        )
    )

    assert result.status == "succeeded"
    assert result.delegate_runner == "ResolvedWorldModelRunner+ExistingResultsRunner"
    results = _read_jsonl(output_dir / "results.jsonl")
    assert results[0]["artifacts"]["generated_video"]["uri"] == "memory://sample-a.webm"
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["model"]["model_id"] == "zoo-model"
    assert scorecard["model"]["resolver"]["source"] == "model_zoo"
    assert scorecard["model"]["resolver"]["diagnostics"]["entry_id"] == "zoo-model"
