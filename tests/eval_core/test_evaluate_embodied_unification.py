from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.cli import main
from worldfoundry.evaluation.runner import EvaluateRunRequest, execute_evaluate_run
from worldfoundry.evaluation.tasks.embodied import (
    VlaVaWamRunRequest,
    materialize_vla_va_wam_requests,
    metric_suite,
    run_vla_va_wam,
)
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_RUNNER_TARGET


EMBODIED_SPEC = {
    "track": "vla",
    "kind": "action",
    "task_name": "pick_place_smoke",
    "action_space": {"kind": "discrete", "actions": ["pick", "place"]},
    "observation_keys": ["instruction"],
    "output_keys": ["actions"],
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evaluate_model_mode_runs_embodied_metric_objects_through_contract_runner(tmp_path: Path) -> None:
    materialized = materialize_vla_va_wam_requests(
        [{"sample_id": "episode-1", "instruction": "pick up the cube"}],
        spec=EMBODIED_SPEC,
    )
    output_dir = tmp_path / "evaluate-embodied"

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            mode="model",
            requests=materialized.requests,
            model_id="test-contract-model",
            model_runner="tests.eval_core.contract_fixture:ContractFixtureRunner",
            model_parameters={"metadata_namespace": "vla_va_wam"},
            metrics=metric_suite(("generation_success", "task_success"), track="vla"),
            run_id="evaluate-embodied-run",
        )
    )

    assert result.mode == "model"
    assert result.delegate_runner == "ResolvedWorldModelRunner+ContractRunner"
    assert result.status == "succeeded"
    summary = _read_json(output_dir / "metrics" / "summary.json")
    assert summary["leaderboard"]["generation_success"] == 1.0
    assert summary["leaderboard"]["task_success"] == 1.0
    assert _read_json(output_dir / "run_manifest.json")["runner"] == "contract_runner"


def test_vla_va_wam_runner_is_evaluate_runner_wrapper(tmp_path: Path) -> None:
    output_dir = tmp_path / "vla-wrapper"

    result = run_vla_va_wam(
        VlaVaWamRunRequest(
            output_dir=output_dir,
            spec=EMBODIED_SPEC,
            samples=[{"sample_id": "episode-1", "instruction": "pick up the cube"}],
            model_id="test-contract-model",
            model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
            metric_ids=("generation_success", "task_success"),
            run_id="vla-wrapper-run",
        )
    )

    assert result.mode == "model"
    assert result.delegate_runner == "ResolvedWorldModelRunner+ContractRunner"
    assert result.status == "succeeded"
    scorecard = _read_json(output_dir / "scorecard.json")
    assert scorecard["benchmark"]["evaluation_protocol"] == "worldfoundry_evaluate_model"
    assert scorecard["model"]["model_type"] == "world_model"


def test_evaluate_existing_results_accepts_embodied_metric_objects(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing-embodied"

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=output_dir,
            requests=[{"sample_id": "episode-1", "task_name": "pick_place_smoke"}],
            results=[
                {
                    "sample_id": "episode-1",
                    "model_id": "offline-vla",
                    "metadata": {"metrics": {"task_success": 1.0}},
                }
            ],
            metrics=metric_suite(("task_success",), track="vla"),
        )
    )

    assert result.mode == "existing-results"
    assert result.delegate_runner == "ExistingResultsRunner"
    summary = _read_json(output_dir / "metrics" / "summary.json")
    assert summary["leaderboard"]["task_success"] == 1.0


def test_evaluate_cli_materializes_embodied_spec_samples(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        json.dumps({"sample_id": "episode-1", "instruction": "pick up the cube"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "cli-embodied"

    exit_code = main(
        [
            "evaluate",
            "--embodied-spec",
            json.dumps(EMBODIED_SPEC),
            "--samples-path",
            str(samples_path),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "test-contract-model",
            "--model-runner",
            CONTRACT_FIXTURE_RUNNER_TARGET,
            "--metric",
            "task_success",
            "--run-id",
            "cli-embodied-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "model"
    assert payload["delegate_runner"] == "ResolvedWorldModelRunner+ContractRunner"
    summary = _read_json(output_dir / "metrics" / "summary.json")
    assert summary["leaderboard"]["task_success"] == 1.0
    scorecard = _read_json(output_dir / "scorecard.json")
    assert scorecard["run"]["run_id"] == "cli-embodied-run"
    assert scorecard["benchmark"]["track"] == "vla"
