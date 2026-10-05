from __future__ import annotations

from pathlib import Path

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET
from worldfoundry.evaluation.runner import (
    build_run_plan,
    build_run_plan_from_task_registry,
    evaluate_request_from_run_plan,
    execute_evaluate_run,
    validate_run_plan,
)
from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest


def _write_plan_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_root = tmp_path / "tasks"
    data_root = tmp_path / "data"
    task_root.mkdir()
    data_root.mkdir()
    (task_root / "plan_task.yaml").write_text(
        """
name: plan-task
benchmark_name: plan-benchmark
protocol: open_loop
input_keys: [prompt]
output_keys: [generated_video]
data:
  metadata_path: samples.jsonl
generation_defaults:
  seed: 7
""",
        encoding="utf-8",
    )
    (data_root / "samples.jsonl").write_text(
        '{"sample_id": "sample-a", "prompt": "a small room"}\n'
        '{"sample_id": "sample-b", "prompt": "a large room"}\n',
        encoding="utf-8",
    )
    return task_root, data_root


def test_run_plan_materializes_task_yaml_requests_and_executes_model(tmp_path: Path) -> None:
    task_root, data_root = _write_plan_fixture(tmp_path)
    plan = build_run_plan_from_task_registry(
        task_name="plan-task",
        task_roots=[task_root],
        output_dir=tmp_path / "run",
        mode="model",
        model_id=CONTRACT_FIXTURE_MODEL_ID,
        model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
        dataset_root=data_root,
        materialize_requests=True,
        limit=1,
        metrics=("artifact_count",),
    )

    assert plan.schema_version == "worldfoundry-run-plan"
    assert len(plan.requests) == 1
    assert plan.requests[0]["sample_id"] == "sample-a"
    assert plan.requests[0]["generation_kwargs"]["seed"] == 7
    assert validate_run_plan(plan)["ok"] is True

    result = execute_evaluate_run(evaluate_request_from_run_plan(plan))

    assert result.status == "succeeded"
    assert result.sample_count == 1
    assert result.scorecard_path.is_file()


def test_run_plan_materializes_task_yaml_from_dataset_manifest(tmp_path: Path) -> None:
    task_root, data_root = _write_plan_fixture(tmp_path)
    manifest = build_dataset_manifest(
        samples_path=data_root / "samples.jsonl",
        root=data_root,
        dataset_id="manifest-plan-dataset",
        split="validation",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)

    plan = build_run_plan_from_task_registry(
        task_name="plan-task",
        task_roots=[task_root],
        output_dir=tmp_path / "run",
        mode="model",
        model_id=CONTRACT_FIXTURE_MODEL_ID,
        dataset_manifest=manifest_path,
        materialize_requests=True,
        limit=1,
    )

    validation = validate_run_plan(plan)

    assert plan.dataset["manifest_path"] == str(manifest_path)
    assert plan.dataset["dataset_id"] == "manifest-plan-dataset"
    assert plan.materialization["source"] == "dataset_manifest"
    assert len(plan.requests) == 1
    assert plan.requests[0]["sample_id"] == "sample-a"
    assert plan.requests[0]["split"] == "validation"
    assert validation["ok"] is True
    assert validation["dataset"]["ok"] is True


def test_run_plan_validate_reports_dataset_manifest_drift(tmp_path: Path) -> None:
    task_root, data_root = _write_plan_fixture(tmp_path)
    manifest = build_dataset_manifest(
        samples_path=data_root / "samples.jsonl",
        root=data_root,
        dataset_id="drift-plan-dataset",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    (data_root / "samples.jsonl").write_text(
        '{"sample_id": "sample-a", "prompt": "changed"}\n',
        encoding="utf-8",
    )
    plan = build_run_plan_from_task_registry(
        task_name="plan-task",
        task_roots=[task_root],
        output_dir=tmp_path / "run",
        mode="model",
        model_id=CONTRACT_FIXTURE_MODEL_ID,
        dataset_manifest=manifest_path,
    )

    validation = validate_run_plan(plan)

    assert validation["ok"] is False
    assert "dataset manifest: sha256 mismatch" in validation["issues"]
    assert validation["dataset"]["ok"] is False


def test_run_plan_validate_reports_missing_results_for_existing_results_mode(tmp_path: Path) -> None:
    plan = build_run_plan(output_dir=tmp_path / "run", mode="existing-results")
    validation = validate_run_plan(plan)

    assert validation["ok"] is False
    assert validation["issues"] == ["existing-results mode requires results_path"]


def test_run_plan_validate_reports_unsupported_metric_ids(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"sample_id":"a","artifacts":{}}\n', encoding="utf-8")
    plan = build_run_plan(
        output_dir=tmp_path / "run",
        mode="existing-results",
        results_path=str(results_path),
        metrics=("artifact_count", "nope"),
    )
    validation = validate_run_plan(plan)

    assert validation["ok"] is False
    assert validation["issues"] == ["unsupported metric: 'nope'"]
    assert validation["metrics"]["unknown_metrics"] == ["nope"]
