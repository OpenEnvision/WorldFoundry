from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry import cli
from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest


def _write_plan_cli_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_root = tmp_path / "tasks"
    data_root = tmp_path / "data"
    task_root.mkdir()
    data_root.mkdir()
    (task_root / "task.yaml").write_text(
        """
name: cli-plan-task
benchmark_name: cli-plan-benchmark
protocol: open_loop
input_keys: [prompt]
output_keys: [generated_video]
data:
  metadata_path: samples.jsonl
""",
        encoding="utf-8",
    )
    (data_root / "samples.jsonl").write_text(
        '{"sample_id": "sample-001", "prompt": "camera pan"}\n',
        encoding="utf-8",
    )
    return task_root, data_root


def test_plan_help_exposes_create_show_validate(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["plan", "--help"])
    output = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert "create" in output
    assert "show" in output
    assert "validate" in output


def test_plan_create_show_validate_and_evaluate_from_plan(tmp_path, capsys) -> None:
    task_root, data_root = _write_plan_cli_fixture(tmp_path)
    plan_path = tmp_path / "plan.json"

    exit_code = cli.main(
        [
            "plan",
            "create",
            "--task-name",
            "cli-plan-task",
            "--task-root",
            str(task_root),
            "--data-path",
            str(data_root),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(plan_path),
            "--materialize-requests",
            "--json",
        ]
    )
    created = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert plan_path.is_file()
    assert created["schema_version"] == "worldfoundry-run-plan"
    assert created["task"]["task_name"] == "cli-plan-task"
    assert created["requests"][0]["sample_id"] == "sample-001"

    assert cli.main(["plan", "show", str(plan_path), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["fingerprint"] == created["fingerprint"]

    assert cli.main(["plan", "validate", str(plan_path), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["ok"] is True

    assert cli.main(["evaluate", "--plan", str(plan_path), "--json"]) == 0
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["status"] == "succeeded"
    assert evaluated["sample_count"] == 1


def test_plan_create_materializes_from_dataset_manifest(tmp_path, capsys) -> None:
    task_root, data_root = _write_plan_cli_fixture(tmp_path)
    manifest = build_dataset_manifest(
        samples_path=data_root / "samples.jsonl",
        root=data_root,
        dataset_id="cli-manifest-dataset",
        split="manifest-split",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    plan_path = tmp_path / "plan.json"
    write_dataset_manifest(manifest, manifest_path)

    exit_code = cli.main(
        [
            "plan",
            "create",
            "--task-name",
            "cli-plan-task",
            "--task-root",
            str(task_root),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(plan_path),
            "--materialize-requests",
            "--json",
        ]
    )
    created = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert created["dataset"]["dataset_id"] == "cli-manifest-dataset"
    assert created["materialization"]["source"] == "dataset_manifest"
    assert created["requests"][0]["split"] == "manifest-split"

    assert cli.main(["plan", "validate", str(plan_path), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["ok"] is True
    assert validation["dataset"]["ok"] is True


def test_plan_validate_cli_returns_nonzero_for_drifted_dataset_manifest(tmp_path, capsys) -> None:
    task_root, data_root = _write_plan_cli_fixture(tmp_path)
    manifest = build_dataset_manifest(
        samples_path=data_root / "samples.jsonl",
        root=data_root,
        dataset_id="cli-drift-dataset",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    plan_path = tmp_path / "plan.json"
    write_dataset_manifest(manifest, manifest_path)

    cli.main(
        [
            "plan",
            "create",
            "--task-name",
            "cli-plan-task",
            "--task-root",
            str(task_root),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(plan_path),
            "--json",
        ]
    )
    capsys.readouterr()
    (data_root / "samples.jsonl").write_text(
        '{"sample_id": "sample-001", "prompt": "changed"}\n',
        encoding="utf-8",
    )

    exit_code = cli.main(["plan", "validate", str(plan_path), "--json"])
    validation = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert validation["ok"] is False
    assert "dataset manifest: sha256 mismatch" in validation["issues"]
    assert validation["dataset"]["ok"] is False


def test_run_plan_executes_without_source_task_selector(tmp_path, capsys) -> None:
    task_root, data_root = _write_plan_cli_fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    cli.main(
        [
            "plan",
            "create",
            "--task-name",
            "cli-plan-task",
            "--task-root",
            str(task_root),
            "--data-path",
            str(data_root),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(plan_path),
            "--materialize-requests",
            "--json",
        ]
    )
    capsys.readouterr()

    exit_code = cli.main(["run", "--plan", str(plan_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["engine"] == "plan"
    assert payload["status"] == "succeeded"
    assert payload["sample_count"] == 1


def test_run_without_plan_still_requires_source_task_selector(capsys) -> None:
    exit_code = cli.main(["run", "--engine", "in-process", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "required unless --plan" in captured.err


def test_plan_create_can_use_default_benchmark_zoo_task_manifest(tmp_path, capsys) -> None:
    plan_path = tmp_path / "vbench-plan.json"

    exit_code = cli.main(
        [
            "plan",
            "create",
            "--task-name",
            "vbench_t2v_standard",
            "--benchmark",
            "vbench",
            "--output-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(plan_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["task"]["task_name"] == "vbench_t2v_standard"
    assert payload["task"]["metadata"]["source_kind"] == "benchmark_zoo"
    assert payload["task"]["metadata"]["artifact_contract"]["required"]["generated_video"]["kind"] == "video"
    assert plan_path.is_file()
