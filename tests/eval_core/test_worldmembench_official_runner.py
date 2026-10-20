# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids
from worldfoundry.evaluation.tasks.catalog.dispatch import official_runner_spec
from worldfoundry.evaluation.tasks.catalog.registry import load_task_registry_from_paths
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
    build_benchmark_runner_registry,
    run_benchmark_execution,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench import run_worldmembench_official_runner as runner
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.worldmembench_metrics import (
    METRIC_SPECS,
    load_official_results,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = "worldfoundry.evaluation.tasks.execution.runners.worldmembench.run_worldmembench_official_runner"


def _write_result(
    root: Path,
    metric: str = "human_identity",
    *,
    models: dict[str, dict[str, Any]] | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the compact summary schema emitted by upstream ResultWriter."""
    directory = root / "metrics" / metric
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_cases": 2,
        "results": models or {"model_a": {"num_cases": 2, "scores": {"score": 0.75}}},
        "groups": [{"model_id": "model_a", "category": "human", "trajectory_id": "LRLR", "num_cases": 2}],
    }
    summary = directory / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    if cases is not None:
        (directory / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    return summary


def _run(tmp_path: Path, *flags: str) -> tuple[dict[str, Any], int]:
    args = runner.build_parser().parse_args(["--output-dir", str(tmp_path / "out"), *flags])
    return runner.run(args)


def test_catalog_contract_task_runtime_and_dispatch_agree() -> None:
    assert "worldmembench" in formal_benchmark_ids()
    contract = get_external_benchmark_contract("worldmembench")
    assert contract.metric_ids == tuple(METRIC_SPECS)
    spec = official_runner_spec("worldmembench")
    assert spec is not None
    assert spec.module == MODULE
    assert spec.supports_official_runtime and spec.artifact_official_run
    assert spec.model_arg == "--model-id"
    assert not spec.supports_fixture
    registry = build_benchmark_runner_registry()
    entry = registry.zoo.get("worldmembench")
    assert entry.integration_status == "integrated"
    assert not entry.official_benchmark_verified
    assert not entry.leaderboard_valid
    task_path = REPO_ROOT / "worldfoundry/data/benchmarks/tasks/external/worldmembench.yaml"
    task = load_task_registry_from_paths(task_path).get("worldmembench_standard", benchmark="worldmembench").task
    assert task.metric_ids == contract.metric_ids
    assert task.evaluation_protocol_names == ("external_official_runner",)
    assert task.metadata["artifact_contract"]
    for relative in [
        "tasks/external/worldmembench.yaml",
        "runtime_profiles/official/worldmembench.yaml",
    ]:
        payload = yaml.safe_load((REPO_ROOT / "worldfoundry/data/benchmarks" / relative).read_text())
        if "metadata" in payload:
            assert set(payload["metrics"]) == set(contract.metric_ids)
        else:
            assert payload["benchmark_ids"] == ["worldmembench"]
            for item in payload["required_paths"]:
                assert (REPO_ROOT / item["path"]).exists()


@pytest.mark.parametrize("input_kind", ["root", "metrics", "summary"])
def test_import_official_summary_layouts(tmp_path: Path, input_kind: str) -> None:
    root = tmp_path / "official"
    cases = [{"item_id": "case_001", "model_id": "model_a", "status": "ok", "scores": {"score": 0.75}}]
    summary = _write_result(root, cases=cases)
    source = {"root": root, "metrics": root / "metrics", "summary": summary}[input_kind]
    model, extracted, imported_cases = load_official_results(source)
    assert model == "model_a"
    assert extracted["human_identity"]["raw_score"] == 0.75
    assert extracted["human_identity"]["sample_count"] == 2
    assert imported_cases == [{**cases[0], "metric_id": "human_identity"}]


def test_models_are_explicitly_selected_without_pooling(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        models={
            "model_a": {"num_cases": 1, "scores": {"score": 0.2}},
            "model_b": {"num_cases": 1, "scores": {"score": 0.9}},
        },
        cases=[
            {"item_id": "a", "model_id": "model_a", "scores": {"score": 0.2}},
            {"item_id": "b", "model_id": "model_b", "scores": {"score": 0.9}},
        ],
    )
    with pytest.raises(ValueError, match="--model-id"):
        load_official_results(tmp_path)
    model, metrics, cases = load_official_results(tmp_path, model_id="model_b")
    assert model == "model_b"
    assert metrics["human_identity"]["raw_score"] == 0.9
    assert [case["item_id"] for case in cases] == ["b"]
    with pytest.raises(ValueError, match="not present"):
        load_official_results(tmp_path, model_id="absent")


def test_ungrouped_model_uses_official_all_group(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        models={"all": {"num_cases": 1, "scores": {"score": 0.4}}},
        cases=[{"item_id": "video.mp4", "model_id": None, "status": "ok"}],
    )
    model, _, cases = load_official_results(tmp_path)
    assert model == "all"
    assert len(cases) == 1


def test_all_metrics_keep_official_units_without_a_composite(tmp_path: Path) -> None:
    root = tmp_path / "official"
    for metric, (key, _) in METRIC_SPECS.items():
        value = -0.25 if metric == "environment_depth_fidelity" else 3.5 if metric.startswith("camera_") else 0.6
        _write_result(root, metric, models={"model_a": {"num_cases": 2, "scores": {key: value}}})
    scorecard, code = _run(tmp_path, "--official-results-path", str(root))
    assert code == 0
    metrics = scorecard["metrics"]["per_metric"]
    assert set(metrics) == set(METRIC_SPECS)
    assert scorecard["metrics"]["summary"]["available_metrics"] == 12
    assert metrics["camera_rotation"]["raw_score"] == metrics["camera_rotation"]["normalized_score"] == 3.5
    assert metrics["camera_translation"]["higher_is_better"] is False
    assert metrics["environment_depth_fidelity"]["normalized_score"] == -0.25
    assert not any("average" in key for key in scorecard["metrics"]["leaderboard"])
    assert scorecard["normalizer_only"] is True
    assert scorecard["integration_evidence"] is False
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False


@pytest.mark.parametrize(
    "status,expected_code",
    [
        ("model_failure", 0),
        ("not_applicable", 0),
        ("eval_failed", 1),
        ("data_invalid", 1),
    ],
)
def test_case_statuses_and_coverage_are_preserved(tmp_path: Path, status: str, expected_code: int) -> None:
    root = tmp_path / "official"
    scored = 2 if status == "model_failure" else 1
    _write_result(
        root,
        models={
            "model_a": {
                "num_cases": 2,
                "num_scored": scored,
                "status_counts": {"ok": 1, status: 1},
                "scores": {"score": 0.25, "identity_score": 0.2},
            }
        },
        cases=[{"item_id": "a", "model_id": "model_a", "status": status, "error": "metric detail"}],
    )
    card, code = _run(tmp_path, "--official-results-path", str(root))
    assert code == expected_code
    row = card["metrics"]["per_metric"]["human_identity"]
    assert row["sample_count"] == scored
    assert row["coverage"]["status_counts"][status] == 1
    assert row["official_scores"]["identity_score"] == 0.2
    assert row["official_groups"][0]["trajectory_id"] == "LRLR"
    case = json.loads((tmp_path / "out/per_sample_scores.jsonl").read_text())
    assert case["status"] == status
    assert case["error"] == "metric detail"
    if expected_code:
        assert card["run"]["status"] == "failed"
        assert card["evaluation"]["available"] is False
        assert card["metrics"]["leaderboard"] == {}


def test_all_failed_cases_keep_coverage_without_fabricated_scores(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _write_result(
        root,
        models={
            "model_a": {
                "num_cases": 2,
                "num_scored": 0,
                "status_counts": {"eval_failed": 2},
            }
        },
    )
    card, code = _run(tmp_path, "--official-results-path", str(root))
    assert code == 1
    row = card["metrics"]["per_metric"]["human_identity"]
    assert row["coverage"]["num_scored"] == 0
    assert row["raw_score"] is None
    assert not row["available"]
    assert card["metrics"]["leaderboard"] == {}


def test_requested_metrics_must_exist_for_the_selected_model(tmp_path: Path) -> None:
    _write_result(tmp_path)
    with pytest.raises(ValueError, match="Missing requested"):
        load_official_results(tmp_path, metrics=("camera_rotation",))
    with pytest.raises(ValueError, match="Unknown"):
        load_official_results(tmp_path, metrics=("invented_metric",))
    _write_result(tmp_path, "object_geometry", models={"model_b": {"num_cases": 1, "scores": {"score": 0.9}}})
    with pytest.raises(ValueError, match="missing from"):
        load_official_results(tmp_path, model_id="model_a")


def test_missing_inputs_write_standard_failed_artifacts(tmp_path: Path) -> None:
    card, code = _run(tmp_path)
    assert code == 1
    assert card["run"]["status"] == "failed"
    for filename in ("scorecard.json", "raw_metric_table.jsonl", "per_sample_scores.jsonl"):
        assert (tmp_path / "out" / filename).is_file()


@pytest.mark.parametrize("upstream_code", [0, 1])
def test_official_execution_uses_in_tree_runtime_and_respects_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream_code: int,
) -> None:
    manifest = tmp_path / "inputs.jsonl"
    manifest.touch()
    config = tmp_path / "config.yaml"
    config.write_text("metrics: {}\n")
    calls: list[list[str]] = []

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert "cwd" not in kwargs
        assert kwargs["timeout"] == 7200
        result_dir = Path(command[command.index("--output") + 1])
        _write_result(result_dir)
        return subprocess.CompletedProcess(command, upstream_code)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    card, code = _run(
        tmp_path,
        "--run-official",
        "--input-path",
        str(manifest),
        "--config",
        str(config),
        "--metrics",
        "human_identity",
        "--gpus",
        "0,1",
        "--expected-revisits",
        "2",
    )
    assert code == upstream_code
    command = calls[0]
    assert command[:5] == [
        sys.executable,
        "-m",
        "worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime",
        "eval",
        str(manifest),
    ]
    assert command[command.index("--metrics") + 1] == "human_identity"
    assert "--gpus" in command and "--device" not in command
    assert command[command.index("--config") + 1] == str(config)
    assert card["validation"]["official_runtime_executed"] is True
    assert card["integration_evidence"] is (upstream_code == 0)
    assert card["evaluation"]["available"] is (upstream_code == 0)
    assert card["validation"]["full_suite_complete"] is False
    assert (tmp_path / "out/runtime_stdout.log").is_file()


def test_official_execution_does_not_reuse_stale_outputs(tmp_path: Path) -> None:
    _write_result(tmp_path / "out/official")
    card, code = _run(tmp_path, "--run-official")
    assert code == 1
    assert "fresh output" in card["run"]["error"]


def test_official_timeout_writes_failure_scorecard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", execute)
    card, code = _run(tmp_path, "--run-official", "--input-path", str(video), "--timeout", "1")
    assert code == 1
    assert card["run"]["status"] == "failed"
    assert "timed out" in card["run"]["error"]


def test_standard_benchmark_contract_and_result_import(tmp_path: Path) -> None:
    contract = run_benchmark_execution("worldmembench", output_dir=tmp_path / "contract", mode="contract")
    assert contract.ok
    assert contract.scorecard_path.is_file()
    root = tmp_path / "official"
    _write_result(root)
    imported = run_benchmark_execution(
        "worldmembench",
        output_dir=tmp_path / "import",
        mode="official-validation",
        official_results_path=root,
    )
    assert imported.ok, imported.to_dict()
    card = json.loads(imported.scorecard_path.read_text())
    assert card["benchmark"]["benchmark_id"] == "worldmembench"
    assert card["normalizer_only"] is True
    assert card["metrics"]["per_metric"]["human_identity"]["raw_score"] == 0.75


def test_standard_official_run_dispatches_generated_videos(tmp_path: Path) -> None:
    interpreter = tmp_path / "metric-python"
    payload = {"num_cases": 1, "results": {"all": {"num_cases": 1, "scores": {"score": 0.5}}}}
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "assert sys.argv[1:4] == ['-m', 'worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime', 'eval']\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "directory = output / 'metrics' / 'imaging_quality'\n"
        "directory.mkdir(parents=True)\n"
        f"(directory / 'summary.json').write_text({json.dumps(payload)!r})\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "case_001.mp4").touch()
    result = run_benchmark_execution(
        "worldmembench",
        output_dir=tmp_path / "official-run",
        mode="official-run",
        generated_artifact_dir=videos,
        env_overrides={"WORLDFOUNDRY_WORLDMEMBENCH_PYTHON": str(interpreter)},
    )
    assert result.ok, result.to_dict()
    card = json.loads(result.scorecard_path.read_text())
    assert card["validation"]["official_runtime_executed"] is True
    assert card["integration_evidence"] is True
    assert card["validation"]["full_suite_complete"] is False
    assert card["metrics"]["leaderboard"] == {"imaging_quality": 0.5}


@pytest.mark.parametrize("select_model", [False, True])
def test_public_cli_imports_results(tmp_path: Path, select_model: bool) -> None:
    root = tmp_path / "official"
    _write_result(
        root,
        models={
            "model_a": {"num_cases": 2, "scores": {"score": 0.75}},
            **({"model_b": {"num_cases": 2, "scores": {"score": 0.2}}} if select_model else {}),
        },
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldfoundry.cli",
            "zoo",
            "benchmark-run",
            "--benchmark-id",
            "worldmembench",
            "--mode",
            "official-validation",
            "--official-results-path",
            str(root),
            "--output-dir",
            str(tmp_path / "cli"),
            *(["--result-model-id", "model_a"] if select_model else []),
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    card = json.loads((tmp_path / "cli/scorecard.json").read_text())
    assert card["metrics"]["leaderboard"] == {"human_identity": 0.75}
    assert card["eligibility"]["leaderboard_valid"] is False
