# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and scoring-batch controls across public evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from worldfoundry.cli.main import main
from worldfoundry.evaluation import framework
from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark, model_benchmark_suite, service
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import EvaluateRunRequest
from worldfoundry.studio.serving import workspace
from worldfoundry.studio.serving.workspace import JobCreateRequest

pytestmark = pytest.mark.unit


def test_workspace_existing_results_honors_explicit_mode_with_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "status": "succeeded",
                "artifacts": {"generated_video": {"uri": str(source), "kind": "video"}},
            }
        )
        + "\n"
    )
    result = workspace._run_evaluation(
        JobCreateRequest(
            job_type="evaluation",
            eval_mode="existing-results",
            benchmark_id="vbench",
            model_id="user-model",
            results_path=str(results),
            output_dir=str(tmp_path / "out"),
            metrics=["artifact_count"],
            resume=True,
            metric_batch_size=7,
        )
    )
    assert result["mode"] == "existing-results"
    assert result["request"]["resume"] is True
    assert result["request"]["metric_batch_size"] == 7
    assert result["sample_count"] == 1
    card = json.loads(Path(result["scorecard_path"]).read_text())
    assert card["metrics"]["leaderboard"]["artifact_count"] == 1.0
    assert card["eligibility"]["leaderboard_valid"] is False


def dataset(tmp_path: Path) -> tuple[Path, Path]:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "prompt": "cube", "status": "succeeded"}) + "\n")
    manifest = tmp_path / "dataset.json"
    write_dataset_manifest(build_dataset_manifest(samples_path=samples, dataset_id="custom"), manifest)
    return manifest, samples


@pytest.mark.parametrize("command", ["score", "generate-score", "reproduce"])
def test_intent_cli_serializes_resume_and_batch_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    manifest, samples = dataset(tmp_path)
    if command == "score":
        arguments = ["score", "--results", str(samples), "--metric", "artifact_count"]
    elif command == "generate-score":
        arguments = [
            "generate-score",
            "--model",
            "hailuo-2p3",
            "--dataset-manifest",
            str(manifest),
            "--metric",
            "artifact_count",
        ]
    else:
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(
            "schema_version: worldfoundry-reproduction-recipe-v1\nid: fixture\n"
            "benchmark: {id: vbench, revision: benchmark-rev}\nmodel: {id: zeroscope, revision: model-rev}\n"
            f"data: {{requests_path: {samples}}}\nevaluation: {{mode: official-run}}\n"
        )
        arguments = ["reproduce", "--recipe", str(recipe)]
    assert (
        main(
            [
                *arguments,
                "--resume",
                "--metric-batch-size",
                "7",
                "--output-dir",
                str(tmp_path / "out"),
                "--plan-only",
                "--json",
            ]
        )
        == 0
    )
    execution = json.loads(capsys.readouterr().out)["execution"]["request"]
    assert execution["resume"] is True
    assert execution["metric_batch_size"] == 7


@pytest.mark.parametrize("kind", ["evaluate", "single", "suite"])
def test_public_facade_preserves_checkpoint_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    captured: list[Any] = []

    def execute(request: Any) -> SimpleNamespace:
        captured.append(request)
        return SimpleNamespace(status="succeeded", exit_code=0, output_dir=tmp_path)

    monkeypatch.setattr(framework, "execute_evaluate_run", execute)
    monkeypatch.setattr(framework, "run_model_benchmark", execute)
    monkeypatch.setattr(framework, "run_model_benchmark_suite", execute)
    benchmarks = () if kind == "evaluate" else ("vbench",) if kind == "single" else ("vbench", "libero")
    framework.run_worldfoundry(
        framework.WorldFoundryRunRequest(
            output_dir=tmp_path,
            model_ids=("custom",),
            benchmark_ids=benchmarks,
            model_manifest_dir=None,
            model_runner="tests:Runner",
            resume=True,
            metric_batch_size=7,
        )
    )
    assert len(captured) == 1
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


@pytest.mark.parametrize("engine", ("auto", "in-process", "existing-results"))
def test_public_facade_scores_imported_results_without_starting_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    captured: list[EvaluateRunRequest] = []

    def evaluate(request: EvaluateRunRequest) -> SimpleNamespace:
        captured.append(request)
        return SimpleNamespace(status="succeeded", exit_code=0, output_dir=tmp_path)

    def generate(request: Any) -> None:
        pytest.fail("existing results must not start benchmark generation")

    monkeypatch.setattr(framework, "execute_evaluate_run", evaluate)
    monkeypatch.setattr(framework, "run_model_benchmark", generate)
    monkeypatch.setattr(framework, "run_model_benchmark_suite", generate)
    results_path = tmp_path / "results.jsonl"
    result = framework.run_worldfoundry(
        framework.WorldFoundryRunRequest(
            output_dir=tmp_path,
            results_path=results_path,
            engine=engine,
            model_ids=("offline-model",),
            model_manifest_dir=None,
            benchmark_ids=("offline-benchmark",),
            benchmark_id="offline-benchmark",
            resume=True,
            metric_batch_size=7,
        )
    )

    assert result.kind == "evaluate"
    assert len(captured) == 1
    request = captured[0]
    assert request.mode == "existing-results"
    assert request.results_path == results_path
    assert request.benchmark_id == "offline-benchmark"
    assert request.model_id == "offline-model"
    assert request.resume is True
    assert request.metric_batch_size == 7


@pytest.mark.parametrize("task_registry", [False, True])
def test_single_benchmark_forwards_controls_to_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_registry: bool
) -> None:
    captured: list[EvaluateRunRequest] = []
    rows = [{"sample_id": "a", "task_name": "fixture"}]
    request = model_benchmark.ModelBenchmarkRunRequest(
        output_dir=tmp_path,
        benchmark_id="vbench",
        benchmark_manifest_path=tmp_path,
        model_id="custom",
        requests=None if task_registry else rows,
        resume=True,
        metric_batch_size=7,
        task_name="fixture" if task_registry else None,
        dataset_root=tmp_path,
    )
    monkeypatch.setattr(model_benchmark, "execute_evaluate_run", lambda row: captured.append(row))
    if task_registry:
        monkeypatch.setattr(
            model_benchmark, "build_run_plan_from_task_registry", lambda **kwargs: SimpleNamespace(requests=rows)
        )
        monkeypatch.setattr(model_benchmark, "write_run_plan", lambda *args: None)
        monkeypatch.setattr(
            model_benchmark, "evaluate_request_from_run_plan", lambda plan: EvaluateRunRequest(output_dir=tmp_path)
        )
    model_benchmark._run_generation(request, tmp_path)
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


def test_suite_cell_forwards_controls_to_single_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[model_benchmark.ModelBenchmarkRunRequest] = []

    def execute(request: model_benchmark.ModelBenchmarkRunRequest, **kwargs: Any) -> SimpleNamespace:
        captured.append(request)
        return SimpleNamespace(
            status="succeeded",
            exit_code=0,
            run_manifest_path=tmp_path / "manifest.json",
            to_dict=lambda: {"benchmark_result": {}, "artifacts": {}},
        )

    monkeypatch.setattr(model_benchmark_suite, "run_model_benchmark", execute)
    plan = SimpleNamespace(
        output_artifact="generated_video",
        model_id="custom",
        requested_model_id="custom",
        benchmark=SimpleNamespace(benchmark_id="vbench"),
        benchmark_manifest_path=tmp_path,
        known_model=False,
        required_artifacts=("generated_video",),
        evaluation_provenance={},
    )
    request = model_benchmark_suite.ModelBenchmarkSuiteRequest(output_dir=tmp_path, resume=True, metric_batch_size=7)
    model_benchmark_suite._run_cell(request, root=tmp_path, plan=plan)
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


def test_metric_batch_size_does_not_change_suite_identity(tmp_path: Path) -> None:
    first = model_benchmark_suite.ModelBenchmarkSuiteRequest(output_dir=tmp_path, metric_batch_size=1)
    second = model_benchmark_suite.ModelBenchmarkSuiteRequest(output_dir=tmp_path, metric_batch_size=7)
    assert model_benchmark_suite._fingerprint_request(first) == model_benchmark_suite._fingerprint_request(second)


@pytest.mark.parametrize("command", ["evaluate", "run"])
def test_materialized_cli_forwards_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    _, samples = dataset(tmp_path)
    captured: list[EvaluateRunRequest] = []

    def execute(request: EvaluateRunRequest) -> SimpleNamespace:
        captured.append(request)
        return SimpleNamespace(exit_code=0, to_dict=lambda: {"status": "succeeded", "exit_code": 0})

    monkeypatch.setattr("worldfoundry.evaluation.runner.execute_evaluate_run", execute)
    monkeypatch.setattr(framework, "execute_evaluate_run", execute)
    arguments = [command, "--results-path", str(samples)]
    if command == "run":
        arguments += ["--engine", "existing-results"]
    assert (
        main([*arguments, "--resume", "--metric-batch-size", "7", "--output-dir", str(tmp_path / "out"), "--json"]) == 0
    )
    capsys.readouterr()
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


def test_direct_model_cli_forwards_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[EvaluateRunRequest] = []
    monkeypatch.setattr(
        "worldfoundry.evaluation.runner.execute_evaluate_run",
        lambda request: (
            captured.append(request) or SimpleNamespace(exit_code=0, to_dict=lambda: {"status": "succeeded"})
        ),
    )
    assert (
        main(
            [
                "run",
                "hailuo-2p3",
                "--prompt",
                "cube",
                "--resume",
                "--metric-batch-size",
                "7",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


def test_evaluate_plan_replay_preserves_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: list[EvaluateRunRequest] = []
    monkeypatch.setattr("worldfoundry.evaluation.runner.load_run_plan", lambda path: object())
    monkeypatch.setattr(
        "worldfoundry.evaluation.runner.evaluate_request_from_run_plan",
        lambda plan: EvaluateRunRequest(output_dir=tmp_path),
    )
    monkeypatch.setattr(
        "worldfoundry.evaluation.runner.execute_evaluate_run",
        lambda row: (captured.append(row) or SimpleNamespace(exit_code=0, to_dict=lambda: {"status": "succeeded"})),
    )
    assert (
        main(["evaluate", "--plan", str(tmp_path / "plan.json"), "--resume", "--metric-batch-size", "7", "--json"]) == 0
    )
    capsys.readouterr()
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


@pytest.mark.parametrize("mode", ["model-benchmark", "generate-and-score"])
def test_workspace_prepared_intent_preserves_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(service, "prepare_evaluation", lambda intent: intent)
    payload = JobCreateRequest(
        eval_mode=mode, model_id="custom", benchmark_id="vbench", resume=True, metric_batch_size=7
    )
    intent = workspace._prepared_evaluation_from_payload(payload, str(tmp_path))
    assert intent.resume is True
    assert intent.metric_batch_size == 7


@pytest.mark.parametrize("plan_replay", [False, True])
def test_workspace_request_preserves_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan_replay: bool
) -> None:
    captured: list[EvaluateRunRequest] = []
    monkeypatch.setattr(
        "worldfoundry.evaluation.runner.run_evaluate",
        lambda row: (captured.append(row) or SimpleNamespace(to_dict=lambda: {"status": "succeeded"})),
    )
    monkeypatch.setattr(
        "worldfoundry.evaluation.tasks.execution.orchestration.plan.load_run_plan", lambda path: object()
    )
    monkeypatch.setattr(
        "worldfoundry.evaluation.tasks.execution.orchestration.plan.validate_run_plan", lambda plan: {"ok": True}
    )
    monkeypatch.setattr(
        "worldfoundry.evaluation.tasks.execution.orchestration.plan.evaluate_request_from_run_plan",
        lambda plan: EvaluateRunRequest(output_dir=tmp_path),
    )
    workspace._run_evaluation(
        JobCreateRequest(
            output_dir=str(tmp_path),
            resume=True,
            metric_batch_size=7,
            run_plan_path="plan.json" if plan_replay else "",
        )
    )
    assert captured[0].resume is True
    assert captured[0].metric_batch_size == 7


def test_preparation_rejects_nonpositive_metric_batch_size(tmp_path: Path) -> None:
    _, samples = dataset(tmp_path)
    prepared = service.prepare_evaluation(
        service.ScoreResultsIntent(
            output_dir=tmp_path,
            results_path=samples,
            metrics=("artifact_count",),
            metric_batch_size=0,
        )
    )
    assert not prepared.ready
    assert any(issue.code == "metric_batch_size_invalid" for issue in prepared.issues)


@pytest.mark.parametrize("entry", ["facade", "single", "suite"])
def test_public_requests_reject_nonpositive_batch_before_model_loading(tmp_path: Path, entry: str) -> None:
    with pytest.raises(ValueError, match="metric_batch_size must be positive"):
        if entry == "facade":
            framework.run_worldfoundry(framework.WorldFoundryRunRequest(output_dir=tmp_path, metric_batch_size=0))
        elif entry == "single":
            model_benchmark.run_model_benchmark(
                model_benchmark.ModelBenchmarkRunRequest(
                    output_dir=tmp_path,
                    benchmark_id="vbench",
                    benchmark_manifest_path=tmp_path,
                    metric_batch_size=0,
                )
            )
        else:
            model_benchmark_suite.run_model_benchmark_suite(
                model_benchmark_suite.ModelBenchmarkSuiteRequest(
                    output_dir=tmp_path,
                    metric_batch_size=0,
                )
            )


@pytest.mark.parametrize("flags", [["--resume"], ["--metric-batch-size", "32"]])
def test_official_artifact_scoring_rejects_in_process_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    assert main(["score", "--benchmark", "vbench", "--artifacts", str(tmp_path), *flags]) == 2
    assert "apply to --results" in capsys.readouterr().err
