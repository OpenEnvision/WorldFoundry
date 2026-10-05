from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldfoundry import cli
from worldfoundry.cli import main
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TASK_TYPE = "worldfoundry_imagetext2video_gen"
LOCAL_BENCHMARK_NAME = "worldfoundry_imagetext2video"

_CLI_FIXTURE_TASK_YAML = """name: worldfoundry-imagetext2video-gen
protocol: open_loop
capability_track: core_video
schema_type: sample
input_keys:
  - generation_text
  - ref_image
output_keys:
  - generated_video
metric_groups:
  - quality
description: WorldFoundry image/text-to-video benchmark with local task schema.
dataset_root: placeholder
output_dir: placeholder
data:
  metadata_path: metadata.jsonl
  media_root: ""
"""


def _patch_cli_imagetext_video_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    from worldfoundry.cli import utils as cli_utils

    task_yaml = tmp_path / "worldfoundry_imagetext2video.yaml"
    task_yaml.write_text(_CLI_FIXTURE_TASK_YAML, encoding="utf-8")

    class _FakeBenchmark:
        source_kind = "task_yaml"
        task_type = LOCAL_TASK_TYPE
        benchmark_name = LOCAL_BENCHMARK_NAME
        suite = "local"
        backend = "task_yaml"
        evaluation_protocol = "open_loop"
        input_keys = ("generation_text", "ref_image")
        output_keys = ("generated_video",)

        def load_samples(self, dataset_root, limit=None):
            rows = []
            metadata_path = Path(dataset_root) / "metadata.jsonl"
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break

            class _TaskCfg:
                generation_defaults = {}

            return _TaskCfg(), rows

    monkeypatch.setattr(
        cli_utils,
        "resolve_cli_benchmark_for_materialize",
        lambda task_type, benchmark_name: _FakeBenchmark(),
    )


def _write_local_video_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "sample_id": "sample-001",
        "generation_text": "A camera push-in over a quiet indoor scene.",
        "ref_image": "memory://sample.png",
        "expected_outputs": {"generated_video": ""},
    }
    (root / "metadata.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return root


def test_top_level_contract_command_is_retired_in_favor_of_zoo(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["contract", "--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "invalid choice: 'contract'" in captured.err
    assert "zoo" in captured.err


def test_root_help_lists_eval_core_evaluate_entrypoint(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "compare-runs" in captured.out
    assert "dataset" in captured.out
    assert "evaluate" in captured.out
    assert "metric" in captured.out
    assert "plan" in captured.out
    assert "run                 Execute or score a WorldFoundry benchmark through the" in captured.out
    assert "unified facade" in captured.out
    assert "run-benchmark" not in captured.out
    assert "run-suite" not in captured.out
    assert "suites" in captured.out
    assert "zoo" in captured.out


def test_evaluate_help_explains_materialized_results_path(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["evaluate", "--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "deterministic eval-core" in captured.out
    assert "--results-path" in captured.out
    assert "scorecard.json" in captured.out


def test_evaluate_cli_scores_existing_results_jsonl(tmp_path, capsys) -> None:
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "evaluate"
    rows = [
        {
            "sample_id": "sample-a",
            "task_name": "offline_t2v",
            "artifacts": {"video": {"uri": "outputs/a.mp4", "kind": "video"}},
            "metrics": {"quality": 0.25},
        },
        {
            "sample_id": "sample-b",
            "task_name": "offline_t2v",
            "metrics": {"quality": 0.75},
        },
    ]
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate",
            "--results-path",
            str(results_path),
            "--output-dir",
            str(output_dir),
            "--benchmark-id",
            "offline-benchmark",
            "--model-id",
            "offline-model",
            "--metric",
            "artifact_count",
            "--metric",
            "numeric:quality",
            "--required-artifact",
            "video",
            "--run-id",
            "evaluate-cli-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-evaluate-run-result"
    assert payload["status"] == "succeeded"
    assert payload["sample_count"] == 2
    assert summary["leaderboard"]["artifact_count"] == 0.5
    assert summary["leaderboard"]["quality"] == 0.5
    assert summary["leaderboard"]["required_artifacts_present"] == 0.5
    assert scorecard["run"]["run_id"] == "evaluate-cli-run"
    assert scorecard["benchmark"]["benchmark_name"] == "offline-benchmark"
    assert scorecard["model"]["model_id"] == "offline-model"
    assert scorecard["eligibility"]["score_valid"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert "missing official/full-suite leaderboard evidence gate" in scorecard["eligibility"]["reasons"]
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.md").is_file()


def test_unified_run_scores_existing_results_with_benchmark_metadata(tmp_path, capsys) -> None:
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "run-existing"
    results_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "task_name": "offline_t2v",
                "artifacts": {"video": {"uri": "memory://sample-a.mp4", "kind": "video"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--results-path",
            str(results_path),
            "--output-dir",
            str(output_dir),
            "--benchmark-id",
            "offline-benchmark",
            "--model-id",
            "offline-model",
            "--metric",
            "artifact_count",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["kind"] == "evaluate"
    assert payload["delegate"]["mode"] == "existing-results"
    assert payload["delegate"]["sample_count"] == 1
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["benchmark"]["benchmark_name"] == "offline-benchmark"
    assert (output_dir / "scorecard.json").is_file()


def test_evaluate_cli_model_mode_materializes_task_and_scores_test_fixture(tmp_path, monkeypatch, capsys) -> None:
    _patch_cli_imagetext_video_registry(monkeypatch, tmp_path)
    output_dir = tmp_path / "evaluate-model"
    data_root = _write_local_video_dataset(tmp_path / "local-benchmark")

    exit_code = main(
        [
            "evaluate",
            "--mode",
            "model",
            "--task-type",
            LOCAL_TASK_TYPE,
            "--benchmark-name",
            LOCAL_BENCHMARK_NAME,
            "--data-path",
            str(data_root),
            "--num-samples",
            "1",
            "--output-dir",
            str(output_dir),
            "--model-id",
            CONTRACT_FIXTURE_MODEL_ID,
            "--model-runner",
            CONTRACT_FIXTURE_RUNNER_TARGET,
            "--metric",
            "artifact_count",
            "--metric",
            "has_artifact:generated_video",
            "--run-id",
            "evaluate-cli-model-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["mode"] == "model"
    assert payload["delegate_runner"] == "ResolvedWorldModelRunner+ExistingResultsRunner"
    assert payload["sample_count"] == 1
    assert summary["leaderboard"]["artifact_count"] == 1.0
    assert summary["leaderboard"]["has_artifact:generated_video"] == 1.0
    assert scorecard["run"]["run_id"] == "evaluate-cli-model-run"
    assert scorecard["benchmark"]["benchmark_name"] == LOCAL_BENCHMARK_NAME
    assert scorecard["model"]["model_id"] == CONTRACT_FIXTURE_MODEL_ID


def test_evaluate_cli_model_mode_resolves_model_zoo_runner(tmp_path, monkeypatch, capsys) -> None:
    _patch_cli_imagetext_video_registry(monkeypatch, tmp_path)
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
    output_dir = tmp_path / "evaluate-zoo-model"
    data_root = _write_local_video_dataset(tmp_path / "local-benchmark")

    exit_code = main(
        [
            "evaluate",
            "--mode",
            "model",
            "--task-type",
            LOCAL_TASK_TYPE,
            "--benchmark-name",
            LOCAL_BENCHMARK_NAME,
            "--data-path",
            str(data_root),
            "--num-samples",
            "1",
            "--output-dir",
            str(output_dir),
            "--model-id",
            "zoo-model",
            "--model-manifest-dir",
            str(manifest_dir),
            "--model-parameter",
            "artifact_ext=webm",
            "--metric",
            "artifact_count",
            "--metric",
            "has_artifact:generated_video",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["mode"] == "model"
    assert results[0]["artifacts"]["generated_video"]["uri"].endswith(".webm")
    assert scorecard["model"]["model_id"] == "zoo-model"
    assert scorecard["model"]["resolver"]["source"] == "model_zoo"


def test_run_cli_in_process_scores_task_materialization(tmp_path, monkeypatch, capsys) -> None:
    _patch_cli_imagetext_video_registry(monkeypatch, tmp_path)
    output_dir = tmp_path / "run"
    data_root = _write_local_video_dataset(tmp_path / "local-benchmark")

    exit_code = main(
        [
            "run",
            "--engine",
            "in-process",
            "--task-type",
            LOCAL_TASK_TYPE,
            "--benchmark-name",
            LOCAL_BENCHMARK_NAME,
            "--data-path",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--model-id",
            CONTRACT_FIXTURE_MODEL_ID,
            "--model-runner",
            CONTRACT_FIXTURE_RUNNER_TARGET,
            "--num-samples",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["engine"] == "in-process"
    assert payload["delegate_runner"] == "ResolvedWorldModelRunner+ExistingResultsRunner"
    assert payload["sample_count"] == 1
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["benchmark"]["benchmark_name"] == LOCAL_BENCHMARK_NAME
    assert scorecard["model"]["model_id"] == CONTRACT_FIXTURE_MODEL_ID


def test_run_cli_in_process_engine_resolves_contract_model(tmp_path, monkeypatch, capsys) -> None:
    _patch_cli_imagetext_video_registry(monkeypatch, tmp_path)
    output_dir = tmp_path / "run-model"
    data_root = _write_local_video_dataset(tmp_path / "local-benchmark")

    exit_code = main(
        [
            "run",
            "--engine",
            "in-process",
            "--task-type",
            LOCAL_TASK_TYPE,
            "--benchmark-name",
            LOCAL_BENCHMARK_NAME,
            "--data-path",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--model-id",
            CONTRACT_FIXTURE_MODEL_ID,
            "--model-runner",
            CONTRACT_FIXTURE_RUNNER_TARGET,
            "--num-samples",
            "1",
            "--metric",
            "artifact_count",
            "--metric",
            "has_artifact:generated_video",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["engine"] == "in-process"
    assert payload["mode"] == "model"
    assert payload["delegate_runner"] == "ResolvedWorldModelRunner+ExistingResultsRunner"
    assert scorecard["model"]["model_id"] == CONTRACT_FIXTURE_MODEL_ID
    assert scorecard["metrics"]["leaderboard"]["artifact_count"] == 1.0
    assert scorecard["metrics"]["leaderboard"]["has_artifact:generated_video"] == 1.0


def test_zoo_benchmark_run_help_exposes_contract_mode(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["zoo", "benchmark-run", "--help"])

    output = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert "--mode {contract,normalizer,official-run,official-validation}" in output
    assert "local schema" in output
    assert "checks" in output




def test_contract_runner_artifact_behavior_is_api_owned() -> None:
    from worldfoundry.evaluation.tasks.execution.orchestration.contract import execute_contract_run, run_contract

    assert execute_contract_run is run_contract


def test_zoo_help_exposes_benchmark_operations(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["zoo", "--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "benchmark-run" in captured.out
    assert "model-benchmark" not in captured.out
    assert "model-benchmark-suite" not in captured.out
    assert "model-show" in captured.out
    assert "benchmark-show" in captured.out
    assert "model-download" in captured.out
    assert "report" in captured.out
    assert "env-check" in captured.out


def test_zoo_models_list_prints_aliases(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "demo-model",
                        "name": "Demo Model",
                        "aliases": ["demo-alias"],
                        "source": {"status": "open_source"},
                        "checkpoint": {"hf_repo_id": "org/demo"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["zoo", "models", "--manifest-dir", str(manifest_dir)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "demo-alias, Demo Model" in output


def test_zoo_benchmarks_list_prints_aliases(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "demo-benchmark",
                        "name": "Demo Benchmark",
                        "aliases": ["DemoBench"],
                        "source": {"status": "open_source"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["zoo", "benchmarks", "--manifest-dir", str(manifest_dir)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DemoBench, Demo Benchmark" in output
    assert "benchmarks.yaml" not in output


def test_zoo_benchmarks_list_filters_ready_and_needs(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "needs-runner",
                        "source": {"status": "open_source"},
                        "requires": ["dataset"],
                    },
                    {
                        "benchmark_id": "dataset-ready",
                        "source": {"status": "open_source"},
                        "requires": ["dataset"],
                        "runner": {"runner_target": "demo.runner:main"},
                    },
                    {
                        "benchmark_id": "official-ready",
                        "source": {"status": "open_source"},
                        "integration_status": "integrated",
                        "official_benchmark_verified": True,
                        "integration_evidence": True,
                        "requires": ["dataset"],
                        "runner": {
                            "runner_target": "demo.official:main",
                            "verification_status": "verified",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmarks",
            "--manifest-dir",
            str(manifest_dir),
            "--ready-now",
            "--needs",
            "dataset",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [item["benchmark_id"] for item in payload] == ["official-ready"]


def test_zoo_benchmarks_list_filters_official_ready(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "contract-only",
                        "source": {"status": "open_source"},
                        "runner": {"runner_target": "demo.contract:main"},
                    },
                    {
                        "benchmark_id": "official-ready",
                        "source": {"status": "open_source"},
                        "integration_status": "integrated",
                        "official_benchmark_verified": True,
                        "integration_evidence": True,
                        "runner": {
                            "runner_target": "demo.official:main",
                            "verification_status": "verified",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["zoo", "benchmarks", "--manifest-dir", str(manifest_dir), "--official-ready", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [item["benchmark_id"] for item in payload] == ["official-ready"]


def test_tasks_list_exposes_benchmark_zoo_contract_surfaces(capsys) -> None:
    exit_code = main(["tasks", "list", "--source-kind", "benchmark_zoo", "--json"])
    payload = json.loads(capsys.readouterr().out)
    by_task = {item["task_type"]: item for item in payload}

    assert exit_code == 0
    assert by_task["vbench"]["suite"] == "benchmark_zoo"
    assert by_task["vbench"]["backend"] == "external_benchmark_contract"
    assert by_task["vbench"]["benchmark_name"] == "vbench"
    assert by_task["vbench"]["name"] == "vbench"
    assert by_task["vbench"]["source_kind"] == "benchmark_zoo"
    assert by_task["vbench"]["official_runtime_validated"] is True
    assert by_task["vbench"]["contract_only_surface"] is False
    assert by_task["worldmodelbench"]["requires_upstream_runtime"] is True
    assert by_task["worldscore"]["evaluation_protocol"] == "external_benchmark_contract"
    assert by_task["worldscore"]["official_runtime_validated"] is False
    assert by_task["rlbench"]["source_kind"] == "benchmark_zoo"
    assert by_task["metaworld"]["output_keys"] == [
        "scorecard",
        "raw_results",
        "per_sample_metrics",
        "rollout_logs",
    ]
    assert by_task["bridgedata-v2"]["benchmark_name"] == "bridgedata-v2"


def test_tasks_list_text_is_grouped_without_compatibility_aliases(capsys) -> None:
    exit_code = main(["tasks", "list"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "benchmark_zoo: external_benchmark_contract" in output
    assert "  vbench" in output
    assert "vbench/external" not in output
    assert "raw alias" not in output


def test_tasks_list_flat_exposes_canonical_registry_keys(capsys) -> None:
    exit_code = main(["tasks", "list", "--flat", "--source-kind", "benchmark_zoo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "vbench [benchmark_zoo/external_benchmark_contract]" in output
    assert "vbench/external" not in output
    assert "raw alias" not in output


def test_tasks_show_benchmark_zoo_uses_canonical_task_id(capsys) -> None:
    exit_code = main(["tasks", "show", "--task-type", "vbench", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["task_type"] == "vbench"
    assert payload["benchmark_name"] == "vbench"
    assert payload["name"] == "vbench"
    assert payload["source_kind"] == "benchmark_zoo"


def test_tasks_catalog_exports_active_registry(capsys) -> None:
    exit_code = main(["tasks", "catalog", "--source-kind", "benchmark_zoo", "--json"])
    payload = json.loads(capsys.readouterr().out)
    by_name = {item["name"]: item for item in payload}

    assert exit_code == 0
    assert by_name["vbench"]["schema_version"] == "worldfoundry-catalog-benchmark"
    task = by_name["vbench"]["tasks"][0]
    assert task["schema_version"] == "worldfoundry-catalog-task"
    assert task["name"] == "vbench"
    assert task["evaluation_protocol"][0]["name"] == "external_benchmark_contract"
    assert task["metadata"]["source_kind"] == "benchmark_zoo"
    assert task["metadata"]["requires_upstream_runtime"] is True


def test_cli_validation_writes_scorecard(tmp_path, capsys) -> None:
    output_dir = tmp_path / "evaluate"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    rows = []
    for index in range(2):
        artifact_path = artifact_dir / f"sample-{index}.txt"
        artifact_path.write_text(f"generated artifact {index}\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "task_name": "scorecard_contract",
                "status": "succeeded",
                "artifacts": {"generated_text": {"uri": str(artifact_path), "kind": "text"}},
            }
        )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    exit_code = main(
        [
            "evaluate",
            "--results-path",
            str(results_path),
            "--output-dir",
            str(output_dir),
            "--benchmark-id",
            "scorecard-contract-benchmark",
            "--model-id",
            "scorecard-contract-model",
            "--metric",
            "artifact_count",
            "--run-id",
            "test-scorecard-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-evaluate-run-result"
    assert payload["status"] == "succeeded"
    assert payload["sample_count"] == 2
    assert payload["scorecard_path"] == str(output_dir.resolve() / "scorecard.json")
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "execution_plan.json").is_file()
    assert (output_dir / "artifacts.jsonl").is_file()
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["run"]["run_id"] == "test-scorecard-run"
    assert scorecard["benchmark"]["benchmark_name"] == "scorecard-contract-benchmark"
    assert scorecard["metrics"]["leaderboard"]["artifact_count"] == 1.0
    assert scorecard["metrics"]["leaderboard"]["generation_success"] == 1.0
    assert scorecard["eligibility"]["score_valid"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert "missing official/full-suite leaderboard evidence gate" in scorecard["eligibility"]["reasons"]


def test_zoo_benchmark_validate_forwards_to_script(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 7

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/benchmark_zoo/validate_integration.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "benchmark-validate",
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--benchmark-id",
            "vbench",
            "--output-root",
            str(tmp_path / "validation"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--timeout",
            "11",
            "--clone-timeout",
            "22",
            "--depth",
            "2",
            "--execute-clone",
            "--fresh-clone",
            "--execute-download",
            "--execute-validation",
            "--disable-xet",
            "--allow-partial",
            "--dataset-id",
            "org/data",
            "--env",
            "FOO=bar",
            "--json",
        ]
    )

    assert exit_code == 7
    assert calls == [
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--benchmark-id",
            "vbench",
            "--output-root",
            str(tmp_path / "validation"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--timeout",
            "11",
            "--clone-timeout",
            "22",
            "--depth",
            "2",
            "--execute-clone",
            "--fresh-clone",
            "--dataset-id",
            "org/data",
            "--execute-download",
            "--disable-xet",
            "--execute-validation",
            "--env",
            "FOO=bar",
            "--allow-partial",
            "--json",
        ]
    ]


def test_zoo_model_env_check_forwards_to_script(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 5

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/model_zoo/env_check.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "env-check",
            "--kind",
            "model",
            "--id",
            "wan2.1",
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--checkpoint-repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--require-repo",
            "--require-checkpoint",
            "--require-demo",
            "--require-runner-demo",
            "--json",
        ]
    )

    assert exit_code == 5
    assert calls == [
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--model-id",
            "wan2.1",
            "--checkpoint-repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--require-repo",
            "--require-checkpoint",
            "--require-demo",
            "--require-runner-demo",
            "--json",
        ]
    ]


def test_zoo_model_env_check_resolves_registry_alias(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/model_zoo/env_check.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "env-check",
            "--kind",
            "model",
            "--id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--json",
        ]
    )

    assert exit_code == 0
    assert "--model-id" in calls[0]
    assert calls[0][calls[0].index("--model-id") + 1] == "wan2.1"


def test_zoo_benchmark_env_check_forwards_to_script(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 6

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/benchmark_zoo/env_check.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "env-check",
            "--kind",
            "benchmark",
            "--id",
            "vbench",
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--require-repo",
            "--require-dataset",
            "--require-validation",
            "--json",
        ]
    )

    assert exit_code == 6
    assert calls == [
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--benchmark-id",
            "vbench",
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--require-repo",
            "--require-dataset",
            "--require-validation",
            "--json",
        ]
    ]


def test_zoo_benchmark_validate_resolves_registry_alias(monkeypatch, tmp_path) -> None:
    calls = []
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "demo-benchmark",
                        "aliases": ["DemoBench"],
                        "source": {"status": "open_source"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/benchmark_zoo/validate_integration.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "benchmark-validate",
            "--manifest-dir",
            str(manifest_dir),
            "--benchmark-id",
            "demobench",
            "--json",
        ]
    )

    assert exit_code == 0
    assert "--benchmark-id" in calls[0]
    assert calls[0][calls[0].index("--benchmark-id") + 1] == "demo-benchmark"




def test_zoo_model_validate_forwards_to_script(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 9

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/model_zoo/validate_integration.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "model-validate",
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--model-id",
            "wan2.1",
            "--output-root",
            str(tmp_path / "validation"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--timeout",
            "11",
            "--clone-timeout",
            "22",
            "--depth",
            "2",
            "--execute-clone",
            "--fresh-clone",
            "--checkpoint-repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--execute-download",
            "--check-local",
            "--disable-xet",
            "--execute-official-demo",
            "--execute-runner-demo",
            "--allow-partial",
            "--json",
        ]
    )

    assert exit_code == 9
    assert calls == [
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--model-id",
            "wan2.1",
            "--output-root",
            str(tmp_path / "validation"),
            "--clone-root",
            str(tmp_path / "repos"),
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--timeout",
            "11",
            "--clone-timeout",
            "22",
            "--depth",
            "2",
            "--execute-clone",
            "--fresh-clone",
            "--checkpoint-repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--execute-download",
            "--check-local",
            "--disable-xet",
            "--execute-official-demo",
            "--execute-runner-demo",
            "--allow-partial",
            "--json",
        ]
    ]


def test_zoo_model_validate_resolves_registry_alias(monkeypatch) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/model_zoo/validate_integration.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "model-validate",
            "--model-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--allow-partial",
            "--json",
        ]
    )

    assert exit_code == 0
    assert "--model-id" in calls[0]
    assert calls[0][calls[0].index("--model-id") + 1] == "wan2.1"


def test_zoo_model_download_forwards_retry_options_to_script(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeScript:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 3

    def fake_load_repo_script(relative_path: str):
        assert relative_path == "scripts/model_zoo/download_checkpoints.py"
        return FakeScript

    monkeypatch.setattr(cli, "_load_repo_script", fake_load_repo_script)

    exit_code = main(
        [
            "zoo",
            "model-download",
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--model-id",
            "wan2.1",
            "--repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--execute",
            "--disable-xet",
            "--disable-hf-transfer",
            "--timeout",
            "120",
            "--retries",
            "2",
            "--max-workers",
            "1",
            "--check-local",
            "--report-path",
            str(tmp_path / "download-report.json"),
            "--json",
        ]
    )

    assert exit_code == 3
    assert calls == [
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--model-id",
            "wan2.1",
            "--repo-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--execute",
            "--disable-xet",
            "--disable-hf-transfer",
            "--timeout",
            "120",
            "--retries",
            "2",
            "--max-workers",
            "1",
            "--check-local",
            "--report-path",
            str(tmp_path / "download-report.json"),
            "--json",
        ]
    ]


def test_zoo_model_show_resolves_registry_alias_and_exports_manifest(capsys) -> None:
    exit_code = main(
        [
            "zoo",
            "model-show",
            "--model-id",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--include-manifest",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_id"] == "wan2.1"
    assert "Wan-AI/Wan2.1-T2V-1.3B" in payload["registry_aliases"]
    assert payload["world_model_manifest"]["model_id"] == "wan2.1"
    assert "hf_repo_ids" in payload["world_model_manifest"]["metadata"]


def test_zoo_benchmark_show_resolves_registry_alias_and_exports_spec(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "demo-benchmark",
                        "name": "Demo Benchmark",
                        "aliases": ["DemoBench"],
                        "source": {"status": "open_source"},
                        "dataset": {"not_applicable": True, "reason": "contract-only validation"},
                        "integration_status": "planned",
                        "runner": {
                            "verification_status": "pending",
                            "expected_artifacts": ["scorecard.json"],
                        },
                        "metrics": [{"metric_id": "quality", "higher_is_better": True}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmark-show",
            "--manifest-dir",
            str(manifest_dir),
            "--benchmark-id",
            "demobench",
            "--include-spec",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark_id"] == "demo-benchmark"
    assert payload["registry_aliases"] == ["DemoBench", "Demo Benchmark"]
    assert payload["benchmark_spec"]["name"] == "demo-benchmark"
    assert payload["benchmark_spec"]["tasks"][0]["metadata"]["benchmark_id"] == "demo-benchmark"
    assert payload["benchmark_spec"]["metrics"][0]["id"] == "quality"


def test_zoo_model_download_real_script_plan_only_handles_dataclasses(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "video_open.yaml").write_text(
        json.dumps(
            {
                "model_id": "demo-model",
                "source": {"status": "open_source"},
                "integration_status": "planned",
                "checkpoint": {"repos": [{"id": "org/demo", "sha": "abc1234"}]},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "download-report.json"

    exit_code = main(
        [
            "zoo",
            "model-download",
            "--manifest-dir",
            str(manifest_dir),
            "--model-id",
            "demo-model",
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--report-path",
            str(report_path),
            "--json",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert written == printed
    assert printed["schema_version"] == "worldfoundry-model-zoo-checkpoint-download-report"
    assert printed["results"][0]["commands"] == [
        [
            "hf",
            "download",
            "org/demo",
            "--cache-dir",
            str(tmp_path / "hfd"),
            "--revision",
            "abc1234",
            "--max-workers",
            "1",
        ]
    ]


def test_zoo_benchmark_runner_writes_contract_scorecard(tmp_path, capsys) -> None:
    output_dir = tmp_path / "runner"

    exit_code = main(
        [
            "zoo",
            "benchmark-run",
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(output_dir),
            "--mode",
            "contract",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["benchmark_id"] == "vbench"
    assert payload["ok"] is False
    assert payload["official_benchmark_verified"] is False
    assert payload["integration_evidence"] is False
    assert scorecard["benchmark"]["contract_only"] is True


def test_zoo_benchmark_runner_official_verified_executes_manifest_command(tmp_path, capsys) -> None:
    output_dir = tmp_path / "runner-official"
    manifest_path = tmp_path / "benchmarks.yaml"
    command = (
        "import json, os; "
        "from pathlib import Path; "
        "root=Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
        "scorecard={'schema_version':'worldfoundry-scorecard',"
        "'official_benchmark_verified':True,'integration_evidence':True,"
        "'run':{'status':'succeeded','command':os.environ['WORLDFOUNDRY_BENCHMARK_COMMAND_KIND']},"
        "'benchmark':{'benchmark_id':os.environ['WORLDFOUNDRY_BENCHMARK_ID']},"
        "'evaluation':{'available':True,'kind':'official_cli_demo'},"
        "'metrics':{'leaderboard':{'quality':1.0}}}; "
        "(root/'scorecard.json').write_text(json.dumps(scorecard), encoding='utf-8')"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", command],
                            "expected_artifacts": ["scorecard.json"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmark-run",
            "--manifest-dir",
            str(manifest_path),
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(output_dir),
            "--mode",
            "official-validation",
            "--timeout",
            "10",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    runtime_report = json.loads((output_dir / "runner_runtime_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["official_benchmark_verified"] is True
    assert payload["integration_evidence"] is True
    assert runtime_report["command_kind"] == "validation"
    assert runtime_report["run_status"] == "succeeded"


def test_zoo_benchmark_runner_requires_integration_evidence_for_success(tmp_path, capsys) -> None:
    output_dir = tmp_path / "runner-normalizer-only"
    manifest_path = tmp_path / "benchmarks.yaml"
    command = (
        "import json, os; "
        "from pathlib import Path; "
        "root=Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
        "scorecard={'schema_version':'worldfoundry-scorecard',"
        "'official_benchmark_verified':True,'integration_evidence':False,"
        "'run':{'status':'succeeded','command':os.environ['WORLDFOUNDRY_BENCHMARK_COMMAND_KIND']},"
        "'benchmark':{'benchmark_id':os.environ['WORLDFOUNDRY_BENCHMARK_ID']},"
        "'evaluation':{'available':True,'kind':'normalizer_only'},"
        "'metrics':{'leaderboard':{'quality':1.0}}}; "
        "(root/'scorecard.json').write_text(json.dumps(scorecard), encoding='utf-8')"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", command],
                            "expected_artifacts": ["scorecard.json"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmark-run",
            "--manifest-dir",
            str(manifest_path),
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(output_dir),
            "--mode",
            "official-validation",
            "--timeout",
            "10",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    runtime_report = json.loads((output_dir / "runner_runtime_report.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["official_benchmark_verified"] is True
    assert payload["integration_evidence"] is False
    assert runtime_report["official_benchmark_verified"] is True
    assert runtime_report["integration_evidence"] is False
    assert runtime_report["run_status"] == "succeeded"


def test_zoo_benchmark_runner_normalizes_official_results_path(tmp_path, capsys) -> None:
    output_dir = tmp_path / "runner-official-results"
    manifest_path = tmp_path / "benchmarks.yaml"
    official_results = tmp_path / "official_results.csv"
    official_results.write_text("sample_id,metric_id,score\nsample-a,quality,1.0\nsample-b,quality,0.5\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {"verification_status": "verified"},
                        "metrics": [{"id": "quality", "leaderboard_key": "quality"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmark-run",
            "--manifest-dir",
            str(manifest_path),
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(output_dir),
            "--mode",
            "official-validation",
            "--official-results-path",
            str(official_results),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["ok"] is False
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["metrics"]["leaderboard"]["quality"] == 0.75


def test_vbench_manifest_script_normalizes_existing_upstream_results(tmp_path, capsys) -> None:
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "aesthetic_quality": [
                    0.64,
                    [{"video_path": "sample.mp4", "video_results": 0.64}],
                ]
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "vbench-zoo"
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [
                                sys.executable,
                                str(REPO_ROOT / "scripts" / "benchmark_zoo" / "run_vbench_official_runner.py"),
                                "run",
                                "--videos-path",
                                str(videos_dir),
                                "--dimension",
                                "aesthetic_quality",
                                "--from-upstream-results",
                                str(upstream_results),
                                "--json",
                            ],
                            "expected_artifacts": ["scorecard.json", "raw_metric_table.jsonl"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "benchmark-run",
            "--manifest-dir",
            str(manifest_path),
            "--benchmark-id",
            "vbench",
            "--output-dir",
            str(output_dir),
            "--mode",
            "official-validation",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["benchmark_id"] == "vbench"
    assert payload["ok"] is False
    assert payload["official_benchmark_verified"] is False
    assert payload["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["run"]["status"] == "normalized"
    assert scorecard["evaluation"]["kind"] == "official_vbench"


def test_vbench_script_lists_dimensions_and_plan_onlys_external_setup(tmp_path, capsys) -> None:
    module = cli._load_repo_script("worldfoundry/evaluation/tasks/execution/runners/vbench/run_vbench_official_runner.py")

    exit_code = module.dimensions_main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark_id"] == "vbench"
    assert payload["presets"]["validation"] == ["aesthetic_quality"]
    assert "full_16" in payload["presets"]

    setup_exit_code = module.setup_main(
        [
            "--vbench-root",
            str(tmp_path / "repos" / "vbench"),
            "--plan-only",
            "--json",
        ]
    )
    setup_payload = json.loads(capsys.readouterr().out)

    assert setup_exit_code == 0
    assert setup_payload["status"] == "planned_clone"
    assert setup_payload["commands"][0][:3] == ["git", "clone", "https://github.com/Vchitect/VBench.git"]
