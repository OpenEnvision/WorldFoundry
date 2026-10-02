"""Integration tests for the vendored PAWBench evaluator and runner wiring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.contracts import PAWBenchContract, get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.integration import (
    BENCHMARK_INTEGRATION_REGISTRY,
    IntegrationTier,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.tasks.execution.runners.pawbench.run_pawbench_official_runner import (
    CONFIG,
    DEFAULT_RUNTIME_ROOT,
    METRIC_IDS,
    build_official_command,
    build_official_environment,
    extract_metrics,
    main,
    prepare_upstream_results,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import (
    CLI_RUNNERS,
    build_workspace_benchmark_command,
)
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, REPO_ROOT

FIXTURE_PATH = REPO_ROOT / "worldfoundry/data/benchmarks/assets/pawbench/sample_results.json"


def _fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _vendored_metrics_module() -> ModuleType:
    path = DEFAULT_RUNTIME_ROOT / "pawbench" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_worldfoundry_vendored_pawbench_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_contract_and_execution_registries_expose_pawbench() -> None:
    entry = load_benchmark_zoo_registry(BENCHMARK_ZOO_DIR).get("pawbench")
    assert entry.name == "PAWBench"
    assert tuple(metric.metric_id for metric in entry.metrics) == METRIC_IDS
    assert entry.runner.runtime["repo_revision"] == "f9ae040227a5b76ee50e37871fbbefc89fcc7fea"
    assert entry.runner.runtime["kind"] == "in_tree_official_judge_runtime"
    assert entry.base_model_dependencies == ()
    assert entry.optional_base_model_dependencies == ()

    assert PAWBenchContract == get_external_benchmark_contract("pawbench")
    assert PAWBenchContract.metric_ids == METRIC_IDS
    assert VIDEO_RUNNER_REGISTRY["pawbench"].script.endswith("pawbench/run_pawbench_official_runner.py")
    integration = BENCHMARK_INTEGRATION_REGISTRY["pawbench"]
    assert integration.tier is IntegrationTier.MODEL_BACKED
    assert integration.hf_dataset_id == "Andrew613/PAWBench"
    assert integration.judge_model_id == "google/gemini-3.5-flash"

    workspace = CLI_RUNNERS["pawbench"]
    assert workspace.generated_arg == "--generated-artifact-dir"
    assert workspace.dataset_root_arg == "--dataset-root"
    assert workspace.model_arg == "--model-name"
    assert workspace.supports_official_runtime is True
    assert workspace.accepts_generated_artifacts is True
    assert workspace.supports_fixture is True


def test_metadata_keeps_hosted_judge_and_model_dependencies_explicit() -> None:
    paths = (
        REPO_ROOT / "worldfoundry/data/benchmarks/catalog/video/pawbench.yaml",
        REPO_ROOT / "worldfoundry/data/benchmarks/tasks/external/pawbench.yaml",
        REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official/pawbench.yaml",
    )
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    catalog, task, profile = payloads

    assert catalog["integration"]["tier"] == "model_backed"
    assert catalog["runner"]["runtime"]["kind"] == "in_tree_official_runtime_hosted_judges"
    assert "--benchmark-data-root" in catalog["runner"]["run_command"]
    assert task["metadata"]["runtime"]["kind"] == "in_tree_official_runtime_hosted_judges"
    assert "OPENROUTER_API_KEY" in profile["optional_env"]
    for payload in (catalog, profile):
        assert payload["checkpoint_refs"] == []
        assert payload["base_model_dependencies"] == []
        assert payload["optional_base_model_dependencies"] == []
    assert task["metadata"]["checkpoint_refs"] == []
    assert task["metadata"]["base_model_dependencies"] == []
    assert task["metadata"]["optional_base_model_dependencies"] == []


def test_vendored_runtime_has_rubrics_and_no_cache_files() -> None:
    assert (DEFAULT_RUNTIME_ROOT / "evaluate.py").is_file()
    for track in ("outcome", "trustworthiness"):
        rubrics = list((DEFAULT_RUNTIME_ROOT / "pawbench/paweval/rubrics" / track).glob("*.yaml"))
        assert len(rubrics) == 50
    assert not list(DEFAULT_RUNTIME_ROOT.rglob("*.pyc"))
    assert not list(DEFAULT_RUNTIME_ROOT.rglob("__pycache__"))
    assert not list(DEFAULT_RUNTIME_ROOT.rglob(".git"))


def test_vendored_upstream_evaluate_help_runs_from_in_tree_runtime() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEFAULT_RUNTIME_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(DEFAULT_RUNTIME_ROOT / "evaluate.py"), "--help"],
        cwd=DEFAULT_RUNTIME_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Evaluate a directory of PAWBench video rollouts" in completed.stdout
    assert "--vlm-api-key-env" in completed.stdout


def test_extracts_official_track_metrics_without_inventing_an_average() -> None:
    metrics = extract_metrics(_fixture_payload(), Path("metrics.json"))
    assert set(metrics) == set(METRIC_IDS)
    assert metrics["calibration_tvd_percent"]["raw_score"] == 12.5
    assert metrics["calibration_tvd_percent"]["normalized_score"] == 0.125
    assert metrics["coverage_percent"]["raw_score"] == 84.0
    assert metrics["coverage_percent"]["normalized_score"] == 0.84
    assert metrics["coverage_percent"]["sample_count"] == 25
    assert all(item["model_name"] == "fixture-model" for item in metrics.values())


def test_rejects_blocked_or_cross_model_official_reports() -> None:
    blocked = _fixture_payload()
    blocked["status"] = "blocked"
    blocked["blockers"] = ["coverage:infrastructure_failure"]
    with pytest.raises(ValueError, match="blocked"):
        extract_metrics(blocked, Path("metrics.json"))

    mismatched = _fixture_payload()
    coverage = mismatched["tracks"]["coverage"]["models"]
    coverage["other-model"] = coverage.pop("fixture-model")
    with pytest.raises(ValueError, match="different evaluated models"):
        extract_metrics(mismatched, Path("metrics.json"))

    incomplete = _fixture_payload()
    incomplete_pass_rate = incomplete["tracks"]["coverage"]["models"]["fixture-model"]["scene_pass_rate"]
    incomplete_pass_rate.update({"passing_scenes": 24, "value": 96.0})
    with pytest.raises(ValueError, match="incomplete"):
        extract_metrics(incomplete, Path("metrics.json"))


def test_report_directory_prefers_canonical_metrics_file(tmp_path: Path) -> None:
    direct = tmp_path / "metrics.json"
    direct.write_text("{}", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "metrics.json").write_text("{}", encoding="utf-8")
    assert prepare_upstream_results(CONFIG, tmp_path, None, tmp_path / "output") == direct

    run_path = tmp_path / "run.json"
    run_path.write_text("{}", encoding="utf-8")
    assert prepare_upstream_results(CONFIG, run_path, None, tmp_path / "output") == direct


def test_fixture_normalization_writes_two_metric_scorecard(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert main(["--run-fixture", "--output-dir", str(output_dir)]) == 0
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["benchmark"]["benchmark_id"] == "pawbench"
    assert scorecard["run"]["status"] == "normalized"
    assert set(scorecard["metrics"]["per_metric"]) == set(METRIC_IDS)
    assert scorecard["metrics"]["per_metric"]["calibration_tvd_percent"]["normalized_score"] == 0.125
    assert scorecard["metrics"]["per_metric"]["coverage_percent"]["normalized_score"] == 0.84
    assert "pawbench_average" not in scorecard["metrics"]["per_metric"]
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    for artifact in ("raw_metric_table.jsonl", "per_sample_scores.jsonl"):
        assert (output_dir / artifact).is_file()


def test_official_command_routes_dataset_rollouts_judge_and_mutable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text("{}", encoding="utf-8")
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setenv("TEST_PAWBENCH_API_KEY", "secret-not-for-command-line")
    args = argparse.Namespace(
        python=sys.executable,
        dataset_root=dataset_root,
        model_name="test-model",
        vlm_base_url="https://openrouter.ai/api/v1",
        vlm_model="google/gemini-3.5-flash",
        vlm_api_key_env="TEST_PAWBENCH_API_KEY",
    )
    command = build_official_command(
        config=CONFIG,
        repo_root=DEFAULT_RUNTIME_ROOT,
        generated_video_dir=rollouts,
        output_dir=output_dir,
        args=args,
    )
    assert command is not None
    assert command[:2] == [sys.executable, str(DEFAULT_RUNTIME_ROOT / "evaluate.py")]
    assert command[command.index("--benchmark") + 1] == str(dataset_root.resolve())
    assert command[command.index("--videos") + 1] == str(rollouts.resolve())
    assert command[command.index("--output") + 1] == str((output_dir / "upstream").resolve())
    assert command[command.index("--model") + 1] == "test-model"
    assert command[command.index("--vlm-api-key-env") + 1] == "TEST_PAWBENCH_API_KEY"
    assert "secret-not-for-command-line" not in command

    command_from_catalog_root = build_official_command(
        config=CONFIG,
        repo_root=DEFAULT_RUNTIME_ROOT.parent.parent,
        generated_video_dir=rollouts,
        output_dir=output_dir,
        args=args,
    )
    assert command_from_catalog_root is not None
    assert command_from_catalog_root[1] == str(DEFAULT_RUNTIME_ROOT / "evaluate.py")

    environment = build_official_environment(
        config=CONFIG,
        repo_root=DEFAULT_RUNTIME_ROOT,
        generated_video_dir=rollouts,
        output_dir=output_dir,
        args=args,
    )
    assert Path(environment["TMPDIR"]).is_relative_to(output_dir)


def test_workspace_command_routes_dataset_rollouts_and_model() -> None:
    command = build_workspace_benchmark_command(
        {
            "benchmark_id": "pawbench",
            "dataset_root": "/benchmark/pawbench",
            "model_id": "demo-world-model",
            "params": {
                "generated_artifact_dir": "/rollouts",
                "run_official": True,
            },
        },
        "/output",
    )
    assert command[command.index("--dataset-root") + 1] == "/benchmark/pawbench"
    assert command[command.index("--generated-artifact-dir") + 1] == "/rollouts"
    assert command[command.index("--model-name") + 1] == "demo-world-model"
    assert "--run-official" in command


def test_packaging_declares_vendored_runtime_and_license() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    notices = (REPO_ROOT / "THIRD-PARTY-NOTICES").read_text(encoding="utf-8")
    assert "runners/pawbench/runtime" in manifest
    assert '"pawbench/runtime/**/*"' in pyproject
    assert "PAWBench" in notices
    assert "f9ae040227a5b76ee50e37871fbbefc89fcc7fea" in notices


def test_vendored_compute_metrics_enforces_and_scores_full_released_grid() -> None:
    metrics_module = _vendored_metrics_module()
    repeats = list(range(50))
    policy: dict[str, object] = {
        "model_or_lanes": ["synthetic-model"],
        "tracks": {
            "calibration": {
                "scenes": [
                    {
                        "scene_id": f"cal-{index:02d}",
                        "track": "calibration",
                        "group": "synthetic",
                        "expected_repeat_indices": repeats,
                        "reference_distribution": {"left": 0.5, "right": 0.5},
                    }
                    for index in range(25)
                ]
            },
            "coverage": {
                "scenes": [
                    {
                        "scene_id": f"cov-{index:02d}",
                        "track": "coverage",
                        "group": "synthetic",
                        "expected_repeat_indices": repeats,
                        "support_labels": ["left", "right"],
                    }
                    for index in range(25)
                ]
            },
        },
    }
    rows = [
        {
            "sample_id": f"synthetic-model::{track}::{scene['scene_id']}::r{repeat:03d}",
            "scene_id": scene["scene_id"],
            "track": track,
            "model_or_lane": "synthetic-model",
            "repeat_index": repeat,
            "observation": "outcome",
            "outcome_label": "left" if repeat % 2 == 0 else "right",
        }
        for track, track_policy in policy["tracks"].items()
        for scene in track_policy["scenes"]
        for repeat in repeats
    ]
    assert len(rows) == 2_500
    result = metrics_module.compute_metrics(rows, policy)
    assert result["status"] == "ok"
    calibration = result["tracks"]["calibration"]["models"]["synthetic-model"]
    coverage = result["tracks"]["coverage"]["models"]["synthetic-model"]
    assert calibration["track_average"] == {
        "name": "calibration_tvd_percent",
        "value": 0.0,
    }
    assert coverage["track_average"] == {"name": "coverage_percent", "value": 100.0}
    assert calibration["scene_pass_rate"]["scene_denominator"] == 25
    assert coverage["scene_pass_rate"]["scene_denominator"] == 25
