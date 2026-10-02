from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import load_benchmark_catalog_shard_entries
from worldfoundry.evaluation.tasks.catalog.schema import (
    BenchmarkRunnerSpec,
    BenchmarkDatasetRef,
    BenchmarkMetricSpec,
    BenchmarkSource,
    BenchmarkZooEntry,
    OPEN_SOURCE_STATUSES,
    load_entries,
)
from worldfoundry.evaluation.tasks.catalog.specs import (
    benchmark_zoo_entries_to_benchmark_specs,
    benchmark_zoo_entry_to_benchmark_spec,
    load_benchmark_specs,
)
from worldfoundry.evaluation.tasks.catalog.zoo_registry import (
    BenchmarkZooRegistry,
    clear_benchmark_zoo_registry_cache,
    load_benchmark_zoo_registry,
)
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.api import BenchmarkSpec
from worldfoundry.cli import zoo as zoo_cli


NEW_VIDEO_WORLD_NORMALIZER_IDS = (
    "aigcbench",
    "devil-dynamics",
    "ewmbench",
    "mirabench",
    "genai-bench",
    "phygenbench",
    "phyeduvideo",
    "physvidbench",
    "t2v-safety-bench",
    "t2vworldbench",
    "videoscience-bench",
    "videophy",
    "videophy2",
    "physics-iq",
    "ipv-bench",
    "phyground",
    "worldarena",
    "world-in-world",
)

NEW_VIDEO_WORLD_CONTRACT_IDS = ()

# Contracts that compute their metrics fully in-tree and therefore do not
# require an upstream benchmark checkout at scoring time.
IN_TREE_COMPUTED_CONTRACT_IDS = frozenset({"worldbench", "physics-iq"})


def _command_text(command: str | tuple[str, ...]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(command)


def _command_contains(command: str | tuple[str, ...], token: str) -> bool:
    if isinstance(command, str):
        return token in command
    return token in command


def test_benchmark_zoo_entry_roundtrips_nested_dict_and_json() -> None:
    entry = BenchmarkZooEntry(
        benchmark_id="example-benchmark",
        name="Example Benchmark",
        aliases=("example",),
        domains=("navigation",),
        modalities=("video", "text"),
        tags=("validation", "simulator"),
        source=BenchmarkSource(
            status="open_source",
            official_repo_url="https://example.invalid/repo",
            paper_url="https://example.invalid/paper",
            requires_auth=False,
            notes=("source note",),
        ),
        dataset=BenchmarkDatasetRef(
            hf_dataset_id="org/example-benchmark",
            revision="main",
            split="test",
            path="cases",
            requires_auth=True,
            notes=("dataset note",),
        ),
        integration_status="integrated",
        runner=BenchmarkRunnerSpec(
            install_profile="default",
            runner_target="worldfoundry.evaluation.example",
            run_command="conda run -n worldfoundry example",
            verification_status="verified",
            notes=("runner note",),
        ),
        metrics=(
            BenchmarkMetricSpec(
                metric_id="success_rate",
                name="Success Rate",
                description="Task completion rate.",
                higher_is_better=True,
                leaderboard_key="success_rate",
                normalizer="identity",
                primary=True,
            ),
        ),
        notes=("entry note",),
    )

    as_dict = entry.to_dict()
    assert as_dict["source"]["status"] == "open_source"
    assert as_dict["aliases"] == ["example"]
    assert as_dict["dataset"]["hf_dataset_id"] == "org/example-benchmark"
    assert as_dict["runner"]["verification_status"] == "verified"
    assert as_dict["metrics"][0]["metric_id"] == "success_rate"
    assert as_dict["metrics"][0]["leaderboard_key"] == "success_rate"
    assert as_dict["metrics"][0]["normalizer"] == "identity"
    assert as_dict["open_source_status"] == "planned"
    assert as_dict["official_benchmark_verified"] is False
    assert as_dict["integration_evidence"] is False
    assert as_dict["leaderboard_valid"] is False

    assert BenchmarkZooEntry.from_dict(as_dict) == entry
    assert BenchmarkZooEntry.from_json(entry.to_json()) == entry


def test_benchmark_zoo_entry_accepts_flat_manifest_fields() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "flat-example",
            "alias": "FlatBench",
            "domain": "navigation",
            "modality": "video",
            "tags": "validation",
            "source_status": "api",
            "official_repo_url": "https://example.invalid/api",
            "paper_url": "https://example.invalid/paper",
            "hf_dataset_id": "org/flat-example",
            "requires_auth": True,
            "open_source_status": "gated",
            "release_status": "normalizer_only",
            "maturity": "contract_ready",
            "official_benchmark_verified": True,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "base_model_dependencies": ["depth_stack"],
            "optional_base_model_dependencies": ["detection_stack"],
            "requires": ["API_KEY"],
            "blockers": ["runtime validation pending"],
            "data_refs": {"task_yaml": "data/benchmarks/tasks/external/flat-example.yaml"},
            "runner_availability": {"available": True, "task_yaml_available": True},
            "integration_status": "planned",
            "install_profile": "api",
            "runner_target": "worldfoundry.evaluation.flat",
            "run_command": "conda run -n worldfoundry flat",
            "verification_status": "pending",
            "metrics": [{"id": "score", "higher_is_better": False}],
            "notes": "flat note",
        }
    )

    assert entry.benchmark_id == "flat-example"
    # contract_validation_command is a deprecated legacy manifest field; it is
    # only populated when explicitly declared and no longer synthesized.
    assert entry.contract_validation_command is None
    assert entry.ready_now_command is None
    assert entry.one_click_command is None
    assert entry.aliases == ("FlatBench",)
    assert entry.domains == ("navigation",)
    assert entry.modalities == ("video",)
    assert entry.tags == ("validation",)
    assert entry.source.status == "api"
    assert entry.source_status == "api"
    assert entry.official_repo_url == "https://example.invalid/api"
    assert entry.paper_url == "https://example.invalid/paper"
    assert entry.hf_dataset_id == "org/flat-example"
    assert entry.requires_auth is True
    assert entry.open_source_status == "gated"
    assert entry.release_status == "normalizer_only"
    assert entry.maturity == "contract_ready"
    assert entry.official_benchmark_verified is True
    assert entry.integration_evidence is False
    assert entry.leaderboard_valid is False
    assert entry.base_model_dependencies == ("depth_stack",)
    assert entry.optional_base_model_dependencies == ("detection_stack",)
    assert entry.requires == ("API_KEY",)
    assert entry.blockers == ("runtime validation pending",)
    assert entry.data_refs["task_yaml"] == "data/benchmarks/tasks/external/flat-example.yaml"
    assert entry.runner_availability["task_yaml_available"] is True
    assert entry.install_profile == "api"
    assert entry.runner_target == "worldfoundry.evaluation.flat"
    assert entry.run_command == "conda run -n worldfoundry flat"
    assert entry.verification_status == "pending"
    assert entry.metrics[0].metric_id == "score"
    assert entry.notes == ("flat note",)


def test_benchmark_zoo_entry_accepts_nested_official_sources_manifest() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "vbench-2.0",
            "name": "VBench-2.0",
            "benchmark_kind": ["physics", "video-generation-quality"],
            "status": "confirmed_official_code_and_hf_data",
            "official_sources": {
                "github": {"url": "https://github.com/Vchitect/VBench"},
                "project_page": "https://vchitect.github.io/VBench-2.0-project/",
                "huggingface_datasets": [
                    {"repo_id": "Vchitect/VBench-2.0_sampled_videos", "gated": False},
                    {"repo_id": "Vchitect/VBench-2.0_human_annotation", "gated": False},
                ],
            },
            "integration": {
                "status": "blocked",
                "blocked_reasons": ["needs heavyweight dependency isolation"],
            },
        }
    )

    assert entry.benchmark_id == "vbench-2.0"
    assert entry.source.status == "open_source"
    assert entry.official_repo_url == "https://github.com/Vchitect/VBench"
    assert entry.paper_url == "https://vchitect.github.io/VBench-2.0-project/"
    assert entry.hf_dataset_id == "Vchitect/VBench-2.0_sampled_videos"
    assert entry.integration_status == "blocked"
    assert entry.tags == ("physics", "video-generation-quality")
    assert entry.notes == ("blocked: needs heavyweight dependency isolation",)


def test_benchmark_zoo_entry_normalizes_status_aliases_and_hf_dataset_urls() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "alias-benchmark",
            "benchmark_kind": "video-generation-quality",
            "status": "confirmed_public_hf_dataset",
            "official_sources": {
                "github": "https://github.com/example/alias-benchmark",
                "huggingface_datasets": [
                    "https://huggingface.co/datasets/org/alias-benchmark",
                ],
            },
            "integration": {"status": "pending_runner"},
            "runner": {"status": "pending_validation"},
        }
    )

    assert entry.source.status == "open_source"
    assert entry.official_repo_url == "https://github.com/example/alias-benchmark"
    assert entry.hf_dataset_id == "org/alias-benchmark"
    assert entry.integration_status == "planned"
    assert entry.verification_status == "pending"
    assert entry.tags == ("video-generation-quality",)


def test_benchmark_zoo_entry_preserves_list_run_command() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "list-command-benchmark",
            "status": "confirmed_official_code",
            "runner": {
                "run_command": ["python", "evaluate.py", "--validation"],
                "status": "pending_validation",
                "blocked_reasons": ["expected artifact contract missing"],
            },
        }
    )

    assert entry.run_command == ("python", "evaluate.py", "--validation")
    assert entry.to_dict()["runner"]["run_command"] == ["python", "evaluate.py", "--validation"]
    assert entry.runner.notes == ("blocked: expected artifact contract missing",)


def test_benchmark_zoo_entry_preserves_validation_artifact_contract() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "artifact-contract-benchmark",
            "status": "confirmed_official_code",
            "runner": {
                "run_command": ["python", "evaluate.py"],
                "validation_command": ["python", "evaluate.py", "--validation"],
                "expected_artifacts": [{"path": "score.json", "sha256": "0" * 64}],
            },
        }
    )

    assert entry.run_command == ("python", "evaluate.py")
    assert entry.validation_command == ("python", "evaluate.py", "--validation")
    assert entry.expected_artifacts == ({"path": "score.json", "sha256": "0" * 64},)
    assert entry.to_dict()["runner"]["expected_artifacts"] == [{"path": "score.json", "sha256": "0" * 64}]


def test_benchmark_zoo_entry_preserves_multiple_dataset_refs() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "multi-dataset-benchmark",
            "status": "confirmed_official_code_and_hf_data",
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/data-a", "sha": "abc123", "license": "mit", "gated": False},
                    {"repo_id": "org/data-b", "sha": "def456", "license": "apache-2.0", "gated": "auto"},
                ],
            },
        }
    )

    assert entry.hf_dataset_id == "org/data-a"
    assert entry.hf_dataset_ids == ("org/data-a", "org/data-b")
    assert entry.dataset_refs[0].revision == "abc123"
    assert entry.dataset_refs[1].license == "apache-2.0"
    assert entry.dataset_refs[1].requires_auth is True


def test_benchmark_zoo_loads_video_world_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog"

    entries = load_benchmark_catalog_shard_entries("video", catalog_root)

    assert len(entries) >= 10
    by_id = {entry.benchmark_id: entry for entry in entries}
    assert by_id["vbench"].official_repo_url == "https://github.com/Vchitect/VBench"
    assert by_id["vbench"].integration_status == "integrated"
    assert by_id["vbench"].verification_status == "verified"
    assert by_id["vbench"].open_source_status == "stable"
    # official_benchmark_verified stays false until full-suite verification is
    # recorded in the manifest (bounded validation only sets
    # integration_evidence).
    assert by_id["vbench"].official_benchmark_verified is False
    assert by_id["vbench"].integration_evidence is True
    assert by_id["vbench"].leaderboard_valid is False
    assert by_id["worldmodelbench"].hf_dataset_id == "Efficient-Large-Model/worldmodelbench"
    assert by_id["worldmodelbench"].open_source_status == "in_tree_runtime"
    assert by_id["video-bench"].integration_status == "integrated"
    assert by_id["video-bench"].verification_status == "verified"
    assert by_id["video-bench"].open_source_status == "gated"
    assert by_id["video-bench"].official_benchmark_verified is False
    assert by_id["video-bench"].integration_evidence is False
    assert by_id["video-bench"].leaderboard_valid is False
    assert by_id["vbench"].metrics[0].metric_id == "overall_quality"
    vbench_metric_ids = tuple(metric.metric_id for metric in by_id["vbench"].metrics)
    assert vbench_metric_ids == get_external_benchmark_contract("vbench").metric_ids
    assert len(vbench_metric_ids) == 20
    assert by_id["vbench"].dataset.not_applicable is True
    assert by_id["vbench"].dataset.reason
    assert by_id["vbench"].runner_target
    assert by_id["vbench"].validation_command is not None
    assert "run_vbench_official_runner.py" in _command_text(by_id["vbench"].validation_command)
    assert _command_contains(by_id["vbench"].validation_command, "--videos-path")
    assert by_id["vbench"].runner_runtime["repo_url"] == "https://github.com/Vchitect/VBench.git"
    assert by_id["vbench"].runner_runtime["override_root_env"] == "WORLDFOUNDRY_VBENCH_ROOT"
    assert by_id["vbench"].runner_runtime["dimension_presets"]["validation"] == ["aesthetic_quality"]
    assert by_id["vbench"].runner.repo_url == "https://github.com/Vchitect/VBench.git"
    assert by_id["vbench"].runner.repo_revision == "45e79ec14e69a2187202c675d2dbce1a71843d53"
    assert by_id["vbench"].runner.install_commands[0][-1] == "setup"
    assert by_id["vbench-plus-plus"].validation_command is not None
    assert "run_vbench_plus_plus_official_runner.py" in _command_text(by_id["vbench-plus-plus"].validation_command)
    assert by_id["vbench-2.0"].validation_command is not None
    assert "run_vbench_2_0_official_runner.py" in _command_text(by_id["vbench-2.0"].validation_command)
    assert by_id["vbench-2.0"].integration_status == "integrated"
    assert by_id["vbench-2.0"].verification_status == "verified"
    assert by_id["vbench-2.0"].open_source_status == "experimental"
    assert by_id["vbench-2.0"].release_status == "experimental"
    assert by_id["vbench-2.0"].maturity == "contract_ready"
    assert by_id["vbench-2.0"].official_benchmark_verified is False
    assert by_id["vbench-2.0"].integration_evidence is True
    assert by_id["vbench-2.0"].leaderboard_valid is False
    vbench2_validation = by_id["vbench-2.0"].runner.assets["official_gpu_validation"]
    assert vbench2_validation["status"] == "verified"
    assert vbench2_validation["scope"] == "bounded_official_gpu_validation"
    assert vbench2_validation["dimension"] == "Diversity"
    assert vbench2_validation["generated_video_count"] == 20
    assert vbench2_validation["full_suite_verified"] is False
    assert vbench2_validation["leaderboard_valid"] is False
    assert by_id["worldmodelbench"].validation_command is not None
    assert "benchmark-run" in _command_text(by_id["worldmodelbench"].validation_command)
    assert _command_contains(by_id["worldmodelbench"].validation_command, "--official-results-path")
    assert by_id["worldscore"].verification_status == "verified"
    assert by_id["worldscore"].integration_status == "integrated"
    assert by_id["worldscore"].open_source_status == "experimental"
    assert by_id["worldscore"].release_status == "experimental"
    assert by_id["worldscore"].maturity == "contract_ready"
    assert by_id["worldscore"].official_benchmark_verified is False
    assert by_id["worldscore"].integration_evidence is False
    worldscore_validation = by_id["worldscore"].runner.assets["official_gpu_validation"]
    assert worldscore_validation["status"] == "verified"
    assert worldscore_validation["scope"] == "bounded_official_gpu_validation"
    assert worldscore_validation["split"] == "dynamic"
    assert worldscore_validation["generated_video_count"] == 1
    assert worldscore_validation["metrics_available"] == 4
    assert worldscore_validation["full_suite_verified"] is False
    assert worldscore_validation["leaderboard_valid"] is False
    assert by_id["worldscore"].runner.clone_dir == "worldfoundry/evaluation/tasks/execution/runners/worldscore/runtime/worldscore"
    assert by_id["worldscore"].run_command is not None
    assert "run_worldscore_official_runner.py" in _command_text(by_id["worldscore"].run_command)
    assert by_id["worldscore"].validation_command is not None
    assert "benchmark-run" in _command_text(by_id["worldscore"].validation_command)
    assert _command_contains(by_id["worldscore"].validation_command, "--official-results-path")
    assert by_id["video-bench"].validation_command is not None
    assert "benchmark-run" in _command_text(by_id["video-bench"].validation_command)
    assert _command_contains(by_id["video-bench"].validation_command, "--official-results-path")
    assert by_id["worldbench"].validation_command is not None
    # worldbench computes its metrics fully in-tree: validation invokes the
    # in-tree official runner module directly instead of the benchmark-run
    # results-import path.
    assert "run_worldbench_official_runner" in _command_text(by_id["worldbench"].validation_command)
    assert _command_contains(by_id["worldbench"].validation_command, "--run-official")
    assert by_id["videoscore"].validation_command is not None
    assert by_id["videoscore"].open_source_status == "experimental"
    assert "run_videoscore_official_runner.py" in _command_text(by_id["videoscore"].validation_command)
    assert by_id["t2v-compbench"].validation_command is not None
    assert by_id["t2v-compbench"].open_source_status == "in_tree_runtime"
    assert "benchmark-run" in _command_text(by_id["t2v-compbench"].validation_command)
    assert _command_contains(by_id["t2v-compbench"].validation_command, "--official-results-path")
    assert by_id["camerabench"].validation_command is not None
    assert "benchmark-run" in _command_text(by_id["camerabench"].validation_command)
    assert _command_contains(by_id["camerabench"].validation_command, "--official-results-path")
    assert _command_contains(by_id["camerabench"].validation_command, "--benchmark-data-root")
    assert by_id["chronomagic-bench"].validation_command is not None
    assert by_id["chronomagic-bench"].open_source_status == "experimental"
    chronomagic_validation = by_id["chronomagic-bench"].runner.assets["official_gpu_validation"]
    assert chronomagic_validation["component"] == "chscore"
    assert chronomagic_validation["generated_video_count"] == 2
    assert chronomagic_validation["full_suite_verified"] is False
    assert chronomagic_validation["leaderboard_valid"] is False
    assert "run_chronomagic_bench_official_runner.py" in _command_text(
        by_id["chronomagic-bench"].validation_command
    )
    assert by_id["vmbench"].validation_command is not None
    assert "benchmark-run" in _command_text(by_id["vmbench"].validation_command)
    assert _command_contains(by_id["vmbench"].validation_command, "--official-results-path")
    assert by_id["vbench-plus-plus"].metrics[-1].metric_id == "vbench_plus_plus_average"
    assert by_id["vbench-plus-plus"].metrics[-1].primary is True
    assert by_id["vbench-2.0"].metrics[-1].metric_id == "vbench2_total"
    assert by_id["vbench-2.0"].metrics[-1].primary is True
    assert by_id["worldscore"].metrics[-1].primary is True
    assert by_id["video-bench"].metrics[-1].metric_id == "videobench_average"
    assert by_id["video-bench"].metrics[-1].primary is True
    # worldbench reports the official per-track metrics (no synthesized
    # average); the primary metric is foreground_miou.
    assert by_id["worldbench"].metrics[0].metric_id == "foreground_miou"
    assert by_id["worldbench"].metrics[0].primary is True
    assert by_id["worldbench"].metrics[-1].metric_id == "binary_accuracy"
    assert by_id["videoscore"].metrics[-1].primary is True
    assert by_id["t2v-compbench"].metrics[-1].metric_id == "t2v_compbench_average"
    assert by_id["t2v-compbench"].metrics[-1].primary is True
    assert by_id["camerabench"].metrics[-1].metric_id == "camerabench_average"
    assert by_id["camerabench"].metrics[-1].primary is True
    assert by_id["vmbench"].metrics[-1].metric_id == "vmbench_average"
    assert by_id["vmbench"].metrics[-1].primary is True


def test_new_video_world_normalizers_do_not_claim_leaderboard_verification() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog"
    entries = load_benchmark_catalog_shard_entries("video", catalog_root)
    by_id = {entry.benchmark_id: entry for entry in entries}

    assert set(NEW_VIDEO_WORLD_NORMALIZER_IDS) <= set(by_id)
    for benchmark_id in NEW_VIDEO_WORLD_NORMALIZER_IDS:
        entry = by_id[benchmark_id]
        all_labels = (*entry.tags, *entry.domains, *entry.modalities)

        assert entry.integration_status in {"planned", "integrated"}
        assert entry.maturity in {"contract_ready", "verified_runner", "planned"}
        assert entry.verification_status != "verified"
        # Protective claims: normalizer-only integrations must never assert
        # full official verification or leaderboard validity.  The
        # integration_evidence bit varies per benchmark (bounded evidence is
        # recorded per manifest), so it is intentionally not pinned here.
        assert entry.official_benchmark_verified is False
        assert entry.leaderboard_valid is False
        surface = entry.runner_availability.get("surface")
        if surface is None:
            # Some manifests (e.g. physics-iq) do not declare a surface label.
            continue
        assert surface in {
            "official_result_normalizer",
            "official_runner",
            "in_tree_artifact_evaluator",
            "aigcbench_in_tree_artifact_importer",
            "in_tree_mock_and_normalizer",
            "in_tree_mock_scorer_and_result_normalizer",
            "phyeduvideo_official_in_tree",
            "physvidbench_official_in_tree",
            "in_tree_mock_judge_and_normalizer",
            "physics_iq_official_in_tree",
            "official_in_tree_runtime",
        } or surface.endswith("_official_in_tree") or surface.startswith("in_tree_result_importer")


def test_benchmark_zoo_loads_embodied_world_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog"

    entries = load_benchmark_catalog_shard_entries("embodied", catalog_root)
    by_id = {entry.benchmark_id: entry for entry in entries}
    expected = {
        "libero",
        "libero-para",
        "simpler-env",
        "robocasa",
        "calvin",
        "maniskill",
        "rlbench",
        "metaworld",
        "bridgedata-v2",
        "robotwin",
    }

    assert expected <= set(by_id)
    assert by_id["rlbench"].official_repo_url == "https://github.com/stepjam/RLBench"
    assert by_id["metaworld"].paper_url == "https://arxiv.org/abs/1910.10897"
    assert by_id["bridgedata-v2"].dataset.path == "${WORLDFOUNDRY_DATA_DIR}/datasets/bridgedata-v2"
    assert by_id["bridgedata-v2"].runner_target.endswith("BridgeDataV2Contract")
    assert by_id["libero-para"].runner_target.endswith("LIBEROParaContract")
    assert by_id["libero-para"].dataset_refs[0].hf_dataset_id == "HAI-Lab/LIBERO-Para"
    assert by_id["robotwin"].name == "RoboTwin 2.0"
    assert by_id["robotwin"].paper_url == "https://arxiv.org/abs/2506.18088"
    assert by_id["robotwin"].runner_target.endswith("RoboTwinContract")
    assert by_id["robotwin"].run_command is not None
    # The robotwin manifest routes runs through the embodied CLI entrypoint
    # (config-driven) instead of the old standalone runner script.
    robotwin_command = _command_text(by_id["robotwin"].run_command)
    assert "worldfoundry-eval embodied run" in robotwin_command
    assert "--config" in robotwin_command
    assert "embodied/robotwin/eval.yaml" in robotwin_command
    assert by_id["robotwin"].runner.clone_dir == "thirdparty/RoboTwin"
    assert by_id["robotwin"].runner_runtime["root_env"] == "WORLDFOUNDRY_ROBOTWIN_ROOT"
    assert by_id["robotwin"].dataset_refs[0].revision == "9dc9299c163db059931898a9f0852098a61155a1"
    assert by_id["robotwin"].dataset_refs[1].revision == "1287871839fae2296bc27b88a5457c3e1eba8e1f"
    for benchmark_id in expected:
        assert by_id[benchmark_id].integration_status == "integrated"
        assert by_id[benchmark_id].open_source_status == "normalizer_only"
        assert by_id[benchmark_id].maturity == "contract_ready"
        assert by_id[benchmark_id].verification_status == "normalizer_only"
        assert by_id[benchmark_id].official_benchmark_verified is False
        assert by_id[benchmark_id].integration_evidence is True
        assert by_id[benchmark_id].leaderboard_valid is False
        assert by_id[benchmark_id].runner_availability["normalizer_verified"] is True
    assert tuple(metric.metric_id for metric in by_id["rlbench"].metrics) == get_external_benchmark_contract(
        "rlbench"
    ).metric_ids
    assert tuple(metric.metric_id for metric in by_id["metaworld"].metrics) == get_external_benchmark_contract(
        "metaworld"
    ).metric_ids
    assert tuple(metric.metric_id for metric in by_id["bridgedata-v2"].metrics) == get_external_benchmark_contract(
        "bridgedata-v2"
    ).metric_ids
    assert tuple(metric.metric_id for metric in by_id["robotwin"].metrics) == get_external_benchmark_contract(
        "robotwin"
    ).metric_ids


def test_priority_external_benchmark_contract_targets_are_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog"
    entries = {
        entry.benchmark_id: entry
        for entry in (
            *load_benchmark_catalog_shard_entries("video", catalog_root),
            *load_benchmark_catalog_shard_entries("embodied", catalog_root),
        )
    }

    for benchmark_id in (
        "bridgedata-v2",
        "calvin",
        "libero",
        "libero-para",
        "maniskill",
        "metaworld",
        "rlbench",
        "robocasa",
        "robotwin",
        "simpler-env",
        "evalcrafter",
        "fetv",
        "vbench",
        "vbench-plus-plus",
        "vbench-2.0",
        "worldmodelbench",
        "worldscore",
        "video-bench",
        "worldbench",
        "t2vworldbench",
        "t2v-compbench",
        "videoscore",
        "chronomagic-bench",
        "camerabench",
        "vmbench",
        "videoverse",
        "physvidbench",
        *NEW_VIDEO_WORLD_NORMALIZER_IDS,
        *NEW_VIDEO_WORLD_CONTRACT_IDS,
    ):
        target = entries[benchmark_id].runner_target
        assert target is not None
        module_name, _, attr_name = target.partition(":")
        module = importlib.import_module(module_name)
        runner = getattr(module, attr_name)
        if not hasattr(runner, "benchmark_id"):
            target = entries[benchmark_id].runner_availability.get("runner_target")
            assert target is not None
            module_name, _, attr_name = target.partition(":")
            module = importlib.import_module(module_name)
            runner = getattr(module, attr_name)
        contract = get_external_benchmark_contract(benchmark_id)

        assert runner.benchmark_id == benchmark_id
        assert contract.metric_ids == runner.metric_ids
        # Contracts default to requiring the upstream benchmark runtime;
        # benchmarks whose metrics are computed fully in-tree opt out.
        expected_requires_upstream = benchmark_id not in IN_TREE_COMPUTED_CONTRACT_IDS
        assert contract.requires_upstream_runtime is expected_requires_upstream


def test_benchmark_zoo_entry_converts_to_public_benchmark_spec() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "runner-example",
            "name": "Runner Example",
            "benchmark_kind": ["world-model", "video"],
            "source_status": "confirmed_official_code",
            "official_repo_url": "https://github.com/example/runner-example",
            "hf_dataset_id": "org/runner-example",
            "integration": {"status": "planned"},
            "base_model_dependencies": ["grounded_depth_segmentation_stack"],
            "optional_base_model_dependencies": ["sam_vit_b"],
            "metrics": [
                {
                    "id": "quality",
                    "name": "Quality",
                    "higher_is_better": True,
                    "normalizer": "minmax:0:1",
                    "aggregator": "mean",
                    "unit": "score",
                    "primary": True,
                    "leaderboard_key": "quality",
                }
            ],
        }
    )

    spec = benchmark_zoo_entry_to_benchmark_spec(entry)

    assert isinstance(spec, BenchmarkSpec)
    assert spec.benchmark_id == "runner-example"
    assert spec.tasks[0].evaluation_protocol == "external_benchmark_contract"
    assert spec.tasks[0].data["hf_dataset_id"] == "org/runner-example"
    # The synthesized contract_validation_command was removed from spec
    # metadata together with the legacy manifest field.
    assert "contract_validation_command" not in spec.tasks[0].metadata
    assert spec.tasks[0].metadata["ready_now_command"] is None
    assert spec.tasks[0].metadata["one_click_command"] is None
    assert spec.tasks[0].metadata["contract_only_surface"] is True
    assert spec.tasks[0].metadata["requires_upstream_runtime"] is True
    assert spec.tasks[0].metadata["official_runtime_validated"] is False
    assert spec.tasks[0].metadata["base_model_dependencies"] == ("grounded_depth_segmentation_stack",)
    assert spec.tasks[0].metadata["optional_base_model_dependencies"] == ("sam_vit_b",)
    assert spec.metrics[0].metric_id == "quality"
    assert spec.metrics[0].higher_is_better is True
    assert spec.metrics[0].normalizer == "minmax:0:1"
    assert spec.metrics[0].output_unit == "score"
    assert spec.metrics[0].primary is True
    assert spec.metrics[0].metadata["leaderboard_key"] == "quality"
    assert spec.metadata["integration_status"] == "planned"
    assert "contract_validation_command" not in spec.metadata
    assert spec.metadata["ready_now_command"] is None
    assert spec.metadata["one_click_command"] is None
    assert spec.metadata["base_model_dependencies"] == ("grounded_depth_segmentation_stack",)
    assert spec.metadata["optional_base_model_dependencies"] == ("sam_vit_b",)


def test_benchmark_zoo_video_world_manifest_converts_to_public_specs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog"

    specs = benchmark_zoo_entries_to_benchmark_specs(load_benchmark_catalog_shard_entries("video", catalog_root))

    assert len(specs) >= 10
    by_id = {spec.benchmark_id: spec for spec in specs}
    assert by_id["vbench"].tasks[0].metadata["source_status"] == "open_source"
    # official_runtime_validated requires official_benchmark_verified, which
    # the vbench manifest keeps false until full-suite verification, so the
    # surface stays contract-only.
    assert by_id["vbench"].tasks[0].metadata["contract_only_surface"] is True
    assert by_id["vbench"].tasks[0].metadata["requires_upstream_runtime"] is True
    assert by_id["vbench"].tasks[0].metadata["official_runtime_validated"] is False
    assert by_id["vbench"].metadata["integration_status"] == "integrated"
    assert by_id["vbench"].tasks[0].input_keys == ("prompt_suite_json", "generated_video_dir")
    assert by_id["vbench"].metrics[0].metric_id == "overall_quality"
    assert tuple(metric.metric_id for metric in by_id["vbench"].metrics) == get_external_benchmark_contract(
        "vbench"
    ).metric_ids
    assert by_id["vbench-plus-plus"].tasks[0].metadata["runner"]["runner_target"].endswith(
        "VBenchPlusPlusContract"
    )
    assert by_id["vbench-plus-plus"].tasks[0].metadata["dataset"]["not_applicable"] is True
    assert by_id["vbench-2.0"].tasks[0].metadata["runner"]["runner_target"].endswith("VBench2Contract")
    assert by_id["vbench-2.0"].metadata["open_source_status"] == "experimental"
    assert by_id["vbench-2.0"].metadata["official_runtime_validated"] is False
    assert by_id["vbench-2.0"].tasks[0].metadata["contract_only_surface"] is True
    assert by_id["vbench-2.0"].tasks[0].metadata["runner"]["assets"]["official_gpu_validation"][
        "scorecard"
    ].endswith("gpu_vbench2_cu113_diversity_probe/scorecard.json")
    assert by_id["vbench-2.0"].metrics[-1].metric_id == "vbench2_total"
    assert by_id["worldmodelbench"].tasks[0].data["hf_dataset_id"] == "Efficient-Large-Model/worldmodelbench"
    assert "judge_responses" in by_id["worldmodelbench"].tasks[0].output_keys
    assert by_id["worldscore"].tasks[0].metadata["runner"]["runner_target"].endswith("WorldScoreContract")
    assert by_id["worldscore"].tasks[0].metadata["official_runtime_validated"] is False
    assert by_id["worldscore"].tasks[0].metadata["open_source_status"] == "experimental"
    assert by_id["worldscore"].tasks[0].metadata["runner"]["assets"]["official_gpu_validation"][
        "scorecard"
    ].endswith("worldscore_bounded_gpu_validation_after_readiness_layers_20260525/scorecard.json")
    assert "per_sample_metrics" in by_id["worldscore"].tasks[0].output_keys
    assert by_id["video-bench"].tasks[0].metadata["runner"]["runner_target"].endswith("VideoBenchContract")
    assert by_id["video-bench"].metadata["integration_status"] == "integrated"
    assert by_id["video-bench"].tasks[0].metadata["official_runtime_validated"] is False
    assert by_id["video-bench"].tasks[0].metadata["leaderboard_valid"] is False
    assert by_id["video-bench"].metrics[-1].metric_id == "videobench_average"
    assert by_id["worldbench"].tasks[0].metadata["runner"]["runner_target"].endswith("WorldBenchContract")
    assert by_id["worldbench"].metrics[0].metric_id == "foreground_miou"
    assert by_id["t2v-compbench"].tasks[0].metadata["runner"]["runner_target"].endswith(
        "T2VCompBenchContract"
    )
    assert "leaderboard_csv_manifest" in by_id["t2v-compbench"].tasks[0].output_keys
    assert by_id["vmbench"].tasks[0].metadata["runner"]["runner_target"].endswith("VMBenchContract")
    assert by_id["vmbench"].tasks[0].metadata["dataset"]["not_applicable"] is True
    assert "per_sample_metrics" in by_id["vmbench"].tasks[0].output_keys


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (BenchmarkSource, {"status": "public"}, "BenchmarkSource.status"),
        (BenchmarkZooEntry, {"benchmark_id": "bad", "integration_status": "done"}, "integration_status"),
        (BenchmarkZooEntry, {"benchmark_id": "bad", "open_source_status": "done"}, "open_source_status"),
        (BenchmarkRunnerSpec, {"verification_status": "matching"}, "BenchmarkRunnerSpec.verification_status"),
    ],
)
def test_status_fields_validate_known_enum_strings(factory: object, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)  # type: ignore[operator]


def test_open_source_status_enum_documents_release_labels() -> None:
    assert OPEN_SOURCE_STATUSES == {
        "stable",
        "experimental",
        "preflight_only",
        "normalizer_only",
        "gated",
        "planned",
        "in_tree_runtime",
        "in_tree_result_normalizer",
        "in_tree_artifact_scores_ready",
    }


def test_benchmark_maturity_comes_from_manifest() -> None:
    """Maturity is declared in catalog YAML, not derived in framework code."""
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "manifest-maturity-only",
            "status": "confirmed_official_code",
            "open_source_status": "stable",
            "maturity": "planned",
            "integration_status": "integrated",
            "official_benchmark_verified": True,
            "integration_evidence": True,
            "runner": {
                "runner_target": "worldfoundry.evaluation.tasks.contracts.external:VBenchContract",
                "verification_status": "verified",
            },
            "metrics": [{"id": "score"}],
            "dataset": {"not_applicable": True},
        }
    )

    assert zoo_cli._benchmark_maturity(entry) == "planned"


def test_benchmark_user_commands_use_manifest_fields_only() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "stable-release-planned-runner",
            "status": "confirmed_official_code",
            "open_source_status": "stable",
            "maturity": "contract_ready",
            "contract_validation_command": (
                "worldfoundry-eval zoo benchmark-run --benchmark-id stable-release-planned-runner "
                "--mode contract --output-dir tmp/benchmark_zoo/benchmark_contract/stable-release-planned-runner --json"
            ),
            "runner": {
                "runner_target": "worldfoundry.evaluation.tasks.contracts.external:VBenchContract"
            },
            "metrics": [{"id": "score"}],
            "dataset": {"not_applicable": True},
        }
    )

    commands = zoo_cli._benchmark_user_commands(entry)

    # User commands come exclusively from manifest-declared run/validation
    # command fields.  The deprecated contract_validation_command no longer
    # surfaces a suggested command, and nothing is synthesized for entries
    # that declare neither run_command nor validation_command.
    assert commands == {}
    assert "contract_run" not in commands
    assert "suite_plan" not in commands


def test_benchmark_normalizer_command_follows_manifest_surface() -> None:
    registry = load_benchmark_zoo_registry()
    by_id = {entry.benchmark_id: entry for entry in registry}

    vbench_commands = zoo_cli._benchmark_user_commands(by_id["vbench"])
    videobench_commands = zoo_cli._benchmark_user_commands(by_id["video-bench"])

    assert "normalizer_run" not in vbench_commands
    assert "normalizer_run" in videobench_commands


def test_benchmark_zoo_registry_queries_by_domain_modality_and_tag() -> None:
    registry = BenchmarkZooRegistry(
        [
            {
                "id": "nav-video",
                "name": "Navigation Video",
                "aliases": ("navbench",),
                "domain": "navigation",
                "modality": "video",
                "tags": ("sim", "validation"),
            },
            {"id": "physics-image", "domain": "physics", "modality": "image", "tags": ("sim",)},
        ]
    )

    assert registry.keys() == ["nav-video", "physics-image"]
    assert registry.get("NAV-VIDEO").benchmark_id == "nav-video"
    assert registry.get("NavBench").benchmark_id == "nav-video"
    assert registry.get("navigation video").benchmark_id == "nav-video"
    assert registry.aliases_for("nav-video") == ("navbench", "Navigation Video")
    assert [entry.benchmark_id for entry in registry.by_domain("Navigation")] == ["nav-video"]
    assert [entry.benchmark_id for entry in registry.by_modality("image")] == ["physics-image"]
    assert [entry.benchmark_id for entry in registry.by_tag("SIM")] == ["nav-video", "physics-image"]
    assert [entry.benchmark_id for entry in registry.query(domain="navigation", tag="validation")] == ["nav-video"]


def test_benchmark_zoo_registry_load_cache_can_be_cleared(tmp_path: Path) -> None:
    manifest = tmp_path / "benchmarks.yaml"
    manifest.write_text(
        """
entries:
  - id: cached-benchmark
    status: confirmed_official_code
    metrics:
      - id: score
    dataset:
      not_applicable: true
""".strip(),
        encoding="utf-8",
    )

    clear_benchmark_zoo_registry_cache()
    try:
        first = load_benchmark_zoo_registry(tmp_path)
        second = load_benchmark_zoo_registry(tmp_path)

        assert first is second
        assert first.get("cached-benchmark").benchmark_id == "cached-benchmark"

        clear_benchmark_zoo_registry_cache()
        refreshed = load_benchmark_zoo_registry(tmp_path)
        assert refreshed is not first
        assert refreshed.keys() == ["cached-benchmark"]
    finally:
        clear_benchmark_zoo_registry_cache()


def test_benchmark_zoo_registry_rejects_duplicate_aliases() -> None:
    with pytest.raises(Exception, match="duplicate benchmark-zoo alias"):
        BenchmarkZooRegistry(
            [
                {"id": "alpha", "aliases": ("shared",)},
                {"id": "beta", "aliases": ("SHARED",)},
            ]
        )


def test_benchmark_zoo_schema_and_registry_imports_are_lightweight() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = [
        repo_root / "worldfoundry" / "evaluation" / "tasks" / "catalog" / "schema.py",
        repo_root / "worldfoundry" / "evaluation" / "tasks" / "catalog" / "zoo_registry.py",
    ]
    allowed_modules = set(sys.stdlib_module_names) | {"__future__", "worldfoundry"}

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                modules = {node.module.split(".", 1)[0]} if node.module else set()
            else:
                continue

            unexpected = modules - allowed_modules
            assert unexpected == set(), f"{path} imports {unexpected}"
