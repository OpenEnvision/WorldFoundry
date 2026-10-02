from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from worldfoundry.base_models.capabilities import resolve_base_model_capability_ids
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    FORMAL_BENCHMARK_INVENTORY_SUITE_ID,
    formal_benchmark_ids,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    DEFAULT_EMBODIED_CATALOG_DIR,
    DEFAULT_VIDEO_CATALOG_DIR,
    benchmark_catalog_ids,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    DEFAULT_BENCHMARK_RUNTIME_PROFILE_PATH,
    load_benchmark_runtime_profiles,
)
from worldfoundry.evaluation.tasks.catalog.integrity import (
    build_benchmark_inventory_integrity,
    build_docs_benchmark_coverage,
)
from worldfoundry.evaluation.tasks import load_task_registry_from_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
EMBODIED_BENCHMARKS_DIR = DEFAULT_EMBODIED_CATALOG_DIR
VIDEO_BENCHMARKS_DIR = DEFAULT_VIDEO_CATALOG_DIR
EMBODIED_BENCHMARKS_PATH = EMBODIED_BENCHMARKS_DIR
VIDEO_BENCHMARKS_PATH = VIDEO_BENCHMARKS_DIR
PACKAGE_VIDEO_BENCHMARKS_PATH = VIDEO_BENCHMARKS_DIR
PACKAGE_EMBODIED_BENCHMARKS_PATH = EMBODIED_BENCHMARKS_DIR
INVENTORY_PATH = REPO_ROOT / "tmp" / "docs_benchmark_inventory.json"
RUNTIME_PROFILE_ROOT = DEFAULT_BENCHMARK_RUNTIME_PROFILE_PATH


def _runtime_profiles() -> list[dict]:
    return list(load_benchmark_runtime_profiles(RUNTIME_PROFILE_ROOT).get("profiles") or ())


def _runtime_profiles_by_id() -> dict[str, dict]:
    return {str(profile["id"]): profile for profile in _runtime_profiles()}
BENCHMARK_DOCS_PATH = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.mdx"
BENCHMARK_DOCS_ZH_PATH = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.zh.mdx"
BENCHMARK_HUB_DATA_PATH = REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-hub-data.ts"
BENCHMARK_HUB_COMPONENT_PATH = REPO_ROOT / "docs" / "fumadocs" / "components" / "benchmark-hub.tsx"
CANONICAL_EXTERNAL_TASK_REF = "worldfoundry/data/benchmarks/tasks/external"
LEGACY_BUNDLED_METADATA_RE = re.compile(
    r"(^|[\" :])data/(benchmarks|models)/(tasks/external|fixtures|scorecards|suites|catalog|runtime_profiles)/"
)

OPEN_SOURCE_STATUSES = {
    "stable",
    "experimental",
    "in_tree_runtime",
    "in_tree_artifact_scores_ready",
    "preflight_only",
    "normalizer_only",
    "gated",
    "planned",
}
IN_TREE_LOCAL_EVALUATOR_STATUSES = {
    "planned",
    "available_for_structured_evidence",
    "implemented",
}
BENCHMARK_MATURITY_STATUSES = {
    "verified_runner",
    "contract_ready",
    "planned",
    "blocked",
}
NEW_VIDEO_WORLD_CONTRACT_PREFLIGHT_ONLY_IDS: set[str] = set()
EMBODIED_NORMALIZER_ONLY_NO_RUNNER_SCRIPT_IDS = frozenset(
    {
        "bridgedata-v2",
        "calvin",
        "libero",
        "libero-para",
        "maniskill",
        "metaworld",
        "rlbench",
        "robocasa",
        "simpler-env",
    }
)
VIDEO_WORLD_OFFICIAL_RESULT_NORMALIZER_IDS = {
    "aigcbench",
    "devil-dynamics",
    "ewmbench",
    "mirabench",
    "genai-bench",
    "ipv-bench",
    "fetv",
    "phygenbench",
    "phyeduvideo",
    "videophy",
    "videophy2",
    "physics-iq",
    "phyground",
    "t2v-safety-bench",
    "t2vphysbench",
    "videoverse",
    "videoscience-bench",
    "t2v-compbench",
    "video-bench",
    "worldbench",
    "worldarena",
    "world-in-world",
    "worldmodelbench",
    "vmbench",
}
FORMAL_BENCHMARK_COUNT = len(formal_benchmark_ids(CATALOG_DIR))
VIDEO_CATALOG_COUNT = len(list(VIDEO_BENCHMARKS_DIR.glob("*.yaml"))) - int((VIDEO_BENCHMARKS_DIR / "_manifest.yaml").is_file())
EMBODIED_CATALOG_COUNT = len(list(EMBODIED_BENCHMARKS_DIR.glob("*.yaml"))) - int((EMBODIED_BENCHMARKS_DIR / "_manifest.yaml").is_file())


def _has_verified_official_start_evidence(source: dict) -> bool:
    runner_availability = source.get("runner_availability")
    if isinstance(runner_availability, dict) and runner_availability.get("runtime_verified") is True:
        return True

    validation_candidates = [source.get("official_gpu_validation")]
    runner = source.get("runner")
    if isinstance(runner, dict):
        assets = runner.get("assets")
        if isinstance(assets, dict):
            validation_candidates.append(assets.get("official_gpu_validation"))

    for validation in validation_candidates:
        if not isinstance(validation, dict):
            continue
        if validation.get("verified") is True or validation.get("status") == "verified":
            return True
    return False


def _inventory_benchmark_ids() -> set[str]:
    if INVENTORY_PATH.exists():
        payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        return {entry["id"] for entry in payload["benchmarks"]}
    return set(formal_benchmark_ids(CATALOG_DIR))


def _catalog_entries_from(paths: tuple[Path, Path]) -> dict[str, dict]:
    entries = {}
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.glob("*.yaml")):
                if candidate.name == "_manifest.yaml":
                    continue
                payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("id"):
                    entries[str(payload["id"])] = payload
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("entries"):
            entries.update({entry["id"]: entry for entry in payload["entries"]})
        elif isinstance(payload, dict) and payload.get("id"):
            entries[str(payload["id"])] = payload
    return entries


def _catalog_entries() -> dict[str, dict]:
    return _catalog_entries_from((VIDEO_BENCHMARKS_PATH, EMBODIED_BENCHMARKS_PATH))


def _benchmark_metric_ids() -> dict[str, tuple[str, ...]]:
    return {
        benchmark_id: tuple(metric["id"] for metric in entry.get("metrics", ()))
        for benchmark_id, entry in _catalog_entries().items()
        if benchmark_id in _inventory_benchmark_ids()
    }


def test_benchmark_zoo_external_task_manifests_load_and_match_json_metrics() -> None:
    registry = load_task_registry_from_paths(TASK_ROOT)
    expected_metric_ids = _benchmark_metric_ids()

    assert set(expected_metric_ids) == _inventory_benchmark_ids()

    for benchmark_id in sorted(expected_metric_ids):
        entries = registry.list(benchmark=benchmark_id)

        assert entries, f"{benchmark_id} should expose at least one task"
        for entry in entries:
            task = entry.task

            assert task.metric_ids == expected_metric_ids[benchmark_id]
            assert task.metadata["source_kind"] == "benchmark_zoo"
            assert (
                "external_official_runner" in task.evaluation_protocol_names
                or "in_tree_official_runtime" in task.evaluation_protocol_names
            )
            for protocol in task.evaluation_protocol:
                if protocol.name == "external_official_runner":
                    if protocol.metric_ids:
                        assert protocol.metric_ids == expected_metric_ids[benchmark_id]
                if protocol.name == "in_tree_local_quality_evaluator":
                    assert protocol.metadata["evaluator_target"]
                    assert protocol.metadata["status"] in IN_TREE_LOCAL_EVALUATOR_STATUSES
            assert task.metadata.get("artifact_contract")


def test_benchmark_zoo_external_manifest_metadata_contracts() -> None:
    expected_metric_ids = _benchmark_metric_ids()
    catalog_entries = _catalog_entries()

    for benchmark_id in sorted(expected_metric_ids):
        manifest_path = TASK_ROOT / f"{benchmark_id}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        catalog_entry = catalog_entries[benchmark_id]

        assert manifest["benchmark"] == benchmark_id
        assert metadata["source_kind"] == "benchmark_zoo"
        assert metadata["contract_validation_command"] == (
            f"worldfoundry-eval zoo benchmark-run --benchmark-id {benchmark_id} "
            f"--mode contract --output-dir tmp/benchmark_zoo/benchmark_contract/{benchmark_id} --json"
        )
        assert metadata.get("one_click_command") is None
        assert metadata["release_status"] == catalog_entry["release_status"]
        assert metadata["maturity"] == catalog_entry["maturity"]
        assert metadata["open_source_status"] == catalog_entry["open_source_status"]
        assert metadata["official_benchmark_verified"] is catalog_entry["official_benchmark_verified"]
        assert metadata["integration_evidence"] is catalog_entry["integration_evidence"]
        assert metadata["leaderboard_valid"] is catalog_entry["leaderboard_valid"]
        assert metadata["requires"]
        assert metadata["blockers"]
        assert metadata["runner_availability"] == catalog_entry["runner_availability"]
        assert metadata["runner_availability"]["available"] is (
            metadata["runner_availability"]["runner_target"] is not None
        )
        if metadata["runner_availability"]["available"]:
            assert metadata["official_runner_target"] == metadata["runner_availability"]["runner_target"]
            assert metadata["runtime"]["root_env"]
        assert metadata["artifact_layout"]
        if "open_source_status" in metadata:
            assert metadata["open_source_status"] in OPEN_SOURCE_STATUSES
        assert tuple(manifest["metrics"]) == expected_metric_ids[benchmark_id]
        assert manifest["tasks"]


def test_formal_benchmark_inventory_exposes_machine_readable_readiness_fields() -> None:
    catalog_entries = _catalog_entries()
    runtime_profiles_by_id = _runtime_profiles_by_id()

    for benchmark_id in sorted(_inventory_benchmark_ids()):
        catalog_entry = catalog_entries[benchmark_id]
        manifest_path = TASK_ROOT / f"{benchmark_id}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        profile = runtime_profiles_by_id[benchmark_id]

        for source in (catalog_entry, metadata):
            assert isinstance(source["data_refs"], dict), benchmark_id
            assert source["data_refs"]["task_yaml"] == f"{CANONICAL_EXTERNAL_TASK_REF}/{benchmark_id}.yaml"
            assert isinstance(source["checkpoint_refs"], list), benchmark_id
            assert source["validation_command"], benchmark_id
            assert isinstance(source["runner_availability"], dict), benchmark_id
            assert source["leaderboard_valid"] is False, benchmark_id
            if source["official_benchmark_verified"] is True:
                assert _has_verified_official_start_evidence(source), benchmark_id

        assert metadata["checkpoint_refs"] == catalog_entry["checkpoint_refs"], benchmark_id
        assert metadata["validation_command"] == catalog_entry["validation_command"], benchmark_id
        assert profile["validation_command"] == catalog_entry["validation_command"], benchmark_id
        assert profile["required_assets"]["task_yaml"] == f"{CANONICAL_EXTERNAL_TASK_REF}/{benchmark_id}.yaml"

        surface = str(catalog_entry["runner_availability"].get("surface") or "")
        status_fields = {
            str(catalog_entry.get("open_source_status") or ""),
            str(catalog_entry.get("release_status") or ""),
            surface,
            str(profile.get("status") or ""),
        }
        if any(token in value for value in status_fields for token in ("planned", "normalizer", "preflight")):
            assert catalog_entry["official_benchmark_verified"] is False, benchmark_id
            assert catalog_entry["leaderboard_valid"] is False, benchmark_id


def test_catalog_entries_link_their_task_yaml_contracts() -> None:
    for catalog_entries in (
        _catalog_entries(),
        _catalog_entries_from((PACKAGE_VIDEO_BENCHMARKS_PATH, PACKAGE_EMBODIED_BENCHMARKS_PATH)),
    ):
        for benchmark_id in sorted(_inventory_benchmark_ids()):
            entry = catalog_entries[benchmark_id]
            expected_task_yaml = f"{CANONICAL_EXTERNAL_TASK_REF}/{benchmark_id}.yaml"

            assert entry["data_refs"]["task_yaml"] == expected_task_yaml
            assert (TASK_ROOT / f"{benchmark_id}.yaml").is_file()
            assert entry["runner_availability"]["task_yaml_available"] is True




def test_video_world_metrics_reuse_base_model_dependencies() -> None:
    expected = {
        "aigcbench": ["video_quality_motion_stack"],
        "camerabench": ["camera_geometry_stack"],
        "evalcrafter": ["grounded_video_quality_stack"],
        "mirabench": ["video_quality_dino_motion_stack"],
        "t2v-compbench": ["grounded_depth_segmentation_stack"],
        "vbench": ["vbench_perception_metric_stack"],
        "vbench-2.0": ["vbench_perception_metric_stack"],
        "vbench-plus-plus": ["vbench_perception_metric_stack"],
        "videoscore": ["videoscore_reward_metric_stack"],
        "worldscore": ["worldscore_spatial_metric_stack"],
    }
    expanded = {
        "aigcbench": ["clip_vit_b32", "raft"],
        "camerabench": ["vggt_1b", "moge_v2_vitl_normal", "unidepth_v2_vitl14", "raft"],
        "evalcrafter": ["clip_vit_b32", "grounding_dino", "raft", "sam_v1", "sam2"],
        "mirabench": ["clip_vit_b32", "dinov2_base", "raft"],
        "t2v-compbench": ["depth_anything_v3", "grounding_dino", "sam_v1", "sam2"],
        "vbench": [
            "clip_vit_b32",
            "dinov2_base",
            "grounding_dino",
            "sam_v1",
            "sam2",
            "sam3",
            "raft",
            "vbench_metric_checkpoint_assets",
        ],
        "vbench-2.0": [
            "clip_vit_b32",
            "dinov2_base",
            "grounding_dino",
            "sam_v1",
            "sam2",
            "sam3",
            "raft",
            "vbench_metric_checkpoint_assets",
        ],
        "vbench-plus-plus": [
            "clip_vit_b32",
            "dinov2_base",
            "grounding_dino",
            "sam_v1",
            "sam2",
            "sam3",
            "raft",
            "vbench_metric_checkpoint_assets",
        ],
        "videoscore": ["videoscore_reward_model_v1_1", "videoscore_bench_dataset_assets"],
        "worldscore": [
            "worldscore_official_assets",
            "droid_slam",
            "grounding_dino",
                "raft",
                "sam_v1",
                "sam2",
                "sea_raft",
                "flowformerplusplus",
                "vfimamba",
            ],
        }
    catalog_entries = _catalog_entries()
    runtime_profiles_by_id = _runtime_profiles_by_id()

    for benchmark_id, dependencies in expected.items():
        manifest = yaml.safe_load((TASK_ROOT / f"{benchmark_id}.yaml").read_text(encoding="utf-8"))
        catalog_entry = catalog_entries[benchmark_id]
        runtime_profile = runtime_profiles_by_id[benchmark_id]

        assert manifest["metadata"]["base_model_dependencies"] == dependencies
        assert catalog_entry["base_model_dependencies"] == dependencies
        assert runtime_profile["base_model_dependencies"] == dependencies
        assert resolve_base_model_capability_ids(dependencies) == expanded[benchmark_id]
        assert manifest["metadata"]["base_model_dependency_preflight"]["status"] == "ready"
        assert catalog_entry["base_model_dependency_preflight"]["status"] == "ready"
        assert runtime_profile["base_model_dependency_preflight"]["status"] == "ready"
        assert manifest["metadata"]["base_model_dependency_preflight"]["missing"] == []

    evalcrafter_manifest = yaml.safe_load((TASK_ROOT / "evalcrafter.yaml").read_text(encoding="utf-8"))
    assert evalcrafter_manifest["metadata"]["optional_base_model_dependencies"] == ["sam_vit_b", "aot_deaot_l"]
    assert catalog_entries["evalcrafter"]["optional_base_model_dependencies"] == ["sam_vit_b", "aot_deaot_l"]
    assert runtime_profiles_by_id["evalcrafter"]["optional_base_model_dependencies"] == ["sam_vit_b", "aot_deaot_l"]


def test_packaged_benchmark_catalogs_mirror_repository_catalogs() -> None:
    assert PACKAGE_VIDEO_BENCHMARKS_PATH == VIDEO_BENCHMARKS_PATH
    assert PACKAGE_EMBODIED_BENCHMARKS_PATH == EMBODIED_BENCHMARKS_PATH
    assert VIDEO_BENCHMARKS_PATH.is_dir()
    assert EMBODIED_BENCHMARKS_PATH.is_dir()


def test_packaged_benchmark_metadata_mirrors_repository_metadata() -> None:
    data_root = REPO_ROOT / "worldfoundry" / "data" / "benchmarks"
    package_root = REPO_ROOT / "worldfoundry" / "data" / "benchmarks"
    suffixes = {".json", ".md", ".yaml", ".yml"}

    for path in sorted(item for item in data_root.rglob("*") if item.suffix in suffixes):
        packaged = package_root / path.relative_to(data_root)
        assert packaged.is_file(), str(packaged)
        assert packaged.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_benchmark_docs_list_every_formal_inventory_entry() -> None:
    docs_text = BENCHMARK_DOCS_PATH.read_text(encoding="utf-8")

    for benchmark_id in sorted(_inventory_benchmark_ids()):
        assert f"`{benchmark_id}`" in docs_text


def test_benchmark_docs_registry_tables_match_formal_inventory_exactly() -> None:
    expected_ids = _inventory_benchmark_ids()

    for path in (BENCHMARK_DOCS_PATH, BENCHMARK_DOCS_ZH_PATH):
        docs_text = path.read_text(encoding="utf-8")
        table_ids = re.findall(r"^\| `([^`]+)`\s*/", docs_text, flags=re.M)

        assert table_ids, str(path)
        assert len(table_ids) == len(set(table_ids)), str(path)
        assert set(table_ids) == expected_ids


def test_benchmark_hub_inventory_renders_every_formal_benchmark_entry() -> None:
    data_text = BENCHMARK_HUB_DATA_PATH.read_text(encoding="utf-8")
    component_text = BENCHMARK_HUB_COMPONENT_PATH.read_text(encoding="utf-8")
    category_blocks = re.findall(r"benchmarkIds:\s*\[([^\]]*)\]", data_text, flags=re.S)
    hub_ids: list[str] = []
    for block in category_blocks:
        hub_ids.extend(re.findall(r"'([^']+)'", block))

    assert hub_ids
    assert len(hub_ids) == len(set(hub_ids))
    assert set(hub_ids) == _inventory_benchmark_ids()
    assert "category.benchmarkIds.map" in component_text
    assert "fallbackBenchmarkCard(category, id)" in component_text


def test_benchmark_status_docs_match_current_catalog_summary() -> None:
    docs_text = BENCHMARK_DOCS_PATH.read_text(encoding="utf-8")
    status_text = (BENCHMARK_DOCS_PATH.parent / "benchmark-status.mdx").read_text(encoding="utf-8")
    status_zh_text = (BENCHMARK_DOCS_ZH_PATH.parent / "benchmark-status.zh.mdx").read_text(encoding="utf-8")
    hub_text = BENCHMARK_HUB_DATA_PATH.read_text(encoding="utf-8")

    assert f"Benchmark inventory catalog | {FORMAL_BENCHMARK_COUNT}" in status_text
    assert f"Video/world catalog | {VIDEO_CATALOG_COUNT}" in status_text
    assert f"Embodied catalog | {EMBODIED_CATALOG_COUNT}" in status_text
    assert f"All {EMBODIED_CATALOG_COUNT} embodied entries are `contract_ready`" in status_text
    assert f"{EMBODIED_CATALOG_COUNT} 个 embodied 条目全部是 `contract_ready`" in status_zh_text
    assert "bridgedata-v2` remains blocked" not in status_text
    assert "bridgedata-v2` 在 dataset materialization" not in status_zh_text
    assert "4 rows have bounded/verified runner evidence" in hub_text
    assert "CameraBench bounded official validation row" in hub_text
    assert "{ label: 'integrated'" not in hub_text
    assert "integrated runner evidence" not in hub_text
    assert "bounded-runner" in hub_text
    assert "Core video-generation quality benchmark with bounded official GPU-validation evidence" in hub_text
    assert "Bounded Diversity GPU validation is recorded" in hub_text
    assert "Extended I2V, long-video, and trustworthiness dimensions with bounded official GPU-validation evidence" in hub_text
    assert "WorldScore" in hub_text and "Bounded official GPU validation is recorded" in hub_text

    for benchmark_id in ("bridgedata-v2", "videoscore", "camerabench"):
        assert f"`{benchmark_id}`" in docs_text
    assert "worldfoundry/data" in docs_text
    assert f"*.yaml ({FORMAL_BENCHMARK_COUNT} external task manifests)" in docs_text
    assert "[https://github.com/EmbodiedCity/iWorld-Bench]" in docs_text
    iworld_doc_line = next(line for line in docs_text.splitlines() if line.startswith("| `iworld-bench`"))
    assert "Project/source page only; runnable benchmark code/data is not confirmed." not in iworld_doc_line


def test_vbench2_bounded_official_gpu_validation_is_recorded_without_full_suite_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "vbench-2.0.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["vbench-2.0"]
    runtime_profile = _runtime_profiles_by_id()["vbench-2.0"]

    validation = manifest["metadata"]["official_gpu_validation"]
    assert validation == catalog_entry["official_gpu_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_gpu_validation"
    assert validation["dimension"] == "Diversity"
    assert validation["generated_video_count"] == 20
    assert validation["scorecard"] == "tmp/local_open_eval/gpu_vbench2_cu113_diversity_probe/scorecard.json"
    assert validation["full_suite_verified"] is False

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is False
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert catalog_entry["runner"]["verification_status"] == "verified"
    assert catalog_entry["integration"]["status"] == "integrated"
    assert catalog_entry["runner"]["assets"]["official_gpu_validation"]["full_suite_verified"] is False
    assert runtime_profile["bounded_official_gpu_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_gpu_validation"]["leaderboard_valid"] is False
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_vbench_bounded_official_gpu_validation_is_recorded_without_full_suite_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "vbench.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["vbench"]
    runtime_profile = _runtime_profiles_by_id()["vbench"]

    validation = manifest["metadata"]["official_gpu_validation"]
    assert validation == catalog_entry["official_gpu_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_gpu_validation"
    assert validation["dimension"] == "aesthetic_quality"
    assert validation["generated_video_count"] == 1
    assert validation["scorecard"] == (
        "tmp/local_open_eval/vbench_zeroscope_aesthetic_official_gpu_validation_after_prompt_file_fix_20260525/scorecard.json"
    )
    assert validation["official_benchmark_verified"] is True
    assert validation["integration_evidence"] is True
    assert validation["leaderboard_valid"] is False
    assert validation["full_suite_verified"] is False

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is False
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert catalog_entry["runner"]["assets"]["official_gpu_validation"]["full_suite_verified"] is False
    assert runtime_profile["bounded_official_gpu_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_gpu_validation"]["leaderboard_valid"] is False
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_vbench_plus_plus_bounded_official_gpu_validation_is_recorded_without_full_suite_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "vbench-plus-plus.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["vbench-plus-plus"]
    runtime_profile = _runtime_profiles_by_id()["vbench-plus-plus"]

    validation = manifest["metadata"]["official_gpu_validation"]
    assert validation == catalog_entry["official_gpu_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_gpu_validation"
    assert validation["variant"] == "long"
    assert validation["dimension"] == "temporal_flickering"
    assert validation["generated_video_count"] == 2
    assert validation["scorecard"] == (
        "tmp/worldfoundry_audit/remaining_benchmarks/vbenchpp_long_temporal_flickering_unified_20260621_002/scorecard.json"
    )
    assert validation["official_benchmark_verified"] is True
    assert validation["integration_evidence"] is True
    assert validation["leaderboard_valid"] is False
    assert validation["full_suite_verified"] is False
    assert validation["metrics_available"] == 3

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is False
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert catalog_entry["runner"]["assets"]["official_gpu_validation"]["full_suite_verified"] is False
    assert catalog_entry["runner_availability"]["surface"] == "official_runner"
    assert runtime_profile["bounded_official_gpu_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_gpu_validation"]["leaderboard_valid"] is False
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_camerabench_strict_official_result_path_is_framework_ready_without_leaderboard_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "camerabench.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["camerabench"]
    runtime_profile = _runtime_profiles_by_id()["camerabench"]

    validation = manifest["metadata"]["official_validation"]
    assert validation == catalog_entry["official_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_validation"
    assert validation["task"] == "binary"
    assert validation["metric_family"] == "camera_motion_binary_classification"
    assert validation["scorecard"] == (
        "tmp/local_open_eval/camerabench_binary_official_validation_after_readiness_layers_20260525/scorecard.json"
    )
    assert validation["official_benchmark_verified"] is True
    assert validation["integration_evidence"] is True
    assert validation["leaderboard_valid"] is False
    assert validation["full_suite_verified"] is False

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is True
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert manifest["metadata"]["maturity"] == "verified_runner"
    assert "official_results_path" in " ".join(manifest["metadata"]["requires"])
    assert "full_suite_valid" in " ".join(manifest["tasks"]["camerabench_camera_motion_standard"]["evaluation_protocol"][0]["acceptance"])
    assert catalog_entry["runner"]["assets"]["official_validation"]["full_suite_verified"] is False
    assert catalog_entry["integration"]["status"] == "integrated"
    assert catalog_entry["integration_evidence"] is True
    assert catalog_entry["runner_availability"]["surface"] == "camerabench_official_in_tree"
    assert catalog_entry["runner_availability"]["framework_ready"] is True
    assert runtime_profile["status"] == "credential_gated_framework_ready"
    assert runtime_profile["bounded_official_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_validation"]["leaderboard_valid"] is False
    assert "WORLDFOUNDRY_CAMERABENCH_STRICT" in runtime_profile["optional_env"]
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_worldscore_bounded_official_gpu_validation_is_recorded_without_full_suite_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "worldscore.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["worldscore"]
    runtime_profile = _runtime_profiles_by_id()["worldscore"]

    validation = manifest["metadata"]["official_gpu_validation"]
    assert validation == catalog_entry["official_gpu_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_gpu_validation"
    assert validation["split"] == "dynamic"
    assert validation["generated_video_count"] == 1
    assert validation["scorecard"] == (
        "tmp/local_open_eval/worldscore_bounded_gpu_validation_after_readiness_layers_20260525/scorecard.json"
    )
    assert validation["official_benchmark_verified"] is True
    assert validation["integration_evidence"] is True
    assert validation["leaderboard_valid"] is False
    assert validation["full_suite_verified"] is False
    assert validation["metrics_available"] == 4

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is False
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert catalog_entry["runner"]["assets"]["official_gpu_validation"]["full_suite_verified"] is False
    assert catalog_entry["runner_availability"]["surface"] == "official_runner"
    assert runtime_profile["bounded_official_gpu_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_gpu_validation"]["leaderboard_valid"] is False
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_videoscore_bounded_official_gpu_validation_is_recorded_without_full_suite_claim() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "videoscore.yaml").read_text(encoding="utf-8"))
    catalog_entry = _catalog_entries()["videoscore"]
    runtime_profile = _runtime_profiles_by_id()["videoscore"]

    validation = manifest["metadata"]["official_gpu_validation"]
    assert validation == catalog_entry["official_gpu_validation"]
    assert validation["verified"] is True
    assert validation["scope"] == "bounded_official_gpu_validation"
    assert validation["bench_name"] == "video_feedback"
    assert validation["sample_count"] == 1
    assert validation["scorecard"] == (
        "tmp/local_open_eval/videoscore_bounded_official_gpu_validation_cachefix_20260525/scorecard.json"
    )
    assert validation["official_benchmark_verified"] is True
    assert validation["integration_evidence"] is True
    assert validation["leaderboard_valid"] is False
    assert validation["full_suite_verified"] is False
    assert validation["metrics_available"] == 6

    assert manifest["metadata"]["official_benchmark_verified"] is False
    assert manifest["metadata"]["integration_evidence"] is False
    assert manifest["metadata"]["leaderboard_valid"] is False
    assert catalog_entry["runner"]["assets"]["official_gpu_validation"]["full_suite_verified"] is False
    assert catalog_entry["runner_availability"]["surface"] == "official_runner"
    assert runtime_profile["bounded_official_gpu_validation"]["status"] == "verified"
    assert runtime_profile["bounded_official_gpu_validation"]["leaderboard_valid"] is False
    assert runtime_profile["required_assets"]["scorecard_paths"] == [validation["scorecard"]]


def test_formal_benchmark_inventory_matches_catalog() -> None:
    inventory_ids = _inventory_benchmark_ids()
    catalog_ids = set(_catalog_entries())

    assert inventory_ids == catalog_ids


def test_formal_benchmark_inventory_has_catalog_task_and_runtime_profile_for_every_id() -> None:
    report = build_benchmark_inventory_integrity(
        catalog_dir=CATALOG_DIR,
        task_dir=TASK_ROOT,
        runtime_profile_path=RUNTIME_PROFILE_ROOT,
    )

    assert report.ok, report.failure_summary()
    assert set(report.benchmark_ids) == _inventory_benchmark_ids()
    assert len(report.benchmark_ids) == FORMAL_BENCHMARK_COUNT


def test_public_docs_benchmark_references_match_formal_inventory() -> None:
    coverage = build_docs_benchmark_coverage()

    assert coverage.ok, coverage.failure_summary()
    assert set(coverage.benchmark_ids) == _inventory_benchmark_ids()
    assert len(coverage.mentioned_ids) == FORMAL_BENCHMARK_COUNT
    assert "docs/fumadocs/content/docs/evaluation/benchmarks.mdx" in coverage.docs_paths
    assert "docs/fumadocs/content/docs/evaluation/benchmark-hub.mdx" in coverage.docs_paths
    assert "docs/fumadocs/lib/benchmark-hub-data.ts" in coverage.docs_paths


def test_docs_benchmark_coverage_flags_unknown_and_excluded_refs(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text(
        """
| `vbench` / VBench | ok |
| `missing-bench` / Missing Bench | not ok |

worldfoundry-eval zoo benchmark-run --benchmark-id met3r --mode contract
worldfoundry-eval zoo benchmark-run --benchmark-id <benchmark-id> --mode contract
""",
        encoding="utf-8",
    )

    coverage = build_docs_benchmark_coverage(docs_roots=(docs_root,), benchmark_ids=("vbench",))

    assert coverage.ok is False
    assert coverage.missing_ids == ()
    assert any(item.endswith("README.md:missing-bench") for item in coverage.unknown_refs)
    assert any(item.endswith("README.md:met3r") for item in coverage.excluded_refs)


def test_raw_catalog_and_task_maturity_values_stay_in_public_schema() -> None:
    catalog_entries = _catalog_entries()

    assert set(catalog_entries) == _inventory_benchmark_ids()
    for benchmark_id, entry in catalog_entries.items():
        assert entry.get("maturity") in BENCHMARK_MATURITY_STATUSES, benchmark_id

        task_path = TASK_ROOT / f"{benchmark_id}.yaml"
        manifest = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task_maturity = manifest.get("metadata", {}).get("maturity")
        if task_maturity is not None:
            assert task_maturity in BENCHMARK_MATURITY_STATUSES, benchmark_id


def test_runtime_profile_assets_do_not_cross_reference_unrelated_benchmarks() -> None:
    profile_by_id = _runtime_profiles_by_id()

    bridgedata_assets = profile_by_id["bridgedata-v2"]["required_assets"]
    bridgedata_paths = json.dumps(bridgedata_assets, ensure_ascii=False)

    assert "FETV" not in bridgedata_paths
    assert "bridgedata-v2" in bridgedata_paths
    assert "rail-berkeley" in bridgedata_paths


def test_task_level_evidence_flags_do_not_overstate_catalog_flags() -> None:
    catalog_entries = _catalog_entries()

    for task_path in sorted(TASK_ROOT.glob("*.yaml")):
        manifest = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        benchmark_id = manifest["benchmark"]
        catalog_entry = catalog_entries[benchmark_id]
        top_metadata = manifest.get("metadata", {})
        task_blocks = manifest.get("tasks", {})

        expected_official = bool(catalog_entry.get("official_benchmark_verified"))
        expected_integration = bool(catalog_entry.get("integration_evidence"))
        assert bool(top_metadata.get("official_benchmark_verified")) is expected_official, benchmark_id
        assert bool(top_metadata.get("integration_evidence")) is expected_integration, benchmark_id

        for task in task_blocks.values():
            task_metadata = task.get("metadata", {})
            if "official_benchmark_verified" in task_metadata:
                assert bool(task_metadata["official_benchmark_verified"]) is expected_official, benchmark_id
            if "integration_evidence" in task_metadata:
                assert bool(task_metadata["integration_evidence"]) is expected_integration, benchmark_id


def test_base_model_dependency_preflight_uses_report_key_only() -> None:
    for benchmark_id, catalog_entry in _catalog_entries().items():
        catalog_preflight = catalog_entry.get("base_model_dependency_preflight", {})
        assert "evidence" not in catalog_preflight, benchmark_id
        if catalog_preflight.get("status") == "ready" and "tmp/local_open_eval" in str(catalog_preflight):
            assert "report" in catalog_preflight, benchmark_id

        manifest_path = TASK_ROOT / f"{benchmark_id}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        task_preflight = manifest.get("metadata", {}).get("base_model_dependency_preflight", {})
        assert "evidence" not in task_preflight, benchmark_id
        if task_preflight.get("status") == "ready" and "tmp/local_open_eval" in str(task_preflight):
            assert "report" in task_preflight, benchmark_id

    profiles = _runtime_profiles()
    for profile in profiles:
        profile_preflight = profile.get("base_model_dependency_preflight", {})
        assert "evidence" not in profile_preflight, profile["id"]
        if profile_preflight.get("status") == "ready" and "tmp/local_open_eval" in str(profile_preflight):
            assert "report" in profile_preflight, profile["id"]


def test_bundled_metadata_uses_package_data_paths() -> None:
    checked_paths = [
        *sorted((REPO_ROOT / "worldfoundry" / "data" / "benchmarks").rglob("*.json")),
        *sorted((REPO_ROOT / "worldfoundry" / "data" / "benchmarks").rglob("*.yaml")),
        *sorted((REPO_ROOT / "worldfoundry" / "data" / "models").rglob("*.yaml")),
    ]

    assert checked_paths
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")

        assert "worldfoundry/worldfoundry/" not in text, path
        assert not LEGACY_BUNDLED_METADATA_RE.search(text), path


def test_referenced_benchmark_zoo_runner_scripts_exist() -> None:
    report = build_benchmark_inventory_integrity(
        catalog_dir=CATALOG_DIR,
        task_dir=TASK_ROOT,
        runtime_profile_path=RUNTIME_PROFILE_ROOT,
    )

    assert report.runner_scripts
    assert report.missing_runner_scripts == ()
    for script_path in report.runner_scripts:
        assert (REPO_ROOT / script_path).is_file(), script_path


def test_task_manifests_distinguish_runner_scripts_from_commands() -> None:
    for manifest_path in sorted(TASK_ROOT.glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        availability = metadata.get("runner_availability") or {}
        script = metadata.get("official_runner_script")
        command = metadata.get("official_runner_command")

        if script is not None:
            assert str(script).startswith(
                (
                    "worldfoundry/evaluation/tasks/execution/",
                    "worldfoundry/evaluation/tasks/embodied/runners/",
                )
            ), manifest_path
            assert (REPO_ROOT / script).is_file(), manifest_path
        if command is not None:
            assert str(command).startswith("worldfoundry-eval "), manifest_path
        assert availability.get("runner_script_available") is bool(script)


def test_task_manifests_expose_user_facing_contract_validation_commands() -> None:
    for manifest_path in sorted(TASK_ROOT.glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        benchmark_id = manifest["metadata"]["benchmark_id"]
        command = str(manifest["metadata"].get("contract_validation_command") or "")

        assert command.startswith("worldfoundry-eval zoo benchmark-run "), manifest_path
        assert f"--benchmark-id {benchmark_id}" in command, manifest_path
        assert "--mode contract" in command, manifest_path
        assert "--output-dir tmp/benchmark_zoo/benchmark_contract/" in command, manifest_path
        assert command.endswith("--json"), manifest_path
        assert manifest["metadata"].get("one_click_command") is None, manifest_path


def test_task_manifests_do_not_use_one_click_for_preflight_or_validation_commands() -> None:
    def find_one_click_keys(value: object, prefix: tuple[str, ...] = ()) -> list[str]:
        if isinstance(value, dict):
            found: list[str] = []
            for key, item in value.items():
                path = (*prefix, str(key))
                if key == "one_click_command":
                    found.append(".".join(path))
                found.extend(find_one_click_keys(item, path))
            return found
        if isinstance(value, list):
            found = []
            for index, item in enumerate(value):
                found.extend(find_one_click_keys(item, (*prefix, str(index))))
            return found
        return []

    for manifest_path in sorted(TASK_ROOT.glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert find_one_click_keys(manifest) == [], manifest_path


def test_runtime_profile_package_required_paths_exist() -> None:
    profiles = _runtime_profiles()
    checked_paths = []

    for profile in profiles:
        for item in profile.get("required_paths") or ():
            path = str(item.get("path", ""))
            if not path.startswith(("worldfoundry/", "scripts/")):
                continue
            checked_paths.append(path)
            assert (REPO_ROOT / path).exists(), f"{profile['id']}: {path}"

    assert "worldfoundry/evaluation/tasks/embodied/normalizer.py" in checked_paths
    assert all("worldfoundry/evaluation/runner/" not in path for path in checked_paths)


def test_new_video_world_contract_profiles_remain_contract_preflight_only() -> None:
    profile_by_id = _runtime_profiles_by_id()

    assert NEW_VIDEO_WORLD_CONTRACT_PREFLIGHT_ONLY_IDS <= set(profile_by_id)
    for benchmark_id in sorted(NEW_VIDEO_WORLD_CONTRACT_PREFLIGHT_ONLY_IDS):
        profile = profile_by_id[benchmark_id]
        required_paths = {item["id"]: item["path"] for item in profile["required_paths"]}

        assert profile["benchmark_ids"] == [benchmark_id]
        assert profile["status"] == "blocked_external_assets_or_upstream_runtime"
        assert profile["requires_cuda_visibility"] is False
        assert "contract-only" in profile["blocked_reason"]
        assert profile["required_assets"]["task_yaml"] == f"{CANONICAL_EXTERNAL_TASK_REF}/{benchmark_id}.yaml"
        assert required_paths["task_yaml"] == f"{CANONICAL_EXTERNAL_TASK_REF}/{benchmark_id}.yaml"


def test_embodied_normalizer_only_manifests_do_not_declare_runner_scripts() -> None:
    catalog_entries = _catalog_entries()
    for benchmark_id in sorted(EMBODIED_NORMALIZER_ONLY_NO_RUNNER_SCRIPT_IDS):
        manifest_path = TASK_ROOT / f"{benchmark_id}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        availability = metadata["runner_availability"]
        catalog_entry = catalog_entries[benchmark_id]

        assert metadata.get("official_runner_script") is None, manifest_path
        assert availability["runner_script_available"] is False, manifest_path
        assert availability["surface"] == "official_result_normalizer", manifest_path
        assert catalog_entry["runner_availability"]["runner_script_available"] is False
        assert catalog_entry["release_status"] == "normalizer_only"


def test_video_world_runtime_profiles_do_not_regress_to_result_normalizer_surfaces() -> None:
    profile_by_id = _runtime_profiles_by_id()
    catalog_entries = _catalog_entries()

    expected_ids = VIDEO_WORLD_OFFICIAL_RESULT_NORMALIZER_IDS - {"t2vphysbench"}
    assert expected_ids <= set(profile_by_id)
    for benchmark_id in sorted(expected_ids):
        profile = profile_by_id[benchmark_id]
        manifest = yaml.safe_load((TASK_ROOT / f"{benchmark_id}.yaml").read_text(encoding="utf-8"))
        catalog_entry = catalog_entries[benchmark_id]
        runtime_kind = str(manifest["metadata"]["runtime"]["kind"])
        surface = str(catalog_entry["runner_availability"]["surface"])

        assert runtime_kind.startswith("in_tree_")
        assert "official_result_normalizer" not in surface
        assert "normalizer_only" not in str(profile.get("status") or "")
        assert manifest["metadata"]["runner_availability"]["surface"] == surface


def test_robotwin_embodied_task_manifest_loads_with_official_task_list() -> None:
    manifest_path = TASK_ROOT / "robotwin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    registry = load_task_registry_from_paths(manifest_path)
    entries = registry.list(benchmark="robotwin")

    assert manifest["benchmark"] == "robotwin"
    assert manifest["metadata"]["source_kind"] == "benchmark_zoo"
    assert manifest["metadata"]["official_runner_script"] == "worldfoundry/evaluation/tasks/embodied/runners/robotwin/run_robotwin_official_runner.py"
    assert manifest["metadata"]["runtime"]["root_env"] == "WORLDFOUNDRY_ROBOTWIN_ROOT"
    assert manifest["metadata"]["runtime"]["task_count"] == 50
    assert len(manifest["metadata"]["official_task_groups"]["all_50_tasks"]) == 50
    assert "handover_block" in manifest["metadata"]["official_task_groups"]["all_50_tasks"]
    assert "stack_blocks_two" in manifest["metadata"]["official_task_groups"]["all_50_tasks"]
    assert len(entries) == 1
    assert entries[0].task.metric_ids == tuple(manifest["metrics"])
    assert entries[0].task.metadata["source_kind"] == "benchmark_zoo"
    assert entries[0].task.evaluation_protocol_names == ("external_official_runner",)
