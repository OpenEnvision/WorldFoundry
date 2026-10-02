from __future__ import annotations

import ast
import sys
from pathlib import Path

import worldfoundry.evaluation.tasks.catalog as catalog_module
from worldfoundry.evaluation.api import EvaluationProtocolSpec as ApiEvaluationProtocolSpec
from worldfoundry.evaluation.api.json_contract import JsonContract, tuple_of_str
from worldfoundry.evaluation.tasks.catalog import (
    BenchmarkSpec,
    CatalogRegistry,
    EvaluationProtocolSpec,
    WorldTaskConfig,
    coerce_task_config,
)


def test_catalog_contracts_build_from_yaml_like_mapping_and_json_roundtrip() -> None:
    benchmark = BenchmarkSpec.from_mapping(
        {
            "schema_version": "worldfoundry-catalog-benchmark",
            "name": "mixed-protocol-smoke",
            "version": "2026.05",
            "tags": ["smoke", "worldfoundry"],
            "tasks": {
                "static-i2v": {
                    "schema_version": "worldfoundry-catalog-task",
                    "protocol": "open_loop",
                    "evaluation_protocol": [
                        "clip_judge",
                        {
                            "name": "geometry_reference",
                            "metric_ids": ["camera_error"],
                            "threshold": 0.1,
                        },
                    ],
                    "input_keys": ["prompt", "image"],
                    "output_keys": ["generated_video"],
                    "tags": ["image"],
                    "data": {"metadata_path": "manifests/static.jsonl"},
                },
                "interactive-nav": {
                    "protocol": "interactive",
                    "evaluation_protocol": {"name": "episode_judge", "judge": "vlm"},
                    "tags": ["navigation"],
                },
            },
        }
    )

    restored = BenchmarkSpec.from_json(benchmark.to_json())

    assert restored == benchmark
    assert restored.schema_version == "worldfoundry-catalog-benchmark"
    assert restored.tasks[0].name == "static-i2v"
    assert restored.tasks[0].evaluation_protocol_names == ("clip_judge", "geometry_reference")
    assert restored.tasks[0].evaluation_protocol[1].metadata["threshold"] == 0.1
    assert restored.tasks[1].evaluation_protocol_names == ("episode_judge",)


def test_world_task_config_does_not_default_to_reference_metrics() -> None:
    task = WorldTaskConfig.from_mapping({"name": "open-task"})

    assert task.evaluation_protocol == ()


def test_catalog_contracts_use_shared_api_json_contract_helpers() -> None:
    assert catalog_module.JsonContract is JsonContract
    assert EvaluationProtocolSpec is ApiEvaluationProtocolSpec
    assert tuple_of_str("clip_judge") == ("clip_judge",)
    assert WorldTaskConfig.from_mapping(
        {"name": "open-task", "evaluation_protocol": "clip_judge"}
    ).evaluation_protocol_names == ("clip_judge",)


def test_catalog_registry_queries_by_benchmark_task_tag_and_protocol() -> None:
    registry = CatalogRegistry(
        [
            {
                "name": "benchmark-a",
                "tags": ["suite-a"],
                "tasks": [
                    {
                        "name": "static-i2v",
                        "protocol": "open_loop",
                        "evaluation_protocol": ["clip_judge"],
                        "tags": ["image"],
                    },
                    {
                        "name": "nav",
                        "protocol": "interactive",
                        "evaluation_protocol": ["episode_judge"],
                        "tags": ["navigation"],
                    },
                ],
            },
            {
                "name": "benchmark-b",
                "tags": ["suite-b"],
                "tasks": [
                    {
                        "name": "physics",
                        "protocol": "physics",
                        "evaluation_protocol": ["state_rollout"],
                        "tags": ["simulation"],
                    }
                ],
            },
        ]
    )

    assert [task.name for task in registry.list_tasks(benchmark="benchmark-a")] == ["static-i2v", "nav"]
    assert [benchmark.name for benchmark in registry.find_benchmarks(task="physics")] == ["benchmark-b"]
    assert [task.name for task in registry.tasks_by_tag("suite-a")] == ["static-i2v", "nav"]
    assert [task.name for task in registry.tasks_by_tag("navigation")] == ["nav"]
    assert [task.name for task in registry.tasks_by_protocol("interactive", protocol_kind="execution")] == ["nav"]
    assert [task.name for task in registry.tasks_by_protocol("clip_judge", protocol_kind="evaluation")] == [
        "static-i2v"
    ]
    assert [benchmark.name for benchmark in registry.benchmarks_by_protocol("state_rollout")] == ["benchmark-b"]


def test_task_config_coercion_preserves_task_config_shape_without_data_import() -> None:
    task_config = {
        "name": "worldfoundry-image-static-i2v",
        "protocol": "open_loop",
        "capability_track": "core_video",
        "schema_type": "sample",
        "input_keys": ["generation_text", "ref_image"],
        "output_keys": ["generated_video"],
        "metric_groups": ["static_consistency", "prompt_alignment"],
        "description": "WorldFoundry image-only static i2v benchmark.",
        "evaluation_protocol": "reference_metrics",
        "dataset_root": "data/benchmarks/WorldFoundry",
        "output_dir": "tmp/worldfoundry",
        "data": {
            "metadata_path": "manifests/sample_image_static.jsonl",
            "media_root": "",
        },
    }

    task = coerce_task_config(task_config)

    assert task.name == "worldfoundry-image-static-i2v"
    assert task.evaluation_protocol_names == ("reference_metrics",)
    assert task.metric_groups == ("static_consistency", "prompt_alignment")
    assert task.data["metadata_path"] == "manifests/sample_image_static.jsonl"
    assert task.metadata["dataset_root"] == "data/benchmarks/WorldFoundry"
    assert task.metadata["output_dir"] == "tmp/worldfoundry"


def test_catalog_imports_are_stdlib_or_local_and_do_not_import_data_benchmarks() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_root = repo_root / "src" / "worldfoundry" / "evaluation" / "catalog"
    allowed_modules = set(sys.stdlib_module_names) | {"__future__"}

    for path in catalog_root.glob("*.py"):
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
            assert unexpected == set(), f"{path} imports non-stdlib modules: {unexpected}"
            assert "data" not in modules, f"{path} must not import data.benchmarks"
