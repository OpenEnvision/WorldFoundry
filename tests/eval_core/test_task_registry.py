from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks import (
    DuplicateTaskRegistryKeyError,
    load_task_registry_from_paths,
    validate_task_yaml_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_task_registry_loads_single_task_and_benchmark_yaml(tmp_path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    (root / "single.yaml").write_text(
        """
name: standalone-task
benchmark_name: standalone-benchmark
protocol: open_loop
evaluation_protocol: reference_metrics
tags: [smoke]
input_keys: [prompt]
output_keys: [generated_video]
metric_ids: [quality]
data:
  metadata_path: manifests/standalone.jsonl
""",
        encoding="utf-8",
    )
    (root / "benchmark.yaml").write_text(
        """
benchmark: mixed-benchmark
tasks:
  task-a:
    protocol: open_loop
    tags: [image]
  task-b:
    protocol: interactive
    evaluation_protocol: episode_judge
    tags: [navigation]
""",
        encoding="utf-8",
    )

    registry = load_task_registry_from_paths(root)

    assert len(registry) == 3
    standalone = registry.get("standalone-task")
    assert standalone.benchmark_name == "standalone-benchmark"
    assert standalone.task.metric_ids == ("quality",)
    assert standalone.task.data["metadata_path"] == "manifests/standalone.jsonl"
    assert [entry.task_name for entry in registry.list(benchmark="mixed-benchmark")] == ["task-a", "task-b"]
    assert [entry.task_name for entry in registry.list(tag="navigation")] == ["task-b"]
    assert [entry.task_name for entry in registry.list(protocol="episode_judge")] == ["task-b"]

    catalog = registry.to_catalog_registry()
    assert sorted(benchmark.name for benchmark in catalog.list_benchmarks()) == [
        "mixed-benchmark",
        "standalone-benchmark",
    ]


def test_task_registry_rejects_duplicate_benchmark_task_pairs(tmp_path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    for index in range(2):
        (root / f"task-{index}.yaml").write_text(
            """
name: duplicate-task
benchmark_name: duplicate-benchmark
protocol: open_loop
""",
            encoding="utf-8",
        )

    with pytest.raises(DuplicateTaskRegistryKeyError):
        load_task_registry_from_paths(root)


def test_validate_task_yaml_file_reports_success_and_loader_errors(tmp_path) -> None:
    valid_path = tmp_path / "valid.yaml"
    invalid_path = tmp_path / "invalid.yaml"
    valid_path.write_text("name: valid-task\nprotocol: open_loop\n", encoding="utf-8")
    invalid_path.write_text("schema_version: not-supported\nname: invalid-task\n", encoding="utf-8")

    valid = validate_task_yaml_file(valid_path)
    invalid = validate_task_yaml_file(invalid_path)

    assert valid["ok"] is True
    assert valid["kind"] == "task"
    assert valid["task_names"] == ["valid-task"]
    assert invalid["ok"] is False
    assert invalid["error_type"] == "ValueError"
    assert "Unsupported" in invalid["error"]


def test_benchmark_zoo_task_manifests_are_filesystem_tasks() -> None:
    registry = load_task_registry_from_paths(REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external")

    by_name = {entry.task_name: entry for entry in registry.list()}

    assert {"vbench_t2v_standard", "worldmodelbench_video_standard", "worldscore_video_standard"} <= set(by_name)
    vbench = by_name["vbench_t2v_standard"]
    assert vbench.benchmark_name == "vbench"
    assert "overall_quality" in vbench.task.metric_ids
    assert vbench.task.metadata["artifact_contract"]["required"]["generated_video"]["kind"] == "video"
    assert vbench.task.evaluation_protocol[0].metadata["runner_script"].endswith("run_vbench_official_runner.py")
