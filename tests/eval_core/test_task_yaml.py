from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.catalog import (
    CATALOG_BENCHMARK_SCHEMA_VERSION,
    CATALOG_TASK_SCHEMA_VERSION,
)
from worldfoundry.evaluation.tasks import (
    TASK_YAML_SCHEMA_VERSION,
    load_benchmark_yaml,
    load_catalog_yaml,
    load_world_task_yaml,
    world_task_config_from_yaml_mapping,
)


def test_task_yaml_extends_deep_merges_and_builds_catalog_task(tmp_path) -> None:
    root = tmp_path / "catalog"
    (root / "templates").mkdir(parents=True)
    (root / "templates" / "base.yaml").write_text(
        """
schema_version: worldfoundry-task
task: base-task
suite: worldfoundry
benchmark_name: base-benchmark
tags: [base]
input_keys: [prompt, image]
protocol:
  type: open_loop_generation
  output_artifacts:
    generated_video:
      kind: video
  generation_defaults:
    fps: 8
dataset:
  path: data/benchmarks/WorldFoundry
  manifest: manifests/base.jsonl
  media_root: media
metrics:
  - id: quality
  - name: consistency
runtime_requirements:
  device: cuda
""",
        encoding="utf-8",
    )
    child_path = root / "child.yaml"
    child_path.write_text(
        """
extends: templates/base.yaml
task: child-task
tags: [child]
dataset:
  manifest: manifests/child.jsonl
generation_defaults:
  seed: 42
evaluation_protocol:
  - name: reference_metrics
    metric_ids: [quality]
""",
        encoding="utf-8",
    )

    task = load_world_task_yaml(child_path, root_dir=root)

    assert task.schema_version == CATALOG_TASK_SCHEMA_VERSION
    assert task.name == "child-task"
    assert task.protocol == "open_loop_generation"
    assert task.input_keys == ("prompt", "image")
    assert task.output_keys == ("generated_video",)
    assert task.metric_ids == ("quality", "consistency")
    assert task.tags == ("child",)
    assert task.data["metadata_path"] == "manifests/child.jsonl"
    assert task.data["media_root"] == "media"
    assert task.generation_defaults == {"fps": 8, "seed": 42}
    assert task.evaluation_protocol_names == ("reference_metrics",)
    assert task.evaluation_protocol[0].metric_ids == ("quality",)
    assert task.metadata["suite"] == "worldfoundry"
    assert task.metadata["benchmark_name"] == "base-benchmark"
    assert task.metadata["runtime_requirements"] == {"device": "cuda"}
    assert task.metadata["dataset"]["path"] == "data/benchmarks/WorldFoundry"
    assert task.metadata["source_schema_version"] == TASK_YAML_SCHEMA_VERSION
    assert task.metadata["source_path"] == str(child_path)


def test_task_yaml_blocks_extends_outside_root(tmp_path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (tmp_path / "outside.yaml").write_text("task: outside\n", encoding="utf-8")
    child_path = root / "child.yaml"
    child_path.write_text("extends: ../outside.yaml\ntask: child\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes task root"):
        load_world_task_yaml(child_path, root_dir=root)


def test_task_yaml_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported task YAML schema_version"):
        world_task_config_from_yaml_mapping(
            {
                "schema_version": "worldfoundry-task-v999",
                "task": "future-task",
            }
        )


def test_benchmark_yaml_loads_embedded_tasks_from_mapping(tmp_path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        """
benchmark: mixed-benchmark
tags: [smoke, yaml]
splits: [mini]
metrics:
  quality:
    aggregation: mean
tasks:
  static-i2v:
    protocol: open_loop
    input_keys: [prompt, image]
    dataset:
      manifest: manifests/static.jsonl
    metrics:
      - id: quality
  interactive-nav:
    protocol: interactive
    evaluation_protocol: episode_judge
""",
        encoding="utf-8",
    )

    benchmark = load_benchmark_yaml(path)

    assert benchmark.schema_version == CATALOG_BENCHMARK_SCHEMA_VERSION
    assert benchmark.name == "mixed-benchmark"
    assert benchmark.version == "1.0"
    assert benchmark.splits == ("mini",)
    assert benchmark.tags == ("smoke", "yaml")
    assert benchmark.metrics == ({"id": "quality", "aggregation": "mean"},)
    assert [task.name for task in benchmark.tasks] == ["static-i2v", "interactive-nav"]
    assert benchmark.tasks[0].metric_ids == ("quality",)
    assert benchmark.tasks[0].data["metadata_path"] == "manifests/static.jsonl"
    assert benchmark.tasks[1].evaluation_protocol_names == ("episode_judge",)
    assert benchmark.metadata["source_schema_version"] == "worldfoundry-benchmark"


def test_catalog_yaml_dispatches_task_or_benchmark(tmp_path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("schema_version: worldfoundry-task\ntask: one-task\n", encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        """
benchmark: one-benchmark
tasks:
  one-task:
    protocol: open_loop
""",
        encoding="utf-8",
    )

    task = load_catalog_yaml(task_path)
    benchmark = load_catalog_yaml(benchmark_path)

    assert task.name == "one-task"
    assert benchmark.name == "one-benchmark"
