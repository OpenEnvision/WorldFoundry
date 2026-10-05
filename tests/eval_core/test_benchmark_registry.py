from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.catalog.specs import (
    build_benchmark_zoo_catalog_registry,
    get_benchmark_zoo_cli_task,
    list_benchmark_zoo_cli_tasks,
    validate_benchmark_zoo_cli_task,
)
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry


def test_benchmark_zoo_entries_are_exposed_as_cli_task_definitions(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "external_benchmarks.yaml"
    manifest_path.write_text(
        """{
  "entries": [
    {
      "id": "external-video-bench",
      "name": "External Video Bench",
      "benchmark_kind": ["text-to-video", "video-generation-quality"],
      "domains": ["video-generation-quality"],
      "modalities": ["text", "video"],
      "status": "confirmed_official_code",
      "official_repo_url": "https://github.com/example/external-video-bench",
      "runner": {
        "runner_target": "worldfoundry.evaluation.tasks.contracts.external:VBenchContract"
      },
      "metrics": [
        {"id": "overall_quality", "higher_is_better": true, "primary": true}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    benchmark = get_benchmark_zoo_cli_task("external-video-bench", manifest_dir=manifest_dir)

    assert benchmark["source_kind"] == "benchmark_zoo"
    assert benchmark["benchmark_zoo_id"] == "external-video-bench"
    assert benchmark["suite"] == "benchmark_zoo"
    assert benchmark["backend"] == "external_benchmark_contract"
    assert benchmark["task_yaml_path"] == "external-video-bench"
    assert benchmark["name"] == "external-video-bench"
    assert benchmark["protocol"] == "external_benchmark"
    assert benchmark["evaluation_protocol"] == "external_benchmark_contract"
    assert benchmark["input_keys"] == ["prompt_suite_json", "generated_video_dir"]
    assert benchmark["output_keys"] == ["scorecard", "dimension_scores", "raw_metric_table"]
    assert benchmark["requires_upstream_runtime"] is True
    assert benchmark["official_runtime_validated"] is False
    assert benchmark["contract_only_surface"] is True

    assert list_benchmark_zoo_cli_tasks(suite="benchmark_zoo", manifest_dir=manifest_dir) == [benchmark]
    assert list_benchmark_zoo_cli_tasks(source_kind="benchmark_zoo", manifest_dir=manifest_dir) == [benchmark]
    assert list_benchmark_zoo_cli_tasks(source_kind="task_yaml", manifest_dir=manifest_dir) == []
    assert validate_benchmark_zoo_cli_task(benchmark, dataset_root=tmp_path)["source_kind"] == "benchmark_zoo"


def test_benchmark_zoo_exports_catalog_v2_specs(tmp_path) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "external_benchmarks.yaml").write_text(
        """{
  "entries": [
    {
      "id": "external-video-bench",
      "name": "External Video Bench",
      "status": "confirmed_official_code",
      "runner": {
        "runner_target": "worldfoundry.evaluation.tasks.contracts.external:VBenchContract"
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    specs = build_benchmark_zoo_catalog_registry(manifest_dir=manifest_dir).list_benchmarks()
    catalog = build_benchmark_zoo_catalog_registry(manifest_dir=manifest_dir)
    by_name = {item.name: item for item in specs}

    assert set(by_name) == {"external-video-bench"}

    external_task = by_name["external-video-bench"].tasks[0]
    assert by_name["external-video-bench"].schema_version == "worldfoundry-catalog-benchmark"
    assert external_task.schema_version == "worldfoundry-catalog-task"
    assert external_task.metadata["source_kind"] == "benchmark_zoo"
    assert external_task.metadata["benchmark_zoo_id"] == "external-video-bench"
    assert external_task.evaluation_protocol_names == ("external_benchmark_contract",)

    assert catalog.get_benchmark("external-video-bench").tasks[0].name == "external-video-bench"
    assert [task.name for task in catalog.tasks_by_protocol("external_benchmark_contract")] == [
        "external-video-bench"
    ]


def test_load_benchmark_zoo_registry_resolves_manifest_entries(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "external_benchmarks.yaml").write_text(
        """{
  "entries": [
    {
      "id": "external-video-bench",
      "name": "External Video Bench",
      "status": "confirmed_official_code",
      "runner": {
        "runner_target": "worldfoundry.evaluation.tasks.contracts.external:VBenchContract"
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    entry = load_benchmark_zoo_registry(manifest_dir).get("external-video-bench")
    assert entry.benchmark_id == "external-video-bench"
