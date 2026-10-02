from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.yaml import load_yaml_mapping_with_extends


def test_catalog_yaml_extends_resolves_relative_paths(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    child = tmp_path / "tasks" / "child.yaml"
    child.parent.mkdir()
    parent.write_text(
        """
schema_version: worldfoundry-catalog-task
name: base-task
metrics:
  - base_metric
metadata:
  source_kind: benchmark_zoo
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
extends: ../base.yaml
name: child-task
metadata:
  benchmark_zoo_id: child-bench
""".strip(),
        encoding="utf-8",
    )

    payload = load_yaml_mapping_with_extends(child, root_dir=tmp_path)

    assert payload["name"] == "child-task"
    assert payload["metrics"] == ["base_metric"]
    assert payload["metadata"] == {
        "source_kind": "benchmark_zoo",
        "benchmark_zoo_id": "child-bench",
    }


def test_native_task_loader_package_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.evaluation.tasks.native")
