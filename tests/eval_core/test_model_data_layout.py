"""Layout contracts for worldfoundry/data/models: one documented schema per file kind."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_ROOT = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models"
CATALOG_FILES = sorted((MODELS_ROOT / "catalog").rglob("*.yaml"))
PROFILE_FILES = sorted((MODELS_ROOT / "runtime" / "profiles").glob("*.yaml"))
BINDING_IDS = {path.stem for path in (MODELS_ROOT / "bindings" / "pipelines").glob("*.yaml")}
STATUS_FIELDS = ("backend_stage", "runtime_status", "integration_status")


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


@pytest.mark.parametrize("path", CATALOG_FILES, ids=lambda p: str(p.relative_to(MODELS_ROOT)))
def test_catalog_entry_uses_documented_shape(path: Path) -> None:
    data = _load(path)

    assert data.get("id") == path.stem
    assert not {"model_id", "schema_version", "models"} & set(data)


@pytest.mark.parametrize("path", PROFILE_FILES, ids=lambda p: p.name)
def test_runtime_profile_uses_schema_v2_layout(path: Path) -> None:
    data = _load(path)

    assert data.get("schema_version") == 2
    assert data.get("model_id") == path.stem
    assert "id" not in data
    assert not set(STATUS_FIELDS) & set(data), "status fields belong in the execution block"
    binding = (data.get("execution") or {}).get("pipeline_binding")
    assert binding is None or binding in BINDING_IDS


@pytest.mark.parametrize("path", sorted((MODELS_ROOT / "bindings" / "pipelines").glob("*.yaml")), ids=lambda p: p.name)
def test_binding_file_is_named_after_its_binding_id(path: Path) -> None:
    assert _load(path).get("binding_id") == path.stem


def test_runtime_configs_root_holds_only_directories() -> None:
    root = MODELS_ROOT / "runtime" / "configs"

    assert sorted(path.name for path in root.iterdir() if path.is_file()) == []


def test_runtime_config_directories_are_named_after_packages() -> None:
    """Config directories use the synthesis or pipeline package name (snake_case), not the model id."""
    root = MODELS_ROOT / "runtime" / "configs"

    assert sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.isidentifier()) == []
