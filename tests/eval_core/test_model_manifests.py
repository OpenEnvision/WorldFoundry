from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from worldfoundry.evaluation.models import (
    build_model_manifest_registry,
    model_manifests_from_source_maps,
)
from worldfoundry.evaluation.api.registry import DuplicateRegistryKeyError, ModelRegistry


def test_model_manifest_facade_resolves_catalog_manifest_helpers() -> None:
    import worldfoundry.evaluation.models.catalog.manifest as canonical_manifest
    import worldfoundry.evaluation.models as models

    assert models.model_manifests_from_source_maps is canonical_manifest.model_manifests_from_source_maps
    assert models.build_model_manifest_registry is canonical_manifest.build_model_manifest_registry


def _load_alpha(model_path: str, device: str) -> object:
    return (model_path, device)


def _infer_alpha(pipe: object) -> object:
    return pipe


def _load_beta(model_path: str, device: str) -> object:
    return (model_path, device)


def _infer_only(pipe: object) -> object:
    return pipe


def test_builds_manifests_from_caller_supplied_source_maps() -> None:
    loaders_by_family = {
        "video_generation": {
            "alpha": _load_alpha,
            "alpha-alias": _load_alpha,
            "beta": _load_beta,
        },
    }
    infers_by_family = {
        "video_generation": {
            "alpha": _infer_alpha,
            "alpha-alias": _infer_alpha,
            "infer-only": _infer_only,
        },
    }

    manifests = model_manifests_from_source_maps(
        loaders_by_family,
        infers_by_family,
        provider={"video_generation": "local"},
        capabilities_by_family={"video_generation": ("i2v", "camera_control")},
    )

    by_id = {manifest.model_id: manifest for manifest in manifests}
    alpha = by_id["alpha"]
    assert alpha.provider == "local"
    assert alpha.capabilities == ("i2v", "camera_control")
    assert alpha.metadata["family"] == "video_generation"
    assert alpha.metadata["capabilities"] == ("i2v", "camera_control")
    assert alpha.metadata["has_loader"] is True
    assert alpha.metadata["has_infer"] is True
    assert alpha.metadata["provider"] == "local"
    assert alpha.metadata["aliases"] == ("alpha-alias",)
    assert alpha.metadata["source_model_ids"] == ("alpha", "alpha-alias")
    assert alpha.metadata["loader"].endswith(":_load_alpha")
    assert alpha.metadata["infer"].endswith(":_infer_alpha")

    beta = by_id["beta"]
    assert beta.metadata["has_loader"] is True
    assert beta.metadata["has_infer"] is False

    infer_only = by_id["infer-only"]
    assert infer_only.metadata["has_loader"] is False
    assert infer_only.metadata["has_infer"] is True


def test_build_model_manifest_registry_returns_core_model_registry() -> None:
    registry = build_model_manifest_registry(
        {"video_generation": {"alpha": _load_alpha, "alpha-alias": _load_alpha}},
        {"video_generation": {"alpha": _infer_alpha, "alpha-alias": _infer_alpha}},
    )

    assert isinstance(registry, ModelRegistry)
    assert registry.get("alpha") is registry.get("ALPHA-ALIAS")
    assert registry.keys() == ["alpha"]


def test_duplicate_keys_and_aliases_are_left_to_registry() -> None:
    manifests = model_manifests_from_source_maps(
        {"video_generation": {"alpha": _load_alpha, "ALPHA": _load_alpha}},
        {"video_generation": {"alpha": _infer_alpha, "ALPHA": _infer_alpha}},
    )

    assert len(manifests) == 1
    assert manifests[0].metadata["aliases"] == ("alpha",)
    with pytest.raises(DuplicateRegistryKeyError):
        ModelRegistry(manifests)

    with pytest.raises(DuplicateRegistryKeyError):
        build_model_manifest_registry(
            {
                "video_generation": {"shared": _load_alpha},
                "three_dimension": {"shared": _load_beta},
            },
            {},
        )


def test_model_package_facade_imports_are_stdlib_or_local() -> None:
    # The repository uses a flat package layout (no src/ directory).
    repo_root = Path(__file__).resolve().parents[2]
    models_root = repo_root / "worldfoundry" / "evaluation" / "models"
    allowed_modules = set(sys.stdlib_module_names) | {"__future__"}

    path = models_root / "__init__.py"
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
