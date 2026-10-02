from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.models.catalog import (
    ModelZooEntry,
    ModelZooRegistry,
    UnknownModelZooKeyError,
    load_entries,
    load_model_zoo_registry,
)
from worldfoundry.evaluation.models.catalog.manifest import model_zoo_entry_to_world_model_manifest


def _model_zoo_dir() -> Path:
    # The repository uses a flat package layout (no src/ directory).
    return Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models" / "catalog"


def _model_manifest_paths() -> list[Path]:
    return sorted(_model_zoo_dir().rglob("*.yaml"))


def test_packaged_model_metadata_mirrors_repository_metadata() -> None:
    # With the flat layout the files under worldfoundry/data/models ARE the
    # packaged metadata (there is no separate src/ copy to mirror), so this
    # contract reduces to the catalog being present at the package path.
    package_root = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models"

    assert package_root.is_dir()
    yaml_paths = sorted(package_root.rglob("*.yaml"))
    assert yaml_paths
    assert any(path.is_relative_to(package_root / "catalog") for path in yaml_paths)


def test_model_zoo_registry_loads_repo_manifests_and_exports_manifests() -> None:
    registry = load_model_zoo_registry(_model_zoo_dir())
    direct_entries = tuple(entry for path in _model_manifest_paths() for entry in load_entries(path))
    expected_count = len({entry.model_id for entry in direct_entries})

    assert len(registry) == expected_count
    integrated_ids = {entry.model_id for entry in registry.by_integration_status("integrated")}
    assert integrated_ids >= {"giga-brain-0", "matrix-game-1", "vchitect-2-t2v"}
    for entry in registry.by_integration_status("integrated"):
        assert entry.verification_status != "failed"
        # Hosted-API and script-infer models are integrated without a conda
        # runtime profile; the contract is that every integrated entry
        # declares at least one integration surface.
        assert (
            entry.runtime_profile
            or entry.runner_target
            or entry.pipeline_target
            or entry.pipeline_binding
            or any(variant.runner_target or variant.runtime_profile for variant in entry.variants)
        ), entry.model_id
    integrated_variants = registry.integrated_variants()
    assert all(record.variant.verification_status != "failed" for record in integrated_variants)
    assert len(registry.to_world_model_manifests()) == len(registry)


def test_model_zoo_registry_resolves_wan21_by_id_and_aliases() -> None:
    registry = ModelZooRegistry.from_directory(_model_zoo_dir())

    assert registry.get("wan2.1").model_id == "wan2.1"
    assert registry.get("Wan2.1").model_id == "wan2.1"
    assert registry.get("Wan-AI/Wan2.1-T2V-1.3B").model_id == "wan2.1"
    assert registry.get("wan2.1-t2v-1.3b").model_id == "wan2.1"
    assert registry.get("Wan-AI/Wan2.1-T2V-1.3B").model_id == "wan2.1"


def test_model_zoo_registry_filters_use_exact_task_and_status_matches() -> None:
    registry = ModelZooRegistry.from_directory(_model_zoo_dir())

    text_to_video = registry.by_task("text-to-video")
    open_source = registry.by_source_status("open_source")
    planned = registry.by_integration_status("planned")

    assert text_to_video
    assert all(
        "text-to-video" in entry.tasks or any(variant.task == "text-to-video" for variant in entry.variants)
        for entry in text_to_video
    )
    assert registry.by_task("video") == ()
    assert open_source
    assert all(entry.source_status == "open_source" for entry in open_source)
    assert registry.by_source_status("source") == ()
    assert planned
    assert all(entry.integration_status == "planned" for entry in planned)
    assert registry.by_integration_status("plan") == ()


def test_model_zoo_registry_filters_runner_entry_kind() -> None:
    registry = ModelZooRegistry(
        [
            ModelZooEntry.from_dict({"model_id": "listed", "source": {"status": "open_source"}}),
            ModelZooEntry.from_dict(
                {
                    "model_id": "candidate",
                    "source": {"status": "open_source"},
                    "runner_target": "worldfoundry.models.candidate:Runner",
                    "integration_status": "planned",
                }
            ),
            ModelZooEntry.from_dict(
                {
                    "model_id": "runnable",
                    "source": {"status": "open_source"},
                    "runner_target": "worldfoundry.models.runnable:Runner",
                    "integration_status": "integrated",
                }
            ),
            ModelZooEntry.from_dict(
                {
                    "model_id": "variant-runnable",
                    "source": {"status": "open_source"},
                    "variants": [
                        {
                            "variant_id": "variant-runnable-t2v",
                            "runner_target": "worldfoundry.models.variant:Runner",
                            "integration_status": "integrated",
                        }
                    ],
                }
            ),
        ]
    )

    runnable = registry.runnable_runner_entries()
    candidates = registry.runner_candidate_entries()
    listed = registry.listed_only_entries()

    assert {entry.model_id for entry in runnable} == {"runnable", "variant-runnable"}
    assert all(entry.runner_entry_kind == "runnable_runner" for entry in runnable)
    assert all(entry.runner_entry_kind == "runner_candidate" for entry in candidates)
    assert {entry.model_id for entry in candidates} == {"candidate"}
    assert all(entry.runner_entry_kind == "listed_only" for entry in listed)
    assert {entry.model_id for entry in listed} == {"listed"}


def test_model_zoo_registry_manifest_count_matches_direct_entry_load() -> None:
    manifest_paths = _model_manifest_paths()
    direct_entries = tuple(entry for path in manifest_paths for entry in load_entries(path))
    registry = ModelZooRegistry.from_paths(manifest_paths)
    manifests = registry.to_world_model_manifests()
    expected_count = len({entry.model_id for entry in direct_entries})

    assert len(manifests) == expected_count == len(registry)
    assert {manifest.model_id for manifest in manifests} == {entry.model_id for entry in direct_entries}
    assert registry.to_manifests() == manifests
    assert "Wan-AI/Wan2.1-T2V-1.3B" in next(manifest.aliases for manifest in manifests if manifest.model_id == "wan2.1")


def test_model_zoo_registry_unknown_lookup_raises() -> None:
    registry = ModelZooRegistry.from_directory(_model_zoo_dir())

    with pytest.raises(UnknownModelZooKeyError):
        registry.get("not-a-real-model")


def test_manifest_omits_default_variant_for_variant_only_runner_candidate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "entry.yaml"
    manifest_path.write_text(
        """
models:
  - model_id: manifest-semantics-demo
    source:
      status: open_source
    integration_status: planned
    variants:
      - variant_id: planned-only
        integration_status: planned
        runner_target: pkg:Cls
""".strip(),
        encoding="utf-8",
    )
    entry = load_entries(manifest_path)[0]
    manifest = model_zoo_entry_to_world_model_manifest(entry)
    assert manifest.metadata["default_variant_id"] is None
