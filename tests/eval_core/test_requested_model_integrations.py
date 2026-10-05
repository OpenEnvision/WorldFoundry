from __future__ import annotations

import importlib
from pathlib import Path

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry


REQUESTED_MODEL_IDS = {
    "worldgrow",
    "inspatio-world",
    "mosaicmem",
    "pointworld",
    "droid-w",
    "openpie-0.6",
    "vggt-world",
    "wildworld",
    "shotstream",
    "omniforcing",
    "worldlabs-marble-1.1",
    "fastwam",
    "hy-embodied",
    "wilddet3d",
    "happyoyster",
    "motionbricks",
    "last-r1",
    "sana-wm",
    "multiworld",
}

REQUESTED_NON_RUNNER_MODEL_IDS: set[str] = set()

PREEXISTING_MODEL_IDS = {
    "worldcam",
    "infinite-world",
    "matrix-game-3",
    "hy-worldplay",
    "hy-world-2.0",
}

PREEXISTING_NON_RUNNER_MODEL_IDS: set[str] = set()


def _catalog_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "worldfoundry" / "data" / "models" / "catalog"


def _split_target(target: str) -> tuple[str, str]:
    module_name, separator, attr_name = target.partition(":")
    assert separator == ":"
    assert module_name.startswith("worldfoundry.")
    assert attr_name
    return module_name, attr_name


def test_requested_model_entries_use_standard_in_tree_runner_targets() -> None:
    registry = load_model_zoo_registry(_catalog_dir())
    runner_model_ids = (REQUESTED_MODEL_IDS - REQUESTED_NON_RUNNER_MODEL_IDS) | PREEXISTING_MODEL_IDS
    entries = {model_id: registry.get(model_id) for model_id in runner_model_ids}

    assert "sekai" not in {entry.model_id for entry in registry.list()}
    for entry in entries.values():
        assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
        assert entry.pipeline_target
        assert entry.pipeline_target.startswith("worldfoundry.")
        assert entry.runner_entry_kind in {"runner_candidate", "runnable_runner"}


def test_non_runner_requested_entries_do_not_register_fake_pipelines() -> None:
    registry = load_model_zoo_registry(_catalog_dir())
    for model_id in REQUESTED_NON_RUNNER_MODEL_IDS | PREEXISTING_NON_RUNNER_MODEL_IDS:
        entry = registry.get(model_id)
        assert entry.runner_target is None
        assert entry.pipeline_target is None
        assert entry.output_artifacts == ()


def test_new_model_specific_pipeline_targets_import() -> None:
    registry = load_model_zoo_registry(_catalog_dir())
    for model_id in REQUESTED_MODEL_IDS - REQUESTED_NON_RUNNER_MODEL_IDS:
        entry = registry.get(model_id)
        assert entry.pipeline_target
        module_name, attr_name = _split_target(entry.pipeline_target)
        pipeline_cls = getattr(importlib.import_module(module_name), attr_name)
        assert callable(getattr(pipeline_cls, "from_pretrained", None))
