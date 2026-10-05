from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import worldfoundry.synthesis.visual_generation as visual_generation
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_lagernvs_catalog_and_runtime_profile_are_metadata_only() -> None:
    entry = load_model_zoo_registry(MODEL_MANIFEST_DIR).get("lagernvs")
    profile = load_runtime_profile("lagernvs")

    assert entry.integration_status == "blocked"
    assert entry.runner_entry_kind == "listed_only"
    assert entry.runner_target is None
    assert entry.pipeline_target is None
    assert entry.output_artifacts == ()
    assert profile.artifact_kind == "metadata_profile"
    assert profile.artifact_filename == "lagernvs_metadata.json"
    assert profile.integration_status == "blocked"
    assert profile.backend_stage == "metadata_only"
    assert profile.runtime_status == "metadata_only_no_runnable_pipeline"


def test_lagernvs_runnable_pipeline_was_removed() -> None:
    assert _module_exists("worldfoundry.pipelines.lagernvs.pipeline_lagernvs") is False


def test_lagernvs_metadata_only_facades_are_removed() -> None:
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation" / "lagernvs"
    runtime_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d" / "lagernvs"
    assert not (synthesis_root / ("lagernvs" + "_synthesis.py")).exists()
    assert not (runtime_root / "runtime" / "runtime.py").exists()

    module_name = ".".join(
        ("worldfoundry", "synthesis", "visual_generation", "lagernvs", "lagernvs" + "_synthesis")
    )
    assert _module_exists(module_name) is False
    assert all("lagernvs" not in export.lower() for export in visual_generation.__all__)
