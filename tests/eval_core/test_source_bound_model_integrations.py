from __future__ import annotations

from pathlib import Path

from worldfoundry.zoo.catalog import load_model_zoo_registry
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
PIPELINE_RUNNER_TARGET = "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"

SOURCE_BOUND_RUNTIME_PLAN_MODEL_IDS = (
    "hma",
    "uwm",
    "droid-w",
    "omniforcing",
    "pointworld",
    "vid2world",
    "wildworld",
    "worldgrow",
    "motionbricks",
    "vggt-world",
)

SOURCE_BOUND_PORTED_MODEL_IDS = (
    "hy-embodied",
    "last-r1",
    "openpie-0.6",
)


def test_source_bound_models_use_in_tree_pipeline_or_plan_routes() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id in SOURCE_BOUND_RUNTIME_PLAN_MODEL_IDS:
        entry = registry.get(model_id)
        profile = load_runtime_profile(model_id)

        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.world_model.pipeline_runtime_manifest:")
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.output_artifacts
        assert entry.integration_status == "integrated"
        assert profile.model_id == model_id
        assert profile.backend_stage == "worldfoundry_runtime_plan"
        assert profile.runtime_status == "route_ready_assets_pending"
        assert profile.integration_status == "runtime_ported"
        assert profile.source_repos or profile.checkpoints

    for model_id in SOURCE_BOUND_PORTED_MODEL_IDS:
        entry = registry.get(model_id)
        profile = load_runtime_profile(model_id)

        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.component_pipelines:")
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.integration_status == "integrated"
        assert profile.backend_stage == "official_in_tree_runtime"
        assert profile.integration_status == "runtime_ported"
