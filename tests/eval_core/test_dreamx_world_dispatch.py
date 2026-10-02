from __future__ import annotations

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


def test_dreamx_world_routes_to_the_in_tree_runnable_pipeline() -> None:
    entry = load_model_zoo_registry().get("dreamx-world-5b-cam")
    profile = load_runtime_profile("dreamx-world-5b-cam")

    assert entry.pipeline_target == (
        "worldfoundry.pipelines.dreamx_world.pipeline_dreamx_world:"
        "DreamXWorld5BCamPipeline"
    )
    assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert entry.output_artifacts == ("generated_world",)
    assert entry.integration_status == "integrated"
    assert profile.artifact_kind == "generated_world"
    assert profile.backend_stage == "in_tree_resident_runtime"
    assert profile.runtime_status == "ready"
    assert profile.integration_status == "runnable"


def test_dreamx_world_pipeline_is_importable() -> None:
    from worldfoundry.pipelines.dreamx_world import DreamXWorld5BCamPipeline

    assert DreamXWorld5BCamPipeline.MODEL_ID == "dreamx-world-5b-cam"
