from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
PIPELINE_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"

SOURCE_BOUND_RUNTIME_PLAN_MODEL_IDS = (
    "hma",
    "uwm",
    "droid-w",
    "omniforcing",
    "pointworld",
    "vid2world",
    "wilddet3d",
    "wildworld",
    "worldgrow",
    "motionbricks",
    "vggt-world",
)

SOURCE_BOUND_RUNNABLE_MODEL_IDS = (
    "egowm",
    "shotstream",
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

    for model_id in SOURCE_BOUND_RUNNABLE_MODEL_IDS:
        entry = registry.get(model_id)
        profile = load_runtime_profile(model_id)

        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.world_model.pipeline_runtime_manifest:")
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.integration_status == "integrated"
        assert profile.backend_stage == "in_tree_runtime_manifest"
        assert profile.runtime_status == "official_runtime_checkpoint_bound"
        assert profile.integration_status == "runtime_ported"


def test_world_model_manifest_uses_explicit_source_root_not_legacy_github_checkout(tmp_path, monkeypatch) -> None:
    from worldfoundry.synthesis.visual_generation.world_model.runtime_manifest import (
        resolve_runtime_manifest,
        runtime_spec,
    )

    workspace = tmp_path / "workspace"
    external_repo = tmp_path / "github_repos" / "egowm"
    external_repo.mkdir(parents=True)
    (external_repo / "inference.py").write_text("raise SystemExit('external checkout')\n", encoding="utf-8")
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_DIR", str(tmp_path / "github_repos"))

    root, entrypoint, _ = resolve_runtime_manifest(runtime_spec("egowm"))

    assert root != external_repo.resolve()
    assert entrypoint != (external_repo / "inference.py").resolve()
    assert root == (tmp_path / "model_sources" / "egowm").resolve()
    assert entrypoint is not None
    assert entrypoint.name == "infer.py"
    assert "worldfoundry/synthesis/visual_generation/world_model/egowm" in entrypoint.as_posix()


def test_mosaicmem_memory_store_is_integrated_without_fake_pipeline() -> None:
    from worldfoundry.core.memory import CameraIntrinsics, CameraPose, MosaicMemoryConfig, MosaicMemoryStore

    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get("mosaicmem")
    profile = load_runtime_profile("mosaicmem")

    assert entry.output_artifacts == ("memory_index",)
    assert entry.integration_status == "integrated"
    assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert entry.pipeline_target == "worldfoundry.pipelines.world_model.pipeline_runtime_manifest:MosaicMemPipeline"
    assert profile.artifact_kind == "memory_index"
    assert profile.backend_stage == "worldfoundry_runtime_plan"
    assert profile.integration_status == "runtime_ported"

    store = MosaicMemoryStore(MosaicMemoryConfig(top_k=4))
    inserted = store.insert_keyframe(
        frame_index=0,
        timestamp=0.0,
        latents=[float(index) for index in range(2 * 2 * 2)],
        latent_height=2,
        latent_width=2,
        channels=2,
        depth_map=[[4.0, 4.0], [4.0, 4.0]],
        intrinsics=CameraIntrinsics(width=2, height=2, fx=2.0, fy=2.0, cx=1.0, cy=1.0),
        pose=CameraPose.identity(timestamp=0.0),
    )

    mosaic = store.retrieve_mosaic(CameraPose.identity(timestamp=1.0), CameraIntrinsics.from_size(2, 2))
    canvas = mosaic.compose_latent_canvas(2, 2, 2)
    assert inserted
    assert store.num_patches() > 0
    assert mosaic.num_patches() > 0
    assert canvas is not None
