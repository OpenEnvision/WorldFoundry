# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: WorldFoundry Contributors

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.zoo.catalog import load_model_zoo_registry
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

MODEL_CATALOG_DIR = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models" / "catalog"


@pytest.mark.unit
def test_mosaicmem_memory_store_is_integrated_without_fake_pipeline() -> None:
    from worldfoundry.synthesis.visual_generation.memory.mosaic import (
        CameraIntrinsics,
        CameraPose,
        MosaicMemoryConfig,
        MosaicMemoryStore,
    )

    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get("mosaicmem")
    profile = load_runtime_profile("mosaicmem")

    assert entry.output_artifacts == ("memory_index",)
    assert entry.integration_status == "integrated"
    assert entry.runner_target == "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"
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
