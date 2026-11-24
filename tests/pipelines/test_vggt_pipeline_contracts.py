"""Weight-free contracts for the VGGT pipeline interaction and rendering entry points."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.runtime import VGGTRepresentation
from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.runtime import VGGTOmegaRepresentation
from worldfoundry.pipelines.vggt.pipeline_vggt import VGGTPipeline, VGGTResult


class _RecordingRepresentation:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def get_representation(self, data: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(data)
        return {"depth_map": np.ones((1, 4, 4), dtype=np.float32)}


def test_template_aligned_interaction_lists_use_the_latest_signal() -> None:
    representation = _RecordingRepresentation()
    pipeline = VGGTPipeline(representation_model=cast(VGGTRepresentation, representation))

    result = pipeline(images=np.zeros((4, 4, 3), dtype=np.uint8), interactions=["forward", "camera_l"])

    assert isinstance(result, VGGTResult)
    assert len(result.images) == 1
    assert pipeline.operator.interaction_history == ["camera_l"]
    assert len(representation.requests) == 1
    assert representation.requests[0]["predict_depth"] is True


def test_vggt_delegate_names_the_representation_that_lacks_3dgs_rendering() -> None:
    pipeline = VGGTPipeline(representation_model=VGGTOmegaRepresentation(device="cpu"))

    with pytest.raises(NotImplementedError, match="VGGTOmegaRepresentation does not implement 3DGS rendering"):
        pipeline.render_with_3dgs(ply_path="scene.ply", camera_config={})
