from __future__ import annotations

from pathlib import Path

import numpy as np

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.lingbot_map.pipeline_lingbot_map import LingBotMapPipeline


class _FakeLingBotMapRepresentation:
    def get_representation(self, data):
        assert data["images"] is not None
        return {
            "extrinsic": np.zeros((2, 3, 4), dtype=np.float32),
            "intrinsic": np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            "depth": np.ones((2, 4, 4, 1), dtype=np.float32),
            "mode": data.get("mode", "streaming"),
        }


def test_lingbot_map_catalog_aliases_are_registered():
    entry = load_model_zoo_registry().get("lingbot-map")

    assert entry.pipeline_target == "worldfoundry.pipelines.lingbot_map.pipeline_lingbot_map:LingBotMapPipeline"
    assert "lingbot_map" in entry.aliases
    assert "robbyant/lingbot-map" in entry.aliases


def test_lingbot_map_pipeline_records_and_saves_result(tmp_path: Path):
    pipe = LingBotMapPipeline(representation_model=_FakeLingBotMapRepresentation(), device="cpu")

    result = pipe(
        images=[np.zeros((4, 4, 3), dtype=np.uint8)] * 2,
        interactions=["streaming_reconstruction"],
        output_path=tmp_path / "out.npz",
        return_dict=True,
    )

    assert result["artifact_path"] == str(tmp_path / "out.npz")
    assert Path(result["artifact_path"]).is_file()
    assert pipe.memory_module.select(prefer_type="lingbot_map_result") is not None
