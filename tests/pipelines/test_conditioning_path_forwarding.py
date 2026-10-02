"""Regression coverage for Studio conditioning-path forwarding."""

from __future__ import annotations

import pytest

from worldfoundry.pipelines.hydra.pipeline_hydra import HydraPipeline
from worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0_worldgen import (
    HYWorld2WorldgenPipeline,
)
from worldfoundry.pipelines.liveworld.pipeline_liveworld import LiveWorldPipeline
from worldfoundry.pipelines.magic_world.pipeline_magic_world import MagicWorldPipeline
from worldfoundry.pipelines.minwm.pipeline_minwm import MinWMHYAction2VPipeline
from worldfoundry.pipelines.spatia.pipeline_spatia import SpatiaPipeline
from worldfoundry.pipelines.versecrafter.pipeline_versecrafter import VerseCrafterPipeline


class _RecordingSynthesis:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return {"video": "result.mp4"}


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        MinWMHYAction2VPipeline,
        MagicWorldPipeline,
        VerseCrafterPipeline,
        HYWorld2WorldgenPipeline,
    ],
)
def test_explicit_image_and_workspace_image_path_do_not_collide(pipeline_cls) -> None:
    synthesis = _RecordingSynthesis()
    pipeline = pipeline_cls(synthesis, device="cpu")

    result = pipeline(images="explicit.png", image_path="workspace.png")

    assert result == "result.mp4"
    assert synthesis.calls == [
        {
            "prompt": "",
            "image_path": "explicit.png",
            "output_path": None,
            "return_dict": True,
        }
    ] if pipeline_cls is not HYWorld2WorldgenPipeline else [
        {"image_path": "explicit.png", "output_path": None, "return_dict": True}
    ]


@pytest.mark.parametrize("pipeline_cls", [LiveWorldPipeline, SpatiaPipeline])
def test_explicit_media_and_workspace_paths_do_not_collide(pipeline_cls) -> None:
    synthesis = _RecordingSynthesis()
    pipeline = pipeline_cls(synthesis, device="cpu")

    result = pipeline(
        images="explicit.png",
        video="explicit.mp4",
        image_path="workspace.png",
        video_path="workspace.mp4",
    )

    assert result == "result.mp4"
    call = synthesis.calls[0]
    assert call["image_path"] == "explicit.png"
    assert call["video_path"] == "explicit.mp4"


def test_explicit_video_and_workspace_video_path_do_not_collide() -> None:
    synthesis = _RecordingSynthesis()
    pipeline = HydraPipeline(synthesis, device="cpu")

    result = pipeline(video="explicit.mp4", video_path="workspace.mp4")

    assert result == "result.mp4"
    assert synthesis.calls[0]["video_path"] == "explicit.mp4"
