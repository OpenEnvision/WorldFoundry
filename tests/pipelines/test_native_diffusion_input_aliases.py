from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.pipelines.sana_wm.pipeline_sana_wm import (
    SanaWMPipeline,
    SanaWMStreamingPipeline,
)
from worldfoundry.pipelines.wan.pipeline_wan_vace import Wan2p1VACEPipeline


def _pipeline(pipeline_type):
    pipeline = object.__new__(pipeline_type)
    pipeline.model_id = pipeline.MODEL_ID
    pipeline.process = lambda **kwargs: kwargs
    captured = []

    def native(request):
        captured.append(request)
        return SimpleNamespace(
            sample=torch.zeros(1, 3, 1, 8, 8),
            latents=None,
            artifacts={},
            metadata={},
        )

    pipeline.native_pipeline = native
    return pipeline, captured


@pytest.mark.parametrize(
    "alias",
    ["images", "reference_images", "src_ref_images", "image_path", "ref_image_path", "input_path"],
)
def test_vace_counts_normalized_reference_images(alias) -> None:
    pipeline, captured = _pipeline(Wan2p1VACEPipeline)

    pipeline(prompt="reference composition", **{alias: Path("reference.png")})

    inputs = captured[0].inputs
    assert inputs["images"] == Path("reference.png")
    assert inputs["video"] is None
    assert inputs["vace_reference_count"] == 1


@pytest.mark.parametrize("alias", ["video", "videos", "src_video", "video_path", "input_path"])
def test_vace_video_aliases_do_not_create_reference_images(alias) -> None:
    pipeline, captured = _pipeline(Wan2p1VACEPipeline)

    pipeline(prompt="video edit", **{alias: "source.mp4"})

    inputs = captured[0].inputs
    assert inputs["images"] is None
    assert inputs["video"] == "source.mp4"
    assert inputs["vace_reference_count"] == 0


def test_vace_keeps_explicit_references_with_generic_video_input() -> None:
    pipeline, captured = _pipeline(Wan2p1VACEPipeline)

    pipeline(images=["first.png", "second.png"], input_path="source.mp4")

    inputs = captured[0].inputs
    assert inputs["images"] == ["first.png", "second.png"]
    assert inputs["video"] == "source.mp4"
    assert inputs["vace_reference_count"] == 2


@pytest.mark.parametrize("pipeline_type", [SanaWMPipeline, SanaWMStreamingPipeline])
@pytest.mark.parametrize("options", [{"num_inference_steps": 3, "guidance_scale": 2.0}, {"step": 3, "cfg_scale": 2.0}])
def test_sana_wm_normalizes_sampling_parameters(pipeline_type, options) -> None:
    pipeline, captured = _pipeline(pipeline_type)

    pipeline(images="reference.png", **options)

    sampling = captured[0].sampling
    assert sampling.num_inference_steps == 3
    assert sampling.guidance_scale == 2.0


def test_sana_wm_canonical_sampling_overrides_alias_defaults() -> None:
    pipeline, captured = _pipeline(SanaWMPipeline)

    pipeline(
        images="reference.png",
        step=60,
        cfg_scale=5.0,
        num_inference_steps=3,
        guidance_scale=2.0,
    )

    assert captured[0].sampling.num_inference_steps == 3
    assert captured[0].sampling.guidance_scale == 2.0
