from __future__ import annotations

from types import SimpleNamespace

import torch

from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.pipelines.ltx2.pipeline_ltx2_3_t2v import LTX23T2VPipeline
from worldfoundry.synthesis.visual_generation.ltx2.ltx2_runtime import LTX2Video


def test_ltx23_t2v_has_distinct_native_recipe() -> None:
    recipe = default_native_diffusion_registry().resolve("ltx-2.3-t2v")

    assert recipe.model_id == "ltx-2.3-t2v"
    assert "text-to-video" in recipe.capabilities
    assert "image-to-video" not in recipe.capabilities


def test_ltx23_t2v_pipeline_rejects_images() -> None:
    pipeline = LTX23T2VPipeline(synthesis_model=SimpleNamespace(generation_type="t2v"))

    assert pipeline.process(prompt="waves", images=None)["images"] is None
    try:
        pipeline.process(prompt="waves", images=torch.zeros(1, 3, 8, 8))
    except ValueError as error:
        assert "does not accept images" in str(error)
    else:
        raise AssertionError("T2V accepted an image")


def test_ltx23_t2v_runtime_builds_text_only_request() -> None:
    captured = {}

    class _Pipeline:
        def __call__(self, request):
            captured["request"] = request
            return SimpleNamespace(sample=torch.zeros(2, 8, 8, 3), artifacts={})

    runtime = object.__new__(LTX2Video)
    runtime.generation_type = "t2v"
    runtime.negative_prompt = ""
    runtime.height = 64
    runtime.width = 64
    runtime.num_frames = 2
    runtime.num_inference_steps = 1
    runtime.guidance_scale = 1.0
    runtime.seed = 7
    runtime.frame_rate = 24
    runtime.image_strength = 1.0
    runtime.pipeline = _Pipeline()
    runtime.last_audio = None
    runtime.last_audio_sampling_rate = None

    output = runtime.generate_video("waves")

    assert output.shape == (2, 8, 8, 3)
    assert captured["request"].inputs == {"frame_rate": 24}
