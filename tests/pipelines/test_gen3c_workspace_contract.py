from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from types import SimpleNamespace

import torch

from worldfoundry.pipelines.gen3c import pipeline_gen3c
from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline
from worldfoundry.runtime.inference_catalog import GEN3C_INFERENCE_SPEC
from worldfoundry.pipelines.gen3c.constants import (
    DEFAULT_GEN3C_NEGATIVE_PROMPT,
    DEFAULT_GEN3C_PROMPT,
)
from worldfoundry.studio.inference.catalog import find_entry


def test_workspace_official_defaults_are_normalized(monkeypatch, tmp_path) -> None:
    captured = {}
    instance = object.__new__(Gen3CPipeline)

    def process(**kwargs):
        captured["interactions"] = kwargs["interactions"]
        return {
            "image": kwargs["images"],
            "prompt": kwargs["prompt"],
            "trajectory": "left",
            "actions": [],
            "mapped_trajectories": [],
        }

    instance.process = process

    def native(request):
        captured["request"] = request
        return SimpleNamespace(
            sample=torch.zeros(1, 3, 2, 4, 4),
            latents=None,
            metadata={},
            artifacts={},
        )

    instance.native_pipeline = native
    monkeypatch.setattr(
        pipeline_gen3c,
        "save_image_or_video_tensor",
        lambda sample, path, fps: str(path),
    )

    result = instance(
        images="image.png",
        interaction_signal=["left"],
        prompt="",
        num_video_frames=121,
        num_steps=35,
        guidance=1.0,
        output_dir=tmp_path,
        num_gpus=6,
        noise_aug_strength=0.0,
        filter_points_threshold=0.05,
        foreground_masking=True,
        disable_prompt_upsampler=True,
        disable_guardrail=True,
        disable_prompt_encoder=True,
        offload_diffusion_transformer=False,
        offload_tokenizer=False,
        offload_text_encoder_model=False,
        offload_prompt_upsampler=False,
        offload_guardrail_models=False,
        return_dict=True,
    )

    request = captured["request"]
    assert request.num_frames == 121
    assert request.sampling.num_inference_steps == 35
    assert request.sampling.guidance_scale == 1.0
    assert request.prompt == DEFAULT_GEN3C_PROMPT
    assert request.negative_prompt == DEFAULT_GEN3C_NEGATIVE_PROMPT
    assert captured["interactions"] == ["left"]
    assert request.metadata["trajectory"] == "left"
    assert result["artifact_path"] == str(tmp_path / "gen3c.mp4")


def test_workspace_native_defaults_do_not_reserve_ignored_legacy_gpus() -> None:
    defaults = find_entry("gen3c").default_call_kwargs
    task_defaults = GEN3C_INFERENCE_SPEC.task("official-single-image").default_call_kwargs

    assert defaults["num_video_frames"] == 121
    assert defaults["num_steps"] == 35
    assert defaults["negative_prompt"] == DEFAULT_GEN3C_NEGATIVE_PROMPT
    assert find_entry("gen3c").default_prompt == DEFAULT_GEN3C_PROMPT
    assert GEN3C_INFERENCE_SPEC.task("official-single-image").inputs[1].default == DEFAULT_GEN3C_PROMPT
    assert "negative_prompt" in find_entry("gen3c").call_params
    assert "num_gpus" not in defaults
    assert not any(key.startswith("offload_") for key in defaults)
    assert task_defaults == defaults
