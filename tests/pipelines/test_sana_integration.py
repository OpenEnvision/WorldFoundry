# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Sana catalog, native loading and request/output integration contracts."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS, get_sana_variant
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.sana.pipeline_sana import (
    LongsanaVideo2b480pPipeline,
    Sana1600M1024pxBf16Pipeline,
    SanaControlnet600M1024pxPipeline,
    SanaPipeline,
    SanaVideo2b480pPipeline,
)

pytestmark = pytest.mark.unit

PIPELINES = (
    Sana1600M1024pxBf16Pipeline,
    SanaControlnet600M1024pxPipeline,
    SanaVideo2b480pPipeline,
    LongsanaVideo2b480pPipeline,
)


class _NativePipeline:
    def __init__(self, model_id: str, sample: Any = None) -> None:
        self.model_id = model_id
        self.sample = sample
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if self.sample is None:
            pytest.fail("invalid inputs must fail before native generation")
        return SimpleNamespace(sample=self.sample, latents=None, artifacts={}, metadata={"backend": "native"})


@pytest.mark.parametrize("model_id", sorted(SANA_VARIANTS))
def test_catalog_binding_resolves_current_sana_pipeline(model_id: str) -> None:
    entry = load_model_zoo_registry().get(model_id)
    module_name, class_name = entry.pipeline_target.split(":", 1)
    pipeline_type = getattr(importlib.import_module(module_name), class_name)

    assert issubclass(pipeline_type, SanaPipeline)
    pipeline = pipeline_type(native_pipeline=_NativePipeline(model_id), device="cpu", model_id=model_id)
    assert pipeline.variant.model_id == model_id
    assert pipeline.plan(model_id=model_id)["model_id"] == model_id


@pytest.mark.parametrize("pipeline_type", PIPELINES)
def test_loading_delegates_to_native_diffusion(
    pipeline_type: type[SanaPipeline],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline

    captured: dict[str, Any] = {}

    def load(cls: type[NativeDiffusionPipeline], model_id: str, **kwargs: Any) -> SimpleNamespace:
        captured.update(model_id=model_id, **kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", classmethod(load))
    pipeline = pipeline_type.from_pretrained(device="cpu")

    assert pipeline.model_id == pipeline_type.MODEL_ID
    assert captured["model_id"] == pipeline_type.MODEL_ID
    assert captured["policy"].device == torch.device("cpu")
    assert captured["checkpoint_overrides"] is None
    assert captured["component_options"] is None


@pytest.mark.parametrize("pipeline_type", PIPELINES)
def test_plan_describes_native_execution(pipeline_type: type[SanaPipeline]) -> None:
    plan = pipeline_type.plan()

    assert plan["model_id"] == pipeline_type.MODEL_ID
    assert plan["backend"] == "worldfoundry-native-diffusion"
    assert plan["native_inference"] is True
    assert plan["blocked"] is False
    assert "command" not in plan


@pytest.mark.parametrize("pipeline_type", PIPELINES)
def test_request_and_output_use_shared_native_contract(pipeline_type: type[SanaPipeline]) -> None:
    import torch

    variant = get_sana_variant(pipeline_type.MODEL_ID)
    is_video = variant.task == "text-to-video"
    sample = torch.zeros((1, 3, 5, 32, 32) if is_video else (1, 3, 32, 32))
    native = _NativePipeline(pipeline_type.MODEL_ID, sample)
    pipeline = pipeline_type(native_pipeline=native, device="cpu")
    image = object() if variant.runner == "controlnet" else None
    result = pipeline(
        prompt="a moving cube",
        images=image,
        height=32,
        width=32,
        num_frames=5 if is_video else 1,
        num_inference_steps=3,
        guidance_scale=2.0,
        seed=7,
        flow_shift=2.5,
        return_dict=True,
    )

    assert len(native.requests) == 1
    request = native.requests[0]
    assert request.prompt == "a moving cube"
    assert (request.height, request.width, request.num_frames) == (32, 32, 5 if is_video else 1)
    assert request.sampling.num_inference_steps == 3
    assert request.sampling.guidance_scale == 2.0
    assert request.sampling.seed == 7
    assert request.sampling.scheduler_options["shift"] == 2.5
    if image is not None:
        assert request.inputs["images"] is image
    assert result["sample"] is sample
    assert result["generated_video" if is_video else "generated_image"] is sample
    assert result["model_name"] == pipeline_type.MODEL_ID
    assert result["metadata"] == {"backend": "native"}


def test_controlnet_requires_reference_image_before_generation() -> None:
    native = _NativePipeline(SanaControlnet600M1024pxPipeline.MODEL_ID)
    pipeline = SanaControlnet600M1024pxPipeline(native_pipeline=native, device="cpu")
    with pytest.raises(ValueError, match="requires one or more reference images"):
        pipeline(prompt="a cube")


def test_prompt_only_pipeline_rejects_image_before_generation() -> None:
    native = _NativePipeline(Sana1600M1024pxBf16Pipeline.MODEL_ID)
    pipeline = Sana1600M1024pxBf16Pipeline(native_pipeline=native, device="cpu")
    with pytest.raises(ValueError, match="does not accept image inputs"):
        pipeline(prompt="a cube", images=object())
