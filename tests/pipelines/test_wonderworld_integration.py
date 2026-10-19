# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Native WonderWorld binding, input preflight and public dispatch contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.operators.wonderworld_operator import WonderWorldOperator
from worldfoundry.pipelines.three_d_four_d.pipeline_wonderworld import WonderWorldPipeline

pytestmark = pytest.mark.unit


def test_catalog_binding_and_runtime_readiness_agree() -> None:
    entry = load_model_zoo_registry().get("wonderworld")
    profile = load_runtime_profile("wonderworld")
    assert entry.pipeline_target == "worldfoundry.pipelines.three_d_four_d.pipeline_wonderworld:WonderWorldPipeline"
    assert entry.integration_status == profile.integration_status == "integrated"
    assert profile.runtime_status == "native_offline_inference"


def test_invalid_inputs_fail_before_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = WonderWorldPipeline.from_pretrained(device="cpu")

    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("input preflight must precede checkpoint loading")

    monkeypatch.setattr(pipeline.synthesis_model, "predict", forbidden)
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({}, "initial RGB image"),
        ({"images": [Image.new("RGB", (8, 8))] * 2}, "exactly one"),
        ({"images": Image.new("RGB", (8, 8)), "video": "clip.mp4"}, "not video"),
        ({"images": Image.new("RGB", (8, 8)), "depth_steps": 0}, "depth_steps"),
        ({"images": Image.new("RGB", (8, 8)), "inpaint_steps": 1000}, "inpaint_steps"),
        ({"images": Image.new("RGB", (8, 8)), "normal_steps": 1000}, "normal_steps"),
        ({"images": Image.new("RGB", (8, 8)), "output_path": "world.mp4"}, "end in .splat"),
        ({"images": Image.new("RGB", (8, 8)), "camera_path": [1, 2, 3]}, "camera_path"),
    )
    for values, message in cases:
        with pytest.raises(ValueError, match=message):
            pipeline(**values)
    assert pipeline.synthesis_model.models == {}


def test_public_loader_is_lazy_and_forwards_native_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.evaluation.models.pipelines.loading import PipelineRunnerSpec, load_pipeline_from_spec

    target = load_model_zoo_registry().get("wonderworld").pipeline_target
    assert target is not None
    pipeline = load_pipeline_from_spec(
        PipelineRunnerSpec(
            model_id="wonderworld",
            pipeline_target=target,
            runtime_profile_id="wonderworld",
            model_path={"checkpoint_root": str(tmp_path)},
            fallback_model_path=None,
            required_components=None,
            device="cpu",
        )
    )
    assert pipeline.synthesis_model.models == {}
    captured = {}

    def predict(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_path": str(tmp_path / "world.splat"), "status": "completed"}

    monkeypatch.setattr(pipeline.synthesis_model, "predict", predict)
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 10), "green").save(image)
    camera = np.eye(4).tolist()
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([camera]))
    result = pipeline(
        input_path=image,
        prompt="a garden",
        camera_path=path,
        seed=9,
        gen_layer=False,
        refinement_steps=0,
        return_dict=True,
    )
    assert result["status"] == "completed"
    assert captured["image"].size == (512, 512)
    assert captured["seed"] == 9 and captured["gen_layer"] is False
    assert captured["refinement_steps"] == 0
    np.testing.assert_array_equal(captured["camera_path"], [camera])
    assert pipeline(input_path=image) == str(tmp_path / "world.splat")
    with pytest.raises(NotImplementedError, match="offline"):
        pipeline.stream()


@pytest.mark.parametrize("image_key", ["image", "ref_image_path", "input_path"])
@pytest.mark.parametrize("viewer", [False, True])
def test_standard_evaluation_invocation_preserves_inputs_and_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_key: str, viewer: bool
) -> None:
    from worldfoundry.evaluation.api import GenerationRequest
    from worldfoundry.evaluation.models.pipelines.lifecycle import (
        PipelineLifecycleContext,
        PipelineRuntimeProfile,
        WorldFoundryPipelineLifecycle,
    )

    pipeline = WonderWorldPipeline.from_pretrained({"checkpoint_root": str(tmp_path)}, device="cpu")
    captured: dict[str, Any] = {}

    def predict(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_path": str(kwargs["output_path"]), "status": "completed"}

    monkeypatch.setattr(pipeline.synthesis_model, "predict", predict)
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 10), "green").save(image)
    expected_camera = np.eye(4, dtype=np.float32)
    expected_camera[0, 3] = -0.35
    controls: dict[str, Any]
    if viewer:
        matrix = expected_camera @ np.diag([-1, -1, 1, 1])
        controls = {"camera_view_matrix_sequence": [matrix.T.reshape(-1).tolist()]}
    else:
        controls = {"camera_path": [expected_camera.tolist()]}
    controls["seed"] = 2
    request = GenerationRequest(
        sample_id="room",
        task_name="image-to-3d-world",
        inputs={image_key: str(image), "prompt": "a garden"},
        controls=controls,
        generation_kwargs={"seed": 19, "gen_layer": False, "refinement_steps": 0},
    )
    lifecycle = WorldFoundryPipelineLifecycle(
        pipeline=pipeline,
        context=PipelineLifecycleContext(
            model_id="wonderworld",
            output_dir=tmp_path,
            pipeline_target="worldfoundry.pipelines.three_d_four_d.pipeline_wonderworld:WonderWorldPipeline",
            profile=PipelineRuntimeProfile("wonderworld.splat", "generated_world", "world_model"),
        ),
    )
    result = lifecycle.generate_in_context(request)
    assert result.status == "succeeded"
    assert "generated_world" in result.artifacts
    assert captured["image"].size == (512, 512)
    assert captured["prompt"] == "a garden"
    assert captured["seed"] == 19 and captured["gen_layer"] is False
    assert captured["refinement_steps"] == 0
    assert captured["output_path"] == tmp_path / "room_wonderworld.splat"
    np.testing.assert_array_equal(captured["camera_path"], [expected_camera])
    assert pipeline.synthesis_model.models == {}


def test_supported_step_limits_reach_synthesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = WonderWorldPipeline.from_pretrained({"checkpoint_root": str(tmp_path)}, device="cpu")
    captured: dict[str, Any] = {}

    def predict(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_path": str(tmp_path / "world.splat")}

    monkeypatch.setattr(pipeline.synthesis_model, "predict", predict)
    pipeline(images=Image.new("RGB", (8, 8)), depth_steps=1000, inpaint_steps=999, normal_steps=999)
    assert (captured["depth_steps"], captured["inpaint_steps"], captured["normal_steps"]) == (1000, 999, 999)


def test_named_viewer_matrix_conversion_and_rigid_validation() -> None:
    matrix = np.diag([-1, -1, 1, 1]).astype(float)
    matrix[0, 3] = -0.35
    actual = WonderWorldOperator.camera_matrices(
        interactions={"camera_view_matrix_sequence": [matrix.T.reshape(-1).tolist()]}
    )
    expected = np.eye(4)
    expected[0, 3] = -0.35
    np.testing.assert_allclose(actual[0], expected)
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        WonderWorldOperator.camera_matrices([invalid])
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        WonderWorldOperator.camera_matrices([invalid])


@pytest.mark.parametrize("error", [RuntimeError("load failure"), asyncio.CancelledError()])
def test_partial_loading_failure_releases_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    from worldfoundry.synthesis.visual_generation.wonderworld.runtime import (
        NativeDiffusionPipeline,
        WonderWorldSynthesis,
    )

    for directory in (
        "stable-diffusion-2-inpainting",
        "prs-eth--marigold-depth-v1-0",
        "prs-eth--marigold-normals-v0-1",
    ):
        path = tmp_path / directory / "unet/diffusion_pytorch_model.fp16.safetensors"
        path.parent.mkdir(parents=True)
        path.touch()
    for relative in ("oneformer_ade20k_swin_large/model.safetensors", "RepViT/repvit_sam.pt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.touch()
    synthesis = WonderWorldSynthesis.from_pretrained({"checkpoint_root": str(tmp_path)}, device="cpu")
    calls = []

    def load(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        if len(calls) == 2:
            raise error
        return object()

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    with pytest.raises(type(error)):
        synthesis._ensure_models()
    assert synthesis.models == {} and synthesis.segmenter is None


def test_public_cli_and_studio_expose_component_assets() -> None:
    from worldfoundry.cli.model_run import load_model_run_schema
    from worldfoundry.runtime.inference_catalog import get_model_inference_spec
    from worldfoundry.studio.inference.catalog import find_entry

    spec = get_model_inference_spec("wonderworld")
    assert spec is not None
    assert spec.default_task_id == "image-to-3d-world"
    schema = load_model_run_schema("wonderworld")
    fields = {field.option for field in schema.fields}
    assert {
        "--pipeline.input-path",
        "--pipeline.depth-steps",
        "--pipeline.camera-path",
        "--pipeline.load.checkpoint-root",
    } <= fields
    config = find_entry("wonderworld")
    assert config.supports_stream is False
    assert "depth_path" in config.load_params
    assert "source_root" not in config.load_params


def test_native_component_paths_are_explicit(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.wonderworld.runtime import WonderWorldSynthesis

    custom_depth = tmp_path / "depth"
    runtime = WonderWorldSynthesis.from_pretrained(
        {"checkpoint_root": str(tmp_path), "depth_path": str(custom_depth)}, device="cpu"
    )
    assert runtime.paths["depth_path"] == str(custom_depth)
    assert runtime.paths["inpainting_path"] == str(tmp_path / "stable-diffusion-2-inpainting")
    assert runtime.paths["repvit_checkpoint"] == str(tmp_path / "RepViT" / "repvit_sam.pt")
    assert runtime.models == {} and runtime.segmenter is None


def test_native_recipes_declare_local_fp16_weights() -> None:
    from worldfoundry.base_models.diffusion_model.recipes.stable_diffusion import sd2_inpaint_recipe

    recipe = sd2_inpaint_recipe()
    assert recipe.checkpoints["text_encoder"].files == ("text_encoder/model.fp16.safetensors",)
    assert recipe.checkpoints["unet"].files == ("unet/diffusion_pytorch_model.fp16.safetensors",)
    assert recipe.checkpoints["vae"].files == ("vae/diffusion_pytorch_model.fp16.safetensors",)
