# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 4 public discovery, loading delegation and video contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
from worldfoundry.base_models.diffusion_model.recipes.matrix_game import MATRIX_GAME4_EGYPT_PROMPT
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.core.contracts import PipelineInvocation
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, WorldModelConfig
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec, import_pipeline_target
from worldfoundry.evaluation.models.pipelines.results import PipelineResultContext, generation_result_from_pipeline
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile_manifest
from worldfoundry.evaluation.models.runtime.validate import validate_runtime_profile_references
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game4 import MatrixGame4EgyptPipeline
from worldfoundry.runtime.inference_catalog import get_model_inference_spec

MODEL_DATA = Path(__file__).resolve().parents[2] / "worldfoundry/data/models"
TARGET = "worldfoundry.pipelines.matrix_game.pipeline_matrix_game4:MatrixGame4EgyptPipeline"


def test_egypt_catalog_binding_recipe_and_cli_agree() -> None:
    entry = load_model_zoo_registry().get("matrix-game-4-ancient-egypt")
    assert entry.model_id == "matrix-game-4-egypt"
    assert entry.integration_status == "integrated"
    assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert entry.hf_repo_id == "Skywork/Matrix-Game-4.0"
    assert entry.demo_parity.status == entry.runner_parity.status == "pending"
    binding = build_pipeline_runner_spec(WorldModelConfig(model_id=entry.model_id, runner="worldfoundry.pipeline"))
    assert binding.pipeline_target == TARGET
    assert import_pipeline_target(TARGET) is MatrixGame4EgyptPipeline
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime/profiles/matrix-game-4-egypt.yaml")
    assert not validate_runtime_profile_references(profile)
    assert "fps" not in profile.execution["defaults"]
    assert "num_frames" not in profile.execution["defaults"]
    recipe = default_native_diffusion_registry().resolve("matrix-game-4-egypt")
    assert recipe.execution.strategy == "autoregressive-window"
    assert recipe.execution.options["fixed_timesteps"] == (1000, 500)
    assert recipe.checkpoints["dit"].files == ("fps-egypt/model.safetensors",)
    assert recipe.checkpoints["sink"].files == ("fps-egypt/sink.png",)
    assert recipe.checkpoints["dit"].revision == "4419accdb7bb7881234c5914de8cd915266e0325"
    assert (
        next(spec for spec in recipe.components if spec.key.kind.value == "decoder").options["dtype"] is torch.bfloat16
    )
    spec = get_model_inference_spec("matrix-game-4-egypt")
    assert spec is not None
    assert spec.default_task_id == "low-poly-rendering"
    assert spec.tasks[0].inputs[0].required
    assert spec.tasks[0].inputs[1].default == MATRIX_GAME4_EGYPT_PROMPT
    assert "fps" not in spec.tasks[0].default_call_kwargs


def test_public_loader_preserves_source_clock_and_generation_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[DiffusionRequest] = []
    loaded: dict[str, Any] = {}

    class Runner:
        model_id = "matrix-game-4-egypt"

        def run(self, request: DiffusionRequest) -> DiffusionOutput:
            calls.append(request)
            video = request.inputs["video"]
            assert isinstance(video, torch.Tensor)
            return DiffusionOutput(sample=video, latents=video, artifacts={"padded_frames": 3})

    def load(_cls: type, model_id: str, **kwargs: Any) -> NativeDiffusionPipeline:
        loaded.update(model_id=model_id, **kwargs)
        return NativeDiffusionPipeline(Runner())

    def read(_path: Any) -> tuple[np.ndarray, dict[str, float]]:
        return np.full((2, 40, 72, 3), 127, np.uint8), {"fps": 23.976}

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", classmethod(load))
    monkeypatch.setattr("worldfoundry.core.media.codecs.video.read_video", read)
    pipeline = cast(
        MatrixGame4EgyptPipeline,
        MatrixGame4EgyptPipeline.from_pretrained(
            model_path="/weights/egypt.safetensors",
            device="cpu",
            torch_dtype="bfloat16",
            offload_mode="none",
            checkpoint_overrides={"dit": "/weights/egypt.safetensors", "sink": "/weights/sink.png"},
        ),
    )
    result = pipeline(input_path="lowpoly.mp4", seed=42, return_dict=True)
    assert loaded["model_id"] == "matrix-game-4-egypt"
    assert str(loaded["policy"].device) == "cpu"
    assert loaded["policy"].dtype is torch.bfloat16
    assert loaded["checkpoint_overrides"] == {"dit": "/weights/egypt.safetensors", "sink": "/weights/sink.png"}
    assert pipeline._checkpoint_overrides("/weights/egypt.safetensors", {}) == {"dit": "/weights/egypt.safetensors"}
    normalized = generation_result_from_pipeline(
        invocation=PipelineInvocation(
            request=GenerationRequest(sample_id="clip", task_id="low-poly-rendering"),
            prompt="",
            image=None,
            video=None,
            interactions=None,
            ref_image_path=None,
            output_path=Path("egypt.mp4"),
            operator_kwargs={},
            pipeline_kwargs={},
        ),
        result=result,
        context=PipelineResultContext("matrix-game-4-egypt", "generated_video", "world_model", TARGET),
    )
    assert isinstance(normalized, GenerationResult)
    assert normalized.status == "succeeded"
    assert normalized.metadata["pipeline_metadata"]["fps"] == 23.976
    assert calls[0].num_frames == 2
    assert calls[0].prompts == (MATRIX_GAME4_EGYPT_PROMPT,)
    video = calls[0].inputs["video"]
    assert isinstance(video, torch.Tensor)
    assert video.shape == (1, 3, 2, 480, 864)
    assert calls[0].inputs["fps"] == 23.976
    assert result["metadata"]["fps"] == 23.976
    assert result["metadata"]["padded_frames"] == 3
    assert result["metadata"]["scene"] == "ancient-egypt"
    request, _ = pipeline._prepare_call(video=calls[0].inputs["video"], fps=0.5)
    assert request.inputs["fps"] == 0.5

    encoded: list[tuple[int, str, float]] = []

    def write(frames: np.ndarray, path: str, *, fps: float) -> None:
        encoded.append((len(frames), path, fps))

    monkeypatch.setattr("worldfoundry.core.media.codecs.video.write_video", write)
    single_frame = video[:, :, :1]
    saved = pipeline(video=single_frame, fps=23.976, output_path="egypt", return_dict=True)
    assert saved["generated_video_path"] == "egypt.mp4"
    assert saved["generated_image_path"] is None
    assert pipeline(video=single_frame, fps=23.976, output_path="egypt.mp4", return_dict=False) == "egypt.mp4"
    assert encoded == [(1, "egypt.mp4", 23.976), (1, "egypt.mp4", 23.976)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 832},
        {"num_frames": 5},
        {"num_inference_steps": 1},
        {"guidance_scale": 2.0},
        {"shift": 3.0},
        {"fps": 0},
        {"fps": float("nan")},
    ],
)
def test_public_route_rejects_settings_outside_egypt_contract(kwargs: dict[str, Any]) -> None:
    pipeline = MatrixGame4EgyptPipeline(
        native_pipeline=SimpleNamespace(model_id="matrix-game-4-egypt"),
        device="cpu",
    )
    with pytest.raises(ValueError):
        pipeline._prepare_call(video=torch.zeros(1, 3, 2, 480, 864), **kwargs)
