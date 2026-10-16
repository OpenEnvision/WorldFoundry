# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free checks for the concrete replay loading and request contracts."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MATRIX = json.loads((_ROOT / "tests/manual/geometry_regression_cases.json").read_text())
_VARIANTS = ("lite", "lite-distill", "lite-pretrain", "pro", "pro-distill", "pro-pretrain")
_CASE_IDS = (
    *("kandinsky6-" + variant + "-video-short" for variant in _VARIANTS),
    "dreamdojo-student-controls-short",
    "dreamdojo-2b-gr1-distilled-controls-short",
    "lingbot-world-v2-1p3b-causal-short",
    "matrix-game4-egypt-video-short",
    "wonderworld-scene-short",
)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_replay_options_bind_to_the_concrete_public_pipeline(case_id: str) -> None:
    case = _MATRIX[case_id]
    module, _, symbol = case["target"].partition(":")
    pipeline = getattr(importlib.import_module(module), symbol)
    inspect.signature(pipeline.from_pretrained).bind(**case["load"])
    calls = [case["call"]] if "call" in case else [step["call"] for step in case["sequence"]]
    for call in calls:
        inspect.signature(pipeline.__call__).bind(None, **call)


@pytest.mark.parametrize("variant", _VARIANTS)
def test_kandinsky_replay_uses_its_own_weights_and_supported_request(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
    from worldfoundry.pipelines.kandinsky6.pipeline_kandinsky6 import Kandinsky6Pipeline

    case = _MATRIX["kandinsky6-" + variant + "-video-short"]
    captured: dict[str, Any] = {}

    def load(model_id: str, **options: Any) -> SimpleNamespace:
        captured.update(model_id=model_id, **options)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    pipeline = Kandinsky6Pipeline.from_pretrained(**{**case["load"], "device": "cpu"})
    assert captured["model_id"] == case["model_id"]
    assert captured["checkpoint_overrides"] == {
        role: case["load"]["model_path"] for role in ("denoiser", "conditioner", "decoder")
    }
    call = {key: value for key, value in case["call"].items() if key not in {"output_path", "return_dict"}}
    request = pipeline.process(**call)
    assert (request.num_frames, request.height, request.width) == (121, 480, 864)
    assert request.sampling.seed == case["seed"]
    assert request.sampling.num_inference_steps == (10 if "distill" in variant else 50)
    assert request.sampling.guidance_scale == (1.0 if "distill" in variant else 5.0)
    assert request.inputs["sample_audio"] is False


def test_matrix_game4_replay_separates_transformer_and_shared_codec_assets() -> None:
    from worldfoundry.pipelines.matrix_game.pipeline_matrix_game4 import MatrixGame4EgyptPipeline

    case = _MATRIX["matrix-game4-egypt-video-short"]
    overrides = MatrixGame4EgyptPipeline._checkpoint_overrides(case["load"]["model_path"], case["load"])
    assert overrides is not None
    assert set(overrides) == set(MatrixGame4EgyptPipeline.CHECKPOINT_ROLES)
    assert overrides["dit"].endswith("Skywork--Matrix-Game-4.0/fps-egypt/model.safetensors")
    assert overrides["sink"].endswith("Skywork--Matrix-Game-4.0/fps-egypt/sink.png")
    assert overrides["vae"].endswith("alibaba-pai--Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth")


@pytest.mark.parametrize("case_id", ("dreamdojo-student-controls-short", "dreamdojo-2b-gr1-distilled-controls-short"))
def test_dreamdojo_replay_requires_explicit_student_not_teacher_weights(case_id: str) -> None:
    from worldfoundry.pipelines.dreamdojo.pipeline_dreamdojo_student import DreamDojoStudentPipeline

    case = _MATRIX[case_id]
    assert DreamDojoStudentPipeline.MODEL_PATH_OPTION == "student_checkpoint_path"
    assert case["load"]["model_path"] == "${WORLDFOUNDRY_REGRESSION_DREAMDOJO_STUDENT_CHECKPOINT}"
    assert case["assets"]["student"] == case["load"]["model_path"]
    assert case["load"]["text_embeddings_path"] == case["assets"]["text"]
    assert case["array_contracts"]["a.video_chunk.frames"]["shape"][2] == 17
