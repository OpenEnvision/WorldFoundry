# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for each Kandinsky6 variant's numerical replay recipe."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_MATRIX = json.loads((_ROOT / "tests/manual/geometry_regression_cases.json").read_text())
_VARIANTS = ("lite", "lite-distill", "lite-pretrain", "pro", "pro-distill", "pro-pretrain")
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant", _VARIANTS)
def test_replay_options_bind_to_the_concrete_public_pipeline(variant: str) -> None:
    from worldfoundry.pipelines.kandinsky6.pipeline_kandinsky6 import Kandinsky6Pipeline

    case = _MATRIX["kandinsky6-" + variant + "-video-short"]
    inspect.signature(Kandinsky6Pipeline.from_pretrained).bind(**case["load"])
    inspect.signature(Kandinsky6Pipeline.__call__).bind(None, **case["call"])


@pytest.mark.parametrize("variant", _VARIANTS)
def test_replay_uses_its_own_checkpoint_and_decoded_outputs(variant: str) -> None:
    model_id = "kandinsky6-" + variant
    case = _MATRIX[model_id + "-video-short"]
    binding = yaml.safe_load((_ROOT / "worldfoundry/data/models/bindings/pipelines" / (model_id + ".yaml")).read_text())
    profile = yaml.safe_load((_ROOT / "worldfoundry/data/models/runtime/profiles" / (model_id + ".yaml")).read_text())
    assert case["model_id"] == case["load"]["model_id"] == model_id
    assert case["target"] == binding["pipeline"]["target"]
    assert case["load"]["model_path"].endswith(profile["checkpoints"][0]["repo_id"].replace("/", "--"))
    assert case["assets"]["checkpoint"] == case["load"]["model_path"]
    assert case["call"]["seed"] == case["seed"] == 42
    assert case["call"]["sample_audio"] is False
    assert case["required_outputs"] == ["artifact.video.mp4.frames", "artifact.video.mp4.fps"]
    assert case["output_arrays"] == case["export_artifacts"] == ["video.mp4"]
    assert case["array_contracts"]["artifact.video.mp4.frames"] == {
        "shape": [121, 480, 864, 3],
        "dtype": "uint8",
    }
    assert case["array_contracts"]["artifact.video.mp4.fps"]["values"] == [24, 1]


@pytest.mark.parametrize("variant", _VARIANTS)
def test_replay_routes_weights_and_sampling_to_the_native_request(
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
