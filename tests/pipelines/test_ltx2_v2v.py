# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Public LTX IC-LoRA loading, request, and audio-video artifact contracts."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import av
import pytest
import torch

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe
from worldfoundry.core.contracts import PipelineInvocation
from worldfoundry.core.media.codecs import pyav
from worldfoundry.evals.api import WorldModelConfig
from worldfoundry.evals.models.pipelines.loading import build_pipeline_runner_spec, load_pipeline_from_spec
from worldfoundry.pipelines.ltx2.pipeline_ltx2_v2v import LTX2V2VPipeline, LTX23V2VPipeline

pytestmark = pytest.mark.unit


def _assets(tmp_path: Path) -> dict[str, Path]:
    paths = {
        key: tmp_path / f"{key}.safetensors" for key in ("checkpoint_path", "spatial_upsampler_path", "ic_lora_path")
    }
    for path in paths.values():
        path.touch()
    paths["gemma_root"] = tmp_path / "gemma"
    paths["gemma_root"].mkdir()
    (paths["gemma_root"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    return paths


def _no_subprocess(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError("LTX IC-LoRA must execute in-process")


@pytest.mark.parametrize("model_id", ["ltx-2-v2v", "ltx-2.3-v2v"])
@pytest.mark.parametrize("offload,expected_mode", [("none", "none"), ("cpu", "block"), ("disk", "disk")])
@pytest.mark.parametrize("skip", [False, True])
def test_ltx_v2v_public_loader_builds_native_recipe_and_forwards_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, offload: str, expected_mode: str, skip: bool
) -> None:
    assets = _assets(tmp_path)
    spec = build_pipeline_runner_spec(WorldModelConfig(model_id=model_id, runner="worldfoundry.pipeline"))
    pipeline = load_pipeline_from_spec(replace(spec, model_path=assets, required_components=None, device="cuda:1"))
    assert isinstance(pipeline, LTX2V2VPipeline | LTX23V2VPipeline)
    builds: list[tuple[NativeDiffusionRecipe, dict[str, Any]]] = []
    requests: list[DiffusionRequest] = []
    writes: list[dict[str, Any]] = []
    frames = torch.full((9, 8, 8, 3), 0.25)
    audio = torch.zeros(2, 16000)

    def run(request: DiffusionRequest) -> SimpleNamespace:
        requests.append(request)
        return SimpleNamespace(sample=frames, artifacts={"audio": audio, "audio_sampling_rate": 48000})

    def build(self: NativeDiffusionAssembler, recipe: NativeDiffusionRecipe, **options: Any) -> SimpleNamespace:
        self.resolve_checkpoints(recipe, options["checkpoint_overrides"])
        builds.append((recipe, options))
        return SimpleNamespace(run=run)

    def write(video: torch.Tensor, path: Path, **kwargs: Any) -> None:
        assert video is frames
        writes.append(kwargs)
        path.write_bytes(b"native-media-fixture")

    monkeypatch.setattr(NativeDiffusionAssembler, "build", build)
    monkeypatch.setattr(pyav, "save_audio_video_h264", write)
    monkeypatch.setattr(platform, "processor", platform.machine)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
    control = tmp_path / "control.mp4"
    control.touch()
    result = pipeline(
        "A dancer.",
        video_path=control,
        control_type="pose",
        height=128,
        width=192,
        num_frames=17,
        fps=25,
        seed=321,
        control_strength=0.6,
        lora_strength=0.7,
        offload=offload,
        skip_stage_2=skip,
        output_path=tmp_path / "result.mp4",
        return_dict=True,
    )
    assert result == {
        "status": "success",
        "artifact_path": str(tmp_path / "result.mp4"),
        "video": str(tmp_path / "result.mp4"),
        "log_path": str(tmp_path / "result.ltx.log"),
    }
    assert "worldfoundry.native_diffusion" in Path(result["log_path"]).read_text()
    assert len(builds) == len(requests) == len(writes) == 1
    recipe, options = builds[0]
    assert recipe.model_id == model_id
    assert recipe.execution.options["stage_steps"] == ((8,) if skip else (8, 3))
    assert options["policy"].device == torch.device("cuda:1")
    assert options["policy"].dtype == torch.bfloat16
    assert options["policy"].offload.mode.value == expected_mode
    expected = {"model": "checkpoint_path", "gemma": "gemma_root", "ic_lora": "ic_lora_path"}
    if not skip:
        expected["upsampler"] = "spatial_upsampler_path"
    assert {key: value.source[0] for key, value in options["checkpoint_overrides"].items()} == {
        key: str(assets[value]) for key, value in expected.items()
    }
    assert options["component_options"] == {"denoiser:main": {"lora_strength": 0.7}}
    request = requests[0]
    assert (request.prompt, request.height, request.width, request.num_frames) == ("A dancer.", 128, 192, 17)
    assert (request.sampling.seed, request.sampling.guidance_scale) == (321, 1.0)
    assert request.sampling.num_inference_steps == (8 if skip else 11)
    assert request.inputs == {"video": str(control), "control_strength": 0.6, "frame_rate": 25}
    assert writes[0]["waveform"] is audio
    assert writes[0]["sample_rate"] == 48000
    assert writes[0]["fps"] == 25


@pytest.mark.parametrize("pipeline_type,control_type", [(LTX2V2VPipeline, "pose"), (LTX23V2VPipeline, "depth")])
def test_ltx_v2v_invocation_writes_real_audio_video_and_recovers_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline_type: Any, control_type: str
) -> None:
    pipeline = pipeline_type.from_pretrained(required_components=_assets(tmp_path), device="cpu")
    control = tmp_path / "control.mp4"
    control.touch()
    requests: list[DiffusionRequest] = []

    def run(request: DiffusionRequest) -> SimpleNamespace:
        requests.append(request)
        if request.prompt == "fail":
            raise ValueError("fixture denoising failure")
        strength = request.inputs["control_strength"]
        assert isinstance(strength, float)
        return SimpleNamespace(
            sample=torch.full((9, 16, 24, 3), strength),
            artifacts={"audio": torch.zeros(2, 18000), "audio_sampling_rate": 48000},
        )

    monkeypatch.setattr(NativeDiffusionAssembler, "build", lambda *args, **kwargs: SimpleNamespace(run=run))
    invocation = PipelineInvocation(
        request=None,
        prompt="A dancer.",
        image=None,
        video=str(control),
        interactions=None,
        ref_image_path=None,
        output_path=tmp_path / "result.mp4",
        operator_kwargs={},
        pipeline_kwargs={
            "control_type": control_type,
            "control_strength": 0.3,
            "offload": "none",
            "operator_kwargs": {},
        },
    )
    with pytest.raises(ValueError, match="does not accept images"):
        pipeline.run_pipeline_invocation(replace(invocation, image="image.png"))
    assert not requests
    first = pipeline.run_pipeline_invocation(invocation)
    with av.open(first["artifact_path"]) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, stream.frames) == (24, 16, 9)
        first_frame = next(container.decode(stream)).to_ndarray(format="rgb24")
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].sample_rate == 48000
    with pytest.raises(RuntimeError, match="IC-LoRA generation failed: fixture denoising failure"):
        pipeline.run_pipeline_invocation(replace(invocation, prompt="fail"))
    assert "failed: ValueError" in Path(first["log_path"]).read_text()
    changed = replace(invocation, pipeline_kwargs={**invocation.pipeline_kwargs, "control_strength": 0.7, "seed": 99})
    pipeline.run_pipeline_invocation(changed)
    final = pipeline.run_pipeline_invocation(invocation)
    with av.open(final["artifact_path"]) as container:
        last_frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
    assert (first_frame == last_frame).all()
    assert [request.inputs["control_strength"] for request in requests] == [0.3, 0.3, 0.7, 0.3]
    assert [request.sampling.seed for request in requests] == [171198, 171198, 99, 171198]


@pytest.mark.parametrize(
    "options,error",
    [
        ({"images": []}, "does not accept images"),
        ({"video": None}, "requires a control video"),
        ({"video_path": "duplicate.mp4"}, "pass either video or video_path"),
        ({"control_type": None}, "control_type must be one of"),
        ({"control_type": "rgb"}, "control_type must be one of"),
        ({"height": 64}, "at least 128 and divisible by 64"),
        ({"width": 130}, "at least 128 and divisible by 64"),
        ({"num_frames": 16}, "num_frames must be 8n"),
        ({"fps": 0}, "fps must be positive"),
        ({"control_strength": 1.1}, "control_strength must be"),
        ({"lora_strength": 0}, "lora_strength must be positive"),
        ({"offload": "invalid"}, "offload must be"),
        ({"unexpected": True}, "unsupported LTX IC-LoRA generation options"),
    ],
)
def test_ltx_v2v_rejects_invalid_requests_before_component_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], error: str
) -> None:
    pipeline = LTX2V2VPipeline.from_pretrained(_assets(tmp_path), device="cpu")
    control = tmp_path / "control.mp4"
    control.touch()
    monkeypatch.setattr(pipeline, "_build_native_pipeline", _no_subprocess)
    arguments: dict[str, Any] = {"prompt": "A dancer.", "video": control, "control_type": "pose", **options}
    with pytest.raises(ValueError, match=error):
        pipeline(**arguments)


@pytest.mark.parametrize("pipeline_type", [LTX2V2VPipeline, LTX23V2VPipeline])
def test_ltx_v2v_retires_interpreter_option_and_preserves_local_path_overrides(
    tmp_path: Path, pipeline_type: Any
) -> None:
    assets = _assets(tmp_path)
    with pytest.raises(ValueError, match="unsupported LTX IC-LoRA loading options:.*python_executable"):
        pipeline_type.from_pretrained(assets, python_executable="external-python")
    with pytest.raises(ValueError, match="unsupported LTX IC-LoRA loading options:.*python_executable"):
        pipeline_type.from_pretrained(assets, required_components={"python_executable": "external-python"})
    checkpoint = assets.pop("checkpoint_path")
    pipeline = pipeline_type.from_pretrained(checkpoint, required_components=assets, device="cpu", lazy=True)
    assert pipeline.checkpoint_path == checkpoint
    assert pipeline.gemma_root == assets["gemma_root"]
    assert pipeline.ic_lora_path == assets["ic_lora_path"]


@pytest.mark.parametrize("model_id", ["ltx-2-v2v", "ltx-2.3-v2v"])
def test_ltx_v2v_cli_spec_retains_control_fields_without_an_interpreter(model_id: str) -> None:
    from worldfoundry.cli.model_run import load_model_run_schema
    from worldfoundry.runtime.inference_specs.ltx_v2v import LTX2_V2V_INFERENCE_SPEC, LTX23_V2V_INFERENCE_SPEC

    spec = LTX2_V2V_INFERENCE_SPEC if model_id == "ltx-2-v2v" else LTX23_V2V_INFERENCE_SPEC
    fields = {field.field_id: field for field in spec.tasks[0].inputs}
    assert "python_executable" not in fields
    assert fields["control_type"].required
    assert fields["input_path"].required
    assert fields["offload"].choices == ("none", "cpu", "disk")
    assert {name for name, field in fields.items() if field.target == "load_kwargs"} == {
        "checkpoint_path",
        "spatial_upsampler_path",
        "ic_lora_path",
        "gemma_root",
    }
    schema = load_model_run_schema(model_id)
    assert not any("python-executable" in field.option for field in schema.fields)


@pytest.mark.parametrize("model_id", ["ltx-2-v2v", "ltx-2.3-v2v"])
def test_ltx_v2v_runtime_profile_selects_the_existing_ltx25_environment(model_id: str) -> None:
    from worldfoundry.runtime.conda import load_runtime_conda_env_spec
    from worldfoundry.zoo.bindings import resolve_pipeline_route, runtime_profile_execution_metadata

    environment = load_runtime_conda_env_spec(model_id)
    ltx25 = load_runtime_conda_env_spec("ltx-2.5")
    assert environment is not None and ltx25 is not None
    assert environment.env_name == ltx25.env_name == "ltx-2.5"
    assert environment.cuda_profile == ltx25.cuda_profile == "cu128"
    assert environment.python == ltx25.python == "3.11"
    assert environment.pip_packages == ltx25.pip_packages
    assert environment.conda_packages == ltx25.conda_packages
    assert environment.pip_extra_index_url == ltx25.pip_extra_index_url
    metadata = runtime_profile_execution_metadata(model_id)
    assert metadata["environment"] == "ltx-2.5"
    assert metadata["pipeline_binding"] == model_id
    route = resolve_pipeline_route(model_id=model_id, runtime_profile=model_id)
    assert route is not None
    target, binding, _source = route
    assert target.startswith("worldfoundry.pipelines.ltx2.pipeline_ltx2_v2v:")
    assert binding == model_id


def test_ltx_t2v_i2v_keep_the_original_gemma_loader_and_unified_environment() -> None:
    from worldfoundry.base_models.diffusion_model.components import ComponentKind
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import build_ltx_prompt_conditioner
    from worldfoundry.base_models.diffusion_model.recipes import default_native_diffusion_registry
    from worldfoundry.runtime.conda import load_runtime_conda_env_spec
    from worldfoundry.zoo.bindings import runtime_profile_execution_metadata

    for model_id in ("ltx-2-t2v", "ltx-2.3-t2v", "ltx-2-i2v", "ltx-2.3-i2v"):
        recipe = default_native_diffusion_registry().resolve(model_id)
        conditioner = next(c for c in recipe.components if c.key.kind is ComponentKind.CONDITIONER)
        assert conditioner.factory is build_ltx_prompt_conditioner
        if model_id.endswith("-t2v"):
            assert runtime_profile_execution_metadata(model_id)["environment"] == "_unified"
        else:
            environment = load_runtime_conda_env_spec(model_id)
            assert environment is not None
            assert environment.env_name == "worldfoundry-unified-cu128"
