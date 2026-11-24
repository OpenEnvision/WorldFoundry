from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from worldfoundry.evals.api import WorldModelConfig
from worldfoundry.evals.models.pipelines.loading import build_pipeline_runner_spec, load_pipeline_from_spec
from worldfoundry.pipelines.ltx25 import LTX25Pipeline
from worldfoundry.synthesis.visual_generation.ltx25.runtime import (
    DEFAULT_COMPONENTS,
    LTX25DistilledRuntime,
)
from worldfoundry.zoo.bindings import load_pipeline_binding
from worldfoundry.zoo.catalog import load_model_catalog_manifest
from worldfoundry.zoo.runtime.environments import load_runtime_environment_profile
from worldfoundry.zoo.runtime.profiles import load_runtime_profile_manifest

MODEL_DATA = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models"


def _local_runtime(tmp_path: Path, **kwargs: Any) -> LTX25DistilledRuntime:
    checkpoint_dir = tmp_path / "Lightricks--LTX-2.5"
    for relative in DEFAULT_COMPONENTS.values():
        path = checkpoint_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local-checkpoint-fixture")
    return LTX25DistilledRuntime(checkpoint_dir=checkpoint_dir, **kwargs)


def test_ltx25_catalog_binding_profile_and_environment_are_wired() -> None:
    catalog = load_model_catalog_manifest(MODEL_DATA / "catalog" / "video" / "ltx-2.5.yaml")
    binding = load_pipeline_binding(MODEL_DATA / "bindings" / "pipelines" / "ltx-2.5.yaml")
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime" / "profiles" / "ltx-2.5.yaml")
    environment = load_runtime_environment_profile(MODEL_DATA / "runtime" / "environments" / "video" / "ltx-2.5.yaml")
    assert catalog.model_id == "ltx-2.5"
    assert catalog.task_family == "video_generation"
    assert catalog.integration["status"] == "integrated"
    assert catalog.integration["pipeline_binding"] == "ltx-2.5"
    assert binding.model_id == "ltx-2.5"
    assert binding.pipeline_target == "worldfoundry.pipelines.ltx25.pipeline_ltx25:LTX25Pipeline"
    assert profile.execution["runtime_status"] == "native_checkpoint_demo_pending"
    assert profile.execution["pipeline_binding"] == "ltx-2.5"
    assert profile.artifact_kind == "generated_video"
    assert profile.artifact_filename == "ltx-2.5.mp4"
    assert environment.model_id == "ltx-2.5"
    assert environment.env_name == "ltx-2.5"
    assert environment.python == "3.11"
    for alias in ("ltx-2.5", "ltx2.5", "ltx25", "ltx-2.5-distilled"):
        spec = build_pipeline_runner_spec(WorldModelConfig(model_id=alias, runner="worldfoundry.pipeline"))
        assert spec.pipeline_target == binding.pipeline_target


def test_ltx25_pipeline_loads_from_standard_runner_spec() -> None:
    config = WorldModelConfig(model_id="ltx-2.5", runner="worldfoundry.pipeline", runtime={"device": "cpu"})
    spec = build_pipeline_runner_spec(config)
    pipeline = load_pipeline_from_spec(spec)
    assert spec.pipeline_target == "worldfoundry.pipelines.ltx25.pipeline_ltx25:LTX25Pipeline"
    assert pipeline.model_id == "ltx-2.5"
    assert pipeline.preflight()["status"] == "blocked"


def test_ltx25_preflight_and_split_checkpoint_plan(tmp_path: Path) -> None:
    runtime = _local_runtime(tmp_path)
    preflight = runtime.preflight()
    assert preflight["missing_checkpoint_files"] == []
    assert preflight["incomplete_checkpoint_files"] == []
    assert preflight["component_paths"]["duration_head_path"].endswith("ltx-2.5-duration-head-bf16.safetensors")

    image = tmp_path / "reference.png"
    image.write_bytes(b"fixture")
    plan = runtime.build_plan(
        request={"prompt": "A city at sunset.", "num_frames": 121, "height": 512, "width": 768, "images": [str(image)]},
        output_path=tmp_path / "result.mp4",
    )
    assert plan.task == "image-to-video"
    assert plan.request["images"] == [{"path": str(image), "frame_index": 0, "strength": 1.0, "crf": None}]
    assert plan.to_dict()["recipe"] == "ltx-2.5"
    assert plan.offload_mode == "cpu"
    assert plan.output_path.endswith("result.mp4")
    assert not {"command", "work_dir", "python_executable"} & plan.to_dict().keys()


def test_ltx25_preflight_rejects_in_progress_component_files(tmp_path: Path) -> None:
    runtime = _local_runtime(tmp_path)
    empty = runtime.checkpoint_dir / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    empty.write_bytes(b"")
    downloading = runtime.checkpoint_dir / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    downloading.with_name(f"{downloading.name}.aria2").write_bytes(b"aria2")

    preflight = runtime.preflight()

    assert preflight["status"] == "blocked"
    assert preflight["missing_checkpoint_files"] == []
    assert set(preflight["incomplete_checkpoint_files"]) == {str(empty), str(downloading)}


def test_ltx25_pipeline_requires_explicit_execution() -> None:
    pipeline = LTX25Pipeline.from_pretrained(device="cpu")
    with pytest.raises(RuntimeError, match="execute=True"):
        pipeline(prompt="A quiet forest")


def test_ltx25_rejects_invalid_shapes_and_load_options(tmp_path: Path) -> None:
    runtime = _local_runtime(tmp_path)
    for request, error in (
        ({"prompt": " "}, "non-empty prompt"),
        ({"prompt": "A scene.", "width": 770}, "multiples of 64"),
        ({"prompt": "A scene.", "num_frames": 120}, "8 \\* k \\+ 1"),
    ):
        with pytest.raises(ValueError, match=error):
            runtime.build_plan(request=request, output_path=tmp_path / "video.mp4")
    with pytest.raises(TypeError, match="Unknown LTX-2.5 load options"):
        LTX25Pipeline.from_pretrained(tmp_path, tyop=True)


@pytest.mark.parametrize("option", ["python_executable", "python_env_dir", "conda_dir"])
def test_removed_interpreter_options_use_unknown_load_option_error(tmp_path: Path, option: str) -> None:
    options: dict[str, Any] = {option: str(tmp_path)}
    with pytest.raises(TypeError, match=f"Unknown LTX-2.5 load options: {option}"):
        LTX25Pipeline.from_pretrained(tmp_path, **options)
    with pytest.raises(TypeError, match=f"Unknown LTX-2.5 load options: {option}"):
        LTX25Pipeline.from_pretrained(tmp_path, required_components={option: str(tmp_path)})


@pytest.mark.parametrize("mode", ["fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"])
def test_native_load_preserves_quantization_and_split_checkpoint_options(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    import torch

    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline

    calls: list[tuple[str, dict[str, Any]]] = []
    native = object()

    def load(recipe: str, **kwargs: Any) -> object:
        calls.append((recipe, kwargs))
        return native

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    runtime = _local_runtime(tmp_path, quantization=mode, device="cuda:1")
    plan = runtime.build_plan(request={"prompt": "A scene."}, output_path=tmp_path / "result.mp4")
    assert runtime._native_pipeline(plan) is native
    assert runtime._native_pipeline(plan) is native
    assert len(calls) == 1
    recipe, options = calls[0]
    assert recipe == "ltx-2.5"
    assert options["policy"].device == torch.device("cuda:1")
    assert options["policy"].dtype is torch.bfloat16
    assert options["policy"].offload.mode.value == "block"
    assert options["policy"].quantization.mode.value == mode
    assert set(options["checkpoint_overrides"]) == {
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "upsampler",
        "duration_head",
    }
    assert all(Path(value.source[0]).is_file() for value in options["checkpoint_overrides"].values())
    assert options["component_options"] == {
        "conditioner:main": {"load_duration_head": True},
        "decoder:main": {"diffvae_optimization": "chunked_eager"},
    }


def test_native_run_translates_request_and_preserves_artifact_contract(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from worldfoundry.core.media.codecs import pyav

    runtime = _local_runtime(tmp_path)
    pipeline = LTX25Pipeline(runtime=runtime)
    requests = []
    writes = []

    def native(request):
        requests.append(request)
        return SimpleNamespace(sample="frames", artifacts={"audio": "waveform", "audio_sampling_rate": 48000})

    def write(frames, path, **kwargs):
        writes.append((frames, kwargs))
        Path(path).write_bytes(b"native-artifact")

    def no_subprocess(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("LTX-2.5 native execution must not invoke a model subprocess")

    monkeypatch.setattr(runtime, "_native_pipeline", lambda plan: native)
    monkeypatch.setattr(runtime, "preflight", lambda: {"status": "ready"})
    monkeypatch.setattr(pyav, "save_audio_video_h264", write)
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    result = pipeline(
        prompt="A scene.",
        output_path=tmp_path / "movie.mp4",
        num_frames=17,
        fps=25,
        seed=123,
        execute=True,
        return_dict=True,
    )

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["runtime"] == "worldfoundry.native_diffusion"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"] == str(tmp_path / "movie.mp4")
    assert result["artifact_files"] == [str(tmp_path / "movie.mp4")]
    assert result["metadata_path"] == str(tmp_path / "movie.json")
    assert requests[0].sampling.seed == 123
    assert requests[0].sampling.num_inference_steps == 11
    assert requests[0].num_frames == 17
    assert requests[0].inputs["frame_rate"] == 25
    assert requests[0].inputs["deadline_monotonic"] > 0
    assert writes[0][0] == "frames"
    assert writes[0][1]["sample_rate"] == 48000
    assert writes[0][1]["waveform"] == "waveform"
    assert writes[0][1]["fps"] == 25
    assert (tmp_path / "movie.json").is_file()


def test_native_timeout_is_cooperative_and_precedes_loading(tmp_path: Path, monkeypatch) -> None:
    runtime = _local_runtime(tmp_path)
    plan = runtime.build_plan(request={"prompt": "A scene."}, output_path=tmp_path / "video.mp4")

    def expired_load(plan: Any) -> NoReturn:
        raise AssertionError("expired request loaded weights")

    monkeypatch.setattr(runtime, "_native_pipeline", expired_load)
    with pytest.raises(TimeoutError, match="component loading"):
        runtime.run_plan(plan, timeout_seconds=-1)
    assert not Path(plan.output_path).exists()


def test_ltx25_runtime_supports_text_only_and_auto_duration(tmp_path: Path) -> None:
    runtime = _local_runtime(tmp_path)
    plan = runtime.build_plan(
        request={"prompt": "A scene.", "auto_duration": [2, 10], "num_generated_keyframes": 2},
        output_path=tmp_path / "video.mp4",
    )
    assert plan.request["num_frames"] is None
    assert plan.request["auto_duration"] == (2.0, 10.0)
    assert plan.request["num_generated_keyframes"] == 2
    runtime.duration_head_path.unlink()
    with pytest.raises(FileNotFoundError, match="duration_head_path"):
        runtime.build_plan(request=plan.request, output_path=plan.output_path)
