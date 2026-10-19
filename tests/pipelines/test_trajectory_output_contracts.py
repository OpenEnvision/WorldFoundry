# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Trajectory output boundaries with real CPU pixels and substituted generation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.runtime_v1 import Lyra1Representation
from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.pipelines.gen3c import pipeline_gen3c
from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline
from worldfoundry.pipelines.kling.pipeline_astra import AstraConfig, AstraPipeline
from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline
from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline
from worldfoundry.studio.inference.execution import _normalize_frame_list, _safe_json
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from worldfoundry.synthesis.visual_generation.lyra_1.synthesis import Lyra1Synthesis
from worldfoundry.synthesis.visual_generation.lyra_1.worldfoundry_runtime import Lyra1Runtime
from worldfoundry.synthesis.visual_generation.memory.kling.astra import AstraMemory

pytestmark = pytest.mark.unit


def _assert_pixels(result: dict[str, Any], count: int, *, layout: str, value_range: str) -> None:
    chunk = result["video_chunk"]
    raw = result["video"] if "video" in result else result["frames"]
    assert isinstance(chunk, VideoChunk) and chunk.frames is raw
    assert chunk.layout == layout and chunk.value_range == value_range
    assert chunk.frame_count == count
    expected = np.full((count, 3, 4, 3), 191, dtype=np.uint8)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)
    preview = _normalize_frame_list(chunk)
    assert preview is not None
    np.testing.assert_array_equal(np.stack([np.asarray(frame) for frame in preview]), expected)
    descriptor = _safe_json(result)["video_chunk"]
    assert descriptor["frame_count"] == count
    assert descriptor["layout"] == layout and descriptor["value_range"] == value_range
    assert chunk.metadata["fps"] == result["fps"]


def _gen3c(
    count: int, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype = torch.float32
) -> tuple[Gen3CPipeline, torch.Tensor, list[Any], list[np.ndarray]]:
    sample = torch.full((1, 3, count, 3, 4), 0.5, dtype=dtype)
    latents = torch.arange(64, dtype=torch.float32).reshape(1, 16, 1, 2, 2)
    requests: list[Any] = []
    encoded: list[np.ndarray] = []
    cameras = torch.eye(4).repeat(1, 121, 1, 1)
    intrinsics = torch.eye(3).repeat(1, 121, 1, 1)

    def native(request: Any) -> Any:
        requests.append(request)
        return SimpleNamespace(
            sample=latents if request.inputs["return_latent"] else sample,
            latents=latents,
            metadata={"trajectory": request.metadata["trajectory"]},
            artifacts={"camera_to_world": cameras, "camera_intrinsics": intrinsics},
        )

    def save(value: torch.Tensor, path: str | Path, *, fps: int, layout: str, value_range: str) -> str:
        assert value is sample and fps == 12
        assert layout == "bcthw" and value_range == "-1,1"
        encoded.append(video_chunk_to_uint8_frames(VideoChunk(frames=value, layout=layout, value_range=value_range)))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"recorded video")
        return str(target)

    monkeypatch.setattr(pipeline_gen3c, "save_image_or_video_tensor", save)
    return Gen3CPipeline(native_pipeline=native, device="cpu"), sample, requests, encoded


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("save", [False, True])
def test_gen3c_declares_signed_bcthw_without_replacing_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, dtype: torch.dtype, route: str, save: bool
) -> None:
    pipeline, sample, requests, encoded = _gen3c(count, monkeypatch, dtype)
    original = sample.clone()
    method = pipeline if route == "batch" else pipeline.stream
    kwargs: dict[str, Any] = {"images": Image.new("RGB", (4, 3)), "interactions": ["left"], "fps": 12}
    if save:
        kwargs["output_path"] = tmp_path / "output.mp4"
    result = method(return_dict=True, **kwargs)
    _assert_pixels(result, count, layout="bcthw", value_range="-1,1")
    assert result["video"] is result["generated_video"] is sample
    assert result["latents"].shape[1] == 16 and result["trajectory"] == "left"
    assert result["artifacts"]["camera_to_world"].shape == (1, 121, 4, 4)
    assert requests[0].inputs["fps"] == 12
    if route == "stream":
        assert pipeline.memory_module.current_video is sample
        assert pipeline.memory_module.select(context_query="latest_result") is result
        assert len(pipeline.memory_module.all_frames) == count
        np.testing.assert_array_equal(np.asarray(pipeline.memory_module.select()), np.full((3, 4, 3), 191, np.uint8))
        kwargs["images"] = None
    if save:
        assert Path(result["artifact_path"]).read_bytes() == b"recorded video"
        np.testing.assert_array_equal(encoded[0], np.full((count, 3, 4, 3), 191, np.uint8))
    else:
        assert result["artifact_path"] is None and encoded == []
    raw = method(**kwargs)
    assert raw == str(kwargs["output_path"]) if save else raw is sample
    torch.testing.assert_close(sample, original, rtol=0, atol=0)


def test_gen3c_declaration_preserves_nonuniform_time_and_rgb_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, sample, _, _ = _gen3c(5, monkeypatch)
    sample.copy_((torch.arange(sample.numel()).reshape_as(sample) % 32) / 32)
    result = pipeline(images=Image.new("RGB", (4, 3)), return_dict=True, fps=12)
    expected = ((sample.permute(0, 2, 3, 4, 1).numpy()[0] + 1) * 127.5).astype(np.uint8)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(result["video_chunk"]), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)


def test_gen3c_latents_do_not_claim_rgb_or_replace_rollout_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, sample, _, encoded = _gen3c(5, monkeypatch)
    pipeline.stream(images=Image.new("RGB", (4, 3)), fps=12)
    last_image = pipeline.memory_module.select()
    result = pipeline.stream(output_type="latent", output_path=tmp_path / "latent.mp4", fps=12, return_dict=True)
    assert result["video"] is result["latents"]
    assert result["video_chunk"] is None and result["artifact_path"] is None
    assert pipeline.memory_module.current_video is sample and pipeline.memory_module.select() is last_image
    assert len(pipeline.memory_module.all_frames) == 5 and encoded == []
    assert not (tmp_path / "latent.mp4").exists()


def _lyra1(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[Lyra1Pipeline, torch.Tensor, list[Any], list[dict[str, Any]], dict[str, Any]]:
    gen3c, sample, requests, _ = _gen3c(count, monkeypatch)
    runtime = Lyra1Runtime(device="cpu")
    monkeypatch.setattr(runtime, "_native_pipeline", lambda: gen3c)
    reconstruction_calls: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {"points": np.zeros((2, 3)), "scene": "scene", "camera": np.eye(4)}

    def reconstruct(value: dict[str, Any]) -> dict[str, Any]:
        reconstruction_calls.append(value)
        return geometry

    return (
        Lyra1Pipeline(
            synthesis_model=Lyra1Synthesis(runtime),
            representation_model=cast(Lyra1Representation, SimpleNamespace(get_representation=reconstruct)),
            device="cpu",
        ),
        sample,
        requests,
        reconstruction_calls,
        geometry,
    )


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
@pytest.mark.parametrize("reconstruct", [False, True])
def test_lyra1_keeps_geometry_and_declares_primary_video_through_native_gen3c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, route: str, mode: str, reconstruct: bool
) -> None:
    pipeline, sample, requests, reconstruction_calls, geometry = _lyra1(count, monkeypatch)
    method = pipeline if route == "batch" else pipeline.stream
    source = Image.new("RGB", (4, 3)) if mode == "static" else np.zeros((2, 3, 4, 3), np.uint8)
    kwargs: dict[str, Any] = {
        "images" if mode == "static" else "videos": source,
        "mode": mode,
        "interactions": ["zoom_in"],
        "fps": 12,
        "execute": True,
        "multi_trajectory": False,
        "reconstruct_3d": reconstruct,
        "output_dir": str(tmp_path),
    }
    if mode == "dynamic":
        kwargs.update(
            rendered_warp_images=torch.zeros(1, 3, 121, 3, 4), rendered_warp_masks=torch.ones(1, 1, 121, 3, 4)
        )
    result = method(return_dict=True, **kwargs)
    _assert_pixels(result, count, layout="bcthw", value_range="-1,1")
    assert result["video"] is sample and result["trajectory"] == "zoom_in"
    assert result["actions"] == ["zoom_in"] and result["mapped_trajectories"] == ["zoom_in"]
    assert result["reconstruction"] is (geometry if reconstruct else None)
    if reconstruct:
        assert reconstruction_calls[0]["generated_root"] == result["generated_root"]
        assert reconstruction_calls[0]["mode"] == mode
    with np.load(tmp_path / "generated" / "pose" / "lyra1.npz") as saved:
        np.testing.assert_array_equal(saved["data"], np.repeat(np.eye(4)[None], 121, axis=0))
    latent = torch.load(tmp_path / "generated" / "latent" / "lyra1.pkl", weights_only=False)
    np.testing.assert_array_equal(latent, np.arange(64, dtype=np.float32).reshape(1, 1, 16, 2, 2))
    if route == "stream":
        assert pipeline.memory_module.current_video is sample
        kwargs["images" if mode == "static" else "videos"] = None
        second = method(return_dict=True, **kwargs)
        _assert_pixels(second, count, layout="bcthw", value_range="-1,1")
        np.testing.assert_array_equal(np.asarray(requests[-1].inputs["image"]), np.full((3, 4, 3), 191, np.uint8))
    else:
        raw = method(**kwargs)
        assert raw["reconstruction"] is geometry if reconstruct else raw is sample


def test_lyra1_multi_trajectory_retains_each_declaration_and_reconstruction_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, sample, requests, _, _ = _lyra1(3, monkeypatch)
    assert pipeline.synthesis_model is not None
    runtime = pipeline.synthesis_model.runtime
    result = runtime.predict(
        visual_input=Image.new("RGB", (4, 3)),
        output_root=str(tmp_path),
        execute=True,
        multi_trajectory=True,
        trajectory="zoom_in",
        fps=12,
    )
    assert len(requests) == len(result["trajectories"]) == 6
    assert result["video_chunk"] is result["trajectories"]["zoom_in"]["video_chunk"]
    for index, trajectory in enumerate(runtime._MULTI_TRAJECTORIES):
        output = result["trajectories"][trajectory]
        _assert_pixels(output, 3, layout="bcthw", value_range="-1,1")
        assert output["video"] is sample and output["trajectory"] == trajectory
        assert (tmp_path / str(index) / "intrinsics" / "lyra1.npz").is_file()


@pytest.mark.parametrize("return_dict", [False, True])
def test_lyra1_stream_plan_does_not_replace_generated_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, return_dict: bool
) -> None:
    pipeline, sample, requests, _, _ = _lyra1(5, monkeypatch)
    pipeline.stream(
        images=Image.new("RGB", (4, 3)),
        interactions=["left"],
        execute=True,
        fps=12,
        output_dir=str(tmp_path),
        multi_trajectory=False,
    )
    memory = pipeline.memory_module
    last_result, last_image, records = memory.latest_result, memory.current_image, len(memory.storage)
    plan = pipeline.stream(
        interactions=["right"],
        execute=False,
        return_dict=return_dict,
        fps=12,
        output_dir=str(tmp_path),
        reconstruct_3d=True,
    )
    assert plan["status"] == "planned" and "video_chunk" not in plan and "video" not in plan
    assert json.loads(Path(plan["artifact_path"]).read_text())["trajectory"] == "right"
    assert len(requests) == 1 and memory.latest_result is last_result and memory.current_image is last_image
    assert memory.current_video is sample and len(memory.storage) == records


def _astra(monkeypatch: pytest.MonkeyPatch) -> tuple[AstraPipeline, list[dict[str, Any]], torch.Tensor]:
    decoded = torch.full((1, 3, 33, 3, 4), 0.5)
    synthesis = AstraSynthesis(
        SimpleNamespace(
            device="cpu",
            decode_video=lambda *args, **kwargs: decoded,
            vae=SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None)),
        )
    )
    calls: list[dict[str, Any]] = []

    def process(image: str, interaction: dict[str, Any], args: AstraConfig) -> dict[str, Any]:
        assert image == "input.png" and args.frames_per_generation == 8
        directions = interaction["direction"]
        direction_count = 1 if isinstance(directions, str) else len(directions)
        total = args.total_frames_to_generate * direction_count
        return {
            "initial_latents": torch.zeros(3, 1, 2, 2),
            "prompt_emb_pos": {"prompt": interaction["prompt"]},
            "prompt_emb_neg": None,
            "camera_embedding_full": torch.zeros(total + 31, 13),
            "camera_embedding_uncond": None,
            "total_frames": total,
        }

    def predict(framepack: dict[str, Any], count: int, *args: Any, generator: torch.Generator) -> torch.Tensor:
        assert count == 8 and framepack["latent_indices"].shape == (8,)
        assert generator.initial_seed() == 7
        calls.append(framepack)
        return torch.randn(1, 3, count, 2, 2, generator=generator)

    monkeypatch.setattr(synthesis, "predict", predict)
    monkeypatch.setattr(AstraSynthesis, "from_pretrained", lambda config, device: synthesis)
    pipeline = AstraPipeline.from_pretrained("astra", {"wan_model_path": "wan"}, "cpu")
    monkeypatch.setattr(pipeline, "process", process)
    return pipeline, calls, decoded


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("save", [False, True])
def test_astra_declares_cropped_pil_rgb_without_changing_latent_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, save: bool
) -> None:
    pipeline, calls, decoded = _astra(monkeypatch)
    original = decoded.clone()
    rng = torch.random.get_rng_state().clone()
    encoded: list[Any] = []

    def write(frames: list[Image.Image], path: Path, *, fps: int) -> None:
        assert fps == 12
        encoded.append(frames)
        path.write_bytes(b"recorded astra")

    monkeypatch.setattr("worldfoundry.core.io.write_video", write)
    kwargs: dict[str, Any] = {
        "image_path": "input.png",
        "interactions": ["forward"],
        "num_frames": count,
        "fps": 12,
        "seed": 7,
    }
    if save:
        kwargs["output_path"] = tmp_path / "astra.mp4"
    result = pipeline(return_dict=True, **kwargs)
    assert isinstance(result, dict)
    _assert_pixels(result, count, layout="frame-list", value_range="0,255")
    assert result["num_frames"] == count and len(calls) == 1
    assert isinstance(pipeline.memory, AstraMemory) and pipeline.memory.history_latents is not None
    assert pipeline.memory.history_latents.shape == (3, 9, 2, 2)
    assert calls[0]["current_length"] == 1 and calls[0]["next_length"] == 9
    assert pipeline.config.total_frames_to_generate == 24 and pipeline.config.frames_per_generation == 8
    if save:
        assert encoded[0] is result["frames"] and Path(result["output_path"]).read_bytes() == b"recorded astra"
    else:
        assert result["output_path"] is None and encoded == []
    raw = pipeline(**kwargs)
    assert isinstance(raw, list) and all(isinstance(frame, Image.Image) for frame in raw)
    np.testing.assert_array_equal(
        np.stack([np.asarray(frame) for frame in raw]), np.full((count, 3, 4, 3), 191, np.uint8)
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(decoded, original, rtol=0, atol=0)


def test_astra_declaration_preserves_decoder_time_rgb_order_and_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, _, decoded = _astra(monkeypatch)
    decoded.copy_((torch.arange(decoded.numel()).reshape_as(decoded) % 32) / 32)
    result = pipeline(image_path="input.png", num_frames=5, seed=7, return_dict=True)
    assert isinstance(result, dict)
    expected = ((decoded.permute(0, 2, 3, 4, 1).numpy()[0, :5] * 0.5 + 0.5) * 255).astype(np.uint8)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(result["video_chunk"]), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)


def test_gen3c_public_loader_delegates_to_canonical_native_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, sample, _, _ = _gen3c(5, monkeypatch)
    calls: list[dict[str, Any]] = []

    def load(model_id: str, **kwargs: Any) -> Any:
        calls.append({"model_id": model_id, **kwargs})
        return pipeline.native_pipeline

    monkeypatch.setattr(pipeline_gen3c.NativeDiffusionPipeline, "from_pretrained", load)
    loaded = Gen3CPipeline.from_pretrained(None, {"torch_dtype": "float32", "offload_mode": "none"}, "cpu")
    result = loaded(images=Image.new("RGB", (4, 3)), fps=12, return_dict=True)
    _assert_pixels(result, 5, layout="bcthw", value_range="-1,1")
    assert result["video"] is sample and loaded.native_pipeline is pipeline.native_pipeline
    assert calls[0]["model_id"] == "gen3c" and calls[0]["policy"].dtype == torch.float32
    assert str(calls[0]["policy"].device) == "cpu" and calls[0]["checkpoint_overrides"] is None


def test_lyra1_public_loader_retains_positional_dtype_and_declared_native_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, sample, _, _, _ = _lyra1(5, monkeypatch)
    calls: list[dict[str, Any]] = []

    def load(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return pipeline.synthesis_model

    monkeypatch.setattr(Lyra1Synthesis, "from_pretrained", load)
    loaded = Lyra1Pipeline.from_pretrained(None, {}, "cpu", torch.float32)
    result = loaded(
        images=Image.new("RGB", (4, 3)),
        interactions=["left"],
        execute=True,
        output_dir=str(tmp_path),
        fps=12,
        return_dict=True,
    )
    _assert_pixels(result, 5, layout="bcthw", value_range="-1,1")
    assert result["video"] is sample and loaded.weight_dtype == torch.float32 and calls[0]["device"] == "cpu"


@pytest.mark.parametrize(
    "pipeline_cls,method,names",
    [
        (Gen3CPipeline, "stream", ["self", "interactions", "images", "prompt", "reset_memory"]),
        (
            Lyra1Pipeline,
            "__call__",
            [
                "self",
                "images",
                "videos",
                "interactions",
                "prompt",
                "mode",
                "trajectory",
                "reconstruct_3d",
                "output_dir",
                "return_dict",
            ],
        ),
        (
            Lyra1Pipeline,
            "stream",
            [
                "self",
                "interactions",
                "images",
                "videos",
                "prompt",
                "mode",
                "trajectory",
                "reconstruct_3d",
                "output_dir",
                "reset_memory",
                "return_dict",
            ],
        ),
        (
            Lyra2Pipeline,
            "stream",
            [
                "self",
                "interactions",
                "images",
                "prompt",
                "camera_path",
                "region_hint",
                "fps",
                "resolution",
                "reconstruct_3d",
                "output_dir",
                "seed",
                "reset_memory",
                "return_dict",
            ],
        ),
        (
            AstraPipeline,
            "__call__",
            [
                "self",
                "image_path",
                "interactions",
                "images",
                "prompt",
                "output_path",
                "fps",
                "return_dict",
                "num_frames",
                "frames_per_generation",
                "total_frames_to_generate",
                "num_inference_steps",
                "seed",
                "height",
                "width",
            ],
        ),
    ],
)
def test_output_declarations_preserve_existing_positional_parameters(
    pipeline_cls: type, method: str, names: list[str]
) -> None:
    parameters = inspect.signature(getattr(pipeline_cls, method)).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == names


def test_astra_default_frame_rate_matches_its_export_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, _, _ = _astra(monkeypatch)
    result = pipeline(image_path="input.png", num_frames=5, seed=7, return_dict=True)
    assert isinstance(result, dict)
    assert result["fps"] == result["video_chunk"].metadata["fps"] == 16


@pytest.mark.parametrize(
    "source",
    [
        np.empty((0, 3, 4, 3), np.uint8),
        VideoChunk(frames=np.empty((0, 3, 4, 3), np.uint8), layout="thwc", value_range="0,255"),
    ],
)
def test_lyra1_dynamic_reference_rejects_empty_video(source: Any) -> None:
    with pytest.raises(ValueError, match="dynamic input contains no frames"):
        Lyra1Runtime._dynamic_reference(source)
