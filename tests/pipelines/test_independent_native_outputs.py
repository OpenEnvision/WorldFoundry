# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Independent native adapters retain raw results and declare decoded pixel semantics."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.studio.inference.execution import _normalize_frame_list, _safe_json
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result

pytestmark = pytest.mark.unit

ROUTES = [
    ("cosmos.pipeline_cosmos_predict2", "CosmosPredict2Pipeline", "cosmos-predict2-2b-video2world", "-1,1", 16),
    ("cosmos.pipeline_cosmos_predict2p5", "CosmosPredict2p5Pipeline", "cosmos-predict2.5-2b", "-1,1", 16),
    (
        "cosmos.pipeline_cosmos_transfer2p5",
        "CosmosTransfer2p5Pipeline",
        "cosmos-transfer2.5-2b-controlled-video",
        "-1,1",
        16,
    ),
    ("cosmos.pipeline_cosmos3", "Cosmos3Pipeline", "cosmos3-nano", "-1,1", 24),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideoT2VPipeline", "hunyuanvideo-t2v", "0,1", 24),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideoI2VPipeline", "hunyuanvideo-i2v", "0,1", 24),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideo15T2VPipeline", "hunyuanvideo-1.5-t2v", "0,1", 24),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideo15I2VPipeline", "hunyuanvideo-1.5-i2v", "0,1", 24),
    ("t2v_turbo.pipeline_t2v_turbo_t2v", "T2VTurboT2VPipeline", "t2v_turbo_t2v", "-1,1", 8),
    ("vchitect.pipeline_vchitect_2_t2v", "Vchitect2T2VPipeline", "vchitect-2-t2v", "-1,1", 8),
]
STREAM_ROUTES = ROUTES[4:]


def _adapter(route: tuple, native: Any) -> Any:
    module, name, model_id, *_ = route
    cls = getattr(importlib.import_module(f"worldfoundry.pipelines.{module}"), name)
    options = {"native_pipeline": native, "device": "cpu"}
    if "model_id" in inspect.signature(cls).parameters:
        options["model_id"] = model_id
    return cls(**options)


def _inputs(route: tuple) -> dict[str, Any]:
    module, name, *_ = route
    if name == "CosmosPredict2Pipeline" or name.endswith("I2VPipeline"):
        return {"images": Image.new("RGB", (4, 3), "red")}
    if "transfer" in module:
        return {"control_video": torch.zeros(1, 3, 5, 3, 4)}
    return {}


def _native(sample: torch.Tensor) -> tuple[Any, list[Any], torch.Tensor]:
    requests: list[Any] = []
    latents = torch.arange(80, dtype=torch.float32).reshape(1, 4, 5, 2, 2)

    def native(request: Any) -> Any:
        requests.append(request)
        return SimpleNamespace(
            sample=latents if request.inputs["return_latent"] else sample,
            latents=latents,
            metadata={"source": "native-boundary"},
            artifacts={},
        )

    return native, requests, latents


def _expected(sample: torch.Tensor, value_range: str) -> np.ndarray:
    pixels = sample.float()
    pixels = (pixels + 1) * 127.5 if value_range == "-1,1" else pixels * 255
    return pixels.clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 3, 0).numpy()


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_decoded_pixels_follow_declared_axes_and_range(route: tuple, count: int, dtype: torch.dtype) -> None:
    sample = torch.full((1, 3, count, 3, 4), 0.5, dtype=dtype)
    sample[:, 1, :, 0, 0] = 0
    sample[:, 2, :, 0, 0] = 1
    sample[:, 0, :, 1, 2] = torch.arange(count, dtype=dtype) / 8
    original = sample.clone()
    native, requests, latents = _native(sample)
    pipeline = _adapter(route, native)
    result = pipeline("scene", return_dict=True, fps=12, num_frames=93, **_inputs(route))
    chunk = result["video_chunk"]
    assert isinstance(chunk, VideoChunk) and chunk.frames is result["video"] is sample
    assert chunk.layout == "bcthw" and chunk.value_range == route[3] and chunk.frame_count == count
    assert chunk.metadata["fps"] == result["fps"] == requests[0].inputs["fps"] == 12
    assert result["latents"] is latents and result["metadata"] == {"source": "native-boundary"}
    assert requests[0].num_frames == 93 and not requests[0].inputs["return_latent"]
    expected = _expected(sample, route[3])
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)
    frames = _normalize_frame_list(chunk)
    assert frames is not None
    np.testing.assert_array_equal(np.stack(frames), expected)
    assert _safe_json(result)["video_chunk"]["frame_count"] == count
    assert pipeline("scene", **_inputs(route)) is sample
    if "generated_video" in result:
        assert result["generated_video"] is sample
    torch.testing.assert_close(sample, original, rtol=0, atol=0)


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
@pytest.mark.parametrize("save", [False, True])
def test_export_and_raw_return_use_the_same_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: tuple, save: bool
) -> None:
    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    native, requests, _ = _native(sample)
    module = importlib.import_module(f"worldfoundry.pipelines.{route[0]}")
    writes: list[np.ndarray] = []

    def write(value: Any, path: Any, *, fps: int, layout: str, value_range: str) -> str:
        assert value is sample and fps == 12 and layout == "bcthw" and value_range == route[3]
        writes.append(video_chunk_to_uint8_frames(VideoChunk(frames=value, layout=layout, value_range=value_range)))
        Path(path).write_bytes(b"native-artifact")
        return str(path)

    monkeypatch.setattr(module, "save_image_or_video_tensor", write)
    pipeline = _adapter(route, native)
    options = {"output_path": tmp_path / "output.mp4"} if save else {}
    result = pipeline("scene", fps=12, return_dict=True, **_inputs(route), **options)
    raw = pipeline("scene", fps=12, **_inputs(route), **options)
    assert raw == str(options["output_path"]) if save else raw is sample
    assert result["artifact_path"] == (str(options["output_path"]) if save else None)
    assert result["fps"] == 12 and len(requests) == 2
    assert len(writes) == (2 if save else 0)
    for pixels in writes:
        np.testing.assert_array_equal(pixels, _expected(sample, route[3]))
    if save:
        assert Path(result["artifact_path"]).read_bytes() == b"native-artifact"


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
def test_single_decoded_frame_roundtrips_through_real_png(tmp_path: Path, route: tuple) -> None:
    sample = torch.full((1, 3, 1, 3, 4), 0.5)
    native, _, _ = _native(sample)
    result = _adapter(route, native)("scene", output_path=tmp_path / "frame.png", return_dict=True, **_inputs(route))
    with Image.open(result["artifact_path"]) as image:
        np.testing.assert_array_equal(np.asarray(image), _expected(sample, route[3])[0])
    assert result["fps"] == result["video_chunk"].metadata["fps"] == route[4]


@pytest.mark.parametrize("route", STREAM_ROUTES, ids=[route[1] for route in STREAM_ROUTES])
@pytest.mark.parametrize("count", [1, 3, 4, 5])
def test_structured_stream_preserves_raw_memory_and_keyword_controls(route: tuple, count: int) -> None:
    sample = torch.full((1, 3, count, 3, 4), 0.5)
    native, requests, _ = _native(sample)
    pipeline = _adapter(route, native)
    image = _inputs(route).get("images")
    result = pipeline.stream("scene", image, return_dict=True, fps=12, num_frames=17)
    assert result["video"] is result["video_chunk"].frames is sample
    assert result["fps"] == requests[0].inputs["fps"] == 12 and requests[0].num_frames == 17
    assert pipeline.stream_state_ready() is False
    assert pipeline.memory_module.select(prefer_type="video") is sample
    np.testing.assert_array_equal(np.asarray(pipeline.memory_module.select()), _expected(sample, route[3])[-1])
    assert pipeline.stream("scene", image) is sample
    assert len(pipeline.memory_module.storage) == 2
    names = inspect.signature(pipeline.stream).parameters
    for name in ("return_dict", "fps", "num_frames"):
        assert names[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
def test_latents_retain_raw_identity_without_rgb_export_or_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: tuple
) -> None:
    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    native, requests, latents = _native(sample)
    module = importlib.import_module(f"worldfoundry.pipelines.{route[0]}")
    monkeypatch.setattr(module, "save_image_or_video_tensor", lambda *args, **kwargs: pytest.fail("latent RGB export"))
    pipeline = _adapter(route, native)
    memory = pipeline.memory_module
    if memory is not None:
        memory.record(Image.new("RGB", (4, 3), "blue"))
    before = list(memory.storage) if memory is not None else []
    method = pipeline.stream if route in STREAM_ROUTES else pipeline
    result = method(
        "scene", return_dict=True, output_type="latent", output_path=tmp_path / "latent.mp4", **_inputs(route)
    )
    assert result["video_chunk"] is None and result["artifact_path"] is None
    assert result["video"] is result["latents"] is latents and requests[0].inputs["return_latent"]
    assert method("scene", output_type="latent", **_inputs(route)) is latents
    if memory is not None:
        assert memory.storage == before
    with pytest.raises(RuntimeError, match="no in-memory RGB frames"):
        realtime_frames_from_result(result)
    assert not (tmp_path / "latent.mp4").exists()


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
def test_public_loading_delegates_to_canonical_native_pipeline(monkeypatch: pytest.MonkeyPatch, route: tuple) -> None:
    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    native, requests, _ = _native(sample)
    calls: list[dict[str, Any]] = []
    module = importlib.import_module(f"worldfoundry.pipelines.{route[0]}")

    def load(model_id: str, **kwargs: Any) -> Any:
        calls.append({"model_id": model_id, **kwargs})
        return native

    monkeypatch.setattr(module.NativeDiffusionPipeline, "from_pretrained", load)
    cls = getattr(module, route[1])
    pipeline = cls.from_pretrained(None, {"torch_dtype": "float32", "offload_mode": "none"}, "cpu")
    result = pipeline("scene", return_dict=True, **_inputs(route))
    assert pipeline.native_pipeline is native and calls[0]["model_id"] == route[2]
    assert calls[0]["policy"].device == torch.device("cpu") and calls[0]["policy"].dtype is torch.float32
    assert result["video_chunk"].frames is sample and len(requests) == 1


@pytest.mark.parametrize("main", [True, False])
@pytest.mark.parametrize("output_type", ["video", "latent"])
def test_cosmos_parallel_artifact_ownership_does_not_remove_rank_local_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main: bool, output_type: str
) -> None:
    from worldfoundry.pipelines.cosmos import pipeline_cosmos_predict2p5 as module

    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    native, requests, latents = _native(sample)
    agreements: list[Any] = []
    context = SimpleNamespace(is_main=main, agree=agreements.append, broadcast_seed=lambda value: 987)
    writes: list[Any] = []
    monkeypatch.setattr(
        module, "save_image_or_video_tensor", lambda *args, **kwargs: writes.append(kwargs) or str(args[1])
    )
    pipeline = module.CosmosPredict2p5Pipeline(
        native_pipeline=native, device="cpu", model_id=ROUTES[1][2], parallel_context=context
    )
    result = pipeline("scene", output_type=output_type, output_path=str(tmp_path / "output.mp4"), return_dict=True)
    assert requests[0].sampling.seed == 987 and agreements
    assert len(writes) == int(main and output_type == "video")
    if output_type == "video":
        assert result["video_chunk"].frames is result["video"] is sample
    else:
        assert result["video_chunk"] is None and result["video"] is latents


def test_cosmos3_video_declaration_retains_independent_audio_and_actions() -> None:
    from worldfoundry.pipelines.cosmos.pipeline_cosmos3 import Cosmos3Pipeline
    from worldfoundry.synthesis.visual_generation.cosmos.cosmos3_runtime import Cosmos3Runtime

    video, sound, actions = torch.full((1, 3, 5, 3, 4), 0.5), torch.ones(1, 320), torch.ones(1, 8, 5)
    calls: list[Any] = []

    def native(request: Any) -> Any:
        calls.append(request)
        return SimpleNamespace(
            sample=video,
            latents=actions,
            metadata={},
            artifacts={"video": video, "sound": sound, "action": actions, "audio_sampling_rate": 16000},
        )

    pipeline = Cosmos3Pipeline(native_pipeline=native, device="cpu", model_id="cosmos3-nano")
    action = {"mode": "policy", "chunk_size": 5, "domain_name": "robot"}
    result = pipeline("scene", enable_sound=True, action=action, fps=23.5, return_dict=True)
    assert result["video_chunk"].frames is video and result["fps"] == 23.5
    assert result["sound"] is sound and result["action"] is actions and result["audio_sampling_rate"] == 16000
    assert action == {"mode": "policy", "chunk_size": 5, "domain_name": "robot"}
    assert calls[0].inputs["action_mode"] == "policy" and calls[0].inputs["enable_sound"]
    wrapped = Cosmos3Runtime(pipeline).predict("scene", enable_sound=True, action=action, fps=23.5)
    assert wrapped.video is video and wrapped.sound is sound and wrapped.action is actions
    assert wrapped.video_chunk is not None and wrapped.video_chunk.frames is video and wrapped.fps == 23.5


@pytest.mark.parametrize("version", ["original", "1.5"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hunyuan_codec_normalization_matches_unsigned_declaration(version: str, dtype: torch.dtype) -> None:
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
    from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.component import (
        HunyuanVideo15Codec,
        HunyuanVideoOriginalCodec,
    )

    signed = torch.tensor([-1, 0, 0.5, 1], dtype=dtype).reshape(1, 1, 1, 1, 4).expand(1, 3, 5, 3, 4).clone()
    original = signed.clone()
    calls: list[Any] = []

    class VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
            self.config = SimpleNamespace(scaling_factor=2.0)
            self.scaling_factor = 2.0

        def enable_tiling(self, value: bool) -> None:
            calls.append(value)

        def decode(self, value: torch.Tensor) -> Any:
            calls.append(value)
            return SimpleNamespace(sample=signed)

    vae: Any = VAE()
    codec = HunyuanVideoOriginalCodec(vae) if version == "original" else HunyuanVideo15Codec(vae)
    latents = torch.ones(1, 16, 2, 1, 1, dtype=dtype)
    request = DiffusionRequest(prompt="scene", height=3, width=4, num_frames=5)
    decoded = codec.decode(latents, request)
    torch.testing.assert_close(decoded, (signed.float() / 2 + 0.5).clamp(0, 1), rtol=0, atol=0)
    chunk = VideoChunk(frames=decoded, layout="bcthw", value_range="0,1")
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), _expected(decoded, "0,1"))
    assert len(calls) == 2
    torch.testing.assert_close(calls[1], latents / 2, rtol=0, atol=0)
    assert codec.decode(latents, DiffusionRequest(prompt="scene", inputs={"return_latent": True})) is latents
    assert len(calls) == 2
    torch.testing.assert_close(signed, original, rtol=0, atol=0)


def test_cosmos_codec_crops_signed_decode_without_renormalization() -> None:
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
    from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos2p5.component import Cosmos25VideoCodec

    sample = torch.full((1, 3, 9, 3, 4), 0.5)
    vae: Any = SimpleNamespace(decode=lambda *args, **kwargs: sample)
    codec = Cosmos25VideoCodec(vae, device=torch.device("cpu"), dtype=torch.float32)
    latents = torch.zeros(1, 16, 3, 1, 1)
    decoded = codec.decode(latents, DiffusionRequest(prompt="scene", num_frames=5))
    assert decoded.shape == (1, 3, 5, 3, 4) and decoded.data_ptr() == sample.data_ptr()
    np.testing.assert_array_equal(_expected(decoded, "-1,1"), np.full((5, 3, 4, 3), 191, np.uint8))
    assert codec.decode(latents, DiffusionRequest(prompt="scene", inputs={"return_latent": True})) is latents


def test_turbo_frame_decoder_retains_signed_time_order_and_latent_bypass() -> None:
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
    from worldfoundry.base_models.diffusion_model.models.autoencoders.lvdm.component import LVDMVideoDecoder

    calls: list[torch.Tensor] = []

    def decode(value: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return value[:, :3]

    model: Any = SimpleNamespace(decode=decode)
    decoder = LVDMVideoDecoder(model, scale_factor=0.5)
    latents = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625]).reshape(1, 1, 5, 1, 1).expand(1, 4, 5, 3, 4).clone()
    original = latents.clone()
    decoded = decoder.decode(latents, DiffusionRequest(prompt="scene"))
    torch.testing.assert_close(decoded, (latents[:, :3] / 0.5).clamp(-1, 1), rtol=0, atol=0)
    assert len(calls) == 5
    assert decoder.decode(latents, DiffusionRequest(prompt="scene", inputs={"return_latent": True})) is latents
    assert len(calls) == 5
    torch.testing.assert_close(latents, original, rtol=0, atol=0)
