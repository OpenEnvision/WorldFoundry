# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Lyra-2 output normalization and plans with substituted model and depth inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.execution import inference_runtime
from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.pipelines.lyra import pipeline_lyra2
from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline, LyraPipeline
from worldfoundry.studio.inference.execution import _normalize_frame_list
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.lyra_2.runtime import Lyra2Runtime
from worldfoundry.synthesis.visual_generation.lyra_2.synthesis import Lyra2Synthesis

pytestmark = pytest.mark.unit


def _runtime(count: int, monkeypatch: pytest.MonkeyPatch) -> tuple[Lyra2Runtime, torch.Tensor, list[dict[str, Any]]]:
    sample = torch.full((1, 3, count, 3, 4), 0.5)
    calls: list[dict[str, Any]] = []
    model = SimpleNamespace(tensor_kwargs={"device": "cpu", "dtype": torch.float32})
    runtime = Lyra2Runtime(
        device="cpu",
        weight_dtype=torch.float32,
        model=model,
        da3_model=object(),
        negative_prompt_data={"embeddings": torch.zeros(1, 2, 3)},
    )
    monkeypatch.setattr(runtime, "_install_runtime_aliases", lambda: None)
    monkeypatch.setattr(runtime, "_build_prompt_batch", lambda **kwargs: {"t5_text_embeddings": torch.zeros(1, 2, 3)})
    monkeypatch.setattr(inference_runtime, "set_random_seed", lambda **kwargs: None)
    sample_module = ModuleType(
        "worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.lyra2_ar_inference"
    )
    depth_module = ModuleType(
        "worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.lyra2_zoomgs_inference"
    )

    def run(model: Any, batch: dict[str, Any], args: Any, **kwargs: Any) -> dict[str, Any]:
        assert batch["video"].shape == (1, 3, 1, 3, 4)
        assert batch["camera_w2c"].shape == (1, 81, 4, 4) and args.num_frames == 81
        calls.append(batch)
        return {"video": sample, "warp_video": sample}

    def depth(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        assert kwargs["target_hw"] == (3, 4) and kwargs["img_rgb_uint8"].shape == (3, 4, 3)
        return torch.zeros(1, 3, 3, 4), torch.ones(3, 4), torch.eye(3), None

    monkeypatch.setattr(sample_module, "run_lyra2_sample", run, raising=False)
    monkeypatch.setattr(sample_module, "safe_to", lambda batch, **kwargs: batch, raising=False)
    monkeypatch.setattr(depth_module, "_da3_infer_depth_intrinsics_single", depth, raising=False)
    monkeypatch.setitem(sys.modules, sample_module.__name__, sample_module)
    monkeypatch.setitem(sys.modules, depth_module.__name__, depth_module)
    return runtime, sample, calls


def _assert_pixels(result: dict[str, Any], count: int) -> None:
    pixels = np.full((count, 3, 4, 3), 191, np.uint8)
    chunk = result["video_chunk"]
    assert chunk.frames is result["video"] and chunk.layout == "thwc" and chunk.value_range == "0,255"
    assert chunk.frame_count == count and result["fps"] == chunk.metadata["fps"] == 12
    np.testing.assert_array_equal(result["video"], pixels)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), pixels)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), pixels)
    preview = _normalize_frame_list(chunk)
    assert preview is not None
    np.testing.assert_array_equal(np.stack([np.asarray(frame) for frame in preview]), pixels)


@pytest.mark.parametrize("count", [1, 3, 4, 5])
def test_runtime_declares_existing_byte_video_and_keeps_camera_tensor_and_warp(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sample, calls = _runtime(count, monkeypatch)
    original = sample.clone()
    cameras = torch.eye(4).repeat(81, 1, 1)
    zoom = torch.linspace(1, 2, 81)
    result = runtime.predict(
        input_image=Image.new("RGB", (4, 3)),
        camera_w2c=cameras,
        zoom_factors=zoom,
        execute=True,
        resolution=(3, 4),
        fps=12,
    )
    _assert_pixels(result, count)
    assert result["video_tensor"] is result["warp_video"] is sample
    torch.testing.assert_close(result["camera_w2c"], cameras, rtol=0, atol=0)
    torch.testing.assert_close(result["zoom_factors"], zoom, rtol=0, atol=0)
    torch.testing.assert_close(result["intrinsics"][:, 0, 0], zoom, rtol=0, atol=0)
    torch.testing.assert_close(result["intrinsics"][:, 1, 1], zoom, rtol=0, atol=0)
    assert calls[0]["fps"].item() == 12
    torch.testing.assert_close(sample, original, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("reconstruct", [False, True])
def test_pipeline_preserves_raw_video_camera_reconstruction_and_frame_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, route: str, reconstruct: bool
) -> None:
    runtime, sample, _ = _runtime(count, monkeypatch)
    representation_calls: list[dict[str, Any]] = []
    geometry = {"points": np.zeros((2, 3)), "camera": np.eye(4)}
    writes: list[np.ndarray] = []

    def representation(value: dict[str, Any]) -> dict[str, Any]:
        representation_calls.append(value)
        return geometry

    def write(video: np.ndarray, path: str, *, fps: int) -> None:
        assert fps == 12
        writes.append(video)
        Path(path).write_bytes(b"recorded lyra2")

    monkeypatch.setattr(pipeline_lyra2.Lyra2Synthesis, "from_pretrained", lambda **kwargs: Lyra2Synthesis(runtime))
    monkeypatch.setattr(
        pipeline_lyra2.Lyra2Representation,
        "from_pretrained",
        lambda **kwargs: SimpleNamespace(get_representation=representation),
    )
    monkeypatch.setattr(pipeline_lyra2, "save_video_frames", write)
    pipeline = Lyra2Pipeline.from_pretrained(None, {}, "cpu", torch.float32)
    method = pipeline if route == "batch" else pipeline.stream
    kwargs: dict[str, Any] = {
        "images": Image.new("RGB", (4, 3)),
        "interactions": ["camera_zoom_in"],
        "fps": 12,
        "resolution": (3, 4),
        "output_dir": str(tmp_path),
        "reconstruct_3d": reconstruct,
    }
    result = method(return_dict=True, **kwargs)
    _assert_pixels(result, count)
    assert result["video_tensor"] is sample and writes[0] is result["video"]
    assert result["camera_w2c"].shape == (81, 4, 4) and result["zoom_factors"].shape == (81,)
    assert result["actions"] == ["camera_zoom_in"]
    assert result["reconstruction"] is (geometry if reconstruct else None)
    assert Path(result["generated_video_path"]).read_bytes() == b"recorded lyra2"
    if reconstruct:
        assert representation_calls[0]["video_tensor"] is sample
        assert representation_calls[0]["video_path"] == result["generated_video_path"]
        assert representation_calls[0]["video"] is None
    if route == "stream":
        assert pipeline.memory_module.current_video is result["video"]
        assert len(pipeline.memory_module.all_frames) == count
        assert pipeline.memory_module.current_image is not None
        assert pipeline.memory_module.current_image.getpixel((0, 0)) == (191, 191, 191)
        kwargs["images"] = None
    raw = method(**kwargs)
    assert raw["reconstruction"] is geometry if reconstruct else isinstance(raw, np.ndarray)


@pytest.mark.parametrize("pipeline_cls", [Lyra2Pipeline, LyraPipeline])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("return_dict", [False, True])
def test_blocked_plan_retains_camera_fields_and_does_not_export_or_reconstruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline_cls: type, route: str, return_dict: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = Lyra2Runtime(device="cpu")
    pipeline = pipeline_cls(synthesis_model=Lyra2Synthesis(runtime), device="cpu")
    camera_path = {
        "keyframes": [
            {"position": [0, 0, 0], "target": [0, 0, 1], "fov": 60},
            {"position": [0, 0, 1], "target": [0, 0, 2], "fov": 55},
        ]
    }
    method = pipeline if route == "batch" else pipeline.stream
    kwargs = {
        "images": Image.new("RGB", (4, 3)),
        "camera_path": camera_path,
        "output_dir": str(tmp_path / "plans"),
        "execute": False,
        "reconstruct_3d": True,
        "fps": 12,
    }
    result = method(return_dict=return_dict, **kwargs)
    assert result["status"] == "blocked" and result["camera_w2c"].shape == (81, 4, 4)
    assert result["actions"] == ["camera_path"] and result["fps"] == 12
    assert "video_chunk" not in result and "video" not in result and "reconstruction" not in result
    plan = Path(result["artifact_path"])
    assert plan == tmp_path / "plans" / "lyra2_plan.json"
    assert json.loads(plan.read_text())["camera_w2c_shape"] == [81, 4, 4]
    assert not (tmp_path / "plans" / "generated.mp4").exists()
    if route == "stream":
        assert pipeline.memory_module.latest_result is None and len(pipeline.memory_module.storage) == 1
        assert pipeline.memory_module.current_image.getpixel((0, 0)) == (0, 0, 0)


def test_camera_path_only_stream_declares_output_without_fabricating_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _ = _runtime(5, monkeypatch)
    pipeline = Lyra2Pipeline(synthesis_model=Lyra2Synthesis(runtime), device="cpu")
    camera_path = {
        "keyframes": [{"position": [0, 0, 0], "target": [0, 0, 1]}, {"position": [0, 0, 1], "target": [0, 0, 2]}]
    }
    result = pipeline.stream(
        images=Image.new("RGB", (4, 3)), camera_path=camera_path, resolution=(3, 4), return_dict=True, fps=12
    )
    _assert_pixels(result, 5)
    assert result["actions"] == ["camera_path"]
    assert pipeline.memory_module.storage[-1]["metadata"]["interactions"] == []


def test_runtime_rejects_short_camera_path_before_depth_or_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _, calls = _runtime(5, monkeypatch)
    with pytest.raises(ValueError, match="1 \\+ 80k"):
        runtime.predict(
            input_image=Image.new("RGB", (4, 3)),
            camera_w2c=torch.eye(4).repeat(5, 1, 1),
            execute=True,
            resolution=(3, 4),
        )
    assert calls == []
