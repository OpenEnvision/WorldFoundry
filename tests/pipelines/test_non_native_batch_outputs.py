# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Batch/continuation pixel declarations with substituted generation and real media operations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import (
    IdentityVideoPostProcessor,
    VideoChunk,
    VideoPostprocessChain,
    infer_video_spec,
)
from worldfoundry.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
from worldfoundry.pipelines.longvie.pipeline_longvie import LongVie1Pipeline, LongVie2Pipeline
from worldfoundry.pipelines.worldfm.pipeline_worldfm import WorldFMPipeline
from worldfoundry.studio.inference.execution import _normalize_frame_list
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_synthesis import LingBotSynthesis
from worldfoundry.synthesis.visual_generation.lingbot_world.runtime import LingBotWorldRuntime
from worldfoundry.synthesis.visual_generation.worldfm.runtime import WorldFMRuntime
from worldfoundry.synthesis.visual_generation.worldfm.worldfm_synthesis import WorldFMSynthesis

pytestmark = pytest.mark.unit


def _assert_pixels(result: dict[str, Any], expected: np.ndarray, *, layout: str, value_range: str) -> None:
    chunk: VideoChunk = result["video_chunk"]
    assert chunk.frames is result["video"]
    assert (chunk.layout, chunk.value_range, chunk.frame_count) == (layout, value_range, len(expected))
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)
    materialized = _normalize_frame_list(chunk)
    assert materialized is not None
    np.testing.assert_array_equal(np.stack(materialized), expected)
    spec = infer_video_spec(chunk.frames, layout=chunk.layout, fps=16)
    assert (spec.height, spec.width, spec.channels) == expected.shape[1:]
    session = VideoPostprocessChain((IdentityVideoPostProcessor(),)).start(spec)
    [output] = session.process(chunk)
    assert output is chunk
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(output), expected)
    assert session.flush() == []


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("step", [2, 4])
@pytest.mark.parametrize("route", ["batch", "stream"])
def test_worldfm_declares_frames_without_changing_decoder_or_scene(count: int, step: int, route: str) -> None:
    decoded = torch.linspace(-1, 1, count * 3 * 3 * 4).reshape(count, 3, 3, 4)
    original = decoded.clone()
    conditions = [np.full((3, 4, 3), index, dtype=np.uint8) for index in range(count)]

    class Service:
        device = "cpu"

        def set_cond2_from_array(self, condition: np.ndarray) -> None:
            assert condition is conditions[self.calls % count]

        def infer_from_render_u8(self, render: torch.Tensor) -> torch.Tensor:
            assert step == 2
            return self._decode(render)

        def infer_from_render_u8_multistep(self, render: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            assert step == 4 and kwargs == {"sample_steps": 4, "cfg_scale": 4.5}
            return self._decode(render)

        def _decode(self, render: torch.Tensor) -> torch.Tensor:
            index = self.calls % count
            assert render.dtype == torch.uint8
            self.calls += 1
            return decoded[index : index + 1]

        calls = 0

    runtime = WorldFMRuntime(Service(), "checkpoint", "vae", step=step, device="cpu")
    frame_conditions = [{"render_rgb_u8": frame, "cond_nearest_rgb": frame} for frame in conditions]
    scene = {"scene_name": "A", "K": np.eye(3)}

    class Representation:
        def get_representation(self, inputs: dict[str, Any]) -> dict[str, Any]:
            assert inputs["scene_context"] is scene
            return {**inputs, "scene_name": "A", "K": scene["K"], "rendered_conditions": frame_conditions}

    pipeline = WorldFMPipeline(
        representation_model=cast(Any, Representation()), synthesis_model=WorldFMSynthesis(runtime), device="cpu"
    )
    pipeline.memory_module.record({"scene_context": scene})
    poses = [np.eye(4) for _ in range(count)]
    kwargs: dict[str, Any] = {"interactions": poses, "return_dict": True}
    result = pipeline(scene_context=scene, **kwargs) if route == "batch" else pipeline.stream(**kwargs)
    expected = (127.5 * decoded + 128).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
    assert result["frames"] is result["video"]
    assert result["scene_context"] is scene
    assert result["K"] is scene["K"]
    assert pipeline.memory_module.select() is scene
    np.testing.assert_array_equal(result["c2w_list"], poses)
    _assert_pixels(result, expected, layout="frame-list", value_range="0,255")
    raw = runtime.predict(frame_conditions, return_dict=False)
    assert isinstance(raw, list)
    np.testing.assert_array_equal(np.stack(raw), expected)
    assert runtime.service.calls == 2 * count
    torch.testing.assert_close(decoded, original, rtol=0, atol=0)


def test_worldfm_saved_pngs_match_declared_frames(tmp_path: Path) -> None:
    class Service:
        device = "cpu"

        def set_cond2_from_array(self, condition: np.ndarray) -> None:
            pass

        def infer_from_render_u8(self, render: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 3, 3, 4)

    runtime = WorldFMRuntime(Service(), "checkpoint", "vae", device="cpu")
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    result = runtime.predict(
        [{"render_rgb_u8": frame, "cond_nearest_rgb": frame}] * 3,
        output_dir=tmp_path,
        scene_name="A",
        save_mode="image",
    )
    assert isinstance(result, dict)
    expected = np.full((3, 3, 4, 3), 128, dtype=np.uint8)
    _assert_pixels(result, expected, layout="frame-list", value_range="0,255")
    assert result["generated_video_path"] is None
    for path, expected_frame in zip(result["generated_image_paths"], expected, strict=True):
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), expected_frame)


def _lingbot(monkeypatch: pytest.MonkeyPatch, count: int, model_id: str = "lingbot-world") -> tuple[Any, torch.Tensor]:
    decoded = torch.linspace(-1, 1, count * 3 * 3 * 4).reshape(3, count, 3, 4)

    class Core:
        config = SimpleNamespace(sample_shift=5, sample_steps=4, sample_guide_scale=1)

        def generate(
            self, *, input_prompt: str, img: Image.Image, frame_num: int, seed: int, offload_model: bool
        ) -> torch.Tensor:
            assert input_prompt == "scene" and frame_num == count and seed == 7
            return decoded

    synthesis = LingBotSynthesis(LingBotWorldRuntime(Core()))
    monkeypatch.setattr(LingBotSynthesis, "from_pretrained", lambda **kwargs: synthesis)
    pipeline = LingBotPipeline.from_pretrained("checkpoint", model_id=model_id, device="cpu")
    return pipeline, decoded


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("model_id", ["lingbot-world", "lingbot-world-act"])
@pytest.mark.parametrize("route", ["batch", "stream"])
def test_lingbot_batch_and_continuation_preserve_float_returns(
    monkeypatch: pytest.MonkeyPatch, count: int, model_id: str, route: str
) -> None:
    pipeline, decoded = _lingbot(monkeypatch, count, model_id)
    original = decoded.clone()
    image = Image.new("RGB", (4, 3))
    method = pipeline if route == "batch" else pipeline.stream
    kwargs = {"images": image, "prompt": "scene", "num_frames": count, "seed": 7}
    result = method(return_dict=True, **kwargs)
    expected_float = ((decoded.permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1).numpy()
    expected = (expected_float * 255).astype(np.uint8)
    assert result["model_id"] == model_id
    np.testing.assert_array_equal(result["video"], expected_float)
    _assert_pixels(result, expected, layout="thwc", value_range="0,1")
    if route == "stream":
        assert pipeline.memory_module.current_video is result["video"]
        assert len(pipeline.memory_module.all_frames) == count
        np.testing.assert_array_equal(np.asarray(pipeline.memory_module.select()), expected[-1])
        continued = pipeline.stream(prompt="scene", num_frames=count, seed=7, return_dict=True)
        _assert_pixels(continued, expected, layout="thwc", value_range="0,1")
    raw = method(**kwargs)
    assert isinstance(raw, np.ndarray)
    np.testing.assert_array_equal(raw, expected_float)
    torch.testing.assert_close(decoded, original, rtol=0, atol=0)


def test_lingbot_constructor_owns_default_operator_and_frame_memory() -> None:
    from worldfoundry.operators.lingbot_world_operator import LingBotOperator
    from worldfoundry.synthesis.visual_generation.memory.stream import VisualFrameMemory

    pipeline = LingBotPipeline(device="cpu")
    assert isinstance(pipeline.operators, LingBotOperator)
    assert isinstance(pipeline.memory_module, VisualFrameMemory)
    assert pipeline.memory_module.select() is None
    with pytest.raises(RuntimeError, match="synthesis model is not loaded"):
        pipeline(images=Image.new("RGB", (4, 3)))


def test_lingbot_export_matches_transport_pixels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.io as io

    pipeline, decoded = _lingbot(monkeypatch, 5)
    encoded: list[np.ndarray] = []
    target = tmp_path / "video.mp4"

    def save(frames: np.ndarray, output: Path, *, fps: int) -> None:
        assert output == target and fps == 12
        assert frames.dtype == np.uint8
        encoded.append(frames)
        output.write_bytes(b"encoded-video")

    monkeypatch.setattr(io, "save_video_frames", save)
    result = pipeline(
        images=Image.new("RGB", (4, 3)),
        prompt="scene",
        num_frames=5,
        seed=7,
        output_path=target,
        fps=12,
        return_dict=True,
    )
    expected = (((decoded.permute(1, 2, 3, 0) + 1) / 2) * 255).to(torch.uint8).numpy()
    _assert_pixels(result, expected, layout="thwc", value_range="0,1")
    np.testing.assert_array_equal(encoded[0], expected)
    assert result["artifact_path"] == result["generated_video_path"] == str(target)


class _LongVieRuntime:
    use_usp = False
    ring_degree = ulysses_degree = 1
    loaded = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outputs: list[list[Image.Image]] = []

    def generate_segment(self, **kwargs: Any) -> tuple[list[Image.Image], Any]:
        self.calls.append(kwargs)
        frames = [Image.new("RGB", (4, 3), (len(self.calls), index, 128)) for index in range(kwargs["num_frames"])]
        self.outputs.append(frames)
        return frames, {"segment": len(self.calls)}

    def is_output_rank(self) -> bool:
        return True

    def save_video(self, *args: Any, **kwargs: Any) -> Path:
        from worldfoundry.synthesis.visual_generation.longvie.worldfoundry_runtime import LongVieOfficialRuntime

        return LongVieOfficialRuntime.save_video(*args, **kwargs)


@pytest.mark.parametrize("pipeline_cls", [LongVie1Pipeline, LongVie2Pipeline])
@pytest.mark.parametrize("count", [5, 9, 81])
def test_longvie_declares_pil_frames_and_preserves_continuation(
    monkeypatch: pytest.MonkeyPatch, pipeline_cls: type[Any], count: int
) -> None:
    from worldfoundry.synthesis.visual_generation.longvie import longvie_synthesis as module

    runtime = _LongVieRuntime()
    monkeypatch.setattr(module, "LongVieOfficialRuntime", lambda **kwargs: runtime)
    monkeypatch.setenv("WORLD_SIZE", "1")
    pipeline = pipeline_cls.from_pretrained("checkpoint", device="cpu", execute_by_default=True)
    controls = np.zeros((count, 3, 4, 3), dtype=np.uint8)
    kwargs = {
        "prompt": "scene",
        "dense_video": controls,
        "sparse_video": controls,
        "target_size": (4, 3),
        "num_frames": count,
    }
    first = pipeline(images=Image.new("RGB", (4, 3)), return_dict=True, **kwargs)
    second = pipeline.stream(return_dict=True, **kwargs)
    for index, result in enumerate((first, second)):
        assert result["video"] is runtime.outputs[index]
        expected = np.stack([np.asarray(frame) for frame in result["video"]])
        _assert_pixels(result, expected, layout="frame-list", value_range="0,255")
        assert infer_video_spec(result["video_chunk"].frames, layout="frame-list").dtype == "uint8"
        assert result["segment_spec"]["realtime"] is False
        assert result["artifact_path"] is None
    assert runtime.calls[1]["history"] == first["video"][-8:]
    assert runtime.calls[1]["noise"] is first["noise"]
    np.testing.assert_array_equal(np.asarray(runtime.calls[1]["input_image"]), np.asarray(first["last_frame"]))
    assert pipeline.synthesis_model.history == second["video"][-8:]
    assert pipeline.synthesis_model.last_frame is second["last_frame"]
    raw = pipeline.stream(**kwargs)
    assert raw is runtime.outputs[2]
    positional_kwargs = dict(kwargs)
    positional_kwargs.pop("prompt")
    positional = pipeline.stream("scene", None, None, None, None, 12, True, None, None, **positional_kwargs)
    assert positional["video"] is runtime.outputs[3]
    assert positional["metadata"]["fps"] == 12


def test_longvie_encoder_receives_the_declared_pil_pixels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import imageio

    from worldfoundry.synthesis.visual_generation.longvie import worldfoundry_runtime as module
    from worldfoundry.synthesis.visual_generation.longvie.longvie_synthesis import LongVieSynthesis

    encoded: list[np.ndarray] = []
    closed: list[bool] = []

    class Writer:
        def append_data(self, frame: np.ndarray) -> None:
            encoded.append(frame.copy())

        def close(self) -> None:
            closed.append(True)

    target = tmp_path / "video.mp4"

    def writer(path: str, **kwargs: Any) -> Writer:
        assert path == str(target)
        assert kwargs == {"fps": 12, "quality": 10, "ffmpeg_params": None}
        target.write_bytes(b"encoded-video")
        return Writer()

    monkeypatch.setattr(module, "ensure_longvie_runtime", lambda: None)
    monkeypatch.setattr(imageio, "get_writer", writer)
    runtime = _LongVieRuntime()
    synthesis = LongVieSynthesis(runtime=cast(Any, runtime), execute_by_default=True)
    controls = np.zeros((5, 3, 4, 3), dtype=np.uint8)
    result = synthesis.predict(
        prompt="scene",
        images=Image.new("RGB", (4, 3)),
        dense_video=controls,
        sparse_video=controls,
        target_size=(4, 3),
        num_frames=5,
        output_path=target,
        fps=12,
        return_dict=True,
    )
    expected = np.stack([np.asarray(frame) for frame in runtime.outputs[0]])
    _assert_pixels(result, expected, layout="frame-list", value_range="0,255")
    np.testing.assert_array_equal(np.stack(encoded), expected)
    assert closed == [True]
    assert result["artifact_path"] == str(target)


def test_pil_frame_list_rejects_changed_geometry() -> None:
    frames = [Image.new("RGB", (4, 3)), Image.new("RGB", (4, 4))]
    with pytest.raises(ValueError, match="frame-list shape changed at index 1"):
        infer_video_spec(frames, layout="frame-list")


def test_lingbot_none_frame_count_selects_the_public_default(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, _ = _lingbot(monkeypatch, 81)
    result = pipeline(images=Image.new("RGB", (4, 3)), prompt="scene", seed=7, num_frames=None, return_dict=True)
    assert result["video_chunk"].frame_count == 81


@pytest.mark.parametrize("family", ["lingbot", "longvie-1", "longvie-2"])
def test_studio_requests_declared_continuation_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    from worldfoundry.studio.inference.catalog import CatalogEntry
    from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager
    from worldfoundry.synthesis.visual_generation.longvie.longvie_synthesis import LongVieSynthesis

    controls = np.zeros((5, 3, 4, 3), dtype=np.uint8)
    call_kwargs: dict[str, Any] = {"seed": 7}
    if family == "lingbot":
        pipeline, _ = _lingbot(monkeypatch, 5)
        pipeline.memory_module.record(Image.new("RGB", (4, 3)))
    else:
        runtime = _LongVieRuntime()
        synthesis = LongVieSynthesis(model_id=family, runtime=cast(Any, runtime), execute_by_default=True)
        pipeline_cls = LongVie1Pipeline if family == "longvie-1" else LongVie2Pipeline
        pipeline = pipeline_cls(synthesis_model=synthesis, device="cpu")
        call_kwargs.update(dense_video=controls, sparse_video=controls, target_size=(4, 3))
        pipeline(images=Image.new("RGB", (4, 3)), prompt="scene", num_frames=5, return_dict=True, **call_kwargs)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = CatalogEntry(
        model_id=family,
        display_name=family,
        module_path=__name__,
        class_name="Pipeline",
        family="video",
        category="Video Generation",
        summary="",
        supports_stream=True,
    )
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key=family,
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=None,
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=None,
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=12,
        num_frames=5,
        output_dir=str(tmp_path),
        output_path=str(tmp_path / "video.mp4"),
        call_kwargs=call_kwargs,
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    result = manager.runtime_driver_for(entry)._invoke(ctx, request, "stream", materialize_outputs=False)
    assert isinstance(result, dict)
    chunk = result["video_chunk"]
    _assert_pixels(result, video_chunk_to_uint8_frames(chunk), layout=chunk.layout, value_range=chunk.value_range)
    assert chunk.frame_count == 5
    assert result["metadata"]["fps"] == 12 if family.startswith("longvie") else result["fps"] == 12
    assert not (tmp_path / "video.mp4").exists()
