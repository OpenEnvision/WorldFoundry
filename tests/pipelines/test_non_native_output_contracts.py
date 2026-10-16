"""Weight-free contracts for model-owned video adapters and artifact runtimes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.pipelines.abot_world.pipeline_abot_world import ABotWorldPipeline
from worldfoundry.pipelines.video_official.pipeline_official_video import OfficialVideoPipeline
from worldfoundry.studio.inference.execution import _normalize_frame_list, _safe_json
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis

pytestmark = pytest.mark.unit


def _runtime_synthesis(frames: Any) -> RuntimeVideoSynthesis:
    class Runtime:
        def generate_video(self, *, prompt: str, image_path: str | None) -> Any:
            assert prompt == "scene"
            assert image_path is None
            return frames

    return RuntimeVideoSynthesis(
        model_name="video-contract", generation_type="t2v", runtime_cls=Runtime, runtime_kwargs={}
    )


def _assert_rgb_result(result: dict[str, Any], expected: np.ndarray, alias: str = "video") -> None:
    chunk = result["video_chunk"]
    assert chunk.frames is result[alias]
    assert chunk.frame_count == len(expected)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), expected)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), expected)
    materialized = _normalize_frame_list(chunk)
    assert materialized is not None
    np.testing.assert_array_equal(np.stack(materialized), expected)
    assert _safe_json(result)["video_chunk"] == {
        "frames": {"shape": list(chunk.frames.shape), "dtype": str(chunk.frames.dtype)},
        "layout": chunk.layout,
        "value_range": chunk.value_range,
        "frame_count": len(expected),
        "metadata": {},
    }


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("source_kind", ["numpy", "tensor", "pil", "declared"])
def test_shared_runtime_declares_rgb_without_changing_raw_returns(count: int, source_kind: str) -> None:
    expected = np.arange(count * 3 * 4 * 3, dtype=np.uint8).reshape(count, 3, 4, 3)
    source: Any = expected.copy()
    if source_kind == "tensor":
        source = torch.from_numpy(source).permute(0, 3, 1, 2)
    elif source_kind == "pil":
        source = [Image.fromarray(frame) for frame in source]
    elif source_kind == "declared":
        source = VideoChunk(frames=torch.from_numpy(source), layout="thwc", value_range="0,255")
    synthesis = _runtime_synthesis(source)

    result = synthesis.predict("scene", return_dict=True)
    assert result["video_chunk"].layout == "thwc"
    assert result["video_chunk"].value_range == "0,255"
    assert result["generated_video_path"] is None
    _assert_rgb_result(result, expected)
    np.testing.assert_array_equal(synthesis.predict("scene"), expected)
    if source_kind == "numpy":
        assert result["video"] is source
        np.testing.assert_array_equal(source, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("value_range", "pixel"), [("-1,1", 191), ("0,1", 127), ("0,255", 0)])
def test_shared_runtime_honors_declared_positive_only_pixels(dtype: torch.dtype, value_range: str, pixel: int) -> None:
    source = torch.full((1, 3, 5, 3, 4), 0.5, dtype=dtype)
    synthesis = _runtime_synthesis(VideoChunk(frames=source, layout="bcthw", value_range=value_range))
    result = synthesis.predict("scene", return_dict=True)
    _assert_rgb_result(result, np.full((5, 3, 4, 3), pixel, dtype=np.uint8))
    torch.testing.assert_close(source, torch.full_like(source, 0.5), rtol=0, atol=0)


def test_shared_runtime_preserves_chunk_metadata_during_normalization() -> None:
    source = VideoChunk(
        frames=np.full((3, 3, 4, 3), 0.5),
        layout="thwc",
        value_range="-1,1",
        metadata={"frame_index": 12, "fps": 16},
    )
    result = _runtime_synthesis(source).predict("scene", return_dict=True)
    chunk = result["video_chunk"]
    assert chunk is not source
    assert chunk.frames is result["video"]
    assert chunk.metadata == {"frame_index": 12, "fps": 16}
    assert chunk.value_range == "0,255"
    assert source.value_range == "-1,1"
    np.testing.assert_array_equal(source.frames, np.full((3, 3, 4, 3), 0.5))
    np.testing.assert_array_equal(chunk.frames, np.full((3, 3, 4, 3), 191, dtype=np.uint8))


@pytest.mark.parametrize("channels", [1, 4])
def test_shared_runtime_keeps_rgb_conversion_for_declared_frames(channels: int) -> None:
    source = np.full((3, 3, 4, channels), 0.5, dtype=np.float32)
    synthesis = _runtime_synthesis(VideoChunk(frames=source, layout="thwc", value_range="-1,1"))
    result = synthesis.predict("scene", return_dict=True)
    _assert_rgb_result(result, np.full((3, 3, 4, 3), 191, dtype=np.uint8))
    np.testing.assert_array_equal(source, np.full_like(source, 0.5))


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_shared_runtime_rejects_nonfinite_pixels(declared: bool, value: float) -> None:
    source: Any = torch.full((3, 3, 3, 4), value)
    if declared:
        source = VideoChunk(frames=source, layout="tchw", value_range="-1,1")
    with pytest.raises(ValueError, match="NaN or infinite"):
        _runtime_synthesis(source).predict("scene", return_dict=True)


def test_shared_runtime_does_not_guess_an_invalid_declared_batch() -> None:
    source = VideoChunk(frames=np.zeros((2, 3, 5, 3, 4)), layout="bcthw", value_range="-1,1")
    with pytest.raises(ValueError, match="batch size 1"):
        _runtime_synthesis(source).predict("scene", return_dict=True)


def test_shared_runtime_empty_chunk_remains_buffering() -> None:
    source = VideoChunk(frames=np.empty((0, 3, 4, 3), dtype=np.uint8), layout="thwc", value_range="0,255")
    result = _runtime_synthesis(source).predict("scene", return_dict=True)
    assert result["video_chunk"].frame_count == 0
    assert result["generated_video_path"] is None
    assert realtime_frames_from_result(result) == []


def test_shared_runtime_encoder_and_transport_receive_identical_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import imageio

    encoded: list[np.ndarray] = []
    output = tmp_path / "video.mp4"

    def save(path: str, frames: np.ndarray, *, fps: int) -> None:
        assert path == str(output)
        assert fps == 12
        encoded.append(frames)
        Path(path).write_bytes(b"encoded-artifact")

    monkeypatch.setattr(imageio, "mimsave", save)
    source = VideoChunk(frames=torch.full((5, 3, 3, 4), 0.5), layout="tchw", value_range="-1,1")
    result = _runtime_synthesis(source).predict("scene", output_path=str(output), fps=12, return_dict=True)
    assert result["generated_video_path"] == str(output)
    assert encoded[0] is result["video"]
    _assert_rgb_result(result, np.full((5, 3, 4, 3), 191, dtype=np.uint8))


@pytest.mark.parametrize("count", [1, 3, 4, 5])
def test_abot_batch_and_realtime_preserve_unit_tchw(count: int) -> None:
    video = torch.arange(count * 3 * 3 * 4, dtype=torch.float32).reshape(count, 3, 3, 4) / 256
    original = video.clone()

    class Runtime:
        def generate(self, images: Any, **kwargs: Any) -> torch.Tensor:
            assert images == "source"
            return video

        def step(self, interactions: Any) -> torch.Tensor:
            assert interactions == ["forward"]
            return video

    pipeline = ABotWorldPipeline(Runtime(), device="cpu")
    expected = (video.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()
    for result, alias in (
        (pipeline(images="source", return_dict=True), "video"),
        (pipeline.stream_realtime(interactions=["forward"]), "frames"),
    ):
        assert result["video_chunk"].layout == "tchw"
        assert result["video_chunk"].value_range == "0,1"
        _assert_rgb_result(result, expected, alias)
        assert result[alias] is video
    assert pipeline(images="source") is video
    torch.testing.assert_close(video, original, rtol=0, atol=0)


def test_abot_artifact_export_uses_the_declared_time_axis(tmp_path: Path) -> None:
    video = torch.tensor([0.25, 0.5, 0.75]).view(1, 3, 1, 1).expand(1, 3, 3, 4)

    class Runtime:
        def generate(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            return video

    pipeline = ABotWorldPipeline(Runtime(), device="cpu")
    result = pipeline(images="source", output_path=tmp_path / "frame.png", return_dict=True)
    with Image.open(result["artifact_path"]) as image:
        assert image.size == (4, 3)
        np.testing.assert_array_equal(np.asarray(image), video_chunk_to_uint8_frames(result["video_chunk"])[0])
    assert result["video"] is video


def test_artifact_loader_requires_a_model_identity() -> None:
    with pytest.raises(ValueError, match="requires model_id/profile_id"):
        OfficialVideoPipeline.from_pretrained(device="cpu")


@pytest.mark.parametrize("return_dict", [False, True])
def test_artifact_runtime_stream_preserves_paths_and_metadata(tmp_path: Path, return_dict: bool) -> None:
    from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory

    payload = {"status": "ok", "artifact_path": str(tmp_path / "video.mp4"), "metadata": {"fps": 12}}

    class Runtime:
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            return payload

    pipeline = OfficialVideoPipeline(model_id="artifact-contract", runtime=Runtime(), device="cpu")  # ty: ignore[invalid-argument-type]
    result = pipeline.stream("scene", output_path=tmp_path / "video.mp4", return_dict=return_dict)
    assert result is payload if return_dict else result == payload["artifact_path"]
    assert "video_chunk" not in payload
    assert pipeline.stream_state_ready() is False
    assert isinstance(pipeline.memory_module, VideoArtifactMemory)
    assert len(pipeline.memory_module.storage) == 1


@pytest.mark.parametrize("route", ["realtime", "worker"])
def test_artifact_history_does_not_replace_the_next_explicit_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    from worldfoundry.studio.inference.catalog import CatalogEntry
    from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

    received: list[tuple[int, ...]] = []
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("WORLDFOUNDRY_ARTIFACT_DIR", str(artifact_root))
    artifact_path = artifact_root / "pipeline_eval" / "artifact-contract.mp4"

    class Runtime:
        def generate(self, *, image_path: str, output_path: Path, **kwargs: Any) -> dict[str, Any]:
            with Image.open(image_path) as image:
                received.append(tuple(np.asarray(image)[0, 0].tolist()))
            return {"status": "ok", "artifact_path": str(output_path)}

    class Pipeline(OfficialVideoPipeline):
        GENERATION_TYPE = "i2v"

    pipeline = Pipeline(model_id="artifact-contract", runtime=Runtime(), device="cpu")  # ty: ignore[invalid-argument-type]
    pipeline.memory_module.record(Image.new("RGB", (4, 3), "blue"))
    entry = CatalogEntry(
        model_id="artifact-contract",
        display_name="Artifact runtime",
        module_path=__name__,
        class_name="Pipeline",
        family="video",
        category="Image-to-Video",
        summary="",
        supports_stream=True,
    )
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key="artifact",
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    manager.pipeline_cache[ctx.cache_key] = ctx
    driver = manager.runtime_driver_for(entry)
    monkeypatch.setattr(driver, "load_pipeline", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(manager, "_torchrun_worker_request", lambda entry, request: request)
    first, second = Image.new("RGB", (4, 3)), Image.new("RGB", (4, 3), "red")
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=first,
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
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    assert not driver.can_init(ctx, request)
    for image in (first, second, first):
        current = replace(request, image=image)
        if route == "realtime":
            result = manager.run_realtime(entry=entry, request=current, action="stream")
        else:
            result = manager._run_local_torchrun_lingbot_fast(
                entry=entry, request=current, action="stream", materialize=False
            )
        assert result == {"status": "ok", "artifact_path": str(artifact_path)}
        assert not ctx.state
        assert ctx.active_leases == 0
    assert received == [(0, 0, 0), (255, 0, 0), (0, 0, 0)]
