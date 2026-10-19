# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Matrix-Game batch output boundaries with CPU decoding and substituted generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline
from worldfoundry.studio.inference.execution import _normalize_frame_list, _safe_json
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime import worldfoundry_runtime as mg3
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_synthesis import MatrixGame3Synthesis

pytestmark = pytest.mark.unit


def _assert_chunk(result: dict[str, Any], pixels: np.ndarray, *, layout: str, value_range: str) -> None:
    chunk = result["video_chunk"]
    assert isinstance(chunk, VideoChunk)
    assert chunk.frames is result["video"]
    assert chunk.layout == layout and chunk.value_range == value_range
    assert chunk.frame_count == len(pixels)
    np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), pixels)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), pixels)
    normalized = _normalize_frame_list(chunk)
    assert normalized is not None
    np.testing.assert_array_equal(np.stack([np.asarray(frame) for frame in normalized]), pixels)
    descriptor = _safe_json(result)["video_chunk"]
    assert descriptor["layout"] == layout and descriptor["value_range"] == value_range
    assert descriptor["frame_count"] == len(pixels)


def _mg2(count: int) -> tuple[MatrixGame2Pipeline, torch.Tensor]:
    decoded = torch.linspace(-1, 1, count * 3 * 3 * 4).reshape(1, count, 3, 3, 4)

    class VAE:
        clip = SimpleNamespace(encode_video=lambda image: torch.zeros(1, 2, 4))

        def encode(self, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            assert images.ndim == 5 and images.shape[1] == 3
            return torch.zeros(1, 16, (images.shape[2] - 1) // 4 + 1, 2, 2)

    class Model:
        num_frame_per_block = 1

        def inference(self, **kwargs: Any) -> list[torch.Tensor]:
            assert kwargs["conditional_dict"]["cond_concat"].shape[1] == 20
            assert kwargs["conditional_dict"]["keyboard_cond"].shape[-1] == 4
            assert kwargs["conditional_dict"]["mouse_cond"].shape[-1] == 2
            return [decoded]

    synthesis = MatrixGame2Synthesis(pipeline=Model(), vae=VAE(), weight_dtype=torch.float32, device="cpu")
    return MatrixGame2Pipeline(synthesis_model=synthesis, device="cpu", weight_dtype=torch.float32), decoded


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("save", [False, True])
def test_mg2_declares_real_post_visualization_unit_frame_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, route: str, save: bool
) -> None:
    pipeline, decoded = _mg2(count)
    original = decoded.clone()
    encoded: list[np.ndarray] = []
    target = tmp_path / "out" / "video.mp4"

    def write(frames: np.ndarray, path: str, fps: int) -> str:
        assert Path(path) == target and fps == 19
        encoded.append(frames)
        Path(path).write_bytes(b"encoded output")
        return path

    monkeypatch.setattr("worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2.write_video", write)
    method = pipeline if route == "batch" else pipeline.stream
    kwargs: dict[str, Any] = {
        "images": Image.new("RGB", (16, 16)),
        "interactions": ["forward"],
        "num_frames": count,
        "size": (16, 16),
        "visualize_ops": False,
        "seed": 7,
        "fps": 19,
    }
    if save:
        kwargs["output_path"] = target
    result = method(return_dict=True, **kwargs)
    pixels = ((decoded.float() + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)[0].transpose(0, 2, 3, 1)
    _assert_chunk(result, pixels, layout="frame-list", value_range="0,1")
    assert isinstance(result["video"], list)
    np.testing.assert_array_equal(np.stack(result["video"]), pixels / 255)
    assert result["fps"] == 19
    torch.testing.assert_close(decoded, original)
    if save:
        assert result["artifact_path"] == str(target)
        np.testing.assert_array_equal(encoded[0], pixels)
    else:
        assert result["artifact_path"] == "" and not encoded
    if route == "stream":
        assert all(
            actual is original
            for actual, original in zip(pipeline.memory_module.current_video, result["video"], strict=True)
        )
        assert len(pipeline.memory_module.all_frames) == count
        np.testing.assert_array_equal(np.asarray(pipeline.memory_module.select()), pixels[-1])
        kwargs["images"] = None
    raw = method(**kwargs)
    assert isinstance(raw, list)
    np.testing.assert_array_equal(np.stack(raw), pixels / 255)


def _mg3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, rank: int
) -> tuple[MatrixGame3Pipeline, np.ndarray, list[list[str]], list[str]]:
    pixels = np.arange(count * 3 * 4 * 3, dtype=np.uint8).reshape(count, 3, 4, 3)
    commands: list[list[str]] = []
    reads: list[str] = []
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    runtime = mg3.MatrixGame3Runtime(runtime_path=str(tmp_path), checkpoint_dir=str(tmp_path), device="cpu")
    monkeypatch.setattr(runtime, "_ensure_checkpoint_layout", lambda path: None)
    monkeypatch.setattr(mg3, "_wait_for_video_readable", lambda path: None)

    def run(command: list[str], **kwargs: Any) -> None:
        assert kwargs["check"] is True
        commands.append(command)
        root = Path(command[command.index("--output_dir") + 1])
        name = command[command.index("--save_name") + 1]
        if rank == 0:
            (root / f"{name}.mp4").write_bytes(b"producer mp4")
        controls = json.loads(Path(command[command.index("--actions_json") + 1]).read_text())
        assert len(controls["keyboard_condition"]) == len(controls["mouse_condition"]) == 57
        with Image.open(command[command.index("--image_path") + 1]) as image:
            assert image.mode == "RGB"

    def read(path: str) -> np.ndarray:
        assert rank == 0 and Path(path).read_bytes() == b"producer mp4"
        reads.append(path)
        return pixels

    monkeypatch.setattr(mg3.subprocess, "run", run)
    monkeypatch.setattr(mg3, "load_video_frames", read)
    pipeline = MatrixGame3Pipeline(synthesis_model=MatrixGame3Synthesis(runtime=runtime), device="cpu")
    return pipeline, pixels, commands, reads


@pytest.mark.parametrize("count", [1, 3, 4, 5])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("route", ["batch", "stream"])
@pytest.mark.parametrize("copy_artifact", [False, True])
def test_mg3_declares_readback_preserves_artifact_and_non_output_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, rank: int, route: str, copy_artifact: bool
) -> None:
    pipeline, pixels, commands, reads = _mg3(tmp_path, monkeypatch, count, rank)
    original = pixels.copy()
    target = tmp_path / "export" / "video.mp4"
    kwargs: dict[str, Any] = {
        "images": Image.new("RGB", (4, 3)),
        "interactions": ["forward"],
        "prompt": "scene",
        "num_iterations": 1,
        "output_dir": str(tmp_path / "generation"),
        "fps": 19,
        "size": (3, 4),
        "visualize_ops": False,
        "show_progress": False,
    }
    if copy_artifact:
        kwargs["output_path"] = target
    method = pipeline if route == "batch" else pipeline.stream
    result = method(return_dict=True, **kwargs)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--fps") + 1] == "19"
    if rank == 0:
        _assert_chunk(result, pixels, layout="thwc", value_range="0,255")
        assert result["video_chunk"].metadata["fps"] == 19
        expected_artifact = target if copy_artifact else Path(result["generated_video_path"])
        assert result["artifact_path"] == str(expected_artifact)
        assert expected_artifact.read_bytes() == b"producer mp4"
        assert len(reads) == 1
        if route == "stream":
            assert pipeline.memory_module.current_video is pixels
            assert len(pipeline.memory_module.all_frames) == count
            np.testing.assert_array_equal(np.asarray(pipeline.memory_module.select()), pixels[-1])
    else:
        assert result["video"] is None and result["video_chunk"] is None
        assert result["artifact_path"] is None
        with pytest.raises(RuntimeError, match="no in-memory RGB frames"):
            realtime_frames_from_result(result)
        assert not target.exists() and not Path(result["generated_video_path"]).exists()
        assert reads == []
        if route == "stream":
            assert pipeline.memory_module.all_frames == []
            assert len(pipeline.memory_module.storage) == 1
    np.testing.assert_array_equal(pixels, original)
    kwargs.pop("output_path", None)
    if route == "stream":
        kwargs["images"] = None
    raw = method(**kwargs)
    assert raw is (pixels if rank == 0 else None)


def _studio_stream(pipeline: Any, tmp_path: Path, model_id: str, call_kwargs: dict[str, Any]) -> Any:
    from worldfoundry.studio.inference.catalog import CatalogEntry
    from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = CatalogEntry(
        model_id=model_id,
        display_name=model_id,
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
        cache_key=model_id,
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=Image.new("RGB", (16, 16)),
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=["forward"],
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=19,
        num_frames=3,
        output_dir=str(tmp_path / "generation"),
        output_path=str(tmp_path / "export.mp4"),
        call_kwargs=call_kwargs,
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    return manager.runtime_driver_for(entry)._invoke(ctx, request, "stream", materialize_outputs=False)


def test_studio_requests_mg2_declared_stream_with_raw_frame_memory(tmp_path: Path) -> None:
    pipeline, decoded = _mg2(3)
    result = _studio_stream(pipeline, tmp_path, "matrix-game-2", {"size": (16, 16), "visualize_ops": False, "seed": 7})
    pixels = ((decoded.float() + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)[0].transpose(0, 2, 3, 1)
    _assert_chunk(result, pixels, layout="frame-list", value_range="0,1")
    assert result["fps"] == 19 and len(pipeline.memory_module.all_frames) == 3
    assert result["artifact_path"] == "" and not (tmp_path / "export.mp4").exists()


@pytest.mark.parametrize("rank", [0, 1])
def test_studio_requests_mg3_declared_stream_on_each_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    pipeline, pixels, _, reads = _mg3(tmp_path, monkeypatch, 4, rank)
    result = _studio_stream(
        pipeline,
        tmp_path,
        "matrix-game-3",
        {"num_iterations": 1, "size": (3, 4), "visualize_ops": False, "output_dir": str(tmp_path / "generation")},
    )
    if rank == 0:
        _assert_chunk(result, pixels, layout="thwc", value_range="0,255")
        assert len(reads) == 1
    else:
        assert result["video_chunk"] is None and result["artifact_path"] is None and not reads
    assert not (tmp_path / "export.mp4").exists()


@pytest.mark.parametrize("pipeline_cls", [MatrixGame2Pipeline, MatrixGame3Pipeline])
def test_unloaded_generation_reports_the_model_phase(pipeline_cls: type) -> None:
    pipeline = pipeline_cls(device="cpu")
    with pytest.raises(RuntimeError, match="synthesis model is not loaded"):
        pipeline(images=Image.new("RGB", (4, 3)), num_frames=3)
