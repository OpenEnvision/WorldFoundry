# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Batch artifact readback contracts; generation, encoding and process groups are substituted."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.pipelines.lingbot_world_v2.pipeline_lingbot_world_v2 import (
    LingBotWorldV2Pipeline,
    LingBotWorldV2SmallPipeline,
)
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.lingbot_world_v2 import runtime as module
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.lingbot_world_v2_synthesis import LingBotWorldV2Synthesis
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.runtime import LingBotWorldV2Runtime

pytestmark = pytest.mark.unit


def _runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_id: str) -> LingBotWorldV2Runtime:
    np.save(tmp_path / "poses.npy", np.repeat(np.eye(4)[None], 5, axis=0))
    np.save(tmp_path / "intrinsics.npy", np.repeat(np.eye(3)[None], 5, axis=0))
    runtime = LingBotWorldV2Runtime(
        model_id=model_id,
        defaults={"frame_num": 5, "chunk_size": 1, "nproc_per_node": 2, "t5_fsdp": False, "dit_fsdp": False},
    )
    monkeypatch.setattr(runtime, "_resolve_checkpoint", lambda **kwargs: tmp_path)
    monkeypatch.setattr(runtime, "_resolve_shared_assets", lambda **kwargs: tmp_path)
    return runtime


def _assert_readback(result: dict[str, Any], frames: np.ndarray, target: Path) -> None:
    chunk: VideoChunk = result["video_chunk"]
    assert chunk.frames is result["video"] is frames
    assert (chunk.layout, chunk.value_range, chunk.frame_count) == ("thwc", "0,255", 5)
    assert result["artifact_path"] == str(target)
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), frames)


@pytest.mark.parametrize("model_id", ["lingbot-world-v2", "lingbot-world-v2-1.3b"])
@pytest.mark.parametrize("rank", [0, 1])
def test_existing_group_declares_readback_and_preserves_rank_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, rank: int
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, model_id)
    latent_video = torch.full((3, 5, 3, 4), 0.5)
    expected_encoded = np.full((5, 3, 4, 3), 191, dtype=np.uint8)
    readback = np.full((5, 3, 4, 3), 190, dtype=np.uint8)
    target = tmp_path / "video.mp4"
    generated: list[bool] = []
    encoded: list[bool] = []

    class Core:
        def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor | None:
            generated.append(True)
            return latent_video if rank == 0 else None

    def save(video: torch.Tensor, path: Path, *, fps: int, value_range: Any, layout: Any) -> None:
        assert rank == 0 and path == target and fps == 16
        assert layout == "bcthw" and value_range == (-1, 1)
        assert video.shape == (1, 3, 5, 3, 4)
        assert video.data_ptr() == latent_video.data_ptr()
        np.testing.assert_array_equal(
            video_tensor_to_uint8_frames(video, layout=layout, value_range=value_range), expected_encoded
        )
        encoded.append(True)
        path.write_bytes(b"encoded-artifact")

    decoded: list[Path] = []

    def load(path: Path) -> np.ndarray:
        assert rank == 0 and path == target
        decoded.append(path)
        return readback

    group = SimpleNamespace(broadcast_object_list=lambda payload, **kwargs: None)
    monkeypatch.setattr(runtime, "_initialized_distributed_context", lambda: (torch, group, rank, 2))
    monkeypatch.setattr(runtime, "_get_distributed_core", lambda **kwargs: Core())
    monkeypatch.setattr(module, "save_image_or_video_tensor", save)
    monkeypatch.setattr(module, "load_video_frames", load)
    result = runtime.predict(prompt="scene", images=Image.new("RGB", (4, 3)), action_path=tmp_path, output_path=target)
    assert generated == [True]
    if rank == 0:
        assert result is not None
        _assert_readback(result, readback, target)
        assert result["execution_mode"] == "existing_torchrun_process_group"
        assert encoded == [True] and decoded == [target]
    else:
        assert result is None
        assert encoded == [] and decoded == [] and not target.exists()
    torch.testing.assert_close(latent_video, torch.full_like(latent_video, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("model_id", ["lingbot-world-v2", "lingbot-world-v2-1.3b"])
def test_subprocess_declares_readback_without_replacing_or_reencoding_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, model_id)
    monkeypatch.setattr(runtime, "_initialized_distributed_context", lambda: None)
    frames = np.arange(5 * 3 * 4 * 3, dtype=np.uint8).reshape(5, 3, 4, 3)
    target = tmp_path / "video.mp4"
    generated: list[Path] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        assert command[command.index("--frame-num") + 1] == "5"
        assert command[command.index("--nproc_per_node") + 1] == "2"
        path = Path(command[command.index("--output") + 1])
        generated.append(path)
        path.write_bytes(b"preserved-encoded-artifact")
        return SimpleNamespace(returncode=0)

    def load(path: Path) -> np.ndarray:
        assert path == target and path.read_bytes() == b"preserved-encoded-artifact"
        return frames

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "load_video_frames", load)
    monkeypatch.setattr(module, "save_image_or_video_tensor", lambda *args, **kwargs: pytest.fail("no re-encoding"))
    result = runtime.predict(prompt="scene", images=Image.new("RGB", (4, 3)), action_path=tmp_path, output_path=target)
    assert result is not None
    _assert_readback(result, frames, target)
    assert len(generated) == 1 and not generated[0].exists()
    assert Path(result["stdout_path"]).is_file() and Path(result["stderr_path"]).is_file()


@pytest.mark.parametrize("pipeline_cls", [LingBotWorldV2Pipeline, LingBotWorldV2SmallPipeline])
@pytest.mark.parametrize("route", ["batch", "stream"])
def test_v2_public_pipeline_keeps_path_defaults_and_structured_chunk(
    tmp_path: Path, pipeline_cls: type[Any], route: str
) -> None:
    frames = np.zeros((5, 3, 4, 3), dtype=np.uint8)
    target = tmp_path / "video.mp4"
    payload = {
        "artifact_path": str(target),
        "video": frames,
        "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255"),
    }

    class Runtime:
        defaults = {"frame_num": 5}
        model_id = pipeline_cls.MODEL_ID
        model_name = "LingBot V2"
        generation_type = "i2v"
        device = "cpu"

        def predict(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["frame_num"] == 5
            assert kwargs.get("fps") in (None, 16)
            return payload

    pipeline = pipeline_cls(
        model_id=pipeline_cls.MODEL_ID, synthesis_model=LingBotWorldV2Synthesis(cast(Any, Runtime()))
    )
    method = pipeline if route == "batch" else pipeline.stream
    assert "return_dict" in inspect.signature(method).parameters
    if route == "stream":
        assert {"num_frames", "fps"} <= inspect.signature(method).parameters.keys()
    assert method(num_frames=5, fps=16, return_dict=True) is payload
    assert method() == str(target)
    assert method(None, "scene", None, None, None, target, False, 5, None, 480, 832) == str(target)
