# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Shared Studio dispatch retains trajectory output declarations and request cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.processing.postprocess import VideoChunk, VideoTensorLayout
from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline
from worldfoundry.pipelines.kling.pipeline_astra import AstraConfig, AstraPipeline
from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline
from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager
from worldfoundry.synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from worldfoundry.synthesis.visual_generation.memory.kling.astra import AstraMemory

pytestmark = pytest.mark.unit


def _invoke(pipeline: Any, tmp_path: Path, model_id: str, *, mode: str = "stream", input_path: str = "") -> Any:
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
        input_path=input_path,
        image=None if input_path else Image.new("RGB", (4, 3)),
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=["left"],
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=12,
        num_frames=121,
        output_dir=str(tmp_path),
        output_path=str(tmp_path / "output.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    return manager.runtime_driver_for(entry)._invoke(ctx, request, mode, materialize_outputs=False)


@pytest.mark.parametrize("mode", ["run", "stream"])
def test_studio_requests_gen3c_declaration_frame_count_and_fps(tmp_path: Path, mode: str) -> None:
    requests: list[Any] = []
    sample = torch.full((1, 3, 5, 3, 4), 0.5)

    def native(request: Any) -> Any:
        requests.append(request)
        return SimpleNamespace(sample=sample, latents=sample, metadata={}, artifacts={})

    pipeline = Gen3CPipeline(native_pipeline=native, device="cpu")
    result = _invoke(pipeline, tmp_path, "gen3c", mode=mode)
    assert result["video_chunk"].frames is sample and result["video_chunk"].frame_count == 5
    assert result["video_chunk"].layout == "bcthw" and result["video_chunk"].value_range == "-1,1"
    assert result["fps"] == requests[0].inputs["fps"] == 12 and requests[0].num_frames == 121
    assert result["artifact_path"] is None and not (tmp_path / "output.mp4").exists()
    if mode == "stream":
        assert pipeline.memory_module.current_video is sample
        assert pipeline.memory_module.current_image.getpixel((0, 0)) == (191, 191, 191)


@pytest.mark.parametrize(
    "pipeline_cls,layout,value_range", [(Lyra1Pipeline, "bcthw", "-1,1"), (Lyra2Pipeline, "thwc", "0,255")]
)
@pytest.mark.parametrize("mode", ["run", "stream"])
def test_studio_requests_lyra_structured_outputs_and_forwards_fps(
    tmp_path: Path, pipeline_cls: type, layout: VideoTensorLayout, value_range: str, mode: str
) -> None:
    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    pixels = np.full((5, 3, 4, 3), 191, np.uint8)
    calls: list[dict[str, Any]] = []
    video = sample if pipeline_cls is Lyra1Pipeline else pixels
    chunk = VideoChunk(frames=video, layout=layout, value_range=value_range, metadata={"fps": 12})

    def predict(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "video": video,
            "video_tensor": sample,
            "video_chunk": chunk,
            "fps": kwargs["fps"],
            "generated_root": str(tmp_path),
            "generated_video_path": str(tmp_path / "generated.mp4"),
        }

    pipeline = pipeline_cls(synthesis_model=SimpleNamespace(predict=predict), device="cpu")
    result = _invoke(pipeline, tmp_path, "lyra-1" if pipeline_cls is Lyra1Pipeline else "lyra-2", mode=mode)
    assert result["video_chunk"] is chunk and result["fps"] == calls[0]["fps"] == 12
    assert result["video"] is video and result["reconstruction"] is None
    if mode == "stream":
        assert pipeline.memory_module.current_video is video and len(pipeline.memory_module.all_frames) == 5
        assert pipeline.memory_module.current_image.getpixel((0, 0)) == (191, 191, 191)
    assert not (tmp_path / "output.mp4").exists()


def test_studio_requests_astra_cropped_byte_frames_and_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def decode(latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.full((1, 3, 4 * (latents.shape[2] - 1) + 1, 3, 4), 0.5)

    synthesis = AstraSynthesis(
        SimpleNamespace(
            device="cpu", decode_video=decode, vae=SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))
        )
    )
    monkeypatch.setattr(synthesis, "predict", lambda data, count, *args, **kwargs: torch.zeros(1, 3, count, 2, 2))
    pipeline = AstraPipeline(SimpleNamespace(), synthesis, AstraMemory(capacity=100), AstraConfig(device="cpu"))

    def process(image: str, interaction: dict[str, Any], args: AstraConfig) -> dict[str, Any]:
        assert image == "input.png" and interaction["prompt"] == "scene"
        return {
            "initial_latents": torch.zeros(3, 1, 2, 2),
            "total_frames": args.total_frames_to_generate,
            "camera_embedding_full": torch.zeros(args.total_frames_to_generate + 31, 13),
            "camera_embedding_uncond": None,
            "prompt_emb_pos": {},
            "prompt_emb_neg": None,
        }

    monkeypatch.setattr(pipeline, "process", process)
    result = _invoke(pipeline, tmp_path, "astra", mode="run", input_path="input.png")
    assert result["video_chunk"].frames is result["frames"] and result["video_chunk"].frame_count == 121
    assert result["video_chunk"].layout == "frame-list" and result["video_chunk"].value_range == "0,255"
    assert result["fps"] == result["video_chunk"].metadata["fps"] == 12 and result["output_path"] is None
    assert result["frames"][-1].getpixel((0, 0)) == (191, 191, 191)
    assert pipeline.memory.history_latents is not None and pipeline.memory.history_latents.shape[1] == 33
    assert not (tmp_path / "output.mp4").exists()
