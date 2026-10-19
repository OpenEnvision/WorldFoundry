# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Shared dispatch and materialization honor independent native video declarations."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.studio.inference import execution
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result

pytestmark = pytest.mark.unit

ROUTES = [
    ("cosmos.pipeline_cosmos_predict2", "CosmosPredict2Pipeline", "cosmos-predict2"),
    ("cosmos.pipeline_cosmos_predict2p5", "CosmosPredict2p5Pipeline", "cosmos-predict2.5"),
    ("cosmos.pipeline_cosmos_transfer2p5", "CosmosTransfer2p5Pipeline", "cosmos-transfer-2.5"),
    ("cosmos.pipeline_cosmos3", "Cosmos3Pipeline", "cosmos3"),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideoT2VPipeline", "hunyuanvideo-t2v"),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideoI2VPipeline", "hunyuanvideo-i2v"),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideo15T2VPipeline", "hunyuanvideo-1.5-t2v"),
    ("hunyuan_video.pipeline_hunyuan_video", "HunyuanVideo15I2VPipeline", "hunyuanvideo-1.5-i2v"),
    ("t2v_turbo.pipeline_t2v_turbo_t2v", "T2VTurboT2VPipeline", "t2v_turbo_t2v"),
    ("vchitect.pipeline_vchitect_2_t2v", "Vchitect2T2VPipeline", "vchitect-2-t2v"),
]


def _setup(tmp_path: Path, route: tuple, *, latent: bool = False) -> tuple[Any, ...]:
    module = importlib.import_module(f"worldfoundry.pipelines.{route[0]}")
    cls = getattr(module, route[1])
    requests: list[Any] = []
    sample = torch.full((1, 3, 5, 3, 4), 0.5)
    latents = torch.full((1, 4, 5, 3, 4), 0.25)

    def native(request: Any) -> Any:
        requests.append(request)
        return SimpleNamespace(
            sample=latents if request.inputs["return_latent"] else sample,
            latents=latents,
            metadata={},
            artifacts={},
        )

    options = {"native_pipeline": native, "device": "cpu"}
    if (
        cls.__name__ in {"CosmosPredict2Pipeline", "CosmosPredict2p5Pipeline", "Cosmos3Pipeline"}
        or "Hunyuan" in route[1]
    ):
        options["model_id"] = route[2]
    pipeline = cls(**options)
    entry = CatalogEntry(
        model_id=route[2],
        display_name=route[1],
        module_path=module.__name__,
        class_name=route[1],
        family="video",
        category="Video Generation",
        summary="",
        supports_stream=route in ROUTES[4:],
    )
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key=route[2],
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    image = (
        Image.new("RGB", (4, 3), "red")
        if route[1] == "CosmosPredict2Pipeline" or route[1].endswith("I2VPipeline")
        else None
    )
    call_kwargs: dict[str, Any] = {"output_type": "latent"} if latent else {}
    if "transfer" in route[0]:
        call_kwargs["control_video"] = torch.zeros(1, 3, 17, 3, 4)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=image,
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
        num_frames=17,
        output_dir=str(output_dir),
        output_path=str(output_dir / "output.mp4"),
        call_kwargs=call_kwargs,
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    return manager, ctx, request, requests, sample, latents


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
def test_studio_batch_requests_declarations_and_preserves_frame_rate(tmp_path: Path, route: tuple) -> None:
    manager, ctx, request, requests, sample, _ = _setup(tmp_path, route)
    result = manager.runtime_driver_for(ctx.entry)._invoke(ctx, request, "run", materialize_outputs=False)
    assert result["video_chunk"].frames is result["video"] is sample
    assert result["video_chunk"].frame_count == 5
    assert result["fps"] == requests[0].inputs["fps"] == 12 and requests[0].num_frames == 17
    assert result["artifact_path"] is None and not Path(request.output_path).exists()


@pytest.mark.parametrize("route", ROUTES[4:], ids=[route[1] for route in ROUTES[4:]])
def test_studio_continuation_requests_structured_output_without_reconditioning(tmp_path: Path, route: tuple) -> None:
    manager, ctx, request, requests, sample, _ = _setup(tmp_path, route)
    driver = manager.runtime_driver_for(ctx.entry)
    assert not driver.can_init(ctx, request)
    result = driver.continue_stream(ctx, request, materialize_outputs=False)
    assert result["video_chunk"].frames is sample and result["fps"] == requests[0].inputs["fps"] == 12
    assert requests[0].num_frames == 17 and ctx.state == {}
    if request.image is not None:
        assert requests[0].inputs["image"] is request.image
    pixels = 127 if "hunyuan" in route[0] else 191
    assert ctx.pipeline.memory_module.select().getpixel((0, 0)) == (pixels, pixels, pixels)
    assert not Path(request.output_path).exists()


@pytest.mark.parametrize("route", ROUTES, ids=[route[1] for route in ROUTES])
@pytest.mark.parametrize("latent", [False, True])
def test_materializer_exports_declared_pixels_but_not_latents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: tuple, latent: bool
) -> None:
    manager, ctx, request, _, sample, latents = _setup(tmp_path, route, latent=latent)
    result = manager.runtime_driver_for(ctx.entry)._invoke(ctx, request, "run", materialize_outputs=False)
    writes: list[Any] = []

    def export(frames: Any, path: str, *, fps: int) -> str:
        writes.append(frames)
        assert fps == 12
        np.testing.assert_array_equal(
            np.stack(frames), np.full((5, 3, 4, 3), 127 if "hunyuan" in route[0] else 191, np.uint8)
        )
        Path(path).write_bytes(b"preview-artifact")
        return path

    monkeypatch.setattr(execution, "export_frames_to_video", export)
    record = manager.materialize_run(ctx, request, result=result, mode="run")
    assert len(writes) == int(not latent)
    assert result["video"] is (latents if latent else sample)
    assert bool(record.preview_video) == (not latent)
    if latent:
        assert result["video_chunk"] is None and not Path(request.output_path).exists()
        with pytest.raises(RuntimeError, match="no in-memory RGB frames"):
            realtime_frames_from_result(result)


@pytest.mark.parametrize("route", [ROUTES[5], ROUTES[7]], ids=["original", "1.5"])
@pytest.mark.parametrize("dispatch", ["realtime", "worker"])
def test_i2v_artifact_history_does_not_replace_explicit_a_b_a_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: tuple, dispatch: str
) -> None:
    manager, ctx, request, requests, sample, _ = _setup(tmp_path, route)
    manager.pipeline_cache[ctx.cache_key] = ctx
    driver = manager.runtime_driver_for(ctx.entry)
    monkeypatch.setattr(driver, "load_pipeline", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(manager, "_torchrun_worker_request", lambda entry, value: value)
    first, second = request.image, Image.new("RGB", (4, 3), "blue")
    for image in (first, second, first):
        current = replace(request, image=image)
        if dispatch == "realtime":
            result = manager.run_realtime(entry=ctx.entry, request=current, action="stream")
        else:
            result = manager._run_local_torchrun_lingbot_fast(
                entry=ctx.entry,
                request=current,
                action="stream",
                materialize=False,
            )
        assert result["video_chunk"].frames is sample and ctx.active_leases == 0 and not ctx.state
    assert [item.inputs["image"] for item in requests] == [first, second, first]
    assert len(ctx.pipeline.memory_module.storage) == 3
    assert ctx.pipeline.memory_module.select().getpixel((0, 0)) == (127, 127, 127)


@pytest.mark.parametrize(
    "declared", [None, VideoChunk(frames=np.empty((0, 3, 4, 3), np.uint8), layout="thwc", value_range="0,255")]
)
def test_explicit_no_rgb_or_buffering_declaration_disables_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: VideoChunk | None
) -> None:
    manager, ctx, request, *_ = _setup(tmp_path, ROUTES[-1])
    result = {"video_chunk": declared, "video": torch.ones(1, 4, 5, 3, 4), "frames": [Image.new("RGB", (4, 3))]}
    monkeypatch.setattr(
        execution, "export_frames_to_video", lambda *args, **kwargs: pytest.fail("undeclared RGB fallback")
    )
    record = manager.materialize_run(ctx, request, result=result, mode="run")
    assert record.preview_video is None
    if declared is None:
        with pytest.raises(RuntimeError, match="no in-memory RGB frames"):
            realtime_frames_from_result(result)
    else:
        assert realtime_frames_from_result(result) == []


def test_undeclared_legacy_pixels_keep_existing_materialization_and_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, ctx, request, *_ = _setup(tmp_path, ROUTES[-1])
    raw = np.full((5, 3, 4, 3), 128, np.uint8)
    result = {"video": raw}
    writes: list[Any] = []

    def export(frames: Any, path: str, *, fps: int) -> str:
        writes.append(frames)
        Path(path).write_bytes(b"legacy-preview")
        return path

    monkeypatch.setattr(execution, "export_frames_to_video", export)
    record = manager.materialize_run(ctx, request, result=result, mode="run")
    assert record.preview_video == request.output_path and len(writes) == 1
    np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), raw)
