# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Native product output semantics and their shared Studio transport boundary."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result

pytestmark = pytest.mark.unit


def _native_types() -> list[type[NativeVisualDiffusionPipeline]]:
    root = Path(__file__).resolve().parents[2]
    definitions = []
    for path in (root / "worldfoundry/pipelines").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                definitions.append((path, node.name, {base.id for base in node.bases if isinstance(base, ast.Name)}))
    names = {"NativeVisualDiffusionPipeline"}
    pending = list(definitions)
    targets = []
    while pending:
        children = [definition for definition in pending if definition[2] & names]
        if not children:
            break
        for path, name, bases in children:
            names.add(name)
            pending.remove((path, name, bases))
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            pipeline_type = getattr(importlib.import_module(module), name)
            if pipeline_type.MODEL_ID:
                targets.append(pipeline_type)
    assert targets
    return targets


@pytest.mark.parametrize("pipeline_type", _native_types(), ids=lambda cls: cls.__name__)
def test_every_native_visual_product_declares_pixel_semantics(
    pipeline_type: type[NativeVisualDiffusionPipeline],
) -> None:
    assert issubclass(pipeline_type, NativeVisualDiffusionPipeline)
    assert pipeline_type.OUTPUT_VALUE_RANGE in {"-1,1", "0,1"}
    assert pipeline_type.OUTPUT_LAYOUT in {"BCTHW", "FHWC"}


class _Product(NativeVisualDiffusionPipeline):
    MODEL_ID = "output-contract"


@pytest.mark.parametrize("shape", [(1, 3, 3, 4), (1, 3, 1, 3, 4), (1, 3, 5, 3, 4)])
@pytest.mark.parametrize("stream", [False, True])
def test_shared_product_carries_decoded_pixels_without_changing_raw_return(
    shape: tuple[int, ...], stream: bool
) -> None:
    sample = torch.full(shape, 0.5)

    class Native:
        model_id = "output-contract"

        def __call__(self, request: Any) -> Any:
            return SimpleNamespace(sample=sample, latents=None, artifacts={}, metadata={})

    pipeline = _Product(native_pipeline=Native(), device="cpu")
    method = pipeline.stream if stream else pipeline
    result = method(prompt="scene", return_dict=True)
    assert result["sample"] is sample
    chunk = result["video_chunk"]
    assert chunk.layout == "bcthw" and chunk.value_range == "-1,1"
    assert chunk.frames.data_ptr() == sample.data_ptr()
    frames = realtime_frames_from_result(result)
    assert len(frames) == (1 if len(shape) == 4 else shape[2])
    assert np.stack(frames).shape[1:] == (3, 4, 3)
    assert np.all(np.stack(frames) == 191)
    assert method(prompt="scene", return_dict=False) is sample
    assert torch.all(sample == 0.5)
    if stream:
        last_frame = pipeline.memory_module.select()
        assert last_frame.size == (4, 3)
        assert last_frame.getpixel((0, 0)) == (191, 191, 191)
        assert pipeline.memory_module.select(prefer_type="video") is sample


def test_frame_major_decoder_output_is_described_after_public_layout_conversion() -> None:
    class Product(_Product):
        OUTPUT_LAYOUT = "FHWC"
        OUTPUT_VALUE_RANGE = "0,1"

    sample = torch.full((5, 3, 4, 3), 0.5)

    class Native:
        model_id = "output-contract"

        def __call__(self, request: Any) -> Any:
            return SimpleNamespace(sample=sample, latents=None, artifacts={}, metadata={})

    result = Product(native_pipeline=Native(), device="cpu")(return_dict=True)
    assert result["sample"].shape == (1, 3, 5, 3, 4)
    assert result["video_chunk"].frames is result["sample"]
    frames = realtime_frames_from_result(result)
    assert len(frames) == 5
    assert np.all(np.stack(frames) == 127)


@pytest.mark.parametrize("fps,expected_fps", [(None, 18), (24, 24)])
def test_native_request_defaults_keep_frame_rate_and_structured_conditioning(
    fps: int | None,
    expected_fps: int,
) -> None:
    conditioning = {"camera": [1.0, 2.0], "enabled": True}

    class Product(_Product):
        REQUEST_INPUT_DEFAULTS = {"frame_rate": "18", "conditioning": conditioning}

    class Native:
        model_id = "output-contract"

        def __call__(self, request: Any) -> Any:
            assert request.inputs["fps"] == request.inputs["frame_rate"] == expected_fps
            assert request.inputs["conditioning"] is conditioning
            return SimpleNamespace(sample=torch.zeros(1, 3, 2, 4, 4), latents=None, artifacts={}, metadata={})

    result = Product(native_pipeline=Native(), device="cpu")(fps=fps, return_dict=True)
    assert result["model_name"] == "output-contract"
    assert result["video_chunk"].frame_count == 2


def test_latents_are_not_declared_as_rgb() -> None:
    class Native:
        model_id = "output-contract"

        def __call__(self, request: Any) -> Any:
            assert request.inputs["return_latent"]
            return SimpleNamespace(sample=torch.zeros(1, 16, 5, 2, 2), latents=None, artifacts={}, metadata={})

    result = _Product(native_pipeline=Native(), device="cpu")(output_type="latent", return_dict=True)
    assert "video_chunk" not in result


@pytest.mark.parametrize("route", ["realtime", "worker"])
def test_native_artifact_history_keeps_explicit_conditioning_across_stream_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    from dataclasses import replace

    from PIL import Image

    from worldfoundry.studio.inference.catalog import CatalogEntry
    from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

    class Product(_Product):
        ACCEPTS_IMAGES = True
        REQUIRES_IMAGES = True

    class Native:
        model_id = "output-contract"

        def __init__(self) -> None:
            self.images: list[Any] = []

        def __call__(self, request: Any) -> Any:
            self.images.append(request.inputs["images"])
            return SimpleNamespace(sample=torch.full((1, 3, 5, 3, 4), 0.5), latents=None, artifacts={}, metadata={})

    native = Native()
    pipeline = Product(native_pipeline=native, device="cpu")
    entry = CatalogEntry(
        model_id=Product.MODEL_ID,
        display_name="Native product",
        module_path=__name__,
        class_name="Product",
        family="native",
        category="Image-to-Video",
        summary="",
        supports_stream=True,
    )
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key="native",
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
        fps=8,
        num_frames=5,
        output_dir=str(tmp_path / "output"),
        output_path=str(tmp_path / "output" / "video.mp4"),
        call_kwargs={"height": 3, "width": 4},
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
                entry=entry,
                request=current,
                action="stream",
                materialize=False,
            )
        assert result["video_chunk"].frame_count == 5
        assert not ctx.state
        assert ctx.active_leases == 0
    assert native.images == [first, second, first]
    assert len(pipeline.memory_module.storage) == 3
    assert pipeline.memory_module.select().getpixel((0, 0)) == (191, 191, 191)
