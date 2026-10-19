# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Shared dispatch contracts; no model weights, Torch, or artifact exporters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldfoundry.studio.inference.catalog import discover_catalog, find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager

pytestmark = pytest.mark.unit


class _FrameMemory:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def manage(self, action: str) -> None:
        assert action == "reset"
        self.records.clear()

    def record(self, value: Any, **kwargs: Any) -> None:
        self.records.append(value)


class _FramePipeline:
    def __init__(self) -> None:
        self.memory_module = _FrameMemory()
        self.calls: list[Any] = []

    def stream(self, images: Any, interactions: Any = None) -> Image.Image:
        assert images is None
        assert len(self.memory_module.records) == 1
        self.calls.append(interactions)
        return Image.new("RGB", (8, 8))


class _OwnedPipeline:
    def __init__(self) -> None:
        self.scene: Any = None
        self.calls: list[Any] = []
        self.fail = False

    def init_stream(self, images: Any = None, output_dir: str | None = None) -> dict[str, Any]:
        if self.fail:
            raise ValueError("invalid source")
        self.scene = images
        self.calls.append(("init", output_dir))
        return {"scene_context": self.scene}

    def stream_state_ready(self) -> bool:
        return self.scene is not None

    def stream(self, images: Any = None, interactions: Any = None) -> Image.Image:
        assert self.scene is not None
        self.calls.append(("stream", interactions))
        return Image.new("RGB", (8, 8))

    def reset_memory(self) -> None:
        self.scene = None


class _NativeRealtimePipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare_realtime(self) -> None:
        self.calls.append("prepare")

    def configure_realtime(self, **kwargs: Any) -> None:
        self.calls.append("configure")

    def stream_realtime(self, **kwargs: Any) -> Image.Image:
        self.calls.append("stream")
        return Image.new("RGB", (8, 8))

    def reset_realtime(self) -> None:
        self.calls.append("reset")


@pytest.fixture
def dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = replace(find_entry("infinite-world"), model_id="dispatch-contract", default_call_kwargs={})
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=Image.new("RGB", (8, 8)),
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
        fps=8,
        num_frames=1,
        output_dir=str(tmp_path / "output"),
        output_path=str(tmp_path / "output" / "video.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    ctx = PipelineContext(
        entry=entry,
        pipeline=None,
        cache_key="contract",
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    manager.pipeline_cache[ctx.cache_key] = ctx
    driver = BaseRuntimeDriver()
    monkeypatch.setattr(manager, "runtime_driver_for", lambda entry: driver)
    monkeypatch.setattr(driver, "load_pipeline", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(manager, "_torchrun_worker_request", lambda entry, request: request)
    monkeypatch.setattr(manager, "materialize_run", lambda ctx, request, **kwargs: kwargs["result"])
    monkeypatch.setattr(manager, "_make_message_record", lambda entry, request, message: message)

    def call(route: str, action: str, value: PreparedInputs | None = None) -> Any:
        kwargs: dict[str, Any] = dict(entry=entry, request=request if value is None else value, action=action)
        if route == "ordinary":
            return manager._run_local_prepared_request(**kwargs)
        if route == "realtime":
            return manager.run_realtime(**kwargs)
        return manager._run_local_torchrun_lingbot_fast(
            **kwargs,
            materialize=route == "rank0",
            realtime=route == "realtime-rank",
        )

    return manager, ctx, request, call


@pytest.mark.parametrize("route", ["ordinary", "realtime", "rank0", "worker-rank", "realtime-rank"])
@pytest.mark.parametrize("pipeline_type", [_FramePipeline, _OwnedPipeline])
def test_init_stream_reset_share_state_ownership(dispatch: Any, route: str, pipeline_type: type) -> None:
    _, ctx, request, call = dispatch
    pipeline = ctx.pipeline = pipeline_type()
    call(route, "init")
    assert ctx.state["studio_stream_initialized"]
    call(route, "stream")
    assert len(pipeline.calls) == (2 if pipeline_type is _OwnedPipeline else 1)
    if pipeline_type is _OwnedPipeline:
        assert pipeline.calls[0] == ("init", request.output_dir if route == "ordinary" else None)
    call(route, "reset")
    assert ctx.state == {}
    assert ctx.active_leases == 0
    if pipeline_type is _OwnedPipeline:
        assert pipeline.scene is None
    else:
        assert pipeline.memory_module.records == []


@pytest.mark.parametrize("route", ["ordinary", "realtime", "rank0", "worker-rank", "realtime-rank"])
def test_failed_init_releases_lease_and_keeps_previous_scene(dispatch: Any, route: str) -> None:
    _, ctx, request, call = dispatch
    pipeline = ctx.pipeline = _OwnedPipeline()
    call(route, "init")
    previous = pipeline.scene
    pipeline.fail = True
    with pytest.raises(ValueError, match="invalid source"):
        call(route, "init", replace(request, image=Image.new("RGB", (8, 8), "red")))
    assert pipeline.scene is previous
    assert ctx.active_leases == 0
    pipeline.fail = False
    call(route, "init")
    call(route, "stream")


@pytest.mark.parametrize("route", ["realtime", "realtime-rank"])
def test_native_realtime_prepare_configure_stream_and_public_reset(dispatch: Any, route: str) -> None:
    manager, ctx, request, call = dispatch
    pipeline = ctx.pipeline = _NativeRealtimePipeline()
    call(route, "configure", replace(request, image=None))
    assert not ctx.state
    with pytest.raises(RuntimeError, match="not been configured"):
        call(route, "stream")
    call(route, "configure")
    call(route, "stream")
    manager.pipeline_cache[ctx.cache_key] = ctx
    manager._reset_cached_model_local(ctx.entry.model_id)
    assert pipeline.calls == ["prepare", "configure", "stream", "reset"]
    assert not ctx.state
    assert ctx.active_leases == 0


@pytest.mark.parametrize("terminal", ["exhausted", "stream-error", "closed"])
@pytest.mark.parametrize("close_error", [False, True])
def test_lazy_result_finalizes_lease_once_and_can_be_reused(
    dispatch: Any,
    terminal: str,
    close_error: bool,
) -> None:
    manager, ctx, _, _ = dispatch
    ctx.pipeline = _OwnedPipeline()

    class Chunks:
        def __init__(self) -> None:
            self.emitted = False
            self.close_calls = 0

        def __iter__(self) -> Chunks:
            return self

        def __next__(self) -> str:
            if not self.emitted:
                self.emitted = True
                return "frame"
            if terminal == "stream-error":
                raise ValueError("stream failed")
            raise StopIteration

        def close(self) -> None:
            self.close_calls += 1
            if close_error:
                raise RuntimeError("close failed")

    chunks = Chunks()
    leased = manager._lease_lazy_pipeline_result(ctx, chunks)
    assert next(leased) == "frame"
    assert ctx.active_leases == 1
    if terminal == "closed":
        if close_error:
            with pytest.raises(RuntimeError, match="close failed"):
                leased.close()
        else:
            leased.close()
    else:
        with pytest.raises(ValueError if terminal == "stream-error" else StopIteration):
            next(leased)
    leased.close()
    assert chunks.close_calls == 1
    assert ctx.active_leases == 0
    assert leased._context is None and leased._manager is None
    assert ctx.lifecycle_lock.acquire(blocking=False)
    ctx.lifecycle_lock.release()
    assert list(manager._lease_lazy_pipeline_result(ctx, iter(("next request",)))) == ["next request"]
    assert ctx.active_leases == 0


def test_realtime_init_does_not_materialize_artifacts(dispatch: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, ctx, _, call = dispatch
    ctx.pipeline = _OwnedPipeline()

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("realtime INIT must not materialize artifacts")

    monkeypatch.setattr(manager, "materialize_run", unexpected)
    call("realtime", "configure")
    assert ctx.pipeline.calls == [("init", None)]


def test_every_catalog_entry_resolves_its_runtime_driver(tmp_path: Path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path))
    entries = discover_catalog()
    assert entries
    for entry in entries:
        driver = manager.runtime_driver_for(entry)
        assert driver.kind == ("default" if entry.runtime_kind == "worldfm" else entry.runtime_kind), entry.model_id
    assert {entry.runtime_kind for entry in entries} >= {"default", "two_stage_3dgs", "pointcloud_nav"}
    assert any(entry.supports_api_init for entry in entries)


@pytest.mark.parametrize("route", ["ordinary", "realtime", "rank0", "worker-rank"])
def test_stateless_generation_does_not_acquire_stream_state(dispatch: Any, route: str) -> None:
    _, ctx, _, call = dispatch

    class Stateless:
        def __call__(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
            assert prompt == "scene"
            return {"actions": [1, 2, 3]}

    ctx.pipeline = Stateless()
    assert call(route, "run") == {"actions": [1, 2, 3]}
    assert not ctx.state
    assert ctx.active_leases == 0


@pytest.mark.parametrize("route", ["ordinary", "rank0", "worker-rank"])
def test_action_pipelines_keep_their_non_streaming_step_contract(dispatch: Any, route: str) -> None:
    _, ctx, _, call = dispatch
    ctx.entry = replace(ctx.entry, category="Embodied Action", supports_stream=False)

    class Action:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"actions": [1, 2, 3]}

    ctx.pipeline = Action()
    assert call(route, "stream") == {"actions": [1, 2, 3]}
    assert not ctx.state


@pytest.mark.parametrize("kind", ["two_stage_3dgs", "pointcloud_nav"])
def test_geometry_drivers_reuse_reconstruction_instead_of_generic_stream_init(
    dispatch: Any,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from worldfoundry.studio.inference import execution

    manager, ctx, request, call = dispatch
    ctx.entry = replace(ctx.entry, runtime_kind=kind, category="3D Scene")
    driver = StudioManager.runtime_driver_for(manager, ctx.entry)
    monkeypatch.setattr(manager, "runtime_driver_for", lambda entry: driver)
    monkeypatch.setattr(driver, "load_pipeline", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(execution, "export_frames_to_video", lambda frames, path, **kwargs: path)
    Path(request.output_dir).mkdir(parents=True)

    class Geometry:
        builds = 0
        renders = 0

        def reconstruct_ply(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.builds += 1
            return {"camera_range": {}, "default_camera": {}, "ply_path": "scene.ply"}

        def render_interaction_video_with_3dgs(self, **kwargs: Any) -> list[Image.Image]:
            self.renders += 1
            return [Image.new("RGB", (8, 8))]

        def __call__(self, task_type: str, **kwargs: Any) -> Any:
            if task_type == "reconstruction":
                self.builds += 1
                return SimpleNamespace(save=lambda path: [], camera_range={})
            self.renders += 1
            return Image.new("RGB", (8, 8))

        def stream(self, interactions: Any = None) -> Image.Image:
            self.renders += 1
            return Image.new("RGB", (8, 8))

    pipeline = ctx.pipeline = Geometry()
    with pytest.raises(RuntimeError, match="no cached reconstruction"):
        call("ordinary", "stream")
    call("ordinary", "run", replace(request, input_path="source.png"))
    call("ordinary", "stream")
    assert (pipeline.builds, pipeline.renders) == (1, 2)
    call("ordinary", "reset")
    assert not ctx.state
