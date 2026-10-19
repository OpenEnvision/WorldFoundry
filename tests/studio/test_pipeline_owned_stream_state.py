from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PipelineContext,
    PreparedInputs,
    StudioManager,
    _prepared_inputs_from_payload,
    _prepared_inputs_payload,
)

pytestmark = pytest.mark.unit


class _OwnedMemory:
    def __init__(self) -> None:
        self.current_scene_context: dict[str, Any] | None = None

    def manage(self, action: str = "reset") -> None:
        assert action == "reset"
        self.current_scene_context = None

    def record(self, value: Any, metadata: Any = None) -> None:
        raise AssertionError("Studio must not seed or overwrite pipeline-owned scene state")


class _OwnedStatePipeline:
    def __init__(self, preview: str) -> None:
        self.memory_module = _OwnedMemory()
        self.preview = preview
        self.calls: list[dict[str, Any]] = []
        self.fail_init = False

    def stream_state_ready(self) -> bool:
        return self.memory_module.current_scene_context is not None

    def init_stream(
        self,
        images: Any = None,
        K: Any = None,
        meta_path: str | None = None,
        panorama_path: str | None = None,
        scene_name: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"mode": "init", "images": images, "K": K, "meta_path": meta_path, "output_dir": output_dir})
        if self.fail_init:
            raise ValueError("invalid scene source")
        self.memory_module.current_scene_context = {"scene_name": scene_name, "K": K}
        return {"scene_context": self.memory_module.current_scene_context, "preview_image": self.preview}

    def __call__(self, images: Any = None, interactions: Any = None, K: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"mode": "run", "images": images, "interactions": interactions, "K": K, **kwargs})
        return {"preview_image": self.preview}

    def stream(self, images: Any = None, interactions: Any = None, K: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"mode": "stream", "images": images, "interactions": interactions, "K": K, **kwargs})
        if not self.stream_state_ready():
            assert images is not None
            self.memory_module.current_scene_context = {"source": images}
        return {"scene_context": self.memory_module.current_scene_context, "preview_image": self.preview}


@pytest.fixture
def owned_context(tmp_path: Path) -> tuple[StudioManager, PipelineContext, PreparedInputs, _OwnedStatePipeline]:
    entry = replace(find_entry("worldfm"), default_call_kwargs={"return_dict": True})
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    request = manager.prepare_inputs(
        entry=entry,
        prompt="scene",
        input_path="",
        image=Image.new("RGB", (24, 12)),
        video=None,
        last_frame=None,
        reference_files=None,
        interactions_text='["forward", "camera_r"]',
        camera_view_text="",
        task_type="",
        intrinsics_text="",
        meta_path="",
        panorama_path="",
        scene_name="A",
        fps=8,
        num_frames=2,
        call_kwargs_text="{}",
        load_kwargs_text="{}",
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    assert request.image_path is not None
    pipeline = _OwnedStatePipeline(request.image_path)
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key="worldfm",
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
        state={},
    )
    return manager, ctx, request, pipeline


def test_generic_driver_delegates_init_without_generating_or_priming_memory(owned_context: Any) -> None:
    manager, ctx, request, pipeline = owned_context
    driver = manager.runtime_driver_for(ctx.entry)
    assert type(driver) is BaseRuntimeDriver
    assert driver.can_init(ctx, request)
    record = driver.run_init(manager, ctx, request)
    assert [call["mode"] for call in pipeline.calls] == ["init"]
    assert pipeline.calls[0]["images"] is request.image
    assert pipeline.calls[0]["K"] is None
    assert pipeline.calls[0]["output_dir"] == request.output_dir
    assert pipeline.stream_state_ready()
    assert ctx.state["studio_stream_initialized"]
    assert record.mode == "init" and record.status == "succeeded"
    assert record.preview_image == request.image_path
    assert Path(record.manifest_path).is_file()


def test_shared_driver_forwards_navigation_and_calibration_without_model_math(owned_context: Any) -> None:
    manager, ctx, request, pipeline = owned_context
    driver = manager.runtime_driver_for(ctx.entry)
    driver.run_fresh(manager, ctx, request)
    assert not pipeline.stream_state_ready()
    assert not ctx.state
    driver.run_continue(manager, ctx, request)
    assert [call["mode"] for call in pipeline.calls] == ["run", "stream"]
    for call in pipeline.calls:
        assert call["images"] is request.image
        assert call["interactions"] == ["forward", "camera_r"]
        assert call["K"] is None
        assert call["return_dict"] is True


def test_owned_reset_clears_state_before_next_stream(owned_context: Any) -> None:
    manager, ctx, request, pipeline = owned_context
    driver = manager.runtime_driver_for(ctx.entry)
    driver.run_init(manager, ctx, request)
    driver.reset(manager, ctx)
    assert not pipeline.stream_state_ready()
    assert not ctx.state
    driver.run_continue(manager, ctx, request)
    assert pipeline.calls[-1]["images"] is request.image
    assert pipeline.stream_state_ready()


def test_failed_owned_init_does_not_publish_ready_state_and_can_retry(owned_context: Any) -> None:
    manager, ctx, request, pipeline = owned_context
    driver = manager.runtime_driver_for(ctx.entry)
    pipeline.fail_init = True
    with pytest.raises(ValueError, match="invalid scene source"):
        driver.run_init(manager, ctx, request)
    assert not pipeline.stream_state_ready()
    assert not ctx.state
    assert [call["mode"] for call in pipeline.calls] == ["init"]
    pipeline.fail_init = False
    assert driver.run_init(manager, ctx, request).status == "succeeded"
    assert pipeline.stream_state_ready()


def test_shared_driver_keeps_scene_state_instance_local(owned_context: Any) -> None:
    manager, ctx, request, first = owned_context
    second = _OwnedStatePipeline(request.image_path)
    second_context = replace(ctx, pipeline=second, cache_key="second", state={})
    driver = manager.runtime_driver_for(ctx.entry)
    driver.run_init(manager, ctx, request)
    first_scene = first.memory_module.current_scene_context
    assert not second.stream_state_ready()
    driver.run_continue(manager, second_context, request)
    assert second.calls[0]["images"] is request.image
    assert second.memory_module.current_scene_context is not first_scene
    assert first.memory_module.current_scene_context is first_scene


def test_load_override_provenance_survives_worker_payload(owned_context: Any) -> None:
    _, _, request, _ = owned_context
    assert request.explicit_load_params == ()
    explicit = replace(request, load_kwargs={"dit_fsdp": True}, explicit_load_params=("dit_fsdp",))
    restored = _prepared_inputs_from_payload(_prepared_inputs_payload(explicit))
    assert restored.explicit_load_params == ("dit_fsdp",)
    assert restored.load_kwargs == {"dit_fsdp": True}
    direct = replace(explicit, explicit_load_params=None)
    assert _prepared_inputs_from_payload(_prepared_inputs_payload(direct)).explicit_load_params is None


def test_worldfm_catalog_requests_structured_output() -> None:
    entry = find_entry("worldfm")
    assert entry.default_call_kwargs["return_dict"] is True
    assert entry.module_path == "worldfoundry.pipelines.worldfm.pipeline_worldfm"
    assert entry.class_name == "WorldFMPipeline"
