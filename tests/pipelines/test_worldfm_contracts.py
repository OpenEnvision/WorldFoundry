from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.pipelines.worldfm.pipeline_worldfm import WorldFMPipeline

pytestmark = pytest.mark.unit


class _Representation:
    def __init__(self) -> None:
        self.builds: list[dict[str, Any]] = []
        self.renders: list[dict[str, Any]] = []
        self.fail = False

    def build_scene_context(self, **kwargs: Any) -> dict[str, Any]:
        self.builds.append(kwargs)
        if self.fail:
            raise ValueError("scene reconstruction failed")
        return {"scene_name": kwargs["scene_name"], "K": kwargs["K"]}

    def get_representation(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.renders.append(inputs)
        return {
            **inputs,
            "scene_name": inputs["scene_context"]["scene_name"],
            "K": inputs["scene_context"]["K"],
            "rendered_conditions": [Image.new("RGB", (16, 16))],
        }


class _Synthesis:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"frames": [Image.new("RGB", (16, 16))]}


def _pipeline() -> Any:
    return WorldFMPipeline(
        representation_model=cast(Any, _Representation()),
        synthesis_model=cast(Any, _Synthesis()),
        device="cpu",
    )


def test_public_loader_delegates_to_canonical_components(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.worldfm import pipeline_worldfm as module

    representation, synthesis = _Representation(), _Synthesis()
    loaded: dict[str, dict[str, Any]] = {}

    def load_representation(**kwargs: Any) -> _Representation:
        loaded["representation"] = kwargs
        return representation

    def load_synthesis(**kwargs: Any) -> _Synthesis:
        loaded["synthesis"] = kwargs
        return synthesis

    monkeypatch.setattr(module.WorldFMRepresentation, "from_pretrained", load_representation)
    monkeypatch.setattr(module.WorldFMSynthesis, "from_pretrained", load_synthesis)
    pipeline = WorldFMPipeline.from_pretrained(
        model_path="local-worldfm",
        required_components={"num_views": 3, "step": 2},
        device="cpu",
        weight_dtype=torch.float32,
    )
    assert pipeline.representation_model is representation
    assert pipeline.synthesis_model is synthesis
    assert loaded["representation"]["num_views"] == 3
    assert loaded["synthesis"]["pretrained_model_path"] == "local-worldfm"
    assert loaded["synthesis"]["weight_dtype"] == torch.float32
    assert not pipeline.stream_state_ready()


def test_init_reconstructs_scene_without_rendering_or_generating() -> None:
    pipeline = _pipeline()
    initialized = pipeline.init_stream(images=Image.new("RGB", (24, 12)), scene_name="A")
    assert pipeline.stream_state_ready()
    assert pipeline.memory_module.select() is initialized["scene_context"]
    np.testing.assert_array_equal(initialized["scene_context"]["K"], [[24, 0, 12], [0, 24, 6], [0, 0, 1]])
    assert len(pipeline.representation_model.builds) == 1
    assert not pipeline.representation_model.renders
    assert not pipeline.synthesis_model.calls


def test_stream_reuses_owned_scene_and_accepts_navigation_without_studio() -> None:
    pipeline = _pipeline()
    initialized = pipeline.init_stream(images=Image.new("RGB", (24, 12)), scene_name="A")
    result = pipeline.stream(interactions=["forward"], return_dict=True)
    assert len(pipeline.representation_model.builds) == 1
    assert len(pipeline.synthesis_model.calls) == 1
    assert result["scene_context"] is initialized["scene_context"]
    assert pipeline.memory_module.select() is initialized["scene_context"]
    np.testing.assert_array_equal(result["c2w_list"][-1][:3, 3], [0.0, 0.0, -0.5])


def test_first_stream_builds_scene_and_preserves_explicit_poses() -> None:
    pipeline = _pipeline()
    pose = np.eye(4)
    pose[0, 3] = 2.0
    result = pipeline.stream(images=Image.new("RGB", (24, 12)), interactions=[pose], return_dict=True)
    assert pipeline.stream_state_ready()
    assert len(pipeline.representation_model.builds) == 1
    np.testing.assert_array_equal(result["c2w_list"][0], pose)
    assert len(pipeline.synthesis_model.calls) == 1


def test_failed_init_preserves_existing_scene_and_can_retry() -> None:
    pipeline = _pipeline()
    initialized = pipeline.init_stream(images=Image.new("RGB", (24, 12)), scene_name="A")
    pipeline.representation_model.fail = True
    with pytest.raises(ValueError, match="scene reconstruction failed"):
        pipeline.init_stream(images=Image.new("RGB", (24, 12)), scene_name="B")
    assert pipeline.memory_module.select() is initialized["scene_context"]
    assert pipeline.stream_state_ready()
    assert not pipeline.synthesis_model.calls
    pipeline.representation_model.fail = False
    retried = pipeline.init_stream(images=Image.new("RGB", (24, 12)), scene_name="B")
    assert pipeline.memory_module.select() is retried["scene_context"]
    assert retried["scene_context"]["scene_name"] == "B"


def test_scene_state_is_instance_local_across_a_b_a_and_reset() -> None:
    first, second = _pipeline(), _pipeline()
    first.init_stream(images=Image.new("RGB", (24, 12)), scene_name="A")
    second_context = second.init_stream(images=Image.new("RGB", (24, 12)), scene_name="B")["scene_context"]
    assert first.stream(interactions=["left"], return_dict=True)["scene_context"]["scene_name"] == "A"
    first.memory_module.manage(action="reset")
    assert not first.stream_state_ready()
    with pytest.raises(ValueError, match="No WorldFM scene context"):
        first.stream(interactions=["left"])
    first.init_stream(images=Image.new("RGB", (24, 12)), scene_name="A")
    assert first.stream(interactions=["right"], return_dict=True)["scene_context"]["scene_name"] == "A"
    assert second.memory_module.select() is second_context
    assert len(second.representation_model.builds) == 1


@pytest.mark.parametrize("distributed", [False, True])
def test_studio_realtime_initializes_real_scene_memory_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distributed: bool,
) -> None:
    from worldfoundry.studio.inference.catalog import find_entry
    from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = find_entry("worldfm")
    pipeline = _pipeline()
    ctx = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key="scene",
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
    request = PreparedInputs(
        prompt="scene",
        input_path="",
        image=Image.new("RGB", (24, 12)),
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
        scene_name="A",
        fps=8,
        num_frames=2,
        output_dir=str(tmp_path / "output"),
        output_path=str(tmp_path / "output" / "video.mp4"),
        call_kwargs={"return_dict": True},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )

    def invoke(action: str, inputs: PreparedInputs) -> Any:
        if distributed:
            return manager._run_local_torchrun_lingbot_fast(
                entry=entry,
                request=inputs,
                action=action,
                materialize=False,
                realtime=True,
            )
        return manager.run_realtime(entry=entry, request=inputs, action=action)

    invoke("configure", request)
    assert pipeline.stream_state_ready()
    assert len(pipeline.representation_model.builds) == 1
    assert not pipeline.synthesis_model.calls
    result = invoke("stream", replace(request, image=None))
    assert len(pipeline.representation_model.builds) == 1
    assert len(pipeline.synthesis_model.calls) == 1
    np.testing.assert_array_equal(result["c2w_list"][-1][:3, 3], [0.0, 0.0, -0.5])
    invoke("reset", request)
    assert not pipeline.stream_state_ready()
    assert ctx.active_leases == 0
