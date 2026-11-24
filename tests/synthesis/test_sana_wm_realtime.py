# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM realtime contracts through the public native request boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.pipelines.sana_wm import SanaWMPipeline, SanaWMStreamingPipeline
from worldfoundry.studio.inference.catalog import CURATED_OVERRIDES, find_entry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PipelineContext,
    StudioManager,
    _prepared_inputs_from_payload,
)
from worldfoundry.synthesis.visual_generation.sana_wm.realtime import (
    DEFAULT_CFG_SCALE,
    DEFAULT_REALTIME_WINDOW_FRAMES,
    DEFAULT_SAMPLING_STEPS,
    SanaWMRealtimeSession,
    _compress_action_frames,
    _frames_for_segments,
    _snap_num_frames,
    _validate_output_video,
)

pytestmark = pytest.mark.unit


class _NativeRunner:
    """Record normalized requests in place of checkpoint-backed execution."""

    def __init__(self) -> None:
        self.model_id = "sana-wm"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.requests: list[DiffusionRequest] = []
        self.fail_next = False

    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        self.requests.append(request)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("native execution failed")
        return DiffusionOutput(
            sample=torch.zeros(1, 3, request.num_frames, 2, 2),
            latents=torch.zeros(1),
        )


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> _NativeRunner:
    runner = _NativeRunner()

    def build(cls: type[NativeDiffusionPipeline], model_id: str, **kwargs: Any) -> NativeDiffusionPipeline:
        runner.model_id = model_id
        runner.device = kwargs["policy"].device
        runner.dtype = kwargs["policy"].dtype
        runner.loads.append((model_id, kwargs))
        return cls(runner)

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", classmethod(build))
    return runner


def _pipeline() -> SanaWMPipeline:
    return SanaWMPipeline.from_pretrained("/tmp/checkpoint", device="cpu", weight_dtype="float32")


def _session() -> SanaWMRealtimeSession:
    session = SanaWMRealtimeSession("/tmp/checkpoint", device="cpu", weight_dtype="float32")
    session.configure(image=Image.new("RGB", (32, 18)), prompt="old prompt")
    return session


def test_sana_wm_temporal_window_preserves_8k_plus_one_layout() -> None:
    assert _snap_num_frames(9) == 9
    assert _snap_num_frames(159) == 161
    assert _snap_num_frames(161) == 161


def test_sana_wm_maps_local_component_checkpoints_to_their_real_layout(tmp_path: Path) -> None:
    sana_root = tmp_path / "Efficient-Large-Model--SANA-WM_bidirectional"
    gemma_root = tmp_path / "Efficient-Large-Model--gemma-2-2b-it"
    assert SanaWMPipeline._checkpoint_overrides(str(sana_root), {}) == {
        "dit": str(sana_root),
        "text-encoder": str(gemma_root),
        "tokenizer": str(gemma_root),
        "codec": str(sana_root),
    }


def test_sana_wm_segments_cover_exact_output_window() -> None:
    frames = _frames_for_segments(
        [{"duration": 0.25, "keys": ["w"]}, {"duration": 0.25, "keys": ["l"]}],
        [],
        frame_count=8,
        fps=16,
    )
    assert frames == [frozenset("w")] * 4 + [frozenset("l")] * 4
    assert _compress_action_frames(frames) == "w-4,l-4"


@pytest.mark.parametrize("pipeline_type", [SanaWMPipeline, SanaWMStreamingPipeline])
@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_sana_wm_public_loader_delegates_to_native_policy(
    native: _NativeRunner, pipeline_type: type[SanaWMPipeline], device: str
) -> None:
    pipeline = pipeline_type.from_pretrained("/tmp/checkpoint", device=device, weight_dtype="float32")
    model_id, options = native.loads[0]
    assert model_id == pipeline_type.MODEL_ID
    assert options["policy"].device == torch.device(device)
    assert options["checkpoint_overrides"]["dit"] == "/tmp/checkpoint"
    assert pipeline.native_pipeline.runner is native
    assert pipeline._realtime_config is None
    assert not native.requests


def test_sana_wm_quality_defaults_use_short_native_window_and_full_sampler() -> None:
    assert DEFAULT_REALTIME_WINDOW_FRAMES == 81
    assert (DEFAULT_REALTIME_WINDOW_FRAMES - 1) % 8 == 0
    assert DEFAULT_SAMPLING_STEPS == 60
    assert DEFAULT_CFG_SCALE == 5.0


def test_sana_wm_catalog_has_no_baked_image_or_prompt_fixture() -> None:
    override = CURATED_OVERRIDES["sana-wm"]
    assert "default_input_path" not in override
    assert override["default_prompt"] == ""
    assert override["default_call_kwargs"]["window_frames"] == 81
    entry = find_entry("sana-wm")
    assert entry.default_prompt == ""
    assert entry.default_input_path == ""


def test_sana_wm_requires_a_user_image_before_native_execution(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="requires a PIL image"):
        pipeline.configure_realtime(images=None, prompt="user prompt")
    assert pipeline._realtime_config is None
    assert not native.requests


def test_sana_wm_requires_a_user_prompt_before_native_execution(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="requires a prompt"):
        pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="  ")
    assert pipeline._realtime_config is None
    assert not native.requests


def test_sana_wm_forwards_user_window_and_quality_settings(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    image = Image.new("RGB", (32, 18))
    configured = pipeline.configure_realtime(
        images=image, prompt="A user-authored world prompt", window_frames=161, fps=12, step=60, cfg_scale=5.0, seed=7
    )
    spec = RealtimeSpec.from_payload(configured)
    assert (spec.fps, spec.first_chunk_frames, spec.steady_chunk_frames) == (12, 161, 161)
    assert (spec.resolved_first_input_frames, spec.resolved_steady_input_frames) == (160, 160)
    assert not native.requests
    image.putpixel((0, 0), (255, 0, 0))
    result = pipeline.stream_realtime(interactions=["w"] * 160)
    request = native.requests[0]
    assert request.num_frames == 161
    assert request.inputs["fps"] == 12
    configured_image = request.inputs["images"]
    assert isinstance(configured_image, Image.Image)
    assert configured_image is not image
    assert configured_image.getpixel((0, 0)) == (0, 0, 0)
    assert request.sampling.num_inference_steps == 60
    assert request.sampling.guidance_scale == 5.0
    assert request.sampling.seed == 7
    assert result["video"].shape == (1, 3, 161, 2, 2)


def test_sana_wm_rejects_mismatched_native_frame_count() -> None:
    with pytest.raises(RuntimeError, match="returned 79 frames; expected 80"):
        _validate_output_video(np.empty((79, 4, 4, 3), dtype=np.uint8), expected_frames=80)


def test_sana_wm_prompt_update_reconditions_without_reconfiguring_the_session(native: _NativeRunner) -> None:
    session = _session()
    config = session.pipeline._realtime_config
    assert session.update_prompt("  a rainy night  ") is True
    assert session.update_prompt("a rainy night") is False
    assert session.update_prompt("  ") is False
    assert session.pipeline._realtime_config is config
    assert not native.requests
    session.generate(interactions=["forward"], seed=9)
    assert native.requests[0].prompt == "a rainy night"
    assert len(native.loads) == 1


def test_sana_wm_generate_commits_prompt_before_the_same_chunk(native: _NativeRunner) -> None:
    session = _session()
    result = session.generate(prompt="a rainy night", interactions=["forward"], seed=9)
    request = native.requests[0]
    assert request.prompt == "a rainy night"
    assert request.inputs["interactions"] == ["w"] * 80
    assert request.sampling.seed == 9
    assert result["frames"].shape == (1, 3, 81, 2, 2)
    assert result["realtime_metrics"] == {}


def test_sana_wm_native_failure_preserves_the_previous_prompt_and_allows_reuse(native: _NativeRunner) -> None:
    session = _session()
    native.fail_next = True
    with pytest.raises(RuntimeError, match="native execution failed"):
        session.generate(prompt="a rainy night", interactions=["forward"], seed=9)
    assert session._prompt == "old prompt"
    assert session.pipeline._realtime_config is not None
    assert session.pipeline._realtime_config["prompt"] == "old prompt"
    session.generate(interactions=["backward"], seed=10)
    assert native.requests[-1].prompt == "old prompt"
    assert native.requests[-1].inputs["interactions"] == ["s"] * 80
    assert native.requests[0].inputs["images"] is native.requests[-1].inputs["images"]
    assert len(native.loads) == 1


def test_sana_wm_pipeline_maps_studio_segments_to_native_frame_actions(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="old prompt", window_frames=9)
    pipeline.stream_realtime(
        prompt="a rainy night",
        interactions=["camera_r"],
        realtime_segments=[{"duration": 0.25, "keys": ["w"]}, {"duration": 0.25, "keys": ["l"]}],
        seed=9,
    )
    assert native.requests[0].prompt == "a rainy night"
    assert native.requests[0].inputs["interactions"] == ["w"] * 4 + ["l"] * 4
    assert native.requests[0].sampling.seed == 9


def test_sana_wm_pipeline_keeps_the_last_successful_prompt(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="old prompt", window_frames=9)
    pipeline.stream_realtime(prompt="a rainy night", interactions=["w"] * 8)
    native.fail_next = True
    with pytest.raises(RuntimeError, match="native execution failed"):
        pipeline.stream_realtime(prompt="a snowy night", interactions=["s"] * 8)
    pipeline.stream_realtime(interactions=["l"] * 8)
    assert [request.prompt for request in native.requests] == ["a rainy night", "a snowy night", "a rainy night"]


def test_sana_wm_session_forwards_segments_and_reset_requires_configuration(native: _NativeRunner) -> None:
    session = _session()
    session.generate(control_segments=[{"duration": 1, "keys": ["w"]}, {"duration": 1, "keys": ["l"]}])
    assert native.requests[-1].inputs["interactions"] == ["w"] * 40 + ["l"] * 40
    session.reset()
    with pytest.raises(RuntimeError, match="not configured"):
        session.generate()
    assert len(native.requests) == 1
    session.configure(image=Image.new("RGB", (32, 18)), prompt="a new world")
    session.generate()
    assert native.requests[-1].prompt == "a new world"
    assert native.requests[-1].inputs["interactions"] == ["none"] * 80
    assert len(native.loads) == 1


def test_sana_wm_failed_reconfiguration_keeps_the_active_session(native: _NativeRunner) -> None:
    session = _session()
    config = session.pipeline._realtime_config
    with pytest.raises(ValueError, match="requires a PIL image"):
        session.configure(image=None, prompt="a new world")
    with pytest.raises(ValueError, match="fps"):
        session.pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="a new world", fps=0)
    assert session.pipeline._realtime_config is config
    assert session._prompt == "old prompt"
    assert not native.requests
    session.generate()
    assert native.requests[-1].prompt == "old prompt"


def test_sana_wm_per_call_window_keeps_sampling_and_output_spec_aligned(native: _NativeRunner) -> None:
    pipeline = _pipeline()
    pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="a world", window_frames=81)
    result = pipeline.stream_realtime(window_frames=9, fps=12, realtime_segments=[{"duration": 1, "keys": ["w"]}])
    spec = RealtimeSpec.from_payload(result)
    assert (spec.fps, spec.first_chunk_frames, spec.resolved_first_input_frames) == (12, 9, 8)
    assert native.requests[-1].num_frames == 9
    assert native.requests[-1].inputs["interactions"] == ["w"] * 8
    following = pipeline.stream_realtime(interactions=[])
    assert RealtimeSpec.from_payload(following).first_chunk_frames == 81
    assert native.requests[-1].num_frames == 81


def test_sana_wm_a_b_a_requests_do_not_reuse_the_other_chunks_controls(native: _NativeRunner) -> None:
    session = _session()
    session.generate(prompt="a world", interactions=["forward"], seed=7)
    session.generate(prompt="another world", interactions=["backward"], seed=8)
    session.generate(prompt="a world", interactions=["forward"], seed=7)
    first, middle, last = native.requests
    assert first.prompt == last.prompt == "a world"
    assert middle.prompt == "another world"
    assert first.inputs["interactions"] == last.inputs["interactions"] == ["w"] * 80
    assert middle.inputs["interactions"] == ["s"] * 80
    assert first.sampling == last.sampling


def test_sana_wm_studio_dispatch_uses_catalog_defaults_and_sampled_segments(
    native: _NativeRunner, tmp_path: Path
) -> None:
    entry = find_entry("sana-wm")
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 18)).save(image_path)
    request = _prepared_inputs_from_payload(
        {
            "prompt": "a user-authored world",
            "image_path": str(image_path),
            "call_kwargs": entry.default_call_kwargs,
            "device": "cpu",
        }
    )
    context = PipelineContext(
        entry=entry,
        pipeline=_pipeline(),
        cache_key="sana-wm",
        backend="from_pretrained",
        model_ref="/tmp/checkpoint",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    manager = object.__new__(StudioManager)
    driver = BaseRuntimeDriver()
    handled, configured = manager._run_native_realtime_pipeline_action(
        driver=driver, ctx=context, request=request, action="configure"
    )
    assert handled
    assert RealtimeSpec.from_payload(configured).resolved_first_input_frames == 80
    assert not native.requests
    request.call_kwargs["realtime_segments"] = [
        {"duration": 2.5, "keys": ["w"]},
        {"duration": 2.5, "keys": ["l"]},
    ]
    handled, result = manager._run_native_realtime_pipeline_action(
        driver=driver, ctx=context, request=request, action="stream"
    )
    assert handled
    assert result["video"].shape == (1, 3, 81, 2, 2)
    assert native.requests[0].inputs["interactions"] == ["w"] * 40 + ["l"] * 40
    assert native.requests[0].prompt == request.prompt
    assert native.requests[0].sampling.num_inference_steps == 60
    assert native.requests[0].sampling.guidance_scale == 5.0
    assert len(native.loads) == 1
