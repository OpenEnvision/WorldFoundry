from __future__ import annotations

import pytest

from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline
from worldfoundry.pipelines.wan.pipeline_wan_vace import Wan2p1VACEPipeline


def test_vace_merges_workspace_canonical_sampling_fields(monkeypatch) -> None:
    captured = {}

    def fake_call(self, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(NativeTextToVideoPipeline, "__call__", fake_call)
    pipeline = object.__new__(Wan2p1VACEPipeline)

    result = pipeline(
        num_frames=17,
        num_inference_steps=4,
        guidance_scale=3.5,
        shift=2.0,
        seed=9,
    )

    assert result["num_frames"] == 17
    assert result["num_inference_steps"] == 4
    assert result["guidance_scale"] == 3.5
    assert result["shift"] == 2.0
    assert result["seed"] == 9
    assert captured == result


def test_vace_rejects_conflicting_alias_and_canonical_value(monkeypatch) -> None:
    monkeypatch.setattr(NativeTextToVideoPipeline, "__call__", lambda self, **kwargs: kwargs)
    pipeline = object.__new__(Wan2p1VACEPipeline)

    with pytest.raises(ValueError, match="frame_num conflicts with explicit num_frames"):
        pipeline(frame_num=17, num_frames=33)
