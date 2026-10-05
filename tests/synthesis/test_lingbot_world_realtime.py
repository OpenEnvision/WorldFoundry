from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch.distributed as dist

from worldfoundry.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import (
    LingBotRealtimeSession,
    RealtimeCameraState,
)


class _NoVideoSynthesis:
    def predict(self, **_kwargs: object) -> None:
        return None


def _no_video_pipeline() -> LingBotPipeline:
    pipeline = object.__new__(LingBotPipeline)
    pipeline.model_id = "lingbot-world"
    pipeline.synthesis_model = _NoVideoSynthesis()
    pipeline.process = lambda **_kwargs: {
        "prompt": "world",
        "pil_image": object(),
        "action_path": None,
        "c2ws": None,
        "Ks": None,
    }
    return pipeline


def test_lingbot_pipeline_accepts_empty_nonzero_rank_result(monkeypatch) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 2)

    assert _no_video_pipeline()(return_dict=True) is None


def test_lingbot_pipeline_rejects_empty_rank_zero_result(monkeypatch) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="did not return video frames"):
        _no_video_pipeline()(return_dict=True)


def _one_frame_pose(*keys: str) -> np.ndarray:
    camera = RealtimeCameraState(
        move_speed_per_second=0.8,
        rotate_speed_radians_per_second=float(np.deg2rad(32.0)),
    )
    return camera.integrate(
        [{"duration": 1.0 / 16.0, "keys": keys}],
        num_frames=1,
        fps=16,
    )[0]


def test_lingbot_realtime_ad_strafes_without_rotating() -> None:
    left = _one_frame_pose("a")
    right = _one_frame_pose("d")

    np.testing.assert_allclose(left[:3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(right[:3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(left[:3, 3], [-0.05, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(right[:3, 3], [0.05, 0.0, 0.0], atol=1e-7)


def test_lingbot_realtime_jl_rotates_without_strafing() -> None:
    yaw_left = _one_frame_pose("j")
    yaw_right = _one_frame_pose("l")

    np.testing.assert_allclose(yaw_left[:3, 3], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(yaw_right[:3, 3], np.zeros(3), atol=1e-7)
    assert yaw_left[0, 2] < 0.0
    assert yaw_right[0, 2] > 0.0


def test_lingbot_realtime_combines_forward_and_strafe() -> None:
    forward_right = _one_frame_pose("w", "d")

    np.testing.assert_allclose(
        forward_right[:3, 3],
        [0.05, 0.0, 0.05],
        atol=1e-7,
    )


def test_lingbot_prompt_update_rebuilds_only_cross_attention_state() -> None:
    session = object.__new__(LingBotRealtimeSession)
    session.device = SimpleNamespace(type="cpu")
    session.configured = True
    session._state_lock = threading.RLock()
    session._state_ready_event = None
    session._prompt = "old prompt"
    session._context = ["old context"]
    old_self_cache = [{"video": "kv"}]
    old_cross_cache = [{"text": "old kv"}]
    old_vae_state = {"history": object()}
    old_generator = object()
    session._self_cache = old_self_cache
    session._cross_cache = old_cross_cache
    session._vae_state = old_vae_state
    session._generator = old_generator
    session._cross_attention_initialized = True
    session.autoregressive_index = 7

    encoded_prompts: list[str] = []
    new_cross_cache = [{"text": "empty kv"}]

    def encode_prompt(prompt: str) -> list[str]:
        encoded_prompts.append(prompt)
        return [f"context:{prompt}"]

    session._encode_prompt = encode_prompt
    session._new_cross_attention_cache = lambda: new_cross_cache

    assert session.update_prompt("new prompt") is True
    assert encoded_prompts == ["new prompt"]
    assert session._prompt == "new prompt"
    assert session._context == ["context:new prompt"]
    assert session._cross_cache is new_cross_cache
    assert session._cross_attention_initialized is False
    assert session._self_cache is old_self_cache
    assert session._vae_state is old_vae_state
    assert session._generator is old_generator
    assert session.autoregressive_index == 7

    assert session.update_prompt("new prompt") is False
    assert encoded_prompts == ["new prompt"]


def test_lingbot_pipeline_applies_runtime_prompt_before_next_chunk() -> None:
    calls: list[tuple[str, object]] = []

    class FakeSession:
        def update_prompt(self, prompt: str) -> None:
            calls.append(("prompt", prompt))

        def generate(self, **kwargs: object) -> dict[str, object]:
            calls.append(("generate", kwargs))
            return {"video": None}

    pipeline = object.__new__(LingBotPipeline)
    pipeline._realtime_session = FakeSession()
    pipeline.memory_module = SimpleNamespace(record=lambda *args, **kwargs: None)

    pipeline.stream_realtime(
        prompt="a rainy night",
        interactions=["forward"],
        realtime_segments=[{"duration": 0.25, "keys": ["w"]}],
        seed=9,
    )

    assert calls[0] == ("prompt", "a rainy night")
    assert calls[1][0] == "generate"
    assert calls[1][1] == {
        "interactions": ["forward"],
        "control_segments": [{"duration": 0.25, "keys": ["w"]}],
        "seed": 9,
    }
