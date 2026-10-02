from __future__ import annotations

import threading
from types import SimpleNamespace

from worldfoundry.pipelines.lingbot_world_v2.pipeline_lingbot_world_v2 import (
    LingBotWorldV2Pipeline,
)
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import (
    LingBotWorldV2RealtimeSession,
)


def test_lingbot_world_v2_prompt_boundary_keeps_video_state() -> None:
    session = object.__new__(LingBotWorldV2RealtimeSession)
    session.device = SimpleNamespace(type="cpu")
    session.configured = True
    session._state_lock = threading.RLock()
    session._state_ready_event = None
    session._prompt = "day"
    session._context = ["day context"]
    session._cross_attention_initialized = True

    self_cache = [{"video": object()}]
    cross_cache = [{"text": "day"}]
    vae_state = {"history": object()}
    generator = object()
    camera_pose = object()
    session._self_cache = self_cache
    session._cross_cache = cross_cache
    session._vae_state = vae_state
    session._generator = generator
    session._last_camera_pose = camera_pose
    session.autoregressive_index = 11

    replacement_cross_cache = [{"text": "empty"}]
    encoded: list[str] = []

    def encode_prompt(prompt: str) -> list[str]:
        encoded.append(prompt)
        return [f"context:{prompt}"]

    session._encode_prompt = encode_prompt
    session._new_cross_attention_cache = lambda: replacement_cross_cache

    assert session.update_prompt("night") is True
    assert encoded == ["night"]
    assert session._context == ["context:night"]
    assert session._cross_cache is replacement_cross_cache
    assert session._cross_attention_initialized is False
    assert session._self_cache is self_cache
    assert session._vae_state is vae_state
    assert session._generator is generator
    assert session._last_camera_pose is camera_pose
    assert session.autoregressive_index == 11


def test_lingbot_world_v2_pipeline_updates_prompt_before_chunk() -> None:
    calls: list[tuple[str, object]] = []

    class FakeSession:
        def update_prompt(self, prompt: str) -> None:
            calls.append(("prompt", prompt))

        def generate(self, **kwargs: object) -> dict[str, object]:
            calls.append(("generate", kwargs))
            return {"video": None}

    pipeline = object.__new__(LingBotWorldV2Pipeline)
    pipeline._realtime_session = FakeSession()

    pipeline.stream_realtime(
        prompt="heavy rain begins",
        interactions=["forward"],
        realtime_segments=[{"duration": 0.5, "keys": ["w"]}],
        seed=17,
    )

    assert calls == [
        ("prompt", "heavy rain begins"),
        (
            "generate",
            {
                "interactions": ["forward"],
                "control_segments": [{"duration": 0.5, "keys": ["w"]}],
                "seed": 17,
            },
        ),
    ]
