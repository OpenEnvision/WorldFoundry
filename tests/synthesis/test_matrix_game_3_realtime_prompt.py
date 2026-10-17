from __future__ import annotations

import pytest

from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.realtime import (
    MatrixGame3RealtimeSession,
)


class _TextEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], object]] = []
        self.fail_on: str | None = None

    def __call__(self, texts: list[str], *, device: object) -> object:
        self.calls.append((list(texts), device))
        if texts[0] == self.fail_on:
            raise RuntimeError("encode failed")
        return {"prompt": texts[0]}


def _configured_session() -> tuple[MatrixGame3RealtimeSession, _TextEncoder]:
    encoder = _TextEncoder()
    session = MatrixGame3RealtimeSession.__new__(MatrixGame3RealtimeSession)
    session._configured = True
    session.device = "cuda:0"
    session.core = type("Core", (), {"text_encoder": encoder})()
    session._prompt = "base prompt"
    session._prompt_context = {"prompt": "base prompt"}
    session._clip_index = 3
    session._latent_history = [object()]
    session._vae_cache = [object()]
    session._last_pose = object()
    return session, encoder


def test_matrix_game_3_prompt_update_preserves_rollout_state() -> None:
    session, encoder = _configured_session()
    latent_history = session._latent_history
    vae_cache = session._vae_cache
    last_pose = session._last_pose

    assert session.update_prompt("storm arrives") is True
    assert session._prompt == "storm arrives"
    assert session._prompt_context == {"prompt": "storm arrives"}
    assert encoder.calls == [(["storm arrives"], "cuda:0")]
    assert session._clip_index == 3
    assert session._latent_history is latent_history
    assert session._vae_cache is vae_cache
    assert session._last_pose is last_pose

    assert session.update_prompt("storm arrives") is False
    assert len(encoder.calls) == 1


def test_matrix_game_3_prompt_update_is_atomic_on_encoder_failure() -> None:
    session, encoder = _configured_session()
    previous_context = session._prompt_context
    encoder.fail_on = "bad prompt"

    with pytest.raises(RuntimeError, match="encode failed"):
        session.update_prompt("bad prompt")

    assert session._prompt == "base prompt"
    assert session._prompt_context is previous_context


def test_matrix_game_3_pipeline_updates_prompt_before_next_window() -> None:
    calls: list[tuple[str, object]] = []

    class _Session:
        def update_prompt(self, prompt: str) -> bool:
            calls.append(("prompt", prompt))
            return True

        def generate(self, *, interactions: list[str], control_segments: object) -> dict[str, bool]:
            calls.append(("generate", (interactions, control_segments)))
            return {"ok": True}

    pipeline = MatrixGame3Pipeline.__new__(MatrixGame3Pipeline)
    pipeline._realtime_session = _Session()
    segments = [{"duration": 0.25, "keys": ["a"]}]

    assert pipeline.stream_realtime(
        prompt="night falls",
        interactions=["left"],
        realtime_segments=segments,
    ) == {"ok": True}
    assert calls == [
        ("prompt", "night falls"),
        ("generate", (["left"], segments)),
    ]
