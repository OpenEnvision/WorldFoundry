from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime import (
    realtime as realtime_module,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
    MatrixGame2RealtimeSession,
)


class _FakeClip:
    def __init__(self) -> None:
        self.calls = 0

    def encode_video(self, image: Any) -> torch.Tensor:
        self.calls += 1
        assert image == "perception-image"
        return torch.full((1, 2, 4), 0.25, dtype=torch.float32)


class _FakeVae:
    def __init__(self) -> None:
        self.clip = _FakeClip()
        self.encode_calls = 0

    def encode(self, img_cond: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.encode_calls += 1
        assert img_cond.shape == (1, 3, 57, 4, 5)
        assert kwargs["device"] == "cpu"
        assert kwargs["tile_size"] == (4, 5)
        return torch.arange(15, dtype=torch.float32).view(1, 1, 15, 1, 1).expand(
            1, 16, 15, 4, 5
        )


class _FakeOperator:
    def __init__(self) -> None:
        self.calls = 0

    def process_perception(
        self,
        image: Image.Image,
        num_frames: int,
        height: int,
        width: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        assert image.mode == "RGB"
        assert (num_frames, height, width) == (15, 352, 640)
        assert kwargs == {"device": "cpu", "weight_dtype": torch.float32}
        return {
            "img_cond": torch.zeros((1, 3, 57, 4, 5), dtype=torch.float32),
            "image": "perception-image",
            "tiler_kwargs": {"tile_size": (4, 5)},
        }


class _FakeCausalCore:
    num_frame_per_block = 3

    def __init__(self) -> None:
        self._session: SimpleNamespace | None = None
        self._last_block: SimpleNamespace | None = None
        self.start_conditions: list[dict[str, Any]] = []
        self.step_conditions: list[dict[str, Any]] = []
        self.reset_calls = 0

    def start_session(self, condition: dict[str, Any], *, mode: str) -> None:
        assert mode == "universal"
        self.start_conditions.append(condition)
        self._session = SimpleNamespace(current_start_frame=0)

    def step_session(
        self,
        noise: torch.Tensor,
        condition: dict[str, Any],
        *,
        mode: str,
    ) -> torch.Tensor:
        assert mode == "universal"
        assert noise.shape == (1, 16, 3, 4, 5)
        assert self._session is not None
        output_frames = 9 if self._session.current_start_frame == 0 else 12
        self.step_conditions.append(condition)
        self._session.current_start_frame += 3
        self._last_block = SimpleNamespace(model_ms=12.5, decode_ms=3.25)
        return torch.zeros((1, output_frames, 3, 4, 5), dtype=torch.float32)

    def reset_session(self) -> None:
        self.reset_calls += 1
        self._session = None
        self._last_block = None


class _FakeRuntime:
    def __init__(self, core: _FakeCausalCore) -> None:
        self.pipeline = core
        self.vae = _FakeVae()
        self.device = "cpu"
        self.weight_dtype = torch.float32
        self.mode = "universal"


def test_vae_zero_cache_template_import_is_allocation_free(monkeypatch: pytest.MonkeyPatch) -> None:
    constant = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.matrix_game."
        "matrix_game_2_runtime.utils.vae_runtime.constant"
    )
    allocations: list[tuple[Any, ...]] = []

    def record_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        allocations.append(args)
        return torch.empty(0)

    monkeypatch.setattr(torch, "zeros", record_zeros)
    constant = importlib.reload(constant)

    assert len(constant.VAE_CACHE_SHAPES) == 32
    assert len(constant.ZERO_VAE_CACHE) == 32
    assert allocations == []
    assert vars(constant.ZERO_VAE_CACHE) == {}
    assert not any(
        isinstance(value, torch.Tensor)
        for value in vars(constant.ZERO_VAE_CACHE).values()
    )


def test_realtime_session_keeps_causal_state_and_native_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_actions: list[tuple[str, ...]] = []

    def fake_encode_actions(actions: list[str], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert mode == "universal"
        encoded_actions.append(tuple(actions))
        keyboard = torch.tensor(
            [float("forward" in actions), float("back" in actions), 0.0, 0.0]
        )
        mouse = torch.tensor(
            [0.0, -0.1 if "camera_l" in actions else 0.0], dtype=torch.float32
        )
        return keyboard, mouse

    monkeypatch.setattr(realtime_module, "encode_actions", fake_encode_actions)
    monkeypatch.setattr(realtime_module, "set_seed", lambda _seed: None)
    monkeypatch.delenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", raising=False)

    core = _FakeCausalCore()
    runtime = _FakeRuntime(core)
    operator = _FakeOperator()
    session = MatrixGame2RealtimeSession(runtime, operator)

    assert session.next_output_frames() == 9
    configured = session.configure(Image.new("RGB", (8, 8)), seed=7)

    assert configured["status"] == "configured"
    assert configured["realtime_spec"] == {
        "fps": 12,
        "first_chunk_frames": 9,
        "steady_chunk_frames": 12,
        "controls": [
            "forward",
            "backward",
            "left",
            "right",
            "camera_up",
            "camera_down",
            "camera_l",
            "camera_r",
        ],
        "transport": "in-memory-rgb",
        "stateful": True,
    }
    assert set(configured["realtime_metrics"]) == {
        "perception_ms",
        "vae_encode_ms",
        "clip_encode_ms",
        "cache_init_ms",
        "decoder_warmup_ms",
        "total_ms",
    }
    assert core.reset_calls == 1
    assert len(core.start_conditions) == 1
    assert core.start_conditions[0]["_cond_concat_start_frame"] == 0
    assert core.start_conditions[0]["cond_concat"].shape == (1, 20, 3, 4, 5)
    assert runtime.vae.encode_calls == 1
    assert runtime.vae.clip.calls == 1

    first = session.generate(interactions=["forward", "camera_left"])
    second = session.generate(interactions=["backward"])

    assert first["video"].shape == (9, 4, 5, 3)
    assert second["video"].shape == (12, 4, 5, 3)
    assert first["video"].dtype == np.uint8
    assert first["video"].flags.c_contiguous
    assert np.all(first["video"] == 127)
    assert first["realtime_metrics"]["model_ms"] == 12.5
    assert first["realtime_metrics"]["decode_ms"] == 3.25
    assert first["realtime_metrics"]["condition_ms"] >= 0.0

    assert len(core.step_conditions) == 2
    assert [item["_cond_concat_start_frame"] for item in core.step_conditions] == [0, 3]
    assert core.step_conditions[0]["keyboard_cond"].shape == (1, 9, 4)
    assert core.step_conditions[1]["keyboard_cond"].shape == (1, 21, 4)
    assert core.step_conditions[0]["mouse_cond"].shape == (1, 9, 2)
    assert core.step_conditions[1]["mouse_cond"].shape == (1, 21, 2)
    assert core.step_conditions[0]["cond_concat"].shape == (1, 20, 3, 4, 5)
    assert core.step_conditions[1]["cond_concat"].shape == (1, 20, 3, 4, 5)
    assert torch.equal(
        core.step_conditions[0]["cond_concat"], session._condition_concat[:, :, 0:3]
    )
    assert torch.equal(
        core.step_conditions[1]["cond_concat"], session._condition_concat[:, :, 3:6]
    )
    assert encoded_actions[:9] == [("forward", "camera_l")] * 9
    assert encoded_actions[9:] == [("back",)] * 12
    assert session.next_output_frames() == 12

    session.reset()

    assert core.reset_calls == 2
    assert core._session is None
    assert session.configured is False
    assert session.configure_metrics == {}
    assert session._visual_context is None
    assert session._condition_concat is None
    assert session._keyboard_history is None
    assert session._mouse_history is None
    assert session._noise_generator is None
    assert session.next_output_frames() == 9
