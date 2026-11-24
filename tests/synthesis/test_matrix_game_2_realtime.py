from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import DEFAULT_REALTIME_KEYS, RealtimeAxis
from worldfoundry.operators.matrix_game_2_operator import encode_actions
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime import (
    realtime as realtime_module,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
    MatrixGame2RealtimeSession,
)

_AXIS_NAMES = ("camera_pitch", "camera_yaw", "camera_pitch_stick", "camera_yaw_stick")


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
        return torch.arange(15, dtype=torch.float32).view(1, 1, 15, 1, 1).expand(1, 16, 15, 4, 5)


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
        "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference"
    )
    allocations: list[tuple[Any, ...]] = []

    def record_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        allocations.append(args)
        return torch.empty(0)

    monkeypatch.setattr(torch, "zeros", record_zeros)
    constant = importlib.reload(constant)

    assert constant.VAE_CACHE_SLOTS == 32
    assert allocations == []
    assert not hasattr(constant, "ZERO_VAE_CACHE")
    assert not any(isinstance(value, torch.Tensor) for value in vars(constant).values())


def test_realtime_session_keeps_causal_state_and_native_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_actions: list[tuple[str, ...]] = []

    def fake_encode_actions(actions: list[str], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert mode == "universal"
        encoded_actions.append(tuple(actions))
        keyboard = torch.tensor([float("forward" in actions), float("back" in actions), 0.0, 0.0])
        mouse = torch.tensor([0.0, -0.1 if "camera_l" in actions else 0.0], dtype=torch.float32)
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
        "axes": [
            {
                "name": "camera_pitch",
                "source": "mouse_y",
                "mode": "delta",
                "scale": -0.0025,
                "limit": 0.1,
                "deadzone": 0.0,
            },
            {
                "name": "camera_yaw",
                "source": "mouse_x",
                "mode": "delta",
                "scale": 0.0025,
                "limit": 0.1,
                "deadzone": 0.0,
            },
            {"name": "camera_pitch_stick", "source": "gamepad_ry", "mode": "level", "scale": -0.1, "deadzone": 0.15},
            {"name": "camera_yaw_stick", "source": "gamepad_rx", "mode": "level", "scale": 0.1, "deadzone": 0.15},
        ],
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
    for result, count in ((first, 9), (second, 12)):
        chunk = result["video_chunk"]
        assert chunk.frames is result["video"]
        assert chunk.layout == "thwc"
        assert chunk.value_range == "0,255"
        assert chunk.frame_count == count
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
    assert session._condition_concat is not None
    assert torch.equal(core.step_conditions[0]["cond_concat"], session._condition_concat[:, :, 0:3])
    assert torch.equal(core.step_conditions[1]["cond_concat"], session._condition_concat[:, :, 3:6])
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


def _condition_session(mode: str) -> MatrixGame2RealtimeSession:
    """Session holding only the state that action encoding reads."""

    session = object.__new__(MatrixGame2RealtimeSession)
    session.runtime = SimpleNamespace(mode=mode)
    session.device = torch.device("cpu")
    session.dtype = torch.float32
    session._keyboard_history = None
    session._mouse_history = None
    return session


def _encode_frames(
    session: MatrixGame2RealtimeSession,
    frame_keys: list[list[str]],
    realtime_axes: dict[str, list[float]] | None = None,
) -> None:
    """Append one chunk whose frame ``i`` holds ``frame_keys[i]``."""

    segments = [{"duration": 1.0, "keys": keys} for keys in frame_keys]
    num_frames = len(frame_keys)
    session._append_actions(
        session._actions_for_frames([], segments, num_frames=num_frames),
        session._camera_offsets(realtime_axes, num_frames=num_frames),
    )


def test_universal_spec_keeps_default_keys_and_declares_camera_axes() -> None:
    spec = _condition_session("universal").realtime_spec()

    assert spec.keys == DEFAULT_REALTIME_KEYS
    assert spec.axes == (
        RealtimeAxis("camera_pitch", "mouse_y", "delta", scale=-0.0025, limit=0.1),
        RealtimeAxis("camera_yaw", "mouse_x", "delta", scale=0.0025, limit=0.1),
        RealtimeAxis("camera_pitch_stick", "gamepad_ry", "level", scale=-0.1, deadzone=0.15),
        RealtimeAxis("camera_yaw_stick", "gamepad_rx", "level", scale=0.1, deadzone=0.15),
    )


def test_universal_keys_without_axis_input_encode_the_key_conditions() -> None:
    frame_keys = [["j", "w"], ["i"], [], ["l", "s"], ["a", "k"]]
    actions = [["camera_l", "forward"], ["camera_up"], [], ["camera_r", "back"], ["left", "camera_down"]]
    expected = [encode_actions(frame_actions, "universal") for frame_actions in actions]
    zero_axes = {name: [0.0] * len(frame_keys) for name in _AXIS_NAMES}

    for realtime_axes in (None, zero_axes):
        session = _condition_session("universal")
        _encode_frames(session, frame_keys, realtime_axes)
        expected_keyboard = torch.stack([keyboard for keyboard, _ in expected])
        torch.testing.assert_close(session._keyboard_history, expected_keyboard, rtol=0, atol=0)
        torch.testing.assert_close(
            session._mouse_history, torch.stack([mouse for _, mouse in expected]), rtol=0, atol=0
        )


def test_universal_axes_add_to_key_camera_and_clamp_each_component() -> None:
    session = _condition_session("universal")
    _encode_frames(
        session,
        [["l"], ["l"], [], ["i"], ["k"]],
        {
            "camera_pitch": [0.0, 0.0, 0.0, -0.05, -0.08],
            "camera_yaw": [0.05, -0.15, 0.03, 0.0, 0.0],
            "camera_pitch_stick": [0.0, 0.0, 0.0, -0.02, -0.08],
            "camera_yaw_stick": [0.0, 0.0, 0.04, 0.0, 0.0],
        },
    )

    torch.testing.assert_close(
        session._mouse_history,
        torch.tensor([[0.0, 0.1], [0.0, -0.05], [0.0, 0.07], [0.03, 0.0], [-0.1, 0.0]]),
    )


def test_gta_drive_keys_match_the_official_keymap() -> None:
    session = _condition_session("gta_drive")
    spec = session.realtime_spec()
    assert [(binding.key, binding.action) for binding in spec.keys] == [
        ("w", "forward"),
        ("s", "back"),
        ("a", "camera_l"),
        ("d", "camera_r"),
    ]
    assert spec.keys[0].group == spec.keys[1].group != spec.keys[2].group == spec.keys[3].group
    assert [axis.name for axis in spec.axes] == ["camera_yaw_stick"]

    _encode_frames(session, [["w"], ["a", "s"], ["d"], []], {"camera_yaw_stick": [0.0, 0.0, 0.0, 0.05]})

    torch.testing.assert_close(
        session._keyboard_history,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        session._mouse_history,
        torch.tensor([[0.0, 0.0], [0.0, -0.1], [0.0, 0.1], [0.0, 0.05]]),
    )


def test_templerun_keys_match_the_official_one_hot_and_idle_is_nomove() -> None:
    session = _condition_session("templerun")
    spec = session.realtime_spec()
    assert [(binding.key, binding.action) for binding in spec.keys] == [
        ("w", "jump"),
        ("s", "slide"),
        ("a", "leftside"),
        ("d", "rightside"),
        ("z", "turnleft"),
        ("c", "turnright"),
    ]
    assert len({binding.group for binding in spec.keys}) == 1
    assert spec.axes == ()

    _encode_frames(session, [["w"], ["s"], ["a"], ["d"], ["z"], ["c"], []])

    # Rows of the official ``get_current_action`` table for W, S, A, D, Z, C and Q.
    torch.testing.assert_close(
        session._keyboard_history,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        rtol=0,
        atol=0,
    )
    assert session._mouse_history is None
    assert torch.count_nonzero(encode_actions([], "templerun")[0]) == 0


def test_realtime_session_applies_axes_and_rejects_axis_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime_module, "set_seed", lambda _seed: None)
    monkeypatch.delenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", raising=False)
    core = _FakeCausalCore()
    session = MatrixGame2RealtimeSession(_FakeRuntime(core), _FakeOperator())
    session.configure(Image.new("RGB", (8, 8)), seed=7)

    with pytest.raises(ValueError, match="'camera_yaw' has 12 values; the chunk has 9 output frames"):
        session.generate(interactions=[], realtime_axes={"camera_yaw": [0.01] * 12})
    assert session._keyboard_history is None
    assert core.step_conditions == []

    session.generate(interactions=[], realtime_axes={"camera_yaw": [0.01] * 9})

    torch.testing.assert_close(core.step_conditions[0]["mouse_cond"][0], torch.tensor([[0.0, 0.01]] * 9))


def test_pipeline_stream_realtime_forwards_axes_to_the_session() -> None:
    calls: list[dict[str, Any]] = []

    class RecordingSession:
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {}

    pipeline = MatrixGame2Pipeline(device="cpu", weight_dtype=torch.float32)
    pipeline._realtime_session = RecordingSession()
    segments = [{"duration": 0.75, "keys": ["w"]}]
    axes = {"camera_yaw": [0.01] * 9}

    pipeline.stream_realtime(["forward"], realtime_segments=segments, realtime_axes=axes, seed=3, prompt="")
    pipeline.stream_realtime(["forward"], realtime_segments=segments)

    assert calls == [
        {"interactions": ["forward"], "control_segments": segments, "realtime_axes": axes},
        {"interactions": ["forward"], "control_segments": segments, "realtime_axes": None},
    ]
