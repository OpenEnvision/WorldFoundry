from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, NoReturn, cast

import numpy as np
import pytest

from worldfoundry.core.execution.realtime.contracts import (
    DEFAULT_REALTIME_KEYS,
    RealtimeAxis,
    RealtimeKey,
    RealtimeSpec,
)
from worldfoundry.studio.serving.realtime.input import (
    ControlSegment,
    RealtimeControlResampler,
    axis_samples_from_message,
    interactions_from_segments,
)
from worldfoundry.studio.visualization.backends.world_realtime import (
    LatestFrameBuffer,
    RealtimePeerManager,
    _ActivePeer,
    _ActiveSocket,
)


def _edge(resampler: RealtimeControlResampler, stamp: float, event: str, key: str) -> None:
    assert resampler.on_edge(arrival_time=stamp, event=event, key=key)


def _states(segments: list[ControlSegment]) -> list[frozenset[str]]:
    return [keys for start, end, keys in segments if end > start]


def test_realtime_input_retains_tap_during_inference_after_state_queries() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100, "keydown", "w")
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset({"w"})]
    _edge(resampler, 100.02, "keyup", "w")
    _edge(resampler, 100.05, "keydown", "d")
    _edge(resampler, 100.10, "keyup", "d")
    for _ in range(3):
        assert resampler.effective_keys == frozenset()
        assert resampler.has_pending_input
    segments = resampler.sample_chunk(4, wall_time=100.25)
    assert _states(segments) == [frozenset(), frozenset({"d"}), frozenset()]
    assert segments[1][1] - segments[1][0] == pytest.approx(0.05)
    assert not resampler.has_pending_input
    assert _states(resampler.sample_chunk(4, wall_time=100.5)) == [frozenset()]


def test_realtime_input_releases_consumed_hold_without_replaying_it() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100, "keydown", "w")
    resampler.sample_chunk(4, wall_time=100)
    _edge(resampler, 100.01, "keydown", "w")  # Browser key repeat is not a new tap.
    _edge(resampler, 100.02, "keyup", "w")
    assert _states(resampler.sample_chunk(4, wall_time=101)) == [frozenset()]
    assert not resampler.has_pending_input


@pytest.mark.parametrize("chunk_frames", [1, 4])
def test_realtime_input_preserves_repeated_and_mixed_taps_across_chunks(chunk_frames: int) -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for stamp, event, key in (
        (100.01, "keydown", "d"),
        (100.02, "keyup", "d"),
        (100.03, "keydown", "d"),
        (100.04, "keyup", "d"),
        (100.05, "keydown", "a"),
        (100.06, "keydown", "w"),
        (100.07, "keyup", "a"),
        (100.08, "keyup", "w"),
    ):
        _edge(resampler, stamp, event, key)
    states: list[frozenset[str]] = []
    for _ in range(16):
        for keys in _states(resampler.sample_chunk(chunk_frames, wall_time=101)):
            if not states or states[-1] != keys:
                states.append(keys)
        if not resampler.has_pending_input:
            break
    assert states == [
        frozenset({"d"}),
        frozenset(),
        frozenset({"d"}),
        frozenset(),
        frozenset({"a"}),
        frozenset({"a", "w"}),
        frozenset({"w"}),
        frozenset(),
    ]
    assert not resampler.has_pending_input


def test_realtime_input_retains_chunk_tail_tap_until_a_complete_sample_fits() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100.19, "keydown", "d")
    _edge(resampler, 100.191, "keyup", "d")
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset()]
    assert resampler.has_pending_input
    segments = resampler.sample_chunk(4, wall_time=100.25)
    assert _states(segments) == [frozenset({"d"}), frozenset()]
    assert segments[0][1] - segments[0][0] == pytest.approx(0.05)
    assert not resampler.has_pending_input


def test_realtime_input_new_edges_follow_retained_backlog() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100.01, "keydown", "d")
    _edge(resampler, 100.011, "keyup", "d")
    assert _states(resampler.sample_chunk(1, wall_time=101)) == [frozenset({"d"})]
    _edge(resampler, 100.02, "keydown", "a")
    _edge(resampler, 100.021, "keyup", "a")
    segments = resampler.sample_chunk(4, wall_time=101.06)
    assert _states(segments) == [frozenset(), frozenset({"a"}), frozenset()]
    assert not resampler.has_pending_input


def test_realtime_input_preserves_long_duration_and_last_pressed_conflicts() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for stamp, event, key in (
        (100.0, "keydown", "w"),
        (100.1, "keydown", "s"),
        (100.3, "keyup", "s"),
        (100.4, "keyup", "w"),
    ):
        _edge(resampler, stamp, event, key)
    segments = resampler.sample_chunk(16, wall_time=101)
    assert _states(segments) == [frozenset({"w"}), frozenset({"s"}), frozenset({"w"}), frozenset()]
    assert [end - start for start, end, _ in segments[:3]] == pytest.approx([0.1, 0.2, 0.1])


def test_realtime_input_focus_loss_releases_chord_and_reset_discards_backlog() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for key in ("w", "ArrowRight"):
        _edge(resampler, 100, "keydown", key)
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset({"w", "l"})]
    for key in ("w", "ArrowRight"):
        _edge(resampler, 100.01, "keyup", key)
    assert _states(resampler.sample_chunk(4, wall_time=100.25)) == [frozenset()]
    _edge(resampler, 100.3, "keydown", "d")
    _edge(resampler, 100.31, "keyup", "d")
    resampler.reset(start_time=200)
    assert not resampler.has_pending_input
    assert resampler.effective_keys == frozenset()
    assert _states(resampler.sample_chunk(4, wall_time=200)) == [frozenset()]
    _edge(resampler, 200.21, "keydown", "a")
    assert frozenset({"a"}) in _states(resampler.sample_chunk(4, wall_time=200.25))


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
@pytest.mark.parametrize("boundary", [None, "step", "prompt", "prompt_scheduled"])
def test_realtime_input_workers_keep_inflight_tap_and_isolate_step(
    transport: str,
    boundary: str | None,
) -> None:
    class Sink:
        readyState = "open"
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

        async def send_str(self, raw: str) -> None:
            self.send(raw)

    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[tuple[list[str], Any, str | None]] = []

        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(
                self,
                interactions: list[str],
                *,
                control_segments: Any = None,
                prompt: str | None = None,
                **kwargs: Any,
            ) -> tuple[list[np.ndarray], float]:
                calls.append((list(interactions), control_segments, prompt))
                if len(calls) == 1:
                    started.set()
                    await release.wait()
                return [np.zeros((4, 6, 3), dtype=np.uint8)], 1.0

        sink = Sink()
        now = asyncio.get_running_loop().time()
        resampler = RealtimeControlResampler(fps=20, start_time=now)
        if transport == "socket":
            active = _ActiveSocket(socket=sink, resampler=resampler, frame_packets=asyncio.Queue(maxsize=8))
        else:
            active = _ActivePeer(peer=None, channel=sink, resampler=resampler, frames=LatestFrameBuffer(maxsize=8))
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=20, chunk_frames=4)
        if boundary == "prompt_scheduled":
            active.prompt_scheduled = True
            active.initial_segment_pending = True
        _edge(resampler, now, "keydown", "w")
        active.first_action.set()
        if isinstance(active, _ActiveSocket):
            worker = asyncio.create_task(peers._socket_generation_worker(active))
        else:
            worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            _edge(resampler, now + 0.02, "keyup", "w")
            _edge(resampler, now + 0.05, "keydown", "d")
            _edge(resampler, now + 0.10, "keyup", "d")
            if boundary in {"step", "prompt_scheduled"}:
                active.pending_steps = 1
            elif boundary == "prompt":
                active.pending_prompt = "Turn right"
                active.pending_prompt_dirty = True
            assert resampler.effective_keys == frozenset()
            release.set()

            async def finished() -> None:
                expected = 3 if boundary == "step" else 2
                while len(calls) < expected or active.first_action.is_set():
                    if worker.done():
                        await worker
                        cast(Callable[[str], NoReturn], pytest.fail)(f"worker stopped: {sink.messages}")
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(finished(), timeout=2)
            if boundary == "prompt_scheduled":
                assert calls == [([], None, None), ([], None, None)]
                assert resampler.has_pending_input
                return
            assert calls[0][0] == ["forward"]
            if boundary == "step":
                assert calls[1] == ([], None, None)
            assert calls[-1][0] == ["right"]
            assert frozenset({"d"}) in _states(calls[-1][1])
            assert calls[-1][2] == ("Turn right" if boundary == "prompt" else None)
            assert not resampler.has_pending_input
            assert not any(message["type"] == "error" for message in sink.messages)
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def _axis_spec_resampler(*axes: RealtimeAxis, start_time: float = 100) -> RealtimeControlResampler:
    return RealtimeControlResampler(fps=20, start_time=start_time, axes=axes)


def test_realtime_spec_payload_round_trips_declared_keys_and_axes() -> None:
    spec = RealtimeSpec(
        controls=("jump", "slide", "turn_left"),
        keys=(
            RealtimeKey("w", "jump", "vertical", "Jump"),
            RealtimeKey("s", "slide", "vertical"),
            RealtimeKey("z", "turn_left"),
        ),
        axes=(
            RealtimeAxis("camera_yaw", "mouse_x", scale=0.01, limit=0.1),
            RealtimeAxis("move_x", "gamepad_lx", mode="level", deadzone=0.15),
        ),
    )
    payload = json.loads(json.dumps(spec.to_payload()))
    assert payload["keys"][0] == {"key": "w", "action": "jump", "group": "vertical", "label": "Jump"}
    assert payload["axes"] == [
        {"name": "camera_yaw", "source": "mouse_x", "mode": "delta", "scale": 0.01, "limit": 0.1, "deadzone": 0.0},
        {"name": "move_x", "source": "gamepad_lx", "mode": "level", "scale": 1.0, "deadzone": 0.15},
    ]
    assert RealtimeSpec.from_payload(payload) == spec
    no_keys = RealtimeSpec(keys=())
    assert no_keys.to_payload()["keys"] == []
    assert RealtimeSpec.from_payload(no_keys.to_payload()).keys == ()


def test_realtime_spec_default_payload_omits_keys_and_axes() -> None:
    assert RealtimeSpec().to_payload() == {
        "fps": 16,
        "first_chunk_frames": 9,
        "steady_chunk_frames": 9,
        "controls": ["forward", "backward", "left", "right", "camera_up", "camera_down", "camera_l", "camera_r"],
        "transport": "in-memory-rgb",
        "stateful": True,
    }
    parsed = RealtimeSpec.from_payload(RealtimeSpec(controls=("prompt_update",)).to_payload())
    assert parsed.keys == DEFAULT_REALTIME_KEYS
    assert parsed.axes == ()


@pytest.mark.parametrize(
    "build",
    [
        lambda: RealtimeSpec(keys=(RealtimeKey("w", "forward"), RealtimeKey("w", "jump"))),
        lambda: RealtimeKey("", "forward"),
        lambda: RealtimeKey("w", ""),
        lambda: RealtimeSpec(axes=(RealtimeAxis("yaw", "mouse_x"), RealtimeAxis("yaw", "mouse_y"))),
        lambda: RealtimeAxis("", "mouse_x"),
        lambda: RealtimeAxis("yaw", "mouse_wheel"),
        lambda: RealtimeAxis("yaw", "mouse_x", mode="absolute"),
        lambda: RealtimeAxis("yaw", "mouse_x", limit=0.0),
        lambda: RealtimeAxis("yaw", "gamepad_rx", mode="level", deadzone=-0.1),
    ],
)
def test_realtime_spec_rejects_invalid_keys_and_axes(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("key", ["W", " ", "arrowup"])
def test_realtime_resampler_rejects_key_names_it_cannot_match(key: str) -> None:
    with pytest.raises(ValueError, match="normalized key names"):
        RealtimeControlResampler(fps=16, realtime_keys=(RealtimeKey(key, "jump"),))


def test_realtime_spec_from_payload_falls_back_on_malformed_axes() -> None:
    fallback = RealtimeSpec(fps=24)
    assert (
        RealtimeSpec.from_payload({"axes": [{"name": "yaw", "source": "mouse_wheel"}]}, fallback=fallback) == fallback
    )
    assert RealtimeSpec.from_payload({"keys": ["w"]}, fallback=fallback) == fallback


def test_realtime_spec_payload_round_trips_continuous_advance_and_max_chunks() -> None:
    spec = RealtimeSpec(controls=("prompt_update",), keys=(), advance="continuous", max_chunks=7)
    payload = json.loads(json.dumps(spec.to_payload()))
    assert payload["advance"] == "continuous"
    assert payload["max_chunks"] == 7
    assert RealtimeSpec.from_payload(payload) == spec
    assert RealtimeSpec.from_payload({"realtime_spec": payload}) == spec

    unbounded = RealtimeSpec(advance="continuous")
    assert "max_chunks" not in unbounded.to_payload()
    assert RealtimeSpec.from_payload(unbounded.to_payload()) == unbounded
    defaults = RealtimeSpec(max_chunks=None).to_payload()
    assert "advance" not in defaults and "max_chunks" not in defaults
    assert RealtimeSpec.from_payload(defaults).advance == "on_input"


@pytest.mark.parametrize(
    "build",
    [
        lambda: RealtimeSpec(advance="auto"),
        lambda: RealtimeSpec(advance="continuous", max_chunks=0),
        lambda: RealtimeSpec(max_chunks=-1),
    ],
)
def test_realtime_spec_rejects_unknown_advance_and_non_positive_max_chunks(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize(
    "payload",
    [{"advance": "auto"}, {"advance": "continuous", "max_chunks": 0}, {"max_chunks": "seven"}],
)
def test_realtime_spec_from_payload_falls_back_on_malformed_advance(payload: dict[str, Any]) -> None:
    fallback = RealtimeSpec(fps=24, advance="continuous", max_chunks=3)
    assert RealtimeSpec.from_payload(payload, fallback=fallback) == fallback


def test_realtime_input_declared_keys_drive_acceptance_groups_and_actions() -> None:
    realtime_keys = (
        RealtimeKey("w", "jump", "vertical"),
        RealtimeKey("s", "slide", "vertical"),
        RealtimeKey("z", "turn_left"),
        RealtimeKey("c", "turn_right"),
    )
    resampler = RealtimeControlResampler(fps=20, start_time=100, realtime_keys=realtime_keys)
    assert not resampler.on_edge(arrival_time=100, event="keydown", key="a")
    for key in ("w", "s", "z", "c"):
        _edge(resampler, 100, "keydown", key)
    assert resampler.effective_keys == frozenset({"s", "z", "c"})
    segments = resampler.sample_chunk(4, wall_time=100)
    assert _states(segments) == [frozenset({"s", "z", "c"})]
    assert interactions_from_segments(segments, realtime_keys) == ["slide", "turn_left", "turn_right"]


def test_axis_samples_from_message_maps_sources_and_ignores_missing_fields() -> None:
    assert axis_samples_from_message({"type": "axis", "source": "mouse", "dx": 3, "dy": -1.5}) == {
        "mouse_x": 3.0,
        "mouse_y": -1.5,
    }
    assert axis_samples_from_message({"source": "gamepad", "lx": 0.5, "ry": -0.25, "ly": "up", "rx": None}) == {
        "gamepad_lx": 0.5,
        "gamepad_ry": -0.25,
    }
    assert axis_samples_from_message({"source": "mouse", "dx": True, "dy": float("nan")}) == {}
    assert axis_samples_from_message({"source": "touch", "dx": 1.0}) == {}


def test_realtime_input_delta_axis_conserves_irregular_samples_across_chunks() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("yaw", "mouse_x"), RealtimeAxis("pitch", "mouse_y"))
    arrivals = [
        (100.003, 1.0),
        (100.011, 2.5),
        (100.049, -0.5),
        (100.051, 4.0),
        (100.137, 0.25),
        (100.21, 3.0),
        (100.262, -1.0),
        (100.263, 2.0),
        (100.33, 0.75),
        (100.39, 0.5),
        (100.47, 1.5),
        (100.56, -0.25),
    ]
    frames: list[float] = []
    pending = list(arrivals)
    # Inference runs slower than playback, so most chunks start in the past.
    for wall_time in (100.0, 100.35, 100.58, 100.95):
        while pending and pending[0][0] <= wall_time:
            stamp, value = pending.pop(0)
            assert resampler.on_axis(arrival_time=stamp, samples={"mouse_x": value})
        chunk = resampler.sample_input_chunk(4, wall_time=wall_time)
        assert chunk.axes["pitch"] == [0.0] * 4
        frames.extend(chunk.axes["yaw"])
    assert frames == pytest.approx([0.0] * 4 + [10.25, 1.0, 0.75, 0.0] + [0.5, 1.5, 0.0, -0.25] + [0.0] * 4)
    assert sum(frames) == pytest.approx(sum(value for _, value in arrivals))
    assert not resampler.has_pending_input


def test_realtime_input_delta_axis_keeps_future_samples_and_moves_late_samples_forward() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("yaw", "mouse_x"))
    resampler.on_axis(arrival_time=100.1, samples={"mouse_x": 2.0})
    resampler.on_axis(arrival_time=100.3, samples={"mouse_x": 5.0})
    chunk = resampler.sample_input_chunk(4, wall_time=100)
    assert chunk.axes == {"yaw": [0.0, 0.0, 2.0, 0.0]}
    assert resampler.has_pending_input
    chunk = resampler.sample_input_chunk(4, wall_time=100.21)
    assert chunk.axes == {"yaw": [0.0, 0.0, 5.0, 0.0]}
    assert not resampler.has_pending_input
    # Arrives after its chunk [100.2, 100.4) was sampled; lands on the next first frame.
    resampler.on_axis(arrival_time=100.25, samples={"mouse_x": 1.0})
    chunk = resampler.sample_input_chunk(4, wall_time=100.41)
    assert chunk.axes == {"yaw": [1.0, 0.0, 0.0, 0.0]}
    assert chunk.has_axis_input


def test_realtime_input_axis_samples_stay_behind_a_postponed_key_tap() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("yaw", "mouse_x"))
    _edge(resampler, 100.19, "keydown", "d")
    _edge(resampler, 100.191, "keyup", "d")
    resampler.on_axis(arrival_time=100.195, samples={"mouse_x": 1.0}, control_id="after-tap")
    first = resampler.sample_input_chunk(4, wall_time=100)
    assert _states(first.segments) == [frozenset()]
    assert first.axes == {"yaw": [0.0] * 4}
    assert first.axis_control_ids == ()
    assert not first.has_axis_input
    # The tap moves to the next chunk and holds a full frame; the delta follows its release.
    second = resampler.sample_input_chunk(4, wall_time=100.25)
    assert _states(second.segments) == [frozenset({"d"}), frozenset()]
    assert second.axes == {"yaw": [0.0, 1.0, 0.0, 0.0]}
    assert second.axis_control_ids == ("after-tap",)


def test_realtime_input_level_axis_holds_until_a_neutral_sample() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("strafe", "gamepad_lx", mode="level"))
    assert not resampler.on_axis(arrival_time=100.01, samples={"gamepad_ry": 0.9})
    assert not resampler.has_pending_input
    assert resampler.on_axis(arrival_time=100.06, samples={"gamepad_lx": 0.5})
    assert resampler.sample_input_chunk(4, wall_time=100).axes == {"strafe": [0.0, 0.5, 0.5, 0.5]}
    assert resampler.has_pending_input
    assert resampler.sample_input_chunk(4, wall_time=100.2).axes == {"strafe": [0.5] * 4}
    resampler.on_axis(arrival_time=100.47, samples={"gamepad_lx": 0.0})
    assert resampler.sample_input_chunk(4, wall_time=100.4).axes == {"strafe": [0.5, 0.0, 0.0, 0.0]}
    assert not resampler.has_pending_input
    assert resampler.sample_input_chunk(4, wall_time=100.6).axes == {"strafe": [0.0] * 4}


def test_realtime_input_axis_applies_deadzone_then_scale_then_limit() -> None:
    resampler = _axis_spec_resampler(
        RealtimeAxis("move", "gamepad_lx", mode="level", scale=2.0, limit=0.5, deadzone=0.2),
        RealtimeAxis("yaw", "mouse_x", scale=2.0, limit=0.5, deadzone=1.0),
    )
    for stamp, level in ((100.0, 0.15), (100.05, 0.2), (100.1, 0.4), (100.15, -0.4)):
        resampler.on_axis(arrival_time=stamp, samples={"gamepad_lx": level})
    for stamp, delta in ((100.0, 0.1), (100.06, 0.2), (100.07, 0.2), (100.15, -0.1)):
        resampler.on_axis(arrival_time=stamp, samples={"mouse_x": delta})
    axes = resampler.sample_input_chunk(4, wall_time=100).axes
    assert axes["move"] == pytest.approx([0.0, 0.4, 0.5, -0.5])
    assert axes["yaw"] == pytest.approx([0.2, 0.5, 0.0, -0.2])


def test_realtime_input_pending_axis_input_and_reset() -> None:
    resampler = _axis_spec_resampler(
        RealtimeAxis("yaw", "mouse_x"),
        RealtimeAxis("move", "gamepad_ly", mode="level", deadzone=0.1),
    )
    resampler.on_axis(arrival_time=100.01, samples={"gamepad_ly": 0.05})
    resampler.sample_input_chunk(4, wall_time=100)
    assert not resampler.has_pending_input
    resampler.on_axis(arrival_time=100.21, samples={"mouse_x": 2.0, "gamepad_ly": 0.8})
    assert resampler.has_pending_input
    resampler.reset(start_time=200)
    assert not resampler.has_pending_input
    assert resampler.sample_input_chunk(4, wall_time=200).axes == {"yaw": [0.0] * 4, "move": [0.0] * 4}


@pytest.mark.unit
def test_axis_control_ids_follow_consumed_deltas_without_changing_samples() -> None:
    axes = (RealtimeAxis("yaw", "mouse_x"), RealtimeAxis("pitch", "mouse_y"))
    tracked = _axis_spec_resampler(*axes)
    plain = _axis_spec_resampler(*axes)
    for stamp, samples, control_id in (
        (100.01, {"mouse_x": 1.0, "mouse_y": 2.0}, "mouse-1"),
        (100.025, {"mouse_x": 3.0}, "mouse-2"),
        (100.25, {"mouse_x": 5.0}, "future"),
    ):
        assert tracked.on_axis(arrival_time=stamp, samples=samples, control_id=control_id)
        assert plain.on_axis(arrival_time=stamp, samples=samples)
    assert not tracked.on_axis(arrival_time=100.3, samples={"gamepad_rx": 0.5}, control_id="undeclared")
    for wall_time, expected in ((100, ("mouse-1", "mouse-2")), (100.2, ("future",)), (100.4, ())):
        chunk = tracked.sample_input_chunk(4, wall_time=wall_time)
        untracked = plain.sample_input_chunk(4, wall_time=wall_time)
        assert chunk.axes == untracked.axes
        assert chunk.segments == untracked.segments
        assert chunk.axis_control_ids == expected
        assert untracked.axis_control_ids == ()


@pytest.mark.unit
def test_axis_control_ids_exclude_overwritten_levels_and_keep_consumed_holds() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("yaw", "gamepad_rx", mode="level"))
    resampler.on_axis(arrival_time=100.001, samples={"gamepad_rx": 0.5}, control_id="overwritten")
    resampler.on_axis(arrival_time=100.011, samples={"gamepad_rx": 0.8}, control_id="held")
    first = resampler.sample_input_chunk(1, wall_time=100)
    assert first.axes == {"yaw": [0.8]}
    assert first.axis_control_ids == ("held",)
    held = resampler.sample_input_chunk(1, wall_time=100.05)
    assert held.axes == first.axes
    assert held.axis_control_ids == ("held",)

    resampler.on_axis(arrival_time=100.11, samples={"gamepad_rx": 0.6})
    untracked = resampler.sample_input_chunk(1, wall_time=100.1)
    assert untracked.axes == {"yaw": [0.6]}
    assert untracked.axis_control_ids == ()
    resampler.on_axis(arrival_time=100.151, samples={"gamepad_rx": 0.0}, control_id="neutral")
    neutral = resampler.sample_input_chunk(1, wall_time=100.15)
    assert neutral.axes == {"yaw": [0.0]}
    assert neutral.axis_control_ids == ("neutral",)
    assert resampler.sample_input_chunk(1, wall_time=100.2).axis_control_ids == ()


@pytest.mark.unit
def test_axis_control_ids_reset_with_queued_samples_and_held_state() -> None:
    resampler = _axis_spec_resampler(RealtimeAxis("yaw", "gamepad_rx", mode="level"))
    resampler.on_axis(arrival_time=100.01, samples={"gamepad_rx": 0.5}, control_id="held")
    assert resampler.sample_input_chunk(1, wall_time=100).axis_control_ids == ("held",)
    resampler.on_axis(arrival_time=100.3, samples={"gamepad_rx": -0.5}, control_id="queued")
    resampler.reset(start_time=200)
    chunk = resampler.sample_input_chunk(4, wall_time=200)
    assert chunk.axes == {"yaw": [0.0] * 4}
    assert chunk.axis_control_ids == ()
