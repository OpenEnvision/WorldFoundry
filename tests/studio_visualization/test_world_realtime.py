from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import DEFAULT_REALTIME_KEYS, RealtimeAxis, RealtimeSpec
from worldfoundry.core.execution.realtime.prewarm import PrewarmTimeoutError
from worldfoundry.core.media.processing.postprocess import VideoSpec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PipelineContext,
    PreparedInputs,
    StudioManager,
)
from worldfoundry.studio.serving.realtime.input import RealtimeInputChunk
from worldfoundry.studio.serving.realtime.media import FrameQueuePolicy
from worldfoundry.studio.serving.realtime.telemetry import DeliveryTelemetry
from worldfoundry.studio.visualization.backends import world_realtime as realtime_backend
from worldfoundry.studio.visualization.backends.world import world_frontend_css, world_frontend_html, world_frontend_js
from worldfoundry.studio.visualization.backends.world_realtime import (
    LatestFrameBuffer,
    OutputResolutionState,
    RealtimeControlResampler,
    RealtimeControlState,
    RealtimePeerManager,
    ResidentWorldRuntime,
    _ActivePeer,
    _ActiveSocket,
    _build_video_track,
    _default_realtime_chunk_frames,
    _default_realtime_inference_steps,
    _encode_jpeg_frames,
    _output_resolution_options,
    _realtime_frame_budget,
    _realtime_overrides,
    _resize_rgb_frames,
    interactions_from_segments,
    normalize_output_resolution,
    normalize_text_events,
    realtime_frames_from_result,
)
from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig


async def _wait_for_condition(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0)


def test_websocket_fallback_encodes_decodable_stable_frames() -> None:
    frames = [np.full((6, 8, 3), value, dtype=np.uint8) for value in (32, 192)]

    packets = _encode_jpeg_frames(frames, quality=88)

    assert len(packets) == 2
    assert all(packet.startswith(b"\xff\xd8") for packet in packets)
    assert [Image.open(io.BytesIO(packet)).size for packet in packets] == [(8, 6), (8, 6)]


def test_realtime_postprocess_defaults_to_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", raising=False)

    stream = realtime_backend._realtime_postprocess_stream(
        fps=16,
        launch_config=StudioLaunchConfig(model_id="test", device="cuda:3"),
    )

    assert stream.processor_names == ("identity",)


def test_realtime_postprocess_resolves_explicit_rtx_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.core.media.processing.rtx as rtx_module

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "rtx-super-resolution-ultra")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_RTX_SCALE", "1.5")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_RTX_NON_BLOCKING", "true")
    probe: dict[str, object] = {}

    def _require_runtime(*, device: int, config: object, input_spec: object, run_inference: bool) -> object:
        probe.update(
            device=device,
            config=config,
            input_spec=input_spec,
            run_inference=run_inference,
        )
        return types.SimpleNamespace(
            device=device,
            gpu_name="Fake RTX",
            package_version="0.1.0.1",
        )

    monkeypatch.setattr(
        rtx_module,
        "require_rtx_vfx_runtime",
        _require_runtime,
    )

    stream = realtime_backend._realtime_postprocess_stream(
        fps=24,
        launch_config=StudioLaunchConfig(model_id="test", device="cuda:3"),
        native_resolution=(640, 360),
    )

    [processor] = stream.chain.processors
    processor = cast(rtx_module.RTXVideoSuperResolutionPostProcessor, processor)
    assert stream.processor_names == ("rtx-video-super-resolution",)
    assert processor.config.quality == "ULTRA"
    assert processor.config.scale == 1.5
    assert processor.config.device == 3
    assert processor.config.non_blocking is True
    assert probe["device"] == 3
    assert probe["config"] is processor.config
    assert probe["input_spec"] == VideoSpec(height=360, width=640, fps=24, dtype="uint8")
    assert probe["run_inference"] is True


def test_realtime_postprocess_rejects_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "rtx-super-resolution")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_RTX_NON_BLOCKING", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        realtime_backend._realtime_postprocess_stream(
            fps=16,
            launch_config=StudioLaunchConfig(model_id="test", device="cuda"),
        )


def test_websocket_output_scaler_preserves_canvas_and_aspect_ratio() -> None:
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)

    [packet] = _encode_jpeg_frames(
        [frame],
        quality=90,
        output_resolution=(320, 240),
    )

    image = np.asarray(Image.open(io.BytesIO(packet)).convert("RGB"))
    assert image.shape == (240, 320, 3)
    assert int(image[0, 0].max()) < 20
    assert int(image[120, 160].min()) > 230

    with pytest.raises(ValueError, match="cannot upscale"):
        _encode_jpeg_frames(
            [np.zeros((90, 160, 3), dtype=np.uint8)],
            quality=90,
            output_resolution=(320, 180),
        )


def test_text_event_and_output_resolution_validation() -> None:
    events = normalize_text_events([{"event_id": "weather:rain", "label": "Rain", "prompt": "Heavy rain begins."}])

    assert events[0]["category"] == "event"
    assert normalize_output_resolution("960x540") == (960, 540)
    assert OutputResolutionState.from_value({"mode": "native"}).to_payload() == {"mode": "native"}
    for invalid in (
        [{"event_id": "bad id", "prompt": "x"}],
        [{"event_id": "same", "prompt": "x"}, {"event_id": "same", "prompt": "y"}],
    ):
        try:
            normalize_text_events(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid event catalog was accepted")
    try:
        normalize_output_resolution({"width": 641, "height": 360})
    except ValueError:
        pass
    else:
        raise AssertionError("odd output resolution was accepted")
    with pytest.raises(ValueError, match="between"):
        normalize_output_resolution({"width": 3840, "height": 2160})

    bounded = OutputResolutionState.from_value(
        {"width": 640, "height": 360},
        maximum=(1280, 720),
    )
    bounded.observe_source(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert bounded.snapshot().dimensions == (640, 360)
    revision = bounded.revision
    with pytest.raises(ValueError, match="does not upscale"):
        bounded.update({"width": 1280, "height": 1080})
    assert bounded.dimensions == (640, 360)
    assert bounded.revision == revision

    early = OutputResolutionState.from_value({"width": 960, "height": 540})
    early.observe_source(np.zeros((360, 640, 3), dtype=np.uint8))
    snapshot = early.snapshot()
    assert snapshot.dimensions is None
    assert snapshot.to_payload() == {"mode": "native", "width": 640, "height": 360}
    assert snapshot.revision == 1


def test_output_resolution_options_never_guess_unknown_native_size() -> None:
    assert _output_resolution_options(find_entry("sana-wm")) == [{"mode": "native", "label": "Native"}]
    dreamx = _output_resolution_options(find_entry("dreamx-world-5b-cam"))
    assert dreamx[0] == {
        "mode": "native",
        "label": "Native · 1280×704",
        "width": 1280,
        "height": 704,
    }
    assert all(option["mode"] == "native" or (option["width"] <= 1280 and option["height"] <= 704) for option in dreamx)
    postprocessed = _output_resolution_options(
        find_entry("dreamx-world-5b-cam"),
        native_override=(1920, 1056),
    )
    assert postprocessed[0] == {
        "mode": "native",
        "label": "Native · 1920×1056",
        "width": 1920,
        "height": 1056,
    }
    assert postprocessed[1] == {
        "mode": "fixed",
        "label": "Stream · 1440×792",
        "width": 1440,
        "height": 792,
    }


def test_close_active_is_bounded_when_workers_and_runtime_ignore_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Runtime:
        def __init__(self) -> None:
            self.reset_started = asyncio.Event()
            self.reset_cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def reset(self) -> None:
            self.reset_started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.reset_cancelled.set()
                await self.release.wait()

    class Peer:
        closed = 0

        async def close(self) -> None:
            self.closed += 1

    async def exercise() -> None:
        runtime = Runtime()
        peer = Peer()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        worker_started = asyncio.Event()
        worker_cancelled = asyncio.Event()

        async def cancellation_resistant_worker() -> None:
            worker_started.set()
            try:
                await runtime.release.wait()
            except asyncio.CancelledError:
                worker_cancelled.set()
                await runtime.release.wait()

        worker = asyncio.create_task(cancellation_resistant_worker())
        active = _ActivePeer(
            peer=peer,
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            generation_task=worker,
        )
        peers._active = active
        await worker_started.wait()

        await asyncio.wait_for(peers.close_active(), timeout=0.25)

        assert active.closed is True
        assert peer.closed == 1
        assert worker_cancelled.is_set()
        assert runtime.reset_started.is_set()
        await asyncio.wait_for(runtime.reset_cancelled.wait(), timeout=0.1)
        assert peers.draining is False
        assert peers._drain_done.is_set()

        lingering = list(peers._shutdown_tasks)
        runtime.release.set()
        await asyncio.wait_for(
            asyncio.gather(worker, *lingering, return_exceptions=True),
            timeout=0.25,
        )

    asyncio.run(exercise())

    assert "deadline expired while waiting for WebRTC workers" in caplog.text
    assert "deadline expired while waiting for session runtime reset" in caplog.text


def test_close_active_normal_teardown_is_idempotent() -> None:
    class Runtime:
        resets = 0

        async def reset(self) -> None:
            self.resets += 1

    class Peer:
        closes = 0

        async def close(self) -> None:
            self.closes += 1

    async def exercise() -> None:
        runtime = Runtime()
        peer = Peer()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=peer,
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )

        await asyncio.gather(peers.close_active(), peers.close_active())
        await peers.close_active()

        assert peer.closes == 1
        assert runtime.resets == 1
        assert peers.active is False
        assert peers.draining is False

    asyncio.run(exercise())


def test_socket_generation_cancel_does_not_wait_forever_for_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Socket:
        closed = False

        async def send_str(self, _raw: str) -> None:
            pass

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, _interactions, **_kwargs):
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            return [np.zeros((4, 6, 3), dtype=np.uint8)], 1.0

    async def exercise() -> None:
        runtime = Runtime()
        active = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=1),
            pending_steps=1,
        )
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        worker = asyncio.create_task(peers._socket_generation_worker(active))
        await runtime.started.wait()

        worker.cancel()
        await asyncio.wait_for(
            asyncio.gather(worker, return_exceptions=True),
            timeout=0.25,
        )

        assert worker.cancelled()
        assert runtime.cancelled.is_set()
        lingering = list(peers._shutdown_tasks)
        assert lingering
        runtime.release.set()
        await asyncio.wait_for(
            asyncio.gather(*lingering, return_exceptions=True),
            timeout=0.25,
        )

    asyncio.run(exercise())


def test_runtime_reset_and_close_have_independent_deadlines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")
    reset_entered = threading.Event()
    release_thread = threading.Event()

    class BlockingPostprocess:
        def reset(self) -> None:
            reset_entered.set()
            release_thread.wait()

    class Manager:
        workspace_root = str(tmp_path)

    runtime = ResidentWorldRuntime(
        manager=cast(Any, Manager()),
        entry=find_entry("sana-wm"),
        launch_config=StudioLaunchConfig(model_id="sana-wm", frontend="world"),
        fps=16,
        postprocess=cast(Any, BlockingPostprocess()),
    )
    try:
        asyncio.run(asyncio.wait_for(runtime.reset(), timeout=0.25))
        assert reset_entered.is_set()
        assert runtime._configured is False
        assert runtime._base_request is None
    finally:
        release_thread.set()
        runtime._executor.shutdown(wait=True, cancel_futures=True)

    assert "Realtime postprocess reset did not finish within" in caplog.text

    close_runtime = ResidentWorldRuntime(
        manager=cast(Any, Manager()),
        entry=find_entry("sana-wm"),
        launch_config=StudioLaunchConfig(model_id="sana-wm", frontend="world"),
        fps=16,
    )

    async def exercise_close() -> None:
        reset_started = asyncio.Event()
        reset_cancelled = asyncio.Event()
        release = asyncio.Event()

        async def cancellation_resistant_reset() -> None:
            reset_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                reset_cancelled.set()
                await release.wait()

        monkeypatch.setattr(close_runtime, "reset", cancellation_resistant_reset)
        await asyncio.wait_for(close_runtime.close(), timeout=0.25)
        assert reset_started.is_set()
        assert reset_cancelled.is_set()
        lingering = list(close_runtime._shutdown_tasks)
        assert lingering
        release.set()
        await asyncio.wait_for(
            asyncio.gather(*lingering, return_exceptions=True),
            timeout=0.25,
        )

    asyncio.run(exercise_close())

    assert "deadline expired while waiting for runtime reset" in caplog.text


def test_datachannel_events_ack_and_idle_step_are_ordered() -> None:
    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    async def exercise() -> None:
        channel = Channel()
        active = _ActivePeer(
            peer=None,
            channel=channel,
            frames=LatestFrameBuffer(maxsize=2),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            base_prompt="base scene",
        )
        peers = RealtimePeerManager(runtime=cast(Any, None), fps=16, chunk_frames=9)

        await peers._handle_message(
            active,
            json.dumps(
                {
                    "type": "event_catalog",
                    "request_id": "catalog-1",
                    "base_revision": 0,
                    "events": [
                        {
                            "event_id": "portal",
                            "label": "Portal",
                            "prompt": "A portal opens ahead.",
                        }
                    ],
                }
            ),
        )
        await peers._handle_message(
            active,
            json.dumps(
                {
                    "type": "event",
                    "event_id": "portal",
                    "state": "trigger",
                    "request_id": "event-1",
                }
            ),
        )
        await peers._handle_message(
            active,
            json.dumps(
                {
                    "type": "output_config",
                    "resolution": {"width": 640, "height": 360},
                    "request_id": "output-1",
                }
            ),
        )
        await peers._handle_message(
            active,
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "step"},
                    "request_id": "step-1",
                }
            ),
        )

        assert [message["type"] for message in channel.messages] == [
            "event_catalog_ack",
            "event_ack",
            "output_config_ack",
            "step_ack",
        ]
        assert all(message["ok"] is True for message in channel.messages)
        assert channel.messages[1]["request_id"] == "event-1"
        assert channel.messages[1]["active_event_id"] == "portal"
        assert active.pending_prompt == "A portal opens ahead."
        assert active.pending_prompt_dirty is True
        assert active.pending_steps == 1
        assert active.output_resolution.dimensions == (640, 360)
        assert active.resampler.effective_keys == frozenset()

    asyncio.run(exercise())


def test_idle_step_generates_exactly_one_chunk_without_keyboard_input() -> None:
    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str | None]] = []
            self.generated = asyncio.Event()

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, interactions, *, prompt=None, **kwargs):
            del kwargs
            self.calls.append((list(interactions), prompt))
            self.generated.set()
            return [np.zeros((4, 6, 3), dtype=np.uint8)], 1.0

    async def exercise() -> None:
        runtime = Runtime()
        channel = Channel()
        active = _ActivePeer(
            peer=None,
            channel=channel,
            frames=LatestFrameBuffer(maxsize=2),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await peers._handle_message(
                active,
                json.dumps({"type": "action", "action": {"event": "step"}}),
            )
            await asyncio.wait_for(runtime.generated.wait(), timeout=1)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert runtime.calls == [([], None)]
            assert active.pending_steps == 0
            assert active.first_action.is_set() is False
            chunk = next(message for message in channel.messages if message["type"] == "chunk_done")
            assert chunk["interactions"] == ["step"]
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def test_generation_worker_uses_split_input_frame_count() -> None:
    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Resampler:
        has_pending_input = False
        realtime_keys = DEFAULT_REALTIME_KEYS

        def __init__(self) -> None:
            self.sampled: list[int] = []

        def sample_input_chunk(self, num_frames: int, *, wall_time: float) -> RealtimeInputChunk:
            self.sampled.append(num_frames)
            return RealtimeInputChunk([(wall_time, wall_time + 1.0, frozenset({"w"}))], {})

    class Runtime:
        last_generation_metrics: dict[str, float] = {}
        realtime_spec = RealtimeSpec(
            fps=24,
            first_chunk_frames=5,
            steady_chunk_frames=5,
            input_fps=6.0,
            first_input_frames=2,
            steady_input_frames=2,
        )

        def __init__(self) -> None:
            self.active: _ActivePeer | None = None

        def next_chunk_frames(self, default: int) -> int:
            del default
            return 5

        def next_input_frames(self, default: int) -> int:
            del default
            return 2

        async def generate(self, _interactions, **_kwargs):
            assert self.active is not None
            self.active.closed = True
            return [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(5)], 1.0

    async def exercise() -> None:
        runtime = Runtime()
        channel = Channel()
        resampler = Resampler()
        active = _ActivePeer(
            peer=None,
            channel=channel,
            frames=LatestFrameBuffer(maxsize=5),
            resampler=cast(Any, resampler),
        )
        runtime.active = active
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=24, chunk_frames=5)
        assert peers._runtime_input_fps() == pytest.approx(6.0)
        active.first_action.set()

        await asyncio.wait_for(peers._generation_worker(active), timeout=1)

        assert resampler.sampled == [2]
        chunk = next(message for message in channel.messages if message["type"] == "chunk_done")
        assert chunk["frames"] == 5
        assert chunk["interactions"] == ["forward"]

    asyncio.run(exercise())


def test_generation_worker_logs_periodic_perf(caplog: pytest.LogCaptureFixture) -> None:
    class Resampler:
        has_pending_input = True
        realtime_keys = DEFAULT_REALTIME_KEYS

        def sample_input_chunk(self, num_frames: int, *, wall_time: float) -> RealtimeInputChunk:
            del num_frames
            return RealtimeInputChunk([(wall_time, wall_time + 1.0, frozenset({"w"}))], {})

    class Runtime:
        last_generation_metrics = {
            "runtime_ms": 8.0,
            "model_ms": 5.0,
            "decode_ms": 2.0,
            "copy_ms": 1.5,
            "cache_frames": 17.0,
            "cache_tokens": 512.0,
        }

        def __init__(self) -> None:
            self.active: _ActivePeer | None = None
            self.calls = 0

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, _interactions, **_kwargs):
            self.calls += 1
            assert self.active is not None
            if self.calls == 3:
                self.active.closed = True
            return [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)], 10.0

    async def exercise() -> None:
        runtime = Runtime()
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=4),
            resampler=cast(Any, Resampler()),
        )
        runtime.active = active
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=24, chunk_frames=2)
        peers._perf_log_interval_chunks = 2
        active.first_action.set()

        await asyncio.wait_for(peers._generation_worker(active), timeout=1)

    caplog.set_level("INFO", logger=realtime_backend.__name__)
    asyncio.run(exercise())

    messages = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Realtime perf")]
    assert len(messages) == 2
    assert "chunk=1" in messages[0]
    assert "chunk=3" in messages[1]
    assert "copy_ms=1.5" in messages[0]
    assert "cache_frames=17" in messages[0]
    assert "cache_tokens=512" in messages[0]
    assert "generation_p50_ms=10.0" in messages[0]
    assert "generation_p90_ms=10.0" in messages[0]


def test_perf_timing_excludes_warmup_and_writes_jsonl(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "realtime-timing.jsonl"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PERF_WARMUP_CHUNKS", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PERF_JSONL", str(trace_path))

    class Runtime:
        last_generation_metrics = {"model_ms": 5.0, "cache_frames": 8.0}

    active = _ActivePeer(
        peer=None,
        channel=None,
        frames=LatestFrameBuffer(maxsize=2),
        resampler=RealtimeControlResampler(fps=16, start_time=0.0),
    )
    peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
    peers._perf_log_interval_chunks = 2
    caplog.set_level("INFO", logger=realtime_backend.__name__)

    for chunk_index, generation_ms in ((1, 100.0), (2, 20.0), (3, 30.0)):
        active.chunk_index = chunk_index
        peers._record_perf(
            active,
            transport="webrtc",
            generation_started=float(chunk_index),
            now=float(chunk_index) + generation_ms / 1000.0,
            output_frames=2,
            generation_ms=generation_ms,
            pixel_post_ms=1.0,
            enqueue_ms=0.5,
            queue_depth=1,
            dropped_frames=0,
            control_latency_ms=None,
        )

    messages = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Realtime perf")]
    assert len(messages) == 2
    assert "chunk=1" in messages[0]
    assert "measured_chunks=0" in messages[0]
    assert "chunk=3" in messages[1]
    assert "measured_chunks=2" in messages[1]
    assert "generation_p50_ms=25.0" in messages[1]
    assert "generation_p90_ms=29.0" in messages[1]

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all(row["model_id"] is None for row in rows)
    assert [row["fields"]["warmup"] for row in rows] == [True, False, False]
    assert [row["fields"]["chunk_index"] for row in rows] == [1, 2, 3]


def test_disabled_perf_summary_does_not_accumulate_timing_samples() -> None:
    class Runtime:
        last_generation_metrics: dict[str, float] = {}

    active = _ActivePeer(
        peer=None,
        channel=None,
        frames=LatestFrameBuffer(maxsize=2),
        resampler=RealtimeControlResampler(fps=16, start_time=0.0),
    )
    peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
    peers._perf_log_interval_chunks = 0

    for chunk_index in range(1, 101):
        active.chunk_index = chunk_index
        peers._record_perf(
            active,
            transport="webrtc",
            generation_started=float(chunk_index),
            now=float(chunk_index) + 0.01,
            output_frames=2,
            generation_ms=10.0,
            pixel_post_ms=1.0,
            enqueue_ms=0.5,
            queue_depth=1,
            dropped_frames=0,
            control_latency_ms=None,
        )

    assert active.perf_window.observed_chunks == 0
    assert active.perf_window.measured_chunks == 0


def test_prompt_scheduled_initial_segment_and_acknowledged_step_are_distinct() -> None:
    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str | None]] = []

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, interactions, *, prompt=None, **kwargs):
            del kwargs
            self.calls.append((list(interactions), prompt))
            return [np.zeros((180, 320, 3), dtype=np.uint8)], 1.0

    async def exercise() -> None:
        runtime = Runtime()
        channel = Channel()
        active = _ActivePeer(
            peer=None,
            channel=channel,
            frames=LatestFrameBuffer(maxsize=4),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            prompt_scheduled=True,
            initial_segment_pending=True,
        )
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        await peers._handle_message(
            active,
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "step"},
                    "request_id": "step-before-initial",
                }
            ),
        )
        worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(runtime.calls) >= 2), timeout=1)
            assert runtime.calls == [([], None), ([], None)]
            assert active.pending_steps == 0
            chunks = [message for message in channel.messages if message["type"] == "chunk_done"]
            assert len(chunks) == 2
            assert chunks[0]["interactions"] == []
            assert chunks[1]["interactions"] == ["step"]
            step_ack = next(message for message in channel.messages if message["type"] == "step_ack")
            assert step_ack["request_id"] == "step-before-initial"
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def test_socket_prompt_scheduled_initial_segment_does_not_consume_step() -> None:
    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_str(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str | None]] = []

        async def generate(self, interactions, *, prompt=None, **kwargs):
            del kwargs
            self.calls.append((list(interactions), prompt))
            return [np.zeros((180, 320, 3), dtype=np.uint8)], 1.0

    async def exercise() -> None:
        socket = Socket()
        runtime = Runtime()
        active = _ActiveSocket(
            socket=socket,
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=4),
            prompt_scheduled=True,
            initial_segment_pending=True,
        )
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=2)
        await peers._handle_socket_message(
            active,
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "step"},
                    "request_id": "socket-step-before-initial",
                }
            ),
        )
        worker = asyncio.create_task(peers._socket_generation_worker(active))
        try:
            await asyncio.wait_for(
                _wait_for_condition(lambda: sum(message["type"] == "chunk_done" for message in socket.messages) >= 2),
                timeout=2,
            )
            assert runtime.calls == [([], None), ([], None)]
            chunks = [message for message in socket.messages if message["type"] == "chunk_done"]
            assert len(chunks) == 2
            assert chunks[0]["interactions"] == []
            assert chunks[1]["interactions"] == ["step"]
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def test_non_object_control_json_is_ignored_without_killing_sessions() -> None:
    class Socket:
        closed = False

        async def send_str(self, _raw: str) -> None:
            pass

    async def exercise() -> None:
        peers = RealtimePeerManager(runtime=cast(Any, None), fps=16, chunk_frames=2)
        rtc = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        worker = asyncio.create_task(peers._input_worker(rtc))
        try:
            rtc.input_messages.put_nowait("[]")
            rtc.input_messages.put_nowait(json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}))
            await asyncio.wait_for(_wait_for_condition(rtc.input_messages.empty), timeout=1)
            await asyncio.sleep(0)
            assert worker.done() is False
            assert rtc.resampler.effective_keys == frozenset({"w"})
        finally:
            rtc.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        ws = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=1),
        )
        await peers._handle_socket_message(ws, "[]")
        await peers._handle_socket_message(
            ws,
            json.dumps({"type": "action", "action": {"event": "keydown", "key": "d"}}),
        )
        assert ws.closed is False
        assert ws.resampler.effective_keys == frozenset({"d"})

    asyncio.run(exercise())


def test_rtc_generation_resizes_off_loop_before_frame_buffer(monkeypatch) -> None:
    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, _interactions, **_kwargs):
            return [np.full((360, 640, 3), 255, dtype=np.uint8)], 1.0

    resize_threads: list[int] = []
    resize_started = threading.Event()
    release_first_resize = threading.Event()
    original_resize = _resize_rgb_frames

    def slow_resize(frames, *, output_resolution):
        resize_threads.append(threading.get_ident())
        if len(resize_threads) == 1:
            resize_started.set()
            assert release_first_resize.wait(timeout=2)
        time.sleep(0.025)
        return original_resize(frames, output_resolution=output_resolution)

    monkeypatch.setattr(realtime_backend, "_resize_rgb_frames", slow_resize)

    async def exercise() -> None:
        channel = Channel()
        active = _ActivePeer(
            peer=None,
            channel=channel,
            frames=LatestFrameBuffer(maxsize=2),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            output_resolution=OutputResolutionState.from_value({"width": 320, "height": 240}),
        )
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        ticks = 0
        ticking = True

        async def ticker() -> None:
            nonlocal ticks
            while ticking:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker_task = asyncio.create_task(ticker())
        worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await peers._handle_message(
                active,
                json.dumps({"type": "action", "action": {"event": "step"}}),
            )
            await asyncio.wait_for(_wait_for_condition(resize_started.is_set), timeout=1)
            await peers._handle_message(
                active,
                json.dumps(
                    {
                        "type": "output_config",
                        "resolution": {"width": 160, "height": 90},
                        "request_id": "rtc-resize-race",
                    }
                ),
            )
            release_first_resize.set()
            await asyncio.wait_for(
                _wait_for_condition(lambda: any(message["type"] == "chunk_done" for message in channel.messages)),
                timeout=1,
            )
            frame = cast(np.ndarray, await active.frames.get())
            assert frame.shape == (90, 160, 3)
            assert int(frame[45, 80].min()) == 255
            assert resize_threads and resize_threads[0] != threading.get_ident()
            assert len(resize_threads) == 2
            assert ticks >= 3
            chunk = next(message for message in channel.messages if message["type"] == "chunk_done")
            assert chunk["resolution"] == {"mode": "fixed", "width": 160, "height": 90}
            assert chunk["resolution_revision"] == 1
        finally:
            release_first_resize.set()
            ticking = False
            active.closed = True
            worker.cancel()
            ticker_task.cancel()
            await asyncio.gather(worker, ticker_task, return_exceptions=True)

    asyncio.run(exercise())


def test_rtc_track_repeats_prepared_frame_without_resize(monkeypatch) -> None:
    class FakeMediaStreamTrack:
        pass

    class FakeMediaStreamError(Exception):
        pass

    class FakeVideoFrame:
        shapes: list[tuple[int, ...]] = []

        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.pts = None
            self.time_base = None

        @classmethod
        def from_ndarray(cls, array: np.ndarray, *, format: str):
            assert format == "rgb24"
            cls.shapes.append(array.shape)
            return cls(array)

        def reformat(self, **_kwargs):
            raise AssertionError("RTC recv hot path attempted to resize")

    aiortc: Any = types.ModuleType("aiortc")
    aiortc.MediaStreamTrack = FakeMediaStreamTrack
    mediastreams: Any = types.ModuleType("aiortc.mediastreams")
    mediastreams.MediaStreamError = FakeMediaStreamError
    av: Any = types.ModuleType("av")
    av.VideoFrame = FakeVideoFrame
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)
    monkeypatch.setitem(sys.modules, "aiortc.mediastreams", mediastreams)
    monkeypatch.setitem(sys.modules, "av", av)

    async def exercise() -> None:
        frames = LatestFrameBuffer(maxsize=1)
        await frames.put_chunk([np.zeros((180, 320, 3), dtype=np.uint8)])
        track = _build_video_track(frames=frames, fps=1000)
        first = await track.recv()
        second = await track.recv()
        assert first.array is second.array
        assert FakeVideoFrame.shapes == [(180, 320, 3), (180, 320, 3)]

    asyncio.run(exercise())


def test_socket_resolution_ack_preserves_old_chunk_and_reencodes_inflight_chunk(
    monkeypatch,
) -> None:
    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_str(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, _interactions, **_kwargs):
            return [np.zeros((360, 640, 3), dtype=np.uint8)], 1.0

    encode_started = threading.Event()
    release_first_encode = threading.Event()
    encoded_resolutions: list[tuple[int, int] | None] = []

    def controlled_encode(
        _frames,
        *,
        quality,
        subsampling=1,
        output_resolution=None,
    ):
        del quality, subsampling
        encoded_resolutions.append(output_resolution)
        if len(encoded_resolutions) == 1:
            encode_started.set()
            assert release_first_encode.wait(timeout=2)
        return [str(output_resolution).encode()]

    monkeypatch.setattr(realtime_backend, "_encode_jpeg_frames", controlled_encode)

    async def exercise() -> None:
        socket = Socket()
        active = _ActiveSocket(
            socket=socket,
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=4),
            output_resolution=OutputResolutionState.from_value({"width": 640, "height": 360}),
        )
        # This packet belongs to a previously announced chunk. Resolution
        # changes must not truncate it or invalidate chunk_done.frames.
        active.frame_packets.put_nowait(b"announced-old-chunk")
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        worker = asyncio.create_task(peers._socket_generation_worker(active))
        try:
            await peers._handle_socket_message(
                active,
                json.dumps({"type": "action", "action": {"event": "step"}}),
            )
            await asyncio.wait_for(_wait_for_condition(encode_started.is_set), timeout=1)
            await peers._handle_socket_message(
                active,
                json.dumps(
                    {
                        "type": "output_config",
                        "resolution": {"width": 320, "height": 240},
                        "request_id": "resolution-race",
                    }
                ),
            )
            assert active.frame_packets.qsize() == 1
            release_first_encode.set()
            await asyncio.wait_for(
                _wait_for_condition(lambda: any(message["type"] == "chunk_done" for message in socket.messages)),
                timeout=2,
            )

            assert encoded_resolutions == [(640, 360), (320, 240)]
            assert active.frame_packets.get_nowait() == b"announced-old-chunk"
            assert active.frame_packets.get_nowait() == b"(320, 240)"
            ack = next(message for message in socket.messages if message["type"] == "output_config_ack")
            chunk = next(message for message in socket.messages if message["type"] == "chunk_done")
            assert ack["status"] == "queued"
            assert ack["applies_at"] == "next_chunk"
            assert ack["resolution_revision"] == 1
            assert chunk["resolution_revision"] == 1
            assert chunk["resolution"] == {"mode": "fixed", "width": 320, "height": 240}
        finally:
            release_first_encode.set()
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def test_prompt_segment_socket_backpressures_instead_of_dropping_counted_frames(
    monkeypatch,
) -> None:
    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_str(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        async def generate(self, _interactions, **_kwargs):
            return [np.full((180, 320, 3), value, dtype=np.uint8) for value in (1, 2, 3)], 1.0

    monkeypatch.setattr(
        realtime_backend,
        "_encode_jpeg_frames",
        lambda frames, **_kwargs: [bytes([int(frame[0, 0, 0])]) for frame in frames],
    )

    async def exercise() -> None:
        socket = Socket()
        active = _ActiveSocket(
            socket=socket,
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=1),
            prompt_scheduled=True,
            initial_segment_pending=True,
        )
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=3)
        received: list[bytes] = []

        async def slow_consumer() -> None:
            while len(received) < 3:
                received.append(await active.frame_packets.get())
                await asyncio.sleep(0.01)

        consumer = asyncio.create_task(slow_consumer())
        worker = asyncio.create_task(peers._socket_generation_worker(active))
        try:
            await asyncio.wait_for(
                _wait_for_condition(lambda: any(message["type"] == "chunk_done" for message in socket.messages)),
                timeout=2,
            )
            await consumer
            chunk = next(message for message in socket.messages if message["type"] == "chunk_done")
            assert received == [b"\x01", b"\x02", b"\x03"]
            assert chunk["frames"] == 3
            assert chunk["dropped_frames"] == 0
        finally:
            active.closed = True
            worker.cancel()
            consumer.cancel()
            await asyncio.gather(worker, consumer, return_exceptions=True)

    asyncio.run(exercise())


def test_rtc_resolution_ack_keeps_frames_from_announced_chunk() -> None:
    async def exercise() -> None:
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        buffer = LatestFrameBuffer(maxsize=2)
        await buffer.put_chunk([frame])
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=buffer,
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            output_resolution=OutputResolutionState.from_value({"mode": "native"}),
        )
        peers = RealtimePeerManager(runtime=cast(Any, None), fps=16, chunk_frames=2)

        ack = peers._handle_session_control(
            active,
            {
                "type": "output_config",
                "resolution": {"width": 160, "height": 90},
                "request_id": "rtc-next-chunk",
            },
        )

        assert ack is not None
        assert ack["ok"] is True
        assert ack["status"] == "queued"
        assert ack["applies_at"] == "next_chunk"
        assert buffer.qsize() == 1
        assert await buffer.get() is frame

    asyncio.run(exercise())


def test_realtime_client_exposes_event_resolution_step_and_panel_controls() -> None:
    entry = find_entry("lingbot-world")
    html = world_frontend_html(
        entry,
        StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
    )
    for element_id in (
        "statusPanel",
        "controlsPanel",
        "scenePanel",
        "runtimeLog",
        "stepButton",
        "pauseButton",
        "pointerHint",
        "interactionHint",
        "resolutionSelect",
        "eventEditorRows",
        "eventTriggerBar",
    ):
        assert f'id="{element_id}"' in html
    for marker in (
        'type: "event_catalog"',
        'type: "event"',
        'acceptAck("catalog", message)',
        'acceptAck("event", message)',
        'acceptAck("step", message)',
        'acceptAck("output", message)',
        "message.request_id !== expected",
        'action: { event: "step" }',
        'type: "output_config"',
        "setupPanels()",
        "requestAnimationFrame(paintDrag)",
        "if (record.floating && Number.isFinite(record.left)",
        "if (state.channel || state.peer || state.socket) await closePeer(false)",
    ):
        assert marker in world_frontend_js()


def test_control_state_resolves_conflicts_by_last_press() -> None:
    state = RealtimeControlState()
    assert state.apply("keydown", "w")
    assert state.apply("keydown", "s")
    assert state.effective() == frozenset({"s"})
    assert state.apply("keyup", "s")
    assert state.effective() == frozenset({"w"})


def test_resampler_preserves_edges_inside_chunk() -> None:
    resampler = RealtimeControlResampler(fps=10, start_time=10.0)
    assert resampler.on_edge(arrival_time=10.05, event="keydown", key="w")
    assert resampler.on_edge(arrival_time=10.25, event="keyup", key="w")

    segments = resampler.sample_chunk(4, wall_time=10.0)

    assert any(keys == frozenset({"w"}) for _, _, keys in segments)
    assert interactions_from_segments(segments) == ["forward"]
    assert resampler.effective_keys == frozenset()


def test_datachannel_mailbox_applies_edges_in_order() -> None:
    async def exercise() -> None:
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        peers = RealtimePeerManager(
            runtime=cast(Any, None),
            fps=16,
            chunk_frames=9,
        )
        task = asyncio.create_task(peers._input_worker(active))
        try:
            for event, key in (("keydown", "w"), ("keydown", "s")):
                active.input_messages.put_nowait(json.dumps({"type": "action", "action": {"event": event, "key": key}}))
            while not active.input_messages.empty():
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert active.resampler.effective_keys == frozenset({"s"})
        finally:
            active.closed = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_frame_buffer_backpressures_then_evicts_stale_frames() -> None:
    async def exercise() -> None:
        buffer = LatestFrameBuffer(maxsize=1, backpressure_ms=100)
        first = np.full((2, 2, 3), 1, dtype=np.uint8)
        second = np.full((2, 2, 3), 2, dtype=np.uint8)
        await buffer.put_chunk([first])

        async def consume() -> None:
            await asyncio.sleep(0.01)
            assert int(cast(np.ndarray, await buffer.get())[0, 0, 0]) == 1

        consumer = asyncio.create_task(consume())
        assert await buffer.put_chunk([second]) == 1
        await consumer
        assert buffer.dropped_frames == 0
        assert int(cast(np.ndarray, await buffer.get())[0, 0, 0]) == 2

        stale = LatestFrameBuffer(maxsize=1)
        await stale.put_chunk([first, second])
        assert stale.dropped_frames == 1
        assert int(cast(np.ndarray, await stale.get())[0, 0, 0]) == 2

    asyncio.run(exercise())


def test_ordered_quality_frame_buffer_backpressures_without_drops() -> None:
    async def exercise() -> None:
        buffer = LatestFrameBuffer(
            maxsize=1,
            policy=FrameQueuePolicy.ORDERED_QUALITY,
        )
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.ones((2, 2, 3), dtype=np.uint8)
        assert await buffer.put_chunk([first]) == 1

        blocked = asyncio.create_task(buffer.put_chunk([second]))
        await asyncio.sleep(0.01)
        assert blocked.done() is False
        np.testing.assert_array_equal(cast(np.ndarray, await buffer.get()), first)

        assert await asyncio.wait_for(blocked, timeout=0.5) == 1
        np.testing.assert_array_equal(cast(np.ndarray, await buffer.get()), second)
        assert buffer.dropped_frames == 0
        buffer.close()

    asyncio.run(exercise())


def test_frame_queue_policy_aliases_and_validation() -> None:
    assert FrameQueuePolicy.from_value("latest") is FrameQueuePolicy.LATEST_INTERACTIVE
    assert FrameQueuePolicy.from_value("ordered_quality") is FrameQueuePolicy.ORDERED_QUALITY
    with pytest.raises(ValueError, match="Unknown frame queue policy"):
        FrameQueuePolicy.from_value("unbounded")


def test_realtime_profile_preserves_quality_and_disables_offline_actions() -> None:
    lingbot = find_entry("lingbot-world")
    quality = _realtime_overrides(
        lingbot,
        {"action_path": "/tmp/replay", "sampling_steps": 20},
    )
    distilled = _realtime_overrides(
        lingbot,
        {"sampling_steps": 20},
        inference_steps=4,
    )
    matrix = _realtime_overrides(
        find_entry("matrix-game-2"),
        {"official_bench_actions": True},
    )

    assert quality["action_path"] is None
    assert quality["sampling_steps"] == 20
    assert distilled["sampling_steps"] == 4
    assert matrix["official_bench_actions"] is False
    assert _realtime_frame_budget(find_entry("matrix-game-2"), 9) == 9


def test_dreamx_interactive_defaults_are_latency_oriented_and_overridable(
    monkeypatch,
) -> None:
    entry = find_entry("dreamx-world-5b-cam")
    launch = StudioLaunchConfig(model_id=entry.model_id, frontend="world")

    assert _default_realtime_chunk_frames(entry) == 5
    assert _default_realtime_inference_steps(entry, launch) == 4

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_INFERENCE_STEPS", "7")
    assert _default_realtime_inference_steps(entry, launch) == 7


def test_resident_warmup_deadline_resets_configured_model_without_cancelling_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "seed.png"
    Image.new("RGB", (8, 6), "gray").save(image_path)
    actions: list[str] = []

    class Manager:
        def run_realtime(self, *, entry, request, action):
            del entry, request
            actions.append(action)
            if action == "configure":
                time.sleep(0.02)
            return {}

    entry = find_entry("infinite-world")
    request = PreparedInputs(
        prompt="world",
        input_path=str(image_path),
        image=Image.new("RGB", (8, 6), "gray"),
        image_path=str(image_path),
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=[],
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=16,
        num_frames=2,
        output_dir=str(tmp_path / "unused"),
        output_path=str(tmp_path / "unused" / "output.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    runtime = ResidentWorldRuntime(
        manager=cast(Any, Manager()),
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
        fps=16,
        warmup_image_path=str(image_path),
        warmup_chunks=2,
    )
    cast(Any, runtime)._build_request = lambda **_kwargs: request
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PREWARM_TIMEOUT_SECONDS", "0.001")

    try:
        with pytest.raises(PrewarmTimeoutError):
            asyncio.run(runtime._warmup())
    finally:
        runtime._executor.shutdown(wait=True, cancel_futures=True)

    assert actions == ["configure", "reset"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_realtime_shutdown_timeout_must_be_positive_and_finite(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="positive finite"):
        realtime_backend._realtime_shutdown_timeout_s()


def test_close_active_uses_one_deadline_for_all_teardown_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deadlines: list[float] = []
    original_finish = realtime_backend._finish_tasks_by_deadline
    original_run = realtime_backend._run_by_shutdown_deadline

    async def recording_finish(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return await original_finish(*args, **kwargs)

    async def recording_run(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return await original_run(*args, **kwargs)

    monkeypatch.setattr(realtime_backend, "_finish_tasks_by_deadline", recording_finish)
    monkeypatch.setattr(realtime_backend, "_run_by_shutdown_deadline", recording_run)

    class Peer:
        async def close(self) -> None:
            pass

    class Runtime:
        async def reset(self) -> None:
            pass

    async def exercise() -> None:
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        await peers.close_active()

    asyncio.run(exercise())
    assert len(deadlines) >= 3
    assert all(deadline == deadlines[0] for deadline in deadlines)


def test_resident_runtime_close_bounds_cancellation_resistant_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    async def exercise() -> None:
        release = asyncio.Event()
        started = asyncio.Event()
        entry = find_entry("infinite-world")
        runtime = ResidentWorldRuntime(
            manager=cast(Any, object()),
            entry=entry,
            launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
            fps=16,
        )

        async def resistant_reset() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        cast(Any, runtime).reset = resistant_reset
        close_task = asyncio.create_task(runtime.close())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert started.is_set()
            assert runtime._shutdown_tasks

            # close() is idempotent even while the cooperatively cancelled
            # reset is still winding down in the background.
            await runtime.close()
        finally:
            release.set()
            await asyncio.gather(
                close_task,
                *tuple(runtime._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_resident_runtime_close_bounds_cancellation_resistant_preload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    async def exercise() -> None:
        release = asyncio.Event()
        started = asyncio.Event()
        entry = find_entry("infinite-world")
        runtime = ResidentWorldRuntime(
            manager=cast(Any, object()),
            entry=entry,
            launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
            fps=16,
        )

        async def resistant_preload() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        preload_task = asyncio.create_task(resistant_preload())
        runtime._preload_task = preload_task
        await started.wait()
        close_task = asyncio.create_task(runtime.close())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert preload_task in runtime._shutdown_tasks
        finally:
            release.set()
            await asyncio.gather(close_task, preload_task, return_exceptions=True)

    asyncio.run(exercise())


def test_resident_runtime_close_does_not_join_running_executor_preload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    async def exercise() -> None:
        entry = find_entry("infinite-world")
        runtime = ResidentWorldRuntime(
            manager=cast(Any, object()),
            entry=entry,
            launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
            fps=16,
        )

        def blocking_preload() -> None:
            started.set()
            release.wait()
            finished.set()

        async def run_preload() -> None:
            await runtime._run(blocking_preload)

        preload_task = asyncio.create_task(run_preload())
        runtime._preload_task = preload_task
        while not started.is_set():
            await asyncio.sleep(0)
        close_task = asyncio.create_task(runtime.close())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
        finally:
            release.set()
            await asyncio.gather(close_task, preload_task, return_exceptions=True)

    asyncio.run(exercise())
    assert finished.wait(timeout=1)


def test_close_active_bounds_peer_and_socket_close_and_releases_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Runtime:
        async def reset(self) -> None:
            pass

    async def exercise_peer() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        class Peer:
            async def close(self) -> None:
                started.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass

        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        close_task = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert started.is_set()
            assert peers.draining is False
            assert peers._drain_done.is_set()
            assert peers._shutdown_tasks
            await peers.close_active()
        finally:
            release.set()
            await asyncio.gather(
                close_task,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    async def exercise_socket() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        class Socket:
            closed = False

            async def close(self) -> None:
                started.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass

        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active_socket = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=1),
        )
        close_task = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert started.is_set()
            assert peers.draining is False
            assert peers._drain_done.is_set()
            assert peers._shutdown_tasks
        finally:
            release.set()
            await asyncio.gather(
                close_task,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise_peer())
    asyncio.run(exercise_socket())


def test_concurrent_close_active_waiter_shares_bounded_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Runtime:
        async def reset(self) -> None:
            pass

    async def exercise() -> None:
        release = asyncio.Event()
        peer_close_started = asyncio.Event()

        class Peer:
            async def close(self) -> None:
                peer_close_started.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass

        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        leader = asyncio.create_task(peers.close_active())
        await peer_close_started.wait()
        follower = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({leader, follower}, timeout=1)
            assert done == {leader, follower}
            await asyncio.gather(leader, follower)
            assert peers.draining is False
            assert peers._drain_done.is_set()
        finally:
            release.set()
            await asyncio.gather(
                leader,
                follower,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_close_active_bounds_cancellation_resistant_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Peer:
        async def close(self) -> None:
            pass

    class Runtime:
        async def reset(self) -> None:
            pass

    async def exercise() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        async def resistant_worker() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        worker = asyncio.create_task(resistant_worker())
        await started.wait()
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            generation_task=worker,
        )
        close_task = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert peers.draining is False
            assert worker in peers._shutdown_tasks
        finally:
            release.set()
            await asyncio.gather(
                close_task,
                worker,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_close_active_bounds_cancellation_resistant_runtime_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Peer:
        async def close(self) -> None:
            pass

    async def exercise() -> None:
        release = asyncio.Event()
        reset_started = asyncio.Event()

        class Runtime:
            async def reset(self) -> None:
                reset_started.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass

        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
        )
        close_task = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert reset_started.is_set()
            assert peers.draining is False
            assert peers._drain_done.is_set()
            assert peers._shutdown_tasks
        finally:
            release.set()
            await asyncio.gather(
                close_task,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_close_active_bounds_session_lock_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    class Peer:
        async def close(self) -> None:
            pass

    class Runtime:
        async def reset(self) -> None:
            pass

    async def exercise() -> None:
        worker_cancelled = asyncio.Event()

        async def worker() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                worker_cancelled.set()
                raise

        worker_task = asyncio.create_task(worker())
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        peers._active = _ActivePeer(
            peer=Peer(),
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            generation_task=worker_task,
        )
        await peers._lock.acquire()
        lock_held = True
        close_task = asyncio.create_task(peers.close_active())
        try:
            done, _ = await asyncio.wait({close_task}, timeout=1)
            assert close_task in done
            await close_task
            assert worker_cancelled.is_set() is False
            assert peers._deferred_close_task is not None
            peers._lock.release()
            lock_held = False
            await asyncio.wait_for(worker_cancelled.wait(), timeout=1)
            deferred = peers._deferred_close_task
            if deferred is not None:
                await asyncio.wait_for(asyncio.shield(deferred), timeout=1)
            assert peers._active is None
            assert worker_task.done()
        finally:
            if lock_held:
                peers._lock.release()
            worker_task.cancel()
            deferred = peers._deferred_close_task
            await asyncio.gather(
                close_task,
                worker_task,
                *(task for task in (deferred,) if task is not None),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_socket_worker_does_not_gather_cancellation_resistant_inference() -> None:
    class Socket:
        closed = False

        async def send_str(self, _raw: str) -> None:
            pass

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def next_chunk_frames(self, default: int) -> int:
            return default

    async def exercise() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        async def resistant_generate(*_args, **_kwargs):
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return [], 0.0

        runtime: Any = Runtime()
        runtime.generate = resistant_generate
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)
        active = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=1),
            pending_steps=1,
        )
        active.first_action.set()
        worker = asyncio.create_task(peers._socket_generation_worker(active))
        await asyncio.wait_for(started.wait(), timeout=1)
        worker.cancel()
        try:
            done, _ = await asyncio.wait({worker}, timeout=1)
            assert worker in done
            await asyncio.gather(worker, return_exceptions=True)
            assert peers._shutdown_tasks
        finally:
            release.set()
            await asyncio.gather(
                worker,
                *tuple(peers._shutdown_tasks),
                return_exceptions=True,
            )

    asyncio.run(exercise())


def test_queued_segment_request_keeps_native_frames_steps_and_user_controls(tmp_path: Path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = find_entry("longvie-2")
    runtime = ResidentWorldRuntime(
        manager=manager,
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
        fps=16,
    )
    try:
        request = runtime._build_request(
            prompt="ride through snow",
            image=Image.new("RGB", (8, 6), "white"),
            input_path="/tmp/seed.png",
            video_path="",
            dense_video_path="/uploads/depth.mp4",
            sparse_video_path="/uploads/track.mp4",
        )
    finally:
        runtime._executor.shutdown(wait=False, cancel_futures=True)

    call_kwargs = request.call_kwargs
    assert call_kwargs["num_frames"] == 81
    assert call_kwargs["num_inference_steps"] == 50
    assert call_kwargs["seed"] == 0
    assert call_kwargs["dense_video"] == "/uploads/depth.mp4"
    assert call_kwargs["sparse_video"] == "/uploads/track.mp4"
    assert call_kwargs["return_dict"] is True
    assert _realtime_frame_budget(entry, 9) == 81


def test_queued_segment_runtime_runs_fresh_then_extends_resident_state(tmp_path: Path) -> None:
    entry = find_entry("longvie-2")
    calls: list[tuple[str, PreparedInputs]] = []

    class Manager:
        workspace_root = str(tmp_path)

        def run_realtime(self, *, entry, request, action):
            del entry
            calls.append((action, request))
            return {
                "video": np.zeros((81, 2, 3, 3), dtype=np.uint8),
                "realtime_spec": {
                    "fps": 16,
                    "first_chunk_frames": 81,
                    "steady_chunk_frames": 81,
                    "controls": ["dense_depth_video", "sparse_pointmap_or_track_video"],
                    "transport": "queued-segment-rgb",
                },
            }

    request = PreparedInputs(
        prompt="first segment",
        input_path="/uploads/seed.png",
        image=Image.new("RGB", (3, 2), "white"),
        image_path="/uploads/seed.png",
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=[],
        camera_view=None,
        task_type="image-to-video",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=16,
        num_frames=81,
        output_dir=str(tmp_path / "unused"),
        output_path=str(tmp_path / "unused" / "must-not-exist.mp4"),
        call_kwargs={
            "execute": True,
            "num_frames": 81,
            "num_inference_steps": 50,
            "seed": 0,
            "dense_video": "/uploads/depth-0.mp4",
            "sparse_video": "/uploads/track-0.mp4",
            "return_dict": True,
        },
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cuda",
    )
    runtime = ResidentWorldRuntime(
        manager=cast(Any, Manager()),
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
        fps=16,
    )
    runtime._base_request = request
    runtime._configured = True

    async def exercise() -> None:
        first, _ = await runtime.generate([], seed=99)
        second, _ = await runtime.generate(
            [],
            seed=100,
            prompt="second segment",
            dense_video_path="/uploads/depth-1.mp4",
            sparse_video_path="/uploads/track-1.mp4",
        )
        assert len(first) == len(second) == 81
        await runtime.close()

    asyncio.run(exercise())

    assert [action for action, _ in calls[:2]] == ["run", "stream"]
    assert calls[0][1].call_kwargs["seed"] == 99
    continuation = calls[1][1]
    assert continuation.call_kwargs["seed"] == 100
    assert continuation.prompt == "second segment"
    assert continuation.input_path == ""
    assert continuation.image is None
    assert continuation.image_path is None
    assert continuation.call_kwargs["dense_video"] == "/uploads/depth-1.mp4"
    assert continuation.call_kwargs["sparse_video"] == "/uploads/track-1.mp4"
    assert not (tmp_path / "unused" / "must-not-exist.mp4").exists()


def test_realtime_spec_uses_model_owned_causal_cadence() -> None:
    fallback = RealtimeSpec(fps=16, first_chunk_frames=9, steady_chunk_frames=9)

    parsed = RealtimeSpec.from_payload(
        {
            "realtime_spec": {
                "fps": 16,
                "first_chunk_frames": 13,
                "steady_chunk_frames": 16,
            }
        },
        fallback=fallback,
    )

    assert parsed.first_chunk_frames == 13
    assert parsed.steady_chunk_frames == 16


def test_realtime_spec_can_split_input_and_output_cadence() -> None:
    legacy = RealtimeSpec(fps=24, first_chunk_frames=8, steady_chunk_frames=16)
    assert legacy.resolved_input_fps == 24.0
    assert legacy.resolved_first_input_frames == 8
    assert legacy.resolved_steady_input_frames == 16
    assert "input_fps" not in legacy.to_payload()

    split = RealtimeSpec.from_payload(
        {
            "realtime_spec": {
                "fps": 24,
                "first_chunk_frames": 8,
                "steady_chunk_frames": 16,
                "input_fps": 6.25,
                "first_input_frames": 2,
                "steady_input_frames": 4,
            }
        }
    )

    assert split.resolved_input_fps == pytest.approx(6.25)
    assert split.resolved_first_input_frames == 2
    assert split.resolved_steady_input_frames == 4
    assert split.to_payload()["input_fps"] == pytest.approx(6.25)


def test_resampler_accepts_fractional_input_fps() -> None:
    resampler = RealtimeControlResampler(fps=6.25, start_time=1.0)

    assert resampler.fps == pytest.approx(6.25)
    assert resampler.dt == pytest.approx(0.16)


def test_frame_extraction_prefers_in_memory_video() -> None:
    video = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    frames = realtime_frames_from_result({"artifact_path": "/tmp/must-not-read.mp4", "video": video})
    assert len(frames) == 3
    assert frames[0].shape == (4, 5, 3)


def test_frame_extraction_supports_torch_thwc_uint8_chunks() -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros((3, 4, 5, 3), dtype=torch.uint8)
    video[2, 0, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8)

    frames = realtime_frames_from_result({"video": video})

    assert len(frames) == 3
    assert frames[0].shape == (4, 5, 3)
    assert frames[0].dtype == np.uint8
    assert frames[2][0, 0].tolist() == [10, 20, 30]


def test_frame_extraction_prefers_tchw_when_width_is_three() -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros((2, 3, 4, 3), dtype=torch.uint8)
    video[1, :, 0, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8)

    frames = realtime_frames_from_result({"video": video})

    assert len(frames) == 2
    assert frames[1].shape == (4, 3, 3)
    assert frames[1][0, 0].tolist() == [10, 20, 30]


def test_frame_extraction_consumes_and_closes_lazy_iterator() -> None:
    closed = False

    def chunks():
        nonlocal closed
        try:
            yield np.zeros((4, 5, 3), dtype=np.uint8)
            yield np.full((4, 5, 3), 255, dtype=np.uint8)
        finally:
            closed = True

    frames = realtime_frames_from_result(chunks())

    assert len(frames) == 2
    assert closed is True


def test_frame_extraction_starts_all_lazy_prefetches_before_materializing() -> None:
    events: list[str] = []

    class LazyFrame:
        def __init__(self, index: int) -> None:
            self.index = index

        def prefetch_to_numpy(self) -> None:
            events.append(f"prefetch:{self.index}")

        def to_numpy(self) -> np.ndarray:
            assert events[:2] == ["prefetch:0", "prefetch:1"]
            events.append(f"materialize:{self.index}")
            return np.full((4, 5, 3), self.index, dtype=np.uint8)

    frames = realtime_frames_from_result({"frames": [LazyFrame(0), LazyFrame(1)]})

    assert events == ["prefetch:0", "prefetch:1", "materialize:0", "materialize:1"]
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1]


class _Memory:
    def __init__(self) -> None:
        self.value = None

    def manage(self, action: str = "reset") -> None:
        if action == "reset":
            self.value = None

    def record(self, value, **_kwargs) -> None:
        self.value = value


class _Pipeline:
    def __init__(self) -> None:
        self.memory_module = _Memory()
        self.calls: list[dict[str, object]] = []

    def stream(self, images, interactions, **kwargs):
        self.calls.append({"images": images, "interactions": interactions, **kwargs})
        return np.full((2, 6, 8, 3), 127, dtype=np.uint8)


class _Driver(BaseRuntimeDriver):
    def __init__(self, pipeline: _Pipeline) -> None:
        self.pipeline = pipeline

    def load_pipeline(self, manager, entry, request, progress_callback=None):
        del manager, request, progress_callback
        return PipelineContext(
            entry=entry,
            pipeline=self.pipeline,
            cache_key="realtime-test",
            backend="from_pretrained",
            model_ref="",
            endpoint="",
            load_kwargs={},
            device="cpu",
        )


def test_realtime_manager_keeps_seed_in_memory_and_skips_materialization(
    tmp_path: Path,
) -> None:
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = find_entry("infinite-world")
    pipeline = _Pipeline()
    driver = _Driver(pipeline)
    cast(Any, manager).runtime_driver_for = lambda _entry: driver
    output_dir = tmp_path / "live"
    request = PreparedInputs(
        prompt="world",
        input_path=str(tmp_path / "seed.png"),
        image=Image.new("RGB", (8, 6), "red"),
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
        scene_name="",
        fps=16,
        num_frames=9,
        output_dir=str(output_dir),
        output_path=str(output_dir / "must-not-exist.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )

    manager.run_realtime(entry=entry, request=request, action="configure")
    result = manager.run_realtime(entry=entry, request=request, action="stream")

    assert isinstance(result, np.ndarray)
    assert pipeline.calls[0]["images"] is None
    assert not output_dir.exists()


class _AxisSink:
    readyState = "open"
    closed = False

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))

    async def send_str(self, raw: str) -> None:
        self.send(raw)


def _axis_session(transport: str, sink: _AxisSink) -> _ActivePeer | _ActiveSocket:
    resampler = RealtimeControlResampler(
        fps=20,
        start_time=asyncio.get_running_loop().time(),
        axes=(RealtimeAxis("yaw", "mouse_x"),),
    )
    if transport == "socket":
        return _ActiveSocket(socket=sink, resampler=resampler, frame_packets=asyncio.Queue(maxsize=8))
    return _ActivePeer(peer=None, channel=sink, resampler=resampler, frames=LatestFrameBuffer(maxsize=8))


async def _handle_axis_message(
    peers: RealtimePeerManager,
    active: _ActivePeer | _ActiveSocket,
    payload: dict[str, Any],
) -> None:
    if isinstance(active, _ActiveSocket):
        await peers._handle_socket_message(active, json.dumps(payload))
    else:
        await peers._handle_message(active, json.dumps(payload))


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_axis_messages_feed_declared_axes_on_both_transports(transport: str) -> None:
    async def exercise() -> None:
        active = _axis_session(transport, _AxisSink())
        peers = RealtimePeerManager(runtime=cast(Any, None), fps=20, chunk_frames=4)
        await _handle_axis_message(peers, active, {"type": "axis", "source": "gamepad", "lx": 0.8})
        assert not active.first_action.is_set()
        assert not active.resampler.has_pending_input
        await _handle_axis_message(peers, active, {"type": "axis", "source": "mouse", "dx": 3.0, "dy": 1.0})
        assert active.first_action.is_set()
        assert len(active.action_arrivals) == 1
        chunk = active.resampler.sample_input_chunk(4, wall_time=asyncio.get_running_loop().time())
        assert sum(chunk.axes["yaw"]) == pytest.approx(3.0)

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_axis_only_input_generates_with_chunk_axes(transport: str) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Runtime:
        last_generation_metrics: dict[str, float] = {}

        def next_chunk_frames(self, default: int) -> int:
            return default

        async def generate(self, interactions: list[str], **kwargs: Any) -> tuple[list[np.ndarray], float]:
            calls.append((list(interactions), kwargs))
            return [np.zeros((4, 6, 3), dtype=np.uint8)], 1.0

    async def exercise() -> None:
        sink = _AxisSink()
        active = _axis_session(transport, sink)
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=20, chunk_frames=4)
        await _handle_axis_message(peers, active, {"type": "axis", "source": "mouse", "dx": 2.5})
        if isinstance(active, _ActiveSocket):
            worker = asyncio.create_task(peers._socket_generation_worker(active))
        else:
            worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: bool(calls) and not active.first_action.is_set()), 2)
            assert len(calls) == 1
            interactions, kwargs = calls[0]
            assert interactions == []
            assert kwargs["control_segments"] is not None
            assert sum(kwargs["control_axes"]["yaw"]) == pytest.approx(2.5)
            assert not any(message["type"] == "error" for message in sink.messages)
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())


def test_interactive_call_kwargs_add_realtime_axes_only_when_declared() -> None:
    segments = [(0.0, 0.1, frozenset({"w"}))]
    without_axes = realtime_backend._interactive_call_kwargs(
        {}, ["forward"], seed=3, control_segments=segments, control_axes={}
    )
    assert without_axes == {"seed": 3, "realtime_segments": [{"duration": 0.1, "keys": ["w"]}]}
    with_axes = realtime_backend._interactive_call_kwargs(
        {}, ["forward"], seed=3, control_segments=segments, control_axes={"yaw": [0.0, 0.5]}
    )
    assert with_axes["realtime_axes"] == {"yaw": [0.0, 0.5]}
    assert with_axes["realtime_segments"] == without_axes["realtime_segments"]


@pytest.mark.unit
def test_interactive_call_kwargs_do_not_replay_previous_chunk_controls() -> None:
    base = {
        "realtime_segments": [{"duration": 0.75, "keys": ["w"]}],
        "realtime_axes": {"yaw": [0.5] * 9},
        "num_inference_steps": 4,
    }
    step = realtime_backend._interactive_call_kwargs(base, [], seed=4)
    assert step == {"num_inference_steps": 4, "seed": 4}
    assert base["realtime_segments"] == [{"duration": 0.75, "keys": ["w"]}]
    assert base["realtime_axes"] == {"yaw": [0.5] * 9}


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_sampled_controls_and_idle_step_reach_matrix_game_2_stream_realtime(tmp_path: Path, transport: str) -> None:
    from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
        MatrixGame2RealtimeSession,
    )

    declaring_session = object.__new__(MatrixGame2RealtimeSession)
    declaring_session.runtime = types.SimpleNamespace(mode="universal")
    spec = declaring_session.realtime_spec()
    generate_calls: list[dict[str, Any]] = []

    class RecordingSession:
        """MG2-like resident session that declares the universal camera axes."""

        def realtime_spec(self) -> RealtimeSpec:
            return spec

        def configure(self, image: Image.Image, *, seed: int) -> dict[str, Any]:
            del image, seed
            return {"status": "configured", "realtime_spec": spec.to_payload()}

        def generate(self, **kwargs: Any) -> dict[str, Any]:
            frames = spec.steady_chunk_frames if generate_calls else spec.first_chunk_frames
            generate_calls.append(kwargs)
            return {"video": np.zeros((frames, 6, 8, 3), dtype=np.uint8), "realtime_spec": spec.to_payload()}

        def reset(self) -> None:
            pass

    pipeline = MatrixGame2Pipeline(device="cpu")
    pipeline._realtime_session = RecordingSession()
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    driver = _Driver(cast(Any, pipeline))
    # A cached context keeps the configured stream state between Studio actions.
    context = driver.load_pipeline(manager, find_entry("matrix-game-2"), None)
    manager.pipeline_cache[context.cache_key] = context
    cast(Any, driver).load_pipeline = lambda *_args, **_kwargs: context
    cast(Any, manager).runtime_driver_for = lambda _entry: driver
    seed_path = tmp_path / "seed.png"
    Image.new("RGB", (8, 6), "red").save(seed_path)
    runtime = ResidentWorldRuntime(
        manager=manager,
        entry=find_entry("matrix-game-2"),
        launch_config=StudioLaunchConfig(model_id="matrix-game-2", frontend="world"),
        fps=12,
    )

    async def exercise() -> None:
        await runtime.configure(prompt="", image_path=str(seed_path), video_path="")
        assert runtime.realtime_spec.axes == spec.axes
        sink = _AxisSink()
        resampler = RealtimeControlResampler(
            fps=runtime.realtime_spec.resolved_input_fps,
            start_time=asyncio.get_running_loop().time(),
            realtime_keys=runtime.realtime_spec.keys,
            axes=runtime.realtime_spec.axes,
        )
        active: _ActivePeer | _ActiveSocket
        if transport == "socket":
            active = _ActiveSocket(socket=sink, resampler=resampler, frame_packets=asyncio.Queue(maxsize=64))
        else:
            active = _ActivePeer(peer=None, channel=sink, resampler=resampler, frames=LatestFrameBuffer(maxsize=64))
        peers = RealtimePeerManager(runtime=runtime, fps=12, chunk_frames=9)

        async def tap(key: str) -> None:
            for event in ("keydown", "keyup"):
                await _handle_axis_message(peers, active, {"type": "action", "action": {"event": event, "key": key}})

        async def wait_for_chunks(count: int) -> None:
            await asyncio.wait_for(
                _wait_for_condition(
                    lambda: len([message for message in sink.messages if message["type"] == "chunk_done"]) == count
                    and not active.first_action.is_set()
                ),
                timeout=10,
            )

        await _handle_axis_message(peers, active, {"type": "axis", "source": "mouse", "dx": 8.0})
        await tap("w")
        worker = asyncio.create_task(
            peers._socket_generation_worker(active)
            if isinstance(active, _ActiveSocket)
            else peers._generation_worker(active)
        )
        try:
            await wait_for_chunks(1)
            await _handle_axis_message(
                peers, active, {"type": "action", "action": {"event": "step"}, "request_id": "idle-step"}
            )
            await wait_for_chunks(2)
            await tap("a")
            await wait_for_chunks(3)
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await runtime.close()
        assert not any(message["type"] == "error" for message in sink.messages)
        assert any(
            message["type"] == "step_ack" and message["ok"] and message["request_id"] == "idle-step"
            for message in sink.messages
        )

    asyncio.run(exercise())

    realtime_axes = generate_calls[0]["realtime_axes"]
    assert set(realtime_axes) == {axis.name for axis in spec.axes}
    assert all(len(values) == 9 for values in realtime_axes.values())
    assert sum(realtime_axes["camera_yaw"]) == pytest.approx(0.02)
    assert not any(realtime_axes["camera_pitch"])
    assert [call["interactions"] for call in generate_calls] == [["forward"], [], ["left"]]
    assert any(segment["keys"] == ["w"] for segment in generate_calls[0]["control_segments"])
    assert generate_calls[1]["control_segments"] is None
    assert generate_calls[1]["realtime_axes"] is None
    assert any(segment["keys"] == ["a"] for segment in generate_calls[2]["control_segments"])
    assert all(len(values) == 12 and not any(values) for values in generate_calls[2]["realtime_axes"].values())


class _FakeChannel:
    readyState = "open"

    def __init__(self, label: str) -> None:
        self.label = label
        self.handlers: dict[str, Callable[..., Any]] = {}
        self.sent: list[dict[str, Any]] = []

    def on(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers[name] = handler
            return handler

        return register

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _install_fake_aiortc(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace aiortc and the video track with fakes; return the peer connections ``create_answer`` builds."""
    peers_created: list[Any] = []

    class FakePeerConnection:
        connectionState = "new"

        def __init__(self, _configuration: object) -> None:
            self.handlers: dict[str, Callable[..., Any]] = {}
            self.localDescription = types.SimpleNamespace(sdp="", type="answer")
            peers_created.append(self)

        def addTrack(self, _track: object) -> object:  # noqa: N802 - aiortc compatibility
            return object()

        def on(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def register(handler: Callable[..., Any]) -> Callable[..., Any]:
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description: object) -> None:  # noqa: N802
            pass

        async def createAnswer(self) -> object:  # noqa: N802
            return types.SimpleNamespace(sdp="candidate", type="answer")

        async def setLocalDescription(self, _description: object) -> None:  # noqa: N802
            self.localDescription = types.SimpleNamespace(sdp="answer-sdp", type="answer")

        async def close(self) -> None:
            self.connectionState = "closed"

    async def no_ice_wait(_peer: object) -> None:
        pass

    aiortc: Any = types.ModuleType("aiortc")
    aiortc.RTCConfiguration = lambda **kwargs: types.SimpleNamespace(**kwargs)
    aiortc.RTCIceServer = lambda **kwargs: types.SimpleNamespace(**kwargs)
    aiortc.RTCPeerConnection = FakePeerConnection
    aiortc.RTCSessionDescription = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)
    monkeypatch.setattr(realtime_backend, "_build_video_track", lambda **_kwargs: types.SimpleNamespace())
    monkeypatch.setattr(realtime_backend, "bind_video_sender", lambda *_args: None)
    monkeypatch.setattr(realtime_backend, "_configure_nvenc_h264", lambda *_args: None)
    monkeypatch.setattr(realtime_backend, "_wait_for_ice_gathering", no_ice_wait)
    return peers_created


def test_axes_datachannel_feeds_input_without_restarting_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    peers_created = _install_fake_aiortc(monkeypatch)

    class Runtime:
        queued_segment_generation = False
        entry = types.SimpleNamespace(default_prompt="", default_call_kwargs={})
        realtime_spec = RealtimeSpec(
            fps=20, first_chunk_frames=4, steady_chunk_frames=4, axes=(RealtimeAxis("yaw", "mouse_x"),)
        )
        transport_native_resolution = None
        supports_text_events = False
        last_generation_metrics: dict[str, float] = {}

        async def configure(self, **_kwargs: Any) -> None:
            pass

        async def reset(self) -> None:
            pass

        def steady_chunk_frames(self, _default: int) -> int:
            return 4

    async def exercise() -> None:
        manager = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=20, chunk_frames=4)
        await manager.create_answer(offer={"sdp": "offer-sdp", "type": "offer"}, session={})
        active = manager._active
        assert active is not None
        assert active.resampler.axes == Runtime.realtime_spec.axes
        try:
            on_datachannel = peers_created[0].handlers["datachannel"]
            controls = _FakeChannel("controls")
            on_datachannel(controls)
            assert active.channel is controls
            assert [message["type"] for message in controls.sent] == ["ready"]
            tasks = (active.generation_task, active.liveness_task, active.input_task)
            assert all(task is not None for task in tasks)
            active.resampler.on_edge(arrival_time=asyncio.get_running_loop().time(), event="keydown", key="w")

            axes = _FakeChannel("axes")
            on_datachannel(axes)
            assert active.channel is controls
            assert (active.generation_task, active.liveness_task, active.input_task) == tasks
            assert active.resampler.effective_keys == frozenset({"w"})
            assert set(axes.handlers) == {"message"}
            assert axes.sent == []
            assert [message["type"] for message in controls.sent] == ["ready"]

            raw = json.dumps({"type": "axis", "source": "mouse", "dx": 1.5})
            axes.handlers["message"](raw)
            assert active.input_messages.get_nowait() == raw
        finally:
            await manager.close_active()

    asyncio.run(exercise())


class _ContinuousRuntime:
    """Fake resident runtime; ``gate`` holds each generation until the test releases it."""

    last_generation_metrics: dict[str, float] = {}

    def __init__(self, *, gated: bool = False) -> None:
        self.prompts: list[str | None] = []
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()

    def next_chunk_frames(self, default: int) -> int:
        return default

    async def generate(self, interactions: list[str], *, prompt: str | None = None, **kwargs: Any):
        del interactions, kwargs
        self.prompts.append(prompt)
        await self.gate.wait()
        return [np.zeros((180, 320, 3), dtype=np.uint8)], 1.0


def _continuous_session(
    transport: str,
    sink: _AxisSink,
    *,
    continuous: bool = True,
    max_chunks: int | None = 3,
) -> _ActivePeer | _ActiveSocket:
    """A prompt-scheduled session as ``create_answer`` / ``serve_socket`` build it for the spec."""
    fields: dict[str, Any] = {
        "resampler": RealtimeControlResampler(fps=16, start_time=0.0, realtime_keys=()),
        "prompt_scheduled": True,
        "initial_segment_pending": True,
        "base_prompt": "fox",
        "continuous": continuous,
        "max_chunks": max_chunks if continuous else None,
    }
    if transport == "socket":
        active: _ActivePeer | _ActiveSocket = _ActiveSocket(
            socket=sink, frame_packets=asyncio.Queue(maxsize=64), **fields
        )
    else:
        active = _ActivePeer(peer=None, channel=sink, frames=LatestFrameBuffer(maxsize=64), **fields)
    active.first_action.set()
    return active


def _start_worker(peers: RealtimePeerManager, active: _ActivePeer | _ActiveSocket) -> asyncio.Task[Any]:
    if isinstance(active, _ActiveSocket):
        return asyncio.create_task(peers._socket_generation_worker(active))
    return asyncio.create_task(peers._generation_worker(active))


def _chunks_done(sink: _AxisSink) -> list[dict[str, Any]]:
    return [message for message in sink.messages if message["type"] == "chunk_done"]


async def _settle() -> None:
    """Give an idle worker enough loop turns to start another chunk if it wrongly would."""
    await asyncio.sleep(0.05)


async def _stop_worker(active: _ActivePeer | _ActiveSocket, worker: asyncio.Task[Any]) -> None:
    active.closed = True
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_continuous_session_generates_max_chunks_without_input_then_stops(transport: str) -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime()
        active = _continuous_session(transport, sink, max_chunks=3)
        policy = active.frame_queue_policy if isinstance(active, _ActiveSocket) else active.frames.policy
        assert policy is FrameQueuePolicy.ORDERED_QUALITY
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=9)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 3), 5)
            await _settle()
            assert runtime.prompts == [None, None, None]
            chunks = _chunks_done(sink)
            assert [chunk["rollout_chunks"] for chunk in chunks] == [1, 2, 3]
            assert {chunk["max_chunks"] for chunk in chunks} == {3}
            assert {chunk["advance"] for chunk in chunks} == {"continuous"}
            assert not active.first_action.is_set()
            assert not any(message["type"] == "error" for message in sink.messages)
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_continuous_session_pause_stops_auto_advance_and_resume_continues(transport: str) -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime(gated=True)
        active = _continuous_session(transport, sink, max_chunks=3)
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=9)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(runtime.prompts) == 1), 5)
            await _handle_axis_message(peers, active, {"type": "pause", "request_id": "pause-1"})
            pause_ack = next(message for message in sink.messages if message["type"] == "pause_ack")
            assert pause_ack == {
                "type": "pause_ack",
                "ok": True,
                "request_id": "pause-1",
                "status": "paused",
                "advance": "continuous",
                "rollout_chunks": 0,
                "max_chunks": 3,
                "paused": True,
            }
            runtime.gate.set()
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) == 1), 5)
            await _settle()
            assert len(runtime.prompts) == 1
            assert _chunks_done(sink)[0]["paused"] is True
            assert not active.first_action.is_set()

            await _handle_axis_message(peers, active, {"type": "resume", "request_id": "resume-1"})
            resume_ack = next(message for message in sink.messages if message["type"] == "resume_ack")
            assert resume_ack["ok"] is True and resume_ack["status"] == "resumed"
            assert resume_ack["rollout_chunks"] == 1
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 3), 5)
            await _settle()
            assert len(runtime.prompts) == 3
            assert [chunk["rollout_chunks"] for chunk in _chunks_done(sink)] == [1, 2, 3]
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_continuous_session_prompt_update_restarts_rollout_count(transport: str) -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime()
        active = _continuous_session(transport, sink, max_chunks=2)
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=9)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 2), 5)
            await _settle()
            if isinstance(active, _ActivePeer):
                active.telemetry = DeliveryTelemetry(lambda _message: None)
            # A complete rollout has no blocks left: STEP is rejected and the unchanged prompt is dropped.
            await _handle_axis_message(peers, active, {"type": "step", "request_id": "step-1"})
            step_ack = next(message for message in sink.messages if message["type"] == "step_ack")
            assert step_ack == {
                "type": "step_ack",
                "ok": False,
                "request_id": "step-1",
                "message": "The rollout is complete; send a new prompt to start another.",
            }
            if isinstance(active, _ActivePeer):
                assert active.telemetry is not None
                assert list(active.telemetry.step_controls) == []
            await _handle_axis_message(peers, active, {"type": "prompt_update", "prompt": " fox "})
            await _settle()
            assert runtime.prompts == [None, None]
            assert active.pending_steps == 0

            await _handle_axis_message(peers, active, {"type": "prompt_update", "prompt": "owl"})
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 4), 5)
            await _settle()
            assert runtime.prompts == [None, None, "owl", None]
            assert [chunk["rollout_chunks"] for chunk in _chunks_done(sink)] == [1, 2, 1, 2]
            assert not any(message["type"] == "error" for message in sink.messages)
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_input_driven_session_rejects_pause_and_keeps_chunk_payload(transport: str) -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime()
        active = _continuous_session(transport, sink, continuous=False)
        policy = active.frame_queue_policy if isinstance(active, _ActiveSocket) else active.frames.policy
        assert policy is FrameQueuePolicy.ORDERED_QUALITY
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=9)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 1), 5)
            await _handle_axis_message(peers, active, {"type": "resume", "request_id": "resume-1"})
            await _settle()
            assert runtime.prompts == [None]
            chunk = _chunks_done(sink)[0]
            assert not {"advance", "rollout_chunks", "max_chunks", "paused"} & chunk.keys()
            resume_ack = next(message for message in sink.messages if message["type"] == "resume_ack")
            assert resume_ack == {
                "type": "resume_ack",
                "ok": False,
                "request_id": "resume-1",
                "message": "This session generates only on input.",
            }
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


def test_continuous_controls_session_advances_without_held_keys() -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime()
        active = _ActivePeer(
            peer=None,
            channel=sink,
            frames=LatestFrameBuffer(maxsize=8),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            continuous=True,
            max_chunks=2,
        )
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=4)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 2), 5)
            await _settle()
            assert len(runtime.prompts) == 2
            assert [chunk["interactions"] for chunk in _chunks_done(sink)] == [[], []]
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


class _ContinuousSpecRuntime:
    """Resident runtime reporting the native Self-Forcing spec; generation waits until the session closes."""

    queued_segment_generation = False
    entry = types.SimpleNamespace(default_prompt="fox", default_call_kwargs={})
    realtime_spec = RealtimeSpec(controls=("prompt_update",), keys=(), advance="continuous", max_chunks=7)
    transport_native_resolution = None
    supports_text_events = True
    last_generation_metrics: dict[str, float] = {}
    preserve_cuda_frames = False

    async def configure(self, **_kwargs: Any) -> None:
        pass

    async def reset(self) -> None:
        pass

    def steady_chunk_frames(self, _default: int) -> int:
        return self.realtime_spec.steady_chunk_frames

    async def generate(self, _interactions: list[str], **_kwargs: Any) -> tuple[list[np.ndarray], float]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


CONTINUOUS_READY_STATE = {"advance": "continuous", "rollout_chunks": 0, "max_chunks": 7, "paused": False}


def test_create_answer_maps_continuous_spec_onto_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    peers_created = _install_fake_aiortc(monkeypatch)

    async def exercise() -> None:
        manager = RealtimePeerManager(runtime=cast(Any, _ContinuousSpecRuntime()), fps=16, chunk_frames=9)
        await manager.create_answer(offer={"sdp": "offer-sdp", "type": "offer"}, session={})
        active = manager._active
        assert active is not None
        try:
            assert active.continuous is True
            assert active.max_chunks == 7
            assert active.prompt_scheduled is True
            assert active.frames.policy is FrameQueuePolicy.ORDERED_QUALITY
            controls = _FakeChannel("controls")
            peers_created[0].handlers["datachannel"](controls)
            ready = controls.sent[0]
            assert ready["type"] == "ready"
            assert ready["frame_queue_policy"] == FrameQueuePolicy.ORDERED_QUALITY.value
            assert {name: ready[name] for name in CONTINUOUS_READY_STATE} == CONTINUOUS_READY_STATE
            assert active.first_action.is_set()
        finally:
            await manager.close_active()

    asyncio.run(exercise())


def test_serve_socket_maps_continuous_spec_onto_the_session() -> None:
    from aiohttp import WSMsgType

    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []
            self.release = asyncio.Event()

        async def receive(self) -> object:
            return types.SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps({"type": "configure", "session": {}}))

        async def send_str(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

        async def send_bytes(self, _packet: bytes) -> None:
            pass

        async def close(self) -> None:
            self.closed = True

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> object:
            await self.release.wait()
            raise StopAsyncIteration

    async def exercise() -> None:
        socket = Socket()
        manager = RealtimePeerManager(runtime=cast(Any, _ContinuousSpecRuntime()), fps=16, chunk_frames=9)
        serving = asyncio.create_task(manager.serve_socket(socket))
        try:
            await asyncio.wait_for(
                _wait_for_condition(lambda: any(message["type"] == "ready" for message in socket.messages)), 5
            )
            active = manager._active_socket
            assert active is not None
            assert active.continuous is True
            assert active.max_chunks == 7
            assert active.prompt_scheduled is True
            assert active.frame_queue_policy is FrameQueuePolicy.ORDERED_QUALITY
            ready = next(message for message in socket.messages if message["type"] == "ready")
            assert ready["frame_queue_policy"] == FrameQueuePolicy.ORDERED_QUALITY.value
            assert {name: ready[name] for name in CONTINUOUS_READY_STATE} == CONTINUOUS_READY_STATE
        finally:
            socket.release.set()
            await asyncio.wait_for(serving, 10)
        assert socket.closed is True

    asyncio.run(exercise())


def test_realtime_http_healthz_status_load_and_duplicate_ws_upload_preservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiohttp import ClientSession, FormData, WSMsgType, web

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_WEBSOCKET_ONLY", "1")
    captured: dict[str, Any] = {}

    def fake_run_app(app: web.Application, **_kwargs: Any) -> None:
        captured["app"] = app

    monkeypatch.setattr(web, "run_app", fake_run_app)

    preload_state = {"fail": True, "calls": 0}
    entry = find_entry("matrix-game-2")

    class FakeManager:
        workspace_root = str(tmp_path / "studio")

        def prepare_inputs(self, **kwargs: Any) -> PreparedInputs:
            return PreparedInputs(
                prompt=str(kwargs.get("prompt") or ""),
                input_path=str(kwargs.get("input_path") or ""),
                image=kwargs.get("image"),
                image_path=str(kwargs.get("input_path") or "") or None,
                video_path=kwargs.get("video"),
                last_frame=None,
                last_frame_path=None,
                reference_images=[],
                reference_image_paths=[],
                interactions=[],
                camera_view=None,
                task_type="",
                intrinsics=None,
                meta_path="",
                panorama_path="",
                scene_name="",
                fps=16,
                num_frames=9,
                output_dir=str(tmp_path / "out"),
                output_path=str(tmp_path / "out" / "out.mp4"),
                call_kwargs={},
                load_kwargs={},
                model_ref="",
                backend="from_pretrained",
                endpoint="",
                api_key="",
                device="cpu",
            )

        def run_realtime(self, *, entry: Any, request: Any, action: str) -> dict[str, Any]:
            del entry, request
            if action == "configure":
                preload_state["calls"] += 1
                if preload_state["fail"]:
                    raise RuntimeError("missing weights on disk")
                return {
                    "status": "configured",
                    "realtime_spec": RealtimeSpec(controls=("prompt_update",), keys=()).to_payload(),
                }
            if action == "reset":
                return {"status": "reset"}
            return {"video": np.zeros((1, 4, 4, 3), dtype=np.uint8)}

    realtime_backend.serve_realtime_world_frontend(
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world", device="cpu"),
        manager=cast(Any, FakeManager()),
        host="127.0.0.1",
        port=0,
        demo_images=(),
        allowed_roots=(tmp_path,),
    )
    app: web.Application = captured["app"]

    async def exercise() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        server: Any = site._server
        port = server.sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        try:
            async with ClientSession() as client:
                # Wait for initial startup preload failure to settle.
                async def wait_error() -> None:
                    while True:
                        async with client.get(f"{base}/healthz") as resp:
                            body = await resp.json()
                            if body["error"] is not None:
                                return
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_error(), timeout=5)

                async with client.get(f"{base}/healthz") as resp:
                    assert resp.status == 200
                    health = await resp.json()
                    assert health == {
                        "ok": True,
                        "frontend": "world",
                        "model_id": entry.model_id,
                        "ready": False,
                        "loading": False,
                        "error": "missing weights on disk",
                    }

                async with client.get(f"{base}/api/runtime/status") as resp:
                    status = await resp.json()
                    assert status["capabilities"] == {"text_events": False}

                # POST /api/runtime/load returns structured JSON error instead of bare HTTP 500.
                async with client.post(f"{base}/api/runtime/load", json={}) as resp:
                    assert resp.status == 500
                    body = await resp.json()
                    assert body == {"loaded": False, "ready": False, "error": "missing weights on disk"}

                # Retry after fixing the failure updates both /healthz and capabilities.text_events.
                preload_state["fail"] = False
                async with client.post(f"{base}/api/runtime/load", json={}) as resp:
                    assert resp.status == 200
                    assert await resp.json() == {"loaded": True, "ready": True}

                async with client.get(f"{base}/healthz") as resp:
                    health = await resp.json()
                    assert health["ok"] is True
                    assert health["ready"] is True
                    assert health["loading"] is False
                    assert health["error"] is None

                async with client.get(f"{base}/api/runtime/status") as resp:
                    status = await resp.json()
                    assert status["ready"] is True
                    assert status["capabilities"] == {"text_events": True}

                # Upload an image and hold an active WebSocket session.
                image_bytes = io.BytesIO()
                Image.new("RGB", (8, 8), "blue").save(image_bytes, format="PNG")
                form = FormData()
                form.add_field("file", image_bytes.getvalue(), filename="seed.png", content_type="image/png")
                async with client.post(f"{base}/api/session/input", data=form) as resp:
                    assert resp.status == 200
                    uploaded = Path((await resp.json())["path"])
                assert uploaded.is_file()

                ws1 = await client.ws_connect(f"{base}/api/realtime/ws")
                await ws1.send_json({"type": "configure", "session": {"init_image_path": str(uploaded)}})
                while True:
                    msg = await asyncio.wait_for(ws1.receive(), timeout=5)
                    assert msg.type is WSMsgType.TEXT
                    if json.loads(msg.data).get("type") == "ready":
                        break

                # A rejected duplicate tab must not delete the active tab's uploads.
                ws2 = await client.ws_connect(f"{base}/api/realtime/ws")
                await ws2.send_json({"type": "configure", "session": {}})
                messages_ws2: list[dict[str, Any]] = []
                async for msg in ws2:
                    if msg.type is WSMsgType.TEXT:
                        messages_ws2.append(json.loads(msg.data))
                assert any(
                    item.get("type") == "error" and "already active" in str(item.get("message", ""))
                    for item in messages_ws2
                )
                assert uploaded.is_file(), "rejected duplicate websocket must not delete active session uploads"

                # Closing the accepted session cleans up the owned upload.
                await ws1.close()
                await asyncio.wait_for(_wait_for_condition(lambda: not uploaded.exists()), timeout=5)
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
def test_continuous_non_prompt_scheduled_session_accepts_prompt_update(transport: str) -> None:
    async def exercise() -> None:
        sink = _AxisSink()
        runtime = _ContinuousRuntime()
        fields: dict[str, Any] = {
            "resampler": RealtimeControlResampler(fps=16, start_time=0.0),
            "prompt_scheduled": False,
            "base_prompt": "fox",
            "continuous": True,
            "max_chunks": 2,
        }
        active: _ActivePeer | _ActiveSocket
        if transport == "socket":
            active = _ActiveSocket(socket=sink, frame_packets=asyncio.Queue(maxsize=64), **fields)
        else:
            active = _ActivePeer(peer=None, channel=sink, frames=LatestFrameBuffer(maxsize=64), **fields)
        active.first_action.set()
        peers = RealtimePeerManager(runtime=cast(Any, runtime), fps=16, chunk_frames=4)
        worker = _start_worker(peers, active)
        try:
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 2), 5)
            await _settle()
            assert runtime.prompts == [None, None]
            await _handle_axis_message(peers, active, {"type": "prompt_update", "prompt": "owl"})
            await asyncio.wait_for(_wait_for_condition(lambda: len(_chunks_done(sink)) >= 4), 5)
            await _settle()
            assert runtime.prompts == [None, None, "owl", None]
            assert [chunk["rollout_chunks"] for chunk in _chunks_done(sink)] == [1, 2, 1, 2]
        finally:
            await _stop_worker(active, worker)

    asyncio.run(exercise())


def test_queued_segment_value_error_preserves_session_and_applies_segment1_retry_paths(tmp_path: Path) -> None:
    entry = find_entry("longvie-2")
    calls: list[tuple[str, PreparedInputs]] = []
    mode = {"behavior": "value_error"}

    class Manager:
        workspace_root = str(tmp_path)

        def run_realtime(self, *, entry: Any, request: PreparedInputs, action: str) -> dict[str, Any]:
            del entry
            calls.append((action, request))
            if action == "reset":
                return {"status": "reset"}
            if mode["behavior"] == "value_error":
                raise ValueError("bad depth video length")
            if mode["behavior"] == "fatal":
                raise RuntimeError("cuda device fault")
            return {
                "video": np.zeros((4, 2, 3, 3), dtype=np.uint8),
                "realtime_spec": {
                    "fps": 16,
                    "first_chunk_frames": 4,
                    "steady_chunk_frames": 4,
                    "controls": ["dense_depth_video", "sparse_pointmap_or_track_video"],
                    "transport": "queued-segment-rgb",
                },
            }

    request = PreparedInputs(
        prompt="first segment",
        input_path="/uploads/seed.png",
        image=Image.new("RGB", (3, 2), "white"),
        image_path="/uploads/seed.png",
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=[],
        camera_view=None,
        task_type="image-to-video",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=16,
        num_frames=4,
        output_dir=str(tmp_path / "unused"),
        output_path=str(tmp_path / "unused" / "out.mp4"),
        call_kwargs={
            "num_frames": 4,
            "dense_video": "/uploads/bad-depth.mp4",
            "sparse_video": "/uploads/bad-track.mp4",
            "return_dict": True,
        },
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )
    runtime = ResidentWorldRuntime(
        manager=cast(Any, Manager()),
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
        fps=16,
    )
    runtime._base_request = request
    runtime._configured = True

    async def exercise() -> None:
        with pytest.raises(ValueError, match="bad depth video length"):
            await runtime.generate([], seed=1)
        assert runtime._configured is True
        assert runtime._base_request is request
        assert runtime._first_stream_step is True

        mode["behavior"] = "ok"
        frames1, _ = await runtime.generate(
            [],
            seed=2,
            prompt="retried segment 1",
            dense_video_path="/uploads/fixed-depth.mp4",
            sparse_video_path="/uploads/fixed-track.mp4",
        )
        assert len(frames1) == 4
        assert runtime._first_stream_step is False

        frames2, _ = await runtime.generate(
            [],
            seed=3,
            prompt="segment 2",
            dense_video_path="/uploads/depth-2.mp4",
            sparse_video_path="/uploads/track-2.mp4",
        )
        assert len(frames2) == 4

        mode["behavior"] = "fatal"
        with pytest.raises(RuntimeError, match="cuda device fault"):
            await runtime.generate(
                [],
                seed=4,
                prompt="segment 3",
                dense_video_path="/uploads/depth-3.mp4",
                sparse_video_path="/uploads/track-3.mp4",
            )
        assert runtime._configured is False
        assert runtime._base_request is None
        await runtime.close()

    asyncio.run(exercise())

    assert [action for action, _ in calls[:4]] == ["run", "run", "stream", "stream"]
    assert calls[1][1].prompt == "retried segment 1"
    assert calls[1][1].call_kwargs["dense_video"] == "/uploads/fixed-depth.mp4"
    assert calls[1][1].call_kwargs["sparse_video"] == "/uploads/fixed-track.mp4"
    assert calls[1][1].image is not None
    assert calls[2][1].image is None
    assert calls[2][1].call_kwargs["dense_video"] == "/uploads/depth-2.mp4"


def test_stale_session_teardown_callbacks_cannot_close_newer_session(monkeypatch: pytest.MonkeyPatch) -> None:
    peers_created = _install_fake_aiortc(monkeypatch)

    class Runtime:
        queued_segment_generation = False
        entry = types.SimpleNamespace(default_prompt="", default_call_kwargs={})
        realtime_spec = RealtimeSpec(fps=16, first_chunk_frames=2, steady_chunk_frames=2)
        transport_native_resolution = None
        supports_text_events = False
        last_generation_metrics: dict[str, float] = {}
        preserve_cuda_frames = False

        async def configure(self, **_kwargs: Any) -> None:
            pass

        async def reset(self) -> None:
            pass

        def steady_chunk_frames(self, _default: int) -> int:
            return 2

    async def exercise() -> None:
        manager = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=16, chunk_frames=2)
        await manager.create_answer(offer={"sdp": "offer-1", "type": "offer"}, session={})
        old_peer_conn = peers_created[0]
        old_active = manager._active
        assert old_active is not None
        old_channel = _FakeChannel("controls")
        old_peer_conn.handlers["datachannel"](old_channel)

        await manager.close_active()
        assert manager._active is None

        await manager.create_answer(offer={"sdp": "offer-2", "type": "offer"}, session={})
        new_active = manager._active
        assert new_active is not None and new_active is not old_active

        # Late callbacks from the previous peer must not touch the replacement session.
        old_peer_conn.connectionState = "failed"
        await old_peer_conn.handlers["connectionstatechange"]()
        old_channel.handlers["close"]()
        await manager.close_active(expected=old_active)
        await asyncio.sleep(0)

        assert manager._active is new_active
        assert new_active.closed is False

        # Same identity guard applies to WebSocket teardown.
        await manager.close_active()
        old_socket = _ActiveSocket(
            socket=types.SimpleNamespace(closed=False, close=lambda: asyncio.sleep(0)),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=4),
        )
        new_socket = _ActiveSocket(
            socket=types.SimpleNamespace(closed=False, close=lambda: asyncio.sleep(0)),
            resampler=RealtimeControlResampler(fps=16, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=4),
        )
        manager._active_socket = new_socket
        await manager.close_active(expected=old_socket)
        assert manager._active_socket is new_socket
        assert new_socket.closed is False
        await manager.close_active()

    asyncio.run(exercise())


def test_world_css_sensor_grid_rows_and_immersive_top_controls_do_not_overlap() -> None:
    css = world_frontend_css()
    assert "grid-auto-rows: minmax(0, 1fr);" in css
    assert ".sensor-tile { position: relative; min-width: 0; min-height: 0;" in css
    control_bar_block = css.split(".is-immersive .realtime-control-bar {", 1)[1].split("}", 1)[0]
    session_actions_block = css.split(".is-immersive .session-actions {", 1)[1].split("}", 1)[0]
    control_top = next(line.strip() for line in control_bar_block.splitlines() if line.strip().startswith("top:"))
    actions_top = next(line.strip() for line in session_actions_block.splitlines() if line.strip().startswith("top:"))
    assert control_top != actions_top
