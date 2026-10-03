from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.execution.realtime.prewarm import PrewarmTimeoutError
from worldfoundry.core.media.processing.postprocess import VideoSpec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PipelineContext,
    PreparedInputs,
    StudioManager,
)
from worldfoundry.studio.serving.realtime.media import FrameQueuePolicy
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends import world_realtime as realtime_backend
from worldfoundry.studio.visualization.backends.world import world_frontend_html
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
from worldfoundry.studio.visualization.backends.world_realtime_client import (
    WORLD_REALTIME_CLIENT_JS,
)


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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        manager=Manager(),  # type: ignore[arg-type]
        entry=find_entry("sana-wm"),
        launch_config=StudioLaunchConfig(model_id="sana-wm", frontend="world"),
        fps=16,
        postprocess=BlockingPostprocess(),  # type: ignore[arg-type]
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
        manager=Manager(),  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=None, fps=16, chunk_frames=9)  # type: ignore[arg-type]

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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        effective_keys = frozenset()

        def __init__(self) -> None:
            self.sampled: list[int] = []

        def sample_chunk(self, num_frames: int, *, wall_time: float):
            self.sampled.append(num_frames)
            return [(wall_time, wall_time + 1.0, frozenset({"w"}))]

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
            resampler=resampler,  # type: ignore[arg-type]
        )
        runtime.active = active
        peers = RealtimePeerManager(runtime=runtime, fps=24, chunk_frames=5)  # type: ignore[arg-type]
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
        effective_keys = frozenset({"w"})

        def sample_chunk(self, num_frames: int, *, wall_time: float):
            del num_frames
            return [(wall_time, wall_time + 1.0, frozenset({"w"}))]

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
            resampler=Resampler(),  # type: ignore[arg-type]
        )
        runtime.active = active
        peers = RealtimePeerManager(runtime=runtime, fps=24, chunk_frames=2)  # type: ignore[arg-type]
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
    peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
    peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=None, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
            frame = await active.frames.get()
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

    aiortc = types.ModuleType("aiortc")
    aiortc.MediaStreamTrack = FakeMediaStreamTrack
    mediastreams = types.ModuleType("aiortc.mediastreams")
    mediastreams.MediaStreamError = FakeMediaStreamError
    av = types.ModuleType("av")
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=3)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=None, fps=16, chunk_frames=2)  # type: ignore[arg-type]

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
        assert marker in WORLD_REALTIME_CLIENT_JS


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
            runtime=None,  # type: ignore[arg-type]
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
            assert int((await buffer.get())[0, 0, 0]) == 1

        consumer = asyncio.create_task(consume())
        assert await buffer.put_chunk([second]) == 1
        await consumer
        assert buffer.dropped_frames == 0
        assert int((await buffer.get())[0, 0, 0]) == 2

        stale = LatestFrameBuffer(maxsize=1)
        await stale.put_chunk([first, second])
        assert stale.dropped_frames == 1
        assert int((await stale.get())[0, 0, 0]) == 2

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
        np.testing.assert_array_equal(await buffer.get(), first)

        assert await asyncio.wait_for(blocked, timeout=0.5) == 1
        np.testing.assert_array_equal(await buffer.get(), second)
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
        manager=Manager(),  # type: ignore[arg-type]
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
        fps=16,
        warmup_image_path=str(image_path),
        warmup_chunks=2,
    )
    runtime._build_request = lambda **_kwargs: request  # type: ignore[method-assign]
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
            manager=object(),  # type: ignore[arg-type]
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

        runtime.reset = resistant_reset  # type: ignore[method-assign]
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
            manager=object(),  # type: ignore[arg-type]
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
            manager=object(),  # type: ignore[arg-type]
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

        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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

        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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

        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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

        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        peers = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=2)  # type: ignore[arg-type]
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

        runtime = Runtime()
        runtime.generate = resistant_generate  # type: ignore[attr-defined]
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)  # type: ignore[arg-type]
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
        manager=manager,  # type: ignore[arg-type]
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
        manager=Manager(),  # type: ignore[arg-type]
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
    manager.runtime_driver_for = lambda _entry: driver  # type: ignore[method-assign]
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
