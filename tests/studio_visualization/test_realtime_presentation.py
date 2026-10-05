from __future__ import annotations

import asyncio
import json
import sys
import types

import numpy as np
import pytest
from aiohttp import ClientSession, WSMsgType, web

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.studio.serving.realtime.input import RealtimeControlResampler
from worldfoundry.studio.serving.realtime.media import (
    ChunkPresentationBuffer,
    FrameQueuePolicy,
    RealtimePresentationMode,
    WebSocketPacingMode,
)
from worldfoundry.studio.visualization.backends.world_realtime import (
    RealtimePeerManager,
    _ActiveSocket,
    _build_video_track,
)


def _frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def test_presentation_mode_aliases_and_validation() -> None:
    assert RealtimePresentationMode.from_value("legacy") is RealtimePresentationMode.HOLD_LAST
    assert RealtimePresentationMode.from_value("no_repeat") is RealtimePresentationMode.REAL_FRAMES
    with pytest.raises(ValueError, match="Unknown realtime presentation mode"):
        RealtimePresentationMode.from_value("duplicate-everything")
    assert WebSocketPacingMode.from_value("legacy") is WebSocketPacingMode.LEGACY_DUAL
    assert WebSocketPacingMode.from_value("browser") is WebSocketPacingMode.CLIENT_ONLY
    with pytest.raises(ValueError, match="Unknown WebSocket pacing mode"):
        WebSocketPacingMode.from_value("nobody")


def test_peer_manager_defaults_to_compatible_hold_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_PRESENTATION_MODE", raising=False)
    compatible = RealtimePeerManager(runtime=None, fps=16, chunk_frames=2)  # type: ignore[arg-type]
    experimental = RealtimePeerManager(
        runtime=None,  # type: ignore[arg-type]
        fps=16,
        chunk_frames=2,
        presentation_mode="real-frames",
        socket_pacing_mode="client-only",
    )

    assert compatible.presentation_mode is RealtimePresentationMode.HOLD_LAST
    assert compatible.socket_pacing_mode is WebSocketPacingMode.LEGACY_DUAL
    assert experimental.presentation_mode is RealtimePresentationMode.REAL_FRAMES
    assert experimental.socket_pacing_mode is WebSocketPacingMode.CLIENT_ONLY


def test_client_only_websocket_mode_does_not_pace_on_server() -> None:
    async def exercise() -> None:
        sent: list[bytes] = []
        active: _ActiveSocket

        class Socket:
            closed = False

            async def send_bytes(self, packet: bytes) -> None:
                sent.append(packet)
                if len(sent) == 2:
                    active.closed = True

        active = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=1, start_time=0.0),
            frame_packets=asyncio.Queue(maxsize=2),
            pacing_mode=WebSocketPacingMode.CLIENT_ONLY,
        )
        active.frame_packets.put_nowait(b"first")
        active.frame_packets.put_nowait(b"second")
        manager = RealtimePeerManager(
            runtime=None,  # type: ignore[arg-type]
            fps=1,
            chunk_frames=2,
        )

        await asyncio.wait_for(manager._socket_sender(active), timeout=0.25)
        assert sent == [b"first", b"second"]

    asyncio.run(exercise())


def test_socket_config_selects_client_only_pacing() -> None:
    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def receive(self) -> object:
            return types.SimpleNamespace(
                type=WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "configure",
                        "session": {"socket_pacing_mode": "client-only"},
                    }
                ),
            )

        async def send_str(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

        async def send_bytes(self, _packet: bytes) -> None:
            pass

        async def close(self) -> None:
            self.closed = True

        def __aiter__(self) -> "Socket":
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    class Runtime:
        queued_segment_generation = False
        entry = types.SimpleNamespace(default_prompt="", default_call_kwargs={})
        realtime_spec = RealtimeSpec(
            fps=20,
            first_chunk_frames=3,
            steady_chunk_frames=3,
            controls=("forward",),
        )
        transport_native_resolution = None
        supports_text_events = False

        async def configure(self, **_kwargs) -> None:
            pass

        async def reset(self) -> None:
            pass

        def steady_chunk_frames(self, _default: int) -> int:
            return self.realtime_spec.steady_chunk_frames

    async def exercise() -> None:
        socket = Socket()
        manager = RealtimePeerManager(
            runtime=Runtime(),  # type: ignore[arg-type]
            fps=16,
            chunk_frames=3,
        )
        await manager.serve_socket(socket)

        ready = next(message for message in socket.messages if message["type"] == "ready")
        assert ready["socket_pacing_mode"] == "client-only"
        assert socket.closed is True

    asyncio.run(exercise())


def test_explicit_socket_disconnect_does_not_wait_for_a_nonreading_client(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "2")

    class Runtime:
        queued_segment_generation = False
        entry = types.SimpleNamespace(default_prompt="", default_call_kwargs={})
        realtime_spec = RealtimeSpec(
            fps=20,
            first_chunk_frames=3,
            steady_chunk_frames=3,
            controls=("forward",),
        )
        transport_native_resolution = None
        supports_text_events = False

        def __init__(self) -> None:
            self.reset_calls = 0

        async def configure(self, **_kwargs) -> None:
            pass

        async def reset(self) -> None:
            self.reset_calls += 1

        def steady_chunk_frames(self, _default: int) -> int:
            return self.realtime_spec.steady_chunk_frames

    async def exercise() -> None:
        runtime = Runtime()
        manager = RealtimePeerManager(
            runtime=runtime,  # type: ignore[arg-type]
            fps=20,
            chunk_frames=3,
        )
        handler_done = asyncio.Event()
        handler_errors: list[BaseException] = []

        async def handler(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            try:
                await manager.serve_socket(socket)
            except BaseException as exc:
                handler_errors.append(exc)
            finally:
                handler_done.set()
            return socket

        app = web.Application()
        app.router.add_get("/realtime", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]

        try:
            async with ClientSession() as session:
                socket = await session.ws_connect(f"http://127.0.0.1:{port}/realtime")
                await socket.send_json({"type": "configure", "session": {}})
                while True:
                    message = await asyncio.wait_for(socket.receive(), timeout=1)
                    assert message.type is WSMsgType.TEXT
                    if json.loads(message.data).get("type") == "ready":
                        break

                started = asyncio.get_running_loop().time()
                await socket.send_json({"type": "disconnect"})
                # Intentionally perform no further receive. A client that has
                # declared disconnect must not own the runtime-reset budget.
                await asyncio.wait_for(handler_done.wait(), timeout=1)
                elapsed = asyncio.get_running_loop().time() - started
                await socket.close()

            assert elapsed < 1
            assert runtime.reset_calls == 1
            assert manager.active is False
            assert handler_errors == []
        finally:
            if manager.active:
                await manager.close_active()
            await runner.cleanup()

    asyncio.run(exercise())
    assert "session runtime reset" not in caplog.text
    assert "shutdown deadline expired" not in caplog.text


def test_create_answer_selects_real_frame_presentation(monkeypatch: pytest.MonkeyPatch) -> None:
    peers_created: list[object] = []

    class FakeMediaStreamTrack:
        pass

    class FakeMediaStreamError(Exception):
        pass

    class FakeVideoFrame:
        @classmethod
        def from_ndarray(cls, _array: np.ndarray, *, format: str) -> "FakeVideoFrame":
            assert format == "rgb24"
            return cls()

    class FakePeerConnection:
        iceGatheringState = "complete"
        connectionState = "new"

        def __init__(self, _configuration: object) -> None:
            self.handlers: dict[str, object] = {}
            self.tracks: list[object] = []
            self.localDescription = types.SimpleNamespace(sdp="", type="answer")
            peers_created.append(self)

        def addTrack(self, track: object) -> None:  # noqa: N802 - aiortc compatibility
            self.tracks.append(track)

        def on(self, name: str):
            def register(handler):
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

    aiortc = types.ModuleType("aiortc")
    aiortc.MediaStreamTrack = FakeMediaStreamTrack
    aiortc.MediaStreamError = FakeMediaStreamError
    aiortc.RTCConfiguration = lambda **kwargs: types.SimpleNamespace(**kwargs)
    aiortc.RTCIceServer = lambda **kwargs: types.SimpleNamespace(**kwargs)
    aiortc.RTCPeerConnection = FakePeerConnection
    aiortc.RTCSessionDescription = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mediastreams = types.ModuleType("aiortc.mediastreams")
    mediastreams.MediaStreamError = FakeMediaStreamError
    av = types.ModuleType("av")
    av.VideoFrame = FakeVideoFrame
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)
    monkeypatch.setitem(sys.modules, "aiortc.mediastreams", mediastreams)
    monkeypatch.setitem(sys.modules, "av", av)

    class Runtime:
        queued_segment_generation = False
        entry = types.SimpleNamespace(default_prompt="", default_call_kwargs={})
        realtime_spec = RealtimeSpec(
            fps=20,
            first_chunk_frames=3,
            steady_chunk_frames=3,
            controls=("forward",),
        )
        transport_native_resolution = None
        supports_text_events = False

        async def configure(self, **_kwargs) -> None:
            pass

        async def reset(self) -> None:
            pass

        def steady_chunk_frames(self, _default: int) -> int:
            return self.realtime_spec.steady_chunk_frames

    async def exercise() -> None:
        manager = RealtimePeerManager(
            runtime=Runtime(),  # type: ignore[arg-type]
            fps=16,
            chunk_frames=3,
        )
        answer = await manager.create_answer(
            offer={"sdp": "offer-sdp", "type": "offer"},
            session={"presentation_mode": "real-frames"},
        )
        try:
            assert answer == {"sdp": "answer-sdp", "type": "answer"}
            assert len(peers_created) == 1
            assert len(peers_created[0].tracks) == 1  # type: ignore[attr-defined]
            assert manager._active is not None
            assert isinstance(manager._active.frames, ChunkPresentationBuffer)
            assert manager._active.presentation_mode is RealtimePresentationMode.REAL_FRAMES
        finally:
            await manager.close_active()

    asyncio.run(exercise())


def test_presentation_buffer_replaces_only_the_complete_pending_chunk() -> None:
    async def exercise() -> None:
        buffer = ChunkPresentationBuffer(fps=1000, maximum_fps=1000)
        await buffer.put_chunk([_frame(1), _frame(2), _frame(3)])
        await buffer.put_chunk([_frame(4), _frame(5)])
        await buffer.put_chunk([_frame(6), _frame(7)])

        assert buffer.dropped_chunks == 1
        assert buffer.dropped_frames == 2
        values = [
            int((await asyncio.wait_for(buffer.get(), timeout=0.5))[0, 0, 0])
            for _ in range(5)
        ]
        assert values == [1, 2, 3, 6, 7]
        assert buffer.presented_frames == 5
        assert buffer.qsize() == 0
        buffer.close()

    asyncio.run(exercise())


def test_presentation_buffer_bounds_sender_lag_to_two_real_frames() -> None:
    async def exercise() -> None:
        buffer = ChunkPresentationBuffer(fps=1000, maximum_fps=1000)
        await buffer.put_chunk([_frame(value) for value in range(5)])

        async def wait_until_presented() -> None:
            while buffer.presented_frames < 5:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_presented(), timeout=0.5)

        assert buffer.qsize() == 2
        assert buffer.dropped_frames == 3
        assert [int((await buffer.get())[0, 0, 0]) for _ in range(2)] == [3, 4]
        buffer.close()

    asyncio.run(exercise())


def test_ordered_presentation_backpressures_at_chunk_boundary() -> None:
    async def exercise() -> None:
        buffer = ChunkPresentationBuffer(
            fps=1000,
            maximum_fps=1000,
            policy=FrameQueuePolicy.ORDERED_QUALITY,
        )
        await buffer.put_chunk([_frame(1), _frame(2), _frame(3)])
        await buffer.put_chunk([_frame(4), _frame(5)])
        blocked = asyncio.create_task(buffer.put_chunk([_frame(6)]))
        await asyncio.sleep(0)
        assert blocked.done() is False

        first = [int((await buffer.get())[0, 0, 0]) for _ in range(3)]
        assert await asyncio.wait_for(blocked, timeout=0.5) == 1
        rest = [int((await buffer.get())[0, 0, 0]) for _ in range(3)]
        assert first + rest == [1, 2, 3, 4, 5, 6]
        assert buffer.dropped_frames == 0
        assert buffer.dropped_chunks == 0
        buffer.close()

    asyncio.run(exercise())


def test_real_frame_track_blocks_instead_of_repeating(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMediaStreamTrack:
        pass

    class FakeMediaStreamError(Exception):
        pass

    class FakeVideoFrame:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.pts: int | None = None
            self.time_base = None

        @classmethod
        def from_ndarray(cls, array: np.ndarray, *, format: str) -> "FakeVideoFrame":
            assert format == "rgb24"
            return cls(array)

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
        buffer = ChunkPresentationBuffer(fps=1000, maximum_fps=1000)
        await buffer.put_chunk([_frame(1)])
        track = _build_video_track(
            frames=buffer,
            fps=1000,
            presentation_mode=RealtimePresentationMode.REAL_FRAMES,
        )

        first = await asyncio.wait_for(track.recv(), timeout=0.5)
        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0.01)
        assert waiting.done() is False

        await buffer.put_chunk([_frame(2)])
        second = await asyncio.wait_for(waiting, timeout=0.5)
        assert int(first.array[0, 0, 0]) == 1
        assert int(second.array[0, 0, 0]) == 2
        assert first.pts == 0
        assert second.pts is not None and second.pts >= 1

        buffer.close()
        with pytest.raises(FakeMediaStreamError):
            await track.recv()

    asyncio.run(exercise())
