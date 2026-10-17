# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixture sensor producers traverse the real resident worker and WebSocket transport."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSensor, RealtimeSpec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import StudioManager
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends.world_realtime import RealtimePeerManager, ResidentWorldRuntime

pytestmark = pytest.mark.unit


class _SensorProducer:
    """Model-boundary fixture; all worker, atlas, codec, and socket code stays native."""

    def __init__(self) -> None:
        self.chunk = 0
        self.failure: str | None = None
        self.single_rgb = False
        self.resets = 0
        self.threads: set[int] = set()
        self.spec = RealtimeSpec(
            fps=16,
            first_chunk_frames=2,
            steady_chunk_frames=2,
            controls=("forward",),
            sensors=(
                RealtimeSensor("front", kind="rgb", channel_index=2),
                RealtimeSensor("depth", kind="depth", channel_index=0, range_m=10),
                RealtimeSensor("lidar", kind="lidar", channel_index=7, range_m=10),
            ),
        )

    def __call__(self, *, action: str, **_kwargs: Any) -> dict[str, Any]:
        self.threads.add(threading.get_ident())
        if action == "reset":
            self.resets += 1
            self.chunk = 0
            return {}
        if action == "configure":
            spec = RealtimeSpec(fps=16, first_chunk_frames=2, steady_chunk_frames=2) if self.single_rgb else self.spec
            return {"realtime_spec": spec.to_payload()}
        assert action == "stream"
        self.chunk += 1
        if self.single_rgb:
            return {
                "video": np.full((2, 32, 48, 3), (16, 32, 224), dtype=np.uint8),
                "realtime_spec": RealtimeSpec(fps=16, first_chunk_frames=2, steady_chunk_frames=2).to_payload(),
            }
        red = np.stack(
            [np.full((32, 48, 3), (96 + 32 * self.chunk + 16 * time, 0, 0), dtype=np.uint8) for time in range(2)]
        )
        depth = [np.full((32, 48), 2.0 + self.chunk + time, dtype=np.float32) for time in range(2)]
        x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
        sensors = {"lidar": [points, points], "depth": depth, "front": red}
        if self.failure == "missing":
            sensors.pop("depth")
        elif self.failure == "cadence":
            sensors["depth"] = depth[:1]
        return {"realtime_spec": self.spec.to_payload(), "sensor_frames": sensors}


async def _receive_json(socket: ClientWebSocketResponse[bool], expected: str) -> dict[str, Any]:
    while True:
        message = await asyncio.wait_for(socket.receive(), timeout=30)
        assert message.type is WSMsgType.TEXT, message
        payload = json.loads(message.data)
        assert payload["type"] != "error", payload
        if payload["type"] == expected:
            return payload


async def _receive_chunk(socket: ClientWebSocketResponse[bool]) -> tuple[dict[str, Any], list[np.ndarray]]:
    metadata = None
    frames = []
    while metadata is None or len(frames) < metadata["frames"]:
        message = await asyncio.wait_for(socket.receive(), timeout=30)
        if message.type is WSMsgType.BINARY:
            with Image.open(io.BytesIO(message.data)) as image:
                frames.append(np.asarray(image.convert("RGB")))
        else:
            assert message.type is WSMsgType.TEXT, message
            payload = json.loads(message.data)
            assert payload["type"] != "error", payload
            if payload["type"] == "chunk_done":
                metadata = payload
    assert metadata is not None and len(frames) == metadata["frames"] == 2
    return metadata, frames


def _assert_sensor_chunk(metadata: dict[str, Any], frames: list[np.ndarray], chunk: int) -> None:
    layout = metadata["sensor_layout"]
    assert [item["key"] for item in layout] == ["front", "depth", "lidar"]
    assert [item["kind"] for item in layout] == ["rgb", "depth", "lidar"]
    assert [item["channel_index"] for item in layout] == [2, 0, 7]
    assert [(item["x"], item["y"]) for item in layout] == [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]
    assert metadata["dropped_frames"] == 0
    for time, frame in enumerate(frames):
        assert frame.shape == (1024, 1024, 3)
        np.testing.assert_allclose(frame[256, 256], (96 + 32 * chunk + 16 * time, 0, 0), atol=3)
        shade = (2 + chunk + time) / 10
        np.testing.assert_allclose(frame[256, 768], (int(shade * 255), int((1 - shade) * 255), 0), atol=3)
        assert frame[744:792, 232:280].mean() > 20
        assert not frame[768, 768].any()


@pytest.mark.parametrize("failure", ["missing", "cadence"])
def test_resident_sensor_socket_order_reset_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SOCKET_PACING_MODE", "client-only")
    producer = _SensorProducer()
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    monkeypatch.setattr(manager, "run_realtime", producer)
    entry = find_entry("infinite-world")
    runtime = ResidentWorldRuntime(
        manager=manager,
        entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, device="cpu"),
        fps=16,
        warmup_chunks=0,
    )

    async def exercise() -> None:
        peers = RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)
        handler_errors: list[BaseException] = []
        completed: asyncio.Queue[None] = asyncio.Queue()

        async def handler(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            try:
                await peers.serve_socket(socket)
            except BaseException as error:
                handler_errors.append(error)
            finally:
                completed.put_nowait(None)
            return socket

        app = web.Application()
        app.router.add_get("/realtime", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with ClientSession() as client:
                async with client.ws_connect(f"http://127.0.0.1:{port}/realtime") as socket:
                    await socket.send_json({"type": "configure", "session": {"output_resolution": "native"}})
                    ready = await _receive_json(socket, "ready")
                    assert ready["fps"] == 16
                    initial = None
                    for chunk in (1, 2):
                        await socket.send_json({"type": "step"})
                        metadata, frames = await _receive_chunk(socket)
                        assert metadata["chunk_index"] == chunk
                        _assert_sensor_chunk(metadata, frames, chunk)
                        if chunk == 1:
                            initial = frames
                    producer.failure = failure
                    await socket.send_json({"type": "step"})
                    while True:
                        message = await asyncio.wait_for(socket.receive(), timeout=30)
                        assert message.type is WSMsgType.TEXT, message
                        payload = json.loads(message.data)
                        if payload["type"] == "error":
                            assert "ValueError" in payload["message"]
                            expected = "sensor keys" if failure == "missing" else "identical frame counts"
                            assert expected in payload["message"]
                            break
                    await asyncio.wait_for(socket.receive(), timeout=10)
                await asyncio.wait_for(completed.get(), timeout=10)
                producer.failure = None
                async with client.ws_connect(f"http://127.0.0.1:{port}/realtime") as socket:
                    await socket.send_json({"type": "configure", "session": {"output_resolution": "native"}})
                    await _receive_json(socket, "ready")
                    await socket.send_json({"type": "step"})
                    metadata, recovered = await _receive_chunk(socket)
                    _assert_sensor_chunk(metadata, recovered, 1)
                    assert initial is not None
                    for reference, candidate in zip(initial, recovered):
                        np.testing.assert_array_equal(reference, candidate)
                    await socket.send_json({"type": "disconnect"})
                    await asyncio.wait_for(socket.receive(), timeout=10)
                await asyncio.wait_for(completed.get(), timeout=10)
                producer.single_rgb = True
                async with client.ws_connect(f"http://127.0.0.1:{port}/realtime") as socket:
                    await socket.send_json({"type": "configure", "session": {"output_resolution": "native"}})
                    await _receive_json(socket, "ready")
                    await socket.send_json({"type": "step"})
                    metadata, frames = await _receive_chunk(socket)
                    assert metadata["sensor_layout"] == []
                    assert frames[0].shape == (32, 48, 3)
                    np.testing.assert_allclose(frames[0][16, 24], (16, 32, 224), atol=3)
                    await socket.send_json({"type": "disconnect"})
                    await asyncio.wait_for(socket.receive(), timeout=10)
            await asyncio.wait_for(completed.get(), timeout=10)
            assert producer.resets > 0 and not peers.active
            assert handler_errors == []
            assert len(producer.threads) == 1 and threading.get_ident() not in producer.threads
        finally:
            await peers.close_active()
            await runtime.close()
            await runner.cleanup()

    asyncio.run(exercise())
