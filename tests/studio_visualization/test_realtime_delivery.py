# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from worldfoundry.studio.serving.realtime.input import RealtimeControlResampler
from worldfoundry.studio.serving.realtime.media import ChunkPresentationBuffer, FrameQueuePolicy, LatestFrameBuffer
from worldfoundry.studio.serving.realtime.telemetry import (
    HISTORY_SIZE,
    DeliveryTelemetry,
    FrameContext,
    PresentationFrame,
    browser_transport_metrics,
)
from worldfoundry.studio.serving.realtime.webrtc import bind_video_sender
from worldfoundry.studio.visualization.backends.world_realtime import (
    RealtimePeerManager,
    _ActivePeer,
    _build_video_track,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("presentation", [False, True])
def test_frame_identity_survives_normalization_and_ordered_pacing(presentation: bool) -> None:
    async def exercise() -> None:
        queue = (
            ChunkPresentationBuffer(fps=1000, policy=FrameQueuePolicy.ORDERED_QUALITY)
            if presentation
            else LatestFrameBuffer(maxsize=2, policy=FrameQueuePolicy.ORDERED_QUALITY)
        )
        pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::2]
        context = FrameContext(1, 0, ("key-1",), 0.0)
        await queue.put_chunk([PresentationFrame(pixels, context)])
        queued = await queue.get()
        assert queued.context is context
        assert queued.pixels.flags.c_contiguous
        np.testing.assert_array_equal(queued.pixels, pixels)
        assert queue.dropped_frames == 0
        queue.close()

    asyncio.run(exercise())


def test_evicted_frame_metadata_does_not_shift_to_new_pixels() -> None:
    async def exercise() -> None:
        queue = LatestFrameBuffer(maxsize=1)
        await queue.put_chunk(
            [
                PresentationFrame(np.full((2, 2, 3), index, np.uint8), FrameContext(index, 0, (str(index),), 0))
                for index in (1, 2)
            ]
        )
        frame = await queue.get()
        assert frame.context.chunk_index == 2
        assert (frame.pixels == 2).all()
        assert queue.dropped_frames == 1
        queue.close()

    asyncio.run(exercise())


def test_presentation_samples_require_actual_sent_identity_and_are_bounded() -> None:
    messages: list[dict[str, Any]] = []
    telemetry = DeliveryTelemetry(messages.append)
    context = FrameContext(3, 2, ("key-1",), 42)
    payload = {"rtp_timestamp": 4294967295, "control_id": "key-1", "control_to_display_ms": 120.0}
    assert telemetry.presented(payload) is None
    telemetry.sent(context, rtp_timestamp=4294967295, queue_ms=10, encode_ms=2)
    assert telemetry.presented({**payload, "control_id": "unsampled"}) is None
    assert telemetry.presented({**payload, "control_to_display_ms": float("nan")}) is None
    assert telemetry.presented({**payload, "control_to_display_ms": -1}) is None
    sample = telemetry.presented(payload)
    assert sample is not None
    assert sample["chunk_index"] == 3
    assert sample["frame_index"] == 2
    assert sample["queue_ms"] == 10
    assert messages[-1]["control_to_display"]["p90_ms"] == 120
    assert telemetry.presented(payload) is None
    for timestamp in range(HISTORY_SIZE + 2):
        telemetry.sent(context, rtp_timestamp=timestamp, queue_ms=0, encode_ms=1)
    assert len(telemetry._sent) == HISTORY_SIZE
    assert 4294967295 not in telemetry._sent
    assert DeliveryTelemetry(messages.append).presented(payload) is None
    warmup_messages: list[dict[str, Any]] = []
    warmup = DeliveryTelemetry(warmup_messages.append, warmup_chunks=3)
    warmup.sent(context, rtp_timestamp=4294967295, queue_ms=0, encode_ms=1)
    assert warmup.presented(payload) is not None
    assert [item["type"] for item in warmup_messages] == ["frame_sent"]


def test_client_timing_trace_is_separate_from_chunk_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.runtime.realtime_regression import read_realtime_trace

    path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PERF_JSONL", str(path))
    manager = RealtimePeerManager(runtime=SimpleNamespace(), fps=16, chunk_frames=1)
    active = _ActivePeer(
        peer=None,
        channel=None,
        frames=LatestFrameBuffer(maxsize=1),
        resampler=RealtimeControlResampler(fps=16),
        telemetry=DeliveryTelemetry(lambda _: None),
    )
    assert active.telemetry is not None
    active.telemetry.sent(FrameContext(1, 2, ("control",), 0), rtp_timestamp=30, queue_ms=5, encode_ms=1)

    async def exercise() -> None:
        await manager._handle_message(
            active,
            json.dumps(
                {
                    "type": "frame_presented",
                    "rtp_timestamp": 30,
                    "control_id": "control",
                    "control_to_display_ms": 75,
                }
            ),
        )
        await manager._handle_message(active, json.dumps({"type": "browser_transport", "rtt_ms": 10}))

    asyncio.run(exercise())
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["event"] for item in events] == ["realtime.frame_presented", "realtime.browser_transport"]
    assert events[0]["fields"]["control_to_display_ms"] == 75
    assert all(item["fields"]["session_id"] == active.session_id for item in events)
    assert read_realtime_trace(path) == []


def test_congested_channel_skips_diagnostics_without_suppressing_controls() -> None:
    messages: list[str] = []
    channel = SimpleNamespace(readyState="open", bufferedAmount=65_536, send=messages.append)
    active = _ActivePeer(
        peer=None,
        channel=channel,
        frames=LatestFrameBuffer(maxsize=1),
        resampler=RealtimeControlResampler(fps=16),
    )
    manager = RealtimePeerManager(runtime=SimpleNamespace(), fps=16, chunk_frames=1)
    manager._send_telemetry(active, {"type": "frame_sent"})
    assert messages == []
    manager._send(channel, {"type": "step_ack"})
    assert json.loads(messages[-1])["type"] == "step_ack"
    channel.bufferedAmount = 0
    manager._send_telemetry(active, {"type": "frame_sent"})
    assert json.loads(messages[-1])["type"] == "frame_sent"


def test_control_snapshots_do_not_include_unsampled_keys_or_queued_steps() -> None:
    telemetry = DeliveryTelemetry(lambda _: None)
    telemetry.key_control = "key"
    telemetry.pending_prompt_control = "prompt"
    telemetry.step_controls.extend(("step-1", "step-2"))
    assert telemetry.sample_controls(keys=True, prompt=True, step=False) == ("key", "prompt")
    assert telemetry.sample_controls(keys=False, prompt=False, step=True) == ("prompt", "step-1")
    assert telemetry.sample_controls(keys=False, prompt=False, step=True) == ("prompt", "step-2")
    assert browser_transport_metrics({"rtt_ms": 2, "decode_ms": float("inf"), "frames_dropped": -1}) == {"rtt_ms": 2}


def test_generation_stamps_input_snapshot_before_inflight_control_changes() -> None:
    async def exercise() -> None:
        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(self, interactions: Any, **kwargs: Any) -> tuple[list[np.ndarray], float]:
                assert interactions == ["forward"]
                await manager._handle_message(
                    active,
                    json.dumps(
                        {
                            "type": "action",
                            "action": {"event": "keydown", "key": "a"},
                            "control_id": "later",
                        }
                    ),
                )
                active.closed = True
                return [np.zeros((2, 2, 3), np.uint8)], 1.0

        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=2),
            resampler=RealtimeControlResampler(fps=16),
            telemetry=DeliveryTelemetry(lambda _: None),
        )
        manager = RealtimePeerManager(runtime=Runtime(), fps=16, chunk_frames=1)
        await manager._handle_message(
            active,
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "keydown", "key": "w"},
                    "control_id": "original",
                }
            ),
        )
        await manager._generation_worker(active)
        frame = await active.frames.get()
        assert frame.context.control_ids == ("original",)
        assert frame.context.chunk_index == 1
        assert active.telemetry.key_control == "later"
        active.frames.close()

    asyncio.run(exercise())


def test_real_aiortc_sender_receipts_match_wire_timestamp_including_final_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiortc import RTCRtpSender
    from aiortc.rtcrtpparameters import RTCRtpCodecParameters
    from aiortc.rtp import RtpPacket

    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_NVENC", raising=False)

    async def exercise() -> None:
        packets: list[Any] = []
        messages: list[dict[str, Any]] = []

        async def send_rtp(data: bytes) -> None:
            packets.append(RtpPacket.parse(data))

        queue = LatestFrameBuffer(maxsize=2)
        track = _build_video_track(frames=queue, fps=30, presentation_mode="real-frames")
        sender = RTCRtpSender(track, SimpleNamespace(state="connected", _send_rtp=send_rtp))
        bind_video_sender(sender, track, DeliveryTelemetry(messages.append))
        now = asyncio.get_running_loop().time()
        await queue.put_chunk(
            [
                PresentationFrame(np.full((32, 32, 3), index * 40, np.uint8), FrameContext(1, index, ("control",), now))
                for index in range(2)
            ]
        )
        task = asyncio.create_task(
            sender._run_rtp(RTCRtpCodecParameters(mimeType="video/VP8", clockRate=90000, payloadType=96))
        )
        try:

            async def received() -> None:
                while len(messages) < 2:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(received(), 5)
            markers = [packet for packet in packets if packet.marker]
            assert len(markers) == 2
            assert [message["rtp_timestamp"] for message in messages] == [packet.timestamp for packet in markers]
            assert [message["frame_index"] for message in messages] == [0, 1]
            assert all(message["encode_ms"] >= 0 and message["queue_ms"] >= 0 for message in messages)
            assert not task.done()  # The final receipt did not need another source frame.
        finally:
            queue.close()
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())


@pytest.mark.gpu
@pytest.mark.parametrize("presentation", [False, True])
def test_device_frame_receipt_keeps_cuda_storage(presentation: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from worldfoundry.core.execution.realtime.frame_prefetch import LazyCudaFrame
    from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
    from worldfoundry.studio.serving.realtime.nvenc import _rgb_frame_to_abgr_cuda

    assert torch.cuda.is_available(), "device receipts require a real CUDA device"
    source = torch.full((1, 4, 4, 3), 17, dtype=torch.uint8, device="cuda")
    frames = realtime_frames_from_result({"frames": source}, preserve_cuda=True)
    source.fill_(99)

    def reject_host(self: Any) -> Any:
        raise AssertionError("frame telemetry materialized CUDA pixels on the host")

    monkeypatch.setattr(LazyCudaFrame, "to_numpy", reject_host)

    async def exercise() -> None:
        queue = ChunkPresentationBuffer(fps=1000) if presentation else LatestFrameBuffer(maxsize=1)
        queue.preserve_cuda_frames = True
        context = FrameContext(1, 0, ("control",), 0)
        await queue.put_chunk([PresentationFrame(frames[0], context)])
        frame = await queue.get()
        assert frame.context is context
        assert frame.pixels is frames[0]
        pixels = _rgb_frame_to_abgr_cuda(frame.pixels, torch).cpu().numpy()
        assert (pixels[..., :3] == 17).all()
        assert (pixels[..., 3] == 255).all()
        queue.close()

    asyncio.run(exercise())


def test_native_feedback_preserves_aiortc_handling_and_filters_remb_ssrc() -> None:
    from aiortc import RTCRtpSender
    from aiortc.rtp import RTCP_PSFB_APP, RTCP_PSFB_FIR, RTCP_PSFB_PLI, RtcpPsfbPacket, pack_remb_fci

    async def exercise() -> None:
        calls: list[Any] = []
        track = SimpleNamespace(
            preserves_cuda_frames=True,
            request_keyframe=lambda: calls.append("idr"),
            set_target_bitrate=lambda value: calls.append(value),
        )
        sender = RTCRtpSender("video", SimpleNamespace(state="connected"))
        bind_video_sender(sender, track, DeliveryTelemetry(lambda _: None))
        for fmt in (RTCP_PSFB_PLI, RTCP_PSFB_FIR):
            await sender._handle_rtcp_packet(RtcpPsfbPacket(fmt=fmt, ssrc=1, media_ssrc=sender._ssrc))
        for ssrc in (sender._ssrc, sender._ssrc ^ 1):
            await sender._handle_rtcp_packet(
                RtcpPsfbPacket(
                    fmt=RTCP_PSFB_APP,
                    ssrc=1,
                    media_ssrc=0,
                    fci=pack_remb_fci(400_000, [ssrc]),
                )
            )
        await sender._handle_rtcp_packet(RtcpPsfbPacket(fmt=RTCP_PSFB_APP, ssrc=1, media_ssrc=0, fci=b"invalid"))
        assert calls == ["idr", "idr", 400_000]
        assert sender._RTCRtpSender__force_keyframe is True

    asyncio.run(exercise())
