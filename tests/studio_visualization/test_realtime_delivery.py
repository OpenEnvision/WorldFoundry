# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from worldfoundry.core.execution.realtime.contracts import RealtimeAxis
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
    ResidentWorldRuntime,
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
        assert isinstance(queued, PresentationFrame)
        assert isinstance(queued.pixels, np.ndarray)
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
        assert isinstance(frame, PresentationFrame)
        assert isinstance(frame.pixels, np.ndarray)
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
    manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, SimpleNamespace()), fps=16, chunk_frames=1)
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
    manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, SimpleNamespace()), fps=16, chunk_frames=1)
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
        manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, Runtime()), fps=16, chunk_frames=1)
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
        assert isinstance(frame, PresentationFrame)
        assert frame.context.control_ids == ("original",)
        assert frame.context.chunk_index == 1
        assert active.telemetry is not None
        assert active.telemetry.key_control == "later"
        active.frames.close()

    asyncio.run(exercise())


def test_sampled_axis_control_snapshot_is_deduplicated_and_bounded() -> None:
    telemetry = DeliveryTelemetry(lambda _: None)
    assert telemetry.sample_controls(
        keys=False, prompt=False, step=False, axis_control_ids=("mouse", "stick", "mouse")
    ) == ("mouse", "stick")
    controls = tuple(f"axis-{index}" for index in range(HISTORY_SIZE + 10))
    assert (
        telemetry.sample_controls(keys=False, prompt=False, step=False, axis_control_ids=controls)
        == controls[-HISTORY_SIZE:]
    )


def test_generation_snapshots_sampled_axes_before_inflight_messages() -> None:
    async def exercise() -> None:
        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(self, interactions: Any, **kwargs: Any) -> tuple[list[np.ndarray], float]:
                assert interactions == []
                assert kwargs["control_axes"] == {"yaw": [2.0, 0.0, 0.0, 0.0]}
                assert "axis_control_ids" not in kwargs
                await manager._handle_message(
                    active, json.dumps({"type": "axis", "source": "mouse", "dx": 9, "control_id": "later"})
                )
                active.closed = True
                return [np.zeros((2, 2, 3), np.uint8)], 1.0

        telemetry = DeliveryTelemetry(lambda _: None)
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=2),
            resampler=RealtimeControlResampler(fps=20, axes=(RealtimeAxis("yaw", "mouse_x"),)),
            telemetry=telemetry,
        )
        manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, Runtime()), fps=20, chunk_frames=4)
        await manager._handle_message(
            active, json.dumps({"type": "axis", "source": "mouse", "dx": 2, "control_id": "original"})
        )
        await manager._generation_worker(active)
        frame = await active.frames.get()
        assert isinstance(frame, PresentationFrame)
        assert frame.context.control_ids == ("original",)
        following = active.resampler.sample_input_chunk(4, wall_time=asyncio.get_running_loop().time())
        assert following.axis_control_ids == ("later",)
        assert following.axes == {"yaw": [9.0, 0.0, 0.0, 0.0]}
        telemetry.sent(frame.context, rtp_timestamp=123, queue_ms=1, encode_ms=2)
        assert telemetry.presented({"rtp_timestamp": 123, "control_id": "later", "control_to_display_ms": 50}) is None
        assert (
            telemetry.presented({"rtp_timestamp": 123, "control_id": "original", "control_to_display_ms": 50})
            is not None
        )
        active.frames.close()

    asyncio.run(exercise())


def test_idle_step_does_not_claim_queued_axis_controls() -> None:
    async def exercise() -> None:
        calls: list[dict[str, Any]] = []

        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(self, interactions: Any, **kwargs: Any) -> tuple[list[np.ndarray], float]:
                assert interactions == []
                calls.append(kwargs)
                if len(calls) == 2:
                    active.closed = True
                return [np.zeros((2, 2, 3), np.uint8)], 1.0

        telemetry = DeliveryTelemetry(lambda _: None)
        telemetry.step_controls.append("step")
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=4),
            resampler=RealtimeControlResampler(fps=20, axes=(RealtimeAxis("yaw", "gamepad_rx", mode="level"),)),
            telemetry=telemetry,
            pending_steps=1,
        )
        manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, Runtime()), fps=20, chunk_frames=4)
        await manager._handle_message(
            active, json.dumps({"type": "axis", "source": "gamepad", "rx": 0.5, "control_id": "stick"})
        )
        await manager._generation_worker(active)
        assert calls[0]["control_axes"] is None
        assert calls[1]["control_axes"] == {"yaw": [0.5] * 4}
        step_frame = await active.frames.get()
        axis_frame = await active.frames.get()
        assert isinstance(step_frame, PresentationFrame)
        assert isinstance(axis_frame, PresentationFrame)
        assert step_frame.context.control_ids == ("step",)
        assert axis_frame.context.control_ids == ("stick",)
        active.frames.close()

    asyncio.run(exercise())


def test_real_aiortc_sender_receipts_match_wire_timestamp_including_final_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiortc import RTCDtlsTransport, RTCRtpSender
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
        sender = RTCRtpSender(track, cast(RTCDtlsTransport, SimpleNamespace(state="connected", _send_rtp=send_rtp)))
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
        assert isinstance(frame, PresentationFrame)
        assert frame.context is context
        assert frame.pixels is frames[0]
        pixels = _rgb_frame_to_abgr_cuda(frame.pixels, torch).cpu().numpy()
        assert (pixels[..., :3] == 17).all()
        assert (pixels[..., 3] == 255).all()
        queue.close()

    asyncio.run(exercise())


def test_native_feedback_preserves_aiortc_handling_and_filters_remb_ssrc() -> None:
    from aiortc import RTCDtlsTransport, RTCRtpSender
    from aiortc.rtp import RTCP_PSFB_APP, RTCP_PSFB_FIR, RTCP_PSFB_PLI, RtcpPsfbPacket, pack_remb_fci

    async def exercise() -> None:
        calls: list[Any] = []
        track = SimpleNamespace(
            preserves_cuda_frames=True,
            request_keyframe=lambda: calls.append("idr"),
            set_target_bitrate=lambda value: calls.append(value),
        )
        sender = RTCRtpSender("video", cast(RTCDtlsTransport, SimpleNamespace(state="connected")))
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
        assert vars(sender)["_RTCRtpSender__force_keyframe"] is True

    asyncio.run(exercise())


def test_idle_keyup_clears_key_control_and_action_arrivals() -> None:
    async def exercise() -> None:
        chunk_count = 0

        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(self, interactions: Any, **kwargs: Any) -> tuple[list[np.ndarray], float]:
                nonlocal chunk_count
                del interactions, kwargs
                chunk_count += 1
                return [np.zeros((2, 2, 3), np.uint8)], 1.0

        telemetry = DeliveryTelemetry(lambda _: None)
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=8),
            resampler=RealtimeControlResampler(fps=16),
            telemetry=telemetry,
        )
        manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, Runtime()), fps=16, chunk_frames=2)
        worker = asyncio.create_task(manager._generation_worker(active))
        try:
            await manager._handle_message(
                active,
                json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}, "control_id": "down-1"}),
            )
            while chunk_count < 1:
                await asyncio.sleep(0.005)
            await manager._handle_message(
                active,
                json.dumps({"type": "action", "action": {"event": "keyup", "key": "w"}, "control_id": "up-1"}),
            )
            for _ in range(40):
                if not active.first_action.is_set() and telemetry.key_control is None:
                    break
                await asyncio.sleep(0.005)
            assert not active.first_action.is_set()
            assert telemetry.key_control is None
            assert list(active.action_arrivals) == []
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            active.frames.close()

    asyncio.run(exercise())


def test_dropped_steps_on_max_chunks_clear_step_controls_alignment() -> None:
    async def exercise() -> None:
        gate = asyncio.Event()
        calls: list[str | None] = []

        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(self, interactions: Any, *, prompt: str | None = None, **kwargs: Any):
                del interactions, kwargs
                calls.append(prompt)
                await gate.wait()
                return [np.zeros((2, 2, 3), np.uint8)], 1.0

        telemetry = DeliveryTelemetry(lambda _: None)
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=8),
            resampler=RealtimeControlResampler(fps=16, realtime_keys=()),
            telemetry=telemetry,
            prompt_scheduled=True,
            initial_segment_pending=True,
            base_prompt="fox",
            continuous=True,
            max_chunks=1,
        )
        active.first_action.set()
        manager = RealtimePeerManager(runtime=cast(ResidentWorldRuntime, Runtime()), fps=16, chunk_frames=1)
        worker = asyncio.create_task(manager._generation_worker(active))
        try:
            while len(calls) < 1:
                await asyncio.sleep(0.005)
            # Queue two steps while rollout_chunks is still 0; chunk 1 then completes max_chunks=1.
            await manager._handle_message(
                active, json.dumps({"type": "step", "request_id": "s1", "control_id": "step-old-1"})
            )
            await manager._handle_message(
                active, json.dumps({"type": "step", "request_id": "s2", "control_id": "step-old-2"})
            )
            assert list(telemetry.step_controls) == ["step-old-1", "step-old-2"]
            gate.set()
            for _ in range(50):
                if active.rollout_chunks == 1 and active.pending_steps == 0 and not telemetry.step_controls:
                    break
                await asyncio.sleep(0.005)
            assert active.pending_steps == 0
            assert list(telemetry.step_controls) == []

            # Restart rollout paused with a new prompt, then verify the next step keeps its own control_id.
            active.max_chunks = 2
            active.paused = True
            await manager._handle_message(
                active, json.dumps({"type": "prompt_update", "prompt": "owl", "control_id": "prompt-new"})
            )
            while len(calls) < 2:
                await asyncio.sleep(0.005)
            await manager._handle_message(
                active, json.dumps({"type": "step", "request_id": "s3", "control_id": "step-new"})
            )
            while len(calls) < 3:
                await asyncio.sleep(0.005)
            frames = [await active.frames.get() for _ in range(3)]
            assert isinstance(frames[2], PresentationFrame)
            assert "step-new" in frames[2].context.control_ids
            assert "step-old-1" not in frames[2].context.control_ids
        finally:
            gate.set()
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            active.frames.close()

    asyncio.run(exercise())


def test_hold_last_sender_emits_receipt_once_without_spamming_held_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiortc import RTCDtlsTransport, RTCRtpSender
    from aiortc.rtcrtpparameters import RTCRtpCodecParameters

    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_NVENC", raising=False)

    async def exercise() -> None:
        rtp_packets: list[bytes] = []
        messages: list[dict[str, Any]] = []

        async def send_rtp(data: bytes) -> None:
            rtp_packets.append(data)

        queue = LatestFrameBuffer(maxsize=2)
        track = _build_video_track(frames=queue, fps=100, presentation_mode="hold-last")
        sender = RTCRtpSender(track, cast(RTCDtlsTransport, SimpleNamespace(state="connected", _send_rtp=send_rtp)))
        bind_video_sender(sender, track, DeliveryTelemetry(messages.append))
        context = FrameContext(1, 0, ("control-1",), asyncio.get_running_loop().time())
        await queue.put_chunk([PresentationFrame(np.full((32, 32, 3), 24, np.uint8), context)])
        task = asyncio.create_task(
            sender._run_rtp(RTCRtpCodecParameters(mimeType="video/VP8", clockRate=90000, payloadType=96))
        )
        try:

            async def multiple_frames_sent() -> None:
                while len(rtp_packets) < 4:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(multiple_frames_sent(), 5)
            assert len(messages) == 1
            assert messages[0]["chunk_index"] == 1
            assert messages[0]["frame_index"] == 0
            assert messages[0]["control_ids"] == ["control-1"]
        finally:
            queue.close()
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
