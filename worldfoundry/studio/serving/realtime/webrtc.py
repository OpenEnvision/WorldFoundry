# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Instance-local aiortc sender hooks for NVENC feedback and frame receipts."""

from __future__ import annotations

import asyncio
from typing import Any

from worldfoundry.studio.serving.realtime.telemetry import DeliveryTelemetry


def bind_video_sender(sender: Any, track: Any, telemetry: DeliveryTelemetry) -> None:
    """Bind one aiortc 1.x sender without modifying process-global classes.

    aiortc exposes neither an encoded-packet feedback callback nor a frame-sent
    callback. Its next-frame boundary follows the complete RTP send, including
    the random timestamp origin. Reading the previous timestamp here also
    accounts for the final frame before the producer blocks waiting for input.
    The real-sender contract tests cover these aiortc-private integration points.
    """
    next_encoded = sender._next_encoded_frame
    pending: tuple[Any, float, float] | None = None

    async def next_frame(codec: Any) -> Any:
        nonlocal pending
        if pending is not None:
            context, queue_ms, encode_ms = pending
            pending = None
            telemetry.sent(
                context,
                rtp_timestamp=sender._RTCRtpSender__rtp_timestamp,
                queue_ms=queue_ms,
                encode_ms=encode_ms,
            )
        encoded = await next_encoded(codec)
        context = track.delivery_context
        if encoded is not None and context is not None:
            now = asyncio.get_running_loop().time()
            queue_ms = (track.delivery_started_at - context.queued_at_s) * 1000.0
            # Native tracks finish encoding before recv(); software tracks are
            # encoded by aiortc after recv() returns the tagged source frame.
            encode_ms = track.delivery_encode_ms
            if encode_ms is None:
                encode_ms = (now - track.delivery_started_at) * 1000.0
            pending = (context, queue_ms, encode_ms)
        return encoded

    sender._next_encoded_frame = next_frame

    if not getattr(track, "preserves_cuda_frames", False):
        return

    from aiortc.rtp import RTCP_PSFB_APP, RTCP_PSFB_FIR, RTCP_PSFB_PLI, RtcpPsfbPacket, unpack_remb_fci

    handle_rtcp = sender._handle_rtcp_packet

    async def feedback(packet: Any) -> None:
        # Retain aiortc's retransmission, receiver-report and software-codec
        # bookkeeping. NVENC consumes feedback on its own encoding worker.
        await handle_rtcp(packet)
        if not isinstance(packet, RtcpPsfbPacket):
            return
        if packet.fmt in (RTCP_PSFB_PLI, RTCP_PSFB_FIR):
            track.request_keyframe()
        elif packet.fmt == RTCP_PSFB_APP:
            try:
                bitrate, ssrcs = unpack_remb_fci(packet.fci)
            except ValueError:
                return
            if sender._ssrc in ssrcs:
                track.set_target_bitrate(bitrate)

    sender._handle_rtcp_packet = feedback
