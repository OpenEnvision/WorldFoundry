# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Frame identity and browser-clock control latency for one realtime peer."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from worldfoundry.core.observability.realtime_timing import TimingDistribution

HISTORY_SIZE = 256


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Model input snapshot attached to a decoded frame, outside causal state."""

    chunk_index: int
    frame_index: int
    control_ids: tuple[str, ...]
    queued_at_s: float


@dataclass(frozen=True, slots=True)
class PresentationFrame:
    """Pixels and their identity travel together through eviction and pacing."""

    pixels: Any
    context: FrameContext


def frame_pixels(frame: Any) -> Any:
    """Unwrap transport metadata without materializing CUDA pixels."""
    return frame.pixels if isinstance(frame, PresentationFrame) else frame


def frame_context(frame: Any) -> FrameContext | None:
    """Return the identity of a generated frame; untagged frames have none."""
    return frame.context if isinstance(frame, PresentationFrame) else None


class DeliveryTelemetry:
    """Bounded per-session correlation of sampled controls and presented frames.

    Browser durations use only the browser's monotonic clock. Server queue and
    encode durations use only the server clock. No clock offset is inferred.
    """

    def __init__(self, send: Callable[[dict[str, Any]], None], *, warmup_chunks: int = 0) -> None:
        self._send = send
        self._warmup_chunks = warmup_chunks
        self.key_control: str | None = None
        self.pending_prompt_control: str | None = None
        self.prompt_control: str | None = None
        self.step_controls: deque[str | None] = deque()
        self._sent: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._reported: OrderedDict[str, None] = OrderedDict()
        self._latencies: deque[float] = deque(maxlen=HISTORY_SIZE)

    @staticmethod
    def control_id(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("control_id")
        return value if isinstance(value, str) and 0 < len(value) <= 128 else None

    def sample_controls(self, *, keys: bool, prompt: bool, step: bool) -> tuple[str, ...]:
        """Snapshot only controls consumed by this generation call."""
        if prompt:
            self.prompt_control = self.pending_prompt_control
            self.pending_prompt_control = None
        step_control = self.step_controls.popleft() if step and self.step_controls else None
        return tuple(
            dict.fromkeys(
                value
                for value in (
                    self.key_control if keys else None,
                    self.prompt_control,
                    step_control,
                )
                if value is not None
            )
        )

    def sent(self, context: FrameContext, *, rtp_timestamp: int, queue_ms: float, encode_ms: float) -> None:
        """Publish the actual RTP timestamp after all packets of a frame leave."""
        message = {
            "type": "frame_sent",
            "rtp_timestamp": rtp_timestamp,
            "chunk_index": context.chunk_index,
            "frame_index": context.frame_index,
            "control_ids": list(context.control_ids),
            "queue_ms": max(queue_ms, 0.0),
            "encode_ms": max(encode_ms, 0.0),
        }
        self._sent[rtp_timestamp] = message
        self._sent.move_to_end(rtp_timestamp)
        while len(self._sent) > HISTORY_SIZE:
            self._sent.popitem(last=False)
        self._send(message)

    def presented(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Accept one browser presentation sample per control on a sent frame."""
        timestamp = payload.get("rtp_timestamp")
        if type(timestamp) is not int:
            return None
        sent = self._sent.get(timestamp)
        control = self.control_id(payload)
        if sent is None or control is None or control not in sent["control_ids"] or control in self._reported:
            return None
        value = payload.get("control_to_display_ms")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            return None
        self._reported[control] = None
        while len(self._reported) > HISTORY_SIZE:
            self._reported.popitem(last=False)
        if sent["chunk_index"] > self._warmup_chunks:
            self._latencies.append(float(value))
            summary = TimingDistribution.from_values(list(self._latencies)).to_payload()
            self._send({"type": "presentation_timing", "control_to_display": summary})
        return {
            "rtp_timestamp": timestamp,
            "chunk_index": sent["chunk_index"],
            "frame_index": sent["frame_index"],
            "control_id": control,
            "control_to_display_ms": float(value),
            "queue_ms": sent["queue_ms"],
            "encode_ms": sent["encode_ms"],
        }


def browser_transport_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Keep finite browser WebRTC gauges in their declared units."""
    names = (
        "rtt_ms",
        "jitter_ms",
        "jitter_buffer_ms",
        "decode_ms",
        "bitrate_bps",
        "packets_lost",
        "frames_dropped",
        "frames_decoded",
    )
    return {
        name: float(value)
        for name in names
        if isinstance(value := payload.get(name), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    }
