"""Frame normalization, transport encoding, and bounded realtime queues."""

from __future__ import annotations

import asyncio
import io
import math
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

from worldfoundry.core.execution.realtime.frame_prefetch import LazyCudaFrame, prefetch_to_numpy
from worldfoundry.studio.inference.execution import _normalize_frame_list, _to_uint8_rgb

MIN_OUTPUT_WIDTH = 160
MIN_OUTPUT_HEIGHT = 90
MAX_OUTPUT_WIDTH = 1920
MAX_OUTPUT_HEIGHT = 1920
MAX_OUTPUT_PIXELS = 1920 * 1080


class FrameQueuePolicy(str, Enum):
    """Congestion behavior for decoded realtime frames."""

    LATEST_INTERACTIVE = "latest-interactive"
    ORDERED_QUALITY = "ordered-quality"

    @classmethod
    def from_value(cls, value: "FrameQueuePolicy | str") -> "FrameQueuePolicy":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower().replace("_", "-")
        aliases = {
            "latest": cls.LATEST_INTERACTIVE,
            "interactive": cls.LATEST_INTERACTIVE,
            "ordered": cls.ORDERED_QUALITY,
            "quality": cls.ORDERED_QUALITY,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown frame queue policy {value!r}; expected one of: {supported}.") from exc


class RealtimePresentationMode(str, Enum):
    """Ownership model for realtime video pacing."""

    HOLD_LAST = "hold-last"
    REAL_FRAMES = "real-frames"

    @classmethod
    def from_value(cls, value: "RealtimePresentationMode | str") -> "RealtimePresentationMode":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower().replace("_", "-")
        aliases = {
            "legacy": cls.HOLD_LAST,
            "repeat": cls.HOLD_LAST,
            "repeat-last": cls.HOLD_LAST,
            "real": cls.REAL_FRAMES,
            "source": cls.REAL_FRAMES,
            "no-repeat": cls.REAL_FRAMES,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown realtime presentation mode {value!r}; expected one of: {supported}.") from exc


class WebSocketPacingMode(str, Enum):
    """Select which WebSocket layer owns frame pacing."""

    LEGACY_DUAL = "legacy-dual"
    CLIENT_ONLY = "client-only"

    @classmethod
    def from_value(cls, value: "WebSocketPacingMode | str") -> "WebSocketPacingMode":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower().replace("_", "-")
        aliases = {
            "legacy": cls.LEGACY_DUAL,
            "dual": cls.LEGACY_DUAL,
            "client": cls.CLIENT_ONLY,
            "browser": cls.CLIENT_ONLY,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown WebSocket pacing mode {value!r}; expected one of: {supported}.") from exc


class LatestFrameBuffer:
    """Bounded frame queue with selectable latency/quality congestion policy.

    The historical name remains as a compatibility surface. In
    ``latest-interactive`` mode it briefly backpressures and then evicts stale
    frames. In ``ordered-quality`` mode it waits for capacity and never drops a
    generated frame.
    """

    def __init__(
        self,
        *,
        maxsize: int,
        backpressure_ms: int = 0,
        policy: FrameQueuePolicy | str = FrameQueuePolicy.LATEST_INTERACTIVE,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self.backpressure_s = max(float(backpressure_ms), 0.0) / 1000.0
        self.policy = FrameQueuePolicy.from_value(policy)
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=maxsize)
        self._closed_event = asyncio.Event()
        self.dropped_frames = 0
        self.last_enqueue_ms = 0.0
        self.closed = False
        self.preserve_cuda_frames = False

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _put_ordered(self, frame: np.ndarray) -> bool:
        while not self.closed:
            try:
                self._queue.put_nowait(frame)
                return True
            except asyncio.QueueFull:
                try:
                    await asyncio.wait_for(self._closed_event.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
        return False

    async def put_chunk(self, frames: list[np.ndarray]) -> int:
        if self.closed:
            return 0
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.backpressure_s
        accepted = 0
        for frame in frames:
            rgb = frame if self.preserve_cuda_frames and isinstance(frame, LazyCudaFrame) else np.ascontiguousarray(frame)
            if self.policy is FrameQueuePolicy.ORDERED_QUALITY:
                if not await self._put_ordered(rgb):
                    break
                accepted += 1
                continue
            if self._queue.full() and self.backpressure_s > 0:
                remaining = deadline - loop.time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(self._queue.put(rgb), timeout=remaining)
                        accepted += 1
                        continue
                    except asyncio.TimeoutError:
                        pass
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped_frames += 1
                except asyncio.QueueEmpty:
                    break
            if self.closed:
                break
            self._queue.put_nowait(rgb)
            accepted += 1
        self.last_enqueue_ms = (loop.time() - started) * 1000.0
        return accepted

    async def get(self) -> np.ndarray:
        frame = await self._queue.get()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def get_nowait(self) -> np.ndarray:
        frame = self._queue.get_nowait()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._closed_event.set()
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(None)


class ChunkPresentationBuffer:
    """Present complete generated chunks through a two-frame sender mailbox.

    Active and pending chunks share an event-loop clock so transports request
    real frames; they do not add another pacing loop or repeat a held frame.
    """

    def __init__(
        self,
        *,
        fps: float,
        maximum_fps: float | None = None,
        mailbox_size: int = 2,
        policy: FrameQueuePolicy | str = FrameQueuePolicy.LATEST_INTERACTIVE,
    ) -> None:
        fps = float(fps)
        maximum_fps = fps if maximum_fps is None else float(maximum_fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a positive finite number")
        if not math.isfinite(maximum_fps) or maximum_fps < fps:
            raise ValueError("maximum_fps must be finite and >= fps")
        if mailbox_size < 1:
            raise ValueError("mailbox_size must be >= 1")

        self.maxsize = int(mailbox_size)
        self.policy = FrameQueuePolicy.from_value(policy)
        self._frame_interval = 1.0 / fps
        self._backlog_interval = 1.0 / maximum_fps
        self._mailbox: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=self.maxsize)
        self._pending_available = asyncio.Event()
        self._pending_available.set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._scheduled: asyncio.Handle | None = None
        self._next_frame_at: float | None = None
        self._active_chunk: list[np.ndarray] | None = None
        self._active_index = 0
        self._pending_chunk: list[np.ndarray] | None = None
        self._blocked_on_mailbox = False
        self.dropped_frames = 0
        self.dropped_chunks = 0
        self.presented_frames = 0
        self.last_enqueue_ms = 0.0
        self.closed = False
        self.preserve_cuda_frames = False

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("presentation buffer cannot move between event loops")
        return loop

    def qsize(self) -> int:
        active = 0 if self._active_chunk is None else len(self._active_chunk) - self._active_index
        pending = 0 if self._pending_chunk is None else len(self._pending_chunk)
        return self._mailbox.qsize() + active + pending

    async def put_chunk(self, frames: list[np.ndarray]) -> int:
        if self.closed:
            return 0
        loop = self._bind_loop()
        started = loop.time()
        prepared = [
            frame if self.preserve_cuda_frames and isinstance(frame, LazyCudaFrame) else np.ascontiguousarray(frame)
            for frame in frames
        ]
        if not prepared:
            self.last_enqueue_ms = (loop.time() - started) * 1000.0
            return 0

        while not self.closed:
            if self._active_chunk is None:
                self._active_chunk = prepared
                self._active_index = 0
                self._schedule_next(0.0)
                break
            if self._pending_chunk is None:
                self._pending_chunk = prepared
                self._pending_available.clear()
                self._accelerate_for_backlog()
                break
            if self.policy is FrameQueuePolicy.LATEST_INTERACTIVE:
                self.dropped_frames += len(self._pending_chunk)
                self.dropped_chunks += 1
                self._pending_chunk = prepared
                self._accelerate_for_backlog()
                break
            self._pending_available.clear()
            await self._pending_available.wait()

        self.last_enqueue_ms = (loop.time() - started) * 1000.0
        return 0 if self.closed else len(prepared)

    async def get(self) -> np.ndarray:
        frame = await self._mailbox.get()
        if frame is None:
            raise EOFError("presentation buffer closed")
        self._resume_after_mailbox_read()
        return frame

    def get_nowait(self) -> np.ndarray:
        frame = self._mailbox.get_nowait()
        if frame is None:
            raise EOFError("presentation buffer closed")
        self._resume_after_mailbox_read()
        return frame

    def _resume_after_mailbox_read(self) -> None:
        if not self.closed and self._blocked_on_mailbox:
            self._blocked_on_mailbox = False
            self._schedule_next(0.0)

    def _schedule_next(self, delay: float) -> None:
        if self.closed or self._scheduled is not None:
            return
        loop = self._loop
        if loop is None:
            return
        delay = max(float(delay), 0.0)
        self._next_frame_at = loop.time() + delay
        if delay == 0.0:
            self._scheduled = loop.call_soon(self._present_next)
        else:
            self._scheduled = loop.call_later(delay, self._present_next)

    def _accelerate_for_backlog(self) -> None:
        loop = self._loop
        if loop is None or self._scheduled is None or self._next_frame_at is None:
            return
        accelerated_at = loop.time() + self._backlog_interval
        if accelerated_at >= self._next_frame_at:
            return
        self._scheduled.cancel()
        self._scheduled = None
        self._next_frame_at = None
        self._schedule_next(self._backlog_interval)

    def _promote_pending(self) -> bool:
        if self._pending_chunk is None:
            return False
        self._active_chunk = self._pending_chunk
        self._active_index = 0
        self._pending_chunk = None
        self._pending_available.set()
        return True

    def _present_next(self) -> None:
        self._scheduled = None
        self._next_frame_at = None
        if self.closed:
            return
        if self._active_chunk is None and not self._promote_pending():
            return
        assert self._active_chunk is not None

        if self._mailbox.full():
            if self.policy is FrameQueuePolicy.ORDERED_QUALITY:
                self._blocked_on_mailbox = True
                return
            try:
                stale = self._mailbox.get_nowait()
                if stale is not None:
                    self.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass

        had_backlog = self._pending_chunk is not None
        self._mailbox.put_nowait(self._active_chunk[self._active_index])
        self._active_index += 1
        self.presented_frames += 1
        if self._active_index >= len(self._active_chunk):
            self._active_chunk = None
            self._active_index = 0
            self._promote_pending()
        if self._active_chunk is not None:
            interval = self._backlog_interval if had_backlog else self._frame_interval
            self._schedule_next(interval)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        self._next_frame_at = None
        self._active_chunk = None
        self._active_index = 0
        self._pending_chunk = None
        self._blocked_on_mailbox = False
        self._pending_available.set()
        while True:
            try:
                self._mailbox.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._mailbox.put_nowait(None)


def _cuda_transport_frames(candidate: Any) -> list[LazyCudaFrame] | None:
    """Snapshot supported byte frames on device, retaining producer ordering."""
    if isinstance(candidate, (list, tuple)) and candidate and all(isinstance(frame, LazyCudaFrame) for frame in candidate):
        try:
            device_sources = [frame.to_cuda_tensor() for frame in candidate]
        except RuntimeError:
            return None
        if all(source.is_cuda and str(source.dtype) == "torch.uint8" and source.ndim == 3 and source.shape[-1] == 3 for source in device_sources):
            return [frame.snapshot_cuda() for frame in candidate]
        return None
    try:
        import torch
    except ImportError:
        return None

    if not torch.is_tensor(candidate) or not candidate.is_cuda or candidate.dtype != torch.uint8:
        return None
    if candidate.ndim == 3 and candidate.shape[-1] == 3:
        candidate = candidate.unsqueeze(0)
    if candidate.ndim != 4 or candidate.shape[-1] != 3:
        return None
    with torch.cuda.device(candidate.device), torch.inference_mode():
        owned = candidate.detach().clone(memory_format=torch.contiguous_format)
        candidate.record_stream(torch.cuda.current_stream(candidate.device))
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(candidate.device))
    return [LazyCudaFrame(owned, index, source_event=ready) for index in range(owned.shape[0])]


def realtime_frames_from_result(result: Any, *, preserve_cuda: bool = False) -> list[Any]:
    """Extract in-memory RGB frames without invoking artifact exporters."""

    if isinstance(result, Iterator):
        close = getattr(result, "close", None)
        try:
            result = list(result)
        finally:
            if callable(close):
                close()

    candidates: list[Any] = []
    if isinstance(result, Mapping):
        for key in ("sr_videos", "videos", "frames", "video", "output", "images"):
            if key in result:
                candidates.append(result[key])
    else:
        candidates.append(result)

    for candidate in candidates:
        if preserve_cuda:
            device_frames = _cuda_transport_frames(candidate)
            if device_frames:
                return device_frames
        if isinstance(candidate, Image.Image):
            return [_to_uint8_rgb(candidate)]
        if isinstance(candidate, (list, tuple)):
            # Start every eligible device-to-host copy before waiting on the
            # first frame. Plain NumPy/PIL/tensor lists remain a no-op here.
            for frame in candidate:
                prefetch_to_numpy(frame)
            materializers = [getattr(frame, "to_numpy", None) for frame in candidate]
            if candidate and all(callable(materialize) for materialize in materializers):
                return [_to_uint8_rgb(materialize()) for materialize in materializers if callable(materialize)]
        frames = _normalize_frame_list(candidate)
        if frames:
            return [np.ascontiguousarray(frame) for frame in frames]
    raise RuntimeError(
        "Realtime stream returned no in-memory RGB frames. The model integration "
        "must return a tensor/array/image chunk instead of only an artifact path."
    )


def resize_rgb_frame(
    frame: np.ndarray,
    output_resolution: tuple[int, int] | None,
) -> np.ndarray:
    """Prepare one transport frame without ever enlarging model output."""

    rgb = np.ascontiguousarray(frame)
    if output_resolution is None or (rgb.shape[1], rgb.shape[0]) == output_resolution:
        return rgb
    width, height = output_resolution
    source_height, source_width = rgb.shape[:2]
    if width > MAX_OUTPUT_WIDTH or height > MAX_OUTPUT_HEIGHT or width * height > MAX_OUTPUT_PIXELS:
        raise ValueError(f"realtime transport resolution {width}x{height} exceeds its safe output budget.")
    if width > source_width or height > source_height:
        raise ValueError(
            f"realtime transport cannot upscale {source_width}x{source_height} model output to {width}x{height}."
        )
    scale = min(width / source_width, height / source_height)
    fitted = (
        max(min(int(round(source_width * scale)), width), 1),
        max(min(int(round(source_height * scale)), height), 1),
    )
    resized = Image.fromarray(rgb, mode="RGB").resize(
        fitted,
        resample=Image.Resampling.BILINEAR,
    )
    if fitted == output_resolution:
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    canvas = Image.new("RGB", output_resolution, "black")
    canvas.paste(resized, ((width - fitted[0]) // 2, (height - fitted[1]) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def resize_rgb_frames(
    frames: list[np.ndarray],
    *,
    output_resolution: tuple[int, int] | None,
) -> list[np.ndarray]:
    return [resize_rgb_frame(frame, output_resolution) for frame in frames]


def encode_jpeg_frames(
    frames: list[np.ndarray],
    *,
    quality: int,
    subsampling: int = 1,
    output_resolution: tuple[int, int] | None = None,
) -> list[bytes]:
    """Encode a generated chunk for the same-port WebSocket fallback."""

    packets: list[bytes] = []
    for frame in frames:
        output = io.BytesIO()
        image = Image.fromarray(
            resize_rgb_frame(frame, output_resolution),
            mode="RGB",
        )
        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=False,
            subsampling=subsampling,
        )
        packets.append(output.getvalue())
    return packets


__all__ = [
    "ChunkPresentationBuffer",
    "FrameQueuePolicy",
    "LatestFrameBuffer",
    "MAX_OUTPUT_HEIGHT",
    "MAX_OUTPUT_PIXELS",
    "MAX_OUTPUT_WIDTH",
    "MIN_OUTPUT_HEIGHT",
    "MIN_OUTPUT_WIDTH",
    "RealtimePresentationMode",
    "WebSocketPacingMode",
    "encode_jpeg_frames",
    "realtime_frames_from_result",
    "resize_rgb_frame",
    "resize_rgb_frames",
]
