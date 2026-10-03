# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral streaming video post-processing contracts.

Responsibility: per-stream session, ordered-chain, and layout-aware
chunk types that a generator, processors, and a sink can exchange
without copying frames. NumPy frame lists are accepted so Studio can
use the same boundary without a tensor stack.

Not this module: NVIDIA RTX VSR lives in
:mod:`worldfoundry.core.media.processing.rtx` and is imported only
when a preset selects it. Encode / decode lives in
:mod:`worldfoundry.core.media.codecs.video` and
:mod:`worldfoundry.core.safety.video_io`.

Public surface: :class:`VideoSpec`, :class:`VideoChunk`,
:class:`VideoPostProcessor`, :class:`VideoPostProcessorSession`,
:class:`VideoPostprocessChain`, :class:`VideoPostprocessStream`,
:func:`infer_video_spec`, :func:`video_frame_count`,
:func:`frame_list_from_chunks`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from worldfoundry.core.observability.nvtx import nvtx_range

VideoTensorLayout = Literal[
    "frame-list",
    "thwc",
    "tchw",
    "bthwc",
    "btchw",
    "bcthw",
    "bvthwc",
    "bvtchw",
]

# ──────────────────────────────────────────────────────────────────────────
# Stream ABI — static spec plus one chunk; layouts never imply a copy
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoSpec:
    """Static dimensions and cadence for one processor stream."""

    height: int
    width: int
    fps: float | None = None
    channels: int = 3
    dtype: str | None = None

    def __post_init__(self) -> None:
        """Reject non-positive geometry so a later processor cannot invent pixels."""

        if self.height < 1 or self.width < 1:
            raise ValueError("VideoSpec height and width must be positive.")
        if self.channels < 1:
            raise ValueError("VideoSpec channels must be positive.")
        if self.fps is not None and self.fps <= 0.0:
            raise ValueError("VideoSpec fps must be positive when provided.")


@dataclass(slots=True, kw_only=True)
class VideoChunk:
    """One segment exchanged between a generator, processors, and a sink."""

    frames: Any
    layout: VideoTensorLayout = "frame-list"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        """Count frames from the declared layout; empty chunks are valid (buffering)."""

        return video_frame_count(self.frames, layout=self.layout)


# ──────────────────────────────────────────────────────────────────────────
# Processor factory / session — one session owns one logical output stream
# ──────────────────────────────────────────────────────────────────────────


class VideoPostProcessorSession(ABC):
    """State owned by one logical output stream."""

    def prepare(self) -> None:
        """Prepare expensive state before the first measured chunk."""

    @abstractmethod
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Consume one chunk and return zero or more ordered output chunks."""

    @abstractmethod
    def flush(self) -> list[VideoChunk]:
        """Emit buffered tail frames and transition to a closed state."""

    def close(self) -> None:
        """Release the session while discarding any flushed tail output."""

        self.flush()


class VideoPostProcessor(ABC):
    """Factory for per-stream processor sessions."""

    @property
    def name(self) -> str:
        """Stable telemetry id; subclasses override when the class name is not the preset."""

        return type(self).__name__

    @property
    def processing_device(self) -> str | None:
        """Declare an explicit GPU device for optional stream profiling."""

        return None

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Declare the spec this processor emits; default is identity (no resize)."""

        return input_spec

    @abstractmethod
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Create isolated state for a new output stream."""


class IdentityVideoPostProcessor(VideoPostProcessor):
    """Zero-copy processor used to exercise the production integration path."""

    @property
    def name(self) -> str:
        """Preset id used by Studio when no real processor is selected."""

        return "identity"

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Open a passthrough session that still enforces flush-then-reject."""

        return _IdentityVideoPostProcessorSession(spec)


class _IdentityVideoPostProcessorSession(VideoPostProcessorSession):
    """Zero-copy session: return the same chunk until ``flush`` closes it."""

    def __init__(self, spec: VideoSpec) -> None:
        """Record the stream spec; identity does not allocate buffers."""

        self.spec = spec
        self._closed = False

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Return ``chunk`` unchanged; reject work after ``flush``."""

        if self._closed:
            raise RuntimeError("cannot process video after the identity session was flushed")
        return [chunk]

    def flush(self) -> list[VideoChunk]:
        """Close the session; identity never buffers a tail."""

        self._closed = True
        return []


@dataclass(frozen=True, slots=True)
class VideoPostprocessChain:
    """Ordered processor factories applied to each generated video stream."""

    processors: tuple[VideoPostProcessor, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Ordered processor ids for telemetry; empty when the chain is a no-op."""

        return tuple(processor.name for processor in self.processors)

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Compose declared specs left-to-right so the sink can size buffers first."""

        current = input_spec
        for processor in self.processors:
            current = processor.output_spec(current)
        return current

    def start(self, input_spec: VideoSpec) -> VideoPostProcessorSession:
        """Open one session per factory, feeding each the previous output spec."""

        sessions: list[VideoPostProcessorSession] = []
        current = input_spec
        try:
            for processor in self.processors:
                sessions.append(processor.start(current))
                current = processor.output_spec(current)
        except BaseException as error:
            _close_sessions(sessions, primary_error=error)
            raise
        return _VideoPostprocessChainSession(sessions)


def _close_sessions(
    sessions: Iterable[VideoPostProcessorSession], *, primary_error: BaseException | None = None,
) -> None:
    """Attempt every close while retaining the original failure."""

    error = primary_error
    for session in sessions:
        try:
            session.close()
        except BaseException as caught:
            if error is None:
                error = caught
            else:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"Postprocess cleanup also failed: {caught!r}")
    if primary_error is None and error is not None:
        raise error


class _VideoPostprocessChainSession(VideoPostProcessorSession):
    """Pipeline of sessions; a flush tail is fed into the remaining stages only."""

    def __init__(self, sessions: list[VideoPostProcessorSession]) -> None:
        """Own the already-started sessions; ``_closed`` is terminal after a failed flush."""

        self._sessions = sessions
        self._closed = False
        self._released = False

    def prepare(self) -> None:
        """Warm every stage before the first measured chunk so latency is not front-loaded."""

        if self._closed:
            raise RuntimeError("cannot prepare a closed post-processing chain")
        try:
            for session in self._sessions:
                session.prepare()
        except BaseException as error:
            self._closed = True
            self._released = True
            _close_sessions(self._sessions, primary_error=error)
            raise

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Push one chunk through every stage; reject work after ``flush``."""

        if self._closed:
            raise RuntimeError("cannot process a post-processing chain after flush()")
        try:
            return self._run(first_session_index=0, chunks=[chunk])
        except BaseException as error:
            self._closed = True
            self._released = True
            _close_sessions(self._sessions, primary_error=error)
            raise

    def flush(self) -> list[VideoChunk]:
        """Drain each stage once; a partial failure must not be retried.

        Retrying would re-emit tails already produced by an earlier stage.
        """

        if self._closed:
            return []
        self._closed = True
        outputs: list[VideoChunk] = []
        for index, session in enumerate(self._sessions):
            try:
                tail = session.flush()
                if tail:
                    outputs.extend(self._run(first_session_index=index + 1, chunks=tail))
            except BaseException as error:
                self._released = True
                _close_sessions(self._sessions, primary_error=error)
                raise
        return outputs

    def close(self) -> None:
        """Discard remaining tails; idempotent so ``reset`` can call it twice."""

        if self._released:
            return
        self._released = True
        self._closed = True
        _close_sessions(self._sessions)

    def _run(
        self,
        *,
        first_session_index: int,
        chunks: Iterable[VideoChunk],
    ) -> list[VideoChunk]:
        """Feed ``chunks`` through ``sessions[first_session_index:]`` in order.

        Each stage may expand or collapse the pending list (buffering).
        Starting mid-chain is how a flushed tail skips stages that already ran.
        """

        pending = list(chunks)
        for session in self._sessions[first_session_index:]:
            emitted: list[VideoChunk] = []
            for chunk in pending:
                emitted.extend(session.process(chunk))
            pending = emitted
        return pending


@dataclass(frozen=True, slots=True)
class VideoPostprocessStepStats:
    """Host dispatch, optional CUDA stream time, and frame-count telemetry."""

    elapsed_ms: float
    input_frames: int
    output_frames: int
    buffering: bool
    gpu_elapsed_ms: float | None = None

    def to_payload(self) -> dict[str, float | int | bool]:
        """JSON-safe receipt; ``buffering`` is True when the stage emitted no frames."""

        payload = {
            "elapsed_ms": self.elapsed_ms,
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "buffering": self.buffering,
        }
        if self.gpu_elapsed_ms is not None:
            payload["gpu_elapsed_ms"] = self.gpu_elapsed_ms
        return payload


class VideoPostprocessStream:
    """Own one stateful processor chain across autoregressive chunks."""

    def __init__(
        self,
        *,
        chain: VideoPostprocessChain | None = None,
        fps: float | None = None,
        profile_cuda: bool = False,
    ) -> None:
        """Bind a chain; the session is created lazily on the first ``process``."""

        self.chain = chain or VideoPostprocessChain()
        self.fps = fps
        if not isinstance(profile_cuda, bool):
            raise TypeError("profile_cuda must be a bool")
        self.profile_cuda = profile_cuda
        self.input_spec: VideoSpec | None = None
        self.output_spec: VideoSpec | None = None
        self.last_stats: VideoPostprocessStepStats | None = None
        self._session: VideoPostProcessorSession | None = None
        self._chunk_index = 0
        self._closed = False

    @property
    def processor_names(self) -> tuple[str, ...]:
        """Chain ids for Studio overlays; empty when the default identity chain is used."""

        return self.chain.names

    @property
    def chunk_index(self) -> int:
        """Number of ``process`` calls completed; used as ``autoregressive_index``."""

        return self._chunk_index

    def process(
        self,
        frames: Any,
        *,
        layout: VideoTensorLayout,
        metadata: dict[str, Any] | None = None,
    ) -> list[VideoChunk]:
        """Run one AR chunk; the first call freezes the input spec for the stream.

        A later chunk whose inferred spec differs is rejected so a processor
        cannot silently resize mid-stream. ``finish`` is terminal until
        :meth:`reset`.
        """

        if self._closed:
            raise RuntimeError("cannot process video after finish()")
        spec = infer_video_spec(frames, layout=layout, fps=self.fps)
        if self.input_spec is None:
            output_spec = self.chain.output_spec(spec)
            session = self.chain.start(spec)
            session.prepare()
            self.input_spec = spec
            self.output_spec = output_spec
            self._session = session
        elif spec != self.input_spec:
            raise ValueError(f"postprocess input stream specification changed from {self.input_spec!r} to {spec!r}.")
        if self._session is None:  # pragma: no cover - guarded by initialization above
            raise RuntimeError("postprocess session was not initialized")

        chunk_metadata = dict(metadata or {})
        chunk_metadata.setdefault("autoregressive_index", self._chunk_index)
        chunk = VideoChunk(frames=frames, layout=layout, metadata=chunk_metadata)
        cuda_timer = self._start_cuda_timer(frames) if self.profile_cuda else None
        started = time.perf_counter()
        with nvtx_range("worldfoundry.postprocess"):
            outputs = self._session.process(chunk)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        gpu_elapsed_ms = None
        if cuda_timer is not None:
            torch, stream, start_event = cuda_timer
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(stream)
            # Synchronization is restricted to explicitly requested profiling.
            end_event.synchronize()
            gpu_elapsed_ms = start_event.elapsed_time(end_event)
        self._validate_outputs(outputs)
        output_frames = sum(output.frame_count for output in outputs)
        self.last_stats = VideoPostprocessStepStats(
            elapsed_ms=elapsed_ms,
            input_frames=chunk.frame_count,
            output_frames=output_frames,
            buffering=output_frames == 0,
            gpu_elapsed_ms=gpu_elapsed_ms,
        )
        self._chunk_index += 1
        return outputs

    def _start_cuda_timer(self, frames: Any):
        """Time one resolved CUDA device; CPU paths keep host-only telemetry.

        Processors must enqueue work on that device's current stream, or join
        any private stream before returning. Mixed-device chains cannot be
        measured correctly with one CUDA event pair and are rejected only
        when CUDA profiling is explicitly requested.
        """

        declared = [processor.processing_device for processor in self.chain.processors]
        frame_values = frames if isinstance(frames, (list, tuple)) else (frames,)
        requested = [value for value in declared if value is not None]
        requested.extend(
            value.device for value in frame_values
            if getattr(getattr(value, "device", None), "type", None) == "cuda"
        )
        if not requested:
            return None
        import torch

        devices = set()
        for value in requested:
            device = torch.device(value)
            if device.type != "cuda":
                continue
            devices.add(device.index if device.index is not None else torch.cuda.current_device())
        if not devices:
            return None
        if len(devices) != 1:
            raise ValueError("CUDA postprocess profiling requires one CUDA device")
        device_index = devices.pop()
        stream = torch.cuda.current_stream(device_index)
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return torch, stream, start_event

    def finish(self) -> list[VideoChunk]:
        """Flush buffered tails once; later calls return an empty list."""

        if self._closed:
            return []
        self._closed = True
        outputs = [] if self._session is None else self._session.flush()
        self._validate_outputs(outputs)
        return outputs

    def reset(self) -> None:
        """Discard buffered output, release state, and accept a fresh stream."""

        session = self._session
        self.input_spec = None
        self.output_spec = None
        self.last_stats = None
        self._session = None
        self._chunk_index = 0
        self._closed = False
        if session is not None:
            session.close()

    def _validate_outputs(self, outputs: Iterable[VideoChunk]) -> None:
        """Require every non-empty chunk to match the chain's declared output spec."""

        expected = self.output_spec
        if expected is None:
            return
        for output in outputs:
            if output.frame_count == 0:
                continue
            actual = infer_video_spec(
                output.frames,
                layout=output.layout,
                fps=expected.fps,
            )
            if actual != expected:
                raise ValueError(f"postprocess chain declared output {expected!r} but emitted {actual!r}.")


# ──────────────────────────────────────────────────────────────────────────
# Layout helpers — infer H/W/C and T without copying; Studio needs frame-list
# ──────────────────────────────────────────────────────────────────────────


def infer_video_spec(
    frames: Any,
    *,
    layout: VideoTensorLayout,
    fps: float | None = None,
) -> VideoSpec:
    """Infer stream metadata from a declared layout without copying frames."""

    if layout == "frame-list":
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
            raise TypeError("frame-list video must be a sequence of HWC frames")
        if not frames:
            raise ValueError("cannot infer a video specification from an empty frame list")
        first_shape = _shape(frames[0], layout)
        if len(first_shape) != 3:
            raise ValueError(f"frame-list expects HWC frames; got shape {first_shape}.")
        height, width, channels = first_shape
        for index, frame in enumerate(frames[1:], start=1):
            if _shape(frame, layout) != first_shape:
                raise ValueError(f"frame-list shape changed at index {index}.")
        dtype = str(getattr(frames[0], "dtype", "unknown"))
    else:
        shape = _shape(frames, layout)
        if layout == "thwc":
            _require_ndim(shape, 4, layout)
            _, height, width, channels = shape
        elif layout == "tchw":
            _require_ndim(shape, 4, layout)
            _, channels, height, width = shape
        elif layout == "bthwc":
            _require_ndim(shape, 5, layout)
            _, _, height, width, channels = shape
        elif layout == "btchw":
            _require_ndim(shape, 5, layout)
            _, _, channels, height, width = shape
        elif layout == "bcthw":
            _require_ndim(shape, 5, layout)
            _, channels, _, height, width = shape
        elif layout == "bvthwc":
            _require_ndim(shape, 6, layout)
            _, _, _, height, width, channels = shape
        elif layout == "bvtchw":
            _require_ndim(shape, 6, layout)
            _, _, _, channels, height, width = shape
        else:  # pragma: no cover - Literal plus callers guard this at type-check time
            raise ValueError(f"unsupported video layout: {layout}")
        dtype = str(getattr(frames, "dtype", "unknown"))
    return VideoSpec(
        height=int(height),
        width=int(width),
        fps=fps,
        channels=int(channels),
        dtype=dtype,
    )


def video_frame_count(frames: Any, *, layout: VideoTensorLayout) -> int:
    """Return the time-axis length for ``layout``; ``frame-list`` is ``len(frames)``."""

    if layout == "frame-list":
        return len(frames)
    shape = _shape(frames, layout)
    time_dimension = {
        "thwc": 0,
        "tchw": 0,
        "bthwc": 1,
        "btchw": 1,
        "bcthw": 2,
        "bvthwc": 2,
        "bvtchw": 2,
    }.get(layout)
    if time_dimension is None:
        raise ValueError(f"unsupported video layout: {layout}")
    expected_dimensions = 4 if layout in {"thwc", "tchw"} else 5 if layout in {"bthwc", "btchw", "bcthw"} else 6
    _require_ndim(shape, expected_dimensions, layout)
    return int(shape[time_dimension])


def frame_list_from_chunks(chunks: Iterable[VideoChunk]) -> list[Any]:
    """Flatten frame-list chunks while preserving frame objects and order."""

    frames: list[Any] = []
    for chunk in chunks:
        if chunk.layout != "frame-list":
            raise ValueError(f"Studio frame transport requires frame-list output, got {chunk.layout!r}.")
        frames.extend(chunk.frames)
    return frames


def _shape(value: Any, layout: VideoTensorLayout) -> tuple[int, ...]:
    """Read ``value.shape`` without importing torch/numpy; missing shape is a TypeError."""

    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{layout} video value does not expose a shape")
    return tuple(int(dimension) for dimension in shape)


def _require_ndim(shape: tuple[int, ...], expected: int, layout: str) -> None:
    """Fail closed when a layout's rank does not match the tensor; never infer a missing axis."""

    if len(shape) != expected:
        raise ValueError(f"layout {layout!r} requires {expected} dimensions; got shape {shape}.")


__all__ = [
    "IdentityVideoPostProcessor",
    "VideoChunk",
    "VideoPostProcessor",
    "VideoPostProcessorSession",
    "VideoPostprocessChain",
    "VideoPostprocessStepStats",
    "VideoPostprocessStream",
    "VideoSpec",
    "VideoTensorLayout",
    "frame_list_from_chunks",
    "infer_video_spec",
    "video_frame_count",
]
