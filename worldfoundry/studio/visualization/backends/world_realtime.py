"""Low-latency WebRTC runtime for the standalone interactive world frontend.

The offline Studio run contract is intentionally not used after session setup.
One model context stays resident on one execution thread, sparse input edges are
resampled at chunk boundaries, and decoded RGB frames move through a bounded
in-memory queue into one long-lived WebRTC video track.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import numpy as np
from PIL import Image

from worldfoundry.core.acceleration.prewarm import run_async_prewarm_sequence
from worldfoundry.core.execution.realtime import RealtimeSpec
from worldfoundry.core.observability.logging_setup import get_logger, write_jsonl_event
from worldfoundry.core.observability.realtime_timing import RealtimeChunkTiming, RealtimeTimingWindow
from worldfoundry.core.video.postprocess import (
    IdentityVideoPostProcessor,
    VideoPostprocessChain,
    VideoPostprocessStream,
    VideoSpec,
    frame_list_from_chunks,
)
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.execution import IMAGE_EXTS, VIDEO_EXTS, PreparedInputs, StudioManager
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.serving import (
    bind_security_warning,
    path_allowed,
    request_token_valid,
    require_auth_token_for_host,
)
from worldfoundry.studio.serving.realtime import media as realtime_media
from worldfoundry.studio.serving.realtime.input import (
    ControlSegment,
    RealtimeControlResampler,
    RealtimeControlState,
    interactions_from_keys,
    interactions_from_segments,
)
from worldfoundry.studio.serving.realtime.media import (
    ChunkPresentationBuffer,
    FrameQueuePolicy,
    LatestFrameBuffer,
    RealtimePresentationMode,
    WebSocketPacingMode,
    realtime_frames_from_result,
)
from worldfoundry.studio.serving.realtime.media import encode_jpeg_frames as _encode_jpeg_frames
from worldfoundry.studio.serving.realtime.media import resize_rgb_frames as _resize_rgb_frames

_MIN_OUTPUT_WIDTH = realtime_media.MIN_OUTPUT_WIDTH
_MIN_OUTPUT_HEIGHT = realtime_media.MIN_OUTPUT_HEIGHT
_MAX_OUTPUT_WIDTH = realtime_media.MAX_OUTPUT_WIDTH
_MAX_OUTPUT_HEIGHT = realtime_media.MAX_OUTPUT_HEIGHT
_MAX_OUTPUT_PIXELS = realtime_media.MAX_OUTPUT_PIXELS
_DREAMX_WORLD_MODEL_ID = "dreamx-world-5b-cam"
_PROMPT_BOUNDARY_MODEL_IDS = frozenset(
    {
        "dreamx-world-5b-cam",
        "helios",
        "lingbot-world",
        "lingbot-world-v2",
        "matrix-game-3",
        "sana-wm",
    }
)
_TEXT_EVENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_TEXT_EVENTS = 12
_MAX_TEXT_EVENT_PROMPT = 1000
_EXPLICIT_DISCONNECT_CLOSE_GRACE_SECONDS = 0.25
_CHUNK_METRIC_NAMES = (
    "condition_ms",
    "model_ms",
    "decode_ms",
    "postprocess_ms",
    "runtime_ms",
    "copy_ms",
    "cache_ms",
    "cache_frames",
    "cache_tokens",
    "compile_active",
    "cuda_graph",
)

logger = logging.getLogger(__name__)
event_logger = get_logger(__name__)


def _metric_float(
    metrics: Mapping[str, Any],
    name: str,
    default: float = 0.0,
) -> float:
    value = metrics.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunk_metric_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name in _CHUNK_METRIC_NAMES:
        if name not in metrics:
            continue
        try:
            payload[name] = round(float(metrics[name]), 1)
        except (TypeError, ValueError):
            continue
    return payload


def normalize_text_events(value: Any) -> list[dict[str, str]]:
    """Validate the small, user-authored event catalog carried by a session."""

    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("text_events must be a JSON list.")
    if len(value) > _MAX_TEXT_EVENTS:
        raise ValueError(f"text_events supports at most {_MAX_TEXT_EVENTS} events.")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"text_events[{index - 1}] must be an object.")
        event_id = str(item.get("event_id") or item.get("id") or "").strip()
        if not _TEXT_EVENT_ID.fullmatch(event_id):
            raise ValueError(f"text_events[{index - 1}].event_id must use 1-64 letters, numbers, or _.:-.")
        if event_id in seen:
            raise ValueError(f"Duplicate text event id: {event_id}.")
        label = str(item.get("label") or event_id).strip()
        prompt = str(item.get("prompt") or "").strip()
        category = str(item.get("category") or "event").strip()
        if not label or len(label) > 64:
            raise ValueError(f"Text event {event_id!r} label must contain 1-64 characters.")
        if not prompt or len(prompt) > _MAX_TEXT_EVENT_PROMPT:
            raise ValueError(f"Text event {event_id!r} prompt must contain 1-{_MAX_TEXT_EVENT_PROMPT} characters.")
        if not category or len(category) > 64:
            raise ValueError(f"Text event {event_id!r} category must contain 1-64 characters.")
        seen.add(event_id)
        result.append(
            {
                "event_id": event_id,
                "label": label,
                "prompt": prompt,
                "category": category,
            }
        )
    return result


def normalize_output_resolution(value: Any) -> tuple[int, int] | None:
    """Return an even transport resolution, or ``None`` for model-native output."""

    if value in (None, "", "native"):
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if not match:
            raise ValueError("output_resolution must be 'native' or WIDTHxHEIGHT.")
        width, height = (int(match.group(1)), int(match.group(2)))
    elif isinstance(value, Mapping):
        mode = str(value.get("mode") or "").strip().lower()
        if mode == "native":
            return None
        try:
            width = int(value.get("width") or 0)
            height = int(value.get("height") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("output_resolution width and height must be integers.") from exc
    else:
        raise ValueError("output_resolution must be an object, WIDTHxHEIGHT, or 'native'.")
    if not (_MIN_OUTPUT_WIDTH <= width <= _MAX_OUTPUT_WIDTH):
        raise ValueError(f"output width must be between {_MIN_OUTPUT_WIDTH} and {_MAX_OUTPUT_WIDTH}.")
    if not (_MIN_OUTPUT_HEIGHT <= height <= _MAX_OUTPUT_HEIGHT):
        raise ValueError(f"output height must be between {_MIN_OUTPUT_HEIGHT} and {_MAX_OUTPUT_HEIGHT}.")
    if width % 2 or height % 2:
        raise ValueError("output width and height must be even for realtime video encoding.")
    return width, height


@dataclass(frozen=True, slots=True)
class OutputResolutionSnapshot:
    """Immutable resolution decision for one generated transport chunk."""

    dimensions: tuple[int, int] | None
    source_dimensions: tuple[int, int] | None
    revision: int

    def to_payload(self) -> dict[str, Any]:
        if self.dimensions is not None:
            return {
                "mode": "fixed",
                "width": self.dimensions[0],
                "height": self.dimensions[1],
            }
        payload: dict[str, Any] = {"mode": "native"}
        if self.source_dimensions is not None:
            payload.update(
                width=self.source_dimensions[0],
                height=self.source_dimensions[1],
            )
        return payload


@dataclass(slots=True)
class OutputResolutionState:
    """Per-session output boundary with source-aware anti-upscale checks."""

    width: int | None = None
    height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_pixels: int = _MAX_OUTPUT_PIXELS
    source_width: int | None = None
    source_height: int | None = None
    revision: int = 0

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        maximum: tuple[int, int] | None = None,
        max_pixels: int | None = None,
    ) -> "OutputResolutionState":
        state = cls(
            max_width=maximum[0] if maximum is not None else None,
            max_height=maximum[1] if maximum is not None else None,
            max_pixels=min(max(int(max_pixels or _MAX_OUTPUT_PIXELS), 1), _MAX_OUTPUT_PIXELS),
        )
        state._set(value, increment_revision=False)
        return state

    @property
    def dimensions(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    @property
    def source_dimensions(self) -> tuple[int, int] | None:
        if self.source_width is None or self.source_height is None:
            return None
        return self.source_width, self.source_height

    def _validate_dimensions(self, dimensions: tuple[int, int] | None) -> None:
        if dimensions is None:
            return
        width, height = dimensions
        if self.max_width is not None and width > self.max_width:
            raise ValueError(
                f"output width {width} exceeds the model-native width {self.max_width}; "
                "realtime output does not upscale generated frames."
            )
        if self.max_height is not None and height > self.max_height:
            raise ValueError(
                f"output height {height} exceeds the model-native height {self.max_height}; "
                "realtime output does not upscale generated frames."
            )
        if width * height > self.max_pixels:
            raise ValueError(
                f"output resolution {width}x{height} exceeds the realtime pixel budget of {self.max_pixels} pixels."
            )
        source = self.source_dimensions
        if source is not None and (width > source[0] or height > source[1]):
            raise ValueError(
                f"output resolution {width}x{height} exceeds the generated frame "
                f"{source[0]}x{source[1]}; realtime output does not upscale."
            )

    def _set(self, value: Any, *, increment_revision: bool) -> bool:
        dimensions = normalize_output_resolution(value)
        self._validate_dimensions(dimensions)
        if dimensions == self.dimensions:
            return False
        self.width, self.height = dimensions or (None, None)
        if increment_revision:
            self.revision += 1
        return True

    def update(self, value: Any) -> bool:
        return self._set(value, increment_revision=True)

    def observe_source(self, source: np.ndarray | None) -> bool:
        """Bind the real model output and safely downgrade an invalid early request."""

        if source is None or source.ndim < 2:
            return False
        source_dimensions = (int(source.shape[1]), int(source.shape[0]))
        self.source_width, self.source_height = source_dimensions
        try:
            self._validate_dimensions(self.dimensions)
        except ValueError:
            self.width = None
            self.height = None
            self.revision += 1
            return True
        return False

    def snapshot(self) -> OutputResolutionSnapshot:
        return OutputResolutionSnapshot(
            dimensions=self.dimensions,
            source_dimensions=self.source_dimensions,
            revision=self.revision,
        )

    def to_payload(self) -> dict[str, Any]:
        return self.snapshot().to_payload()


def _entry_output_resolution(entry: CatalogEntry) -> tuple[int, int] | None:
    values = entry.default_call_kwargs
    try:
        width = int(values.get("width") or values.get("user_width") or 0)
        height = int(values.get("height") or values.get("user_height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    resolution = values.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        try:
            # Catalog resolution tuples follow the model convention (height, width).
            height, width = (int(resolution[0]), int(resolution[1]))
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height
    size = values.get("size")
    if isinstance(size, str):
        # World-model integrations commonly spell size as HEIGHT*WIDTH.
        match = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", size)
        if match:
            height, width = (int(match.group(1)), int(match.group(2)))
            if width > 0 and height > 0:
                return width, height
    return None


def _entry_output_pixel_budget(entry: CatalogEntry) -> int:
    native = _entry_output_resolution(entry)
    if native is not None:
        return min(native[0] * native[1], _MAX_OUTPUT_PIXELS)
    try:
        max_area = int(entry.default_call_kwargs.get("max_area") or 0)
    except (TypeError, ValueError):
        max_area = 0
    return min(max_area, _MAX_OUTPUT_PIXELS) if max_area > 0 else _MAX_OUTPUT_PIXELS


def _new_output_resolution_state(
    entry: CatalogEntry,
    value: Any,
    *,
    native_override: tuple[int, int] | None = None,
) -> OutputResolutionState:
    native = native_override or _entry_output_resolution(entry)
    return OutputResolutionState.from_value(
        value,
        maximum=native,
        max_pixels=(
            min(native[0] * native[1], _MAX_OUTPUT_PIXELS) if native is not None else _entry_output_pixel_budget(entry)
        ),
    )


def _output_resolution_options(
    entry: CatalogEntry,
    *,
    native_override: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    native = native_override or _entry_output_resolution(entry)
    options: list[dict[str, Any]] = [
        {
            "mode": "native",
            "label": (f"Native · {native[0]}×{native[1]}" if native is not None else "Native"),
            **({"width": native[0], "height": native[1]} if native is not None else {}),
        }
    ]
    candidates: list[tuple[int, int]] = []
    if native is not None:
        for scale in (0.75, 0.5):
            width = max(int(round(native[0] * scale / 2.0)) * 2, _MIN_OUTPUT_WIDTH)
            height = max(int(round(native[1] * scale / 2.0)) * 2, _MIN_OUTPUT_HEIGHT)
            candidates.append((width, height))
    # Without a model-owned native size, guessed presets can accidentally
    # upscale the first generated frame. Keep only native until the runtime
    # has observed a real source size.
    pixel_budget = (
        min(native[0] * native[1], _MAX_OUTPUT_PIXELS) if native is not None else _entry_output_pixel_budget(entry)
    )
    for width, height in dict.fromkeys(candidates):
        if not (_MIN_OUTPUT_WIDTH <= width <= _MAX_OUTPUT_WIDTH and _MIN_OUTPUT_HEIGHT <= height <= _MAX_OUTPUT_HEIGHT):
            continue
        if width * height > pixel_budget:
            continue
        if native is not None and (width > native[0] or height > native[1]):
            continue
        options.append(
            {
                "mode": "fixed",
                "width": width,
                "height": height,
                "label": f"Stream · {width}×{height}",
            }
        )
    return options


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), minimum)
    except Exception:
        return max(default, minimum)


def _env_optional_timeout_s(name: str) -> float | None:
    """Read a positive timeout in seconds; zero disables the deadline."""

    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number of seconds.") from exc
    if timeout_s < 0.0:
        raise ValueError(f"{name} must be non-negative, got {timeout_s}.")
    return timeout_s or None


def _realtime_shutdown_timeout_s() -> float:
    """Return the strict cooperative-shutdown budget."""

    name = "WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS"
    raw = os.getenv(name, "5").strip()
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number of seconds.") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of seconds, got {raw!r}.")
    return timeout_s


def _shutdown_deadline(deadline: float | None = None) -> float:
    if deadline is not None:
        return deadline
    return asyncio.get_running_loop().time() + _realtime_shutdown_timeout_s()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        error = task.exception()
    except BaseException:
        logger.warning(
            "Could not retrieve realtime teardown task result for %s.",
            task.get_name(),
            exc_info=True,
        )
        # Done callbacks must never re-raise into the event loop.
        return
    else:
        if error is not None:
            logger.warning(
                "Realtime teardown task %s failed: %s",
                task.get_name(),
                error,
                exc_info=(type(error), error, error.__traceback__),
            )


def _retain_shutdown_task(
    task: asyncio.Task[Any],
    retained: set[asyncio.Task[Any]],
) -> None:
    """Keep cancellation-resistant tasks alive until their result is consumed."""

    if task.done():
        _consume_task_result(task)
        return
    if task in retained:
        return
    retained.add(task)

    def discard(completed: asyncio.Task[Any]) -> None:
        retained.discard(completed)
        _consume_task_result(completed)

    task.add_done_callback(discard)


async def _finish_tasks_by_deadline(
    tasks: tuple[asyncio.Task[Any] | None, ...],
    *,
    deadline: float,
    retained: set[asyncio.Task[Any]],
    label: str,
    cancel_now: bool,
    warn_on_timeout: bool = True,
) -> bool:
    """Wait within the remaining budget, then cancel and retain stragglers."""

    current = asyncio.current_task()
    pending = {
        task
        for task in tasks
        if task is not None and task is not current and not task.done()
    }
    for task in tasks:
        if task is not None and task is not current and task.done():
            _consume_task_result(task)
    if cancel_now:
        for task in pending:
            task.cancel()
    if not pending:
        return True
    remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
    try:
        if remaining > 0.0:
            done, pending = await asyncio.wait(pending, timeout=remaining)
        else:
            done = set()
    except BaseException:
        for task in pending:
            task.cancel()
            _retain_shutdown_task(task, retained)
        raise
    for task in done:
        _consume_task_result(task)
    if not pending:
        return True
    for task in pending:
        task.cancel()
        _retain_shutdown_task(task, retained)
    if warn_on_timeout:
        logger.warning(
            "Realtime shutdown deadline expired while waiting for %s (%d task(s) still running).",
            label,
            len(pending),
        )
    return False


async def _run_by_shutdown_deadline(
    awaitable: Any,
    *,
    deadline: float,
    retained: set[asyncio.Task[Any]],
    label: str,
    warn_on_timeout: bool = True,
) -> bool:
    task = asyncio.create_task(awaitable, name=f"world-realtime-shutdown-{label}")
    return await _finish_tasks_by_deadline(
        (task,),
        deadline=deadline,
        retained=retained,
        label=label,
        cancel_now=False,
        warn_on_timeout=warn_on_timeout,
    )


async def _acquire_lock_by_shutdown_deadline(
    lock: asyncio.Lock,
    *,
    deadline: float,
    label: str,
) -> bool:
    """Acquire a cooperative asyncio lock without exceeding the budget."""

    remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
    if remaining <= 0.0:
        logger.warning("Realtime shutdown deadline expired before acquiring %s.", label)
        return False
    try:
        await asyncio.wait_for(lock.acquire(), timeout=remaining)
    except asyncio.TimeoutError:
        logger.warning("Realtime shutdown deadline expired while acquiring %s.", label)
        return False
    return True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value (1/0, true/false, yes/no, or on/off).")


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _realtime_rtx_device(launch_device: str) -> int:
    configured = _env_optional_int("WORLDFOUNDRY_REALTIME_RTX_DEVICE")
    if configured is not None:
        if configured < 0:
            raise ValueError("WORLDFOUNDRY_REALTIME_RTX_DEVICE must be non-negative.")
        return configured
    match = re.fullmatch(r"\s*cuda(?::(\d+))?\s*", launch_device, flags=re.IGNORECASE)
    return int(match.group(1) or 0) if match else 0


def _realtime_postprocess_stream(
    *,
    fps: int,
    launch_config: StudioLaunchConfig,
    native_resolution: tuple[int, int] | None = None,
) -> VideoPostprocessStream:
    """Resolve the explicitly selected realtime output processor."""

    preset = os.getenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "identity").strip()
    normalized = preset.lower().replace("_", "-")
    if normalized in {"", "none", "off", "identity"}:
        return VideoPostprocessStream(
            chain=VideoPostprocessChain((IdentityVideoPostProcessor(),)),
            fps=fps,
        )

    from worldfoundry.core.video.rtx import (
        require_rtx_vfx_runtime,
        rtx_postprocessor_from_preset,
    )

    device = _realtime_rtx_device(launch_config.device or "cuda")
    processor = rtx_postprocessor_from_preset(
        normalized,
        scale=_env_optional_float("WORLDFOUNDRY_REALTIME_RTX_SCALE"),
        output_width=_env_optional_int("WORLDFOUNDRY_REALTIME_RTX_OUTPUT_WIDTH"),
        output_height=_env_optional_int("WORLDFOUNDRY_REALTIME_RTX_OUTPUT_HEIGHT"),
        quality=(os.getenv("WORLDFOUNDRY_REALTIME_RTX_QUALITY", "").strip().upper() or None),
        device=device,
        non_blocking=_env_bool("WORLDFOUNDRY_REALTIME_RTX_NON_BLOCKING", False),
        use_current_stream=_env_bool("WORLDFOUNDRY_REALTIME_RTX_USE_CURRENT_STREAM", True),
    )
    input_spec = None
    if native_resolution is not None:
        input_spec = VideoSpec(
            height=native_resolution[1],
            width=native_resolution[0],
            fps=fps,
            channels=3,
            dtype="uint8",
        )
    capability = require_rtx_vfx_runtime(
        device=device,
        config=processor.config,
        input_spec=input_spec,
        run_inference=True,
    )
    logger.info(
        "Enabled realtime RTX postprocess preset=%s quality=%s device=cuda:%d gpu=%s nvidia_vfx=%s",
        normalized,
        processor.config.quality,
        device,
        capability.gpu_name or "unknown",
        capability.package_version or "unknown",
    )
    return VideoPostprocessStream(
        chain=VideoPostprocessChain((processor,)),
        fps=fps,
    )


def _frame_queue_policy(*, ordered_required: bool) -> FrameQueuePolicy:
    """Resolve congestion policy without weakening count-exact sessions."""

    if ordered_required:
        return FrameQueuePolicy.ORDERED_QUALITY
    configured = os.getenv(
        "WORLDFOUNDRY_REALTIME_FRAME_QUEUE_POLICY",
        FrameQueuePolicy.LATEST_INTERACTIVE.value,
    )
    return FrameQueuePolicy.from_value(configured)


def _realtime_frame_budget(entry: CatalogEntry, requested: int) -> int:
    """Resolve the bootstrap budget before a model reports its native spec."""

    if "queued-segment-generation" in entry.tags:
        # Queued diffusion segments are not latency-clamped realtime chunks.
        # Their native frame count is part of the model's quality/continuation
        # contract and must be known before the first result reports its spec.
        native_frames = entry.default_call_kwargs.get("num_frames")
        if native_frames is not None:
            try:
                return max(int(native_frames), 1)
            except (TypeError, ValueError):
                pass
    return max(int(requested), 1)


def _default_realtime_chunk_frames(entry: CatalogEntry) -> int:
    # DreamX uses 1 + 4k model windows. Five model frames are the smallest
    # native control interval and avoid silently inheriting the generic
    # nine-frame latency profile.
    return 5 if entry.model_id == _DREAMX_WORLD_MODEL_ID else 9


def _default_realtime_inference_steps(
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
) -> int | None:
    configured_steps = os.getenv("WORLDFOUNDRY_REALTIME_INFERENCE_STEPS", "").strip()
    if configured_steps:
        return _env_int("WORLDFOUNDRY_REALTIME_INFERENCE_STEPS", 4)
    if entry.model_id == _DREAMX_WORLD_MODEL_ID:
        return 4
    if entry.model_id == "lingbot-world" and launch_config.variant_id == "fast":
        return 4
    return None


def _ice_server_payload() -> list[dict[str, Any]]:
    """Parse browser/aiortc-compatible STUN/TURN configuration."""

    raw = os.getenv("WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON", "").strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON is not valid JSON.") from exc
    if isinstance(decoded, Mapping):
        decoded = decoded.get("iceServers")
    if not isinstance(decoded, list):
        raise RuntimeError("Realtime ICE config must be a JSON list or an {iceServers: [...]} object.")
    result: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, Mapping) or not item.get("urls"):
            raise RuntimeError("Each realtime ICE server must define a non-empty 'urls' value.")
        server = {"urls": item["urls"]}
        for key in ("username", "credential"):
            if item.get(key) is not None:
                server[key] = str(item[key])
        result.append(server)
    return result


def _validate_control_video_frames(path: str, *, expected_frames: int) -> int:
    """Decode enough control frames to reject a bad EXTEND transaction early."""

    from av import open as av_open

    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"Control video does not exist: {source}")
    try:
        with av_open(str(source)) as container:
            if not container.streams.video:
                raise ValueError(f"Control input has no video stream: {source.name}")
            count = 0
            for _frame in container.decode(video=0):
                count += 1
                if count >= expected_frames:
                    return count
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot decode control video {source.name}: {exc}") from exc
    raise ValueError(
        f"Control video {source.name} has {count} decoded frame(s); this segment needs at least {expected_frames}."
    )


def _realtime_overrides(
    entry: CatalogEntry,
    base: Mapping[str, Any],
    *,
    inference_steps: int | None = None,
) -> dict[str, Any]:
    """Clamp catalog demo settings to a latency-oriented chunk profile."""

    values = dict(base)
    frame_budget = _realtime_frame_budget(
        entry,
        _env_int(
            "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
            _default_realtime_chunk_frames(entry),
        ),
    )
    params = set(entry.call_params) | set(entry.stream_params) | set(values)

    if "num_frames" in params:
        values["num_frames"] = frame_budget
    if "video_length" in params:
        values["video_length"] = frame_budget
    if "frame_num" in params:
        values["frame_num"] = frame_budget
    if "num_chunks" in params:
        values["num_chunks"] = 1
    if "num_iterations" in params:
        values["num_iterations"] = 1
    if inference_steps is not None:
        for key in ("num_inference_steps", "inference_steps", "sampling_steps", "infer_steps"):
            if key in params:
                values[key] = inference_steps
    for key in ("visualize_ops", "visualize_warning", "show_progress"):
        if key in params:
            values[key] = False
    # A live session must always consume the DataChannel control state.  Several
    # catalog presets intentionally point at offline benchmark action files; if
    # those values leak into the resident runtime the UI looks responsive while
    # the model is actually replaying a fixed trajectory.
    if "action_path" in params:
        values["action_path"] = None
    if "official_bench_actions" in params:
        values["official_bench_actions"] = False
    values.pop("return_dict", None)

    return values


def _interactive_call_kwargs(
    base: Mapping[str, Any],
    interactions: list[str],
    *,
    seed: int,
    control_segments: list[ControlSegment] | None = None,
) -> dict[str, Any]:
    values = {**base, "seed": int(seed)}
    if control_segments is not None:
        values["realtime_segments"] = [
            {
                "duration": max(float(end) - float(start), 0.0),
                "keys": sorted(keys),
            }
            for start, end, keys in control_segments
        ]
    if "interaction_speed" in values:
        configured_speed = values["interaction_speed"]
        if isinstance(configured_speed, (list, tuple)) and configured_speed:
            speed = configured_speed[0]
        elif configured_speed is None:
            speed = 0.2
        else:
            speed = configured_speed
        values["interaction_speed"] = [speed] * len(interactions)
    return values


class ResidentWorldRuntime:
    """One resident model context executed on one stable worker thread."""

    def __init__(
        self,
        *,
        manager: StudioManager,
        entry: CatalogEntry,
        launch_config: StudioLaunchConfig,
        fps: int,
        warmup_image_path: str = "",
        warmup_chunks: int = 0,
        postprocess: VideoPostprocessStream | None = None,
    ) -> None:
        self.manager = manager
        self.entry = entry
        self.launch_config = launch_config
        self.fps = fps
        self.warmup_image_path = warmup_image_path
        self.warmup_chunks = max(int(warmup_chunks), 0)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world-realtime-runtime")
        self._base_request: PreparedInputs | None = None
        self._preload_future: asyncio.Future[Any] | None = None
        self._preload_task: asyncio.Task[Any] | None = None
        self._preload_error: str | None = None
        self._shutdown_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self.warmup_ms = 0.0
        self._configured = False
        self._first_stream_step = True
        self._seed_image: Image.Image | None = None
        self.last_generation_metrics: dict[str, Any] = {}
        self._postprocess = postprocess or VideoPostprocessStream(
            chain=VideoPostprocessChain((IdentityVideoPostProcessor(),)),
            fps=self.fps,
        )
        self.queued_segment_generation = "queued-segment-generation" in entry.tags
        bootstrap_frames = _realtime_frame_budget(
            entry,
            _env_int(
                "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
                _default_realtime_chunk_frames(entry),
            ),
        )
        if self.queued_segment_generation:
            self.realtime_spec = RealtimeSpec(
                fps=int(entry.default_call_kwargs.get("fps") or self.fps),
                first_chunk_frames=bootstrap_frames,
                steady_chunk_frames=bootstrap_frames,
                controls=("dense_depth_video", "sparse_pointmap_or_track_video"),
                transport="queued-segment-rgb",
                stateful=True,
            )
        else:
            self.realtime_spec = RealtimeSpec(
                fps=self.fps,
                first_chunk_frames=bootstrap_frames,
                steady_chunk_frames=bootstrap_frames,
            )

    def _accept_realtime_spec(self, result: Any) -> None:
        self.realtime_spec = RealtimeSpec.from_payload(
            result,
            fallback=self.realtime_spec,
        )
        # Playback, input resampling, and the model request must share the
        # cadence advertised by the resident adapter. Keeping the bootstrap
        # frontend FPS here made 12/17/24-FPS models play at a hard-coded 16.
        self.fps = self.realtime_spec.fps
        self._postprocess.fps = self.fps

    def _postprocess_frames(
        self,
        frames: list[np.ndarray],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        chunks = self._postprocess.process(
            frames,
            layout="frame-list",
            metadata=metadata,
        )
        return frame_list_from_chunks(chunks)

    async def _run(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)

    def _build_request(
        self,
        *,
        prompt: str,
        image: Image.Image | None,
        input_path: str,
        video_path: str,
        dense_video_path: str = "",
        sparse_video_path: str = "",
    ) -> PreparedInputs:
        from worldfoundry.studio.visualization.backends.world import (
            _api_key,
            _launch_runtime_overrides,
        )

        load_text, call_text, model_ref = _launch_runtime_overrides(
            self.entry,
            self.launch_config,
            interactions_text="",
        )
        call_values = json.loads(call_text or "{}")
        if self.queued_segment_generation:
            # Do not apply the low-latency frame/step clamps to full-quality
            # queued segments. Launch call kwargs remain the explicit tuning
            # surface for users who intentionally choose another profile.
            if dense_video_path:
                call_values["dense_video"] = dense_video_path
            else:
                call_values.pop("dense_video", None)
            if sparse_video_path:
                call_values["sparse_video"] = sparse_video_path
            else:
                call_values.pop("sparse_video", None)
            call_values["return_dict"] = True
        else:
            call_values = _realtime_overrides(
                self.entry,
                call_values,
                inference_steps=_default_realtime_inference_steps(
                    self.entry,
                    self.launch_config,
                ),
            )
        return self.manager.prepare_inputs(
            entry=self.entry,
            prompt=prompt,
            input_path=input_path,
            image=image,
            video=video_path or None,
            last_frame=None,
            reference_files=None,
            interactions_text="",
            camera_view_text="",
            task_type=self.entry.default_task_type or "",
            intrinsics_text="",
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=self.fps,
            num_frames=int(call_values.get("num_frames") or 0),
            call_kwargs_text=json.dumps(call_values),
            load_kwargs_text=load_text,
            model_ref=model_ref,
            backend=self.launch_config.backend or self.entry.default_backend or "auto",
            endpoint=self.launch_config.endpoint or self.entry.default_endpoint or "",
            api_key=_api_key(),
            device=self.launch_config.device or "cuda",
        )

    async def preload(self) -> None:
        if self._preload_future is not None:
            await asyncio.shield(self._preload_future)
            return
        loop = asyncio.get_running_loop()
        self._preload_future = loop.create_future()
        current_task = asyncio.current_task()
        self._preload_task = current_task
        try:
            request = await self._run(
                self._build_request,
                prompt=self.entry.default_prompt or "",
                image=None,
                input_path="",
                video_path="",
            )
            result = await self._run(
                self.manager.run_realtime,
                entry=self.entry,
                request=request,
                action="configure",
            )
            self._accept_realtime_spec(result)
            if self.warmup_chunks and self.warmup_image_path and self.entry.supports_stream:
                await self._warmup()
            self._preload_future.set_result(None)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                if not self._preload_future.done():
                    self._preload_future.cancel()
            else:
                self._preload_error = str(exc)
                if not self._preload_future.done():
                    self._preload_future.set_exception(exc)
                traceback.print_exc()
            raise
        finally:
            if self._preload_task is current_task:
                self._preload_task = None

    async def _warmup(self) -> None:
        """Compile and stabilize the actual resident stream before user input."""

        def open_seed() -> Image.Image:
            with Image.open(self.warmup_image_path) as source:
                return source.convert("RGB")

        started = time.perf_counter()
        image = await self._run(open_seed)
        request = await self._run(
            self._build_request,
            prompt=self.entry.default_prompt or "",
            image=image,
            input_path=self.warmup_image_path,
            video_path="",
        )
        interactions = ["forward"]
        configured = False

        async def configure_warmup() -> None:
            nonlocal configured
            result = await self._run(
                self.manager.run_realtime,
                entry=self.entry,
                request=request,
                action="configure",
            )
            self._accept_realtime_spec(result)
            configured = True

        async def warmup_step(index: int) -> None:
            stream_request = replace(
                request,
                image=request.image if index == 0 else None,
                image_path=request.image_path if index == 0 else None,
                interactions=interactions,
                call_kwargs=_interactive_call_kwargs(
                    request.call_kwargs,
                    interactions,
                    seed=41_000 + index,
                ),
            )
            result = await self._run(
                self.manager.run_realtime,
                entry=self.entry,
                request=stream_request,
                action="stream",
            )
            frames = await self._run(realtime_frames_from_result, result)
            await self._run(
                self._postprocess_frames,
                frames,
                metadata={"warmup": True, "warmup_index": index},
            )

        try:
            await run_async_prewarm_sequence(
                cold_start=configure_warmup,
                steady_state=warmup_step,
                steady_steps=self.warmup_chunks,
                label=f"studio.{self.entry.model_id}",
                timeout_s=_env_optional_timeout_s("WORLDFOUNDRY_REALTIME_PREWARM_TIMEOUT_SECONDS"),
            )
        finally:
            await self._run(self._postprocess.reset)
            if configured:
                await self._run(
                    self.manager.run_realtime,
                    entry=self.entry,
                    request=request,
                    action="reset",
                )
        self.warmup_ms = (time.perf_counter() - started) * 1000.0

    async def configure(
        self,
        *,
        prompt: str,
        image_path: str,
        video_path: str,
        dense_video_path: str = "",
        sparse_video_path: str = "",
    ) -> Image.Image:
        await self.preload()
        await self._run(self._postprocess.reset)
        if self.queued_segment_generation:
            missing = [
                label
                for value, label in (
                    (prompt.strip(), "prompt"),
                    (image_path, "initial image"),
                    (dense_video_path, "dense depth control video"),
                    (sparse_video_path, "sparse pointmap/track control video"),
                )
                if not value
            ]
            if missing:
                raise ValueError("Queued segment setup is missing: " + ", ".join(missing) + ".")
        image: Image.Image | None = None
        if image_path:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        input_path = image_path or video_path
        request = await self._run(
            self._build_request,
            prompt=prompt,
            image=image,
            input_path=input_path,
            video_path=video_path,
            dense_video_path=dense_video_path,
            sparse_video_path=sparse_video_path,
        )
        configured = await self._run(
            self.manager.run_realtime,
            entry=self.entry,
            request=request,
            action="configure",
        )
        self._accept_realtime_spec(configured)
        self._base_request = request
        self._configured = True
        self._first_stream_step = True
        self._seed_image = image or Image.new("RGB", (1280, 720), "black")
        return self._seed_image.copy()

    def next_chunk_frames(self, default: int) -> int:
        """Return the cadence advertised by the resident model adapter."""

        del default
        if self._first_stream_step:
            return self.realtime_spec.first_chunk_frames
        return self.realtime_spec.steady_chunk_frames

    def next_input_frames(self, default: int) -> int:
        """Return the model-owned control/input sampling chunk size."""

        del default
        if self._first_stream_step:
            return self.realtime_spec.resolved_first_input_frames
        return self.realtime_spec.resolved_steady_input_frames

    def steady_chunk_frames(self, default: int) -> int:
        del default
        return self.realtime_spec.steady_chunk_frames

    @property
    def supports_text_events(self) -> bool:
        """Whether this resident adapter can change text without resetting world state."""

        return bool(
            self.queued_segment_generation
            or "prompt_update" in self.realtime_spec.controls
            or self.entry.model_id in _PROMPT_BOUNDARY_MODEL_IDS
        )

    @property
    def postprocess_names(self) -> tuple[str, ...]:
        return self._postprocess.processor_names

    @property
    def transport_native_resolution(self) -> tuple[int, int] | None:
        """Expected postprocessed native size advertised to transport controls."""

        native = _entry_output_resolution(self.entry)
        if native is None:
            return None
        output = self._postprocess.chain.output_spec(
            VideoSpec(
                height=native[1],
                width=native[0],
                fps=self.fps,
                channels=3,
                dtype="uint8",
            )
        )
        return output.width, output.height

    async def generate(
        self,
        interactions: list[str],
        *,
        seed: int,
        control_segments: list[ControlSegment] | None = None,
        prompt: str | None = None,
        dense_video_path: str | None = None,
        sparse_video_path: str | None = None,
    ) -> tuple[list[np.ndarray], float]:
        if not self._configured or self._base_request is None:
            raise RuntimeError("Realtime runtime is not configured.")
        if self.queued_segment_generation:
            call_kwargs = dict(self._base_request.call_kwargs)
            if not self._first_stream_step:
                if not dense_video_path or not sparse_video_path:
                    raise ValueError(
                        "Every EXTEND request needs a new dense depth video and sparse pointmap/track video."
                    )
                call_kwargs["dense_video"] = dense_video_path
                call_kwargs["sparse_video"] = sparse_video_path
            request = replace(
                self._base_request,
                prompt=str(prompt).strip() if prompt is not None else self._base_request.prompt,
                interactions=[],
                call_kwargs=call_kwargs,
            )
            if not request.prompt:
                raise ValueError("Queued segment generation requires a non-empty prompt.")
            action = "run" if self._first_stream_step else "stream"
            if not self._first_stream_step:
                request = replace(
                    request,
                    input_path="",
                    image=None,
                    image_path=None,
                    video_path=None,
                )
        else:
            call_kwargs = _interactive_call_kwargs(
                self._base_request.call_kwargs,
                interactions,
                seed=seed,
                control_segments=control_segments,
            )
            request = replace(
                self._base_request,
                prompt=str(prompt).strip() if prompt is not None else self._base_request.prompt,
                interactions=list(interactions),
                call_kwargs=call_kwargs,
            )
            if not self._first_stream_step:
                request = replace(request, image=None, image_path=None)
            action = "stream"
        started = time.perf_counter()
        result = await self._run(
            self.manager.run_realtime,
            entry=self.entry,
            request=request,
            action=action,
        )
        runtime_finished = time.perf_counter()
        self._accept_realtime_spec(result)
        self._first_stream_step = False
        self._base_request = request
        self.last_generation_metrics = dict(result.get("realtime_metrics") or {}) if isinstance(result, Mapping) else {}
        copy_started = time.perf_counter()
        frames = await self._run(realtime_frames_from_result, result)
        copy_ms = (time.perf_counter() - copy_started) * 1000.0
        postprocess_started = time.perf_counter()
        frames = await self._run(
            self._postprocess_frames,
            frames,
            metadata={
                "model_id": self.entry.model_id,
                "seed": int(seed),
            },
        )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        self.last_generation_metrics.setdefault(
            "runtime_ms",
            (runtime_finished - started) * 1000.0,
        )
        self.last_generation_metrics.setdefault("copy_ms", copy_ms)
        self.last_generation_metrics["postprocess_ms"] = postprocess_ms
        generation_ms = (time.perf_counter() - started) * 1000.0
        return frames, generation_ms

    async def reset(self) -> None:
        await self._run(self._postprocess.reset)
        if self._base_request is not None:
            try:
                await self._run(
                    self.manager.run_realtime,
                    entry=self.entry,
                    request=self._base_request,
                    action="reset",
                )
            except Exception:
                logger.warning(
                    "Realtime runtime reset action failed; continuing session teardown.",
                    exc_info=True,
                )
        self._configured = False
        self._base_request = None
        self._seed_image = None
        self._first_stream_step = True

    async def close(self, *, deadline: float | None = None) -> None:
        if self._closed:
            return
        deadline = _shutdown_deadline(deadline)
        self._closed = True
        try:
            await _finish_tasks_by_deadline(
                (self._preload_task,),
                deadline=deadline,
                retained=self._shutdown_tasks,
                label="runtime preload",
                cancel_now=True,
            )
            await _run_by_shutdown_deadline(
                self.reset(),
                deadline=deadline,
                retained=self._shutdown_tasks,
                label="runtime reset",
            )
        finally:
            # Running Python/CUDA callables cannot be killed safely. Do not
            # join them here; reject queued work and let running work finish
            # cooperatively after the async shutdown handler returns.
            self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def ready(self) -> bool:
        return bool(
            self._preload_future
            and self._preload_future.done()
            and not self._preload_future.cancelled()
            and self._preload_future.exception() is None
        )

    @property
    def preload_error(self) -> str | None:
        return self._preload_error


@dataclass(slots=True)
class _ActivePeer:
    peer: Any
    channel: Any | None
    frames: LatestFrameBuffer | ChunkPresentationBuffer
    resampler: RealtimeControlResampler
    presentation_mode: RealtimePresentationMode = RealtimePresentationMode.HOLD_LAST
    generation_task: asyncio.Task[Any] | None = None
    liveness_task: asyncio.Task[Any] | None = None
    input_task: asyncio.Task[Any] | None = None
    setup_task: asyncio.Task[Any] | None = None
    close_task: asyncio.Task[Any] | None = None
    input_messages: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    closed: bool = False
    first_action: asyncio.Event = field(default_factory=asyncio.Event)
    action_arrivals: deque[float] = field(default_factory=deque)
    chunk_index: int = 0
    seed: int = 42
    last_client_message_at: float = 0.0
    prompt_scheduled: bool = False
    initial_segment_pending: bool = False
    pending_prompt: str | None = None
    pending_prompt_dirty: bool = False
    pending_steps: int = 0
    base_prompt: str = ""
    text_events: list[dict[str, str]] = field(default_factory=list)
    catalog_revision: int = 0
    active_event_id: str | None = None
    text_events_supported: bool = True
    output_resolution: OutputResolutionState = field(default_factory=OutputResolutionState)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    perf_window: RealtimeTimingWindow = field(default_factory=RealtimeTimingWindow)

    def __post_init__(self) -> None:
        if self.prompt_scheduled:
            self.frames.policy = FrameQueuePolicy.ORDERED_QUALITY


@dataclass(slots=True)
class _ActiveSocket:
    socket: Any
    resampler: RealtimeControlResampler
    frame_packets: asyncio.Queue[bytes]
    pacing_mode: WebSocketPacingMode = WebSocketPacingMode.LEGACY_DUAL
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation_task: asyncio.Task[Any] | None = None
    sender_task: asyncio.Task[Any] | None = None
    liveness_task: asyncio.Task[Any] | None = None
    closed: bool = False
    disconnect_requested: bool = False
    first_action: asyncio.Event = field(default_factory=asyncio.Event)
    action_arrivals: deque[float] = field(default_factory=deque)
    chunk_index: int = 0
    seed: int = 42
    dropped_frames: int = 0
    last_client_message_at: float = 0.0
    prompt_scheduled: bool = False
    initial_segment_pending: bool = False
    pending_prompt: str | None = None
    pending_prompt_dirty: bool = False
    pending_steps: int = 0
    base_prompt: str = ""
    text_events: list[dict[str, str]] = field(default_factory=list)
    catalog_revision: int = 0
    active_event_id: str | None = None
    text_events_supported: bool = True
    output_resolution: OutputResolutionState = field(default_factory=OutputResolutionState)
    frame_queue_policy: FrameQueuePolicy = FrameQueuePolicy.LATEST_INTERACTIVE
    queued_segments: bool = False
    pending_segment: dict[str, str] | None = None
    segment_inflight: bool = False
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    perf_window: RealtimeTimingWindow = field(default_factory=RealtimeTimingWindow)

    def __post_init__(self) -> None:
        if self.prompt_scheduled or self.queued_segments:
            self.frame_queue_policy = FrameQueuePolicy.ORDERED_QUALITY


class RealtimePeerManager:
    def __init__(
        self,
        *,
        runtime: ResidentWorldRuntime,
        fps: int,
        chunk_frames: int,
        ice_servers: list[dict[str, Any]] | None = None,
        presentation_mode: RealtimePresentationMode | str | None = None,
        socket_pacing_mode: WebSocketPacingMode | str | None = None,
    ) -> None:
        self.runtime = runtime
        self.fps = fps
        self.chunk_frames = chunk_frames
        self.ice_servers = list(ice_servers or [])
        configured_presentation_mode = (
            presentation_mode
            if presentation_mode is not None
            else os.environ.get(
                "WORLDFOUNDRY_REALTIME_PRESENTATION_MODE",
                RealtimePresentationMode.HOLD_LAST.value,
            )
        )
        self.presentation_mode = RealtimePresentationMode.from_value(configured_presentation_mode)
        configured_socket_pacing_mode = (
            socket_pacing_mode
            if socket_pacing_mode is not None
            else os.environ.get(
                "WORLDFOUNDRY_REALTIME_SOCKET_PACING_MODE",
                WebSocketPacingMode.LEGACY_DUAL.value,
            )
        )
        self.socket_pacing_mode = WebSocketPacingMode.from_value(configured_socket_pacing_mode)
        self._active: _ActivePeer | None = None
        self._active_socket: _ActiveSocket | None = None
        self._lock = asyncio.Lock()
        self._draining = False
        self._drain_done = asyncio.Event()
        self._drain_done.set()
        self._shutdown_tasks: set[asyncio.Task[Any]] = set()
        self._deferred_close_task: asyncio.Task[Any] | None = None
        self._perf_log_interval_chunks = _env_int(
            "WORLDFOUNDRY_REALTIME_PERF_LOG_INTERVAL_CHUNKS",
            5,
            minimum=0,
        )
        self._perf_warmup_chunks = _env_int(
            "WORLDFOUNDRY_REALTIME_PERF_WARMUP_CHUNKS",
            0,
            minimum=0,
        )
        self._perf_jsonl_path = os.environ.get("WORLDFOUNDRY_REALTIME_PERF_JSONL", "").strip() or None

    @staticmethod
    def _positive_int_runtime_value(value: Any, *, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if parsed <= 0:
            raise ValueError(f"{label} must be > 0.")
        return parsed

    @staticmethod
    def _positive_float_runtime_value(value: Any, *, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if parsed <= 0.0:
            raise ValueError(f"{label} must be > 0.")
        return parsed

    def _runtime_input_fps(self) -> float:
        spec = getattr(self.runtime, "realtime_spec", None)
        value = getattr(spec, "resolved_input_fps", self.fps)
        return self._positive_float_runtime_value(
            value,
            label="realtime input_fps",
        )

    def _runtime_next_input_frames(self) -> int:
        method = getattr(self.runtime, "next_input_frames", None)
        if callable(method):
            return self._positive_int_runtime_value(
                method(self.chunk_frames),
                label="next_input_frames",
            )
        return self._positive_int_runtime_value(
            self.runtime.next_chunk_frames(self.chunk_frames),
            label="next_chunk_frames",
        )

    def _record_perf(
        self,
        active: _ActivePeer | _ActiveSocket,
        *,
        transport: str,
        generation_started: float,
        now: float,
        output_frames: int,
        generation_ms: float,
        pixel_post_ms: float,
        enqueue_ms: float,
        queue_depth: int,
        dropped_frames: int,
        control_latency_ms: float | None,
    ) -> None:
        window = active.perf_window
        metrics = getattr(self.runtime, "last_generation_metrics", {}) or {}
        presentation_mode = getattr(active, "presentation_mode", None)
        socket_pacing_mode = getattr(active, "pacing_mode", None)
        stage_ms = {
            "generation_ms": generation_ms,
            "pixel_post_ms": pixel_post_ms,
            "enqueue_ms": enqueue_ms,
            "server_chunk_ms": max((now - generation_started) * 1000.0, 0.0),
        }
        if control_latency_ms is not None:
            stage_ms["control_latency_ms"] = control_latency_ms
        gauges: dict[str, float] = {}
        for name in _CHUNK_METRIC_NAMES:
            if name not in metrics:
                continue
            try:
                value = float(metrics[name])
            except (TypeError, ValueError):
                continue
            if name.endswith("_ms"):
                stage_ms[name] = value
            else:
                gauges[name] = value
        timing = RealtimeChunkTiming(
            session_id=active.session_id,
            chunk_index=active.chunk_index,
            transport=transport,
            started_at_s=generation_started,
            completed_at_s=now,
            output_frames=output_frames,
            queue_depth=queue_depth,
            dropped_frames=dropped_frames,
            warmup=active.chunk_index <= self._perf_warmup_chunks,
            stage_ms=stage_ms,
            gauges=gauges,
        )
        window.record(timing)
        model_id = getattr(getattr(self.runtime, "entry", None), "model_id", None)
        if self._perf_jsonl_path is not None:
            try:
                write_jsonl_event(
                    self._perf_jsonl_path,
                    level="INFO",
                    event="realtime.chunk_timing",
                    message="Realtime chunk timing",
                    logger_name=__name__,
                    model_id=model_id,
                    postprocess=list(getattr(self.runtime, "postprocess_names", ())),
                    presentation_mode=(
                        presentation_mode.value if isinstance(presentation_mode, RealtimePresentationMode) else None
                    ),
                    socket_pacing_mode=(
                        socket_pacing_mode.value if isinstance(socket_pacing_mode, WebSocketPacingMode) else None
                    ),
                    **timing.to_payload(),
                )
            except (OSError, TypeError, ValueError):
                logger.warning(
                    "Disabling realtime timing JSONL after a write failure: %s",
                    self._perf_jsonl_path,
                    exc_info=True,
                )
                self._perf_jsonl_path = None
        interval = self._perf_log_interval_chunks
        if active.chunk_index != 1 and (interval <= 0 or (active.chunk_index - 1) % interval != 0):
            if interval <= 0:
                window.reset_interval()
            return

        summary = window.summary()
        generation_fps = output_frames / max(generation_ms / 1000.0, 1.0e-6)
        interval_fps = 0.0 if summary is None else summary.throughput_fps
        dropped_delta = 0 if summary is None else summary.dropped_frames
        measured_chunks = 0 if summary is None else summary.chunk_count
        measured_frames = 0 if summary is None else summary.frame_count

        def stat_text(stage_name: str, field_name: str) -> str:
            if summary is None:
                return "-"
            distribution = summary.stages.get(stage_name)
            if distribution is None:
                return "-"
            return f"{getattr(distribution, field_name):.1f}"

        latency_text = "-" if control_latency_ms is None else f"{control_latency_ms:.1f}"
        message = (
            "Realtime perf transport=%s chunk=%d interval_chunks=%d measured_chunks=%d frames=%d "
            "gen_fps=%.1f interval_fps=%.1f playback_fps=%s generation_ms=%.1f "
            "generation_p50_ms=%s generation_p90_ms=%s model_p50_ms=%s model_p90_ms=%s "
            "pixel_post_ms=%.1f enqueue_ms=%.1f runtime_ms=%.1f condition_ms=%.1f "
            "model_ms=%.1f decode_ms=%.1f postprocess_ms=%.1f copy_ms=%.1f cache_ms=%.1f queue_depth=%d "
            "dropped_delta=%d control_latency_ms=%s cache_frames=%d cache_tokens=%d "
            "compile_active=%d cuda_graph=%d"
        ) % (
            transport,
            active.chunk_index,
            window.observed_chunks,
            measured_chunks,
            measured_frames,
            generation_fps,
            interval_fps,
            self.fps,
            generation_ms,
            stat_text("generation_ms", "p50_ms"),
            stat_text("generation_ms", "p90_ms"),
            stat_text("model_ms", "p50_ms"),
            stat_text("model_ms", "p90_ms"),
            pixel_post_ms,
            enqueue_ms,
            _metric_float(metrics, "runtime_ms", generation_ms),
            _metric_float(metrics, "condition_ms"),
            _metric_float(metrics, "model_ms"),
            _metric_float(metrics, "decode_ms"),
            _metric_float(metrics, "postprocess_ms"),
            _metric_float(metrics, "copy_ms"),
            _metric_float(metrics, "cache_ms"),
            queue_depth,
            dropped_delta,
            latency_text,
            int(round(_metric_float(metrics, "cache_frames"))),
            int(round(_metric_float(metrics, "cache_tokens"))),
            int(round(_metric_float(metrics, "compile_active"))),
            int(round(_metric_float(metrics, "cuda_graph"))),
        )
        event_logger.event(
            "INFO",
            "realtime.performance.summary",
            message,
            transport=transport,
            model_id=model_id,
            session_id=active.session_id,
            chunk_index=active.chunk_index,
            warmup_chunks=self._perf_warmup_chunks,
            summary=None if summary is None else summary.to_payload(),
        )
        window.reset_interval()

    @staticmethod
    def _request_id(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("request_id")
        return str(value)[:128] if value is not None else None

    @staticmethod
    def _event_for(active: _ActivePeer | _ActiveSocket, event_id: str) -> dict[str, str] | None:
        return next(
            (event for event in active.text_events if event["event_id"] == event_id),
            None,
        )

    @staticmethod
    def _schedule_prompt(
        active: _ActivePeer | _ActiveSocket,
        prompt: str,
        *,
        generate: bool,
    ) -> None:
        already_scheduled = active.pending_prompt_dirty
        active.pending_prompt = prompt
        active.pending_prompt_dirty = True
        if not generate or already_scheduled:
            return
        arrival = asyncio.get_running_loop().time()
        active.action_arrivals.append(arrival)
        active.first_action.set()

    def _handle_session_control(
        self,
        active: _ActivePeer | _ActiveSocket,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Apply an ordered control message and return its acknowledgement."""

        message_type = str(payload.get("type") or "").strip().lower()
        request_id = self._request_id(payload)
        if message_type == "event_catalog":
            if not active.text_events_supported:
                return {
                    "type": "event_catalog_ack",
                    "ok": False,
                    "request_id": request_id,
                    "catalog_revision": active.catalog_revision,
                    "message": "This model has no state-preserving realtime text update path.",
                }
            base_revision = payload.get("base_revision")
            if base_revision is not None:
                try:
                    revision_matches = int(base_revision) == active.catalog_revision
                except (TypeError, ValueError):
                    revision_matches = False
                if not revision_matches:
                    return {
                        "type": "event_catalog_ack",
                        "ok": False,
                        "request_id": request_id,
                        "catalog_revision": active.catalog_revision,
                        "event_catalog": active.text_events,
                        "active_event_id": active.active_event_id,
                        "message": "The event catalog changed; refresh before applying this edit.",
                    }
            try:
                events = normalize_text_events(payload.get("events"))
            except ValueError as exc:
                return {
                    "type": "event_catalog_ack",
                    "ok": False,
                    "request_id": request_id,
                    "catalog_revision": active.catalog_revision,
                    "message": str(exc),
                }
            previous = self._event_for(active, active.active_event_id or "")
            active.text_events = events
            active.catalog_revision += 1
            current = self._event_for(active, active.active_event_id or "")
            if active.active_event_id and current is None:
                active.active_event_id = None
                self._schedule_prompt(
                    active,
                    active.base_prompt,
                    generate=not getattr(active, "queued_segments", False),
                )
            elif current is not None and previous is not None and current["prompt"] != previous["prompt"]:
                self._schedule_prompt(
                    active,
                    current["prompt"],
                    generate=not getattr(active, "queued_segments", False),
                )
            return {
                "type": "event_catalog_ack",
                "ok": True,
                "request_id": request_id,
                "catalog_revision": active.catalog_revision,
                "event_catalog": active.text_events,
                "active_event_id": active.active_event_id,
                "status": "applied",
            }
        if message_type == "event":
            if not active.text_events_supported:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": str(payload.get("event_id") or "") or None,
                    "state": str(payload.get("state") or "trigger"),
                    "active_event_id": active.active_event_id,
                    "message": "This model has no state-preserving realtime text update path.",
                }
            state = str(payload.get("state") or "trigger").strip().lower()
            if state in {"release", "off", "none"}:
                state = "clear"
            event_id = str(payload.get("event_id") or payload.get("id") or "").strip()
            if state not in {"trigger", "clear"}:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": event_id or None,
                    "state": state,
                    "active_event_id": active.active_event_id,
                    "message": "event state must be 'trigger' or 'clear'.",
                }
            event = self._event_for(active, event_id) if state == "trigger" else None
            if state == "trigger" and event is None:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": event_id or None,
                    "state": state,
                    "active_event_id": active.active_event_id,
                    "message": f"Unknown text event: {event_id or '<empty>'}.",
                }
            active.active_event_id = event_id if event is not None else None
            prompt = event["prompt"] if event is not None else active.base_prompt
            queued = bool(getattr(active, "queued_segments", False))
            self._schedule_prompt(active, prompt, generate=not queued)
            return {
                "type": "event_ack",
                "ok": True,
                "request_id": request_id,
                "event_id": event_id or None,
                "state": state,
                "active_event_id": active.active_event_id,
                "catalog_revision": active.catalog_revision,
                "applies_at": "next_segment" if queued else "next_chunk",
                "status": "accepted",
            }
        if message_type == "output_config":
            try:
                changed = active.output_resolution.update(payload.get("resolution"))
            except ValueError as exc:
                return {
                    "type": "output_config_ack",
                    "ok": False,
                    "request_id": request_id,
                    "resolution": active.output_resolution.to_payload(),
                    "resolution_revision": active.output_resolution.revision,
                    "message": str(exc),
                }
            return {
                "type": "output_config_ack",
                "ok": True,
                "request_id": request_id,
                "resolution": active.output_resolution.to_payload(),
                "resolution_revision": active.output_resolution.revision,
                "status": "queued" if changed else "unchanged",
                "applies_at": "next_chunk",
            }
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        if message_type == "step" or (message_type == "action" and str(action.get("event") or "").lower() == "step"):
            if bool(getattr(active, "queued_segments", False)):
                return {
                    "type": "step_ack",
                    "ok": False,
                    "request_id": request_id,
                    "message": "Queued segment models require their next control videos.",
                }
            active.pending_steps += 1
            arrival = asyncio.get_running_loop().time()
            active.action_arrivals.append(arrival)
            active.first_action.set()
            return {
                "type": "step_ack",
                "ok": True,
                "request_id": request_id,
                "queued_steps": active.pending_steps,
                "status": "accepted",
            }
        return None

    @property
    def active(self) -> bool:
        return bool(
            self._draining
            or (self._active and not self._active.closed)
            or (self._active_socket and not self._active_socket.closed)
        )

    @property
    def draining(self) -> bool:
        return self._draining

    def _detach_active_locked(
        self,
    ) -> tuple[_ActivePeer | None, _ActiveSocket | None, bool, bool]:
        active = self._active
        self._active = None
        active_socket = self._active_socket
        self._active_socket = None
        has_session = not ((active is None or active.closed) and (active_socket is None or active_socket.closed))
        if has_session:
            self._draining = True
            self._drain_done.clear()
        wait_for_drain = self._draining and not has_session
        return active, active_socket, has_session, wait_for_drain

    def _schedule_deferred_close(self) -> None:
        existing = self._deferred_close_task
        if existing is not None and not existing.done():
            return
        timeout_s = _realtime_shutdown_timeout_s()
        task = asyncio.create_task(
            self._deferred_close_after_lock(timeout_s),
            name="world-realtime-deferred-close",
        )
        self._deferred_close_task = task

        def clear_deferred(completed: asyncio.Task[Any]) -> None:
            if self._deferred_close_task is completed:
                self._deferred_close_task = None

        task.add_done_callback(clear_deferred)
        _retain_shutdown_task(task, self._shutdown_tasks)

    async def _deferred_close_after_lock(self, timeout_s: float) -> None:
        await self._lock.acquire()
        try:
            detached = self._detach_active_locked()
        finally:
            self._lock.release()
        deadline = asyncio.get_running_loop().time() + timeout_s
        await self._finish_active_close(*detached, deadline=deadline)

    async def _finish_active_close(
        self,
        active: _ActivePeer | None,
        active_socket: _ActiveSocket | None,
        has_session: bool,
        wait_for_drain: bool,
        *,
        deadline: float,
    ) -> None:
        if wait_for_drain:
            await _run_by_shutdown_deadline(
                self._drain_done.wait(),
                deadline=deadline,
                retained=self._shutdown_tasks,
                label="concurrent session drain",
            )
            return
        if not has_session:
            return
        try:
            if active is not None and not active.closed:
                active.closed = True
                tasks = (
                    active.generation_task,
                    active.liveness_task,
                    active.input_task,
                    active.setup_task,
                )
                await _finish_tasks_by_deadline(
                    tasks,
                    deadline=deadline,
                    retained=self._shutdown_tasks,
                    label="WebRTC workers",
                    cancel_now=True,
                )
                active.frames.close()
                await _run_by_shutdown_deadline(
                    active.peer.close(),
                    deadline=deadline,
                    retained=self._shutdown_tasks,
                    label="WebRTC peer close",
                )
            if active_socket is not None and not active_socket.closed:
                active_socket.closed = True
                tasks = (
                    active_socket.generation_task,
                    active_socket.sender_task,
                    active_socket.liveness_task,
                )
                await _finish_tasks_by_deadline(
                    tasks,
                    deadline=deadline,
                    retained=self._shutdown_tasks,
                    label="WebSocket workers",
                    cancel_now=True,
                )
                if not active_socket.socket.closed:
                    socket_close_deadline = deadline
                    if active_socket.disconnect_requested:
                        # The application-level disconnect already states the
                        # client's intent. Give the RFC 6455 close handshake a
                        # short chance to complete, but do not let a client
                        # that stops reading consume the runtime-reset budget.
                        socket_close_deadline = min(
                            deadline,
                            asyncio.get_running_loop().time()
                            + _EXPLICIT_DISCONNECT_CLOSE_GRACE_SECONDS,
                        )
                    close_completed = await _run_by_shutdown_deadline(
                        active_socket.socket.close(),
                        deadline=socket_close_deadline,
                        retained=self._shutdown_tasks,
                        label="WebSocket close",
                        warn_on_timeout=not active_socket.disconnect_requested,
                    )
                    if active_socket.disconnect_requested and not close_completed:
                        logger.info(
                            "Explicit WebSocket disconnect close handshake did not complete "
                            "within its bounded grace period; continuing session teardown."
                        )
        finally:
            try:
                await _run_by_shutdown_deadline(
                    self.runtime.reset(),
                    deadline=deadline,
                    retained=self._shutdown_tasks,
                    label="session runtime reset",
                )
            finally:
                # The leader detached the session while holding _lock. These
                # two non-awaiting assignments are atomic on the event loop;
                # reacquiring the lock here could itself exceed the deadline.
                self._draining = False
                self._drain_done.set()

    async def close_active(self, *, deadline: float | None = None) -> None:
        deadline = _shutdown_deadline(deadline)
        acquired = await _acquire_lock_by_shutdown_deadline(
            self._lock,
            deadline=deadline,
            label="realtime session lock",
        )
        if not acquired:
            # The caller's absolute deadline still wins, but preserve the
            # close intent. A single retained task will detach and tear down
            # the session when an in-flight setup operation releases _lock.
            self._schedule_deferred_close()
            return
        try:
            detached = self._detach_active_locked()
        finally:
            self._lock.release()
        await self._finish_active_close(*detached, deadline=deadline)

    async def create_answer(self, *, offer: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, str]:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        if self.runtime.queued_segment_generation:
            raise RuntimeError(
                "Queued segment generation uses the same-origin WebSocket transport; "
                "WebRTC/ICE is intentionally bypassed."
            )
        async with self._lock:
            if self.active:
                raise RuntimeError("A realtime world session is already active.")

            base_prompt = str(session.get("prompt") or self.runtime.entry.default_prompt or "")
            text_events = normalize_text_events(session.get("text_events", session.get("event_catalog")))
            output_resolution = _new_output_resolution_state(
                self.runtime.entry,
                session.get("output_resolution"),
                native_override=self.runtime.transport_native_resolution,
            )
            await self.runtime.configure(
                prompt=base_prompt,
                image_path=str(session.get("init_image_path") or ""),
                video_path=str(session.get("init_video_path") or ""),
                dense_video_path=str(session.get("dense_video_path") or ""),
                sparse_video_path=str(session.get("sparse_video_path") or ""),
            )
            self.fps = self.runtime.realtime_spec.fps
            queue_policy = _frame_queue_policy(
                ordered_required="prompt_update" in self.runtime.realtime_spec.controls,
            )
            requested_presentation_mode = session.get("presentation_mode")
            presentation_mode = (
                self.presentation_mode
                if requested_presentation_mode in (None, "")
                else RealtimePresentationMode.from_value(requested_presentation_mode)
            )
            if presentation_mode is RealtimePresentationMode.REAL_FRAMES:
                frames: LatestFrameBuffer | ChunkPresentationBuffer = ChunkPresentationBuffer(
                    fps=self.fps,
                    maximum_fps=_env_int(
                        "WORLDFOUNDRY_REALTIME_PRESENTATION_MAX_FPS",
                        self.fps * 2,
                        minimum=self.fps,
                    ),
                    mailbox_size=2,
                    policy=queue_policy,
                )
            else:
                frames = LatestFrameBuffer(
                    maxsize=_env_int(
                        "WORLDFOUNDRY_REALTIME_FRAME_QUEUE",
                        max(self.runtime.steady_chunk_frames(self.chunk_frames), 8),
                    ),
                    backpressure_ms=_env_int(
                        "WORLDFOUNDRY_REALTIME_BACKPRESSURE_MS",
                        max(
                            int(self.runtime.steady_chunk_frames(self.chunk_frames) * 500 / self.fps),
                            50,
                        ),
                    ),
                    policy=queue_policy,
                )
            rtc_configuration = RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=server["urls"],
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                    for server in self.ice_servers
                ]
            )
            pc = RTCPeerConnection(rtc_configuration)
            track = _build_video_track(
                frames=frames,
                fps=self.fps,
                presentation_mode=presentation_mode,
            )
            pc.addTrack(track)
            loop = asyncio.get_running_loop()
            active = _ActivePeer(
                peer=pc,
                channel=None,
                frames=frames,
                resampler=RealtimeControlResampler(
                    fps=self._runtime_input_fps(),
                    start_time=loop.time(),
                ),
                presentation_mode=presentation_mode,
                last_client_message_at=loop.time(),
                prompt_scheduled="prompt_update" in self.runtime.realtime_spec.controls,
                initial_segment_pending=("prompt_update" in self.runtime.realtime_spec.controls),
                base_prompt=base_prompt,
                text_events=text_events,
                catalog_revision=1 if text_events else 0,
                text_events_supported=self.runtime.supports_text_events,
                output_resolution=output_resolution,
            )
            self._active = active

            @pc.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                active.channel = channel
                channel_open_time = asyncio.get_running_loop().time()
                active.last_client_message_at = channel_open_time
                active.resampler.reset(start_time=channel_open_time)

                @channel.on("message")
                def on_message(raw: Any) -> None:
                    active.input_messages.put_nowait(raw)

                @channel.on("close")
                def on_close() -> None:
                    active.close_task = asyncio.create_task(self.close_active())
                    active.close_task.add_done_callback(_consume_task_result)

                active.generation_task = asyncio.create_task(self._generation_worker(active))
                active.liveness_task = asyncio.create_task(self._liveness_watchdog(active))
                active.input_task = asyncio.create_task(self._input_worker(active))
                if active.prompt_scheduled:
                    # Start the first native AR segment immediately. Subsequent
                    # segments are triggered by explicit prompt updates.
                    active.first_action.set()
                self._send(
                    channel,
                    {
                        "type": "ready",
                        "fps": self.fps,
                        "event_catalog": active.text_events,
                        "catalog_revision": active.catalog_revision,
                        "active_event_id": active.active_event_id,
                        "frame_queue_policy": active.frames.policy.value,
                        "presentation_mode": active.presentation_mode.value,
                        "resolution": active.output_resolution.to_payload(),
                        "resolution_revision": active.output_resolution.revision,
                    },
                )

            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if pc.connectionState in {"failed", "closed", "disconnected"}:
                    await self.close_active()

            try:
                await pc.setRemoteDescription(RTCSessionDescription(sdp=str(offer["sdp"]), type=str(offer["type"])))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await _wait_for_ice_gathering(pc)
                # A peer that completes signaling but never opens its
                # DataChannel would otherwise hold the single session forever:
                # the liveness watchdog only starts on channel open and the
                # connection state can stay "connected".
                active.setup_task = asyncio.create_task(
                    self._peer_setup_watchdog(active),
                    name="world-realtime-peer-setup-watchdog",
                )
                return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            except Exception:
                active.closed = True
                self._active = None
                active.frames.close()
                await pc.close()
                await self.runtime.reset()
                raise

    async def serve_socket(self, socket: Any) -> None:
        """Run a same-origin fallback session when UDP ICE cannot cross a tunnel."""

        from aiohttp import WSMsgType

        first = await asyncio.wait_for(
            socket.receive(),
            timeout=_env_int("WORLDFOUNDRY_REALTIME_SOCKET_SETUP_TIMEOUT_SECONDS", 15),
        )
        if first.type != WSMsgType.TEXT:
            raise RuntimeError("Expected a WebSocket configure message.")
        try:
            payload = json.loads(first.data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("WebSocket configure message is not valid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("WebSocket configure message must be a JSON object.")
        if str(payload.get("type") or "").lower() != "configure":
            raise RuntimeError("Expected WebSocket message type='configure'.")
        session = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}
        base_prompt = str(session.get("prompt") or self.runtime.entry.default_prompt or "")
        text_events = normalize_text_events(session.get("text_events", session.get("event_catalog")))
        output_resolution = _new_output_resolution_state(
            self.runtime.entry,
            session.get("output_resolution"),
            native_override=self.runtime.transport_native_resolution,
        )
        await socket.send_str(
            json.dumps(
                {"type": "configuring", "transport": "ws"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        async with self._lock:
            if self.active:
                raise RuntimeError("A realtime world session is already active.")
            await self.runtime.configure(
                prompt=base_prompt,
                image_path=str(session.get("init_image_path") or ""),
                video_path=str(session.get("init_video_path") or ""),
                dense_video_path=str(session.get("dense_video_path") or ""),
                sparse_video_path=str(session.get("sparse_video_path") or ""),
            )
            self.fps = self.runtime.realtime_spec.fps
            queue_policy = _frame_queue_policy(
                ordered_required=(
                    self.runtime.queued_segment_generation or "prompt_update" in self.runtime.realtime_spec.controls
                ),
            )
            requested_socket_pacing_mode = session.get("socket_pacing_mode")
            socket_pacing_mode = (
                self.socket_pacing_mode
                if requested_socket_pacing_mode in (None, "")
                else WebSocketPacingMode.from_value(requested_socket_pacing_mode)
            )
            loop = asyncio.get_running_loop()
            active = _ActiveSocket(
                socket=socket,
                resampler=RealtimeControlResampler(
                    fps=self._runtime_input_fps(),
                    start_time=loop.time(),
                ),
                frame_packets=asyncio.Queue(
                    maxsize=_env_int(
                        "WORLDFOUNDRY_REALTIME_SOCKET_FRAME_QUEUE",
                        max(self.runtime.steady_chunk_frames(self.chunk_frames) * 2, 16),
                    )
                ),
                pacing_mode=socket_pacing_mode,
                last_client_message_at=loop.time(),
                prompt_scheduled="prompt_update" in self.runtime.realtime_spec.controls,
                initial_segment_pending=("prompt_update" in self.runtime.realtime_spec.controls),
                base_prompt=base_prompt,
                text_events=text_events,
                catalog_revision=1 if text_events else 0,
                text_events_supported=self.runtime.supports_text_events,
                output_resolution=output_resolution,
                frame_queue_policy=queue_policy,
                queued_segments=self.runtime.queued_segment_generation,
            )
            self._active_socket = active

        active.generation_task = asyncio.create_task(
            self._socket_generation_worker(active),
            name="world-realtime-socket-generation",
        )
        active.sender_task = asyncio.create_task(
            self._socket_sender(active),
            name="world-realtime-socket-sender",
        )
        # aiohttp's protocol heartbeat detects dead sockets without depending
        # on JavaScript timers, which browsers throttle aggressively in
        # background tabs during long segment inference.
        active.liveness_task = None
        await self._socket_send(
            active,
            {
                "type": "ready",
                "fps": self.fps,
                "transport": "ws",
                "event_catalog": active.text_events,
                "catalog_revision": active.catalog_revision,
                "active_event_id": active.active_event_id,
                "frame_queue_policy": active.frame_queue_policy.value,
                "socket_pacing_mode": active.pacing_mode.value,
                "resolution": active.output_resolution.to_payload(),
                "resolution_revision": active.output_resolution.revision,
            },
        )
        # Publish readiness before scheduling the first segment so the browser
        # cannot have a GENERATING status overwritten by a late ready event.
        if active.prompt_scheduled or active.queued_segments:
            active.first_action.set()

        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    await self._handle_socket_message(active, message.data)
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
        finally:
            if self._active_socket is active:
                await self.close_active()

    async def _handle_socket_message(self, active: _ActiveSocket, raw: str) -> None:
        if active.closed:
            return
        active.last_client_message_at = asyncio.get_running_loop().time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        message_type = str(payload.get("type") or "").lower()
        if message_type == "disconnect":
            active.disconnect_requested = True
            await self.close_active()
            return
        if message_type in {"heartbeat", "ping"}:
            return
        acknowledgement = self._handle_session_control(active, payload)
        if acknowledgement is not None:
            await self._socket_send(active, acknowledgement)
            return
        if message_type == "prompt_update" and active.prompt_scheduled:
            prompt = str(payload.get("prompt") or "").strip()
            if prompt:
                arrival = asyncio.get_running_loop().time()
                active.base_prompt = prompt
                active.active_event_id = None
                active.pending_prompt = prompt
                active.pending_prompt_dirty = True
                active.action_arrivals.append(arrival)
                active.first_action.set()
            return
        if message_type == "segment_update" and active.queued_segments:
            prompt = str(payload.get("prompt") or "").strip()
            selected_event = self._event_for(active, active.active_event_id or "")
            if selected_event is not None:
                prompt = selected_event["prompt"]
            dense_video_path = str(payload.get("dense_video_path") or "").strip()
            sparse_video_path = str(payload.get("sparse_video_path") or "").strip()
            if not prompt or not dense_video_path or not sparse_video_path:
                await self._socket_send(
                    active,
                    {
                        "type": "segment_rejected",
                        "message": "EXTEND requires a prompt plus new dense and sparse control videos.",
                    },
                )
                return
            if active.segment_inflight or active.first_action.is_set() or active.pending_segment:
                await self._socket_send(
                    active,
                    {
                        "type": "segment_rejected",
                        "message": "A segment is already generating or queued.",
                    },
                )
                return
            expected_frames = self.runtime.realtime_spec.steady_chunk_frames
            try:
                await asyncio.gather(
                    asyncio.to_thread(
                        _validate_control_video_frames,
                        dense_video_path,
                        expected_frames=expected_frames,
                    ),
                    asyncio.to_thread(
                        _validate_control_video_frames,
                        sparse_video_path,
                        expected_frames=expected_frames,
                    ),
                )
            except ValueError as exc:
                await self._socket_send(
                    active,
                    {"type": "segment_rejected", "message": str(exc)},
                )
                return
            arrival = asyncio.get_running_loop().time()
            active.pending_segment = {
                "prompt": prompt,
                "dense_video_path": dense_video_path,
                "sparse_video_path": sparse_video_path,
            }
            active.action_arrivals.append(arrival)
            active.first_action.set()
            await self._socket_send(
                active,
                {"type": "segment_queued", "segment_index": active.chunk_index + 1},
            )
            return
        if message_type != "action":
            return
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        event = str(action.get("event") or "").lower()
        key = str(action.get("key") or "")
        arrival = asyncio.get_running_loop().time()
        if active.resampler.on_edge(arrival_time=arrival, event=event, key=key):
            active.action_arrivals.append(arrival)
            active.first_action.set()

    async def _socket_generation_worker(self, active: _ActiveSocket) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not active.closed:
                await active.first_action.wait()
                await asyncio.sleep(0)
                next_prompt = None
                prompt_changed = False
                prompt_event_id: str | None = None
                forced_step = False
                segments = None
                dense_video_path = None
                sparse_video_path = None
                if active.queued_segments:
                    active.first_action.clear()
                    active.segment_inflight = True
                    pending_segment = active.pending_segment
                    active.pending_segment = None
                    if pending_segment is not None:
                        next_prompt = pending_segment["prompt"]
                        dense_video_path = pending_segment["dense_video_path"]
                        sparse_video_path = pending_segment["sparse_video_path"]
                    active.pending_prompt = None
                    active.pending_prompt_dirty = False
                    interactions = []
                    await self._socket_send(
                        active,
                        {
                            "type": "segment_started",
                            "segment_index": active.chunk_index + 1,
                        },
                    )
                elif active.prompt_scheduled:
                    # The automatic first segment, each prompt boundary, and
                    # every acknowledged STEP are distinct work items. A STEP
                    # must never disappear into the automatic first segment.
                    active.first_action.clear()
                    active.segment_inflight = True
                    if active.initial_segment_pending:
                        active.initial_segment_pending = False
                        # A prompt received before inference begins still owns
                        # the next boundary, but an explicit STEP stays queued.
                        prompt_changed = active.pending_prompt_dirty
                    elif active.pending_prompt_dirty:
                        prompt_changed = True
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                    else:
                        active.segment_inflight = False
                        active.first_action.clear()
                        continue
                    if prompt_changed:
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                    interactions: list[str] = []
                    await self._socket_send(
                        active,
                        {
                            "type": "segment_started",
                            "segment_index": active.chunk_index + 1,
                        },
                    )
                else:
                    # Prompt changes may share the next native control chunk,
                    # while STEP remains a separate, action-free work item.
                    if active.pending_prompt_dirty:
                        prompt_changed = True
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                        frame_budget = self._runtime_next_input_frames()
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                        interactions = []
                    else:
                        frame_budget = self._runtime_next_input_frames()
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    if not interactions and not forced_step and not prompt_changed:
                        active.first_action.clear()
                        continue
                action_arrival = active.action_arrivals[0] if active.action_arrivals else None
                generation_started = loop.time()
                generation_task = asyncio.create_task(
                    self.runtime.generate(
                        interactions,
                        seed=active.seed + active.chunk_index,
                        control_segments=segments,
                        prompt=next_prompt,
                        dense_video_path=dense_video_path,
                        sparse_video_path=sparse_video_path,
                    ),
                    name="world-realtime-segment-inference",
                )
                try:
                    if active.queued_segments or active.prompt_scheduled:
                        interval = _env_int("WORLDFOUNDRY_SEGMENT_PROGRESS_SECONDS", 1)
                        while not generation_task.done():
                            done, _ = await asyncio.wait({generation_task}, timeout=interval)
                            if done:
                                break
                            await self._socket_send(
                                active,
                                {
                                    "type": "segment_progress",
                                    "segment_index": active.chunk_index + 1,
                                    "elapsed_ms": round(
                                        (loop.time() - generation_started) * 1000.0,
                                        1,
                                    ),
                                },
                            )
                    # Keep worker cancellation separate from the model task so
                    # the handler can explicitly cancel and retain a model
                    # coroutine that does not acknowledge cancellation.
                    frames, generation_ms = await asyncio.shield(generation_task)
                except ValueError as exc:
                    if not active.queued_segments:
                        raise
                    active.segment_inflight = False
                    await self._socket_send(
                        active,
                        {"type": "segment_rejected", "message": str(exc)},
                    )
                    continue
                except BaseException:
                    generation_task.cancel()
                    _retain_shutdown_task(generation_task, self._shutdown_tasks)
                    raise
                pixel_post_started = loop.time()
                quality = _env_int(
                    "WORLDFOUNDRY_QUEUED_SEGMENT_JPEG_QUALITY"
                    if active.queued_segments
                    else "WORLDFOUNDRY_REALTIME_SOCKET_JPEG_QUALITY",
                    95 if active.queued_segments else 88,
                )
                while True:
                    active.output_resolution.observe_source(frames[0] if frames else None)
                    resolution_snapshot = active.output_resolution.snapshot()
                    packets = await asyncio.to_thread(
                        _encode_jpeg_frames,
                        frames,
                        quality=quality,
                        subsampling=0 if active.queued_segments else 1,
                        output_resolution=resolution_snapshot.dimensions,
                    )
                    if resolution_snapshot.revision == active.output_resolution.revision:
                        break
                    # A config ACK raced this encode. Never enqueue obsolete
                    # packets after that ACK; re-encode the same RGB chunk at
                    # the newest boundary without rerunning model inference.
                    active.dropped_frames += len(packets)
                pixel_post_ms = (loop.time() - pixel_post_started) * 1000.0
                enqueue_started = loop.time()
                accepted = 0
                for packet in packets:
                    if active.frame_queue_policy is FrameQueuePolicy.ORDERED_QUALITY:
                        # Segment playback is count-exact: once chunk_done
                        # announces N frames, the browser must receive all N or
                        # it can never finish PLAYING. Backpressure this rare,
                        # explicit-boundary path instead of evicting frames.
                        await active.frame_packets.put(packet)
                        accepted += 1
                        continue
                    while active.frame_packets.full():
                        try:
                            active.frame_packets.get_nowait()
                            active.dropped_frames += 1
                        except asyncio.QueueEmpty:
                            break
                    active.frame_packets.put_nowait(packet)
                    accepted += 1
                enqueue_ms = (loop.time() - enqueue_started) * 1000.0
                active.chunk_index += 1
                now = loop.time()
                while active.action_arrivals and active.action_arrivals[0] <= generation_started:
                    active.action_arrivals.popleft()
                control_latency_ms = (now - action_arrival) * 1000.0 if action_arrival is not None else None
                queue_depth = active.frame_packets.qsize()
                self._record_perf(
                    active,
                    transport="websocket",
                    generation_started=generation_started,
                    now=now,
                    output_frames=accepted,
                    generation_ms=generation_ms,
                    pixel_post_ms=pixel_post_ms,
                    enqueue_ms=enqueue_ms,
                    queue_depth=queue_depth,
                    dropped_frames=active.dropped_frames,
                    control_latency_ms=control_latency_ms,
                )
                await self._socket_send(
                    active,
                    {
                        "type": "chunk_done",
                        "chunk_index": active.chunk_index,
                        "frames": accepted,
                        "generation_ms": round(generation_ms, 1),
                        "enqueue_ms": round(enqueue_ms, 1),
                        "control_latency_ms": round(control_latency_ms, 1) if control_latency_ms is not None else None,
                        "queue_depth": queue_depth,
                        "dropped_frames": active.dropped_frames,
                        "pixel_post_ms": round(pixel_post_ms, 1),
                        "resolution": resolution_snapshot.to_payload(),
                        "resolution_revision": resolution_snapshot.revision,
                        "interactions": (
                            ["queued_segment"]
                            if active.queued_segments
                            else [f"text_event:{prompt_event_id}"]
                            if prompt_event_id
                            else ["prompt_update"]
                            if prompt_changed
                            else ["step"]
                            if forced_step and not interactions
                            else interactions
                        ),
                        **_chunk_metric_payload(getattr(self.runtime, "last_generation_metrics", {}) or {}),
                    },
                )
                # The queue owns each encoded packet now. Do not retain a
                # second full segment of RGB/JPEG data while waiting for the
                # user's next boundary command.
                del packets, frames
                active.segment_inflight = False
                if active.queued_segments:
                    continue
                if (
                    active.initial_segment_pending
                    or active.pending_prompt_dirty
                    or active.pending_steps
                    or active.resampler.effective_keys
                ):
                    active.first_action.set()
                else:
                    active.first_action.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._socket_send(
                active,
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
            await self.close_active()

    async def _socket_sender(self, active: _ActiveSocket) -> None:
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        try:
            while not active.closed:
                packet = await active.frame_packets.get()
                if active.pacing_mode is WebSocketPacingMode.LEGACY_DUAL:
                    now = loop.time()
                    if deadline is not None:
                        deadline += 1.0 / self.fps
                        if deadline > now:
                            await asyncio.sleep(deadline - now)
                        elif now - deadline > 1.0 / self.fps:
                            deadline = now
                    else:
                        deadline = now
                async with active.send_lock:
                    await active.socket.send_bytes(packet)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.close_active()

    async def _socket_liveness_watchdog(self, active: _ActiveSocket) -> None:
        interval = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_INTERVAL_SECONDS", 1)
        timeout = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_TIMEOUT_SECONDS", 30)
        try:
            while not active.closed:
                await asyncio.sleep(interval)
                if asyncio.get_running_loop().time() - active.last_client_message_at > timeout:
                    await self.close_active()
                    return
        except asyncio.CancelledError:
            raise

    @staticmethod
    async def _socket_send(active: _ActiveSocket, payload: Mapping[str, Any]) -> None:
        if active.closed or active.socket.closed:
            return
        async with active.send_lock:
            await active.socket.send_str(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))

    async def _input_worker(self, active: _ActivePeer) -> None:
        """Apply reliable DataChannel messages in their original order."""

        try:
            while not active.closed:
                raw = await active.input_messages.get()
                await self._handle_message(active, raw)
        except asyncio.CancelledError:
            raise

    async def _handle_message(self, active: _ActivePeer, raw: Any) -> None:
        if active.closed or not isinstance(raw, str):
            return
        active.last_client_message_at = asyncio.get_running_loop().time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        message_type = str(payload.get("type") or "").lower()
        if message_type == "disconnect":
            await self.close_active()
            return
        if message_type in {"heartbeat", "ping"}:
            return
        acknowledgement = self._handle_session_control(active, payload)
        if acknowledgement is not None:
            self._send(active.channel, acknowledgement)
            return
        if message_type == "prompt_update" and active.prompt_scheduled:
            prompt = str(payload.get("prompt") or "").strip()
            if prompt:
                arrival = asyncio.get_running_loop().time()
                active.base_prompt = prompt
                active.active_event_id = None
                active.pending_prompt = prompt
                active.pending_prompt_dirty = True
                active.action_arrivals.append(arrival)
                active.first_action.set()
            return
        if message_type != "action":
            return
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        event = str(action.get("event") or "").lower()
        key = str(action.get("key") or "")
        arrival = asyncio.get_running_loop().time()
        if active.resampler.on_edge(arrival_time=arrival, event=event, key=key):
            active.action_arrivals.append(arrival)
            active.first_action.set()

    async def _liveness_watchdog(self, active: _ActivePeer) -> None:
        interval = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_INTERVAL_SECONDS", 1)
        timeout = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_TIMEOUT_SECONDS", 30)
        try:
            while not active.closed:
                await asyncio.sleep(interval)
                if asyncio.get_running_loop().time() - active.last_client_message_at > timeout:
                    await self.close_active()
                    return
        except asyncio.CancelledError:
            raise

    async def _peer_setup_watchdog(self, active: _ActivePeer) -> None:
        """Free the single session when a peer never opens its DataChannel.

        Intentionally not cancelled on channel open: after the timeout it
        re-checks the session state and exits quietly when the channel arrived
        or the session already closed, so its trigger path cannot race
        close_active the way a cancellation from ``on_datachannel`` could.
        ``close_active`` still cancels it (with the current-task guard) so
        teardown never leaves it sleeping on a stale session.
        """

        timeout = _env_int("WORLDFOUNDRY_REALTIME_SOCKET_SETUP_TIMEOUT_SECONDS", 15)
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            raise
        if active.closed or active.channel is not None:
            return
        logger.warning(
            "Realtime peer completed signaling but opened no DataChannel within %ds; "
            "closing the half-open session.",
            timeout,
        )
        await self.close_active()

    async def _generation_worker(self, active: _ActivePeer) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not active.closed:
                await active.first_action.wait()
                # Let the ordered input worker drain the current network burst
                # before sampling. This coalesces adjacent key edges (for
                # example a diagonal joystick gesture) without a timer-based
                # debounce or an extra frame of latency.
                await asyncio.sleep(0)
                next_prompt = None
                prompt_changed = False
                prompt_event_id: str | None = None
                forced_step = False
                segments = None
                if active.prompt_scheduled:
                    active.first_action.clear()
                    if active.initial_segment_pending:
                        active.initial_segment_pending = False
                        prompt_changed = active.pending_prompt_dirty
                    elif active.pending_prompt_dirty:
                        prompt_changed = True
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                    else:
                        active.first_action.clear()
                        continue
                    if prompt_changed:
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                    interactions: list[str] = []
                else:
                    if active.pending_prompt_dirty:
                        prompt_changed = True
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                        frame_budget = self._runtime_next_input_frames()
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                        interactions = []
                    else:
                        frame_budget = self._runtime_next_input_frames()
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    if not interactions and not forced_step and not prompt_changed:
                        active.first_action.clear()
                        continue
                action_arrival = active.action_arrivals[0] if active.action_arrivals else None
                generation_started = loop.time()
                frames, generation_ms = await self.runtime.generate(
                    interactions,
                    seed=active.seed + active.chunk_index,
                    control_segments=segments,
                    prompt=next_prompt,
                )
                pixel_post_started = loop.time()
                while True:
                    active.output_resolution.observe_source(frames[0] if frames else None)
                    resolution_snapshot = active.output_resolution.snapshot()
                    if resolution_snapshot.dimensions is None or all(
                        (frame.shape[1], frame.shape[0]) == resolution_snapshot.dimensions for frame in frames
                    ):
                        transport_frames = frames
                    else:
                        transport_frames = await asyncio.to_thread(
                            _resize_rgb_frames,
                            frames,
                            output_resolution=resolution_snapshot.dimensions,
                        )
                    if resolution_snapshot.revision == active.output_resolution.revision:
                        break
                pixel_post_ms = (loop.time() - pixel_post_started) * 1000.0
                accepted = await active.frames.put_chunk(transport_frames)
                active.chunk_index += 1
                now = loop.time()
                while active.action_arrivals and active.action_arrivals[0] <= generation_started:
                    active.action_arrivals.popleft()
                control_latency_ms = (now - action_arrival) * 1000.0 if action_arrival is not None else None
                queue_depth = active.frames.qsize()
                self._record_perf(
                    active,
                    transport="webrtc",
                    generation_started=generation_started,
                    now=now,
                    output_frames=accepted,
                    generation_ms=generation_ms,
                    pixel_post_ms=pixel_post_ms,
                    enqueue_ms=active.frames.last_enqueue_ms,
                    queue_depth=queue_depth,
                    dropped_frames=active.frames.dropped_frames,
                    control_latency_ms=control_latency_ms,
                )
                self._send(
                    active.channel,
                    {
                        "type": "chunk_done",
                        "chunk_index": active.chunk_index,
                        "frames": accepted,
                        "generation_ms": round(generation_ms, 1),
                        "enqueue_ms": round(active.frames.last_enqueue_ms, 1),
                        "control_latency_ms": round(control_latency_ms, 1) if control_latency_ms is not None else None,
                        "queue_depth": queue_depth,
                        "dropped_frames": active.frames.dropped_frames,
                        "dropped_chunks": getattr(active.frames, "dropped_chunks", 0),
                        "presented_frames": getattr(active.frames, "presented_frames", None),
                        "presentation_mode": active.presentation_mode.value,
                        "pixel_post_ms": round(pixel_post_ms, 1),
                        "resolution": resolution_snapshot.to_payload(),
                        "resolution_revision": resolution_snapshot.revision,
                        "interactions": (
                            [f"text_event:{prompt_event_id}"]
                            if prompt_event_id
                            else ["prompt_update"]
                            if prompt_changed
                            else ["step"]
                            if forced_step and not interactions
                            else interactions
                        ),
                        **_chunk_metric_payload(getattr(self.runtime, "last_generation_metrics", {}) or {}),
                    },
                )
                if (
                    active.initial_segment_pending
                    or active.pending_prompt_dirty
                    or active.pending_steps
                    or active.resampler.effective_keys
                ):
                    active.first_action.set()
                else:
                    active.first_action.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._send(active.channel, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            await self.close_active()

    @staticmethod
    def _send(channel: Any | None, payload: Mapping[str, Any]) -> None:
        if channel is None or getattr(channel, "readyState", "") != "open":
            return
        channel.send(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))


def _build_video_track(
    *,
    frames: LatestFrameBuffer | ChunkPresentationBuffer,
    fps: int,
    presentation_mode: RealtimePresentationMode | str = RealtimePresentationMode.HOLD_LAST,
) -> Any:
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import VideoFrame

    class RealtimeVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._presentation_mode = RealtimePresentationMode.from_value(presentation_mode)
            self._pts = 0
            self._time_base = Fraction(1, fps)
            self._interval = 1.0 / fps
            self._deadline: float | None = None
            self._last_array: np.ndarray | None = None
            self._first_real_frame_at: float | None = None

        async def recv(self) -> VideoFrame:
            loop = asyncio.get_running_loop()
            if self._presentation_mode is RealtimePresentationMode.REAL_FRAMES:
                try:
                    frame_array = await frames.get()
                except EOFError as exc:
                    raise MediaStreamError from exc
                now = loop.time()
                if self._first_real_frame_at is None:
                    self._first_real_frame_at = now
                pts = max(self._pts, round((now - self._first_real_frame_at) * fps))
            else:
                if self._last_array is None:
                    try:
                        self._last_array = await frames.get()
                    except EOFError as exc:
                        raise MediaStreamError from exc
                    self._deadline = loop.time()
                else:
                    assert self._deadline is not None
                    now = loop.time()
                    self._deadline += self._interval
                    if self._deadline > now:
                        await asyncio.sleep(self._deadline - now)
                    elif now - self._deadline > self._interval:
                        self._deadline = now
                    try:
                        self._last_array = frames.get_nowait()
                    except asyncio.QueueEmpty:
                        # Keep a constant RTP clock even between generated chunks.
                        # Besides giving the browser a stable live stream, the
                        # repeated hold frame flushes the encoder's final source
                        # frame instead of leaving it buffered until the next user
                        # action arrives.
                        pass
                    except EOFError as exc:
                        raise MediaStreamError from exc
                frame_array = self._last_array
                pts = self._pts
            # Frames are already resized in the generation worker. Keep the
            # aiortc cadence path free of synchronous resize/canvas work; a
            # held frame is therefore repeated at zero additional scale cost.
            frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
            frame.pts = pts
            frame.time_base = self._time_base
            self._pts = pts + 1
            return frame

    return RealtimeVideoTrack()


async def _wait_for_ice_gathering(peer: Any) -> None:
    if peer.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @peer.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if peer.iceGatheringState == "complete":
            complete.set()

    if peer.iceGatheringState == "complete":
        complete.set()
    await asyncio.wait_for(
        complete.wait(),
        timeout=_env_int("WORLDFOUNDRY_REALTIME_ICE_TIMEOUT_SECONDS", 15),
    )


def _require_realtime_dependencies(*, require_rtc: bool, require_av: bool) -> None:
    missing: list[str] = []
    names = ["aiohttp"]
    if require_rtc:
        names.append("aiortc")
    if require_rtc or require_av:
        names.append("av")
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Realtime World frontend requires " + ", ".join(missing) + ". Install `worldfoundry[studio_realtime]`."
        )


def _prefer_websocket_transport(
    *,
    prompt_scheduled: bool,
    queued_segments: bool,
    remote: str,
) -> bool:
    """Choose the transport without making segment UIs depend on ICE."""

    return prompt_scheduled or queued_segments or remote in {"127.0.0.1", "::1"}


def serve_realtime_world_frontend(
    *,
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
    manager: StudioManager,
    host: str,
    port: int,
    demo_images: tuple[Path, ...],
    allowed_roots: tuple[Path, ...],
) -> None:
    """Serve the model-specific WebRTC frontend."""

    prompt_scheduled = "prompt-scheduled" in entry.tags
    queued_segments = "queued-segment-generation" in entry.tags
    websocket_only = os.getenv("WORLDFOUNDRY_REALTIME_WEBSOCKET_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    _require_realtime_dependencies(
        # Same-origin WebSocket streaming is fully self-contained and is the
        # reliable transport for SSH tunnels.  Keep aiortc optional when the
        # operator explicitly selects that path; the default WebRTC behavior
        # and dependency check remain unchanged.
        require_rtc=not (prompt_scheduled or queued_segments or websocket_only),
        require_av=queued_segments,
    )
    from aiohttp import web

    from worldfoundry.studio.visualization.backends.world import (
        _reference_image_label,
        world_favicon_svg,
        world_frontend_css,
        world_frontend_html,
        world_frontend_js,
    )

    if prompt_scheduled or queued_segments or "user-input-only" in entry.tags:
        # These modes must start from the user's own upload. Do not silently
        # expose packaged reference images as product examples or use them for
        # warmup.
        demo_images = ()

    fps = _env_int("WORLDFOUNDRY_REALTIME_FPS", 16)
    chunk_frames = _realtime_frame_budget(
        entry,
        _env_int(
            "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
            _default_realtime_chunk_frames(entry),
        ),
    )
    ice_servers = _ice_server_payload()
    runtime = ResidentWorldRuntime(
        manager=manager,
        entry=entry,
        launch_config=launch_config,
        fps=fps,
        warmup_image_path=str(demo_images[0]) if demo_images else "",
        warmup_chunks=_env_int(
            "WORLDFOUNDRY_REALTIME_WARMUP_CHUNKS",
            0,
            minimum=0,
        ),
        postprocess=_realtime_postprocess_stream(
            fps=fps,
            launch_config=launch_config,
            native_resolution=_entry_output_resolution(entry),
        ),
    )
    peers = RealtimePeerManager(
        runtime=runtime,
        fps=fps,
        chunk_frames=chunk_frames,
        ice_servers=ice_servers,
    )
    upload_root = Path(manager.workspace_root).expanduser().resolve() / "world_realtime_inputs"
    upload_root.mkdir(parents=True, exist_ok=True)
    owned_uploads: set[Path] = set()
    upload_cleanup_lock = asyncio.Lock()
    file_roots = tuple(dict.fromkeys(root.expanduser().resolve() for root in (*allowed_roots, upload_root)))
    upload_allowed_exts = frozenset(IMAGE_EXTS) | frozenset(VIDEO_EXTS)
    upload_max_bytes = _env_int("WORLDFOUNDRY_UPLOAD_MAX_BYTES", 512 * 1024 * 1024)

    auth_token = require_auth_token_for_host(host, server_name="Realtime world frontend")
    middlewares = []
    if auth_token:
        static_paths = {"/", "/world.css", "/world.js", "/favicon.svg"}

        @web.middleware
        async def require_studio_token(request: Any, handler: Any) -> Any:
            if request.path in static_paths:
                return await handler(request)
            if not request_token_valid(
                auth_token,
                authorization_header=request.headers.get("Authorization"),
                query_token=request.query.get("token"),
            ):
                raise web.HTTPUnauthorized(reason="Missing or invalid Studio auth token.")
            return await handler(request)

        middlewares.append(require_studio_token)

    app = web.Application(client_max_size=2 * 1024**3, middlewares=middlewares)
    preload_task: asyncio.Task[Any] | None = None
    shutdown_tasks: set[asyncio.Task[Any]] = set()

    async def cleanup_owned_uploads() -> None:
        async with upload_cleanup_lock:
            paths = tuple(owned_uploads)
            owned_uploads.clear()
        await asyncio.gather(
            *(asyncio.to_thread(path.unlink, missing_ok=True) for path in paths),
            return_exceptions=True,
        )

    async def index(_: Any) -> Any:
        return web.Response(text=world_frontend_html(entry, launch_config), content_type="text/html")

    async def css(_: Any) -> Any:
        return web.Response(text=world_frontend_css(), content_type="text/css")

    async def js(_: Any) -> Any:
        return web.Response(text=world_frontend_js(), content_type="application/javascript")

    async def favicon(_: Any) -> Any:
        return web.Response(text=world_favicon_svg(), content_type="image/svg+xml")

    async def session_info(request: Any) -> Any:
        remote = str(request.remote or "").split("%", 1)[0]
        # Segment models exchange a small ordered command at each generation
        # boundary, then stream RGB frames. They do not benefit from an RTC
        # control channel, while ICE discovery adds a failure-prone startup
        # round trip for SSH tunnels and reverse proxies.
        segment_transport = prompt_scheduled or queued_segments or websocket_only
        prefer_socket = websocket_only or _prefer_websocket_transport(
            prompt_scheduled=prompt_scheduled,
            queued_segments=queued_segments,
            remote=remote,
        )
        examples = [
            {
                "label": _reference_image_label(path, index),
                "path": str(path),
                "url": f"/api/file?path={quote(str(path), safe='')}",
            }
            for index, path in enumerate(demo_images[:9], start=1)
        ]
        return web.json_response(
            {
                "model_id": entry.model_id,
                "display_name": entry.display_name,
                "transport": "websocket" if segment_transport else "webrtc+websocket",
                "runtime_engine": "worldfoundry-resident",
                "postprocess": list(runtime.postprocess_names),
                "performance_contract": ("in-tree-full-quality-segments" if queued_segments else "in-tree-realtime"),
                "websocket_fallback": True,
                "prefer_websocket": prefer_socket,
                "supports_video_input": not (prompt_scheduled or queued_segments),
                "interaction_mode": (
                    "queued-segments" if queued_segments else "prompt-scheduled" if prompt_scheduled else "controls"
                ),
                "fps": runtime.realtime_spec.fps,
                "chunk_frames": runtime.realtime_spec.first_chunk_frames,
                "steady_chunk_frames": runtime.realtime_spec.steady_chunk_frames,
                "controls": list(runtime.realtime_spec.controls),
                "realtime_spec": runtime.realtime_spec.to_payload(),
                "capabilities": {
                    "text_events": runtime.supports_text_events,
                    "runtime_event_catalog": True,
                    "event_ack": True,
                    "no_action_step": not queued_segments,
                    "output_resolution": True,
                    "draggable_panels": True,
                },
                "event_catalog": [],
                "active_event_id": None,
                "output_resolution": {"mode": "native"},
                "output_resolutions": _output_resolution_options(
                    entry,
                    native_override=runtime.transport_native_resolution,
                ),
                "output_resolution_scope": "transport",
                "ice_servers": ice_servers,
                "examples": examples,
            }
        )

    async def runtime_status(_: Any) -> Any:
        return web.json_response(
            {
                "ready": runtime.ready,
                "loading": not runtime.ready and runtime.preload_error is None,
                "error": runtime.preload_error,
                "warmup_ms": round(runtime.warmup_ms, 1),
                "session_active": peers.active,
                "draining": peers.draining,
                "postprocess": list(runtime.postprocess_names),
                "realtime_spec": runtime.realtime_spec.to_payload(),
            }
        )

    async def load_runtime(_: Any) -> Any:
        await runtime.preload()
        return web.json_response({"loaded": True, "ready": True})

    async def upload_input(request: Any) -> Any:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            raise web.HTTPBadRequest(reason="Expected multipart field named 'file'.")
        filename = Path(field.filename or "input.bin").name
        suffix = Path(filename).suffix[:16].lower()
        if suffix not in upload_allowed_exts:
            raise web.HTTPBadRequest(
                reason=(
                    f"Unsupported upload type {suffix or '<no suffix>'}; expected one of "
                    f"{', '.join(sorted(upload_allowed_exts))}."
                )
            )
        target = upload_root / f"{uuid.uuid4().hex}{suffix}"
        written = 0
        handle = await asyncio.to_thread(target.open, "wb")
        try:
            try:
                while chunk := await field.read_chunk(size=1024 * 1024):
                    written += len(chunk)
                    if written > upload_max_bytes:
                        raise web.HTTPBadRequest(
                            reason=(
                                f"Upload exceeds the per-file limit of {upload_max_bytes} bytes "
                                "(override with WORLDFOUNDRY_UPLOAD_MAX_BYTES)."
                            )
                        )
                    await asyncio.to_thread(handle.write, chunk)
            finally:
                await asyncio.to_thread(handle.close)
        except BaseException:
            # Never leave rejected or interrupted uploads on disk.
            await asyncio.to_thread(target.unlink, missing_ok=True)
            raise
        owned_uploads.add(target)
        return web.json_response({"path": str(target), "url": f"/api/file?path={quote(str(target), safe='')}"})

    async def file_response(request: Any) -> Any:
        raw = str(request.query.get("path") or "")
        path = Path(raw).expanduser().resolve()
        if not path_allowed(path, file_roots):
            raise web.HTTPForbidden(reason="Path is outside allowed Studio roots.")
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def offer(request: Any) -> Any:
        payload = await request.json()
        offer_payload = payload.get("offer") if isinstance(payload.get("offer"), Mapping) else payload
        session_payload = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}
        if not offer_payload.get("sdp") or not offer_payload.get("type"):
            raise web.HTTPBadRequest(reason="Expected WebRTC offer sdp and type.")
        try:
            answer = await peers.create_answer(offer=offer_payload, session=session_payload)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except RuntimeError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        return web.json_response(answer)

    async def websocket(request: Any) -> Any:
        socket = web.WebSocketResponse(autoping=True, heartbeat=10, max_msg_size=1024**2)
        await socket.prepare(request)
        try:
            await peers.serve_socket(socket)
        except Exception as exc:
            if not socket.closed:
                await socket.send_str(
                    json.dumps(
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                await socket.close(code=1011, message=b"realtime session failed")
        finally:
            await cleanup_owned_uploads()
        return socket

    async def reset_session(request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        await peers.close_active()
        if not bool(payload.get("preserve_uploads")):
            await cleanup_owned_uploads()
        return web.json_response({"reset": True})

    async def on_startup(_: Any) -> None:
        # Overlap model residency work with input selection while keeping the
        # HTML server immediately reachable.
        nonlocal preload_task
        preload_task = asyncio.create_task(runtime.preload(), name="world-realtime-preload")

        def consume_preload_result(task: asyncio.Task[Any]) -> None:
            if not task.cancelled():
                task.exception()

        preload_task.add_done_callback(consume_preload_result)

        stale_after = _env_int("WORLDFOUNDRY_UPLOAD_STALE_SECONDS", 24 * 60 * 60)
        cutoff = time.time() - stale_after
        stale = [path for path in upload_root.iterdir() if path.is_file() and path.stat().st_mtime < cutoff]
        await asyncio.gather(
            *(asyncio.to_thread(path.unlink, missing_ok=True) for path in stale),
            return_exceptions=True,
        )

    async def on_shutdown(_: Any) -> None:
        deadline = _shutdown_deadline()
        await _finish_tasks_by_deadline(
            (preload_task,),
            deadline=deadline,
            retained=shutdown_tasks,
            label="application preload",
            cancel_now=True,
        )
        await peers.close_active(deadline=deadline)
        await runtime.close(deadline=deadline)
        await _run_by_shutdown_deadline(
            cleanup_owned_uploads(),
            deadline=deadline,
            retained=shutdown_tasks,
            label="owned upload cleanup",
        )

    app.router.add_get("/", index)
    app.router.add_get("/world.css", css)
    app.router.add_get("/world.js", js)
    app.router.add_get("/favicon.svg", favicon)
    app.router.add_get("/api/session", session_info)
    app.router.add_get("/api/runtime/status", runtime_status)
    app.router.add_post("/api/runtime/load", load_runtime)
    app.router.add_post("/api/session/input", upload_input)
    app.router.add_get("/api/file", file_response)
    app.router.add_get("/api/realtime/ws", websocket)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_post("/api/session/reset", reset_session)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    print(f"WorldFoundry realtime WebRTC UI: http://{host}:{port}", flush=True)
    warning = bind_security_warning(host)
    if warning:
        print(warning, flush=True)
    web.run_app(app, host=host, port=port, print=None, handle_signals=True)


__all__ = [
    "LatestFrameBuffer",
    "OutputResolutionState",
    "RealtimeControlResampler",
    "RealtimeControlState",
    "ResidentWorldRuntime",
    "interactions_from_keys",
    "interactions_from_segments",
    "normalize_output_resolution",
    "normalize_text_events",
    "realtime_frames_from_result",
    "serve_realtime_world_frontend",
]
