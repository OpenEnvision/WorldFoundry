# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deadline-aware prewarm helpers for realtime inference.

Responsibility: run cold-start and steady-state warmup steps under one
shared deadline, recording wall times without cancelling an in-flight
model worker.

This module is not a CUDA Graph capturer, a thread pool, or a request
canceller. Model calls commonly execute on a dedicated thread and
cannot be aborted safely; the deadline is checked *before and after*
each callback so an overrun is reported only once the callback returns.

Public surface:
- :class:`PrewarmTimeoutError` / :class:`PrewarmDeadline`
- :class:`PrewarmTiming` / :class:`PrewarmSequenceTiming`
- :func:`run_timed_prewarm` / :func:`run_prewarm_sequence` /
  :func:`run_async_prewarm_sequence`
- :func:`cuda_graph_prewarm_steps` — warmup + capture + replay count

Asynchronous serving callbacks share the same deadline semantics so
Studio can time a worker without cancelling an in-flight model call.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Timing records — wall clocks only; no CUDA-event attribution
# ──────────────────────────────────────────────────────────────────────────


class PrewarmTimeoutError(TimeoutError):
    """Raised after a synchronous model step crosses its prewarm deadline."""

    def __init__(self, label: str, *, timeout_s: float, elapsed_s: float) -> None:
        """Keep label/budget/elapsed so Studio can log without parsing the message."""
        self.label = label
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s
        super().__init__(f"Prewarm sequence {label!r} exceeded its {timeout_s:.3f}s deadline after {elapsed_s:.3f}s.")


@dataclass(frozen=True, slots=True)
class PrewarmTiming:
    """Wall-clock timing for one prewarm step."""

    label: str
    start_time: float
    end_time: float

    @property
    def elapsed_s(self) -> float:
        """Step duration in seconds (same clock as ``time_fn``)."""
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Step duration in milliseconds for log lines that already use ms."""
        return self.elapsed_s * 1000.0


@dataclass(frozen=True, slots=True)
class PrewarmSequenceTiming:
    """Timings for an optional cold-start step and steady-state steps."""

    label: str
    cold_start: PrewarmTiming | None
    steady_state: tuple[PrewarmTiming, ...]

    @property
    def steps(self) -> tuple[PrewarmTiming, ...]:
        """Cold start first when present, then every steady-state record."""
        if self.cold_start is None:
            return self.steady_state
        return (self.cold_start, *self.steady_state)

    @property
    def elapsed_s(self) -> float:
        """Span from the first step start to the last step end (0 if empty)."""
        if not self.steps:
            return 0.0
        return self.steps[-1].end_time - self.steps[0].start_time

    @property
    def elapsed_ms(self) -> float:
        """Sequence span in milliseconds."""
        return self.elapsed_s * 1000.0


@dataclass(frozen=True, slots=True)
class PrewarmDeadline:
    """Shared deadline checked around non-interruptible prewarm callbacks."""

    label: str
    start_time: float
    timeout_s: float | None = None

    @classmethod
    def start(
        cls,
        *,
        label: str = "prewarm",
        timeout_s: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> "PrewarmDeadline":
        """Stamp ``start_time`` now; ``timeout_s=None`` means no expiry."""
        _validate_timeout(timeout_s)
        return cls(label=label, start_time=time_fn(), timeout_s=timeout_s)

    def elapsed_s(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> float:
        """Seconds since :meth:`start`; ``now`` avoids a second clock read after a step."""
        return (time_fn() if now is None else now) - self.start_time

    def remaining_s(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> float | None:
        """Seconds left, clamped at 0; ``None`` when no timeout was configured."""
        if self.timeout_s is None:
            return None
        return max(self.timeout_s - self.elapsed_s(now=now, time_fn=time_fn), 0.0)

    def raise_if_expired(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Raise :class:`PrewarmTimeoutError` only after the in-flight callback returned."""
        if self.timeout_s is None:
            return
        elapsed_s = self.elapsed_s(now=now, time_fn=time_fn)
        if elapsed_s > self.timeout_s:
            raise PrewarmTimeoutError(
                self.label,
                timeout_s=self.timeout_s,
                elapsed_s=elapsed_s,
            )


# ──────────────────────────────────────────────────────────────────────────
# Runners — check deadline around each callback; never cancel in-flight work
# ──────────────────────────────────────────────────────────────────────────


def run_timed_prewarm(
    step: Callable[[], Any],
    *,
    label: str = "prewarm",
    timeout_s: float | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> PrewarmTiming:
    """Run one synchronous prewarm callback and record its duration."""

    deadline = PrewarmDeadline.start(label=label, timeout_s=timeout_s, time_fn=time_fn)
    return _run_deadlined_step(step, label=label, deadline=deadline, time_fn=time_fn)


def run_prewarm_sequence(
    *,
    steady_state: Callable[[int], Any],
    steady_steps: int,
    cold_start: Callable[[], Any] | None = None,
    label: str = "prewarm",
    timeout_s: float | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> PrewarmSequenceTiming:
    """Run a synchronous cold start and indexed steady-state sequence."""

    _validate_steps(steady_steps)
    deadline = PrewarmDeadline.start(label=label, timeout_s=timeout_s, time_fn=time_fn)
    cold_timing = (
        _run_deadlined_step(
            cold_start,
            label=f"{label}.cold_start",
            deadline=deadline,
            time_fn=time_fn,
        )
        if cold_start is not None
        else None
    )
    steady_timings = tuple(
        _run_deadlined_step(
            lambda index=index: steady_state(index),
            label=f"{label}.steady_state.{index}",
            deadline=deadline,
            time_fn=time_fn,
        )
        for index in range(steady_steps)
    )
    return PrewarmSequenceTiming(label, cold_timing, steady_timings)


async def run_async_prewarm_sequence(
    *,
    steady_state: Callable[[int], Awaitable[Any]],
    steady_steps: int,
    cold_start: Callable[[], Awaitable[Any]] | None = None,
    label: str = "prewarm",
    timeout_s: float | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> PrewarmSequenceTiming:
    """Run asynchronous serving callbacks under one non-cancelling deadline.

    Model calls commonly execute on a dedicated thread and cannot be cancelled
    safely.  The deadline is therefore checked before and after every callback;
    an overrun is reported only once the in-flight callback has returned.
    """

    _validate_steps(steady_steps)
    deadline = PrewarmDeadline.start(label=label, timeout_s=timeout_s, time_fn=time_fn)
    cold_timing = (
        await _run_async_deadlined_step(
            cold_start,
            label=f"{label}.cold_start",
            deadline=deadline,
            time_fn=time_fn,
        )
        if cold_start is not None
        else None
    )
    steady_timings: list[PrewarmTiming] = []
    for index in range(steady_steps):
        timing = await _run_async_deadlined_step(
            lambda index=index: steady_state(index),
            label=f"{label}.steady_state.{index}",
            deadline=deadline,
            time_fn=time_fn,
        )
        steady_timings.append(timing)
    return PrewarmSequenceTiming(label, cold_timing, tuple(steady_timings))


def cuda_graph_prewarm_steps(
    *,
    warmup_iters: int,
    capture_steps: int = 1,
    replay_steps: int = 1,
) -> int:
    """Return calls needed to warm, capture, and replay a steady CUDA graph."""

    for name, value in (
        ("warmup_iters", warmup_iters),
        ("capture_steps", capture_steps),
        ("replay_steps", replay_steps),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    return warmup_iters + capture_steps + replay_steps


def _run_deadlined_step(
    step: Callable[[], Any],
    *,
    label: str,
    deadline: PrewarmDeadline,
    time_fn: Callable[[], float],
) -> PrewarmTiming:
    """Refuse to start if already expired; raise after return if the step overran."""
    deadline.raise_if_expired(time_fn=time_fn)
    started = time_fn()
    result = step()
    del result
    finished = time_fn()
    deadline.raise_if_expired(now=finished, time_fn=time_fn)
    return PrewarmTiming(label, started, finished)


async def _run_async_deadlined_step(
    step: Callable[[], Awaitable[Any]],
    *,
    label: str,
    deadline: PrewarmDeadline,
    time_fn: Callable[[], float],
) -> PrewarmTiming:
    """Same as :func:`_run_deadlined_step` for awaitables that cannot be cancelled."""
    deadline.raise_if_expired(time_fn=time_fn)
    started = time_fn()
    result = await step()
    del result
    finished = time_fn()
    deadline.raise_if_expired(now=finished, time_fn=time_fn)
    return PrewarmTiming(label, started, finished)


def _validate_timeout(timeout_s: float | None) -> None:
    """Reject a negative budget; ``None`` is the explicit 'no deadline' sentinel."""
    if timeout_s is not None and timeout_s < 0.0:
        raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")


def _validate_steps(steady_steps: int) -> None:
    """Zero steady steps is legal (cold-start-only); a negative count is not."""
    if steady_steps < 0:
        raise ValueError(f"steady_steps must be non-negative, got {steady_steps}.")


__all__ = [
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "cuda_graph_prewarm_steps",
    "run_async_prewarm_sequence",
    "run_prewarm_sequence",
    "run_timed_prewarm",
]
