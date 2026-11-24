# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Piecewise CUDA Graph runner for block-stack execution alongside dynamic caches.

Whole-model CUDA Graphs cannot capture dynamic Python control flow such as
TeaCache / MagCache / FirstBlock step-skip decisions or causal KV-cache slot
advancement. :class:`PiecewiseBlockGraphRunner` splits execution at natural
segment boundaries (for example the full transformer block stack on cache-miss
steps, or ``head`` + ``tail`` block segments for FirstBlock/EasyCache) while
keeping cache decisions and request metadata outside the captured graphs.

Each segment delegates capture, signature bucketing, static buffer management,
and request-window telemetry to :class:`InferenceCUDAGraphRunner`.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner


class PiecewiseBlockGraphRunner:
    """Manage named, independently captured CUDA Graph segments for one model.

    Unlike whole-network capture, dynamic step-cache controllers remain in
    Python outside each segment. On a cache hit the caller invokes
    :meth:`record_skip` without replaying the segment; on a cache miss the
    caller invokes :meth:`run_segment` to replay the captured CUDA Graph (with
    FP8 linears, fused attention, and sequence parallelism inside the graph).

    Args:
        warmup: Eager warmup iterations per segment signature before capture.
        max_graphs_per_segment: LRU graph cap for each registered segment.
        clone_outputs: Clone static graph outputs on replay to prevent aliasing.
        enabled: Master switch; when False every segment runs eager.
    """

    def __init__(
        self,
        *,
        warmup: int = 2,
        max_graphs_per_segment: int = 16,
        clone_outputs: bool = True,
        enabled: bool = True,
    ) -> None:
        self._warmup = int(warmup)
        self._max_graphs_per_segment = int(max_graphs_per_segment)
        self._clone_outputs = bool(clone_outputs)
        self._enabled = bool(enabled)
        self._segments: dict[str, InferenceCUDAGraphRunner] = {}
        self._skips: dict[str, int] = {}
        self._window_skips_baseline: dict[str, int] = {}
        self._frozen = False

    def register_segment(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        extra_key: str | None = None,
    ) -> InferenceCUDAGraphRunner:
        """Register a pure-tensor segment callable under ``name``."""
        segment_name = str(name).strip()
        if not segment_name:
            raise ValueError("piecewise graph segment name cannot be empty")
        if segment_name in self._segments:
            raise ValueError(f"piecewise graph segment {segment_name!r} is already registered")
        runner = InferenceCUDAGraphRunner(
            fn,
            warmup=self._warmup,
            max_graphs=self._max_graphs_per_segment,
            clone_outputs=self._clone_outputs,
            enabled=self._enabled,
            extra_key=extra_key or f"piecewise:{segment_name}",
        )
        if self._frozen:
            runner.freeze()
        self._segments[segment_name] = runner
        self._skips[segment_name] = 0
        self._window_skips_baseline[segment_name] = 0
        return runner

    def get_or_register_segment(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        extra_key: str | None = None,
    ) -> InferenceCUDAGraphRunner:
        """Return an existing segment runner or register ``fn`` once."""
        existing = self._segments.get(name)
        if existing is not None:
            return existing
        return self.register_segment(name, fn, extra_key=extra_key)

    def run_segment(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered segment via captured replay or eager fallback."""
        runner = self._segments.get(name)
        if runner is None:
            raise KeyError(f"piecewise graph segment {name!r} is not registered")
        return runner(*args, **kwargs)

    def record_skip(self, name: str, count: int = 1) -> None:
        """Record that a cache controller bypassed segment ``name``."""
        if count < 0:
            raise ValueError("skip count must be non-negative")
        self._skips[name] = self._skips.get(name, 0) + int(count)

    def begin_request_window(self) -> None:
        """Reset per-request telemetry while preserving captured graphs."""
        for name, runner in self._segments.items():
            runner.begin_request_window()
            self._window_skips_baseline[name] = self._skips.get(name, 0)

    def freeze(self) -> None:
        """Forbid new captures on unseen signatures across all segments."""
        self._frozen = True
        for runner in self._segments.values():
            runner.freeze()

    @contextmanager
    def capture_scope(self) -> Generator[None, None, None]:
        """Temporarily permit capture across all segments during explicit warmup."""
        previous_frozen = self._frozen
        self._frozen = False
        with ExitStack() as stack:
            for runner in self._segments.values():
                stack.enter_context(runner.capture_scope())
            try:
                yield
            finally:
                self._frozen = previous_frozen

    def invalidate(self) -> None:
        """Drop all captured graphs and static buffers across every segment."""
        for runner in self._segments.values():
            runner.invalidate()

    def report(self) -> dict[str, Any]:
        """Return aggregate and per-segment capture/replay/skip telemetry."""
        segments: dict[str, Mapping[str, Any]] = {}
        total_captures = 0
        total_replays = 0
        total_eager = 0
        total_skips = 0
        window_captures = 0
        window_replays = 0
        window_eager = 0
        window_skips = 0
        for name, runner in self._segments.items():
            seg_report = runner.report()
            seg_skips = self._skips.get(name, 0)
            seg_win_skips = seg_skips - self._window_skips_baseline.get(name, 0)
            seg_window = dict(seg_report.get("request_window", {}))
            seg_window["cache_skips"] = seg_win_skips
            segments[name] = {
                **seg_report,
                "cache_skips": seg_skips,
                "request_window": seg_window,
            }
            total_captures += int(seg_report.get("capture", 0))
            total_replays += int(seg_report.get("replay", 0))
            total_eager += int(seg_report.get("eager", 0))
            total_skips += seg_skips
            window_captures += int(seg_window.get("capture", 0))
            window_replays += int(seg_window.get("replay", 0))
            window_eager += int(seg_window.get("eager", 0))
            window_skips += seg_win_skips
        return {
            "mode": "piecewise-block-graph",
            "enabled": self._enabled,
            "frozen": self._frozen,
            "segment_count": len(self._segments),
            "capture": total_captures,
            "replay": total_replays,
            "eager": total_eager,
            "cache_skips": total_skips,
            "request_window": {
                "capture": window_captures,
                "replay": window_replays,
                "eager": window_eager,
                "cache_skips": window_skips,
            },
            "segments": segments,
        }


__all__ = ["PiecewiseBlockGraphRunner"]
