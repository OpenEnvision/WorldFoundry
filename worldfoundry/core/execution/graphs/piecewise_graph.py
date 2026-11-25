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

import torch

from worldfoundry.core.execution.graphs.cuda_graph import graph_pool_handle
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
        share_pool: Share one CUDA Graph private memory pool across sequential
            segments so intermediate activation buffers are reused instead of
            duplicated per segment. Automatically disabled when
            ``clone_outputs=False`` to guarantee retained static outputs cannot
            be overwritten by another segment's replay.
    """

    def __init__(
        self,
        *,
        warmup: int = 2,
        max_graphs_per_segment: int = 16,
        clone_outputs: bool = True,
        enabled: bool = True,
        share_pool: bool = True,
    ) -> None:
        self._warmup = int(warmup)
        self._max_graphs_per_segment = int(max_graphs_per_segment)
        self._clone_outputs = bool(clone_outputs)
        self._enabled = bool(enabled)
        self._share_pool = bool(share_pool)
        # Sharing a CUDA Graph memory pool across sequential segments is only
        # alias-safe when outputs are cloned out of the pool on replay; with
        # clone_outputs=False a caller may retain one segment's static output
        # buffer across another segment's replay.
        self._use_shared_pool = self._share_pool and self._clone_outputs
        self._shared_pool: Any | None = None
        self._segments: dict[str, InferenceCUDAGraphRunner] = {}
        self._skips: dict[str, int] = {}
        self._window_skips_baseline: dict[str, int] = {}
        self._frozen = False

    def _ensure_shared_pool(self) -> Any | None:
        """Lazily allocate one shared CUDA Graph pool handle across segments."""
        if not self._use_shared_pool or not self._enabled or not torch.cuda.is_available():
            return None
        if self._shared_pool is None:
            self._shared_pool = graph_pool_handle()
        return self._shared_pool

    @property
    def pool(self) -> Any | None:
        """Shared CUDA Graph memory pool handle, or None when separate/unused."""
        return self._shared_pool

    @property
    def share_pool(self) -> bool:
        """Whether registered segments share one CUDA Graph memory pool."""
        return self._use_shared_pool

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
            pool=self._ensure_shared_pool(),
        )
        if self._frozen:
            runner.freeze()
        self._segments[segment_name] = runner
        self._skips.setdefault(segment_name, 0)
        self._window_skips_baseline.setdefault(segment_name, 0)
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
        if self._use_shared_pool and runner._pool is None:
            shared_pool = self._ensure_shared_pool()
            if shared_pool is not None:
                runner._pool = shared_pool
                runner._owns_pool = False
        return runner(*args, **kwargs)

    def record_skip(self, name: str, count: int = 1) -> None:
        """Record that a cache controller bypassed segment ``name``."""
        segment_name = str(name).strip()
        if not segment_name:
            raise ValueError("piecewise graph segment name cannot be empty")
        if count < 0:
            raise ValueError("skip count must be non-negative")
        self._skips[segment_name] = self._skips.get(segment_name, 0) + int(count)
        self._window_skips_baseline.setdefault(segment_name, 0)

    def begin_request_window(self) -> None:
        """Reset per-request telemetry while preserving captured graphs."""
        for runner in self._segments.values():
            runner.begin_request_window()
        for name in set(self._segments) | set(self._skips):
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
        if self._use_shared_pool:
            # All segments sharing this pool have synchronized and dropped their
            # graphs above, so release the old handle to let the allocator
            # reclaim pool memory and bind registered segments to a fresh handle.
            self._shared_pool = None
            fresh_pool = self._ensure_shared_pool() if self._segments else None
            for runner in self._segments.values():
                runner._pool = fresh_pool
                runner._owns_pool = fresh_pool is None

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
            seg_win_capture = int(seg_report.get("capture", 0))
            seg_win_replay = int(seg_report.get("replay", 0))
            seg_win_eager = int(seg_report.get("eager", 0))
            seg_window = {
                "capture": seg_win_capture,
                "replay": seg_win_replay,
                "eager": seg_win_eager,
                "cache_skips": seg_win_skips,
            }
            segments[name] = {
                **seg_report,
                "cache_skips": seg_skips,
                "request_window": seg_window,
            }
            total_captures += seg_win_capture
            total_replays += seg_win_replay
            total_eager += seg_win_eager
            total_skips += seg_skips
            window_captures += seg_win_capture
            window_replays += seg_win_replay
            window_eager += seg_win_eager
            window_skips += seg_win_skips
        for name, seg_skips in self._skips.items():
            if name in self._segments:
                continue
            seg_win_skips = seg_skips - self._window_skips_baseline.get(name, 0)
            total_skips += seg_skips
            window_skips += seg_win_skips
        return {
            "mode": "piecewise-block-graph",
            "enabled": self._enabled,
            "frozen": self._frozen,
            "shared_pool": self._use_shared_pool,
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
