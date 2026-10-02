# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive CUDA Graph dispatch policy.

Responsibility: pick eager fill, graph drain, or graph replay for each
AR branch, keyed by index. Capture is legal only after the sliding KV
cache is shape-stable (capacity divisible by ``chunk_size``).

This module is not a CUDA Graph runtime. Streams, memory pools, and
capture/replay live in the caller-supplied ``wrapper_factory`` because
resident models differ. Dynamic control flow (CFG branch, AR index)
must stay *outside* the captured region.

Public surface:
- :func:`cuda_graph_capture_ar_index` — first AR index at which KV
  capacity is chunk-aligned.
- :class:`CUDAGraphDispatch` — one wrapper pair per CFG branch plus
  ``select`` / ``reset`` / ``disable``.

The wrapper factory is injected so capture mechanics stay model-owned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Capture index — KV must be shape-stable before the first graph record
# ──────────────────────────────────────────────────────────────────────────


def cuda_graph_capture_ar_index(
    *,
    sink_size: int,
    window_size: int,
    chunk_size: int,
) -> int:
    """Return the first AR index at which a sliding KV cache is shape-stable."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    capacity = sink_size + window_size
    if capacity < 0:
        raise ValueError(f"sink_size + window_size must be non-negative, got {capacity}.")
    if capacity % chunk_size:
        raise ValueError(f"KV capacity {capacity} must be divisible by chunk_size {chunk_size} before graph capture.")
    return capacity // chunk_size


# ──────────────────────────────────────────────────────────────────────────
# Dispatch — two wrappers so CFG cond/uncond never share one captured graph
# ──────────────────────────────────────────────────────────────────────────


class CUDAGraphDispatch:
    """Select eager fill, graph drain, or graph replay for each AR branch."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        enabled: bool,
        capture_ar_index: int,
        warmup_iters: int,
        wrapper_factory: Callable[..., Any] | None = None,
        build: bool = True,
    ) -> None:
        """Require non-negative capture index/warmup; build wrappers only when enabled."""
        if capture_ar_index < 0:
            raise ValueError("capture_ar_index must be non-negative.")
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative.")
        self.fn = fn
        self.enabled = bool(enabled)
        self.capture_ar_index = int(capture_ar_index)
        self.warmup_iters = int(warmup_iters)
        self.wrapper_factory = wrapper_factory
        self._conditional: Any | None = None
        self._unconditional: Any | None = None
        if self.enabled and build:
            self.rebuild()

    @property
    def conditional(self) -> Any | None:
        """CFG-conditional graph wrapper, or ``None`` while disabled/unbuilt."""
        return self._conditional

    @property
    def unconditional(self) -> Any | None:
        """CFG-unconditional graph wrapper, or ``None`` while disabled/unbuilt."""
        return self._unconditional

    def rebuild(self, *, capture_ar_index: int | None = None) -> None:
        """Recreate both wrappers; refuse an enabled dispatch without a factory."""
        if capture_ar_index is not None:
            if capture_ar_index < 0:
                raise ValueError("capture_ar_index must be non-negative.")
            self.capture_ar_index = int(capture_ar_index)
        if not self.enabled:
            self._conditional = None
            self._unconditional = None
            return
        if self.wrapper_factory is None:
            raise RuntimeError("Enabled CUDA Graph dispatch requires a wrapper_factory.")
        self._conditional = self.wrapper_factory(
            self.fn,
            warmup_iters=self.warmup_iters,
        )
        self._unconditional = self.wrapper_factory(
            self.fn,
            warmup_iters=self.warmup_iters,
        )

    def disable(self, *, fn: Callable[..., Any] | None = None) -> None:
        """Drop wrappers so later ``select`` returns eager ``fn`` (optional swap)."""
        if fn is not None:
            self.fn = fn
        self.enabled = False
        self._conditional = None
        self._unconditional = None

    def reset(self) -> None:
        """Reset captured graphs, or rebuild if a wrapper was never constructed."""
        if not self.enabled:
            return
        if self._conditional is None or self._unconditional is None:
            self.rebuild()
            return
        self._conditional.reset()
        self._unconditional.reset()

    def select(self, autoregressive_index: int, *, unconditional: bool) -> Callable[..., Any]:
        """Return drain before capture index, the graph after, eager when disabled.

        ``drain`` is the wrapper's fill path while the KV window is still
        growing. After ``capture_ar_index`` the wrapper itself is the replay
        entry. Missing ``drain`` or a non-callable wrapper is a contract
        error, not a silent eager fallback — that would hide a broken factory.
        """
        if autoregressive_index < 0:
            raise ValueError("autoregressive_index must be non-negative.")
        if not self.enabled:
            return self.fn
        wrapper = self._unconditional if unconditional else self._conditional
        if wrapper is None:
            raise RuntimeError("CUDA Graph dispatch was selected before wrappers were built.")
        if autoregressive_index < self.capture_ar_index:
            drain = getattr(wrapper, "drain", None)
            if not callable(drain):
                raise TypeError("CUDA Graph wrapper must expose a callable drain method.")
            return drain
        if not callable(wrapper):
            raise TypeError("CUDA Graph wrapper must be callable after capture.")
        return wrapper


__all__ = ["CUDAGraphDispatch", "cuda_graph_capture_ar_index"]
