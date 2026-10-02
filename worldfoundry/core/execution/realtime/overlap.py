# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small execution runners for optional realtime work overlap.

Responsibility: one-in-flight host-thread or CUDA-stream overlap with a
shared ``submit`` / ``wait`` / ``close`` surface, plus a synchronous
stand-in so callers do not branch on capability.

This module is not a thread pool, a CUDA Graph, or a general async
runtime. At most one piece of work may be pending; a second ``submit``
raises. Errors are stored and optionally re-raised on ``wait`` so a
realtime loop can decide when to surface them.

Public surface:
- :class:`SynchronousOverlap` — run inline; ``pending`` is always false.
- :class:`HostThreadOverlap` — one daemon thread; join on ``close``.
- :class:`CudaStreamOverlap` — side stream + completion event.

Errors propagate explicitly; CPU and CUDA runners share one lifecycle.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Sync stand-in — same API so realtime loops do not special-case "no overlap"
# ──────────────────────────────────────────────────────────────────────────


class SynchronousOverlap:
    """Execute submitted work immediately behind the overlap-runner API."""

    def __init__(self, *, name: str = "sync-overlap") -> None:
        """Name is telemetry-only; this runner never queues work."""
        self.name = name
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        """Always false: work finished inside ``submit``."""
        return False

    @property
    def last_error(self) -> BaseException | None:
        """Exception from the last ``submit``, if it raised."""
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        """Run ``work`` now; re-raise after recording so the caller still sees it."""
        del name
        self._error = None
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
            raise

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        """No-op completion; optionally re-raise a stored ``submit`` error."""
        del timeout_s
        if raise_error and self._error is not None:
            raise self._error
        return True

    def close(self, *, wait: bool = True) -> None:
        """No resources to join; kept so callers can treat all runners uniformly."""
        del wait


# ──────────────────────────────────────────────────────────────────────────
# Host thread — one in-flight callback; a second submit is a caller bug
# ──────────────────────────────────────────────────────────────────────────


class HostThreadOverlap:
    """Execute at most one deferred callback on a daemon host thread."""

    def __init__(self, *, name: str = "host-overlap", daemon: bool = True) -> None:
        """Start idle (``_done`` set) so the first ``submit`` is not blocked."""
        self.name = name
        self._daemon = daemon
        self._done = threading.Event()
        self._done.set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        """True while the worker thread has not set the completion event."""
        return not self._done.is_set()

    @property
    def last_error(self) -> BaseException | None:
        """Exception captured on the worker; ``None`` until the next ``submit``."""
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        """Start one daemon thread; refuse if the previous job is still running."""
        with self._lock:
            if self.pending:
                raise RuntimeError(f"{self.name} already has pending overlap work.")
            self._error = None
            self._done.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(work,),
                name=name or self.name,
                daemon=self._daemon,
            )
            self._thread.start()

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        """Block until the worker finishes or ``timeout_s`` elapses (``None`` = forever)."""
        if timeout_s is not None and timeout_s < 0.0:
            raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
        completed = self._done.wait(timeout=timeout_s)
        if completed and raise_error and self._error is not None:
            raise self._error
        return completed

    def close(self, *, wait: bool = True) -> None:
        """Optionally join the last worker; does not cancel in-flight work."""
        thread = self._thread
        if wait and thread is not None:
            thread.join()

    def _run(self, work: Callable[[], Any]) -> None:
        """Isolate worker exceptions so they surface on ``wait``, not as a crash."""
        try:
            result = work()
            del result
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()


# ──────────────────────────────────────────────────────────────────────────
# CUDA stream — enqueue under a side stream; poll the event for timeouts
# ──────────────────────────────────────────────────────────────────────────


class CudaStreamOverlap:
    """Enqueue work on a side CUDA stream and publish a completion event."""

    def __init__(
        self,
        *,
        name: str = "cuda-stream-overlap",
        device: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        """Require a live CUDA device; ``torch_module`` is for tests without a GPU import."""
        self.name = name
        self._torch = torch_module if torch_module is not None else importlib.import_module("torch")
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA stream overlap requires an available CUDA device.")
        self._device = self._torch.device(device) if device is not None else None
        self._stream = self._torch.cuda.Stream(device=self._device)
        self._event: Any | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        """True while the recorded event has not yet reported complete."""
        return self._event is not None and not self._event.query()

    @property
    def last_error(self) -> BaseException | None:
        """Exception from the last ``submit`` enqueue, if it raised on the host."""
        return self._error

    def submit(self, work: Callable[[], Any], *, name: str | None = None) -> None:
        """Enqueue on the side stream; refuse if the previous event is still live."""
        del name
        if not self.wait(timeout_s=0.0):
            raise RuntimeError(f"{self.name} already has pending overlap work.")
        self._error = None
        try:
            device_context = nullcontext() if self._device is None else self._torch.cuda.device(self._device)
            with device_context, self._torch.cuda.stream(self._stream):
                result = work()
                del result
                event = self._torch.cuda.Event()
                event.record(self._stream)
        except BaseException as exc:
            self._error = exc
            raise
        self._event = event

    def wait(self, *, timeout_s: float | None = None, raise_error: bool = False) -> bool:
        """Sync the event (or poll with a deadline); ``timeout_s=None`` waits forever."""
        event = self._event
        if event is None:
            completed = True
        elif timeout_s is None:
            event.synchronize()
            completed = True
        else:
            completed = _wait_for_cuda_event(event, timeout_s=timeout_s)
        if completed and raise_error and self._error is not None:
            raise self._error
        return completed

    def close(self, *, wait: bool = True) -> None:
        """Optionally drain the stream and re-raise a stored enqueue error."""
        if wait:
            self.wait(raise_error=True)


def _wait_for_cuda_event(event: Any, *, timeout_s: float) -> bool:
    """Poll ``event.query`` with a 1 ms sleep so a deadline does not busy-spin."""
    if timeout_s < 0.0:
        raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


__all__ = ["CudaStreamOverlap", "HostThreadOverlap", "SynchronousOverlap"]
