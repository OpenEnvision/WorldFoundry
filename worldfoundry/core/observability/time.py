"""Time helpers shared by runtime and evaluation code.

Three small primitives used by inference runners and eval harnesses:

- :func:`utc_now_iso` — timezone-aware UTC timestamps for artifacts
  and JSONL events (``Z`` suffix, not ``+00:00``).
- :class:`BenchmarkTimes` — split a measured interval into model
  invocation vs. host overhead (I/O, encode, post-process).
- :class:`CudaSyncTimer` — optional, env-gated wall/GPU timer that
  works as a context manager or decorator. Timing is off unless
  ``SYNC_TIMER=1`` (or a caller-supplied ``flag_env``) so production
  generate loops do not pay for ``cudaEvent`` + synchronize.

CUDA events are used only when ``torch.cuda.is_available()``; otherwise
the timer falls back to :func:`time.perf_counter`. The torch import is
deferred until a timer is actually enabled.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Callable


@dataclass
class BenchmarkTimes:
    """Seconds spent in the model vs. the full measured interval.

    Attributes:
        model_invocation: Time inside the generate / forward call.
        total: End-to-end time including host-side work.

    ``overhead`` is ``total - model_invocation`` and may be slightly
    negative if the two clocks are not nested the same way.
    """

    model_invocation: float = 0.0
    total: float = 0.0

    @property
    def overhead(self) -> float:
        """Host-side time that is not the model invocation itself."""
        return self.total - self.model_invocation


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form with a ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CudaSyncTimer:
    """Optional CUDA-synchronized timer usable as a context manager or decorator.

    Construction is cheap and has no torch dependency. ``__enter__``
    reads *flag_env* (default ``SYNC_TIMER``) and no-ops unless it is
    ``"1"``. When enabled it records ``elapsed_ms`` and, if *name* is
    set, logs ``"{name} takes X.XXXXs"``.

    As a decorator, each call wraps the function in ``with self:``.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        flag_env: str = "SYNC_TIMER",
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        """Store the log label, enable-flag name, and optional logger."""
        self.name = name
        self.flag_env = flag_env
        self.log_fn = log_fn
        self.elapsed_ms = 0.0
        self._enabled = False
        self._using_cuda = False

    def __enter__(self):
        """Arm CUDA events only when *flag_env* is ``1``; otherwise this is a no-op."""
        self._enabled = os.environ.get(self.flag_env, "0") == "1"
        if not self._enabled:
            return None
        import torch

        self._using_cuda = torch.cuda.is_available()
        if self._using_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self._wall_start = time.perf_counter()
        return lambda: self.elapsed_ms

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Synchronize CUDA (or stop the wall clock) and optionally log ``elapsed_ms``."""
        if not self._enabled:
            return None
        if self._using_cuda:
            import torch

            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = float(self.start.elapsed_time(self.end))
        else:
            self.elapsed_ms = (time.perf_counter() - self._wall_start) * 1000.0
        if self.name is not None:
            message = f"{self.name} takes {self.elapsed_ms / 1000:.4f}s"
            if self.log_fn is not None:
                self.log_fn(message)
            else:
                logging.getLogger(__name__).info(message)
        return None

    def __call__(self, func):
        """Wrap *func* so each call is timed with this same timer instance."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Run *func* inside ``with self`` so decorator and context share state."""
            with self:
                return func(*args, **kwargs)

        return wrapper


__all__ = ["BenchmarkTimes", "CudaSyncTimer", "utc_now_iso"]
