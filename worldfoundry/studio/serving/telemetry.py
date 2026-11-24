from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StudioServiceTelemetry:
    """Small thread-safe health snapshot for Studio's native servers."""

    label: str
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    active_requests: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_latency_ms: float = 0.0
    last_error: str = ""
    last_path: str = ""

    @contextmanager
    def track(self, path: str = "") -> Generator[None, None, None]:
        started = time.perf_counter()
        with self._lock:
            self.active_requests += 1
            self.total_requests += 1
            self.last_path = path
        try:
            yield
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as exc:
            with self._lock:
                self.failed_requests += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.last_latency_ms = latency_ms

    def snapshot(self, **extra: object) -> dict[str, object]:
        with self._lock:
            payload: dict[str, object] = {
                "status": "ok",
                "label": self.label,
                "uptime_s": round(time.time() - self.started_at, 3),
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "last_latency_ms": round(self.last_latency_ms, 3),
                "last_error": self.last_error,
                "last_path": self.last_path,
            }
        payload.update(extra)
        return payload
