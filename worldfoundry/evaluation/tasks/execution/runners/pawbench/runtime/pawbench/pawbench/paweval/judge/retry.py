"""Bounded retry helper for judge calls."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    call: Callable[[], T],
    *,
    attempts: int = 3,
    retry_status: Callable[[T], bool] | None = None,
    sleep_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    max_sleep_seconds: float = 30.0,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be >= 1")
    if max_sleep_seconds < 0:
        raise ValueError("max_sleep_seconds must be >= 0")
    last: T | None = None
    delay = min(sleep_seconds, max_sleep_seconds)
    for attempt in range(attempts):
        last = call()
        if retry_status is None or not retry_status(last):
            return last
        if delay > 0 and attempt < attempts - 1:
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_sleep_seconds)
    if last is None:
        raise RuntimeError("retry_call exhausted without recording a result")
    return last
