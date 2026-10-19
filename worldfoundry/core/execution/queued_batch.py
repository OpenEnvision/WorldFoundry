# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""FIFO compatible-request batching around one serial execution owner."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Thread, current_thread
from time import monotonic
from typing import Generic, TypeVar

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@dataclass
class _Entry(Generic[_Input, _Output]):
    value: _Input
    key: Hashable | None
    future: Future[_Output]
    queued_at: float


class QueuedBatchExecutor(Generic[_Input, _Output]):
    """Batch a compatible FIFO prefix and execute it on one owner thread.

    ``execute`` returns one result per input, in order. An exception or wrong
    result count fails the whole selected batch without retries. ``None`` keys
    are singletons. Futures can cancel queued work; running-work cancellation
    belongs to the execution owner's cooperative lifecycle.
    """

    def __init__(
        self,
        execute: Callable[[Sequence[_Input]], Sequence[_Output]],
        *,
        max_batch_size: int,
        max_delay_seconds: float = 0.005,
    ) -> None:
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not math.isfinite(max_delay_seconds) or max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be finite and non-negative")
        self._execute = execute
        self._max_batch_size = max_batch_size
        self._max_delay_seconds = max_delay_seconds
        self._pending: deque[_Entry[_Input, _Output]] = deque()
        self._condition = Condition()
        self._closed = False
        self._thread = Thread(target=self._run, name="worldfoundry-request-batch", daemon=True)
        self._thread.start()

    def submit(self, value: _Input, *, compatibility_key: Hashable | None) -> Future[_Output]:
        """Enqueue work; raise RuntimeError after close. No model imports occur."""
        future: Future[_Output] = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("request batch executor is closed")
            self._pending.append(_Entry(value, compatibility_key, future, monotonic()))
            self._condition.notify()
        return future

    def close(self, *, wait: bool = True) -> None:
        """Reject new work and cancel queued futures; let the active call finish."""
        with self._condition:
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for entry in pending:
            entry.future.cancel()
        if wait and current_thread() is not self._thread:
            self._thread.join()

    def _take_batch(self) -> list[_Entry[_Input, _Output]]:
        with self._condition:
            while not self._pending and not self._closed:
                self._condition.wait()
            if self._closed:
                return []
            first = self._pending.popleft()
            batch = [first]
            deadline = first.queued_at + self._max_delay_seconds
            while first.key is not None and len(batch) < self._max_batch_size and not self._closed:
                if self._pending:
                    candidate = self._pending[0]
                    if candidate.future.cancelled():
                        self._pending.popleft()
                    elif candidate.key != first.key:
                        break
                    else:
                        batch.append(self._pending.popleft())
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._closed:
                for entry in batch:
                    entry.future.cancel()
                return []
            return [entry for entry in batch if entry.future.set_running_or_notify_cancel()]

    def _run(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                with self._condition:
                    if self._closed:
                        return
                continue
            self._execute_batch(batch)
            del batch

    def _execute_batch(self, batch: Sequence[_Entry[_Input, _Output]]) -> None:
        # This frame ends before the owner waits again, releasing input/output
        # references even when callers release their futures while it is idle.
        try:
            outputs = tuple(self._execute(tuple(entry.value for entry in batch)))
            if len(outputs) != len(batch):
                raise RuntimeError("batch execution must return one result per request")
        except BaseException as error:
            for entry in batch:
                entry.future.set_exception(error)
        else:
            for entry, output in zip(batch, outputs):
                entry.future.set_result(output)
