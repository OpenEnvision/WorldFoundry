# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""FIFO ownership, cancellation, failure fanout and shutdown for queued batches."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from concurrent.futures import CancelledError
from threading import Event

import pytest

from worldfoundry.core.execution.queued_batch import QueuedBatchExecutor

pytestmark = pytest.mark.unit


def test_fifo_prefix_batches_without_crossing_incompatible_work() -> None:
    started, release = Event(), Event()
    calls: list[tuple[int, ...]] = []

    def execute(values: Sequence[int]) -> Sequence[int]:
        calls.append(tuple(values))
        if values == (0,):
            started.set()
            assert release.wait(5)
        return [value * 10 for value in values]

    owner = QueuedBatchExecutor(execute, max_batch_size=2, max_delay_seconds=0)
    try:
        first = owner.submit(0, compatibility_key=None)
        assert started.wait(5)
        futures = [
            owner.submit(value, compatibility_key=key)
            for value, key in ((1, "a"), (2, "a"), (3, "b"), (4, "a"), (5, "a"), (6, "a"))
        ]
        release.set()
        assert first.result(5) == 0
        assert [future.result(5) for future in futures] == [10, 20, 30, 40, 50, 60]
        assert calls == [(0,), (1, 2), (3,), (4, 5), (6,)]
    finally:
        release.set()
        owner.close()


def test_delay_collects_idle_compatible_submissions() -> None:
    calls: list[tuple[int, ...]] = []

    def execute(values: Sequence[int]) -> Sequence[int]:
        calls.append(tuple(values))
        return values

    owner = QueuedBatchExecutor(execute, max_batch_size=2, max_delay_seconds=1)
    try:
        a = owner.submit(1, compatibility_key="same")
        b = owner.submit(2, compatibility_key="same")
        assert a.result(5) == 1 and b.result(5) == 2
        assert calls == [(1, 2)]
    finally:
        owner.close()


def test_queued_cancellation_never_executes_cancelled_input() -> None:
    started, release = Event(), Event()
    seen: list[int] = []

    def execute(values: Sequence[int]) -> Sequence[int]:
        seen.extend(values)
        if values == (0,):
            started.set()
            assert release.wait(5)
        return values

    owner = QueuedBatchExecutor(execute, max_batch_size=3, max_delay_seconds=0)
    try:
        active = owner.submit(0, compatibility_key=None)
        assert started.wait(5)
        cancelled = owner.submit(1, compatibility_key="a")
        kept = owner.submit(2, compatibility_key="a")
        assert cancelled.cancel()
        release.set()
        assert active.result(5) == 0 and kept.result(5) == 2
        with pytest.raises(CancelledError):
            cancelled.result()
        assert seen == [0, 2]
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize("wrong_count", [False, True])
def test_batch_failure_fans_out_once_and_next_batch_runs(wrong_count: bool) -> None:
    calls: list[tuple[int, ...]] = []

    def execute(values: Sequence[int]) -> Sequence[int]:
        calls.append(tuple(values))
        if values == (1, 2):
            if wrong_count:
                return (1,)
            raise RuntimeError("compute failed")
        return values

    owner = QueuedBatchExecutor(execute, max_batch_size=2, max_delay_seconds=1)
    try:
        first = owner.submit(1, compatibility_key="a")
        second = owner.submit(2, compatibility_key="a")
        for future in (first, second):
            with pytest.raises(RuntimeError, match="one result|compute failed"):
                future.result(5)
        assert owner.submit(3, compatibility_key=None).result(5) == 3
        assert calls == [(1, 2), (3,)]
    finally:
        owner.close()


def test_close_cancels_pending_without_terminating_active_execution() -> None:
    started, release = Event(), Event()

    def execute(values: Sequence[int]) -> Sequence[int]:
        started.set()
        assert release.wait(5)
        return values

    owner = QueuedBatchExecutor(execute, max_batch_size=2, max_delay_seconds=0)
    try:
        active = owner.submit(1, compatibility_key=None)
        assert started.wait(5)
        pending = owner.submit(2, compatibility_key=None)
        owner.close(wait=False)
        assert pending.cancelled()
        assert not active.done()
        with pytest.raises(RuntimeError, match="closed"):
            owner.submit(3, compatibility_key=None)
        release.set()
        assert active.result(5) == 1
    finally:
        release.set()
        owner.close()


def test_completed_values_are_released_while_owner_remains_live() -> None:
    class Value:
        pass

    started, release = Event(), Event()

    def execute(values: Sequence[object]) -> Sequence[object]:
        if values == (None,):
            started.set()
            assert release.wait(5)
        return values

    owner = QueuedBatchExecutor(execute, max_batch_size=1)
    try:
        value = Value()
        reference = weakref.ref(value)
        future = owner.submit(value, compatibility_key=None)
        assert future.result(5) is value
        del value, future
        sentinel = owner.submit(None, compatibility_key=None)
        assert started.wait(5)
        gc.collect()
        assert reference() is None
        release.set()
        sentinel.result(5)
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize("size,delay", [(0, 0), (2, -1), (2, float("nan"))])
def test_invalid_queue_limits_are_rejected(size: int, delay: float) -> None:
    with pytest.raises(ValueError):
        QueuedBatchExecutor(lambda values: values, max_batch_size=size, max_delay_seconds=delay)
