from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier


def _race(worker, *, participants: int = 16):
    barrier = Barrier(participants)

    def synchronized():
        barrier.wait()
        return worker()

    with ThreadPoolExecutor(max_workers=participants) as pool:
        return list(pool.map(lambda _index: synchronized(), range(participants)))


def test_global_once_is_atomic_across_threads() -> None:
    from worldfoundry.core.utils.python.misc_utils import global_once

    name = f"test-global-once-{uuid.uuid4().hex}"
    results = _race(lambda: global_once(name))
    assert results.count(True) == 1


def test_global_n_times_is_atomic_across_threads() -> None:
    from worldfoundry.core.utils.python.misc_utils import global_n_times

    name = f"test-global-n-times-{uuid.uuid4().hex}"
    results = _race(lambda: global_n_times(name, 3))
    assert results.count(True) == 3


def test_attention_unavailable_backend_snapshots_are_thread_safe() -> None:
    import pytest

    pytest.importorskip("torch")
    from worldfoundry.core.attention.backends import dispatch

    dispatch.clear_attention_dispatch_cache()
    names = tuple(f"test-backend-{index}" for index in range(32))

    def mutate_and_snapshot(name: str) -> tuple[str, ...]:
        dispatch._quarantine_unavailable_backend(name)
        return dispatch._unavailable_backends_snapshot()

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            snapshots = list(pool.map(mutate_and_snapshot, names))
        assert snapshots
        assert dispatch._unavailable_backends_snapshot() == tuple(sorted(names))
    finally:
        dispatch.clear_attention_dispatch_cache()
