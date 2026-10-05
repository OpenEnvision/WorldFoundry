from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError, Lock

from worldfoundry.studio.serving import workspace as workspace_app


def test_workspace_models_builds_cold_catalog_once_for_concurrent_callers(
    monkeypatch,
) -> None:
    callers_ready = Barrier(3)
    cold_builders = Barrier(2)
    count_lock = Lock()
    build_calls = 0

    def slow_catalog() -> tuple[()]:
        nonlocal build_calls
        with count_lock:
            build_calls += 1
        try:
            cold_builders.wait(timeout=0.5)
        except BrokenBarrierError:
            pass
        return ()

    def load_models() -> tuple[dict[str, object], ...]:
        callers_ready.wait(timeout=5)
        return workspace_app._workspace_models()

    monkeypatch.setattr(workspace_app, "_studio_catalog", slow_catalog)
    workspace_app._workspace_models_cached.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(load_models)
            second = executor.submit(load_models)
            callers_ready.wait(timeout=5)
            assert first.result(timeout=5) == ()
            assert second.result(timeout=5) == ()
        assert build_calls == 1
    finally:
        workspace_app._workspace_models_cached.cache_clear()
