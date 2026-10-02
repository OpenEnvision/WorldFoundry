"""Regression tests for SA-7 (RUF006): fire-and-forget cleanup tasks must be anchored.

The event loop only keeps weak references to tasks, so a bare
``asyncio.create_task(self.close_active())`` inside the WebRTC channel
``on_close`` callback could be garbage collected mid-flight, intermittently
dropping session cleanup.  ``RealtimePeerManager`` now anchors such tasks in
``self._background_tasks`` and discards them on completion.
"""

from __future__ import annotations

import ast
import asyncio
import types
from pathlib import Path

import pytest

realtime_backend = pytest.importorskip(
    "worldfoundry.studio.visualization.backends.world_realtime"
)


def _manager() -> "realtime_backend.RealtimePeerManager":
    return realtime_backend.RealtimePeerManager(
        runtime=types.SimpleNamespace(),
        fps=8,
        chunk_frames=4,
    )


def test_spawn_background_task_holds_reference_until_completion() -> None:
    async def scenario() -> None:
        manager = _manager()
        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        async def cleanup() -> None:
            started.set()
            await release.wait()
            completed.set()

        task = manager._spawn_background_task(cleanup(), name="test-cleanup")
        assert task in manager._background_tasks
        await started.wait()
        # While the task is in flight the manager must hold a strong reference.
        assert task in manager._background_tasks
        release.set()
        await task
        assert completed.is_set()
        # The done callback discards the reference once the task finishes.
        await asyncio.sleep(0)
        assert task not in manager._background_tasks

    asyncio.run(scenario())


def test_spawn_background_task_discards_failed_tasks_too() -> None:
    async def scenario() -> None:
        manager = _manager()

        async def boom() -> None:
            raise RuntimeError("expected test failure")

        task = manager._spawn_background_task(boom(), name="test-boom")
        with pytest.raises(RuntimeError, match="expected test failure"):
            await task
        await asyncio.sleep(0)
        assert task not in manager._background_tasks

    asyncio.run(scenario())


def test_no_unanchored_create_task_expression_statements() -> None:
    """Static regression: no bare ``asyncio.create_task(...)`` statements (RUF006)."""
    source_path = Path(realtime_backend.__file__)
    tree = ast.parse(source_path.read_text())
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "create_task"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "asyncio"
    ]
    assert offenders == [], (
        f"Unanchored asyncio.create_task expression statements at lines {offenders}; "
        "use RealtimePeerManager._spawn_background_task or store the task reference."
    )
