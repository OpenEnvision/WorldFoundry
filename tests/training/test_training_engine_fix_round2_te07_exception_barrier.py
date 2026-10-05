"""TE-07: exception paths must not enter checkpoint barriers.

A pending async checkpoint used to call ``finalize_staged_checkpoint`` from
the session exception handler.  That function always ``dist.barrier()``s; if
only some ranks raised, the handler hung until NCCL timed out.  The
``coordinate=False`` flag joins the local write and skips finalize.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.training.checkpoint.artifacts import OPTIONAL_TRAINING_STATE_NAMES
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import (
    PendingTrainingCheckpoint,
    wait_for_pending_checkpoints,
)

_SESSION = (
    Path(__file__).resolve().parents[2]
    / "worldfoundry/training/engine/sessions/single_device.py"
)
_OPTIONAL_PRESENCE = {name: False for name in OPTIONAL_TRAINING_STATE_NAMES}


class _RecordingFuture:
    def __init__(self) -> None:
        self.result_timeouts: list[float | None] = []

    def result(self, timeout: float | None = None) -> None:
        self.result_timeouts.append(timeout)

    def done(self) -> bool:
        return bool(self.result_timeouts)


class _RecordingFinalizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def finalize_staged_checkpoint(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(path=kwargs["final_path"], global_step=kwargs["global_step"])


def _pending(manager: _RecordingFinalizer, future: _RecordingFuture) -> PendingTrainingCheckpoint:
    return PendingTrainingCheckpoint(
        manager=manager,
        future=future,
        staging_path=Path("/tmp/staging"),
        final_path=Path("/tmp/final"),
        global_step=4,
        identity={"recipe": "te07"},
        gradient_accumulation_phase=0,
        world_size=2,
        staging_strategy="immutable-dtensor-local-shard-snapshot",
        optional_state_presence=_OPTIONAL_PRESENCE,
    )


def test_wait_without_coordinate_joins_write_and_skips_finalize() -> None:
    manager = _RecordingFinalizer()
    future = _RecordingFuture()
    pending = _pending(manager, future)

    assert pending.wait(timeout=1.5, coordinate=False) is None
    assert future.result_timeouts == [1.5]
    assert manager.calls == []
    assert pending.done()


def test_wait_default_still_finalizes() -> None:
    manager = _RecordingFinalizer()
    future = _RecordingFuture()
    pending = _pending(manager, future)

    artifact = pending.wait()
    assert artifact is not None
    assert artifact.global_step == 4
    assert len(manager.calls) == 1
    assert manager.calls[0]["coordinate"] is True
    assert future.result_timeouts == [None]


def test_session_checkpoint_helper_coordinates_on_normal_cleanup() -> None:
    manager = _RecordingFinalizer()
    future = _RecordingFuture()
    pending = [_pending(manager, future)]

    wait_for_pending_checkpoints(pending)

    assert pending == []
    assert len(manager.calls) == 1
    assert manager.calls[0]["coordinate"] is True
    assert future.result_timeouts == [None]


def test_session_checkpoint_helper_skips_finalize_while_exception_propagates() -> None:
    manager = _RecordingFinalizer()
    future = _RecordingFuture()
    pending = [_pending(manager, future)]

    with pytest.raises(RuntimeError, match="training failed"):
        try:
            raise RuntimeError("training failed")
        finally:
            wait_for_pending_checkpoints(pending)

    assert pending == []
    assert manager.calls == []
    assert future.result_timeouts == [None]


def test_finalize_without_coordinate_skips_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    barrier_calls: list[str] = []
    monkeypatch.setattr(
        "worldfoundry.training.checkpoint.checkpointer._barrier",
        lambda: barrier_calls.append("barrier"),
    )
    manager = TrainingCheckpointer(tmp_path / "checkpoints", clean_orphaned_staging=False)
    staging = manager.root / ".step-00000001.0123456789abcdef0123456789abcdef.staging"
    staging.mkdir()
    (staging / ".metadata").write_bytes(b"meta")
    (staging / "payload").write_bytes(b"data")

    artifact = manager.finalize_staged_checkpoint(
        staging_path=staging,
        final_path=manager.root / "step-00000001",
        global_step=1,
        identity={"recipe": "te07"},
        gradient_accumulation_phase=0,
        world_size=1,
        staging_strategy="synchronous-dcp",
        optional_state_presence=_OPTIONAL_PRESENCE,
        coordinate=False,
    )

    assert barrier_calls == []
    assert artifact.global_step == 1
    assert (manager.root / "step-00000001" / "_SUCCESS").is_file()


def test_finalize_default_still_coordinates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    barrier_calls: list[str] = []
    monkeypatch.setattr(
        "worldfoundry.training.checkpoint.checkpointer._barrier",
        lambda: barrier_calls.append("barrier"),
    )
    manager = TrainingCheckpointer(tmp_path / "checkpoints", clean_orphaned_staging=False)
    staging = manager.root / ".step-00000002.0123456789abcdef0123456789abcdef.staging"
    staging.mkdir()
    (staging / ".metadata").write_bytes(b"meta")
    (staging / "payload").write_bytes(b"data")

    manager.finalize_staged_checkpoint(
        staging_path=staging,
        final_path=manager.root / "step-00000002",
        global_step=2,
        identity={"recipe": "te07"},
        gradient_accumulation_phase=0,
        world_size=1,
        staging_strategy="synchronous-dcp",
        optional_state_presence=_OPTIONAL_PRESENCE,
    )

    assert barrier_calls == ["barrier", "barrier"]


def test_session_exception_handler_skips_checkpoint_coordination() -> None:
    tree = ast.parse(_SESSION.read_text(encoding="utf-8"), filename=str(_SESSION))
    run_fn = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SingleDeviceTrainingSession"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    )
    exception_calls = [
        node
        for handler in ast.walk(run_fn)
        if isinstance(handler, ast.ExceptHandler)
        and isinstance(handler.type, ast.Name)
        and handler.type.id == "Exception"
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "finish_pending_checkpoint"
    ]
    assert exception_calls, "session exception handler must join the pending checkpoint"
    for call in exception_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        coordinate = keywords.get("coordinate")
        assert isinstance(coordinate, ast.Constant) and coordinate.value is False
