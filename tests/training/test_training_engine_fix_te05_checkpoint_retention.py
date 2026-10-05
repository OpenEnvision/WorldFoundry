"""TE-05 regression: checkpoint retention and orphaned-staging cleanup.

``TrainingCheckpointer`` historically kept every committed checkpoint and
never removed ``.step-*.<token>.staging`` residue left by crashed writes.
The fix adds:

* opt-in ``keep_last`` retention (constructor argument or the
  ``WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST`` environment variable);
  the default remains "keep everything",
* default-on orphaned-staging cleanup before the first save of an
  instance (disable via ``clean_orphaned_staging=False`` or
  ``WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING=0``).

These tests run real synchronous/asynchronous DCP saves on CPU with
duck-typed engine/dataloader stand-ins, so no torchdata is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    PendingTrainingCheckpoint,
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)

_ORPHAN_TOKEN = "0123456789abcdef0123456789abcdef"


class _StatefulStub:
    def __init__(self, progress: TrainingProgress | None = None) -> None:
        self._progress = progress

    def state_dict(self) -> dict[str, object]:
        if self._progress is not None:
            return {"global_step": self._progress.optimizer_steps}
        return {"position": 0}

    def load_state_dict(self, state_dict: object) -> None:
        del state_dict


def _training_state() -> tuple[TrainingState, TrainingProgress]:
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress = TrainingProgress()
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        engine=_StatefulStub(progress),
        dataloader=_StatefulStub(),
        objective_generator=torch.Generator().manual_seed(23),
        progress=progress,
        identity={"recipe": "te05", "parallel_plan": {"backend": "single", "world_size": 1}},
    )
    return state, progress


def _advance_to(progress: TrainingProgress, step: int) -> None:
    while progress.optimizer_steps < step:
        progress.record_step(microbatches=1, samples=1, latent_tokens=1)


def _committed_steps(root: Path) -> list[str]:
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir() and entry.name.startswith("step-"))


def test_default_behavior_keeps_every_checkpoint(tmp_path: Path) -> None:
    state, progress = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    assert manager.keep_last is None

    for step in (1, 2, 3):
        _advance_to(progress, step)
        manager.save(state)

    assert _committed_steps(manager.root) == ["step-00000001", "step-00000002", "step-00000003"]


def test_keep_last_removes_only_oldest_committed_checkpoints(tmp_path: Path) -> None:
    state, progress = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints", keep_last=2)
    unmanaged = manager.root / "step-notdigits"
    unmanaged.mkdir()
    uncommitted = manager.root / "step-99999999"
    uncommitted.mkdir()

    for step in (1, 2, 3, 4):
        _advance_to(progress, step)
        artifact = manager.save(state)

    assert _committed_steps(manager.root) == [
        "step-00000003",
        "step-00000004",
        "step-99999999",
        "step-notdigits",
    ]
    # Survivors are still valid committed checkpoints and latest points at the newest.
    assert manager.inspect(artifact.path).global_step == 4
    assert manager.load(_training_state()[0], "latest").global_step == 4
    # Unrecognized and uncommitted directories are never touched.
    assert unmanaged.is_dir()
    assert uncommitted.is_dir()


def test_keep_last_applies_after_async_finalize(tmp_path: Path) -> None:
    state, progress = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints", keep_last=1)

    _advance_to(progress, 1)
    manager.save(state)
    _advance_to(progress, 2)
    pending = manager.save(state, asynchronous=True)
    assert isinstance(pending, PendingTrainingCheckpoint)
    # The stale checkpoint must survive until the new one is fully committed.
    assert "step-00000001" in _committed_steps(manager.root)
    artifact = pending.wait()

    assert artifact.global_step == 2
    assert _committed_steps(manager.root) == ["step-00000002"]


def test_keep_last_never_deletes_the_just_committed_checkpoint(tmp_path: Path) -> None:
    # Resuming from an older step can leave higher-numbered checkpoints in the
    # root; the just-committed checkpoint must survive even when keep_last
    # would rank it below them.
    seed_state, seed_progress = _training_state()
    seed_manager = TrainingCheckpointer(tmp_path / "checkpoints")
    _advance_to(seed_progress, 5)
    seed_manager.save(seed_state)

    state, progress = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints", keep_last=1)
    _advance_to(progress, 3)
    artifact = manager.save(state)

    assert artifact.global_step == 3
    remaining = _committed_steps(manager.root)
    assert "step-00000003" in remaining
    assert manager.inspect(manager.root / "step-00000003").global_step == 3


def test_keep_last_env_variable_enables_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST", "1")
    state, progress = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    assert manager.keep_last == 1

    for step in (1, 2):
        _advance_to(progress, step)
        manager.save(state)

    assert _committed_steps(manager.root) == ["step-00000002"]


@pytest.mark.parametrize("raw", ["0", "-3", "two"])
def test_invalid_keep_last_configuration_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST", raw)
    with pytest.raises(ValueError, match="KEEP_LAST"):
        TrainingCheckpointer(tmp_path / "env-checkpoints")
    monkeypatch.delenv("WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST")
    with pytest.raises(ValueError, match="keep_last"):
        TrainingCheckpointer(tmp_path / "arg-checkpoints", keep_last=0)


def test_orphaned_staging_is_removed_before_the_first_save(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    orphan = root / f".step-00000009.{_ORPHAN_TOKEN}.staging"
    orphan.mkdir(parents=True)
    (orphan / "partial.distcp").write_bytes(b"junk")
    unrelated_dir = root / ".not-a-staging-dir"
    unrelated_dir.mkdir()
    unrelated_file = root / f".step-00000008.{_ORPHAN_TOKEN}.staging.txt"
    unrelated_file.write_text("keep me")

    state, progress = _training_state()
    manager = TrainingCheckpointer(root)
    # Construction alone must not delete anything: a load-only checkpointer
    # may point at another live run's root.
    assert orphan.is_dir()

    _advance_to(progress, 1)
    artifact = manager.save(state)

    assert not orphan.exists()
    assert unrelated_dir.is_dir()
    assert unrelated_file.is_file()
    assert artifact.global_step == 1


def test_orphaned_staging_cleanup_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkpoints"
    orphan = root / f".step-00000009.{_ORPHAN_TOKEN}.staging"
    orphan.mkdir(parents=True)

    state, progress = _training_state()
    manager = TrainingCheckpointer(root, clean_orphaned_staging=False)
    _advance_to(progress, 1)
    manager.save(state)
    assert orphan.is_dir()

    monkeypatch.setenv("WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING", "0")
    env_state, env_progress = _training_state()
    env_manager = TrainingCheckpointer(tmp_path / "env-checkpoints")
    env_orphan = env_manager.root / f".step-00000009.{_ORPHAN_TOKEN}.staging"
    env_orphan.mkdir()
    _advance_to(env_progress, 1)
    env_manager.save(env_state)
    assert env_orphan.is_dir()

    monkeypatch.setenv("WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING", "maybe")
    with pytest.raises(ValueError, match="CLEAN_STAGING"):
        TrainingCheckpointer(tmp_path / "bad-env-checkpoints")
