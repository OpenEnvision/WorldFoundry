"""TE-06: persist the process-global NumPy RNG without breaking old checkpoints.

``TrainingState`` already saved torch / Python RNGs.  NumPy's global
``RandomState`` was omitted, so a resume after ``set_seed_everywhere`` (or
any ``np.random`` consumer) could not replay.  The field is optional on
load so checkpoints written before this change still resume.
"""

from __future__ import annotations

import random

import pytest

numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointCompatibilityError,
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)


class _StatefulStub:
    def __init__(self, progress: TrainingProgress | None = None) -> None:
        self._progress = progress

    def state_dict(self) -> dict[str, object]:
        if self._progress is not None:
            return {"global_step": self._progress.optimizer_steps}
        return {"position": 0}

    def load_state_dict(self, state_dict: object) -> None:
        del state_dict


def _training_state() -> TrainingState:
    torch.manual_seed(11)
    random.seed(13)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress = TrainingProgress()
    return TrainingState(
        model=model,
        optimizer=optimizer,
        engine=_StatefulStub(progress),
        dataloader=_StatefulStub(),
        objective_generator=torch.Generator().manual_seed(23),
        progress=progress,
        identity={"recipe": "te06", "parallel_plan": {"backend": "single", "world_size": 1}},
    )


def test_numpy_rng_round_trips_through_dcp(tmp_path) -> None:
    numpy.random.seed(7)
    _ = numpy.random.rand(3)
    expected_next = numpy.random.rand(5).copy()
    numpy.random.seed(7)
    _ = numpy.random.rand(3)

    state = _training_state()
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)

    numpy.random.seed(999)
    restored = _training_state()
    manager.load(restored, artifact.path)
    got = numpy.random.rand(5)
    assert numpy.array_equal(got, expected_next)


def test_legacy_runtime_without_numpy_field_still_loads() -> None:
    numpy.random.seed(3)
    state = _training_state()
    payload = state.state_dict()
    runtime = dict(payload["runtime_by_rank"]["rank-00000000"])
    assert "numpy_random_state" in runtime
    del runtime["numpy_random_state"]
    payload["runtime_by_rank"] = {"rank-00000000": runtime}

    numpy.random.seed(4242)
    scrambled = numpy.random.rand(2).copy()
    numpy.random.seed(4242)
    state.load_state_dict(payload)
    # Old checkpoints omit the field, so the live NumPy stream is left alone.
    assert numpy.array_equal(numpy.random.rand(2), scrambled)


def test_unknown_runtime_field_is_still_rejected() -> None:
    state = _training_state()
    payload = state.state_dict()
    runtime = dict(payload["runtime_by_rank"]["rank-00000000"])
    runtime["not_a_real_field"] = 1
    payload["runtime_by_rank"] = {"rank-00000000": runtime}

    with pytest.raises(TrainingCheckpointCompatibilityError, match="runtime fields"):
        state.load_state_dict(payload)
