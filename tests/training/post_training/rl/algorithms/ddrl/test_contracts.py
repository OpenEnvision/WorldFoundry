from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.ddrl import (  # noqa: E402
    DDRLRolloutBatch,
    DDRLTrajectory,
)


def _trajectory(**overrides) -> DDRLTrajectory:
    values = {
        "trajectory_id": "trajectory-1",
        "sample_ids": ("a", "b", "c", "d"),
        "group_ids": ("first", "first", "second", "second"),
        "train_on": (1, 3),
        "next_latents": torch.zeros(4, 2, 1),
        "old_means": torch.ones(4, 2, 1),
        "reference_means": torch.full((4, 2, 1), 0.5),
        "terminal_latents": torch.zeros(4, 1),
        "replay_inputs": {"noisy": torch.ones(4, 2, 1)},
    }
    values.update(overrides)
    return DDRLTrajectory(**values)


def test_rollout_and_trajectory_require_complete_groups_and_selected_step_shapes() -> None:
    batch = DDRLRolloutBatch(
        batch_id="batch-1",
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        model_inputs={"latents": torch.zeros(4, 1)},
    )
    trajectory = _trajectory()

    assert batch.batch_size == trajectory.batch_size == 4
    assert trajectory.step_count == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        _trajectory(train_on=(3, 1))
    with pytest.raises(TypeError, match="not bool"):
        _trajectory(train_on=(False, True))
    with pytest.raises(ValueError, match=r"\[B,K"):
        _trajectory(next_latents=torch.zeros(4, 3, 1))


def test_trajectory_rejects_differentiable_behavior_and_reference_anchors() -> None:
    with pytest.raises(ValueError, match="frozen rollout tensor"):
        _trajectory(old_means=torch.ones(4, 2, 1, requires_grad=True))
    with pytest.raises(ValueError, match="frozen rollout tensor"):
        _trajectory(reference_means=torch.ones(4, 2, 1, requires_grad=True))
