from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.token_policy import (  # noqa: E402
    PackedTokenTrajectory,
    packed_token_offsets,
    slice_packed_token_trajectory,
)


def _ragged_trajectory() -> PackedTokenTrajectory:
    return PackedTokenTrajectory(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        tokens=torch.tensor([11, 12, 21, 22, 23, 31]),
        lengths=torch.tensor([2, 0, 3, 1]),
        old_log_probs=torch.linspace(-2.0, -0.5, 6),
        sampling_temperature=0.7,
        conditioning={
            "embeddings": torch.arange(8).reshape(4, 2),
            "labels": ("A", "B", "C", "D"),
            "nested": {"scores": [1.0, 2.0, 3.0, 4.0]},
            "shared": torch.tensor(7.0),
        },
    )


def test_packed_ragged_slices_preserve_sequence_and_token_alignment() -> None:
    trajectory = _ragged_trajectory()

    torch.testing.assert_close(
        packed_token_offsets(trajectory.lengths),
        torch.tensor([0, 2, 2, 5, 6]),
    )
    first = slice_packed_token_trajectory(trajectory, 0, 2)
    second = slice_packed_token_trajectory(trajectory, 2, 4)

    assert first.sample_ids == ("a", "b")
    assert first.token_count == 2
    assert first.sampling_temperature == 0.7
    torch.testing.assert_close(first.tokens, torch.tensor([11, 12]))
    torch.testing.assert_close(first.lengths, torch.tensor([2, 0]))
    torch.testing.assert_close(first.conditioning["embeddings"], torch.tensor([[0, 1], [2, 3]]))
    assert first.conditioning["labels"] == ("A", "B")
    assert first.conditioning["nested"]["scores"] == [1.0, 2.0]
    torch.testing.assert_close(first.conditioning["shared"], torch.tensor(7.0))

    assert second.token_start == 2
    assert second.token_end == 6
    torch.testing.assert_close(second.tokens, torch.tensor([21, 22, 23, 31]))


def test_all_empty_sequences_are_a_valid_packed_trajectory() -> None:
    trajectory = PackedTokenTrajectory(
        sample_ids=("a", "b"),
        group_ids=("prompt", "prompt"),
        policy_revision="policy-root",
        tokens=torch.empty(0, dtype=torch.long),
        lengths=torch.tensor([0, 0]),
        old_log_probs=torch.empty(0),
    )

    assert trajectory.batch_size == 2
    assert trajectory.token_count == 0
    empty_slice = slice_packed_token_trajectory(trajectory, 0, 2)
    assert empty_slice.batch_size == 2
    assert empty_slice.token_count == 0


def test_packed_trajectory_rejects_misaligned_or_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        PackedTokenTrajectory(
            sample_ids=("a", "b"),
            group_ids=("first", "second"),
            policy_revision="policy-root",
            tokens=torch.tensor([1, 2]),
            lengths=torch.tensor([1, 1]),
            old_log_probs=torch.zeros(2),
        )

    with pytest.raises(TypeError, match=r"sum\(lengths\)"):
        PackedTokenTrajectory(
            sample_ids=("a", "b"),
            group_ids=("prompt", "prompt"),
            policy_revision="policy-root",
            tokens=torch.tensor([1]),
            lengths=torch.tensor([1, 1]),
            old_log_probs=torch.zeros(2),
        )

    with pytest.raises(ValueError, match="sampling_temperature"):
        PackedTokenTrajectory(
            sample_ids=("a", "b"),
            group_ids=("prompt", "prompt"),
            policy_revision="policy-root",
            tokens=torch.tensor([1, 2]),
            lengths=torch.tensor([1, 1]),
            old_log_probs=torch.zeros(2),
            sampling_temperature=0.0,
        )
