from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.adaptive_video import (  # noqa: E402
    AdaptiveRegressionEMA,
    adaptive_regression_weights,
    temporal_variance_regularization,
)


def test_temporal_regularization_uses_population_variance_and_hard_cutoff() -> None:
    moving = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
    result = temporal_variance_regularization(
        moving,
        epsilon=1.0e-6,
        cutoff=0.8,
    )
    torch.testing.assert_close(result.motion_metric, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        result.raw_loss,
        torch.tensor(-math.log(1.0 + 1.0e-6), dtype=torch.float64),
    )
    assert result.applied_loss.item() == 0.0

    collapsed = torch.tensor([[[0.0], [0.0]]], dtype=torch.float64)
    collapsed_result = temporal_variance_regularization(
        collapsed,
        epsilon=1.0e-6,
        cutoff=0.8,
    )
    torch.testing.assert_close(
        collapsed_result.raw_loss,
        torch.tensor(-math.log(1.0e-6), dtype=torch.float64),
    )
    torch.testing.assert_close(
        collapsed_result.applied_loss,
        collapsed_result.raw_loss,
    )


def test_adaptive_regression_first_observation_has_half_weight() -> None:
    losses = torch.tensor([2.0, 4.0])
    indices = torch.tensor([0, 1])
    result = adaptive_regression_weights(
        losses,
        indices,
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.bool),
        decay=0.95,
        sensitivity=3.0,
    )
    torch.testing.assert_close(result.weights, torch.full((2,), 0.5))
    torch.testing.assert_close(
        result.tentative_ema,
        torch.tensor([2.0, 4.0]),
    )
    assert result.observation.sample_counts.tolist() == [1, 1]
    torch.testing.assert_close(
        result.observation.loss_sums,
        torch.tensor([2.0, 4.0]),
    )


def test_adaptive_regression_updates_before_computing_weight() -> None:
    result = adaptive_regression_weights(
        torch.tensor([4.0]),
        torch.tensor([0]),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([True]),
        decay=0.95,
        sensitivity=3.0,
    )
    expected_ema = 0.95 * 2.0 + 0.05 * 4.0
    expected_weight = 1.0 - torch.sigmoid(torch.tensor(3.0 * (4.0 - expected_ema)))
    torch.testing.assert_close(
        result.tentative_ema,
        torch.tensor([expected_ema]),
    )
    torch.testing.assert_close(result.weights, expected_weight.reshape(1))


def test_slot_ema_commits_accumulated_microbatches_once() -> None:
    state = AdaptiveRegressionEMA(2, decay=0.95)
    first = adaptive_regression_weights(
        torch.tensor([2.0]),
        torch.tensor([0]),
        state.values,
        state.initialized,
        decay=state.decay,
        sensitivity=3.0,
    ).observation
    second = adaptive_regression_weights(
        torch.tensor([4.0]),
        torch.tensor([0]),
        state.values,
        state.initialized,
        decay=state.decay,
        sensitivity=3.0,
    ).observation
    state.commit((first, second))

    assert state.initialized.tolist() == [True, False]
    assert state.update_counts.tolist() == [1, 0]
    torch.testing.assert_close(state.values, torch.tensor([3.0, 0.0], dtype=torch.float64))

    restored = AdaptiveRegressionEMA(2, decay=0.95)
    restored.load_state_dict(state.state_dict())
    torch.testing.assert_close(restored.values, state.values)
    assert torch.equal(restored.initialized, state.initialized)
    assert torch.equal(restored.update_counts, state.update_counts)
