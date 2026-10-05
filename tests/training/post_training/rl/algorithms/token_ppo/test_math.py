from __future__ import annotations

import math

import pytest
import torch

from worldfoundry.training.post_training.rl.algorithms.token_ppo.math import (
    packed_gae,
    scatter_terminal_rewards,
)
from worldfoundry.training.post_training.rl.algorithms.token_ppo.objective import (
    clipped_policy_losses,
    clipped_value_losses,
    token_ppo_loss,
)


def test_terminal_rewards_scatter_and_gae_reset_at_sequence_boundaries() -> None:
    lengths = torch.tensor([2, 3])
    rewards = scatter_terminal_rewards(torch.tensor([2.0, -1.0]), lengths)
    assert torch.equal(rewards, torch.tensor([0.0, 2.0, 0.0, 0.0, -1.0]))

    advantages, returns = packed_gae(
        rewards,
        torch.tensor([0.5, 0.25, 1.0, 0.5, 0.25]),
        lengths,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(
        advantages,
        torch.tensor([1.5, 1.75, -2.0, -1.5, -1.25]),
    )
    assert torch.allclose(returns, torch.tensor([2.0, 2.0, -1.0, -1.0, -1.0]))


def test_policy_and_value_clipping_match_direct_formulas() -> None:
    policy, ratio, clipped = clipped_policy_losses(
        torch.tensor([math.log(0.5), math.log(1.1), math.log(1.4)]),
        torch.zeros(3),
        torch.tensor([-2.0, 1.0, 1.0]),
        clip_range=0.2,
        clip_range_high=0.3,
    )
    assert torch.allclose(ratio, torch.tensor([0.5, 1.1, 1.4]))
    assert torch.allclose(policy, torch.tensor([1.6, -1.1, -1.3]))
    assert torch.equal(clipped, torch.tensor([True, False, True]))

    value, value_clipped = clipped_value_losses(
        torch.tensor([0.7, 0.1]),
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
        clip_range=0.2,
    )
    assert torch.allclose(value, torch.tensor([0.32, 0.005]))
    assert torch.equal(value_clipped, torch.tensor([True, False]))


def test_value_objective_keeps_official_fp32_math_for_bfloat16_critic() -> None:
    values = torch.tensor([0.333], dtype=torch.bfloat16, requires_grad=True)
    old_values = torch.tensor([0.123456], dtype=torch.float32)
    returns = torch.tensor([0.876543], dtype=torch.float32)

    loss, clipped = clipped_value_losses(
        values,
        old_values,
        returns,
        clip_range=0.2,
    )
    values_f = values.float()
    clipped_values = old_values + (values_f - old_values).clamp(-0.2, 0.2)
    expected = 0.5 * torch.maximum(
        (values_f - returns).square(),
        (clipped_values - returns).square(),
    )

    assert loss.dtype is torch.float32
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
    assert clipped.item()
    loss.sum().backward()
    assert values.grad is not None


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("token-mean", 13.0 / 3.0),
        ("seq-mean-token-mean", 5.5),
        ("seq-mean-token-sum-norm", 1.625),
    ],
)
def test_all_packed_token_aggregations_match_direct_formulas(
    reduction: str,
    expected: float,
) -> None:
    result = token_ppo_loss(
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([-1.0, -3.0, -9.0]),
        torch.zeros(3),
        torch.tensor([2, 1]),
        loss_mask=None,
        clip_range=0.0,
        clip_range_high=None,
        value_clip_range=0.0,
        reduction=reduction,
        horizon=4,
    )
    assert result.policy_loss.item() == pytest.approx(expected)
    assert result.value_loss.item() == 0.0
