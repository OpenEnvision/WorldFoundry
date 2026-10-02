from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.bagel_flow_unigrpo.objective import (  # noqa: E402
    bagel_flow_unigrpo_loss,
)


def test_ratio_norm_and_velocity_mse_match_the_explicit_formula() -> None:
    new_logp = torch.tensor([[-0.7, -1.1]], requires_grad=True)
    old_logp = torch.tensor([[-0.8, -1.0]])
    new_means = torch.tensor([[[0.3, -0.1], [0.2, 0.4]]], requires_grad=True)
    old_means = torch.tensor([[[0.1, -0.2], [0.3, 0.1]]])
    transition_std = torch.tensor([[[0.4], [0.25]]])
    sqrt_dt = torch.tensor([[0.5, 0.25]])
    advantages = torch.tensor([1.5])
    policy_velocity = torch.tensor(
        [[[0.6, -0.2], [0.1, 0.5]]],
        requires_grad=True,
    )
    reference_velocity = torch.tensor([[[0.2, -0.1], [0.0, 0.2]]])

    result = bagel_flow_unigrpo_loss(
        new_logp,
        old_logp,
        new_means,
        old_means,
        transition_std,
        sqrt_dt,
        advantages,
        policy_velocity,
        reference_velocity,
        clip_range=0.2,
        velocity_mse_weight=0.7,
        ratio_norm=True,
        grad_reweight=True,
    )

    mean_delta_sq = (new_means - old_means).square().mean(dim=2)
    expected_bias = mean_delta_sq / (2.0 * transition_std.squeeze(-1).square())
    expected_log_ratio = transition_std.squeeze(-1) * (new_logp - old_logp + expected_bias)
    expected_ratio = expected_log_ratio.exp()
    clipped = expected_ratio.clamp(0.8, 1.2)
    per_transition = torch.maximum(
        -advantages[:, None] * expected_ratio,
        -advantages[:, None] * clipped,
    )
    inverse_dt = sqrt_dt.square().reciprocal()
    weights = inverse_dt / inverse_dt.mean()
    expected_surrogate = (per_transition * weights).mean()
    expected_mse = (policy_velocity - reference_velocity).square().mean()

    torch.testing.assert_close(result.ratio_mean_bias, expected_bias)
    torch.testing.assert_close(result.ratio, expected_ratio)
    torch.testing.assert_close(result.surrogate_loss, expected_surrogate)
    torch.testing.assert_close(result.velocity_mse, expected_mse)
    torch.testing.assert_close(
        result.loss,
        expected_surrogate + 0.7 * expected_mse,
    )

    result.loss.backward()
    assert new_logp.grad is not None and bool((new_logp.grad != 0).any())
    assert new_means.grad is not None and bool((new_means.grad != 0).any())
    assert policy_velocity.grad is not None and bool((policy_velocity.grad != 0).any())
    assert reference_velocity.grad is None


def test_plain_mode_is_clipped_grpo_plus_velocity_mse() -> None:
    result = bagel_flow_unigrpo_loss(
        torch.tensor([[0.2, -0.2]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2, 1),
        None,
        torch.ones(1, 2, 1),
        None,
        torch.tensor([-1.0]),
        torch.tensor([[[2.0], [1.0]]]),
        torch.zeros(1, 2, 1),
        clip_range=0.1,
        velocity_mse_weight=0.5,
        ratio_norm=False,
        grad_reweight=False,
    )

    expected_ratio = torch.tensor([[0.2, -0.2]]).exp()
    expected_policy = torch.maximum(
        expected_ratio,
        expected_ratio.clamp(0.9, 1.1),
    ).mean()
    torch.testing.assert_close(result.surrogate_loss, expected_policy)
    torch.testing.assert_close(result.velocity_mse, torch.tensor(2.5))
    torch.testing.assert_close(result.loss, expected_policy + 1.25)
