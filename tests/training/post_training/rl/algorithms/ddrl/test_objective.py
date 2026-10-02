from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.ddrl import (  # noqa: E402
    DDRL_ADVANTAGE_EPSILON,
    ddrl_group_advantages,
    ddrl_loss,
)


def test_group_advantage_default_and_constant_group_behavior() -> None:
    assert DDRL_ADVANTAGE_EPSILON == 1.0e-4

    result = ddrl_group_advantages(
        torch.tensor([1.0, 3.0, 2.0, 2.0]),
        ("first", "first", "second", "second"),
    )

    expected = torch.tensor([-1.0, 1.0]) / (2**0.5 + DDRL_ADVANTAGE_EPSILON)
    torch.testing.assert_close(result.advantages[:2], expected)
    torch.testing.assert_close(result.advantages[2:], torch.zeros(2), rtol=0, atol=0)


def test_ddrl_ratio_uses_latent_mean_without_variance_division_and_ppo_clip() -> None:
    next_latents = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    current_means = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    old_means = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    result = ddrl_loss(
        next_latents=next_latents,
        current_means=current_means,
        old_means=old_means,
        advantages=torch.tensor([1.0, -1.0]),
        clip_range=0.2,
    )

    torch.testing.assert_close(
        result.log_ratio_elements,
        torch.tensor([[-1.0, -3.0], [-1.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(result.log_ratio, torch.tensor([-2.0, 0.0]), rtol=0, atol=0)
    torch.testing.assert_close(result.ratio, torch.tensor([torch.exp(torch.tensor(-2.0)), 1.0]))
    expected_policy = (-torch.exp(torch.tensor(-2.0)) + 1.0) / 2
    torch.testing.assert_close(result.policy_loss, expected_policy)


def test_ddrl_total_loss_and_gradient_isolation() -> None:
    next_latents = torch.tensor([[0.5], [0.1]], requires_grad=True)
    current_means = torch.tensor([[0.2], [0.3]], requires_grad=True)
    old_means = torch.tensor([[0.4], [0.0]], requires_grad=True)
    reference_means = torch.tensor([[1.2], [1.3]], requires_grad=True)
    advantages = torch.tensor([0.5, -0.5], requires_grad=True)
    data_parameter = torch.tensor([2.0], requires_grad=True)

    result = ddrl_loss(
        next_latents=next_latents,
        current_means=current_means,
        old_means=old_means,
        advantages=advantages,
        clip_range=0.2,
        reference_means=reference_means,
        data_loss=data_parameter.square(),
        kl_beta=0.25,
        data_beta=0.5,
    )
    expected = result.policy_loss.detach() + 0.25 * 1.0 + 0.5 * 4.0
    torch.testing.assert_close(result.reference_kl, torch.tensor(1.0))
    torch.testing.assert_close(result.data_loss, torch.tensor(4.0), rtol=0, atol=0)
    torch.testing.assert_close(result.loss.detach(), expected)
    result.loss.backward()

    assert current_means.grad is not None
    assert data_parameter.grad is not None
    assert next_latents.grad is None
    assert old_means.grad is None
    assert reference_means.grad is None
    assert advantages.grad is None
