from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.grpo_guard import (  # noqa: E402
    grpo_guard_policy_loss,
)


def test_grpo_guard_matches_the_batched_formula_and_keeps_bias_per_transition() -> None:
    new_log_probs = torch.tensor(
        [[0.10, -0.20], [0.05, 0.00]],
        dtype=torch.float32,
        requires_grad=True,
    )
    old_log_probs = torch.tensor(
        [[0.00, -0.10], [0.10, -0.05]],
        dtype=torch.float32,
        requires_grad=True,
    )
    new_means = torch.tensor(
        [
            [[0.20, 0.20], [0.10, 0.30]],
            [[0.00, 0.40], [0.20, 0.00]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    old_means = torch.zeros_like(new_means, requires_grad=True)
    std_dev_t = torch.tensor([[[0.4], [0.6]], [[0.8], [1.0]]])
    sqrt_dt = torch.tensor([[0.50, 0.25], [0.75, 0.50]])
    advantages = torch.tensor([-2.0, -9.0])

    result = grpo_guard_policy_loss(
        new_log_probs,
        old_log_probs,
        new_means,
        old_means,
        std_dev_t,
        sqrt_dt,
        advantages,
        clip_range=0.2,
        advantage_clip_max=1.5,
    )

    expected_std_mean = std_dev_t.mean()
    expected_sqrt_dt_mean = sqrt_dt.mean()
    expected_scale = expected_std_mean * expected_sqrt_dt_mean
    expected_mean_diff_sq = (new_means - old_means).square().mean(dim=2)
    expected_bias = expected_mean_diff_sq / (2.0 * expected_scale.square())
    expected_log_ratio = new_log_probs - old_log_probs
    expected_ratio = torch.exp((expected_log_ratio + expected_bias) * expected_scale)
    expected_advantage = advantages.clamp(-1.5, 1.5).reshape(-1, 1)
    expected_per_transition = torch.maximum(
        -expected_advantage * expected_ratio,
        -expected_advantage * expected_ratio.clamp(0.8, 1.2),
    )
    expected_loss = expected_per_transition.mean() / expected_sqrt_dt_mean.square()

    assert result.ratio_mean_bias.shape == (2, 2)
    assert result.ratio_mean_bias.requires_grad
    assert torch.unique(result.ratio_mean_bias.detach()).numel() > 1
    torch.testing.assert_close(result.scale, expected_scale)
    torch.testing.assert_close(result.sqrt_dt_mean, expected_sqrt_dt_mean)
    torch.testing.assert_close(result.ratio_mean_bias, expected_bias)
    torch.testing.assert_close(result.ratio, expected_ratio)
    torch.testing.assert_close(result.per_transition, expected_per_transition)
    torch.testing.assert_close(result.loss, expected_loss)

    result.loss.backward()

    assert new_means.grad is not None
    assert bool((new_means.grad != 0).any())
    assert new_log_probs.grad is not None
    assert bool((new_log_probs.grad != 0).any())
    assert old_means.grad is None
    assert old_log_probs.grad is None


def test_grpo_guard_step_mask_changes_only_the_final_transition_reduction() -> None:
    new_log_probs = torch.tensor([[0.0, 0.2], [-0.1, 0.1]])
    old_log_probs = torch.zeros_like(new_log_probs)
    new_means = torch.tensor(
        [[[0.1], [0.2]], [[0.3], [0.4]]],
        requires_grad=True,
    )
    old_means = torch.zeros_like(new_means)
    std_dev_t = torch.full((2, 2, 1), 0.5)
    sqrt_dt = torch.tensor([[0.5, 0.25], [0.5, 0.25]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = grpo_guard_policy_loss(
        new_log_probs,
        old_log_probs,
        new_means,
        old_means,
        std_dev_t,
        sqrt_dt,
        torch.tensor([-1.0, -1.0]),
        clip_range=0.2,
        advantage_clip_max=5.0,
        step_mask=mask,
    )

    expected = (result.per_transition * mask).sum() / mask.sum()
    expected = expected / sqrt_dt.mean().square()
    torch.testing.assert_close(result.loss, expected)


@pytest.mark.parametrize(
    ("std_dev_t", "sqrt_dt", "message"),
    [
        (torch.ones(2, 2, 1), torch.zeros(2, 2), "must be positive"),
        (torch.ones(2, 3, 1), torch.ones(2, 2), "broadcast"),
    ],
)
def test_grpo_guard_rejects_invalid_transition_geometry(
    std_dev_t: torch.Tensor,
    sqrt_dt: torch.Tensor,
    message: str,
) -> None:
    log_probs = torch.zeros(2, 2)
    means = torch.zeros(2, 2, 1)

    with pytest.raises(ValueError, match=message):
        grpo_guard_policy_loss(
            log_probs,
            log_probs,
            means,
            means,
            std_dev_t,
            sqrt_dt,
            torch.ones(2),
            clip_range=0.2,
            advantage_clip_max=5.0,
        )
