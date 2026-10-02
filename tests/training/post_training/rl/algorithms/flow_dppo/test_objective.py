from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.flow_dppo.objective import (  # noqa: E402
    flow_dppo_policy_loss,
)


def test_flow_dppo_masks_only_high_kl_reward_aligned_ratio_directions() -> None:
    old_logp = torch.zeros(2, 2)
    new_logp = torch.tensor(
        [[math.log(2.0), math.log(0.5)], [math.log(0.5), math.log(2.0)]],
        requires_grad=True,
    )
    old_means = torch.zeros(2, 2, 1)
    new_means = torch.full((2, 2, 1), 2.0, requires_grad=True)

    result = flow_dppo_policy_loss(
        new_logp,
        old_logp,
        new_means,
        old_means,
        torch.ones(1, 1, 1),
        torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        kl_mask_threshold=2.0,
    )

    # Equality is high-KL in the official implementation.  Positive advantage
    # removes ratio>1; negative advantage removes ratio<1.
    assert result.keep_mask.tolist() == [[False, True], [False, True]]
    torch.testing.assert_close(
        result.per_transition,
        torch.tensor([[0.0, -0.5], [0.0, 2.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(result.loss, torch.tensor(0.375), rtol=0, atol=0)
    torch.testing.assert_close(result.masked_fraction, torch.tensor(0.5), rtol=0, atol=0)
    torch.testing.assert_close(
        result.positive_masked_fraction,
        torch.tensor(0.25),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.negative_masked_fraction,
        torch.tensor(0.25),
        rtol=0,
        atol=0,
    )


def test_flow_dppo_uses_stored_transition_scale_only_when_configured() -> None:
    kwargs = {
        "new_log_probs": torch.zeros(1, 1),
        "old_log_probs": torch.zeros(1, 1),
        "new_transition_means": torch.tensor([[[2.0]]]),
        "old_transition_means": torch.zeros(1, 1, 1),
        "transition_scales": torch.tensor([[[2.0]]]),
        "advantages": torch.ones(1),
        "kl_mask_threshold": 10.0,
    }

    normalized = flow_dppo_policy_loss(**kwargs, add_kl_coefficient=True)
    unnormalized = flow_dppo_policy_loss(**kwargs, add_kl_coefficient=False)

    torch.testing.assert_close(normalized.old_policy_kl, torch.tensor([[0.5]]), rtol=0, atol=0)
    torch.testing.assert_close(unnormalized.old_policy_kl, torch.tensor([[2.0]]), rtol=0, atol=0)


def test_flow_dppo_exact_anchor_has_ratio_one_and_zero_old_policy_kl() -> None:
    logp = torch.tensor([[-1.5, -0.25], [-0.5, -2.0]])
    means = torch.randn(2, 2, 3, generator=torch.Generator().manual_seed(71))
    result = flow_dppo_policy_loss(
        logp.clone().requires_grad_(True),
        logp,
        means.clone().requires_grad_(True),
        means,
        torch.full((1, 2, 1), 0.25),
        torch.tensor([1.0, -1.0]),
        kl_mask_threshold=1.0e-5,
    )

    torch.testing.assert_close(result.ratio, torch.ones(2, 2), rtol=0, atol=0)
    torch.testing.assert_close(result.old_policy_kl, torch.zeros(2, 2), rtol=0, atol=0)
    assert bool(result.keep_mask.all())


def test_flow_dppo_where_mask_prevents_overflow_times_zero_nan() -> None:
    result = flow_dppo_policy_loss(
        torch.tensor([[100.0]], requires_grad=True),
        torch.zeros(1, 1),
        torch.ones(1, 1, 1, requires_grad=True),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1),
        torch.ones(1),
        kl_mask_threshold=0.1,
    )

    assert torch.isinf(result.ratio).all()
    assert result.loss.item() == 0.0
    assert torch.isfinite(result.loss)
