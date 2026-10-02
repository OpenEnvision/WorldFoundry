from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.token_policy import (  # noqa: E402
    MAX_SEQUENCE_LOG_RATIO,
    SEQUENCE_MEAN_TOKEN_MEAN,
    SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    TOKEN_MEAN,
    TokenGRPOStage,
    token_cppo_objective,
    token_dppo_objective,
    token_drpo_objective,
    token_grpo_objective,
    token_gspo_objective,
)


def test_grpo_asymmetric_clipping_matches_per_token_surrogate_and_gradient() -> None:
    old_log_probs = torch.log(torch.full((4,), 0.2))
    ratios = torch.tensor([0.7, 1.1, 1.4, 0.6])
    new_log_probs = (old_log_probs + ratios.log()).detach().requires_grad_()
    advantages = torch.tensor([1.0, 1.0, 1.0, -1.0])

    objective = token_grpo_objective(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_range=0.2,
        clip_range_high=0.3,
    )
    objective.losses.sum().backward()

    torch.testing.assert_close(objective.ratio, ratios)
    torch.testing.assert_close(
        objective.losses,
        torch.tensor([-0.7, -1.1, -1.3, 0.8]),
    )
    torch.testing.assert_close(
        new_log_probs.grad,
        torch.tensor([-0.7, -1.1, 0.0, 0.0]),
    )
    assert old_log_probs.grad is None


def test_gspo_uses_nonempty_sequence_mean_and_caps_log_ratio_at_ten() -> None:
    new_log_probs = torch.tensor(
        [12.0, 12.0, torch.log(torch.tensor(2.0))],
        requires_grad=True,
    )
    old_log_probs = torch.zeros(3)
    advantages = torch.tensor([1.0, 99.0, -1.0])

    objective = token_gspo_objective(
        new_log_probs,
        old_log_probs,
        advantages,
        torch.tensor([2, 0, 1]),
        clip_range=0.2,
    )
    objective.losses.sum().backward()

    torch.testing.assert_close(
        objective.log_ratio,
        torch.tensor([MAX_SEQUENCE_LOG_RATIO, torch.log(torch.tensor(2.0))]),
    )
    torch.testing.assert_close(
        objective.ratio,
        torch.tensor([torch.exp(torch.tensor(10.0)), 2.0]),
    )
    torch.testing.assert_close(objective.losses, torch.tensor([-1.2, 2.0]))
    torch.testing.assert_close(new_log_probs.grad, torch.tensor([0.0, 0.0, 2.0]))


def test_dppo_binary_tv_mask_and_gradient_match_hard_gate() -> None:
    old_log_probs = torch.log(torch.tensor([0.2, 0.2, 0.5]))
    new_log_probs = torch.log(torch.tensor([0.5, 0.1, 0.8])).requires_grad_()
    advantages = torch.tensor([1.0, 1.0, -1.0])

    objective = token_dppo_objective(
        new_log_probs,
        old_log_probs,
        advantages,
        delta=0.15,
    )
    objective.losses.sum().backward()

    torch.testing.assert_close(objective.keep_mask, torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(objective.losses, torch.tensor([0.0, -0.5, 1.6]))
    torch.testing.assert_close(new_log_probs.grad, torch.tensor([0.0, -0.5, 1.6]))


@pytest.mark.parametrize(
    ("mu_weighted", "expected_regularizer", "expected_gradient"),
    [
        (True, torch.tensor([0.005, 0.005]), torch.tensor([-2.34, 0.76])),
        (False, torch.tensor([0.02, 0.01]), torch.tensor([-2.16, 0.72])),
    ],
)
def test_drpo_quadratic_variants_match_exact_penalty_and_gradient(
    mu_weighted: bool,
    expected_regularizer: torch.Tensor,
    expected_gradient: torch.Tensor,
) -> None:
    old_log_probs = torch.log(torch.tensor([0.25, 0.5]))
    new_log_probs = (old_log_probs + torch.log(torch.tensor([1.2, 0.8]))).detach()
    new_log_probs.requires_grad_()

    objective = token_drpo_objective(
        new_log_probs,
        old_log_probs,
        torch.tensor([2.0, -1.0]),
        epsilon=2.0,
        mu_weighted=mu_weighted,
    )
    objective.losses.sum().backward()

    torch.testing.assert_close(objective.regularizer, expected_regularizer)
    torch.testing.assert_close(new_log_probs.grad, expected_gradient)


def test_cppo_uses_p90_clamp_and_isolates_each_sequence_prefix() -> None:
    old_probabilities = torch.full((4,), 0.1)
    new_probabilities = torch.tensor([0.2, 0.29, 0.2, 0.29])
    old_log_probs = old_probabilities.log()
    new_log_probs = new_probabilities.log().requires_grad_()

    objective = token_cppo_objective(
        new_log_probs,
        old_log_probs,
        torch.ones(4),
        torch.tensor([2, 0, 2]),
        delta=0.2,
        w_min=0.8,
        delta_b=0.02,
    )
    objective.losses.sum().backward()

    # P90([0.10, 0.19]) is clamped from 0.181 to 0.04.  Each sequence then
    # independently keeps its first token and rejects its second token.
    torch.testing.assert_close(objective.keep_mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(new_log_probs.grad, torch.tensor([-2.0, 0.0, -2.0, 0.0]))


def test_cppo_single_token_position_weight_is_one() -> None:
    old_log_probs = torch.log(torch.tensor([0.2]))
    new_log_probs = torch.log(torch.tensor([0.43])).requires_grad_()

    objective = token_cppo_objective(
        new_log_probs,
        old_log_probs,
        torch.ones(1),
        torch.tensor([1]),
        delta=0.2,
        w_min=0.8,
        delta_b=0.02,
    )
    objective.losses.sum().backward()

    torch.testing.assert_close(objective.keep_mask, torch.tensor([0.0]))
    torch.testing.assert_close(new_log_probs.grad, torch.tensor([0.0]))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (TOKEN_MEAN, -5.0 / 3.0),
        (SEQUENCE_MEAN_TOKEN_MEAN, -4.0 / 3.0),
        (SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED, -5.0 / 6.0),
    ],
)
def test_grpo_reductions_include_zero_length_sequences_exactly(
    mode: str,
    expected: float,
) -> None:
    stage = TokenGRPOStage(reduction=mode, horizon=2, clip_range=0.2)
    loss = stage.loss(
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([1.0, 99.0, 3.0]),
        torch.tensor([2, 0, 1]),
    )

    torch.testing.assert_close(loss.loss, torch.tensor(expected))


def test_grpo_clip_schedule_uses_the_pre_update_optimizer_step() -> None:
    stage = TokenGRPOStage(
        clip_range=0.2,
        clip_range_high=2.0,
        clip_schedule="linear-decay",
        clip_schedule_steps=4,
    )
    loss = stage.loss(
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1, 1]),
        optimizer_step=2,
    )

    torch.testing.assert_close(loss.metrics["clip_range"], torch.tensor(0.15))
