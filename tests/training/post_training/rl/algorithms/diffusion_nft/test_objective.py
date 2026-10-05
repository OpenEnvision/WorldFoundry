from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.objective import (  # noqa: E402
    diffusion_nft_forward_process,
    diffusion_nft_loss,
    diffusion_nft_reward_weights,
)


def test_diffusion_nft_forward_process_has_official_flow_target() -> None:
    clean = torch.tensor([[2.0, -1.0]])
    noise = torch.tensor([[4.0, 3.0]])

    result = diffusion_nft_forward_process(clean, torch.tensor([0.25]), noise)

    torch.testing.assert_close(result.noisy_latents, torch.tensor([[2.5, 0.0]]), rtol=0, atol=0)
    torch.testing.assert_close(result.target_velocity, torch.tensor([[2.0, 4.0]]), rtol=0, atol=0)


def test_diffusion_nft_group_advantage_clamps_then_maps_to_probability() -> None:
    result = diffusion_nft_reward_weights(
        torch.tensor([1.0, 3.0, 2.0, 6.0]),
        ("first", "first", "second", "second"),
        advantage_clip_max=0.5,
    )
    torch.testing.assert_close(
        result.advantages,
        torch.tensor([-0.5, 0.5, -0.5, 0.5]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.reward_probabilities,
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("positive_only", [0.5, 1.0]),
        ("negative_only", [0.0, 0.5]),
        ("one_only", [0.5, 1.0]),
        ("binary", [0.0, 1.0]),
    ),
)
def test_diffusion_nft_reward_mapping_modes(mode: str, expected: list[float]) -> None:
    result = diffusion_nft_reward_weights(
        torch.tensor([1.0, 3.0]),
        ("prompt", "prompt"),
        advantage_clip_max=0.5,
        advantage_mode=mode,
    )

    torch.testing.assert_close(
        result.reward_probabilities,
        torch.tensor(expected),
        rtol=0,
        atol=0,
    )


def test_diffusion_nft_hand_computed_positive_negative_and_reference_loss() -> None:
    policy = torch.full((2, 1), 3.0, requires_grad=True)
    result = diffusion_nft_loss(
        clean_latents=torch.zeros(2, 1),
        noisy_latents=torch.zeros(2, 1),
        times=torch.ones(2),
        target_velocity=torch.zeros(2, 1),
        policy_prediction=policy,
        old_policy_prediction=torch.ones(2, 1),
        reward_probabilities=torch.tensor([1.0, 0.0]),
        beta=0.5,
        advantage_clip_max=4.0,
        reference_prediction=torch.full((2, 1), 2.0),
        reference_mse_weight=0.25,
    )

    # v+ = 2, v- = 0.  At t=1 and x_t=x_0=0 the normalized
    # reconstruction losses are therefore 2 and 0.  Only the positive sample
    # contributes: mean(4 * 2 / 0.5, 0) = 8.  Reference MSE adds 0.25.
    torch.testing.assert_close(
        result.positive_reconstruction,
        torch.tensor([2.0, 2.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.negative_reconstruction,
        torch.zeros(2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(result.policy_loss, torch.tensor(8.0), rtol=0, atol=0)
    torch.testing.assert_close(result.reference_mse, torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(result.loss, torch.tensor(8.25), rtol=0, atol=0)


def test_diffusion_nft_detaches_old_reference_weights_and_mae_denominators() -> None:
    policy = torch.tensor([[0.2], [0.7]], requires_grad=True)
    old = torch.tensor([[0.1], [0.3]], requires_grad=True)
    reference = torch.tensor([[0.0], [0.0]], requires_grad=True)
    probabilities = torch.tensor([0.25, 0.75], requires_grad=True)

    result = diffusion_nft_loss(
        clean_latents=torch.tensor([[0.3], [-0.2]]),
        noisy_latents=torch.tensor([[0.4], [0.6]]),
        times=torch.tensor([0.2, 0.8]),
        target_velocity=torch.tensor([[0.5], [-0.1]]),
        policy_prediction=policy,
        old_policy_prediction=old,
        reward_probabilities=probabilities,
        beta=0.1,
        advantage_clip_max=2.0,
        reference_prediction=reference,
        reference_mse_weight=0.5,
    )
    result.loss.backward()

    assert policy.grad is not None and bool(torch.isfinite(policy.grad).all())
    assert old.grad is None
    assert reference.grad is None
    assert probabilities.grad is None


def test_diffusion_nft_finite_gate_rejects_invalid_policy_prediction() -> None:
    with pytest.raises(FloatingPointError, match="policy_prediction"):
        diffusion_nft_loss(
            clean_latents=torch.zeros(2, 1),
            noisy_latents=torch.zeros(2, 1),
            times=torch.ones(2),
            target_velocity=torch.zeros(2, 1),
            policy_prediction=torch.tensor([[float("nan")], [0.0]], requires_grad=True),
            old_policy_prediction=torch.zeros(2, 1),
            reward_probabilities=torch.full((2,), 0.5),
            beta=0.1,
            advantage_clip_max=1.0,
        )
