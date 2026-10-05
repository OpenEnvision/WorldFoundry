from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_dpo import (  # noqa: E402
    diffusion_dpo_forward_process,
    diffusion_dpo_loss,
    sample_diffusion_dpo_forward_process,
)


def test_forward_process_uses_shared_pair_noise_time_and_flow_target() -> None:
    clean = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    noise = torch.tensor([[4.0, 3.0], [4.0, 3.0]])

    result = diffusion_dpo_forward_process(clean, torch.tensor([0.25, 0.25]), noise)

    torch.testing.assert_close(
        result.noisy_latents,
        torch.tensor([[2.5, 0.0], [1.0, 1.5]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.target_velocity,
        torch.tensor([[2.0, 4.0], [4.0, 2.0]]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("times", "noise", "message"),
    [
        (torch.tensor([0.2, 0.3]), torch.ones(2, 1), "share one timestep"),
        (torch.tensor([0.2, 0.2]), torch.tensor([[1.0], [2.0]]), "share one noise"),
    ],
)
def test_forward_process_rejects_pair_stochasticity_misalignment(times, noise, message) -> None:
    with pytest.raises(ValueError, match=message):
        diffusion_dpo_forward_process(torch.zeros(2, 1), times, noise)


def test_forward_sampler_repeats_pair_stochasticity_exactly() -> None:
    result = sample_diffusion_dpo_forward_process(
        torch.zeros(6, 2, 3),
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(result.times[0::2], result.times[1::2], rtol=0, atol=0)
    torch.testing.assert_close(result.noise[0::2], result.noise[1::2], rtol=0, atol=0)


def test_diffusion_dpo_loss_matches_hand_computed_pair_math() -> None:
    policy = torch.tensor([[1.0], [3.0]], requires_grad=True)
    result = diffusion_dpo_loss(
        target_velocity=torch.zeros(2, 1),
        policy_prediction=policy,
        reference_prediction=torch.full((2, 1), 2.0),
        beta=0.5,
    )

    # Current chosen/rejected errors are 1 and 9; reference errors are 4 and 4.
    # z = -0.5 * 0.5 * ((1 - 9) - (4 - 4)) = 2.
    torch.testing.assert_close(result.current_mse, torch.tensor([1.0, 9.0]), rtol=0, atol=0)
    torch.testing.assert_close(result.reference_mse, torch.tensor([4.0, 4.0]), rtol=0, atol=0)
    torch.testing.assert_close(result.logits, torch.tensor([2.0]), rtol=0, atol=0)
    torch.testing.assert_close(result.loss, torch.nn.functional.softplus(torch.tensor(-2.0)))


def test_diffusion_dpo_gradient_isolates_reference_and_forward_target() -> None:
    policy = torch.tensor([[0.2], [0.7]], requires_grad=True)
    reference = torch.tensor([[0.1], [0.3]], requires_grad=True)
    target = torch.tensor([[0.5], [-0.1]], requires_grad=True)

    result = diffusion_dpo_loss(
        target_velocity=target,
        policy_prediction=policy,
        reference_prediction=reference,
        beta=0.2,
    )
    result.loss.backward()

    assert policy.grad is not None and bool(torch.isfinite(policy.grad).all())
    assert reference.grad is None
    assert target.grad is None
