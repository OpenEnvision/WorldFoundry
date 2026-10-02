from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.training.objectives.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingObjective,
    flow_clean_from_velocity,
    flow_interpolate,
    flow_matching_mse,
    flow_noise_from_velocity,
    flow_shift_sigmas,
    flow_velocity_target,
)


def test_flow_corruption_endpoints_and_inverse_are_exact() -> None:
    clean = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    noise = clean[::-1].copy() + 3.0
    velocity = flow_velocity_target(clean, noise)

    np.testing.assert_array_equal(flow_interpolate(clean, noise, 0.0), clean)
    np.testing.assert_array_equal(flow_interpolate(clean, noise, 1.0), noise)

    sigmas = np.asarray([0.25, 0.75], dtype=np.float32)
    noisy = flow_interpolate(clean, noise, sigmas)
    np.testing.assert_allclose(flow_clean_from_velocity(noisy, velocity, sigmas), clean, rtol=0, atol=1e-6)
    np.testing.assert_allclose(flow_noise_from_velocity(noisy, velocity, sigmas), noise, rtol=0, atol=1e-6)


def test_flow_config_rejects_degenerate_sigma_range() -> None:
    with pytest.raises(ValueError, match="sigma range"):
        FlowMatchingConfig(min_sigma=0.5, max_sigma=0.5)


def test_sana_discrete_shifted_noise_levels_match_official_schedule() -> None:
    torch = pytest.importorskip("torch")
    config = FlowMatchingConfig(
        timestep_sampler="uniform",
        flow_shift=3.0,
        num_train_timesteps=1000,
    )
    objective = FlowMatchingObjective(config)

    sigmas, timesteps = objective._sample_noise_levels(  # noqa: SLF001 - golden schedule contract.
        8,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    base_sigmas = timesteps.float() / 1000.0
    torch.testing.assert_close(sigmas, flow_shift_sigmas(base_sigmas, 3.0))
    assert timesteps.dtype == torch.long
    assert bool(((0 <= timesteps) & (timesteps < 1000)).all())


def test_masked_weighted_mse_uses_effective_token_denominator() -> None:
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([[[1.0, 3.0]], [[2.0, 10.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    weights = torch.tensor([2.0, 0.5])

    result = flow_matching_mse(prediction, target, loss_mask=mask, sample_weights=weights)

    # Numerator = 1^2*2 + 2^2*0.5 + 10^2*0.5 = 54; denominator = 2 + .5 + .5 = 3.
    torch.testing.assert_close(result.loss, torch.tensor(18.0))
    result.loss.backward()
    assert prediction.grad[0, 0, 1].item() == 0.0


def test_flow_objective_corruption_is_generator_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.training.api import PreparedBatch

    prepared = PreparedBatch(sample_ids=("a", "b"), clean_latents=torch.zeros(2, 4, 2, 2))
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    first = objective.corrupt(prepared, generator=torch.Generator().manual_seed(123))
    second = objective.corrupt(prepared, generator=torch.Generator().manual_seed(123))

    torch.testing.assert_close(first.sigmas, second.sigmas)
    torch.testing.assert_close(first.model_input, second.model_input)
    torch.testing.assert_close(first.target, second.target)
