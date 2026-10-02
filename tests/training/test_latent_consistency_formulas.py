from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.latent_consistency.config import (  # noqa: E402
    LatentConsistencyConfig,
    LatentConsistencyNoiseSchedule,
    build_latent_consistency_ddim_schedule,
)
from worldfoundry.training.post_training.distillation.latent_consistency.math import (  # noqa: E402
    add_forward_diffusion_noise,
    boundary_condition_scalings,
    classifier_free_guidance,
    deterministic_ddim_step,
    guidance_scale_embedding,
    latent_consistency_elementwise_loss,
    prediction_to_origin_and_epsilon,
)


def test_ddim_schedule_matches_released_evenly_spaced_solver() -> None:
    alpha_cumprods = tuple(1.0 - 0.0009 * index for index in range(1000))
    noise_schedule = LatentConsistencyNoiseSchedule(alpha_cumprods)
    schedule = build_latent_consistency_ddim_schedule(
        noise_schedule,
        LatentConsistencyConfig(),
    )

    assert schedule.step_size == 20
    assert schedule.start_timesteps == tuple(range(19, 1000, 20))
    assert schedule.end_timesteps == (0, *tuple(range(19, 980, 20)))
    assert schedule.previous_alpha_cumprods[0] == alpha_cumprods[0]
    assert schedule.previous_alpha_cumprods[1:] == tuple(alpha_cumprods[index] for index in range(19, 980, 20))

    with pytest.raises(ValueError, match="divisible"):
        build_latent_consistency_ddim_schedule(
            LatentConsistencyNoiseSchedule(tuple(1.0 - 0.01 * i for i in range(9))),
            LatentConsistencyConfig(num_ddim_timesteps=4),
        )


def test_boundary_scaling_and_guidance_embedding_match_lcm_formulas() -> None:
    timesteps = torch.tensor([0, 2], dtype=torch.int64)
    c_skip, c_out = boundary_condition_scalings(
        timesteps,
        sigma_data=0.5,
        timestep_scaling=10.0,
    )
    torch.testing.assert_close(c_skip[0], torch.tensor(1.0))
    torch.testing.assert_close(c_out[0], torch.tensor(0.0))
    torch.testing.assert_close(c_skip[1], torch.tensor(0.25 / 400.25))
    torch.testing.assert_close(c_out[1], torch.tensor(20.0 / math.sqrt(400.25)))

    coefficients = torch.tensor([0.0, 1.0])
    embedding = guidance_scale_embedding(
        coefficients,
        embedding_dim=5,
        embedding_scale=1.0,
        max_period=10.0,
    )
    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [math.sin(1.0), math.sin(0.1), math.cos(1.0), math.cos(0.1), 0.0],
        ]
    )
    torch.testing.assert_close(embedding, expected)
    assert embedding.device == coefficients.device


def test_epsilon_and_velocity_predictions_resolve_the_same_origin_and_noise() -> None:
    origin = torch.tensor([2.0, -1.0]).reshape(2, 1, 1, 1)
    epsilon = torch.tensor([0.5, 1.5]).reshape(2, 1, 1, 1)
    alpha = torch.full((2, 1, 1, 1), 0.8)
    sigma = torch.full((2, 1, 1, 1), 0.6)
    noisy = add_forward_diffusion_noise(origin, epsilon, alpha, sigma)

    epsilon_origin, epsilon_output = prediction_to_origin_and_epsilon(
        epsilon,
        noisy,
        alpha,
        sigma,
        prediction_type="epsilon",
    )
    velocity = alpha * epsilon - sigma * origin
    velocity_origin, velocity_output = prediction_to_origin_and_epsilon(
        velocity,
        noisy,
        alpha,
        sigma,
        prediction_type="v_prediction",
    )
    torch.testing.assert_close(epsilon_origin, origin)
    torch.testing.assert_close(epsilon_output, epsilon)
    torch.testing.assert_close(velocity_origin, origin)
    torch.testing.assert_close(velocity_output, epsilon)


def test_cfg_ddim_and_robust_loss_use_released_coefficient_conventions() -> None:
    conditional = torch.tensor([[[[3.0]]], [[[2.0]]]])
    unconditional = torch.tensor([[[[1.0]]], [[[4.0]]]])
    coefficients = torch.tensor([2.0, 0.5])
    guided = classifier_free_guidance(
        conditional,
        unconditional,
        coefficients,
    )
    torch.testing.assert_close(guided.flatten(), torch.tensor([7.0, 1.0]))

    previous = deterministic_ddim_step(
        torch.tensor([[[[2.0]]]]),
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[0.25]]]]),
    )
    torch.testing.assert_close(
        previous,
        torch.tensor([[[[1.0 + math.sqrt(0.75)]]]]),
    )

    prediction = torch.tensor([0.0, 3.0])
    target = torch.tensor([0.0, 1.0])
    l2 = latent_consistency_elementwise_loss(
        prediction,
        target,
        loss_type="l2",
        pseudo_huber_c=0.5,
    )
    robust = latent_consistency_elementwise_loss(
        prediction,
        target,
        loss_type="pseudo_huber",
        pseudo_huber_c=0.5,
    )
    torch.testing.assert_close(l2, torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(
        robust,
        torch.tensor([0.0, math.sqrt(4.25) - 0.5]),
    )
