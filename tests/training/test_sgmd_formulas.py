from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.sgmd.config import (  # noqa: E402
    SGMDConfig,
)
from worldfoundry.training.post_training.distillation.sgmd.math import (  # noqa: E402
    sgmd_classifier_free_guidance,
    sgmd_diversity_loss_per_sample,
    sgmd_euler_step,
    sgmd_fake_correction_loss_per_sample,
    sgmd_fake_score_flow_loss_per_sample,
    sgmd_normalized_fisher_loss_per_sample,
)
from worldfoundry.training.post_training.distillation.sgmd.objective import (  # noqa: E402
    sample_sgmd_score_sigmas,
)


def test_released_wan_schedule_and_diversity_anchor_are_exact() -> None:
    config = SGMDConfig()
    assert config.student_timesteps == (1000.0, 750.0, 500.0, 250.0)
    assert config.student_sigmas == pytest.approx((1.0, 0.9375, 5.0 / 6.0, 0.625))
    assert config.teacher_sigmas[0] == 1.0
    assert config.teacher_sigmas[config.diversity_anchor_step] == pytest.approx(25.0 / 26.0)
    assert config.minimum_student_target_index == 1


def test_score_sigma_sampling_matches_discretize_shift_then_clamp_order() -> None:
    config = SGMDConfig()
    reference = torch.zeros(8, 2)
    actual = sample_sgmd_score_sigmas(
        reference,
        config,
        generator=torch.Generator().manual_seed(73),
    )
    indices = torch.randint(
        0,
        1000,
        (8,),
        generator=torch.Generator().manual_seed(73),
    )
    base = indices.float() / 1000.0
    expected = (5.0 * base / (1.0 + 4.0 * base)).clamp(0.02, 0.98)
    torch.testing.assert_close(actual, expected)


def test_sgmd_cfg_fisher_and_fake_correction_match_released_formulas() -> None:
    generated = torch.tensor([[1.0, 3.0], [2.0, -1.0]])
    fake_clean = torch.tensor([[0.5, 2.0], [1.0, 1.0]])
    teacher_clean = torch.tensor([[0.0, 2.5], [1.5, 0.0]])
    sigmas = torch.tensor([0.5, 0.25])

    unconditional = torch.tensor([[1.0, -2.0]])
    conditional = torch.tensor([[3.0, 2.0]])
    torch.testing.assert_close(
        sgmd_classifier_free_guidance(unconditional, conditional, 3.0),
        unconditional + 3.0 * (conditional - unconditional),
    )

    fisher, normalizer = sgmd_normalized_fisher_loss_per_sample(
        generated,
        fake_clean,
        teacher_clean,
    )
    expected_normalizer = (generated - teacher_clean).abs().mean(1, keepdim=True)
    expected_fisher = (
        0.5 * (fake_clean - teacher_clean).square() / (expected_normalizer + 1.0e-8)
    ).mean(1)
    torch.testing.assert_close(normalizer, expected_normalizer)
    torch.testing.assert_close(fisher, expected_fisher)

    correction = sgmd_fake_correction_loss_per_sample(generated, fake_clean, sigmas)
    expanded = sigmas[:, None]
    gradient = (fake_clean - generated) / expanded
    pseudo_target = fake_clean - gradient
    expected_correction = 0.5 * (fake_clean - pseudo_target).square().mean(1)
    torch.testing.assert_close(correction, expected_correction)


def test_sgmd_euler_flow_fake_score_and_diversity_formulas() -> None:
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    velocity = torch.tensor([[0.5, -0.5], [1.0, 2.0]])
    sigma = torch.tensor([1.0, 0.75])
    sigma_next = torch.tensor([0.5, 0.25])
    expected_step = sample + (sigma_next - sigma)[:, None] * velocity
    torch.testing.assert_close(
        sgmd_euler_step(sample, velocity, sigma, sigma_next),
        expected_step,
    )

    generated = torch.tensor([[0.25, -0.5], [1.0, 0.5]])
    noise = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    prediction = torch.tensor([[0.5, 0.5], [-0.5, 1.5]])
    target = noise - generated
    expected_flow = 0.5 * (prediction - target).square().mean(1)
    torch.testing.assert_close(
        sgmd_fake_score_flow_loss_per_sample(prediction, generated, noise),
        expected_flow,
    )

    anchor = torch.tensor([[0.5, 1.0], [2.5, 3.5]])
    anchor_sigma = torch.tensor([0.75, 0.5])
    target_velocity = (sample - anchor) / (1.0 - anchor_sigma)[:, None]
    expected_diversity = (velocity - target_velocity).square().mean(1)
    torch.testing.assert_close(
        sgmd_diversity_loss_per_sample(
            sample,
            anchor,
            anchor_sigma,
            velocity,
        ),
        expected_diversity,
    )
