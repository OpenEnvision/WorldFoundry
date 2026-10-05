"""Regression tests for degenerate DMD gradient cleaning (review TR-20)."""

from __future__ import annotations

import torch

from worldfoundry.training.post_training.distillation.dmd.objective import (
    dmd_distribution_gradient,
    dmd_proxy_loss,
)


def test_normal_path_matches_manual_normalized_difference() -> None:
    generated = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    real = torch.tensor([[0.5, 2.5], [2.0, 5.0]])
    fake = torch.tensor([[1.5, 1.0], [2.5, 4.5]])

    gradient, denominator = dmd_distribution_gradient(generated, fake, real)

    expected_denominator = (generated - real).abs().mean()
    expected = (fake - real) / expected_denominator
    assert torch.equal(denominator, expected_denominator)
    assert torch.allclose(gradient, expected)
    assert bool(torch.isfinite(gradient).all())


def test_zero_denominator_yields_zero_gradient_and_finite_proxy_loss() -> None:
    generated = torch.ones(2, 3)
    real = generated.clone()  # denominator == 0
    fake = generated + 1.0  # non-zero numerator -> +inf before cleaning

    gradient, denominator = dmd_distribution_gradient(generated, fake, real)

    assert float(denominator) == 0.0
    assert torch.equal(gradient, torch.zeros_like(generated))

    loss = dmd_proxy_loss(generated, gradient)
    assert bool(torch.isfinite(loss))
    assert float(loss) == 0.0


def test_per_sample_zero_denominator_only_zeroes_the_degenerate_sample() -> None:
    generated = torch.stack((torch.ones(4), torch.arange(4.0)))
    real = torch.stack((torch.ones(4), torch.arange(4.0) + 2.0))
    fake = generated + 1.0

    gradient, denominator = dmd_distribution_gradient(
        generated,
        fake,
        real,
        per_sample_normalization=True,
    )

    assert float(denominator[0]) == 0.0
    assert torch.equal(gradient[0], torch.zeros(4))
    expected_second = (fake[1] - real[1]) / denominator[1]
    assert torch.allclose(gradient[1], expected_second)
    assert bool(torch.isfinite(gradient).all())
