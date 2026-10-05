from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.adversarial_diffusion import (  # noqa: E402
    ADDDiscriminatorHeadOutput,
    ADDNoiseSchedule,
    add_forward_noise,
    discriminator_hinge_loss_per_sample,
    distillation_weights,
    feature_r1_penalty_per_sample,
    generator_hinge_loss_per_sample,
    pixel_distillation_loss_per_sample,
    schedule_coefficients,
)


def _head(features: torch.Tensor, logits: torch.Tensor, layer: str) -> ADDDiscriminatorHeadOutput:
    return ADDDiscriminatorHeadOutput(
        resolution=8,
        layer=layer,
        features=features,
        logits=logits,
    )


def test_forward_process_uses_vp_alpha_and_sigma_coefficients() -> None:
    schedule = ADDNoiseSchedule((1.0, 0.25, 0.0))
    clean = torch.tensor([[[[2.0]]], [[[4.0]]]])
    noise = torch.tensor([[[[3.0]]], [[[5.0]]]])
    timesteps = torch.tensor([0, 1], dtype=torch.int64)

    alpha, sigma = schedule_coefficients(schedule, timesteps, clean)
    noised = add_forward_noise(clean, noise, alpha, sigma)

    torch.testing.assert_close(alpha, torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(sigma, torch.tensor([0.0, 3.0**0.5 / 2.0]))
    torch.testing.assert_close(
        noised.flatten(),
        torch.tensor([2.0, 2.0 + 5.0 * 3.0**0.5 / 2.0]),
    )


def test_report_exponential_and_sds_distillation_weights() -> None:
    """Check public report equations, not unpublished trainer parity."""

    fixture_path = Path(__file__).parent / "fixtures/source_formulas/adversarial-diffusion.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert set(fixture) == {"inputs", "expected", "atol", "rtol"}
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    reference = torch.zeros(tuple(inputs["reference_shape"]))
    timesteps = torch.tensor(inputs["timesteps"], dtype=torch.int64)
    exponential_schedule = ADDNoiseSchedule(tuple(inputs["alpha_cumprods"]))
    exponential = distillation_weights(
        exponential_schedule,
        timesteps,
        reference,
        weighting="exponential",
    )
    torch.testing.assert_close(
        exponential,
        torch.tensor(expected["exponential"]),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )

    sds_schedule = ADDNoiseSchedule(
        tuple(inputs["alpha_cumprods"]),
        training_loss_weights=tuple(inputs["training_loss_weights"]),
    )
    sds = distillation_weights(
        sds_schedule,
        timesteps,
        reference,
        weighting="sds",
    )
    torch.testing.assert_close(
        sds,
        torch.tensor(expected["sds"]),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )

    with pytest.raises(ValueError, match="inert"):
        distillation_weights(sds_schedule, timesteps, reference, weighting="exponential")
    with pytest.raises(ValueError, match="requires"):
        distillation_weights(exponential_schedule, timesteps, reference, weighting="sds")


def test_pixel_distance_and_hinge_losses_preserve_head_sum() -> None:
    generated = torch.tensor(
        [
            [[[1.0, 3.0]]],
            [[[2.0, 4.0]]],
        ]
    )
    target = torch.zeros_like(generated)
    pixel = pixel_distillation_loss_per_sample(
        generated,
        target,
        torch.tensor([0.5, 2.0]),
    )
    torch.testing.assert_close(pixel, torch.tensor([2.5, 20.0]))

    placeholder = torch.zeros(2, 1)
    fake_heads = (
        _head(placeholder, torch.tensor([[2.0], [-1.0]]), "a"),
        _head(placeholder, torch.tensor([[1.0], [3.0]]), "b"),
    )
    generator = generator_hinge_loss_per_sample(fake_heads)
    torch.testing.assert_close(generator, torch.tensor([-3.0, -2.0]))

    real_heads = (
        _head(placeholder, torch.tensor([[2.0], [0.0]]), "a"),
        _head(placeholder, torch.tensor([[0.5], [2.0]]), "b"),
    )
    discriminator, real, fake = discriminator_hinge_loss_per_sample(real_heads, fake_heads)
    torch.testing.assert_close(real, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(fake, torch.tensor([5.0, 4.0]))
    torch.testing.assert_close(discriminator, torch.tensor([5.5, 5.0]))


def test_r1_is_computed_on_each_head_input_not_pixels() -> None:
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    second = torch.tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
    heads = (
        _head(first, first.sum(dim=1, keepdim=True), "a"),
        _head(second, 2.0 * second.sum(dim=1, keepdim=True), "b"),
    )

    penalty = feature_r1_penalty_per_sample(heads)

    # 0.5 * (1^2 + 1^2) + 0.5 * (2^2 + 2^2)
    torch.testing.assert_close(penalty, torch.full((2,), 5.0))


def test_add_schedules_and_configs_reject_coercive_or_inert_values() -> None:
    from worldfoundry.training.post_training.distillation.adversarial_diffusion import ADDConfig

    with pytest.raises(TypeError, match="real number"):
        ADDNoiseSchedule((1.0, "0.5", 0.0))
    with pytest.raises(TypeError, match="student_timesteps"):
        ADDConfig(
            student_timesteps=(1, 2, 3, True),
            teacher_timestep_min=1,
            teacher_timestep_max=2,
            feature_resolutions=(8,),
            feature_layers=("layer",),
            discriminator_conditioning_keys=(),
        )
