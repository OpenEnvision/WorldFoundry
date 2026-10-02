from __future__ import annotations

import json
from pathlib import Path

import torch

from worldfoundry.training.post_training.distillation.scm_ladd.math import (
    flow_velocity_to_trigflow,
    ladd_discriminator_hinge_loss,
    ladd_generator_hinge_loss,
    sample_trigflow_timesteps,
    scm_tangent_target,
    trigflow_to_flow_input,
)
from worldfoundry.training.recipes import SCMLADDAlgorithmSpec

_FIXTURE = Path(__file__).parent / "fixtures/source_formulas/sana-sprint-scm-ladd.json"


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _case(name: str) -> tuple[dict[str, object], dict[str, object], float, float]:
    fixture = _fixture()
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    assert isinstance(inputs, dict) and isinstance(expected, dict)
    case_inputs = inputs[name]
    case_expected = expected[name]
    assert isinstance(case_inputs, dict) and isinstance(case_expected, dict)
    return case_inputs, case_expected, float(fixture["atol"]), float(fixture["rtol"])


def test_fixed_sana_sprint_profile_is_fully_consumed_by_algorithm_spec() -> None:
    inputs = _fixture()["inputs"]
    assert isinstance(inputs, dict)
    profile = inputs["official_profile"]
    assert isinstance(profile, dict)
    spec = SCMLADDAlgorithmSpec(**profile)
    assert spec.guidance_embedding_scale == 0.1
    assert spec.discriminator_head_block_ids == (2, 8, 14, 19)
    assert spec.misaligned_pairs is True
    assert spec.independent_real_fake_discriminator_times is True
    assert spec.largest_time_enabled is True
    assert spec.largest_time == 1.5708
    assert spec.adversarial_loss == "hinge"


def test_sana_sprint_equations_six_through_eight_match_fixed_source_values() -> None:
    values, expected, atol, rtol = _case("flow_to_trig")
    trig_t = torch.tensor(values["trig_timesteps"], dtype=torch.float64)
    scaled_trig = torch.tensor(values["scaled_trig_latents"], dtype=torch.float64)
    flow_velocity = torch.tensor(values["flow_velocity"], dtype=torch.float64)
    flow_latents, flow_t, _ = trigflow_to_flow_input(scaled_trig, trig_t)
    trig_velocity = flow_velocity_to_trigflow(flow_latents, flow_velocity, flow_t)
    torch.testing.assert_close(
        flow_t,
        torch.tensor(expected["flow_timesteps"], dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        flow_latents,
        torch.tensor(expected["flow_latents"], dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        trig_velocity,
        torch.tensor(expected["trig_velocity"], dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )


def test_sana_sprint_jvp_rearrangement_and_tangent_normalization_match_source() -> None:
    values, expected, atol, rtol = _case("tangent")
    tangent, norm = scm_tangent_target(
        torch.tensor(values["noisy_latents"]),
        torch.tensor(values["stopped_velocity"]),
        torch.tensor(values["directional_derivative"]),
        torch.tensor(values["teacher_path_velocity"]),
        torch.tensor(values["trig_timesteps"]),
        sigma_data=0.5,
        warmup_ratio=values["warmup_ratio"],
        normalization_constant=0.1,
    )
    torch.testing.assert_close(
        tangent,
        torch.tensor(expected["normalized_tangent"]),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        norm.flatten(),
        torch.tensor(expected["tangent_norm"]),
        atol=atol,
        rtol=rtol,
    )


def test_sana_sprint_hinge_losses_match_fixed_trainer_reductions() -> None:
    values, expected, atol, rtol = _case("hinge")
    real = torch.tensor(values["real_logits"])
    fake = torch.tensor(values["fake_logits"])
    generator = ladd_generator_hinge_loss(fake)
    discriminator, real_loss, fake_loss = ladd_discriminator_hinge_loss(real, fake)
    torch.testing.assert_close(generator, torch.tensor(expected["generator"]), atol=atol, rtol=rtol)
    torch.testing.assert_close(real_loss, torch.tensor(expected["real"]), atol=atol, rtol=rtol)
    torch.testing.assert_close(fake_loss, torch.tensor(expected["fake"]), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        discriminator,
        torch.tensor(expected["discriminator"]),
        atol=atol,
        rtol=rtol,
    )


def test_official_largest_time_mixture_is_an_executed_sampler_branch() -> None:
    reference = torch.zeros(4, 2)
    timesteps = sample_trigflow_timesteps(
        reference,
        logit_mean=0.2,
        logit_std=1.6,
        sigma_data=0.5,
        max_time_probability=1.0,
        max_time=1.57080,
        generator=torch.Generator().manual_seed(9),
    )
    torch.testing.assert_close(timesteps, torch.full((4,), 1.57080))
