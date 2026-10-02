from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.senseflow import (  # noqa: E402
    SenseFlowConfig,
    SenseFlowOptimizerConfig,
    SenseFlowSchedule,
    audit_ida_alignment,
    flow_euler_step,
    flow_isg_paths,
    flow_velocity_from_clean,
    implicit_distribution_alignment_,
    isg_loss_per_sample,
    sample_isg_midpoint,
    sample_score_sigmas,
    senseflow_adversarial_time_weight,
    senseflow_discriminator_hinge_loss,
    senseflow_distribution_gradient,
    senseflow_generator_hinge_loss,
    senseflow_proxy_loss_per_sample,
    senseflow_sigma_at_timestep,
)


def _source_formula_fixture() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "source_formulas" / "senseflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_released_sd35_and_flux_controls_match_author_configs() -> None:
    sd35 = SenseFlowConfig.sd35_large_released()
    assert sd35.schedule.timesteps == (999, 749, 499, 249)
    assert sd35.schedule.sigmas == pytest.approx((1.0, 0.9, 0.75, 0.5))
    assert sd35.schedule.isg_margin == 50
    assert sd35.schedule.flow_shift == pytest.approx(3.0)
    assert sd35.schedule.timestep_index_offset == 1
    assert sd35.schedule.terminal_sigma == pytest.approx(3.0 / 1002.0)
    assert sd35.schedule.adversarial_scales == pytest.approx(
        (
            3.0 / 1002.0,
            0.753 / 1.502,
            1.503 / 2.002,
            2.253 / 2.502,
        )
    )
    assert sd35.generator_update_interval == 5
    assert sd35.backward_simulation_probability == pytest.approx(0.5)
    assert sd35.ida_decay == pytest.approx(0.97)
    assert sd35.isg_loss == "charbonnier"
    assert sd35.isg_epsilon == pytest.approx(1.0e-3)
    assert sd35.isg_teacher_guidance == pytest.approx((5.0, 5.0))
    assert sd35.dmd_teacher_guidance == pytest.approx((3.0, 10.0))
    assert sd35.score_sampling == "uniform-schedule-index"
    assert sd35.fake_score_sampling == "logit-normal-scheduler-index"
    assert sd35.score_flow_shift == pytest.approx(3.0)

    medium = SenseFlowConfig.sd35_medium_released()
    assert medium.generator_update_interval == 10
    assert medium.ida_decay == pytest.approx(0.98)
    assert medium.isg_weight == pytest.approx(0.5)
    assert medium.isg_teacher_guidance == pytest.approx((2.0, 4.0))
    assert medium.dmd_teacher_guidance == pytest.approx((2.0, 8.0))

    flux = SenseFlowConfig.flux_released()
    assert flux.schedule.timesteps == (1000, 904, 759, 512)
    assert flux.schedule.sigmas == pytest.approx((1.0, 0.904, 0.759, 0.512))
    assert flux.schedule.isg_margin == 20
    assert flux.isg_teacher_guidance == pytest.approx((1.0, 8.0))
    assert flux.dmd_teacher_guidance == pytest.approx((1.0, 8.0))
    assert flux.score_sampling == "logit-normal-scheduler-index"
    assert flux.score_flow_shift == pytest.approx(1.0)
    assert flux.generator_adversarial_weight == pytest.approx(2.0)

    sd35_optimizer = SenseFlowOptimizerConfig.sd35_released()
    assert sd35_optimizer.student_learning_rate == pytest.approx(1.0e-6)
    assert sd35_optimizer.warmup_steps == 500
    assert sd35_optimizer.warmup_start_ratio == pytest.approx(1.0)
    flux_optimizer = SenseFlowOptimizerConfig.flux_released()
    assert flux_optimizer.student_learning_rate == pytest.approx(1.0e-5)
    assert flux_optimizer.warmup_steps == 500
    assert flux_optimizer.warmup_start_ratio == pytest.approx(0.5)


def test_schedule_rejects_segments_without_a_valid_isg_midpoint() -> None:
    with pytest.raises(ValueError, match="ISG midpoint"):
        SenseFlowSchedule(
            timesteps=(10,),
            sigmas=(0.01,),
            isg_margin=6,
        )


def test_isg_midpoint_is_inclusive_and_sigma_interpolation_is_exact() -> None:
    generator = torch.Generator().manual_seed(17)
    seen = {
        int(
            sample_isg_midpoint(
                7,
                2,
                margin=2,
                device=torch.device("cpu"),
                generator=generator,
            ).item()
        )
        for _ in range(200)
    }
    assert seen == {4, 5}

    midpoint = torch.tensor(5)
    sigma = senseflow_sigma_at_timestep(
        midpoint,
        num_train_timesteps=10,
        flow_shift=1.0,
        timestep_index_offset=0,
    )
    torch.testing.assert_close(sigma, torch.tensor(0.5))

    shifted = senseflow_sigma_at_timestep(
        torch.tensor(249),
        num_train_timesteps=1000,
        flow_shift=3.0,
        timestep_index_offset=1,
    )
    torch.testing.assert_close(shifted, torch.tensor(0.5))


def test_flow_velocity_euler_and_two_isg_paths_match_equations() -> None:
    anchor = torch.tensor([[3.0, 5.0]])
    clean = torch.tensor([[1.0, 1.0]])
    sigma = torch.tensor([0.5])
    velocity = flow_velocity_from_clean(anchor, clean, sigma)
    torch.testing.assert_close(velocity, torch.tensor([[4.0, 8.0]]))
    torch.testing.assert_close(
        flow_euler_step(anchor, velocity, sigma, torch.tensor([0.25])),
        torch.tensor([[2.0, 3.0]]),
    )

    paths = flow_isg_paths(
        anchor,
        torch.tensor([[4.0, 8.0]]),
        torch.tensor([[2.0, 4.0]]),
        torch.tensor([[1.0, 2.0]]),
        anchor_sigmas=torch.tensor([0.5]),
        midpoint_sigmas=torch.tensor([0.25]),
        next_sigmas=torch.tensor([0.0]),
    )
    torch.testing.assert_close(paths.teacher_midpoint, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(paths.target_next, torch.tensor([[1.5, 2.0]]))
    torch.testing.assert_close(paths.direct_next, torch.tensor([[2.5, 4.0]]))


def test_isg_supports_paper_mse_and_released_charbonnier_with_stop_gradient() -> None:
    direct = torch.tensor([[1.0, 4.0]], requires_grad=True)
    target = torch.tensor([[3.0, 1.0]], requires_grad=True)
    mse = isg_loss_per_sample(direct, target, loss_type="mse")
    torch.testing.assert_close(mse, torch.tensor([(4.0 + 9.0) / 2.0]))
    mse.sum().backward()
    torch.testing.assert_close(direct.grad, torch.tensor([[-2.0, 3.0]]))
    assert target.grad is None

    charbonnier = isg_loss_per_sample(
        direct.detach(),
        target.detach(),
        loss_type="charbonnier",
        epsilon=1.0e-3,
    )
    expected = (
        torch.sqrt(torch.tensor(4.0 + 1.0e-6))
        + torch.sqrt(torch.tensor(9.0 + 1.0e-6))
    ) / 2.0 - 1.0e-3
    torch.testing.assert_close(charbonnier, expected.reshape(1))


def test_distribution_matching_field_and_proxy_gradient_are_exact() -> None:
    generated = torch.tensor([[2.0, 4.0], [3.0, 7.0]], requires_grad=True)
    fake = torch.tensor([[2.0, 3.0], [4.0, 6.0]])
    teacher = torch.tensor([[1.0, 1.0], [1.0, 3.0]])
    gradient, normalizer = senseflow_distribution_gradient(generated, fake, teacher)
    torch.testing.assert_close(normalizer, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(
        gradient,
        torch.tensor([[0.5, 1.0], [1.0, 1.0]]),
    )
    proxy = senseflow_proxy_loss_per_sample(generated, gradient)
    proxy.sum().backward()
    torch.testing.assert_close(generated.grad, gradient / 2.0)


@pytest.mark.parametrize(
    "sampling",
    ["uniform-schedule-index", "logit-normal-scheduler-index"],
)
def test_score_sigma_sampling_uses_only_the_owned_generator(sampling: str) -> None:
    reference = torch.zeros(5, 2)
    first = sample_score_sigmas(
        reference,
        sampling=sampling,
        minimum_timestep_fraction=0.02,
        maximum_timestep_fraction=0.98,
        flow_shift=3.0,
        generator=torch.Generator().manual_seed(23),
    )
    second = sample_score_sigmas(
        reference,
        sampling=sampling,
        minimum_timestep_fraction=0.02,
        maximum_timestep_fraction=0.98,
        flow_shift=3.0,
        generator=torch.Generator().manual_seed(23),
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert bool(((first > 0.0) & (first <= 1.0)).all())


def test_score_sigma_sampling_matches_released_scheduler_index_math() -> None:
    reference = torch.zeros(4, 2)
    uniform_generator = torch.Generator().manual_seed(31)
    uniform = sample_score_sigmas(
        reference,
        sampling="uniform-schedule-index",
        minimum_timestep_fraction=0.02,
        maximum_timestep_fraction=0.98,
        flow_shift=3.0,
        generator=uniform_generator,
    )
    expected_generator = torch.Generator().manual_seed(31)
    indices = torch.randint(20, 981, (4,), generator=expected_generator)
    base = (indices.float() + 1.0) / 1000.0
    expected_uniform = 3.0 * base / (1.0 + 2.0 * base)
    torch.testing.assert_close(uniform, expected_uniform, rtol=0, atol=0)

    logit_generator = torch.Generator().manual_seed(37)
    logit = sample_score_sigmas(
        reference,
        sampling="logit-normal-scheduler-index",
        minimum_timestep_fraction=0.0,
        maximum_timestep_fraction=1.0,
        flow_shift=3.0,
        generator=logit_generator,
    )
    expected_generator = torch.Generator().manual_seed(37)
    density = torch.randn((4,), generator=expected_generator).sigmoid()
    indices = (density * 1000).long().clamp_max(999)
    base = (1000.0 - indices.float()) / 1000.0
    expected_logit = 3.0 * base / (1.0 + 2.0 * base)
    torch.testing.assert_close(logit, expected_logit, rtol=0, atol=0)


def test_logit_normal_score_bounds_are_behavior_bearing() -> None:
    reference = torch.zeros(64, 2)
    values = sample_score_sigmas(
        reference,
        sampling="logit-normal-scheduler-index",
        minimum_timestep_fraction=0.4,
        maximum_timestep_fraction=0.6,
        flow_shift=1.0,
        generator=torch.Generator().manual_seed(41),
        num_train_timesteps=1000,
    )
    assert bool((values >= 0.4).all())
    assert bool((values <= 0.6).all())

    with pytest.raises(ValueError, match="integer"):
        sample_score_sigmas(
            torch.zeros(1, 2),
            sampling="uniform-schedule-index",
            minimum_timestep_fraction=0.0,
            maximum_timestep_fraction=1.0,
            flow_shift=1.0,
            generator=torch.Generator().manual_seed(43),
            num_train_timesteps=2.5,
        )


def test_vfm_hinge_losses_and_time_weight_reduce_all_heads_per_sample() -> None:
    fake = torch.tensor([[-2.0, 1.0], [0.5, 1.5]])
    real = torch.tensor([[2.0, 0.0], [1.5, 0.5]])
    torch.testing.assert_close(
        senseflow_generator_hinge_loss(fake),
        torch.tensor([0.5, -1.0]),
    )
    torch.testing.assert_close(
        senseflow_discriminator_hinge_loss(real, fake),
        torch.tensor([1.5, 2.25]),
    )
    torch.testing.assert_close(
        senseflow_adversarial_time_weight(torch.tensor([1.0, 0.5, 0.0])),
        torch.tensor([1.0, 0.25, 0.0]),
    )


class _IDAParameters(torch.nn.Module):
    def __init__(
        self,
        trainable: float | list[float],
        frozen: float | list[float],
    ) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.tensor(trainable))
        self.frozen = torch.nn.Parameter(torch.tensor(frozen), requires_grad=False)


def test_fixed_official_trainer_equations_match_source_formula_fixture() -> None:
    fixture = _source_formula_fixture()
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}

    ida_case = inputs["ida"]
    ida_expected = expected["ida"]
    student = _IDAParameters(
        ida_case["student_trainable"],
        ida_case["student_frozen"],
    )
    fake = _IDAParameters(
        ida_case["fake_trainable"],
        ida_case["fake_frozen"],
    )
    update = implicit_distribution_alignment_(student, fake, decay=ida_case["decay"])
    torch.testing.assert_close(
        fake.trainable,
        torch.tensor(ida_expected["fake_trainable"]),
        **tolerance,
    )
    torch.testing.assert_close(
        fake.frozen,
        torch.tensor(ida_expected["fake_frozen"]),
        **tolerance,
    )
    assert update.parameter_count == ida_expected["parameter_count"]
    torch.testing.assert_close(
        update.mean_absolute_shift,
        torch.tensor(ida_expected["mean_absolute_shift"]),
        **tolerance,
    )

    isg_case = inputs["isg"]
    isg_expected = expected["isg"]
    paths = flow_isg_paths(
        torch.tensor(isg_case["anchor_sample"]),
        torch.tensor(isg_case["teacher_velocity"]),
        torch.tensor(isg_case["midpoint_student_velocity"]),
        torch.tensor(isg_case["anchor_student_velocity"]),
        anchor_sigmas=torch.tensor(isg_case["anchor_sigmas"]),
        midpoint_sigmas=torch.tensor(isg_case["midpoint_sigmas"]),
        next_sigmas=torch.tensor(isg_case["next_sigmas"]),
    )
    for field in ("teacher_midpoint", "target_next", "direct_next"):
        torch.testing.assert_close(
            getattr(paths, field),
            torch.tensor(isg_expected[field]),
            **tolerance,
        )
    torch.testing.assert_close(
        isg_loss_per_sample(paths.direct_next, paths.target_next, loss_type="mse"),
        torch.tensor(isg_expected["mse_per_sample"]),
        **tolerance,
    )

    dmd_case = inputs["dmd"]
    dmd_expected = expected["dmd"]
    generated = torch.tensor(dmd_case["generated_clean"])
    gradient, normalizer = senseflow_distribution_gradient(
        generated,
        torch.tensor(dmd_case["fake_clean"]),
        torch.tensor(dmd_case["teacher_clean"]),
        normalization_epsilon=dmd_case["normalization_epsilon"],
    )
    torch.testing.assert_close(gradient, torch.tensor(dmd_expected["gradient"]), **tolerance)
    torch.testing.assert_close(normalizer, torch.tensor(dmd_expected["normalizer"]), **tolerance)
    torch.testing.assert_close(
        senseflow_proxy_loss_per_sample(generated, gradient),
        torch.tensor(dmd_expected["proxy_per_sample"]),
        **tolerance,
    )


def test_ida_updates_only_student_trainable_parameters_with_post_step_equation() -> None:
    student = _IDAParameters(4.0, 9.0)
    fake = _IDAParameters(2.0, 3.0)
    assert audit_ida_alignment(student, fake) == ("trainable",)
    update = implicit_distribution_alignment_(student, fake, decay=0.75)

    torch.testing.assert_close(fake.trainable, torch.tensor(2.5))
    torch.testing.assert_close(fake.frozen, torch.tensor(3.0))
    assert update.parameter_count == 1
    torch.testing.assert_close(update.mean_absolute_shift, torch.tensor(0.5))


def test_ida_rejects_parameter_inventory_mismatch() -> None:
    with pytest.raises(ValueError, match="inventories differ"):
        audit_ida_alignment(_IDAParameters(1.0, 2.0), torch.nn.Linear(1, 1))

    student = _IDAParameters(1.0, 2.0)
    fake = _IDAParameters(3.0, 4.0)
    fake.frozen.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable parameter masks differ"):
        audit_ida_alignment(student, fake)
