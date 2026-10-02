from __future__ import annotations

from dataclasses import replace

import pytest

from worldfoundry.training.post_training.distillation.senseflow import (
    SenseFlowConfig,
    SenseFlowOptimizerConfig,
)
from worldfoundry.training.recipes.post_training.algorithms.senseflow import (
    SenseFlowAlgorithmSpec,
    SenseFlowScheduleSpec,
    parse_senseflow_algorithm,
)
from worldfoundry.training.recipes.post_training.common import plain_data
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import OptimizerSpec


def _checkpoints() -> dict[str, str]:
    return {
        "teacher_checkpoint": "teacher",
        "fake_score_checkpoint": "fake",
        "discriminator_checkpoint": "discriminator",
    }


def _optimizer_payload(learning_rate: float) -> dict[str, object]:
    return {
        "type": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 2,
    }


def _post_training_payload(spec: SenseFlowAlgorithmSpec) -> dict[str, object]:
    learning_rate = 1.0e-5 if spec.preset == "flux-released" else 1.0e-6
    return {
        "run": {"id": "senseflow-test", "output_dir": "/tmp/senseflow"},
        "model": {"recipe": "toy-flow", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "/tmp/manifest.json"},
        "algorithm": plain_data(spec),
        "optimizer": _optimizer_payload(learning_rate),
        "fake_score_optimizer": _optimizer_payload(learning_rate),
        "discriminator_optimizer": _optimizer_payload(learning_rate),
        "export": {"format": "safetensors"},
    }


def test_released_recipe_presets_are_distinct_and_round_trip_strictly() -> None:
    large = SenseFlowAlgorithmSpec.sd35_large_released(**_checkpoints())
    medium = SenseFlowAlgorithmSpec.sd35_medium_released(**_checkpoints())
    flux = SenseFlowAlgorithmSpec.flux_released(**_checkpoints())

    assert large.generator_update_interval == 5
    assert large.ida_decay == pytest.approx(0.97)
    assert large.isg_teacher_guidance == pytest.approx((5.0, 5.0))
    assert large.dmd_teacher_guidance == pytest.approx((3.0, 10.0))
    assert large.lr_warmup_start_ratio == pytest.approx(1.0)
    assert medium.generator_update_interval == 10
    assert medium.ida_decay == pytest.approx(0.98)
    assert medium.isg_weight == pytest.approx(0.5)
    assert medium.isg_teacher_guidance == pytest.approx((2.0, 4.0))
    assert medium.dmd_teacher_guidance == pytest.approx((2.0, 8.0))
    assert flux.schedule.sigmas == pytest.approx((1.0, 0.904, 0.759, 0.512))
    assert flux.score_sampling == "logit-normal-scheduler-index"
    assert flux.generator_adversarial_weight == pytest.approx(2.0)
    assert flux.lr_warmup_start_ratio == pytest.approx(0.5)
    with pytest.raises(ValueError, match="released preset"):
        replace(flux, generator_adversarial_weight=0.1)

    for spec in (large, medium, flux):
        payload = plain_data(spec)
        assert parse_senseflow_algorithm(payload) == spec
        assert SenseFlowConfig.from_recipe(spec).schedule.sigmas == pytest.approx(
            spec.schedule.sigmas
        )
    payload = plain_data(large)
    assert isinstance(payload, dict)
    payload["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        parse_senseflow_algorithm(payload)


def test_post_training_recipe_round_trip_registers_all_senseflow_roles() -> None:
    spec = SenseFlowAlgorithmSpec.flux_released(**_checkpoints())
    recipe = PostTrainingRecipe.from_mapping(_post_training_payload(spec))
    assert recipe.algorithm == spec
    assert recipe.model.checkpoint == "student"
    assert recipe.fake_score_optimizer is not None
    assert recipe.discriminator_optimizer is not None
    assert recipe.guidance_optimizer is None
    assert recipe.to_dict()["algorithm"] == plain_data(spec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    missing = _post_training_payload(spec)
    missing.pop("discriminator_optimizer")
    with pytest.raises(ValueError, match="discriminator_optimizer"):
        PostTrainingRecipe.from_mapping(missing)


def test_recipe_schedule_rejects_declared_sigmas_that_differ_from_mapping() -> None:
    with pytest.raises(ValueError, match="scheduler mapping"):
        SenseFlowScheduleSpec(
            timesteps=(10,),
            sigmas=(0.6,),
            isg_margin=1,
            num_train_timesteps=20,
        )
    with pytest.raises(TypeError, match="integer"):
        SenseFlowScheduleSpec(
            timesteps=(10,),
            sigmas=(0.5,),
            isg_margin=1,
            num_train_timesteps=20.5,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        SenseFlowAlgorithmSpec(
            schedule=SenseFlowScheduleSpec(
                timesteps=(10,),
                sigmas=(0.5,),
                isg_margin=1,
                num_train_timesteps=20,
            ),
            teacher_checkpoint=None,
            fake_score_checkpoint="fake",
            discriminator_checkpoint="discriminator",
        )


def test_every_algorithm_recipe_field_maps_to_runtime_behavior() -> None:
    spec = SenseFlowAlgorithmSpec(
        schedule=SenseFlowScheduleSpec(
            timesteps=(10,),
            sigmas=(0.5,),
            isg_margin=1,
            num_train_timesteps=20,
            adversarial_scales=(0.4,),
        ),
        **_checkpoints(),
        generator_update_interval=3,
        backward_simulation_probability=0.25,
        ida_decay=0.8,
        isg_weight=0.7,
        isg_loss="mse",
        isg_epsilon=2.0e-3,
        isg_teacher_guidance=(1.5, 2.5),
        dmd_teacher_guidance=(2.5, 4.5),
        score_sampling="logit-normal-scheduler-index",
        fake_score_sampling="uniform-schedule-index",
        score_min_timestep_fraction=0.1,
        score_max_timestep_fraction=0.8,
        fake_score_min_timestep_fraction=0.2,
        fake_score_max_timestep_fraction=0.9,
        score_flow_shift=1.5,
        normalization_epsilon=1.0e-6,
        distribution_matching_weight=0.9,
        generator_adversarial_weight=0.2,
        fake_score_weight=1.1,
        discriminator_weight=1.2,
        seed=123,
        student_scheduler_cadence="generator-update",
        lr_warmup_steps=7,
        lr_warmup_start_ratio=0.25,
    )
    config = SenseFlowConfig.from_recipe(spec)
    for name in (
        "generator_update_interval",
        "backward_simulation_probability",
        "ida_decay",
        "isg_weight",
        "isg_loss",
        "isg_epsilon",
        "isg_teacher_guidance",
        "dmd_teacher_guidance",
        "score_sampling",
        "fake_score_sampling",
        "score_min_timestep_fraction",
        "score_max_timestep_fraction",
        "fake_score_min_timestep_fraction",
        "fake_score_max_timestep_fraction",
        "score_flow_shift",
        "normalization_epsilon",
        "distribution_matching_weight",
        "generator_adversarial_weight",
        "fake_score_weight",
        "discriminator_weight",
        "seed",
        "student_scheduler_cadence",
    ):
        assert getattr(config, name) == getattr(spec, name)
    assert config.schedule.adversarial_scales == pytest.approx((0.4,))

    student = OptimizerSpec(
        type="adamw",
        learning_rate=1.0e-3,
        weight_decay=0.01,
        betas=(0.8, 0.9),
        epsilon=1.0e-7,
        max_grad_norm=2.0,
        gradient_accumulation_steps=3,
    )
    fake = replace(student, learning_rate=2.0e-3, max_grad_norm=3.0)
    discriminator = replace(student, learning_rate=3.0e-3, max_grad_norm=4.0)
    optimizer = SenseFlowOptimizerConfig.from_recipe(
        spec,
        student,
        fake,
        discriminator,
    )
    assert optimizer.student_learning_rate == pytest.approx(1.0e-3)
    assert optimizer.fake_score_learning_rate == pytest.approx(2.0e-3)
    assert optimizer.discriminator_learning_rate == pytest.approx(3.0e-3)
    assert optimizer.betas == pytest.approx((0.8, 0.9))
    assert optimizer.epsilon == pytest.approx(1.0e-7)
    assert optimizer.weight_decay == pytest.approx(0.01)
    assert optimizer.student_max_grad_norm == pytest.approx(2.0)
    assert optimizer.fake_score_max_grad_norm == pytest.approx(3.0)
    assert optimizer.discriminator_max_grad_norm == pytest.approx(4.0)
    assert optimizer.gradient_accumulation_steps == 3
    assert optimizer.warmup_steps == 7
    assert optimizer.warmup_start_ratio == pytest.approx(0.25)


def test_optimizer_recipe_mismatches_fail_instead_of_becoming_ignored_fields() -> None:
    spec = SenseFlowAlgorithmSpec.sd35_large_released(**_checkpoints())
    student = OptimizerSpec(type="adamw", learning_rate=1.0e-6)
    with pytest.raises(ValueError, match="betas"):
        SenseFlowOptimizerConfig.from_recipe(
            spec,
            student,
            replace(student, betas=(0.8, 0.999)),
            student,
        )
    with pytest.raises(ValueError, match="accumulation"):
        SenseFlowOptimizerConfig.from_recipe(
            spec,
            student,
            replace(student, gradient_accumulation_steps=2),
            student,
        )
    with pytest.raises(ValueError, match="AdamW"):
        SenseFlowOptimizerConfig.from_recipe(
            spec,
            student,
            student,
            OptimizerSpec(
                type="came",
                learning_rate=1.0e-6,
                betas=(0.9, 0.999, 0.9999),
                epsilon=(1.0e-30, 1.0e-16),
            ),
        )
    with pytest.raises(ValueError, match="released preset"):
        SenseFlowOptimizerConfig.from_recipe(
            spec,
            replace(student, learning_rate=2.0e-6),
            student,
            student,
        )
