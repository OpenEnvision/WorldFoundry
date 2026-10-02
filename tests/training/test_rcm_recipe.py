from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from worldfoundry.training.post_training.distillation.rcm import (
    CausalRCMConfig,
    RCMConfig,
    causal_rcm_config_from_algorithm,
    rcm_config_from_algorithm,
)
from worldfoundry.training.recipes.post_training.algorithms.rcm import (
    CausalRCMAlgorithmSpec,
    RCMAlgorithmSpec,
    parse_causal_rcm_algorithm,
    parse_rcm_algorithm,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe


def _execution_fields(config_type: type[object]) -> set[str]:
    return {field.name for field in fields(config_type)}


def _algorithm_fields(spec_type: type[object]) -> set[str]:
    return {field.name for field in fields(spec_type)} - {"type"}


_ROLE_FIELDS = {
    "teacher_checkpoint",
    "fake_score_checkpoint",
    "causal_teacher_checkpoint",
    "bidirectional_teacher_checkpoint",
}


def test_bidirectional_recipe_has_exactly_the_execution_config_inventory() -> None:
    assert _algorithm_fields(RCMAlgorithmSpec) - _ROLE_FIELDS == _execution_fields(
        RCMConfig
    )
    spec = RCMAlgorithmSpec(
        consistency_mode="discrete",
        tangent_warmup_steps=7,
        student_update_frequency=4,
        teacher_guidance_scale=4.5,
        consistency_loss_scale=80,
        dmd_loss_scale=0.7,
        max_rollout_steps=3,
        generator_time_mean=-0.7,
        generator_time_std=1.3,
        score_time_mean=0.2,
        score_time_std=1.1,
        tangent_normalization_constant=0.2,
        dcm_total_steps=32,
        dcm_skipping_interval_steps=2,
        dcm_timestep_shift=4,
        fixed_rollout_timesteps=(1.2, 0.8),
    )
    config = rcm_config_from_algorithm(spec)
    expected = asdict(spec)
    expected.pop("type")
    expected.pop("teacher_checkpoint")
    expected.pop("fake_score_checkpoint")
    assert asdict(config) == expected


def test_causal_recipe_has_exactly_the_execution_config_inventory() -> None:
    assert _algorithm_fields(CausalRCMAlgorithmSpec) - _ROLE_FIELDS == _execution_fields(
        CausalRCMConfig
    )
    spec = CausalRCMAlgorithmSpec(
        consistency_mode="continuous",
        tangent_warmup_steps=9,
        student_update_frequency=6,
        causal_teacher_guidance_scale=2.5,
        bidirectional_teacher_guidance_scale=4.0,
        consistency_loss_scale=50,
        dmd_loss_scale=0.5,
        max_rollout_steps=3,
        generator_time_mean=-0.5,
        generator_time_std=1.2,
        score_timestep_shift=4,
        tangent_normalization_constant=0.2,
        dcm_total_steps=24,
        dcm_skipping_interval_steps=2,
        dcm_timestep_shift=2.5,
        first_chunk_frames=1,
        chunk_frames=2,
        spatial_patch_area=16,
        rollout_timesteps=(0.9, 0.7),
    )
    config = causal_rcm_config_from_algorithm(spec)
    expected = asdict(spec)
    expected.pop("type")
    expected.pop("causal_teacher_checkpoint")
    expected.pop("bidirectional_teacher_checkpoint")
    expected.pop("fake_score_checkpoint")
    assert asdict(config) == expected


def test_rcm_parsers_are_strict_and_do_not_accept_governance_metadata() -> None:
    assert parse_rcm_algorithm({"type": "rcm"}).type == "rcm"
    assert parse_causal_rcm_algorithm({"type": "causal-rcm"}).type == "causal-rcm"
    with pytest.raises(ValueError, match="unknown fields"):
        parse_rcm_algorithm({"type": "rcm", "metadata": {"note": "dead"}})
    with pytest.raises(ValueError, match="unknown fields"):
        parse_causal_rcm_algorithm(
            {"type": "causal-rcm", "source_revision": "not-runtime"}
        )


def _post_training_mapping(*, algorithm_type: str, dmd_loss_scale: float) -> dict:
    payload = {
        "run": {"id": f"{algorithm_type}-test", "output_dir": "unused"},
        "model": {
            "recipe": "wan2.1-t2v-1.3b",
            "checkpoint": "student-checkpoint",
        },
        "tuning": {"mode": "full"},
        "data": {"manifest": "training.jsonl"},
        "algorithm": {
            "type": algorithm_type,
            "dmd_loss_scale": dmd_loss_scale,
        },
        "optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
        "export": {"format": "safetensors"},
    }
    algorithm = payload["algorithm"]
    if algorithm_type == "rcm":
        algorithm["teacher_checkpoint"] = "teacher-checkpoint"
        algorithm["fake_score_checkpoint"] = (
            "fake-score-checkpoint" if dmd_loss_scale > 0 else None
        )
    else:
        algorithm["causal_teacher_checkpoint"] = "causal-teacher-checkpoint"
        algorithm["bidirectional_teacher_checkpoint"] = (
            "bidirectional-teacher-checkpoint" if dmd_loss_scale > 0 else None
        )
        algorithm["fake_score_checkpoint"] = (
            "fake-score-checkpoint" if dmd_loss_scale > 0 else None
        )
    if dmd_loss_scale > 0:
        payload["fake_score_optimizer"] = {
            "type": "adamw",
            "learning_rate": 4.0e-7,
        }
    return payload


@pytest.mark.parametrize(
    ("algorithm_type", "algorithm_class"),
    (("rcm", RCMAlgorithmSpec), ("causal-rcm", CausalRCMAlgorithmSpec)),
)
def test_rcm_algorithms_round_trip_through_the_native_recipe(
    algorithm_type: str,
    algorithm_class: type,
) -> None:
    recipe = PostTrainingRecipe.from_mapping(
        _post_training_mapping(algorithm_type=algorithm_type, dmd_loss_scale=1.0)
    )

    assert isinstance(recipe.algorithm, algorithm_class)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe


def test_rcm_recipe_optimizer_inventory_tracks_whether_dmd_executes() -> None:
    missing = _post_training_mapping(algorithm_type="rcm", dmd_loss_scale=1.0)
    missing.pop("fake_score_optimizer")
    with pytest.raises(ValueError, match="DMD requires fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(missing)

    pure_consistency = _post_training_mapping(
        algorithm_type="rcm",
        dmd_loss_scale=0.0,
    )
    pure_consistency["fake_score_optimizer"] = {
        "type": "adamw",
        "learning_rate": 4.0e-7,
    }
    with pytest.raises(ValueError, match="without DMD"):
        PostTrainingRecipe.from_mapping(pure_consistency)
