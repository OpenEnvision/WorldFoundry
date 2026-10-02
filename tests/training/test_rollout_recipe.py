from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.training.recipes import (
    LocalRolloutSpec,
    PostTrainingRecipe,
    RayRolloutSpec,
)

ROOT = Path(__file__).resolve().parents[2]


def test_default_local_rollout_keeps_existing_recipe_serialization_stable() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/t2v_turbo_distillation.yaml")
    assert isinstance(recipe.rollout, LocalRolloutSpec)
    assert "rollout" not in recipe.to_dict()
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe


def test_formal_qwen_ray_profile_round_trips_runtime_placement() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray.yaml")
    assert isinstance(recipe.rollout, RayRolloutSpec)
    assert recipe.rollout.trainer_binding == "actor"
    assert recipe.rollout.placement == "separate"
    assert recipe.rollout.trainer_devices == 1
    assert recipe.rollout.pool.accelerator_resource == "GPU"
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe


def test_ray_lora_transfer_requires_lora_tuning() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray.yaml").to_dict()
    recipe["rollout"]["weight_kind"] = "lora"
    with pytest.raises(ValueError, match="requires tuning.mode=lora"):
        PostTrainingRecipe.from_mapping(recipe)


def test_ray_rollout_is_rejected_when_the_algorithm_has_no_runtime_consumer() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/t2v_turbo_distillation.yaml").to_dict()
    recipe["rollout"] = {
        "backend": "ray",
        "pool": {
            "num_devices": 1,
            "devices_per_node": 1,
        },
        "rollout_devices": 1,
    }
    with pytest.raises(ValueError, match="flow-policy and grouped token-policy"):
        PostTrainingRecipe.from_mapping(recipe)


def test_actor_colocate_requires_a_second_device_slot() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray.yaml").to_dict()
    rollout = recipe["rollout"]
    rollout["trainer_binding"] = "actor"
    rollout["placement"] = "colocate"
    rollout["trainer_devices"] = 1
    with pytest.raises(ValueError, match="workers_per_device"):
        PostTrainingRecipe.from_mapping(recipe)


def test_actor_colocate_cannot_request_more_rollout_than_trainer_devices() -> None:
    recipe = PostTrainingRecipe.from_file(
        ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray_colocated.yaml"
    ).to_dict()
    recipe["rollout"]["rollout_devices"] = 2
    recipe["rollout"]["pool"]["num_devices"] = 2
    with pytest.raises(ValueError, match="cannot exceed trainer_devices"):
        PostTrainingRecipe.from_mapping(recipe)


def test_ray_device_counts_do_not_truncate_fractional_values() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray.yaml").to_dict()
    recipe["rollout"]["pool"]["num_devices"] = 1.5
    with pytest.raises(ValueError, match="positive integer"):
        PostTrainingRecipe.from_mapping(recipe)


def test_formal_qwen_actor_colocate_profile_round_trips() -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray_colocated.yaml")
    assert isinstance(recipe.rollout, RayRolloutSpec)
    assert recipe.rollout.trainer_binding == "actor"
    assert recipe.rollout.placement == "colocate"
    assert recipe.rollout.pool.workers_per_device == 2
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
