from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from dataclasses import replace
from pathlib import Path

import pytest

from worldfoundry.training.distributed import ParallelPlan
from worldfoundry.training.engine import validate_wan_flow_policy_recipe
from worldfoundry.training.engine.wan.rollout_materialization import (
    _audit_distributed_rollout_collectives,
)
from worldfoundry.training.post_training import flow_match_sigma_schedule
from worldfoundry.training.recipes import PostTrainingRecipe


def _recipe(filename: str = "wan_1p3b_flow_grpo.yaml") -> PostTrainingRecipe:
    root = Path(__file__).resolve().parents[2]
    return PostTrainingRecipe.from_file(root / "configs/post_training" / filename)


def test_wan_flow_policy_recipes_resolve_dynamic_world_sizes() -> None:
    recipe = _recipe()
    algorithm, data_plan = validate_wan_flow_policy_recipe(recipe)

    assert algorithm.group_size == 16
    assert algorithm.num_train_timesteps == 1000
    assert algorithm.sigmas == flow_match_sigma_schedule(14, shift=5.0)
    assert algorithm.sde_timestep_fraction == (0.0, 0.5)
    assert algorithm.num_sde_steps == 7
    assert algorithm.guidance_scale == 1.0
    assert algorithm.init_same_noise is False
    assert algorithm.sigma_max == algorithm.sigmas[1]
    assert algorithm.trajectory_dtype == "float16"
    assert algorithm.updates_per_trajectory == 2
    assert algorithm.old_log_prob_source == "replay"
    assert algorithm.advantage_normalization == "group-mean-global-population-std"
    assert data_plan.generation == {
        "height": 720,
        "width": 1280,
        "num_frames": 5,
    }
    assert ParallelPlan.resolve(recipe.distributed, world_size=1).dp_shard == 1
    assert ParallelPlan.resolve(recipe.distributed, world_size=3).dp_shard == 3
    assert ParallelPlan.resolve(recipe.distributed, world_size=12).dp_shard == 12

    dppo, dppo_plan = validate_wan_flow_policy_recipe(_recipe("wan_1p3b_flow_dppo.yaml"))
    assert dppo.type == "flow-dppo"
    assert dppo.updates_per_trajectory == 2
    assert dppo_plan.generation == {
        "height": 256,
        "width": 416,
        "num_frames": 17,
    }


def test_distributed_rollout_rejects_rank_uneven_chunked_collectives() -> None:
    with pytest.raises(ValueError, match="rollout forward"):
        _audit_distributed_rollout_collectives(
            world_size=2,
            tail_policy="uneven",
            rollout_forward_batch_size=1,
            replay_microbatch_size=None,
        )
    _audit_distributed_rollout_collectives(
        world_size=2,
        tail_policy="uneven",
        rollout_forward_batch_size=None,
        replay_microbatch_size=None,
    )
    _audit_distributed_rollout_collectives(
        world_size=7,
        tail_policy="pad",
        rollout_forward_batch_size=1,
        replay_microbatch_size=1,
    )
    _audit_distributed_rollout_collectives(
        world_size=3,
        tail_policy="drop",
        rollout_forward_batch_size=1,
        replay_microbatch_size=1,
    )


def test_source_profile_keeps_semantics_without_fixing_world_size() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan_1p3b_videoalign_flow_grpo.yaml")
    algorithm, data_plan = validate_wan_flow_policy_recipe(recipe)

    assert algorithm.sigmas == flow_match_sigma_schedule(16, shift=3.0)
    assert algorithm.sigma_max == algorithm.sigmas[1]
    assert algorithm.sde_timestep_fraction == (0.0, 0.6)
    assert algorithm.num_sde_steps == 8
    assert algorithm.guidance_scale == 5.0
    assert algorithm.init_same_noise is True
    assert algorithm.eta == 0.25
    assert algorithm.group_size == 24
    assert algorithm.trajectory_dtype == "float32"
    assert data_plan.rollout_forward_batch_size == 1
    assert data_plan.replay_microbatch_size == 1
    assert ParallelPlan.resolve(recipe.distributed, world_size=1).dp_shard == 1
    assert ParallelPlan.resolve(recipe.distributed, world_size=7).dp_shard == 7


@pytest.mark.parametrize(
    ("filename", "algorithm_type", "updates"),
    (
        ("wan_1p3b_dance_grpo.yaml", "dance-grpo", 3),
        ("wan_1p3b_mix_grpo.yaml", "mix-grpo", 4),
    ),
)
def test_dance_mix_profiles_keep_algorithm_semantics_at_arbitrary_world_sizes(
    filename: str,
    algorithm_type: str,
    updates: int,
) -> None:
    recipe = _recipe(filename)
    algorithm, data_plan = validate_wan_flow_policy_recipe(recipe)

    assert algorithm.type == algorithm_type
    assert algorithm.group_size == 12
    assert algorithm.updates_per_trajectory == updates
    assert algorithm.init_same_noise is True
    assert algorithm.advantage_normalization == "group-sample-std"
    assert data_plan.prompt_batch_size == 1
    for world_size in (1, 3, 7, 12):
        assert ParallelPlan.resolve(recipe.distributed, world_size=world_size).dp_shard == world_size


def test_wan_flow_policy_recipe_rejects_ambiguous_data_and_geometry() -> None:
    recipe = _recipe()
    unknown = replace(
        recipe,
        data=replace(
            recipe.data,
            options={**recipe.data.options, "external_launcher": "unirl"},
        ),
    )
    with pytest.raises(ValueError, match="unknown.*external_launcher"):
        validate_wan_flow_policy_recipe(unknown)

    invalid_geometry = replace(
        recipe,
        data=replace(
            recipe.data,
            options={
                **recipe.data.options,
                "generation": {
                    "height": 250,
                    "width": 416,
                    "num_frames": 17,
                },
            },
        ),
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_wan_flow_policy_recipe(invalid_geometry)


def test_wan21_flow_policy_rejects_unconsumed_ray_rollout() -> None:
    root = Path(__file__).resolve().parents[2]
    ray_recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan22_t2v_a14b_ray_flow_grpo.yaml")
    with pytest.raises(ValueError, match="requires local rollout"):
        validate_wan_flow_policy_recipe(replace(_recipe(), rollout=ray_recipe.rollout))


def test_wan_flow_policy_runtime_does_not_import_or_launch_reference_repositories() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "worldfoundry/training/engine/wan"
    source = (package / "flow_policy.py").read_text(encoding="utf-8")
    lowered = source.lower()

    assert not (package / "flow_grpo.py").exists()
    assert not (package / "flow_grpo_recipe.py").exists()
    assert not (package / "flow_grpo_run.py").exists()
    assert "NativeFlowGRPOTrainingSession" not in source
    assert "NativeFlowDPPOTrainingSession" not in source
    assert "stack.session_type(" in source
    assert "import unirl" not in lowered
    assert "subprocess" not in lowered
    assert "external_launcher" not in lowered
