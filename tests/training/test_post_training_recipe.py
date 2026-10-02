from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.training.recipes import (
    POST_TRAINING_RECIPE_SCHEMA,
    DanceGRPOAlgorithmSpec,
    DMDAlgorithmSpec,
    FlowDPPOAlgorithmSpec,
    FlowGRPOAlgorithmSpec,
    FlowPolicyAlgorithmSpec,
    MixGRPOAlgorithmSpec,
    PostTrainingRecipe,
    SIDAlgorithmSpec,
)


def _dmd_mapping() -> dict:
    return {
        "schema": POST_TRAINING_RECIPE_SCHEMA,
        "run": {"id": "native_dmd", "output_dir": "runs/native-dmd"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
        "tuning": {"mode": "lora", "preset": "wan-attention", "rank": 8, "alpha": 16},
        "data": {"manifest": "data/latents.jsonl"},
        "algorithm": {
            "type": "dmd",
            "student_timesteps": [1000, 757, 522],
            "student_sigmas": [1.0, 0.757, 0.522],
            "real_score_checkpoint": "teacher",
            "fake_score_checkpoint": "critic-init",
        },
        "optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
        "fake_score_optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
    }


def test_native_post_training_recipe_is_strict_and_has_no_provider() -> None:
    payload = _dmd_mapping()
    payload["distributed"] = {"backend": "single", "timeout_seconds": 91}
    recipe = PostTrainingRecipe.from_mapping(payload)
    restored = PostTrainingRecipe.from_mapping(recipe.to_dict())

    assert isinstance(recipe.algorithm, DMDAlgorithmSpec)
    assert recipe.algorithm.student_sigmas == (1.0, 0.757, 0.522)
    assert restored == recipe
    assert "provider" not in recipe.to_dict()
    assert recipe.execution_owner == "worldfoundry-native"
    assert recipe.distributed.timeout_seconds == 91

    payload = _dmd_mapping()
    payload["provider"] = {"name": "unirl"}
    with pytest.raises(ValueError, match="unknown fields.*provider"):
        PostTrainingRecipe.from_mapping(payload)


def test_algorithm_specific_optimizer_contracts_fail_closed() -> None:
    dmd = _dmd_mapping()
    del dmd["fake_score_optimizer"]
    with pytest.raises(ValueError, match="DMD requires"):
        PostTrainingRecipe.from_mapping(dmd)

    flow = _dmd_mapping()
    flow["algorithm"] = {
        "type": "flow-grpo",
        "sigmas": [1.0, 0.5, 0.0],
        "sde_step_indices": [0, 1],
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }
    with pytest.raises(ValueError, match="cannot configure fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(flow)

    del flow["fake_score_optimizer"]
    flow["algorithm"]["reference_kl_weight"] = 0.1
    with pytest.raises(ValueError, match="explicit reference_checkpoint"):
        PostTrainingRecipe.from_mapping(flow)

    flow["algorithm"].pop("reference_kl_weight")
    flow["algorithm"]["clip_schedule"] = "cosine-decay"
    with pytest.raises(TypeError, match="requires integer clip_schedule_steps"):
        PostTrainingRecipe.from_mapping(flow)

    flow["algorithm"]["clip_schedule_steps"] = 100
    scheduled_recipe = PostTrainingRecipe.from_mapping(flow)
    assert scheduled_recipe.algorithm.clip_schedule == "cosine-decay"
    assert scheduled_recipe.algorithm.clip_schedule_steps == 100

    flow["algorithm"]["type"] = "flow-dppo"
    flow["algorithm"].pop("clip_range", None)
    flow["algorithm"].pop("clip_schedule")
    flow["algorithm"].pop("clip_schedule_steps")
    flow["algorithm"]["kl_mask_threshold"] = 1.0e-5
    flow["algorithm"]["add_kl_coefficient"] = True
    recipe = PostTrainingRecipe.from_mapping(flow)
    assert isinstance(recipe.algorithm, FlowDPPOAlgorithmSpec)
    assert isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec)


def test_post_training_export_contract_matches_tuning_mode() -> None:
    lora = _dmd_mapping()
    lora["export"] = {"format": "safetensors"}
    with pytest.raises(ValueError, match="LoRA post-training export.format"):
        PostTrainingRecipe.from_mapping(lora)

    full = _dmd_mapping()
    full["tuning"] = {"mode": "full"}
    full["export"] = {"format": "distributed-checkpoint"}
    recipe = PostTrainingRecipe.from_mapping(full)
    assert recipe.export.format == "distributed-checkpoint"

    full["export"] = {
        "format": "safetensors",
        "options": {"max_shard_size_bytes": 1024},
    }
    recipe = PostTrainingRecipe.from_mapping(full)
    assert recipe.export.options["max_shard_size_bytes"] == 1024

    full["export"] = {"format": "safetensors", "merge_adapter": True}
    with pytest.raises(ValueError, match="unknown fields.*merge_adapter"):
        PostTrainingRecipe.from_mapping(full)


@pytest.mark.parametrize(
    ("filename", "algorithm_type"),
    (
        ("wan_1p3b_dmd.yaml", DMDAlgorithmSpec),
        ("wan_1p3b_flow_grpo.yaml", FlowGRPOAlgorithmSpec),
        ("wan_1p3b_videoalign_flow_grpo.yaml", FlowGRPOAlgorithmSpec),
        ("wan_1p3b_flow_dppo.yaml", FlowDPPOAlgorithmSpec),
        ("wan_1p3b_dance_grpo.yaml", DanceGRPOAlgorithmSpec),
        ("wan_1p3b_mix_grpo.yaml", MixGRPOAlgorithmSpec),
        ("sana_sprint_600m_sid.yaml", SIDAlgorithmSpec),
    ),
)
def test_checked_in_native_post_training_recipes_parse(
    filename: str,
    algorithm_type: type,
) -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training" / filename)

    assert isinstance(recipe.algorithm, algorithm_type)
    assert recipe.execution_owner == "worldfoundry-native"
    assert recipe.distributed.dp_shard == "auto"
    if isinstance(recipe.algorithm, DMDAlgorithmSpec):
        assert recipe.data.max_latent_tokens_per_microbatch is not None
    elif isinstance(recipe.algorithm, SIDAlgorithmSpec):
        assert recipe.algorithm.num_train_timesteps == 1000
        assert recipe.fake_score_optimizer is not None
        assert recipe.tuning.mode == "full"
    else:
        assert recipe.algorithm.num_train_timesteps == 1000
        assert recipe.algorithm.reference_checkpoint is None
        assert recipe.algorithm.reward_model.normalization_epsilon == 0.0


def test_dance_and_mix_recipe_invariants_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/post_training"
    dance = PostTrainingRecipe.from_file(root / "wan_1p3b_dance_grpo.yaml").to_dict()
    dance_algorithm = dance["algorithm"]
    assert isinstance(dance_algorithm, dict)
    dance_algorithm["init_same_noise"] = False
    with pytest.raises(ValueError, match="shared initial noise"):
        PostTrainingRecipe.from_mapping(dance)

    mix = PostTrainingRecipe.from_file(root / "wan_1p3b_mix_grpo.yaml").to_dict()
    mix_algorithm = mix["algorithm"]
    assert isinstance(mix_algorithm, dict)
    mix_algorithm["advantage_normalization"] = "group-population-variance"
    with pytest.raises(ValueError, match="group-sample-std"):
        PostTrainingRecipe.from_mapping(mix)


def test_rollout_strategy_fields_are_behavioral_and_mutually_exclusive() -> None:
    flow = _dmd_mapping()
    flow.pop("fake_score_optimizer")
    flow["algorithm"] = {
        "type": "flow-grpo",
        "sigmas": [1.0, 0.75, 0.5, 0.25, 0.0],
        "sde_window": {
            "window_size": 2,
            "iterations_per_window": 3,
            "stride": 1,
            "rollback": True,
        },
        "transition_strategy": "constant-diffusion",
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }
    recipe = PostTrainingRecipe.from_mapping(flow)
    assert recipe.algorithm.sigma_max is None
    assert recipe.algorithm.transition_strategy == "constant-diffusion"
    assert recipe.algorithm.sde_window is not None
    assert recipe.algorithm.sde_window.iterations_per_window == 3

    flow["algorithm"]["sigma_max"] = 0.75
    with pytest.raises(ValueError, match="sigma_max is unused"):
        PostTrainingRecipe.from_mapping(flow)

    del flow["algorithm"]["sigma_max"]
    flow["algorithm"]["sde_step_indices"] = [0]
    with pytest.raises(ValueError, match="cannot be combined"):
        PostTrainingRecipe.from_mapping(flow)


def test_omitted_sde_window_options_match_unirl_defaults() -> None:
    flow = _dmd_mapping()
    flow.pop("fake_score_optimizer")
    flow["algorithm"] = {
        "type": "flow-grpo",
        "sigmas": [1.0, 0.75, 0.5, 0.25, 0.0],
        "sde_window": {
            "window_size": 2,
            "iterations_per_window": 3,
        },
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }

    recipe = PostTrainingRecipe.from_mapping(flow)

    assert recipe.algorithm.sde_window is not None
    assert recipe.algorithm.sde_window.stride == 2
    assert recipe.algorithm.sde_window.rollback is False
