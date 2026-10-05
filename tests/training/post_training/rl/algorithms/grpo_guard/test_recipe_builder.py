from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
import worldfoundry.training.post_training.rl as rl  # noqa: E402
import worldfoundry.training.post_training.rl.algorithms as algorithms  # noqa: E402
import worldfoundry.training.recipes as recipes  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms import grpo_guard  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms.grpo_guard import (  # noqa: E402
    NativeGRPOGuardEngine,
    NativeGRPOGuardTrainingSession,
)
from worldfoundry.training.recipes import (  # noqa: E402
    GRPOGuardAlgorithmSpec,
    PostTrainingRecipe,
)


class _FlowPredictor:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(2, 2, bias=False)

    def predict_velocity(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning, branch
        self.module.train(training)
        return self.module(noisy_latents)

    def predict_clean(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        sigma = torch.as_tensor(
            sigmas,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        ).reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - sigma * velocity


def _recipe_mapping() -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "grpo-guard-test", "output_dir": "runs/grpo-guard-test"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "policy"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "grpo-guard",
            "sigmas": [1.0, 0.7, 0.2, 0.0],
            "sde_step_indices": [0, 2],
            "transition_strategy": "variance-preserving",
            "eta": 0.6,
            "sigma_max": 0.9,
            "clip_range": 0.2,
            "updates_per_trajectory": 2,
            "group_size": 4,
            "old_log_prob_source": "replay",
            "advantage_epsilon": 0.0000001,
            "advantage_clip_max": 3.0,
            "trajectory_dtype": "float32",
            "reward_weights": {
                "video_quality": 1.0,
                "motion_quality": 0.25,
                "text_alignment": 0.5,
            },
            "reward_model": {"type": "videoalign"},
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.02,
            "max_grad_norm": 0.5,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def test_grpo_guard_recipe_is_strict_and_round_trips() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    assert isinstance(recipe.algorithm, GRPOGuardAlgorithmSpec)
    assert recipe.algorithm.clip_range == 0.2
    assert recipe.algorithm.advantage_clip_max == 3.0
    assert recipe.algorithm.requires_reference_policy is False
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    unknown = _recipe_mapping()
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    invalid_clip = _recipe_mapping()
    invalid_clip["algorithm"]["clip_range"] = 1.0
    with pytest.raises(ValueError, match="clip_range must be finite and in"):
        PostTrainingRecipe.from_mapping(invalid_clip)


def test_flow_policy_builder_dispatches_grpo_guard_runtime() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    stack = post_training.build_native_flow_policy_training_stack(
        recipe,
        policy=_FlowPredictor(),
        initial_policy_revision="policy-root",
        fused_adamw=False,
        replay_microbatch_size=1,
    )

    assert isinstance(stack, post_training.NativeFlowPolicyTrainingStack)
    assert isinstance(stack.engine, NativeGRPOGuardEngine)
    assert stack.session_type is NativeGRPOGuardTrainingSession
    assert stack.engine.clip_range == 0.2
    assert stack.engine.advantage_clip_max == 3.0
    assert stack.engine.updates_per_trajectory == 2
    assert stack.reference_replay is None
    assert stack.sde_step_indices == (0, 2)
    assert stack.optimizer.param_groups[0]["lr"] == 0.0003


def test_checked_in_grpo_guard_config_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan_1p3b_grpo_guard.yaml")

    assert isinstance(recipe.algorithm, GRPOGuardAlgorithmSpec)
    assert recipe.algorithm.updates_per_trajectory == 2
    assert recipe.algorithm.reference_checkpoint is None
    assert recipes.GRPOGuardAlgorithmSpec is GRPOGuardAlgorithmSpec
    for name in grpo_guard.__all__:
        canonical = getattr(grpo_guard, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
