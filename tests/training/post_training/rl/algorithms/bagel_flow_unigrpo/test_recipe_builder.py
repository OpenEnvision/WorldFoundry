from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
import worldfoundry.training.post_training.rl as rl  # noqa: E402
import worldfoundry.training.post_training.rl.algorithms as algorithms  # noqa: E402
import worldfoundry.training.recipes as recipes  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms import (  # noqa: E402
    bagel_flow_unigrpo,
)
from worldfoundry.training.post_training.rl.algorithms.bagel_flow_unigrpo import (  # noqa: E402
    BagelFlowUniGRPOStageAlgorithm,
    NativeBagelFlowUniGRPOEngine,
    NativeBagelFlowUniGRPOTrainingSession,
)
from worldfoundry.training.recipes import (  # noqa: E402
    BagelFlowUniGRPOAlgorithmSpec,
    PostTrainingRecipe,
)


class _FlowPredictor:
    def __init__(self, *, trainable: bool = True) -> None:
        self.module = torch.nn.Linear(2, 2, bias=False)
        self.module.requires_grad_(trainable)

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
        "run": {
            "id": "bagel-flow-unigrpo-test",
            "output_dir": "runs/bagel-flow-unigrpo-test",
        },
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "policy"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "bagel-flow-unigrpo",
            "sigmas": [1.0, 0.7, 0.2, 0.0],
            "sde_step_indices": [0, 2],
            "transition_strategy": "variance-preserving",
            "eta": 0.6,
            "sigma_max": 0.9,
            "clip_range": 0.2,
            "velocity_mse_weight": 0.5,
            "ratio_norm": True,
            "grad_reweight": True,
            "updates_per_trajectory": 2,
            "group_size": 4,
            "old_log_prob_source": "replay",
            "reference_checkpoint": "frozen-reference",
            "advantage_epsilon": 0.0000001,
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


def test_bagel_flow_unigrpo_recipe_is_strict_and_round_trips() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    assert isinstance(recipe.algorithm, BagelFlowUniGRPOAlgorithmSpec)
    assert recipe.algorithm.velocity_mse_weight == 0.5
    assert recipe.algorithm.ratio_norm is True
    assert recipe.algorithm.grad_reweight is True
    assert recipe.algorithm.requires_reference_policy is True
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    unknown = _recipe_mapping()
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    missing_reference = _recipe_mapping()
    missing_reference["algorithm"].pop("reference_checkpoint")
    with pytest.raises(ValueError, match="requires an explicit reference_checkpoint"):
        PostTrainingRecipe.from_mapping(missing_reference)

    incompatible_reference_kl = _recipe_mapping()
    incompatible_reference_kl["algorithm"]["reference_kl_weight"] = 0.1
    with pytest.raises(ValueError, match="uses velocity MSE instead of reference KL"):
        PostTrainingRecipe.from_mapping(incompatible_reference_kl)

    invalid_reweight = _recipe_mapping()
    invalid_reweight["algorithm"]["ratio_norm"] = False
    with pytest.raises(ValueError, match="grad_reweight is only defined"):
        PostTrainingRecipe.from_mapping(invalid_reweight)


def test_flow_policy_builder_dispatches_bagel_runtime_and_reference_role() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    policy = _FlowPredictor()

    with pytest.raises(ValueError, match="requires a frozen reference_policy"):
        post_training.build_native_flow_policy_training_stack(
            recipe,
            policy=policy,
            initial_policy_revision="policy-root",
            fused_adamw=False,
        )

    with pytest.raises(ValueError, match="reference policy parameters must be frozen"):
        post_training.build_native_flow_policy_training_stack(
            recipe,
            policy=policy,
            reference_policy=_FlowPredictor(),
            initial_policy_revision="policy-root",
            fused_adamw=False,
        )

    reference = _FlowPredictor(trainable=False)
    stack = post_training.build_native_flow_policy_training_stack(
        recipe,
        policy=policy,
        reference_policy=reference,
        initial_policy_revision="policy-root",
        fused_adamw=False,
        replay_microbatch_size=1,
    )

    assert isinstance(stack, post_training.NativeFlowPolicyTrainingStack)
    assert isinstance(stack.engine, NativeBagelFlowUniGRPOEngine)
    assert stack.session_type is NativeBagelFlowUniGRPOTrainingSession
    assert isinstance(stack.engine.algorithm, BagelFlowUniGRPOStageAlgorithm)
    assert stack.engine.algorithm.clip_range == 0.2
    assert stack.engine.algorithm.velocity_mse_weight == 0.5
    assert stack.engine.algorithm.ratio_norm is True
    assert stack.engine.algorithm.grad_reweight is True
    assert stack.engine.reference_kl_weight == 0.0
    assert stack.reference_replay is not None
    assert all(parameter.grad is None for parameter in reference.module.parameters())
    assert stack.optimizer.param_groups[0]["lr"] == 0.0003


def test_checked_in_bagel_config_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan_1p3b_bagel_flow_unigrpo.yaml")

    assert isinstance(recipe.algorithm, BagelFlowUniGRPOAlgorithmSpec)
    assert recipe.algorithm.reference_checkpoint == "default"
    assert recipe.algorithm.reference_kl_weight == 0.0
    assert recipes.BagelFlowUniGRPOAlgorithmSpec is BagelFlowUniGRPOAlgorithmSpec
    for name in bagel_flow_unigrpo.__all__:
        canonical = getattr(bagel_flow_unigrpo, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
