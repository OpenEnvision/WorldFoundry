from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
import worldfoundry.training.recipes as recipes  # noqa: E402
from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms import (  # noqa: E402
    diffusion_dpo,
)
from worldfoundry.training.recipes import (  # noqa: E402
    DiffusionDPOAlgorithmSpec,
    PostTrainingRecipe,
)


class _FlowPredictor:
    def __init__(self, gain: float, *, trainable: bool) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)
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
        return noisy_latents * self.module.weight.reshape(())

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
        expanded = sigmas.reshape((int(noisy_latents.shape[0]),) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - expanded * velocity


def _recipe_mapping() -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "diffusion-dpo-test", "output_dir": "runs/diffusion-dpo-test"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "policy"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/preference-pairs.jsonl", "shuffle": False},
        "algorithm": {
            "type": "diffusion-dpo",
            "beta": 0.5,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.001,
            "max_grad_norm": 0.75,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def _batch() -> diffusion_dpo.DiffusionDPOBatch:
    return diffusion_dpo.DiffusionDPOBatch(
        batch_id="pair-batch",
        sample_ids=("pair-a-chosen", "pair-a-rejected"),
        pair_ids=("pair-a", "pair-a"),
        clean_latents=torch.tensor([[0.25], [0.8]], dtype=torch.float32),
        conditioning={"context": torch.ones(2, 1)},
    )


def test_diffusion_dpo_recipe_is_strict_and_round_trips() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    assert isinstance(recipe.algorithm, DiffusionDPOAlgorithmSpec)
    assert recipe.algorithm.beta == 0.5
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    unknown = _recipe_mapping()
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    non_positive = _recipe_mapping()
    non_positive["algorithm"]["beta"] = 0
    with pytest.raises(ValueError, match="beta must be finite and positive"):
        PostTrainingRecipe.from_mapping(non_positive)

    misleading_reference = _recipe_mapping()
    misleading_reference["algorithm"]["reference_checkpoint"] = "frozen-reference"
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(misleading_reference)


def test_builder_materializes_roles_optimizer_session_and_checkpoint_seam() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    policy = _FlowPredictor(0.25, trainable=True)
    reference = _FlowPredictor(-0.1, trainable=False)
    stack = post_training.build_native_diffusion_dpo_training_stack(
        recipe,
        policy=policy,
        reference_policy=reference,
        fused_adamw=False,
    )

    assert isinstance(stack, post_training.NativeDiffusionDPOTrainingStack)
    assert stack.engine.policy is policy
    assert stack.engine.reference_policy is reference
    assert stack.engine.beta == 0.5
    assert stack.engine.max_grad_norm == 0.75
    assert stack.checkpoint_state_kwargs() == {
        "lr_scheduler": None,
        "ema": None,
        "algorithm_state": None,
    }
    optimizer_parameters = {id(parameter) for group in stack.optimizer.param_groups for parameter in group["params"]}
    assert optimizer_parameters == {id(policy.module.weight)}
    assert id(reference.module.weight) not in optimizer_parameters

    progress = TrainingProgress()
    session = stack.build_session([_batch()], progress)
    summary = session.run(max_steps=1, generator=torch.Generator().manual_seed(17))
    assert isinstance(session, post_training.NativeDiffusionDPOTrainingSession)
    assert summary.final_step == 1
    assert progress.optimizer_steps == 1

    saved_engine_state = copy.deepcopy(stack.engine.state_dict())
    restored = post_training.build_native_diffusion_dpo_training_stack(
        recipe,
        policy=_FlowPredictor(0.25, trainable=True),
        reference_policy=_FlowPredictor(-0.1, trainable=False),
        fused_adamw=False,
    )
    restored.engine.load_state_dict(saved_engine_state)
    assert restored.engine.state_dict() == saved_engine_state


def test_builder_rejects_mutable_or_aliased_reference_roles() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    policy = _FlowPredictor(0.25, trainable=True)

    with pytest.raises(ValueError, match="parameters must be frozen"):
        post_training.build_native_diffusion_dpo_training_stack(
            recipe,
            policy=policy,
            reference_policy=_FlowPredictor(-0.1, trainable=True),
            fused_adamw=False,
        )

    aliased = _FlowPredictor(-0.1, trainable=False)
    aliased.module = policy.module
    with pytest.raises(ValueError, match="distinct modules"):
        post_training.build_native_diffusion_dpo_training_stack(
            recipe,
            policy=policy,
            reference_policy=aliased,
            fused_adamw=False,
        )


def test_stack_fixture_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(root / "tests/training/fixtures/recipes/wan_1p3b_diffusion_dpo_stack.yaml")

    assert isinstance(recipe.algorithm, DiffusionDPOAlgorithmSpec)
    assert recipe.algorithm.beta == 5000.0
    assert recipes.DiffusionDPOAlgorithmSpec is DiffusionDPOAlgorithmSpec
    assert DiffusionDPOAlgorithmSpec.__module__.endswith(".algorithms.diffusion_dpo")
    assert post_training.NativeDiffusionDPOTrainingStack is diffusion_dpo.NativeDiffusionDPOTrainingStack
    assert (
        post_training.build_native_diffusion_dpo_training_stack
        is diffusion_dpo.build_native_diffusion_dpo_training_stack
    )
