from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
from worldfoundry.training.checkpoint.state import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch  # noqa: E402
from worldfoundry.training.recipes import (  # noqa: E402
    DiffusionNFTAlgorithmSpec,
    DiffusionNFTOldPolicyRefreshSpec,
    DiffusionNFTTerminalLatentCollectionSpec,
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
        expanded = torch.as_tensor(
            sigmas,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        ).reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - expanded * velocity


class _TerminalRewardAdapter:
    reward_ids = ("video_quality", "motion_quality", "text_alignment")

    def score(self, terminal_latents):
        values = terminal_latents.clean_latents.float().flatten(1).mean(dim=1)
        return {
            "video_quality": values + 3.6757,
            "motion_quality": values + 1.1646,
            "text_alignment": values + 2.8105,
        }


def _recipe_mapping() -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "diffusion-nft-test", "output_dir": "runs/diffusion-nft-test"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "model"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "diffusion-nft",
            "collection": {
                "sigmas": [1.0, 0.5, 0.0],
                "group_size": 4,
                "guidance_scale": 1.0,
                "latent_dtype": "float32",
                "forward_batch_size": 2,
            },
            "beta": 0.1,
            "advantage_clip_max": 2.0,
            "advantage_epsilon": 0.0001,
            "advantage_mode": "binary",
            "advantage_normalization": "group-mean-global-population-std",
            "reference_mse_weight": 0.05,
            "reference_checkpoint": "reference",
            "reconstruction_mae_floor": 0.00001,
            "old_policy_refresh": {
                "decay": "linear-to-0-5",
                "interval": 1,
            },
            "reward_weights": {
                "video_quality": 1.0,
                "motion_quality": 0.25,
                "text_alignment": 0.5,
            },
            "reward_model": {"type": "videoalign"},
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.001,
            "max_grad_norm": 0.5,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def _rollout_batch() -> FlowRolloutBatch:
    return FlowRolloutBatch(
        sample_ids=("s0", "s1", "s2", "s3"),
        group_ids=("prompt", "prompt", "prompt", "prompt"),
        policy_revision="old-policy-0",
        initial_latents=torch.tensor(
            [[-1.0, 0.5], [-0.5, 1.0], [0.5, -1.0], [1.0, -0.5]],
            dtype=torch.float32,
        ),
        sigmas=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
    )


def test_diffusion_nft_recipe_is_strict_nested_and_round_trips() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    assert isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec)
    assert isinstance(
        recipe.algorithm.collection,
        DiffusionNFTTerminalLatentCollectionSpec,
    )
    assert isinstance(
        recipe.algorithm.old_policy_refresh,
        DiffusionNFTOldPolicyRefreshSpec,
    )
    assert recipe.algorithm.collection.sigmas == (1.0, 0.5, 0.0)
    assert recipe.algorithm.old_policy_refresh.decay == "linear_to_0_5"
    assert recipe.algorithm.advantage_mode == "binary"
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    payload = _recipe_mapping()
    payload["algorithm"]["collection"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm.collection contains unknown fields"):
        PostTrainingRecipe.from_mapping(payload)


def test_diffusion_nft_reference_and_terminal_schedule_fail_closed() -> None:
    missing_reference = _recipe_mapping()
    missing_reference["algorithm"].pop("reference_checkpoint")
    with pytest.raises(ValueError, match="requires an explicit reference_checkpoint"):
        PostTrainingRecipe.from_mapping(missing_reference)

    incomplete_schedule = _recipe_mapping()
    incomplete_schedule["algorithm"]["collection"]["sigmas"] = [1.0, 0.5]
    with pytest.raises(ValueError, match="start at 1 and end at 0"):
        PostTrainingRecipe.from_mapping(incomplete_schedule)


def test_diffusion_nft_builder_materializes_collection_reward_engine_and_session() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    policy = _FlowPredictor()
    old_policy = _FlowPredictor()
    reference_policy = _FlowPredictor(trainable=False)
    reward_adapter = _TerminalRewardAdapter()

    stack = post_training.build_native_diffusion_nft_training_stack(
        recipe,
        policy=policy,
        old_policy=old_policy,
        initial_old_policy_revision="old-policy-0",
        reference_policy=reference_policy,
        reward_adapter=reward_adapter,
        fused_adamw=False,
    )

    assert isinstance(stack, post_training.NativeDiffusionNFTTrainingStack)
    assert stack.collector.policy is old_policy
    assert stack.collector.sigmas == (1.0, 0.5, 0.0)
    assert stack.collector.forward_batch_size == 2
    assert stack.group_size == 4
    assert stack.engine.beta == 0.1
    assert stack.engine.advantage_mode == "binary"
    assert stack.engine.old_policy_refresh.schedule == "linear_to_0_5"
    assert stack.engine.reference_mse_weight == 0.05
    assert dict(stack.scalarizer.weights) == {
        "video_quality": 1.0,
        "motion_quality": 0.25,
        "text_alignment": 0.5,
    }
    checkpoint_state = stack.checkpoint_state_kwargs()["algorithm_state"]
    assert checkpoint_state is stack.algorithm_state
    assert stack.algorithm_state.component_names == (
        "old_policy",
        "reward_scalarizer",
    )

    session = stack.build_session(
        [_rollout_batch()],
        TrainingProgress(optimizer_steps=0),
    )
    summary = session.run(max_steps=1, generator=torch.Generator().manual_seed(7))

    assert isinstance(session, post_training.NativeDiffusionNFTTrainingSession)
    assert summary.final_step == 1
    assert summary.iterations == 1
    assert stack.engine.old_policy_refreshes == 1
    with pytest.raises(ValueError, match="stale behavior policy"):
        session.train_iteration(
            _rollout_batch(),
            generator=torch.Generator().manual_seed(9),
        )

    saved_algorithm_state = copy.deepcopy(stack.algorithm_state.state_dict())
    expected_old_policy = old_policy.module.weight.detach().clone()
    with torch.no_grad():
        old_policy.module.weight.add_(10)
    stack.algorithm_state.load_state_dict(saved_algorithm_state)
    torch.testing.assert_close(
        old_policy.module.weight,
        expected_old_policy,
        rtol=0,
        atol=0,
    )


def test_diffusion_nft_builder_rejects_reward_contract_and_reference_mismatch() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    policy = _FlowPredictor()
    old_policy = _FlowPredictor()
    reference = _FlowPredictor(trainable=False)

    class _WrongRewards(_TerminalRewardAdapter):
        reward_ids = ("video_quality",)

    with pytest.raises(ValueError, match="reward adapter ids differ"):
        post_training.build_native_diffusion_nft_training_stack(
            recipe,
            policy=policy,
            old_policy=old_policy,
            initial_old_policy_revision="old-policy-initial",
            reference_policy=reference,
            reward_adapter=_WrongRewards(),
            fused_adamw=False,
        )


def test_checked_in_diffusion_nft_recipe_parses_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan_1p3b_diffusion_nft.yaml")

    assert isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec)
    assert recipe.algorithm.collection.group_size == 4
    assert recipe.algorithm.reference_mse_weight == 0.0001
    assert DiffusionNFTTerminalLatentCollectionSpec.__module__.endswith(".algorithms.diffusion_nft")
    assert DiffusionNFTOldPolicyRefreshSpec.__module__.endswith(".algorithms.diffusion_nft")
