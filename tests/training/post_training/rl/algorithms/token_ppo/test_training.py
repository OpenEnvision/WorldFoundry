from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.training.post_training.rl.algorithms.token_ppo import (
    NativeTokenPPOEngine,
    PackedTokenPPOTrajectory,
    TokenPPOReplayResult,
    TokenPPORolloutRequest,
    TokenPPOSample,
    build_native_token_ppo_training_stack,
    materialize_token_ppo_training_run,
)
from worldfoundry.training.recipes.post_training import (
    PostTrainingRecipe,
    TokenPPOAlgorithmSpec,
)
from worldfoundry.training.tuning.full_model import FullModelArtifact


class _ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_weight = nn.Parameter(torch.tensor(0.2))
        self.value_weight = nn.Parameter(torch.tensor(0.1))

    def evaluate(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = tokens.to(self.policy_weight) / 10.0
        return (
            F.logsigmoid(features * self.policy_weight),
            features * self.value_weight,
        )


class _Replay:
    def __init__(self, module: _ActorCritic) -> None:
        self.module = module
        self.training_chunks: list[tuple[tuple[str, ...], torch.Tensor]] = []

    def replay(self, trajectory, *, training: bool) -> TokenPPOReplayResult:
        self.module.train(training)
        if training:
            self.training_chunks.append(
                (trajectory.sample_ids, trajectory.tokens.detach().clone()),
            )
        log_probs, values = self.module.evaluate(trajectory.tokens)
        return TokenPPOReplayResult(
            log_probs=log_probs,
            values=values,
            sampling_temperature=trajectory.sampling_temperature,
        )


class _Rollout:
    def __init__(self, replay: _Replay) -> None:
        self.replay = replay

    def rollout(self, request: TokenPPORolloutRequest, *, generator=None):
        del generator
        lengths = torch.tensor([2 + index % 2 for index in range(request.batch_size)])
        tokens = torch.arange(1, int(lengths.sum().item()) + 1)
        with torch.no_grad():
            old_log_probs, _ = self.replay.module.evaluate(tokens)
        return PackedTokenPPOTrajectory(
            sample_ids=request.sample_ids,
            policy_revision=request.policy_revision,
            tokens=tokens,
            lengths=lengths,
            old_log_probs=old_log_probs,
            loss_mask=torch.ones(tokens.shape[0], dtype=torch.bool),
            sampling_temperature=request.sampling_temperature,
            conditioning=request.conditioning,
        )


class _Rewards:
    reward_ids = ("outcome",)

    def score(self, trajectory: PackedTokenPPOTrajectory):
        return {"outcome": torch.tensor([1.0 if index % 2 == 0 else -0.5 for index in range(trajectory.batch_size)])}


def _recipe_mapping(output_dir: Path | str, *, save_every_steps: int = 0) -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "token-ppo-test", "output_dir": str(output_dir)},
        "model": {"recipe": "toy-actor-critic", "checkpoint": "initial"},
        "tuning": {"mode": "full"},
        "data": {
            "manifest": "unused.jsonl",
            "shuffle": False,
            "tail_policy": "drop",
            "options": {"batch_size": 2},
        },
        "algorithm": {
            "type": "token-ppo",
            "reward_weights": {"outcome": 1.0},
            "update_epochs": 2,
            "clip_range": 0.2,
            "clip_range_high": 0.3,
            "clip_schedule": "linear_decay",
            "clip_schedule_steps": 8,
            "value_clip_range": 0.2,
            "vf_coef": 0.5,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "reduction": "seq-mean-token-mean",
            "horizon": 32,
            "sampling_temperature": 1.0,
            "replay_microbatch_size": 1,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "checkpoint": {"save_every_steps": save_every_steps, "async": False},
        "export": {"format": "safetensors"},
    }


def _adapters(model: _ActorCritic):
    replay = _Replay(model)
    return _Rollout(replay), replay, _Rewards()


def _samples() -> tuple[TokenPPOSample, ...]:
    return tuple(
        TokenPPOSample(sample_id=f"prompt-{index}", conditioning={"prompt": f"p{index}"}) for index in range(4)
    )


def test_token_ppo_recipe_is_strict_round_trips_and_builds(tmp_path: Path) -> None:
    mapping = _recipe_mapping(tmp_path / "run")
    recipe = PostTrainingRecipe.from_mapping(mapping)
    assert isinstance(recipe.algorithm, TokenPPOAlgorithmSpec)
    assert recipe.algorithm.clip_range_high == 0.3
    assert recipe.algorithm.clip_schedule == "linear-decay"
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    mapping["algorithm"]["unused"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(mapping)

    model = _ActorCritic()
    rollout, replay, rewards = _adapters(model)
    stack = build_native_token_ppo_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=rewards,
        initial_policy_revision="initial",
        fused_adamw=False,
    )
    assert isinstance(stack.engine, NativeTokenPPOEngine)
    assert stack.engine.update_epochs == 2
    assert stack.engine.update_partitions == 1


def test_update_partitions_are_disjoint_exhaustive_and_share_one_anchor(tmp_path: Path) -> None:
    mapping = _recipe_mapping(tmp_path / "partitions")
    mapping["algorithm"]["update_epochs"] = 1  # type: ignore[index]
    mapping["algorithm"]["update_partitions"] = 4  # type: ignore[index]
    mapping["algorithm"]["replay_microbatch_size"] = None  # type: ignore[index]
    recipe = PostTrainingRecipe.from_mapping(mapping)
    model = _ActorCritic()
    rollout, replay, rewards = _adapters(model)
    stack = build_native_token_ppo_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=rewards,
        initial_policy_revision="initial",
        fused_adamw=False,
    )
    sample_ids = tuple(f"sample-{index}" for index in range(8))
    trajectory = rollout.rollout(
        TokenPPORolloutRequest(sample_ids=sample_ids, policy_revision="initial"),
    )
    anchor = stack.engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0 if index % 2 == 0 else -0.5 for index in range(8)]),
    )
    frozen = (
        anchor.old_log_probs.clone(),
        anchor.old_values.clone(),
        anchor.advantages.clone(),
        anchor.returns.clone(),
    )

    updates = [stack.engine.train_step() for _ in range(4)]

    assert [update.sample_count for update in updates] == [2, 2, 2, 2]
    assert [int(update.metrics["update_partition"].item()) for update in updates] == [1, 2, 3, 4]
    assert [update.trajectory_complete for update in updates] == [False, False, False, True]
    assert len(replay.training_chunks) == 4
    chunk_ids = [ids for ids, _ in replay.training_chunks]
    assert tuple(sample for ids in chunk_ids for sample in ids) == sample_ids
    assert all(set(left).isdisjoint(right) for index, left in enumerate(chunk_ids) for right in chunk_ids[index + 1 :])
    assert torch.equal(
        torch.cat([tokens for _, tokens in replay.training_chunks]),
        trajectory.tokens,
    )
    assert torch.equal(anchor.old_log_probs, frozen[0])
    assert torch.equal(anchor.old_values, frozen[1])
    assert torch.equal(anchor.advantages, frozen[2])
    assert torch.equal(anchor.returns, frozen[3])
    assert stack.engine.global_step == 4
    assert not stack.engine.has_active_trajectory


def test_multi_update_keeps_old_logp_and_value_anchors_frozen(tmp_path: Path) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "run"))
    model = _ActorCritic()
    rollout, replay, rewards = _adapters(model)
    stack = build_native_token_ppo_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=rewards,
        initial_policy_revision="initial",
        fused_adamw=False,
    )
    request = TokenPPORolloutRequest(sample_ids=("a", "b"), policy_revision="initial")
    trajectory = rollout.rollout(request)
    anchor = stack.engine.prepare_trajectory(trajectory, torch.tensor([1.0, -0.5]))
    old_log_probs = anchor.old_log_probs.clone()
    old_values = anchor.old_values.clone()
    policy_before = model.policy_weight.detach().clone()
    value_before = model.value_weight.detach().clone()

    first = stack.engine.train_step()
    assert not first.trajectory_complete
    assert stack.engine.active_anchor is anchor
    assert torch.equal(anchor.old_log_probs, old_log_probs)
    assert torch.equal(anchor.old_values, old_values)
    second = stack.engine.train_step()
    assert second.trajectory_complete
    assert stack.engine.global_step == 2
    assert not torch.equal(model.policy_weight.detach(), policy_before)
    assert not torch.equal(model.value_weight.detach(), value_before)
    assert first.metrics["clip_range"].item() == pytest.approx(0.2)
    assert second.metrics["clip_range"].item() == pytest.approx(0.1875)


def test_toy_run_update_resume_and_export_match_uninterrupted(tmp_path: Path) -> None:
    full_recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "full", save_every_steps=2))
    full_model = _ActorCritic()
    full_rollout, full_replay, full_rewards = _adapters(full_model)
    full_run = materialize_token_ppo_training_run(
        full_recipe,
        rollout_adapter=full_rollout,
        replay_adapter=full_replay,
        reward_adapter=full_rewards,
        samples=_samples(),
        initialization_seed=17,
        fused_adamw=False,
    )
    full_summary = full_run.run(max_iterations=2)
    assert full_summary.final_optimizer_step == 4
    full_state = {name: value.detach().clone() for name, value in full_model.state_dict().items()}

    split_recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "split", save_every_steps=2))
    first_model = _ActorCritic()
    first_rollout, first_replay, first_rewards = _adapters(first_model)
    first_run = materialize_token_ppo_training_run(
        split_recipe,
        rollout_adapter=first_rollout,
        replay_adapter=first_replay,
        reward_adapter=first_rewards,
        samples=_samples(),
        initialization_seed=17,
        fused_adamw=False,
    )
    assert first_run.run(max_iterations=1).final_optimizer_step == 2

    resumed_model = _ActorCritic()
    resumed_rollout, resumed_replay, resumed_rewards = _adapters(resumed_model)
    resumed_run = materialize_token_ppo_training_run(
        split_recipe,
        rollout_adapter=resumed_rollout,
        replay_adapter=resumed_replay,
        reward_adapter=resumed_rewards,
        samples=_samples(),
        resume_checkpoint="latest",
        initialization_seed=17,
        fused_adamw=False,
    )
    assert resumed_run.resume_artifact is not None
    resumed_summary = resumed_run.run(max_iterations=1)
    assert resumed_summary.initial_optimizer_step == 2
    assert resumed_summary.final_optimizer_step == 4
    for name, expected in full_state.items():
        assert torch.equal(resumed_model.state_dict()[name], expected)

    artifact = resumed_run.export_actor_critic()
    assert isinstance(artifact, FullModelArtifact)
    assert artifact.path.is_dir()
