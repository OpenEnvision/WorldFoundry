from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import worldfoundry.training.post_training as post_training  # noqa: E402
import worldfoundry.training.recipes as recipes  # noqa: E402
from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms import ddrl  # noqa: E402
from worldfoundry.training.recipes import (  # noqa: E402
    DDRLAlgorithmSpec,
    PostTrainingRecipe,
)


class _ReplayAdapter:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(0.25)

    def replay_mean(self, trajectory, train_on_position, *, training):
        self.module.train(training)
        noisy = trajectory.replay_inputs["noisy"][:, train_on_position]
        return self.module(noisy)


class _DataRegularizer:
    def __init__(self, replay: _ReplayAdapter) -> None:
        self.module = replay.module

    def loss(self, trajectory, train_on_position, *, generator, training):
        del generator
        self.module.train(training)
        noisy = trajectory.replay_inputs["data_noisy"][:, train_on_position]
        target = trajectory.replay_inputs["data_target"][:, train_on_position]
        return (self.module(noisy) - target).square()


class _RolloutAdapter:
    def __init__(self, *, train_on: tuple[int, ...] = (0, 2)) -> None:
        self.train_on = train_on

    def collect(self, batch, *, generator=None):
        del generator
        step_count = len(self.train_on)
        noisy = torch.linspace(-0.5, 0.8, batch.batch_size * step_count).reshape(
            batch.batch_size,
            step_count,
            1,
        )
        old_means = noisy * 0.2
        return ddrl.DDRLTrajectory(
            trajectory_id=f"trajectory-{batch.batch_id}",
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            train_on=self.train_on,
            next_latents=noisy * 0.4 + 0.1,
            old_means=old_means,
            reference_means=old_means * 0.5,
            terminal_latents=torch.tensor([[0.0], [2.0], [1.0], [5.0]]),
            replay_inputs={
                "noisy": noisy,
                "data_noisy": noisy + 0.1,
                "data_target": torch.zeros_like(noisy),
            },
        )


class _RewardAdapter:
    reward_ids = ("video_quality", "motion_quality", "text_alignment")

    def score(self, trajectory):
        values = trajectory.terminal_latents.float().flatten(1).mean(dim=1)
        return {
            "video_quality": values + 3.6757,
            "motion_quality": values + 1.1646,
            "text_alignment": values + 2.8105,
        }


def _recipe_mapping() -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "ddrl-test", "output_dir": "runs/ddrl-test"},
        "model": {"recipe": "cosmos-predict2.5-2b", "checkpoint": "policy"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "ddrl",
            "train_on": [0, 2],
            "clip_range": 0.2,
            "advantage_epsilon": 0.0001,
            "advantage_clip_min": -2.0,
            "advantage_clip_max": 3.0,
            "exponential_advantage": False,
            "kl_beta": 0.1,
            "data_beta": 0.25,
            "data_on_first_step_only": True,
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
            "max_grad_norm": 0.75,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def _batch() -> ddrl.DDRLRolloutBatch:
    return ddrl.DDRLRolloutBatch(
        batch_id="batch",
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
    )


def test_ddrl_recipe_is_strict_executable_and_round_trips() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    assert isinstance(recipe.algorithm, DDRLAlgorithmSpec)
    assert recipe.algorithm.train_on == (0, 2)
    assert recipe.algorithm.advantage_epsilon == 1.0e-4
    assert recipe.algorithm.loss_scale == 10.0
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    unknown = _recipe_mapping()
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    duplicate_step = _recipe_mapping()
    duplicate_step["algorithm"]["train_on"] = [0, 2, 2]
    with pytest.raises(ValueError, match="strictly increasing, and unique"):
        PostTrainingRecipe.from_mapping(duplicate_step)

    floating_step = _recipe_mapping()
    floating_step["algorithm"]["train_on"] = [0, 1.5]
    with pytest.raises(TypeError, match="integers, not bool or float"):
        PostTrainingRecipe.from_mapping(floating_step)

    fake_score = _recipe_mapping()
    fake_score["fake_score_optimizer"] = {
        "type": "adamw",
        "learning_rate": 0.001,
    }
    with pytest.raises(ValueError, match="DDRL cannot configure fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(fake_score)


def test_ddrl_recipe_rejects_unused_regularizer_cadence() -> None:
    unused_data_cadence = _recipe_mapping()
    unused_data_cadence["algorithm"]["data_beta"] = 0.0
    with pytest.raises(ValueError, match="data_on_first_step_only is unused"):
        PostTrainingRecipe.from_mapping(unused_data_cadence)


def test_builder_materializes_reward_optimizer_engine_and_recipe_bound_session() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    rollout = _RolloutAdapter()
    replay = _ReplayAdapter()
    data_regularizer = _DataRegularizer(replay)
    reward = _RewardAdapter()

    stack = post_training.build_native_ddrl_training_stack(
        recipe,
        rollout_adapter=rollout,
        replay_adapter=replay,
        reward_adapter=reward,
        data_regularizer=data_regularizer,
        fused_adamw=False,
    )

    assert isinstance(stack, post_training.NativeDDRLTrainingStack)
    assert stack.rollout_adapter is rollout
    assert stack.engine.clip_range == 0.2
    assert stack.engine.loss_scale == 10.0
    assert stack.engine.advantage_clip_min == -2.0
    assert stack.engine.advantage_clip_max == 3.0
    assert stack.engine.kl_beta == 0.1
    assert stack.engine.data_beta == 0.25
    assert stack.engine.data_on_first_step_only is True
    assert stack.train_on == (0, 2)
    assert stack.checkpoint_state_kwargs()["algorithm_state"] is stack.scalarizer

    progress = TrainingProgress()
    session = stack.build_session(progress)
    result = session.train_iteration(
        _batch(),
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(session, post_training.NativeDDRLTrainingSession)
    assert result.update.train_on == (0, 2)
    assert result.update.reference_kl is not None
    assert result.update.data_loss is not None
    assert progress.optimizer_steps == 1
    assert stack.engine.global_step == 1


def test_builder_fails_closed_for_data_adapter_and_collected_train_on() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    replay = _ReplayAdapter()

    with pytest.raises(TypeError, match="requires a data_regularizer adapter"):
        post_training.build_native_ddrl_training_stack(
            recipe,
            rollout_adapter=_RolloutAdapter(),
            replay_adapter=replay,
            reward_adapter=_RewardAdapter(),
            fused_adamw=False,
        )

    stack = post_training.build_native_ddrl_training_stack(
        recipe,
        rollout_adapter=_RolloutAdapter(train_on=(0, 1)),
        replay_adapter=replay,
        reward_adapter=_RewardAdapter(),
        data_regularizer=_DataRegularizer(replay),
        fused_adamw=False,
    )
    with pytest.raises(ValueError, match="train_on differs from the configured recipe"):
        stack.build_session(TrainingProgress()).train_iteration(_batch())


def test_ddrl_stack_fixture_and_public_exports_are_canonical() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(root / "tests/training/fixtures/recipes/cosmos_predict2p5_2b_ddrl_stack.yaml")

    assert isinstance(recipe.algorithm, DDRLAlgorithmSpec)
    assert recipe.algorithm.train_on == tuple(range(0, 20, 2))
    assert recipe.algorithm.clip_range == 1.0e-4
    assert recipe.algorithm.loss_scale == 10.0
    assert recipe.algorithm.data_beta == 0.01
    assert recipe.optimizer.betas == (0.9, 0.99)
    assert recipes.DDRLAlgorithmSpec is DDRLAlgorithmSpec
    assert DDRLAlgorithmSpec.__module__.endswith(".algorithms.ddrl")
    assert post_training.NativeDDRLTrainingStack is ddrl.NativeDDRLTrainingStack
    assert post_training.build_native_ddrl_training_stack is ddrl.build_native_ddrl_training_stack
