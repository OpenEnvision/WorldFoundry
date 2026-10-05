from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training import (  # noqa: E402
    FlowRolloutBatch,
    FlowTrajectorySampler,
    NativeFlowTrajectoryReplay,
    NativeMixGRPOEngine,
    NativeMixGRPOTrainingSession,
    PostTrainingParallelContext,
    WeightedRewardScalarizer,
    build_native_flow_policy_training_stack,
    normalize_weighted_component_advantages,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (  # noqa: E402
    VariancePreservingFlowTransition,
)
from worldfoundry.training.post_training.rl.rollout_strategies.window_sde_steps import (  # noqa: E402
    FlowSDEWindowSchedule,
)
from worldfoundry.training.recipes import MixGRPOAlgorithmSpec, PostTrainingRecipe  # noqa: E402


class _Policy:
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
        sigma = sigmas.reshape((noisy_latents.shape[0], 1))
        return noisy_latents - sigma * velocity


class _FixedReward:
    def score(self, trajectory):
        device = trajectory.latents.device
        return {
            "video_quality": torch.tensor([1.0, 3.0, 2.0, 6.0], device=device),
            "motion_quality": torch.tensor([4.0, 2.0, 8.0, 2.0], device=device),
            "text_alignment": torch.tensor([0.0, 2.0, 3.0, 1.0], device=device),
        }


class _RecordingMixEngine(NativeMixGRPOEngine):
    recorded_advantages: torch.Tensor | None = None

    def prepare_trajectory_from_advantages(self, trajectory, advantages, **kwargs):
        self.recorded_advantages = advantages.detach().clone()
        return super().prepare_trajectory_from_advantages(
            trajectory,
            advantages,
            **kwargs,
        )


def test_mixgrpo_component_first_advantages_match_author_formula() -> None:
    rewards = {
        "quality": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "alignment": torch.tensor([4.0, 2.0, 8.0, 2.0]),
    }
    result = normalize_weighted_component_advantages(
        rewards,
        {"quality": 1.0, "alignment": 3.0},
        ("first", "first", "second", "second"),
        parallel_context=PostTrainingParallelContext.current(),
        epsilon=1.0e-8,
        normalization="group-sample-std",
    )

    root_half = torch.tensor(0.5**0.5)
    expected = torch.tensor([0.5, -0.5, 0.5, -0.5]) * root_half
    torch.testing.assert_close(result.advantages, expected, rtol=1.0e-6, atol=1.0e-7)
    assert dict(result.normalized_weights) == {"quality": 0.25, "alignment": 0.75}


def test_mixgrpo_session_trains_from_component_advantages_not_scalarized_rewards() -> None:
    policy = _Policy()
    sampler = FlowTrajectorySampler(
        policy,
        transition_strategy=VariancePreservingFlowTransition(eta=0.7, sigma_max=0.8),
        trajectory_dtype=torch.float32,
    )
    engine = _RecordingMixEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.02),
        initial_policy_revision="policy-root",
    )
    scalarizer = WeightedRewardScalarizer(
        {
            "video_quality": 1.0,
            "motion_quality": 3.0,
            "text_alignment": 2.0,
        },
        calibration_mean={
            "video_quality": 100.0,
            "motion_quality": -50.0,
            "text_alignment": 20.0,
        },
        calibration_std={
            "video_quality": 2.0,
            "motion_quality": 4.0,
            "text_alignment": 8.0,
        },
    )
    session = NativeMixGRPOTrainingSession(
        sampler=sampler,
        reward_adapter=_FixedReward(),
        scalarizer=scalarizer,
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 1),
        advantage_normalization="group-sample-std",
        advantage_clip_max=5.0,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(4, 2, generator=torch.Generator().manual_seed(41)),
        sigmas=torch.tensor([1.0, 0.6, 0.0]),
    )

    result = session.train_iteration(batch, generator=torch.Generator().manual_seed(43))
    expected = normalize_weighted_component_advantages(
        _FixedReward().score(result.trajectory),
        scalarizer.weights,
        result.trajectory.group_ids,
        parallel_context=engine.parallel_context,
        epsilon=1.0e-8,
        clip_max=5.0,
        normalization="group-sample-std",
    ).advantages

    assert engine.recorded_advantages is not None
    torch.testing.assert_close(engine.recorded_advantages, expected, rtol=0, atol=0)
    scalar_advantages = (
        result.rewards.scalar_rewards[:2] - result.rewards.scalar_rewards[:2].mean()
    ) / (result.rewards.scalar_rewards[:2].std() + 1.0e-8)
    assert not torch.allclose(engine.recorded_advantages[:2], scalar_advantages)
    assert engine.state_dict()["advantage_aggregation"] == "component-first"


def test_mixgrpo_recipe_dispatches_to_windowed_native_runtime() -> None:
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(
        root / "configs/post_training/wan_1p3b_mix_grpo.yaml"
    )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=_Policy(),
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )

    assert isinstance(recipe.algorithm, MixGRPOAlgorithmSpec)
    assert isinstance(stack.engine, NativeMixGRPOEngine)
    assert stack.session_type is NativeMixGRPOTrainingSession
    assert isinstance(stack.transition_strategy, VariancePreservingFlowTransition)
    assert isinstance(stack.sde_index_schedule, FlowSDEWindowSchedule)
    assert stack.sde_index_schedule.resolve(0) == (0, 1, 2, 3)
    assert stack.sde_index_schedule.resolve(25) == (1, 2, 3, 4)
    assert stack.init_same_noise is True
    assert dict(stack.session_kwargs) == {}
    assert stack.engine.state_dict()["schema"] == "worldfoundry-mix-grpo-engine"
