from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training import (  # noqa: E402
    FlowRolloutBatch,
    FlowTrajectorySampler,
    NativeDanceGRPOEngine,
    NativeDanceGRPOTrainingSession,
    NativeFlowGRPOEngine,
    NativeFlowTrajectoryReplay,
    WeightedRewardScalarizer,
    build_native_flow_policy_training_stack,
    sample_dance_update_step_mask,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (  # noqa: E402
    ConstantDiffusionFlowTransition,
)
from worldfoundry.training.recipes import (  # noqa: E402
    DanceGRPOAlgorithmSpec,
    PostTrainingRecipe,
)


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


class _Reward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float().mean(dim=1)
        return {
            "video_quality": terminal,
            "motion_quality": terminal.square(),
            "text_alignment": -terminal,
        }


def _rollout_batch() -> FlowRolloutBatch:
    return FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(4, 2, generator=torch.Generator().manual_seed(11)),
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]),
    )


def test_dance_update_mask_is_independent_per_sample_and_resume_deterministic() -> None:
    generator = torch.Generator().manual_seed(714)
    saved_state = generator.get_state()
    first = sample_dance_update_step_mask(
        batch_size=4,
        transition_count=5,
        timestep_fraction=0.6,
        device=torch.device("cpu"),
        generator=generator,
    )
    generator.set_state(saved_state)
    restored = sample_dance_update_step_mask(
        batch_size=4,
        transition_count=5,
        timestep_fraction=0.6,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert first.dtype is torch.bool
    assert first.sum(dim=1).tolist() == [3, 3, 3, 3]
    assert len({tuple(row.tolist()) for row in first}) > 1
    torch.testing.assert_close(restored, first, rtol=0, atol=0)


def test_dance_session_uses_all_sde_steps_but_updates_only_random_subsets() -> None:
    policy = _Policy()
    sampler = FlowTrajectorySampler(
        policy,
        transition_strategy=ConstantDiffusionFlowTransition(eta=0.3),
        trajectory_dtype=torch.float32,
    )
    engine = NativeDanceGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.02),
        initial_policy_revision="policy-root",
        update_timestep_fraction=0.5,
        updates_per_trajectory=2,
    )
    progress = TrainingProgress()
    session = NativeDanceGRPOTrainingSession(
        sampler=sampler,
        reward_adapter=_Reward(),
        scalarizer=WeightedRewardScalarizer(
            {
                "video_quality": 1.0,
                "motion_quality": 1.0,
                "text_alignment": 1.0,
            }
        ),
        engine=engine,
        progress=progress,
        sde_step_indices=(0, 1, 2, 3),
        advantage_normalization="group-sample-std",
        advantage_clip_max=5.0,
        update_timestep_fraction=0.5,
    )

    result = session.train_iteration(
        _rollout_batch(),
        generator=torch.Generator().manual_seed(29),
    )

    assert result.trajectory.step_indices == (0, 1, 2, 3)
    assert result.trajectory.update_step_mask is not None
    assert result.trajectory.update_step_mask.sum(dim=1).tolist() == [2, 2, 2, 2]
    assert [update.token_count for update in result.updates] == [4, 4]
    assert progress.latent_tokens_seen == 8
    assert engine.global_step == 2
    assert engine.state_dict()["update_timestep_fraction"] == 0.5


def test_non_dance_stage_rejects_a_behavioral_update_mask() -> None:
    policy = _Policy()
    trajectory = FlowTrajectorySampler(policy, eta=0.7).sample(
        _rollout_batch().initial_latents,
        _rollout_batch().sigmas,
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        conditioning={},
        policy_revision="policy-root",
        generator=torch.Generator().manual_seed(31),
    )
    masked = replace(
        trajectory,
        update_step_mask=torch.ones_like(trajectory.old_log_probs, dtype=torch.bool),
    )
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.01),
        initial_policy_revision="policy-root",
    )

    with pytest.raises(ValueError, match="does not support an update_step_mask"):
        engine.prepare_trajectory(masked, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_dance_recipe_dispatches_to_native_engine_session_and_constant_diffusion() -> None:
    root = Path(__file__).resolve().parents[6]
    recipe = PostTrainingRecipe.from_file(
        root / "configs/post_training/wan_1p3b_dance_grpo.yaml"
    )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=_Policy(),
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )

    assert isinstance(recipe.algorithm, DanceGRPOAlgorithmSpec)
    assert isinstance(stack.engine, NativeDanceGRPOEngine)
    assert stack.session_type is NativeDanceGRPOTrainingSession
    assert isinstance(stack.transition_strategy, ConstantDiffusionFlowTransition)
    assert stack.init_same_noise is True
    assert dict(stack.session_kwargs) == {"update_timestep_fraction": 0.6}
    assert stack.engine.state_dict()["schema"] == "worldfoundry-dance-grpo-engine"
