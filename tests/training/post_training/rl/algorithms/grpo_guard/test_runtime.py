from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.grpo_guard import (  # noqa: E402
    GRPO_GUARD_ENGINE_STATE_SCHEMA,
    GRPOGuardIterationResult,
    GRPOGuardStageAlgorithm,
    NativeGRPOGuardEngine,
    NativeGRPOGuardTrainingSession,
)
from worldfoundry.training.post_training.rl.algorithms.stage import (  # noqa: E402
    AnchorField,
    StageAnchor,
)
from worldfoundry.training.post_training.rl.contracts import (  # noqa: E402
    FlowReplayResult,
    FlowRolloutBatch,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (  # noqa: E402
    ConstantDiffusionFlowTransition,
)
from worldfoundry.training.post_training.rl.trajectory import (  # noqa: E402
    FlowTrajectorySampler,
    NativeFlowTrajectoryReplay,
)


class _ToyPolicy:
    def __init__(self, gain: float) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)

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
        sigma = sigmas.reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - sigma * velocity


def _trajectory(policy: _ToyPolicy, revision: str):
    sampler = FlowTrajectorySampler(
        policy,
        transition_strategy=ConstantDiffusionFlowTransition(eta=0.65),
    )
    trajectory = sampler.sample(
        torch.randn(4, 3, generator=torch.Generator().manual_seed(31)),
        torch.tensor([1.0, 0.7, 0.2, 0.0]),
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        conditioning={},
        policy_revision=revision,
        sde_step_indices=(0, 2),
        generator=torch.Generator().manual_seed(37),
    )
    return sampler, trajectory


def test_native_replay_reconstructs_grpo_guard_diffusion_geometry() -> None:
    policy = _ToyPolicy(0.2)
    _, trajectory = _trajectory(policy, "policy-root")

    replay = NativeFlowTrajectoryReplay(policy).replay(trajectory, training=False)

    assert replay.sqrt_dt is not None
    assert replay.std_dev_t is not None
    expected_sqrt_dt = torch.tensor(
        [[(1.0 - 0.7) ** 0.5, (0.2 - 0.0) ** 0.5]] * 4,
        dtype=replay.sqrt_dt.dtype,
    )
    torch.testing.assert_close(replay.sqrt_dt, expected_sqrt_dt)
    torch.testing.assert_close(
        replay.std_dev_t,
        torch.full_like(replay.std_dev_t, 0.65),
    )
    torch.testing.assert_close(
        replay.transition_scales,
        replay.std_dev_t * replay.sqrt_dt.unsqueeze(-1),
    )


def test_grpo_guard_stage_requires_reconstructed_diffusion_geometry() -> None:
    stage = GRPOGuardStageAlgorithm(clip_range=0.2)
    replay = FlowReplayResult(
        log_probs=torch.zeros(2, 2),
        transition_means=torch.zeros(2, 2, 1),
        transition_scales=torch.ones(2, 2, 1),
    )
    anchor = StageAnchor(
        old_log_probs=torch.zeros(2, 2),
        old_transition_means=torch.zeros(2, 2, 1),
        advantages=torch.ones(2),
    )

    with pytest.raises(ValueError, match="std_dev_t and sqrt_dt"):
        stage.loss(replay, anchor)


class _TerminalReward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float()
        return {
            "alignment": terminal.flatten(1).mean(dim=1),
            "quality": terminal.flatten(1).square().mean(dim=1),
        }


def test_grpo_guard_session_runs_native_rollout_replay_and_multi_update() -> None:
    policy = _ToyPolicy(0.2)
    sampler, _ = _trajectory(policy, "policy-root")
    engine = NativeGRPOGuardEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.03),
        initial_policy_revision="policy-root",
        clip_range=0.2,
        updates_per_trajectory=2,
        replay_microbatch_size=1,
    )
    events: list[object] = []
    session = NativeGRPOGuardTrainingSession(
        sampler=sampler,
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 0.25}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 2),
        old_log_prob_source="replay",
        event_sink=events.append,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(
            4,
            3,
            generator=torch.Generator().manual_seed(41),
        ),
        sigmas=torch.tensor([1.0, 0.7, 0.2, 0.0]),
    )

    result = session.train_iteration(
        batch,
        generator=torch.Generator().manual_seed(43),
    )

    assert isinstance(result, GRPOGuardIterationResult)
    assert len(result.updates) == 2
    assert result.updates[0].trajectory_complete is False
    assert result.updates[1].trajectory_complete is True
    torch.testing.assert_close(
        result.updates[0].metrics["ratio_mean_bias"],
        torch.tensor(0.0),
        rtol=0,
        atol=0,
    )
    assert float(result.updates[1].metrics["ratio_mean_bias"]) > 0
    assert session.progress.optimizer_steps == 2
    assert [event["schema"] for event in events] == [
        "worldfoundry-grpo-guard-step-event",
        "worldfoundry-grpo-guard-step-event",
    ]
    assert all("scale" in event and "sqrt_dt_mean" in event for event in events)
    assert engine.state_dict() == {
        "schema": GRPO_GUARD_ENGINE_STATE_SCHEMA,
        "global_step": 2,
        "initial_policy_revision": "policy-root",
        "current_policy_revision": engine.current_policy_revision,
        "updates_per_trajectory": 2,
        "reference_kl_weight": 0.0,
        "replay_microbatch_size": 1,
        "data_parallel_size": 1,
        "clip_range": 0.2,
        "advantage_clip_max": 5.0,
    }
    assert GRPOGuardStageAlgorithm().anchor_fields == frozenset(
        {AnchorField.OLD_LOG_PROBS, AnchorField.OLD_TRANSITION_MEANS}
    )
