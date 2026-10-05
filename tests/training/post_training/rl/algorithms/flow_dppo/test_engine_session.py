from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    PendingTrainingCheckpoint,
    TrainingProgress,
)
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.flow_dppo import (  # noqa: E402
    FLOW_DPPO_ENGINE_STATE_SCHEMA,
    FlowDPPOIterationResult,
    FlowDPPOStageAlgorithm,
    NativeFlowDPPOEngine,
    NativeFlowDPPOTrainingSession,
)
from worldfoundry.training.post_training.rl.algorithms.stage import (  # noqa: E402
    AnchorField,
)
from worldfoundry.training.post_training.rl.contracts import (  # noqa: E402
    FlowReplayResult,
    FlowRolloutBatch,
)
from worldfoundry.training.post_training.rl.trajectory import (  # noqa: E402
    FlowTrajectorySampler,
    NativeFlowTrajectoryReplay,
)


class _ToyPolicy:
    def __init__(self, gain: float, *, trainable: bool = True) -> None:
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
        sigma = sigmas.reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - sigma * velocity


class _GeometryAwareReplay:
    """Make the replay batch size observable in the returned transition means."""

    def __init__(self, policy: _ToyPolicy) -> None:
        self.module = policy.module
        self.delegate = NativeFlowTrajectoryReplay(policy)
        self.calls: list[tuple[bool, int]] = []
        self.training_sample_ids: list[tuple[str, ...]] = []

    def replay(self, trajectory, *, training: bool) -> FlowReplayResult:
        self.calls.append((training, trajectory.batch_size))
        if training:
            self.training_sample_ids.append(trajectory.sample_ids)
        result = self.delegate.replay(trajectory, training=training)
        return FlowReplayResult(
            log_probs=result.log_probs,
            transition_means=result.transition_means + float(trajectory.batch_size),
            transition_scales=result.transition_scales,
        )


class _TrainingLogProbDriftReplay:
    def __init__(self, policy: _ToyPolicy) -> None:
        self.module = policy.module
        self.delegate = NativeFlowTrajectoryReplay(policy)

    def replay(self, trajectory, *, training: bool) -> FlowReplayResult:
        result = self.delegate.replay(trajectory, training=training)
        return FlowReplayResult(
            log_probs=result.log_probs + (0.25 if training else 0.0),
            transition_means=result.transition_means,
            transition_scales=result.transition_scales,
            velocities=result.velocities,
            std_dev_t=result.std_dev_t,
            sqrt_dt=result.sqrt_dt,
        )


class _FailAfterCommitSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("failure after optimizer mutation")


def _trajectory(policy: _ToyPolicy, revision: str):
    return FlowTrajectorySampler(policy, eta=0.7).sample(
        torch.randn(4, 2, generator=torch.Generator().manual_seed(41)),
        torch.tensor([1.0, 0.6, 0.0]),
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        conditioning={},
        policy_revision=revision,
        generator=torch.Generator().manual_seed(43),
    )


def test_flow_dppo_stage_declares_loss_only_anchor_contract() -> None:
    stage = FlowDPPOStageAlgorithm(kl_mask_threshold=0.25, add_kl_coefficient=False)

    assert stage.anchor_fields == frozenset({AnchorField.OLD_LOG_PROBS, AnchorField.OLD_TRANSITION_MEANS})
    assert stage.supports_multi_update is True
    assert stage.state_fields == {
        "kl_mask_threshold": 0.25,
        "add_kl_coefficient": False,
    }
    assert not hasattr(stage, "optimizer")
    assert not hasattr(stage, "replay_adapter")


def test_flow_dppo_replay_anchor_uses_microbatch_geometry_and_stays_frozen() -> None:
    policy = _ToyPolicy(0.2)
    replay = _GeometryAwareReplay(policy)
    engine = NativeFlowDPPOEngine(
        replay,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        kl_mask_threshold=0.0,
        updates_per_trajectory=2,
        replay_microbatch_size=1,
    )
    trajectory = _trajectory(policy, "policy-root")
    anchor = engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
        old_log_prob_source="replay",
    )

    assert replay.calls == [(False, 1)] * 4
    with pytest.raises(RuntimeError, match="trajectory boundary"):
        engine.state_dict()

    first = engine.train_step(anchor_id=anchor)
    calls_after_first = tuple(replay.calls)
    second = engine.train_step(anchor_id=anchor)

    torch.testing.assert_close(first.metrics["ratio_mean"], torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(first.metrics["old_policy_kl"], torch.tensor(0.0), rtol=0, atol=0)
    assert first.trajectory_complete is False
    assert second.trajectory_complete is True
    assert float(second.metrics["old_policy_kl"]) > 0
    assert float(second.metrics["masked_fraction"]) > 0
    assert calls_after_first.count((False, 1)) == 4
    assert replay.calls.count((False, 1)) == 4
    assert replay.calls.count((True, 1)) == 4
    assert replay.training_sample_ids == [("a",), ("b",), ("c",), ("d",)]
    assert (first.sample_count, first.token_count, first.replay_microbatches) == (2, 4, 2)
    assert (second.sample_count, second.token_count, second.replay_microbatches) == (2, 4, 2)
    assert not engine.has_active_trajectory

    state = engine.state_dict()
    assert state == {
        "schema": FLOW_DPPO_ENGINE_STATE_SCHEMA,
        "global_step": 2,
        "initial_policy_revision": "policy-root",
        "current_policy_revision": engine.current_policy_revision,
        "updates_per_trajectory": 2,
        "reference_kl_weight": 0.0,
        "replay_microbatch_size": 1,
        "data_parallel_size": 1,
        "kl_mask_threshold": 0.0,
        "add_kl_coefficient": True,
    }


def test_flow_updates_are_balanced_disjoint_partitions_not_full_batch_epochs() -> None:
    policy = _ToyPolicy(0.2)
    replay = _GeometryAwareReplay(policy)
    engine = NativeFlowDPPOEngine(
        replay,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        kl_mask_threshold=0.0,
        updates_per_trajectory=3,
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(policy, "policy-root"),
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
        old_log_prob_source="replay",
    )

    results = tuple(engine.train_step(anchor_id=anchor_id) for _ in range(3))

    assert replay.calls[:3] == [(False, 2), (False, 1), (False, 1)]
    assert replay.calls[3:] == [(True, 2), (True, 1), (True, 1)]
    assert replay.training_sample_ids == [("a", "b"), ("c",), ("d",)]
    assert [result.sample_count for result in results] == [2, 1, 1]
    assert [result.token_count for result in results] == [4, 2, 2]
    assert [result.replay_microbatches for result in results] == [1, 1, 1]
    assert results[-1].trajectory_complete


def test_flow_partition_plan_rejects_more_updates_than_samples() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowDPPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        updates_per_trajectory=5,
    )

    with pytest.raises(ValueError, match="cannot exceed trajectory batch_size"):
        engine.prepare_trajectory(
            _trajectory(policy, "policy-root"),
            torch.tensor([1.0, 3.0, 2.0, 5.0]),
        )


class _RecordingParallelContext:
    rank = 0
    world_size = 2
    process_group = None

    def audit_synchronized_module(self, module, *, role: str) -> None:
        del module, role

    def audit_local_group_ownership(self, group_ids) -> None:
        del group_ids

    def scale_local_mean(self, local_mean, local_weight):
        del local_weight
        return local_mean


def test_flow_distributed_partition_rejects_mismatched_backward_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_module = importlib.import_module("worldfoundry.training.post_training.rl.algorithms.flow_policy.engine")

    def mismatched_counts(gathered, local, *, group) -> None:
        del group
        for value in gathered:
            value.copy_(local)
        gathered[-1].add_(1)

    monkeypatch.setattr(engine_module.dist, "all_gather", mismatched_counts)
    policy = _ToyPolicy(0.2)
    replay = _GeometryAwareReplay(policy)
    engine = NativeFlowDPPOEngine(
        replay,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
        replay_microbatch_size=1,
        parallel_context=_RecordingParallelContext(),
    )
    anchor_id = engine.prepare_trajectory(
        _trajectory(policy, "policy-root"),
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
    )

    with pytest.raises(ValueError, match="same backward-call count"):
        engine.train_step(anchor_id=anchor_id)

    assert replay.calls == [(False, 1)] * 4
    assert engine.global_step == 0


def test_flow_dppo_rollout_log_probs_use_preupdate_replay_means() -> None:
    policy = _ToyPolicy(0.2)
    replay = _GeometryAwareReplay(policy)
    engine = NativeFlowDPPOEngine(
        replay,
        torch.optim.SGD(policy.module.parameters(), lr=0.01),
        initial_policy_revision="policy-root",
        kl_mask_threshold=0.0,
        replay_microbatch_size=1,
    )
    trajectory = _trajectory(policy, "policy-root")

    anchor = engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
        old_log_prob_source="rollout",
    )
    assert replay.calls == [(False, 1)] * 4

    result = engine.train_step(anchor_id=anchor)

    assert replay.calls == [(False, 1)] * 4 + [(True, 1)] * 4
    assert result.metrics["old_policy_kl"] == pytest.approx(0.0)
    assert engine.has_active_trajectory is False
    assert engine.is_poisoned is False


def test_flow_policy_first_update_rejects_a_non_unit_replay_ratio() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowDPPOEngine(
        _TrainingLogProbDriftReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.01),
        initial_policy_revision="policy-root",
    )
    trajectory = _trajectory(policy, "policy-root")
    anchor = engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
        old_log_prob_source="replay",
    )
    before = policy.module.weight.detach().clone()

    with pytest.raises(ValueError, match="old log-probability anchor"):
        engine.train_step(anchor_id=anchor)

    torch.testing.assert_close(policy.module.weight, before, rtol=0, atol=0)
    assert engine.global_step == 0
    assert engine.is_poisoned is False


def test_flow_policy_optimizer_failure_poisons_the_engine() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowDPPOEngine(
        NativeFlowTrajectoryReplay(policy),
        _FailAfterCommitSGD(policy.module.parameters(), lr=0.01),
        initial_policy_revision="policy-root",
    )
    trajectory = _trajectory(policy, "policy-root")
    anchor = engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0, 3.0, 2.0, 5.0]),
        old_log_prob_source="replay",
    )

    with pytest.raises(RuntimeError, match="failure after optimizer mutation"):
        engine.train_step(anchor_id=anchor)

    assert engine.is_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.finish_trajectory(anchor_id=anchor)


def test_flow_dppo_optional_frozen_reference_kl_is_independent_of_old_policy_kl() -> None:
    policy = _ToyPolicy(0.2)
    reference = _ToyPolicy(0.1, trainable=False)
    engine = NativeFlowDPPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        reference_replay_adapter=NativeFlowTrajectoryReplay(reference),
        reference_kl_weight=0.5,
        add_kl_coefficient=False,
    )
    trajectory = _trajectory(policy, "policy-root")

    result = engine.train_step(
        anchor_id=engine.prepare_trajectory(
            trajectory,
            torch.tensor([1.0, 3.0, 2.0, 5.0]),
            old_log_prob_source="replay",
        )
    )

    torch.testing.assert_close(result.metrics["old_policy_kl"], torch.tensor(0.0), rtol=0, atol=0)
    assert result.reference_kl is not None
    assert float(result.reference_kl) > 0
    torch.testing.assert_close(result.metrics["reference_kl"], result.reference_kl, rtol=0, atol=0)
    assert engine.state_dict()["add_kl_coefficient"] is False


class _TerminalReward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float()
        return {
            "alignment": terminal.flatten(1).mean(dim=1),
            "quality": terminal.flatten(1).square().mean(dim=1),
        }


class _BoundaryCheckpointer:
    def __init__(self, engine: NativeFlowDPPOEngine) -> None:
        self.engine = engine
        self.saved_steps: list[int] = []

    def save(self, state, *, asynchronous: bool):
        del state, asynchronous
        self.engine.state_dict()
        self.saved_steps.append(self.engine.global_step)
        return object()


class _DeferredBoundaryCheckpointer(_BoundaryCheckpointer):
    def __init__(self, engine: NativeFlowDPPOEngine) -> None:
        super().__init__(engine)
        self.wait_calls = 0

    def save(self, state, *, asynchronous: bool):
        del state
        assert asynchronous is True
        self.engine.state_dict()
        self.saved_steps.append(self.engine.global_step)
        pending = object.__new__(PendingTrainingCheckpoint)

        def wait(timeout=None):
            del timeout
            self.wait_calls += 1
            return object()

        pending.wait = wait
        return pending


def test_flow_dppo_session_runs_rollout_reward_replay_multi_update_and_safe_checkpoint() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowDPPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        kl_mask_threshold=0.0,
        updates_per_trajectory=2,
        replay_microbatch_size=1,
    )
    checkpointer = _BoundaryCheckpointer(engine)
    events: list[object] = []
    session = NativeFlowDPPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.7),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 0.25}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 2),
        old_log_prob_source="replay",
        checkpoint_state=object(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=1,
        event_sink=events.append,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(4, 2, generator=torch.Generator().manual_seed(17)),
        sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
    )

    result = session.train_iteration(batch, generator=torch.Generator().manual_seed(19))

    assert isinstance(result, FlowDPPOIterationResult)
    assert len(result.updates) == 2
    torch.testing.assert_close(result.updates[0].metrics["ratio_mean"], torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(result.updates[0].metrics["old_policy_kl"], torch.tensor(0.0), rtol=0, atol=0)
    assert result.updates[-1].trajectory_complete is True
    assert session.progress.optimizer_steps == 2
    assert checkpointer.saved_steps == [2]
    assert [event["schema"] for event in events] == [
        "worldfoundry-flow-dppo-step-event",
        "worldfoundry-flow-dppo-step-event",
    ]
    assert "old_policy_kl" in events[0]
    assert "masked_fraction" in events[1]


def test_flow_policy_async_checkpoint_is_joined_at_the_run_boundary() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowDPPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        kl_mask_threshold=0.0,
        updates_per_trajectory=1,
    )
    checkpointer = _DeferredBoundaryCheckpointer(engine)
    session = NativeFlowDPPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.7),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 0.25}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 2),
        old_log_prob_source="replay",
        checkpoint_state=object(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=1,
        asynchronous_checkpoints=True,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(4, 2, generator=torch.Generator().manual_seed(23)),
        sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
    )

    session.train_iteration(batch, generator=torch.Generator().manual_seed(29))

    assert checkpointer.saved_steps == [1]
    assert checkpointer.wait_calls == 0
    session.wait_for_checkpoints()
    assert checkpointer.wait_calls == 1
    session.wait_for_checkpoints()
    assert checkpointer.wait_calls == 1
