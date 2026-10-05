from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training import (  # noqa: E402
    DMDLossResult,
    DMDTrainingBatch,
    FlowRolloutBatch,
    FlowSDEIndexSchedule,
    FlowTrajectorySampler,
    NativeDMDTrainEngine,
    NativeDMDTrainingSession,
    NativeFlowGRPOEngine,
    NativeFlowGRPOTrainingSession,
    NativeFlowTrajectoryReplay,
    WeightedRewardScalarizer,
)


class _ToyDMDLosses:
    def __init__(self, student: torch.nn.Linear, fake_score: torch.nn.Linear) -> None:
        self.student = student
        self.fake_score = fake_score

    def loss_denominator(self, batch, *, role):
        del role
        return torch.tensor(float(batch.batch_size), device=batch.clean_latents.device)

    def generator_loss(self, batch, *, generator=None) -> DMDLossResult:
        del generator
        loss = (self.student.weight - 0.25).square().mean()
        return DMDLossResult(loss, {"loss_denominator": self.loss_denominator(batch, role="generator")})

    def fake_score_loss(self, batch, *, generator=None) -> DMDLossResult:
        del generator
        target = self.student.weight.detach() + 0.1
        loss = (self.fake_score.weight - target).square().mean()
        return DMDLossResult(loss, {"loss_denominator": self.loss_denominator(batch, role="fake-score")})


def _dmd_stack() -> tuple[NativeDMDTrainEngine, DMDTrainingBatch]:
    student = torch.nn.Linear(1, 1, bias=False)
    fake_score = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    engine = NativeDMDTrainEngine(
        student_module=student,
        real_score_module=teacher,
        fake_score_module=fake_score,
        loss_adapter=_ToyDMDLosses(student, fake_score),
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        fake_score_optimizer=torch.optim.SGD(fake_score.parameters(), lr=0.1),
        generator_update_interval=2,
    )
    batch = DMDTrainingBatch(
        sample_ids=("sample-a", "sample-b"),
        clean_latents=torch.zeros(2, 1, 2, 3),
        conditioning={},
        unconditional_conditioning={},
    )
    return engine, batch


def test_native_dmd_session_drives_cadence_progress_and_events() -> None:
    engine, batch = _dmd_stack()
    progress = TrainingProgress()
    events: list[object] = []
    session = NativeDMDTrainingSession(
        engine,
        [batch],
        progress,
        event_sink=events.append,
    )

    summary = session.run(max_steps=3, generator=torch.Generator().manual_seed(11))

    assert summary.initial_step == 0
    assert summary.final_step == 3
    assert summary.student_optimizer_steps == 1
    assert summary.fake_score_optimizer_steps == 3
    assert progress.optimizer_steps == 3
    assert progress.microbatches_seen == 3
    assert progress.samples_seen == 6
    assert progress.latent_tokens_seen == 36
    assert [event["generator_updated"] for event in events] == [False, True, False]


def test_native_dmd_session_runs_export_callbacks_only_at_safe_boundaries() -> None:
    engine, batch = _dmd_stack()
    boundaries: list[tuple[int, int]] = []
    session = NativeDMDTrainingSession(engine, [batch], TrainingProgress())

    session.run(
        max_steps=5,
        boundary_every_steps=2,
        boundary_sink=lambda previous, current: boundaries.append((previous, current)),
    )

    assert boundaries == [(1, 2), (3, 4)]

    engine, batch = _dmd_stack()
    session = NativeDMDTrainingSession(engine, [batch], TrainingProgress())
    with pytest.raises(ValueError, match="configured together"):
        session.run(max_steps=1, boundary_every_steps=1)


def test_native_dmd_session_keeps_one_iterator_across_run_calls() -> None:
    engine, batch = _dmd_stack()

    class _FiniteLoader:
        def __init__(self) -> None:
            self.iterator_count = 0
            self.yielded: list[int] = []

        def __iter__(self):
            self.iterator_count += 1
            for index in (1, 2):
                self.yielded.append(index)
                yield batch

    loader = _FiniteLoader()
    session = NativeDMDTrainingSession(engine, loader, TrainingProgress())

    first = session.run(max_steps=1)
    second = session.run(max_steps=1)

    assert first.initial_step == 0
    assert second.initial_step == 1
    assert second.final_step == 2
    assert loader.iterator_count == 1
    assert loader.yielded == [1, 2]


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


class _TerminalReward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float()
        return {
            "alignment": terminal.flatten(1).mean(dim=1),
            "quality": terminal.flatten(1).square().mean(dim=1),
        }


def test_native_flow_grpo_session_closes_rollout_reward_replay_update_loop() -> None:
    policy = _ToyPolicy(0.2)
    sampler = FlowTrajectorySampler(policy, eta=0.7, trajectory_dtype=torch.bfloat16)
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
    )
    progress = TrainingProgress()
    events: list[object] = []
    session = NativeFlowGRPOTrainingSession(
        sampler=sampler,
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 0.25}),
        engine=engine,
        progress=progress,
        sde_step_indices=(0, 2),
        event_sink=events.append,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(4, 2, generator=torch.Generator().manual_seed(17)),
        sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
    )
    before = policy.module.weight.detach().clone()

    result = session.train_iteration(batch, generator=torch.Generator().manual_seed(19))

    assert len(result.updates) == 2
    assert result.updates[-1].trajectory_complete is True
    torch.testing.assert_close(result.updates[0].metrics["ratio_mean"], torch.tensor(1.0), rtol=0, atol=0)
    assert result.rewards.scalar_rewards.shape == (4,)
    assert not engine.has_active_trajectory
    assert engine.global_step == 2
    assert progress.optimizer_steps == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 4
    assert progress.latent_tokens_seen == 8
    assert [update.sample_count for update in result.updates] == [2, 2]
    assert [update.token_count for update in result.updates] == [4, 4]
    assert len(events) == 2
    assert not torch.equal(policy.module.weight.detach(), before)


def test_flow_session_rejects_a_stale_rollout_before_sampling() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="active-policy",
    )
    session = NativeFlowGRPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.7),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0,),
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b"),
        group_ids=("group", "group"),
        policy_revision="stale-policy",
        initial_latents=torch.zeros(2, 1),
        sigmas=torch.tensor([1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="policy revision"):
        session.train_iteration(batch)


class _RecordingCheckpointer:
    def __init__(self, engine: NativeFlowGRPOEngine) -> None:
        self.engine = engine
        self.saved_steps: list[int] = []

    def save(self, state, *, asynchronous: bool):
        del state, asynchronous
        self.saved_steps.append(self.engine.global_step)
        return object()


def _flow_batch(policy_revision: str, *, seed: int) -> FlowRolloutBatch:
    return FlowRolloutBatch(
        sample_ids=(f"a-{seed}", f"b-{seed}"),
        group_ids=(f"group-{seed}", f"group-{seed}"),
        policy_revision=policy_revision,
        initial_latents=torch.randn(
            2,
            2,
            generator=torch.Generator().manual_seed(seed),
        ),
        sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
    )


def test_flow_session_checkpoints_after_crossing_cadence_at_safe_boundary() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
    )
    checkpointer = _RecordingCheckpointer(engine)
    session = NativeFlowGRPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.7),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 2),
        checkpoint_state=object(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=3,
    )

    session.train_iteration(
        _flow_batch(engine.current_policy_revision, seed=29),
        generator=torch.Generator().manual_seed(31),
    )
    assert checkpointer.saved_steps == []

    session.train_iteration(
        _flow_batch(engine.current_policy_revision, seed=37),
        generator=torch.Generator().manual_seed(41),
    )

    assert checkpointer.saved_steps == [4]
    assert not engine.has_active_trajectory


def test_flow_session_resolves_dynamic_sde_indices_from_rollout_identity() -> None:
    policy = _ToyPolicy(0.2)
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.01),
        initial_policy_revision="policy-root",
    )
    events: list[object] = []
    session = NativeFlowGRPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.25, sigma_max=0.8),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"alignment": 1.0, "quality": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        sde_index_schedule=FlowSDEIndexSchedule(
            transition_count=16,
            timestep_fraction=(0.0, 0.6),
            num_sde_steps=8,
        ),
        event_sink=events.append,
    )
    sigmas = torch.tensor(
        [
            1.0,
            0.97826087474823,
            0.9545454382896423,
            0.9285714030265808,
            0.8999999761581421,
            0.8684210777282715,
            0.8333333134651184,
            0.7941176295280457,
            0.75,
            0.699999988079071,
            0.6428571343421936,
            0.5769230723381042,
            0.5,
            0.40909090638160706,
            0.30000001192092896,
            0.1666666716337204,
            0.0,
        ]
    )

    first = session.train_iteration(
        FlowRolloutBatch(
            sample_ids=("a-0", "b-0"),
            group_ids=("group-0", "group-0"),
            policy_revision=engine.current_policy_revision,
            initial_latents=torch.randn(2, 2),
            sigmas=sigmas,
        ),
        generator=torch.Generator().manual_seed(107),
    )
    second = session.train_iteration(
        FlowRolloutBatch(
            sample_ids=("a-1", "b-1"),
            group_ids=("group-1", "group-1"),
            policy_revision=engine.current_policy_revision,
            initial_latents=torch.randn(2, 2),
            sigmas=sigmas,
        ),
        generator=torch.Generator().manual_seed(109),
    )

    assert first.trajectory.step_indices == (0, 1, 2, 3, 4, 5, 7, 8)
    assert second.trajectory.step_indices == (0, 1, 3, 4, 5, 6, 7, 8)
    assert events[0]["rollout_id"] == 0
    assert events[1]["rollout_id"] == 1
    assert events[1]["sde_step_indices"] == [0, 1, 3, 4, 5, 6, 7, 8]
