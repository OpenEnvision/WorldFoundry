from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training import (  # noqa: E402
    DMDLossResult,
    DMDTrainingBatch,
    FlowTrajectorySampler,
    NativeDMDTrainEngine,
    NativeFlowGRPOEngine,
    NativeFlowTrajectoryReplay,
)


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


class _FailAfterCommitSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("failure after optimizer mutation")


class _ToyDMDLosses:
    def __init__(self, student: torch.nn.Linear, fake_score: torch.nn.Linear, *, fail_fake: bool = False) -> None:
        self.student = student
        self.fake_score = fake_score
        self.fail_fake = fail_fake

    def loss_denominator(self, batch, *, role):
        del role
        return torch.tensor(float(batch.batch_size), device=batch.clean_latents.device)

    def generator_loss(self, batch, *, generator=None) -> DMDLossResult:
        del generator
        loss = (self.student.weight - 0.25).square().mean()
        return DMDLossResult(loss, {"loss_denominator": self.loss_denominator(batch, role="generator")})

    def fake_score_loss(self, batch, *, generator=None) -> DMDLossResult:
        del generator
        if self.fail_fake:
            raise RuntimeError("intentional fake-score failure")
        target = self.student.weight.detach() + 0.1
        loss = (self.fake_score.weight - target).square().mean()
        return DMDLossResult(loss, {"loss_denominator": self.loss_denominator(batch, role="fake-score")})


def _dmd_engine(*, fail_fake: bool = False, generator_interval: int = 2):
    student = torch.nn.Linear(1, 1, bias=False)
    fake_score = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    student_scheduler = _Scheduler()
    fake_scheduler = _Scheduler()
    engine = NativeDMDTrainEngine(
        student_module=student,
        real_score_module=teacher,
        fake_score_module=fake_score,
        loss_adapter=_ToyDMDLosses(student, fake_score, fail_fake=fail_fake),
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        fake_score_optimizer=torch.optim.SGD(fake_score.parameters(), lr=0.1),
        generator_update_interval=generator_interval,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_scheduler_cadence="iteration",
    )
    return engine, student_scheduler, fake_scheduler


def _empty_dmd_batch() -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=("sample",),
        clean_latents=torch.zeros(1, 1),
        conditioning={},
        unconditional_conditioning={},
    )


def test_native_dmd_engine_owns_official_two_optimizer_cadence_and_state() -> None:
    engine, student_scheduler, fake_scheduler = _dmd_engine()

    results = [engine.train_step(_empty_dmd_batch()) for _ in range(3)]

    assert [result.generator_updated for result in results] == [False, True, False]
    assert engine.global_step == 3
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 3
    assert student_scheduler.steps == 3
    assert fake_scheduler.steps == 3

    restored, _, _ = _dmd_engine()
    restored.load_state_dict(engine.state_dict())
    assert restored.global_step == 3
    assert restored.student_optimizer_steps == 1
    assert restored.fake_score_optimizer_steps == 3


def test_native_dmd_engine_refuses_checkpoint_after_partial_optimizer_commit() -> None:
    engine, _, _ = _dmd_engine(fail_fake=True, generator_interval=1)

    with pytest.raises(RuntimeError, match="intentional"):
        engine.train_step(_empty_dmd_batch())
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="restore the last checkpoint"):
        engine.train_step(_empty_dmd_batch())


def test_native_dmd_engine_poisoning_starts_before_optimizer_returns() -> None:
    student = torch.nn.Linear(1, 1, bias=False)
    fake_score = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    engine = NativeDMDTrainEngine(
        student_module=student,
        real_score_module=teacher,
        fake_score_module=fake_score,
        loss_adapter=_ToyDMDLosses(student, fake_score),
        student_optimizer=_FailAfterCommitSGD(student.parameters(), lr=0.1),
        fake_score_optimizer=torch.optim.SGD(fake_score.parameters(), lr=0.1),
        generator_update_interval=1,
    )

    with pytest.raises(RuntimeError, match="failure after optimizer mutation"):
        engine.train_step(_empty_dmd_batch())

    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="restore the last checkpoint"):
        engine.train_step(_empty_dmd_batch())


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_dmd_stack(seed: int):
    torch.manual_seed(seed)
    engine, _, _ = _dmd_engine()
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "fake_score": engine.fake_score_module,
        }
    )
    progress = TrainingProgress()
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.fake_score_optimizer),
        engine=engine,
        dataloader=_StatefulLoader(),
        objective_generator=torch.Generator().manual_seed(101),
        progress=progress,
        identity={
            "algorithm": "dmd",
            "parallel_plan": {"backend": "single", "world_size": 1},
        },
    )
    return engine, model, progress, state


def test_training_dcp_exactly_restores_dmd_models_both_optimizers_and_cadence(tmp_path: Path) -> None:
    baseline_engine, baseline_model, baseline_progress, baseline_state = _checkpointable_dmd_stack(53)
    first = baseline_engine.train_step(_empty_dmd_batch())
    baseline_progress.record_step(microbatches=1, samples=1, latent_tokens=1)
    assert first.generator_updated is False
    manager = TrainingCheckpointer(tmp_path / "dmd-checkpoints")
    artifact = manager.save(baseline_state)

    expected = baseline_engine.train_step(_empty_dmd_batch())
    expected_parameters = {name: value.detach().clone() for name, value in baseline_model.state_dict().items()}

    restored_engine, restored_model, restored_progress, restored_state = _checkpointable_dmd_stack(59)
    manager.load(restored_state, artifact.path)
    actual = restored_engine.train_step(_empty_dmd_batch())

    assert restored_progress.optimizer_steps == 1
    assert actual.generator_updated is True
    torch.testing.assert_close(actual.fake_score_loss, expected.fake_score_loss, rtol=0, atol=0)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


class _Policy:
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


def _trajectory(policy: _Policy, revision: str):
    return FlowTrajectorySampler(policy, eta=0.7).sample(
        torch.randn(4, 2, generator=torch.Generator().manual_seed(41)),
        torch.tensor([1.0, 0.6, 0.0]),
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        conditioning={},
        policy_revision=revision,
        generator=torch.Generator().manual_seed(43),
    )


def test_native_flow_grpo_freezes_old_anchor_across_multiple_policy_updates() -> None:
    policy = _Policy(0.2)
    replay = NativeFlowTrajectoryReplay(policy)
    engine = NativeFlowGRPOEngine(
        replay,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        updates_per_trajectory=2,
    )
    trajectory = _trajectory(policy, "policy-root")
    anchor = engine.prepare_trajectory(trajectory, torch.tensor([1.0, 3.0, 2.0, 5.0]))
    before = policy.module.weight.detach().clone()

    first = engine.train_step(anchor_id=anchor)
    first_revision = engine.current_policy_revision
    second = engine.train_step(anchor_id=anchor)

    torch.testing.assert_close(first.metrics["ratio_mean"], torch.tensor(1.0), rtol=0, atol=0)
    assert first.trajectory_complete is False
    assert second.trajectory_complete is True
    assert first_revision != "policy-root"
    assert engine.current_policy_revision != first_revision
    assert engine.global_step == 2
    assert not engine.has_active_trajectory
    assert not torch.equal(policy.module.weight.detach(), before)


def test_native_flow_grpo_rejects_stale_policy_revision_and_checkpoints_at_boundaries() -> None:
    policy = _Policy(0.2)
    engine = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
    )
    trajectory = _trajectory(policy, "policy-root")
    anchor = engine.prepare_trajectory(trajectory, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(RuntimeError, match="trajectory boundary"):
        engine.state_dict()
    engine.train_step(anchor_id=anchor)
    state = engine.state_dict()

    with pytest.raises(ValueError, match="revision differs"):
        engine.prepare_trajectory(trajectory, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    restored_policy = _Policy(0.2)
    restored = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(restored_policy),
        torch.optim.SGD(restored_policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
    )
    restored.load_state_dict(state)
    assert restored.global_step == 1
    assert restored.current_policy_revision == engine.current_policy_revision


def test_flow_grpo_replay_microbatch_matches_full_batch_update() -> None:
    full_policy = _Policy(0.2)
    chunked_policy = _Policy(0.2)
    trajectory = _trajectory(full_policy, "policy-root")
    rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
    full = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(full_policy),
        torch.optim.SGD(full_policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
    )
    chunked = NativeFlowGRPOEngine(
        NativeFlowTrajectoryReplay(chunked_policy),
        torch.optim.SGD(chunked_policy.module.parameters(), lr=0.05),
        initial_policy_revision="policy-root",
        replay_microbatch_size=1,
    )

    full_result = full.train_step(anchor_id=full.prepare_trajectory(trajectory, rewards))
    chunked_result = chunked.train_step(anchor_id=chunked.prepare_trajectory(trajectory, rewards))

    torch.testing.assert_close(
        chunked_policy.module.weight,
        full_policy.module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        chunked_result.loss,
        full_result.loss,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    for name in (
        "ratio_mean",
        "ratio_std",
        "ratio_min",
        "ratio_max",
        "approx_kl",
        "clip_fraction",
    ):
        torch.testing.assert_close(
            chunked_result.metrics[name],
            full_result.metrics[name],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    assert chunked.state_dict()["replay_microbatch_size"] == 1
