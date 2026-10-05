from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.ddrl import (  # noqa: E402
    DDRL_ENGINE_STATE_SCHEMA,
    DDRLTrajectory,
    NativeDDRLEngine,
)


class _ReplayAdapter:
    def __init__(self, gain: float = 0.25) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)
        self.calls: list[int] = []

    def replay_mean(self, trajectory, train_on_position, *, training):
        self.module.train(training)
        self.calls.append(train_on_position)
        noisy = trajectory.replay_inputs["noisy"][:, train_on_position]
        return noisy * self.module.weight.reshape(())


class _DataRegularizer:
    def __init__(self, replay: _ReplayAdapter) -> None:
        self.module = replay.module
        self.calls: list[int] = []

    def loss(self, trajectory, train_on_position, *, generator, training):
        del generator
        self.module.train(training)
        self.calls.append(train_on_position)
        noisy = trajectory.replay_inputs["data_noisy"][:, train_on_position]
        target = trajectory.replay_inputs["data_target"][:, train_on_position]
        prediction = noisy * self.module.weight.reshape(())
        return (prediction - target).square()


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.05, momentum=0.2)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _RecordingParallelContext:
    world_size = 3

    def __init__(self) -> None:
        self.weights: list[float] = []

    def audit_synchronized_module(self, module, *, role) -> None:
        del module, role

    def audit_local_group_ownership(self, group_ids) -> None:
        assert group_ids == ("first", "first", "second", "second")

    def scale_local_mean(self, local_mean, local_weight):
        self.weights.append(float(local_weight))
        return local_mean


def _trajectory(trajectory_id: str, *, shift: float = 0.0, reference: bool = True):
    noisy = torch.linspace(-0.4, 0.7, 12).reshape(4, 3, 1) + shift
    old = noisy * 0.2
    return DDRLTrajectory(
        trajectory_id=trajectory_id,
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        train_on=(0, 2, 4),
        next_latents=noisy * 0.3 + 0.1,
        old_means=old,
        reference_means=old * 0.5 if reference else None,
        terminal_latents=torch.tensor([[0.0], [2.0], [1.0], [5.0]]),
        replay_inputs={
            "noisy": noisy,
            "data_noisy": noisy + 0.2,
            "data_target": torch.zeros_like(noisy),
        },
    )


def _engine(*, loss_scale: float = 10.0, max_grad_norm: float = 1.0, parallel_context=None):
    replay = _ReplayAdapter()
    data = _DataRegularizer(replay)
    optimizer = _CountingSGD(replay.module.parameters())
    engine = NativeDDRLEngine(
        replay,
        optimizer,
        clip_range=0.2,
        loss_scale=loss_scale,
        kl_beta=0.1,
        data_beta=0.25,
        data_regularizer=data,
        data_on_first_step_only=True,
        max_grad_norm=max_grad_norm,
        parallel_context=parallel_context,
    )
    return engine, replay, data, optimizer


def test_engine_accumulates_every_train_on_step_before_one_optimizer_commit() -> None:
    parallel = _RecordingParallelContext()
    engine, replay, data, optimizer = _engine(parallel_context=parallel)
    trajectory = _trajectory("trajectory-1")
    old_before = trajectory.old_means.clone()
    reference_before = trajectory.reference_means.clone()

    result = engine.train_trajectory(
        trajectory,
        torch.tensor([0.0, 2.0, 1.0, 5.0]),
    )

    assert replay.calls == [0, 1, 2]
    assert data.calls == [0]
    assert optimizer.step_calls == 1
    assert parallel.weights == [12.0]
    assert result.ratios.shape == (4, 3)
    assert result.train_on == (0, 2, 4)
    assert result.reference_kl is not None
    assert result.data_loss is not None
    torch.testing.assert_close(trajectory.old_means, old_before, rtol=0, atol=0)
    torch.testing.assert_close(trajectory.reference_means, reference_before, rtol=0, atol=0)
    assert engine.state_dict() == {
        "schema": DDRL_ENGINE_STATE_SCHEMA,
        "global_step": 1,
        "optimizer_steps": 1,
        "last_trajectory_id": "trajectory-1",
        "clip_range": 0.2,
        "loss_scale": 10.0,
        "advantage_epsilon": 1.0e-4,
        "advantage_normalization": "group-sample-std",
        "advantage_clip_min": None,
        "advantage_clip_max": None,
        "exponential_advantage": False,
        "kl_beta": 0.1,
        "data_beta": 0.25,
        "data_on_first_step_only": True,
        "max_grad_norm": 1.0,
        "data_parallel_size": 3,
    }


def test_engine_state_restores_exact_next_trajectory_update() -> None:
    engine, replay, _, optimizer = _engine()
    engine.train_trajectory(_trajectory("trajectory-1"), torch.tensor([0.0, 2.0, 1.0, 5.0]))
    engine_state = copy.deepcopy(engine.state_dict())
    policy_state = copy.deepcopy(replay.module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    expected = engine.train_trajectory(
        _trajectory("trajectory-2", shift=0.1),
        torch.tensor([1.0, 3.0, 0.0, 4.0]),
    )

    restored, restored_replay, _, restored_optimizer = _engine()
    restored_replay.module.load_state_dict(policy_state)
    restored_optimizer.load_state_dict(optimizer_state)
    restored.load_state_dict(engine_state)
    assert restored.state_dict() == engine_state
    actual = restored.train_trajectory(
        _trajectory("trajectory-2", shift=0.1),
        torch.tensor([1.0, 3.0, 0.0, 4.0]),
    )

    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(restored_replay.module.weight, replay.module.weight, rtol=0, atol=0)
    assert restored.global_step == engine.global_step == 2


def test_engine_scales_only_backward_gradient_and_parameter_update() -> None:
    unit_engine, unit_replay, _, _ = _engine(loss_scale=1.0, max_grad_norm=1.0e6)
    scaled_engine, scaled_replay, _, _ = _engine(loss_scale=10.0, max_grad_norm=1.0e6)
    initial_weight = unit_replay.module.weight.detach().clone()

    unit_result = unit_engine.train_trajectory(
        _trajectory("unit-scale"),
        torch.tensor([0.0, 2.0, 1.0, 5.0]),
    )
    scaled_result = scaled_engine.train_trajectory(
        _trajectory("official-scale"),
        torch.tensor([0.0, 2.0, 1.0, 5.0]),
    )

    torch.testing.assert_close(scaled_result.loss, unit_result.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        scaled_result.policy_loss,
        unit_result.policy_loss,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scaled_result.reference_kl,
        unit_result.reference_kl,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scaled_result.data_loss,
        unit_result.data_loss,
        rtol=0,
        atol=0,
    )
    unit_update = initial_weight - unit_replay.module.weight.detach()
    scaled_update = initial_weight - scaled_replay.module.weight.detach()
    torch.testing.assert_close(scaled_update, unit_update * 10.0)
    torch.testing.assert_close(
        scaled_result.gradient_norm,
        unit_result.gradient_norm * 10.0,
    )


def test_engine_requires_reference_anchor_when_reference_weight_is_enabled() -> None:
    engine, _, _, _ = _engine()

    with pytest.raises(ValueError, match="reference_means"):
        engine.train_trajectory(
            _trajectory("trajectory-no-reference", reference=False),
            torch.tensor([0.0, 2.0, 1.0, 5.0]),
        )
