from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.causal_consistency.config import (  # noqa: E402
    CausalConsistencyConfig,
    build_causal_consistency_schedule,
)
from worldfoundry.training.post_training.distillation.causal_consistency.contracts import (  # noqa: E402
    CausalConsistencyTrainingBatch,
)
from worldfoundry.training.post_training.distillation.causal_consistency.engine import (  # noqa: E402
    NativeCausalConsistencyTrainEngine,
)
from worldfoundry.training.post_training.distillation.causal_consistency.objective import (  # noqa: E402
    CausalConsistencyObjective,
)
from worldfoundry.training.post_training.distillation.causal_ode.config import (  # noqa: E402
    CausalODEConfig,
    warped_causal_ode_timesteps,
)
from worldfoundry.training.post_training.distillation.causal_ode.contracts import (  # noqa: E402
    CausalODETrainingBatch,
)
from worldfoundry.training.post_training.distillation.causal_ode.engine import (  # noqa: E402
    NativeCausalODETrainEngine,
)
from worldfoundry.training.post_training.distillation.causal_ode.objective import (  # noqa: E402
    CausalODEObjective,
)


class _ScaleModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(value)))


class _CleanAdapter:
    def __init__(self, value: float) -> None:
        self.module = _ScaleModule(value)
        self.calls: list[dict[str, object]] = []

    def predict_clean(
        self,
        noisy_latents,
        timesteps,
        *,
        clean_context,
        sample_ids,
        conditioning,
        training,
    ):
        del conditioning
        self.calls.append(
            {
                "timesteps": timesteps.detach().clone(),
                "clean_context": clean_context.detach().clone(),
                "sample_ids": sample_ids,
                "training": training,
                "module_training": self.module.training,
            }
        )
        return noisy_latents * self.module.scale


class _VelocityAdapter:
    def __init__(self, value: float) -> None:
        self.module = _ScaleModule(value)
        self.calls: list[dict[str, object]] = []

    def predict_velocity(
        self,
        noisy_latents,
        timesteps,
        *,
        clean_context,
        sample_ids,
        conditioning,
        training,
    ):
        del clean_context, sample_ids
        bias = torch.as_tensor(
            conditioning["bias"],
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        self.calls.append(
            {
                "timesteps": timesteps.detach().clone(),
                "training": training,
                "module_training": self.module.training,
                "bias": float(bias.item()),
            }
        )
        return noisy_latents * self.module.scale + bias


def _ode_batch(*, offset: float = 0.0) -> CausalODETrainingBatch:
    trajectory = torch.stack(
        (
            torch.full((1, 1, 3, 1, 1), 1.0 + offset),
            torch.full((1, 1, 3, 1, 1), 2.0 + offset),
            torch.full((1, 1, 3, 1, 1), 4.0 + offset),
        ),
        dim=1,
    )
    return CausalODETrainingBatch(
        sample_ids=(f"sample-{offset}",),
        ode_trajectories=trajectory,
        conditioning={"context": torch.ones(1, 2, 3)},
    )


def _consistency_batch(*, sample_id: str = "sample") -> CausalConsistencyTrainingBatch:
    return CausalConsistencyTrainingBatch(
        sample_ids=(sample_id,),
        clean_latents=torch.ones(1, 1, 2, 1, 1),
        conditioning={"bias": 2.0},
        unconditional_conditioning={"bias": 1.0},
    )


def _consistency_stack(
    *,
    seed: int = 7,
    accumulation: int = 1,
    parallel_context=None,
):
    student = _CleanAdapter(0.5)
    teacher = _VelocityAdapter(0.25)
    ema_student = _CleanAdapter(-3.0)
    config = CausalConsistencyConfig(
        num_levels=4,
        num_train_timesteps=1000,
        flow_shift=1.0,
        extra_terminal_step=True,
        guidance_scale=3.0,
        ema_decay=0.5,
        frame_dim=2,
    )
    objective = CausalConsistencyObjective(
        student=student,
        teacher=teacher,
        ema_student=ema_student,
        config=config,
    )
    optimizer = torch.optim.SGD(student.module.parameters(), lr=0.05)
    engine = NativeCausalConsistencyTrainEngine(
        student_module=student.module,
        teacher_module=teacher.module,
        ema_student_module=ema_student.module,
        objective=objective,
        optimizer=optimizer,
        gradient_accumulation_steps=accumulation,
        parallel_context=parallel_context,
        seed=seed,
    )
    return student, teacher, ema_student, objective, engine


def test_causal_ode_released_warped_schedule_fixture() -> None:
    timesteps = warped_causal_ode_timesteps((1000, 750, 500, 250))
    assert timesteps == pytest.approx((1000.0, 937.5, 833.3333129882812, 625.0))
    with_terminal = warped_causal_ode_timesteps((1000, 0))
    assert with_terminal == pytest.approx((1000.0, 0.0))


def test_causal_ode_samples_one_index_per_sample_and_uses_exact_mask_denominator() -> None:
    student = _CleanAdapter(2.0)
    objective = CausalODEObjective(
        student,
        CausalODEConfig(trajectory_timesteps=(10.0, 0.0), frame_dim=2),
    )
    first = torch.stack(
        (
            torch.ones(1, 3, 1, 1),
            torch.full((1, 3, 1, 1), 9.0),
            torch.full((1, 3, 1, 1), 4.0),
        )
    )
    second = torch.stack(
        (
            torch.full((1, 3, 1, 1), 100.0),
            torch.full((1, 3, 1, 1), 20.0),
            torch.full((1, 3, 1, 1), 8.0),
        )
    )
    batch = CausalODETrainingBatch(
        sample_ids=("first", "second"),
        ode_trajectories=torch.stack((first, second)),
        conditioning={"context": torch.ones(2, 4, 5)},
    )
    prepared = objective.prepare(batch, torch.tensor([0, 1], dtype=torch.int64))
    assert prepared.loss_denominator == 3
    assert prepared.timesteps.shape == (2, 3)
    torch.testing.assert_close(prepared.timesteps[0], torch.full((3,), 10.0))
    torch.testing.assert_close(prepared.timesteps[1], torch.zeros(3))
    torch.testing.assert_close(prepared.clean_context, batch.ode_trajectories[:, -1])
    torch.testing.assert_close(prepared.target_latents, batch.ode_trajectories[:, -2])
    result = objective.loss(prepared)
    assert result.metrics["loss_denominator"] == 3
    assert result.loss.item() == pytest.approx(49.0)
    assert student.calls[-1]["training"] is True

    with pytest.raises(ValueError, match="empty"):
        objective.prepare(batch, torch.tensor([1, 1], dtype=torch.int64))


def test_causal_ode_engine_accumulates_and_restores_trajectory_rng() -> None:
    student = _CleanAdapter(0.5)
    objective = CausalODEObjective(
        student,
        CausalODEConfig(trajectory_timesteps=(10.0, 5.0), frame_dim=2),
    )
    optimizer = torch.optim.SGD(student.module.parameters(), lr=0.05)
    engine = NativeCausalODETrainEngine(
        student_module=student.module,
        objective=objective,
        optimizer=optimizer,
        gradient_accumulation_steps=2,
        seed=19,
    )
    before = student.module.scale.detach().clone()
    result = engine.train_step((_ode_batch(), _ode_batch(offset=1.0)))
    assert result.loss.isfinite()
    assert engine.global_step == engine.optimizer_steps == 1
    assert not torch.equal(student.module.scale.detach(), before)
    state = engine.state_dict()
    expected = objective.sample_trajectory_indices(_ode_batch(), generator=engine._rng)

    restored_student = _CleanAdapter(0.5)
    restored_objective = CausalODEObjective(restored_student, objective.config)
    restored = NativeCausalODETrainEngine(
        student_module=restored_student.module,
        objective=restored_objective,
        optimizer=torch.optim.SGD(restored_student.module.parameters(), lr=0.05),
        gradient_accumulation_steps=2,
        seed=999,
    )
    restored.load_state_dict(state)
    actual = restored_objective.sample_trajectory_indices(
        _ode_batch(),
        generator=restored._rng,
    )
    torch.testing.assert_close(actual, expected)


def test_causal_consistency_48_level_extra_terminal_schedule_fixture() -> None:
    config = CausalConsistencyConfig()
    schedule = build_causal_consistency_schedule(config)
    assert len(schedule.timesteps) == 48
    assert schedule.pair_count == 47
    assert schedule.extra_terminal_step is True
    assert schedule.timesteps[0] == pytest.approx(1000.0)
    assert schedule.timesteps[23] == pytest.approx(125000.0 / 148.0, rel=2e-6)
    assert schedule.timesteps[-1] == pytest.approx(5000.0 / 52.0, rel=2e-6)
    assert schedule.timesteps[-1] > 0

    without_terminal_spacing = build_causal_consistency_schedule(
        CausalConsistencyConfig(extra_terminal_step=False)
    )
    assert without_terminal_spacing.timesteps[-1] == pytest.approx(0.0)
    assert without_terminal_spacing.timesteps != schedule.timesteps


def test_causal_consistency_exact_cfg_euler_and_detached_ema_formula() -> None:
    student = _CleanAdapter(2.0)
    teacher = _VelocityAdapter(0.0)
    ema_student = _CleanAdapter(0.5)
    objective = CausalConsistencyObjective(
        student=student,
        teacher=teacher,
        ema_student=ema_student,
        config=CausalConsistencyConfig(
            num_levels=4,
            num_train_timesteps=1000,
            flow_shift=1.0,
            extra_terminal_step=True,
            guidance_scale=2.0,
            ema_decay=0.5,
            frame_dim=2,
        ),
    )
    batch = _consistency_batch()
    noise = torch.full_like(batch.clean_latents, 3.0)
    result = objective.loss(batch, pair_index=0, noise=noise)
    # x_t=3; CFG velocity=1+2*(2-1)=3; x_next=3-0.25*3=2.25.
    # online=2*x_t=6; EMA target=.5*x_next=1.125.
    assert result.loss.item() == pytest.approx((6.0 - 1.125) ** 2)
    assert result.metrics["pair_index"] == 0
    assert result.metrics["timestep"] == pytest.approx(1000.0)
    assert result.metrics["next_timestep"] == pytest.approx(750.0)
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in ema_student.module.parameters())
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in ema_student.module.parameters())
    assert teacher.calls[0]["bias"] == 2.0
    assert teacher.calls[1]["bias"] == 1.0


def test_causal_consistency_engine_shares_pair_freezes_roles_and_updates_ema() -> None:
    student, teacher, ema_student, _, engine = _consistency_stack(accumulation=2)
    old_student = student.module.scale.detach().clone()
    old_target = ema_student.module.scale.detach().clone()
    torch.testing.assert_close(old_target, old_student)
    result = engine.train_step(
        (_consistency_batch(sample_id="first"), _consistency_batch(sample_id="second"))
    )
    assert 0 <= result.pair_index < 3
    assert engine.last_pair_index == result.pair_index
    assert engine.global_step == engine.optimizer_steps == 1
    assert student.module.training is True
    assert teacher.module.training is False
    assert ema_student.module.training is False
    assert len(student.calls) == 2
    torch.testing.assert_close(student.calls[0]["timesteps"], student.calls[1]["timesteps"])
    new_student = student.module.scale.detach()
    expected_target = old_target * 0.5 + new_student * 0.5
    torch.testing.assert_close(ema_student.module.scale.detach(), expected_target)
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in ema_student.module.parameters())


def test_causal_consistency_engine_state_replays_pair_and_noise_rng() -> None:
    _, _, _, _, engine = _consistency_stack(seed=31)
    state = engine.state_dict()
    assert state["last_pair_index"] == -1
    expected_pair = engine.sample_pair_index()
    expected_noise = torch.randn((2, 3), generator=engine._noise_rng)

    _, _, _, _, restored = _consistency_stack(seed=999)
    restored.load_state_dict(state)
    assert restored.sample_pair_index() == expected_pair
    actual_noise = torch.randn((2, 3), generator=restored._noise_rng)
    torch.testing.assert_close(actual_noise, expected_noise)


class _FakeParallelContext:
    rank = 1
    world_size = 2
    process_group = object()

    def __init__(self) -> None:
        self.broadcasts: list[object] = []

    def audit_synchronized_module(self, module, *, role: str) -> None:
        del module, role

    def broadcast_from_coordinator(self, value):
        self.broadcasts.append(self.process_group)
        value.fill_(2)
        return value


def test_causal_consistency_pair_is_broadcast_from_rank_zero() -> None:
    context = _FakeParallelContext()
    _, _, _, _, engine = _consistency_stack(parallel_context=context)
    assert engine.sample_pair_index() == 2
    assert context.broadcasts == [_FakeParallelContext.process_group]
