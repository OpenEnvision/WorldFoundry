from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDLossResult,
    DMDTrainingBatch,
    NativeDMDTrainEngine,
    NativeDMDTrainingSession,
)


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


class _WeightedDMDLosses:
    def __init__(
        self,
        student: torch.nn.Linear,
        fake_score: torch.nn.Linear,
        *,
        use_generator: bool = False,
        fail_fake_call: int | None = None,
    ) -> None:
        self.student = student
        self.fake_score = fake_score
        self.use_generator = use_generator
        self.fail_fake_call = fail_fake_call
        self.fake_calls = 0

    def loss_denominator(self, batch, *, role):
        del role
        return torch.tensor(
            batch.clean_latents.numel(),
            device=batch.clean_latents.device,
            dtype=torch.float32,
        )

    def _jitter(self, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
        if not self.use_generator:
            return torch.zeros((), device=device)
        return torch.rand((), generator=generator, device=device) * 0.05

    def generator_loss(self, batch, *, generator=None) -> DMDLossResult:
        values = batch.clean_latents.float()
        prediction = self.student(values)
        target = values * (0.25 + self._jitter(generator, values.device))
        loss = (prediction - target).square().mean()
        denominator = torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
        return DMDLossResult(
            loss,
            {
                "loss_numerator": loss.detach() * denominator,
                "loss_denominator": denominator,
            },
        )

    def fake_score_loss(self, batch, *, generator=None) -> DMDLossResult:
        self.fake_calls += 1
        if self.fake_calls == self.fail_fake_call:
            raise RuntimeError("intentional accumulated fake-score failure")
        values = batch.clean_latents.float()
        prediction = self.fake_score(values)
        target = self.student(values).detach() + self._jitter(generator, values.device)
        loss = (prediction - target).square().mean()
        denominator = torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
        return DMDLossResult(
            loss,
            {
                "loss_numerator": loss.detach() * denominator,
                "loss_denominator": denominator,
            },
        )


def _batch(values: list[float], *, prefix: str = "sample") -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(len(values))),
        clean_latents=torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
        conditioning={},
        unconditional_conditioning={},
    )


def _engine(
    *,
    seed: int,
    accumulation_steps: int,
    generator_interval: int = 2,
    scheduler_cadence: str = "generator-update",
    use_generator: bool = False,
    fail_fake_call: int | None = None,
):
    torch.manual_seed(seed)
    student = torch.nn.Linear(1, 1, bias=False)
    fake_score = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    student_scheduler = _Counter()
    fake_scheduler = _Counter()
    losses = _WeightedDMDLosses(
        student,
        fake_score,
        use_generator=use_generator,
        fail_fake_call=fail_fake_call,
    )
    engine = NativeDMDTrainEngine(
        student_module=student,
        real_score_module=teacher,
        fake_score_module=fake_score,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake_score.parameters(), lr=0.05),
        generator_update_interval=generator_interval,
        gradient_accumulation_steps=accumulation_steps,
        student_max_grad_norm=1000.0,
        fake_score_max_grad_norm=1000.0,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_scheduler_cadence=scheduler_cadence,
    )
    return engine, student_scheduler, fake_scheduler


def test_dmd_uneven_microbatch_accumulation_matches_one_combined_batch() -> None:
    accumulated, student_scheduler, fake_scheduler = _engine(seed=13, accumulation_steps=2)
    combined, _, _ = _engine(seed=13, accumulation_steps=1)
    first = _batch([1.0], prefix="first")
    second = _batch([2.0, 3.0, 4.0], prefix="second")
    merged = _batch([1.0, 2.0, 3.0, 4.0], prefix="merged")

    warmup_accumulated = accumulated.train_step((first, second))
    warmup_combined = combined.train_step(merged)
    assert warmup_accumulated.generator_updated is False
    assert warmup_combined.generator_updated is False

    accumulated_result = accumulated.train_step((first, second))
    combined_result = combined.train_step(merged)

    torch.testing.assert_close(
        accumulated.student_module.weight,
        combined.student_module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        accumulated.fake_score_module.weight,
        combined.fake_score_module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(accumulated_result.generator_loss, combined_result.generator_loss)
    torch.testing.assert_close(accumulated_result.fake_score_loss, combined_result.fake_score_loss)
    assert accumulated_result.metrics["accumulated_microbatches"] == 2
    assert accumulated_result.metrics["student"]["loss_denominator"].item() == 4
    assert accumulated_result.metrics["fake_score"]["loss_denominator"].item() == 4
    assert student_scheduler.steps == 1
    assert fake_scheduler.steps == 2

    skipped = accumulated.train_step((first, second))
    assert skipped.generator_updated is False
    assert student_scheduler.steps == 1
    assert fake_scheduler.steps == 3
    assert accumulated.student_optimizer_steps == 1
    assert accumulated.fake_score_optimizer_steps == 3


def test_dmd_accumulation_poisoning_covers_failure_after_student_commit() -> None:
    engine, student_scheduler, fake_scheduler = _engine(
        seed=17,
        accumulation_steps=2,
        generator_interval=1,
        fail_fake_call=2,
    )
    before = engine.student_module.weight.detach().clone()

    with pytest.raises(RuntimeError, match="accumulated fake-score failure"):
        engine.train_step((_batch([1.0]), _batch([2.0])))

    assert not torch.equal(engine.student_module.weight.detach(), before)
    assert student_scheduler.steps == 0
    assert fake_scheduler.steps == 0
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="restore the last checkpoint"):
        engine.train_step((_batch([1.0]), _batch([2.0])))


class _RecordingCheckpointer:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def save(self, state, *, asynchronous: bool):
        del asynchronous
        self.steps.append(state.engine.global_step)
        return object()


def test_dmd_session_counts_accumulated_work_and_checkpoints_only_after_commit() -> None:
    engine, _, _ = _engine(seed=19, accumulation_steps=2)
    progress = TrainingProgress()
    checkpointer = _RecordingCheckpointer()
    boundaries: list[tuple[int, int]] = []
    events: list[object] = []
    session = NativeDMDTrainingSession(
        engine,
        [_batch([1.0]), _batch([2.0, 3.0, 4.0])],
        progress,
        checkpoint_state=type("_State", (), {"engine": engine})(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=2,
        event_sink=events.append,
    )

    summary = session.run(
        max_steps=3,
        boundary_every_steps=2,
        boundary_sink=lambda previous, current: boundaries.append((previous, current)),
    )

    assert summary.final_step == 3
    assert progress.optimizer_steps == 3
    assert progress.microbatches_seen == 6
    assert progress.samples_seen == 12
    assert progress.latent_tokens_seen == 12
    assert checkpointer.steps == [2]
    assert boundaries == [(1, 2)]
    assert [event["microbatches"] for event in events] == [2, 2, 2]
    assert [event["samples"] for event in events] == [4, 4, 4]


class _StatefulDMDLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> DMDTrainingBatch:
        value = float(self.cursor + 1)
        self.cursor += 1
        return _batch([value], prefix=f"cursor-{self.cursor}")

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack(seed: int):
    engine, _, _ = _engine(
        seed=seed,
        accumulation_steps=2,
        use_generator=True,
    )
    loader = _StatefulDMDLoader()
    progress = TrainingProgress()
    objective_generator = torch.Generator().manual_seed(101)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "fake_score": engine.fake_score_module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.fake_score_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=objective_generator,
        progress=progress,
        identity={
            "algorithm": "dmd",
            "gradient_accumulation_steps": engine.gradient_accumulation_steps,
        },
    )
    return engine, loader, progress, objective_generator, model, state


def test_dmd_accumulation_dcp_resume_restores_microbatch_cursor_rng_and_cadence(tmp_path: Path) -> None:
    (baseline_engine, baseline_loader, baseline_progress, baseline_generator, baseline_model, baseline_state) = (
        _checkpointable_stack(23)
    )
    baseline_session = NativeDMDTrainingSession(baseline_engine, baseline_loader, baseline_progress)
    baseline_session.run(max_steps=1, generator=baseline_generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(baseline_state)

    expected = baseline_session.run(max_steps=1, generator=baseline_generator)
    expected_parameters = {name: value.detach().clone() for name, value in baseline_model.state_dict().items()}

    (restored_engine, restored_loader, restored_progress, restored_generator, restored_model, restored_state) = (
        _checkpointable_stack(29)
    )
    manager.load(restored_state, artifact.path)
    restored_session = NativeDMDTrainingSession(restored_engine, restored_loader, restored_progress)
    actual = restored_session.run(max_steps=1, generator=restored_generator)

    assert restored_progress.optimizer_steps == 2
    assert restored_progress.microbatches_seen == 4
    assert restored_loader.cursor == 4
    assert actual.student_optimizer_steps == expected.student_optimizer_steps
    assert actual.fake_score_optimizer_steps == expected.fake_score_optimizer_steps
    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
