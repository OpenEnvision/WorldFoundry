from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection  # noqa: E402
from worldfoundry.training.post_training.distillation.sgmd import (  # noqa: E402
    NativeSGMDLossAdapter,
    NativeSGMDTrainEngine,
    NativeSGMDTrainingSession,
    SGMDConfig,
    SGMDLossResult,
    SGMDTrainingBatch,
    build_native_sgmd_training_stack,
    simulate_sgmd_student,
)
from worldfoundry.training.post_training.distillation.sgmd.math import (  # noqa: E402
    flow_clean_from_velocity,
    flow_interpolate,
    sgmd_fake_correction_loss_per_sample,
    sgmd_normalized_fisher_loss_per_sample,
)
from worldfoundry.training.post_training.distillation.sgmd.objective import (  # noqa: E402
    sample_sgmd_score_sigmas,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class _Adapter:
    noise_process_kind = "flow-matching"

    def __init__(self, value: float, identity: str, *, frozen: bool = False) -> None:
        self.module = _Scale(value)
        self.checkpoint_identity = identity
        self.grad_enabled: list[bool] = []
        self.training_flags: list[bool] = []
        if frozen:
            self.module.requires_grad_(False)

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
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_flags.append(bool(training))
        return noisy_latents * self.module.weight


def _batch(
    values=(0.0, 0.0),
    *,
    prefix="sample",
    weights=None,
) -> SGMDTrainingBatch:
    return SGMDTrainingBatch(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(len(values))),
        latent_template=torch.tensor(values, dtype=torch.float32).reshape(len(values), 1),
        conditioning={},
        unconditional_conditioning={},
        sample_weights=None if weights is None else torch.tensor(weights, dtype=torch.float32),
    )


def _small_config(**changes) -> SGMDConfig:
    values = {
        "student_timesteps": (1000.0, 750.0, 500.0),
        "diversity_teacher_steps": 3,
        "diversity_anchor_step": 1,
    }
    values.update(changes)
    return SGMDConfig(**values)


def _recipe(*, accumulation: int = 1) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "sgmd-test", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "same-base",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "prompts.jsonl"},
            "algorithm": {
                "type": "sgmd",
                "student_timesteps": [1000, 750, 500],
                "teacher_checkpoint": "same-base",
                "fake_score_checkpoint": "same-base",
                "diversity_teacher_steps": 3,
                "diversity_anchor_step": 1,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2.0e-6,
                "weight_decay": 0.01,
                "betas": [0.0, 0.999],
                "max_grad_norm": 10.0,
                "gradient_accumulation_steps": accumulation,
            },
            "fake_score_optimizer": {
                "type": "adamw",
                "learning_rate": 4.0e-7,
                "weight_decay": 0.01,
                "betas": [0.0, 0.999],
                "max_grad_norm": 10.0,
                "gradient_accumulation_steps": accumulation,
            },
            "export": {"format": "safetensors"},
        }
    )


def test_rollout_detaches_prefix_and_uses_euler_selected_clean() -> None:
    student = _Adapter(0.2, "student")
    batch = _batch((0.0, 0.0))
    initial = torch.ones_like(batch.latent_template)
    result = simulate_sgmd_student(
        student,
        batch,
        _small_config(),
        target_index=1,
        initial_noise=initial,
        training=True,
    )
    assert student.grad_enabled == [False, True]
    assert student.training_flags == [True, True]
    first_next = initial + (0.9375 - 1.0) * (0.2 * initial)
    expected_clean = first_next - 0.9375 * (0.2 * first_next)
    torch.testing.assert_close(result.clean_latents, expected_clean)


def _native_objective(*, diversity: bool):
    student = _Adapter(0.2, "student")
    teacher = _Adapter(0.6, "teacher", frozen=True)
    fake = _Adapter(0.4, "fake")
    config = _small_config(
        diversity_enabled=diversity,
        diversity_weight=0.05 if diversity else 0.0,
    )
    return NativeSGMDLossAdapter(student, teacher, fake, config), student, teacher, fake


def test_student_objective_preserves_fake_input_jacobian_without_fake_parameter_gradients() -> None:
    losses, student, teacher, fake = _native_objective(diversity=False)
    batch = _batch((0.0, 0.0))
    result = losses.student_loss(
        batch,
        target_index=0,
        generator=torch.Generator().manual_seed(101),
    )
    result.loss.backward()
    isolated_student_gradient = student.module.weight.grad.detach().clone()
    assert fake.module.weight.grad is None
    assert teacher.module.weight.grad is None

    reference_student = _Adapter(0.2, "student")
    reference_teacher = _Adapter(0.6, "teacher", frozen=True)
    reference_fake = _Adapter(0.4, "fake")
    config = losses.config
    generator = torch.Generator().manual_seed(101)
    rollout = simulate_sgmd_student(
        reference_student,
        batch,
        config,
        target_index=0,
        generator=generator,
        training=True,
    )
    generated = rollout.clean_latents
    sigmas = sample_sgmd_score_sigmas(generated, config, generator=generator)
    noise = torch.randn(generated.shape, generator=generator, dtype=torch.float32)
    noisy = flow_interpolate(generated, noise, sigmas).to(generated)
    fake_velocity = reference_fake.predict_velocity(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=False,
    )
    fake_clean = flow_clean_from_velocity(noisy, fake_velocity, sigmas)
    with torch.no_grad():
        teacher_velocity = reference_teacher.predict_velocity(
            noisy,
            sigmas,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=False,
        )
        teacher_clean = flow_clean_from_velocity(noisy, teacher_velocity, sigmas)
    fisher, _ = sgmd_normalized_fisher_loss_per_sample(
        generated,
        fake_clean,
        teacher_clean,
    )
    correction = sgmd_fake_correction_loss_per_sample(generated, fake_clean, sigmas)
    reference_loss = (fisher - config.fake_correction_weight * correction).mean()
    reference_loss.backward()
    torch.testing.assert_close(
        isolated_student_gradient,
        reference_student.module.weight.grad,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert reference_fake.module.weight.grad is not None


def test_native_losses_isolate_student_and_fake_score_roles() -> None:
    losses, student, teacher, fake = _native_objective(diversity=True)
    student_result = losses.student_loss(
        _batch(),
        target_index=1,
        generator=torch.Generator().manual_seed(11),
    )
    student_result.loss.backward()
    assert student.module.weight.grad is not None
    assert fake.module.weight.grad is None
    assert teacher.module.weight.grad is None
    student.module.zero_grad(set_to_none=True)

    fake_result = losses.fake_score_loss(
        _batch(),
        target_index=0,
        generator=torch.Generator().manual_seed(13),
    )
    fake_result.loss.backward()
    assert student.module.weight.grad is None
    assert fake.module.weight.grad is not None
    assert teacher.module.weight.grad is None
    assert "fake_clean_diagnostic" in fake_result.metrics


def test_builder_uses_released_optimizer_profile_and_rejects_shared_roles() -> None:
    student = _Adapter(0.2, "same-base")
    teacher = _Adapter(0.6, "same-base", frozen=True)
    fake = _Adapter(0.4, "same-base")
    stack = build_native_sgmd_training_stack(
        _recipe(),
        student=student,
        teacher=teacher,
        fake_score=fake,
        fused_adamw=False,
    )
    assert stack.student_optimizer.param_groups[0]["lr"] == 2.0e-6
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 4.0e-7
    assert stack.student_optimizer.param_groups[0]["betas"] == (0.0, 0.999)
    assert stack.engine.student_max_grad_norm == 10.0
    assert stack.engine.fake_score_max_grad_norm == 10.0

    shared = _Adapter(0.1, "shared")
    fake.module = shared.module
    with pytest.raises(ValueError, match="independently materialized"):
        build_native_sgmd_training_stack(
            _recipe(),
            student=shared,
            teacher=teacher,
            fake_score=fake,
            fused_adamw=False,
        )


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


class _EMA:
    def __init__(self, module: torch.nn.Linear) -> None:
        self.shadow = module.weight.detach().clone()

    def update(self, module: torch.nn.Linear) -> None:
        self.shadow.mul_(0.5).add_(module.weight.detach(), alpha=0.5)

    def state_dict(self):
        return {"shadow": self.shadow.clone()}

    def load_state_dict(self, state_dict) -> None:
        self.shadow.copy_(state_dict["shadow"])


class _EngineLosses:
    num_student_steps = 4
    minimum_student_target_index = 1

    def __init__(self, student, fake, *, fail_fake_call=None, use_generator=False) -> None:
        self.student = student
        self.fake = fake
        self.fail_fake_call = fail_fake_call
        self.use_generator = use_generator
        self.fake_calls = 0
        self.calls: list[tuple[str, tuple[str, ...], int]] = []
        self.student_weights_seen_by_fake: list[float] = []
        self.student_modes_seen_by_fake: list[bool] = []

    def loss_denominator(self, batch, *, role):
        del role
        if batch.sample_weights is None:
            return torch.tensor(float(batch.batch_size))
        return batch.sample_weights.sum()

    def _weights(self, batch):
        if batch.sample_weights is None:
            return torch.ones(batch.batch_size)
        return batch.sample_weights

    def _jitter(self, generator):
        if not self.use_generator:
            return torch.tensor(0.0)
        return torch.rand((), generator=generator) * 0.01

    def _result(self, per_sample, batch):
        weights = self._weights(batch).to(per_sample)
        return SGMDLossResult(
            loss=(per_sample * weights).sum() / weights.sum(),
            metrics={"loss_denominator": weights.sum()},
        )

    def student_loss(self, batch, *, target_index, generator=None):
        self.calls.append(("student", batch.sample_ids, target_index))
        values = batch.latent_template.float()
        prediction = self.student(values)
        target = self.fake(values).detach() + self._jitter(generator)
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return self._result(per_sample, batch)

    def fake_score_loss(self, batch, *, target_index, generator=None):
        self.fake_calls += 1
        if self.fake_calls == self.fail_fake_call:
            raise RuntimeError("intentional fake-score failure")
        self.calls.append(("fake", batch.sample_ids, target_index))
        self.student_weights_seen_by_fake.append(float(self.student.weight.detach()))
        self.student_modes_seen_by_fake.append(self.student.training)
        values = batch.latent_template.float()
        prediction = self.fake(values)
        target = self.student(values).detach() + self._jitter(generator)
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return self._result(per_sample, batch)


def _engine(*, seed: int, accumulation: int, fail_fake_call=None, use_generator=False):
    torch.manual_seed(seed)
    student = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    fake = torch.nn.Linear(1, 1, bias=False)
    losses = _EngineLosses(
        student,
        fake,
        fail_fake_call=fail_fake_call,
        use_generator=use_generator,
    )
    student_scheduler = _Counter()
    fake_scheduler = _Counter()
    ema = _EMA(student)
    engine = NativeSGMDTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
        student_max_grad_norm=1000.0,
        fake_score_max_grad_norm=1000.0,
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_ema=ema,
    )
    return engine, losses, student_scheduler, fake_scheduler, ema


def test_engine_commits_student_before_fresh_fake_batches_and_uses_independent_targets() -> None:
    engine, losses, student_scheduler, fake_scheduler, _ = _engine(seed=23, accumulation=2)
    initial_student = float(engine.student_module.weight.detach())
    result = engine.train_step(
        (
            _batch((1.0,), prefix="student-a"),
            _batch((2.0,), prefix="student-b"),
        ),
        (
            _batch((3.0,), prefix="fake-a"),
            _batch((4.0,), prefix="fake-b"),
        ),
        generator=torch.Generator().manual_seed(29),
    )
    assert [call[0] for call in losses.calls] == ["student", "student", "fake", "fake"]
    assert [call[1][0].split("-")[0] for call in losses.calls] == [
        "student",
        "student",
        "fake",
        "fake",
    ]
    assert all(target >= 1 for target in result.student_target_indices)
    assert all(target >= 0 for target in result.fake_score_target_indices)
    assert losses.student_weights_seen_by_fake[0] != initial_student
    assert losses.student_modes_seen_by_fake == [True, True]
    assert engine.student_optimizer_steps == engine.fake_score_optimizer_steps == 1
    assert student_scheduler.steps == fake_scheduler.steps == 1
    assert not engine.fake_score_module.training


def test_engine_poison_blocks_checkpoint_after_student_commit() -> None:
    engine, _, _, _, _ = _engine(seed=31, accumulation=1, fail_fake_call=1)
    before = engine.student_module.weight.detach().clone()
    with pytest.raises(RuntimeError, match="intentional fake-score failure"):
        engine.train_step(
            _batch((1.0,), prefix="student"),
            _batch((2.0,), prefix="fake"),
        )
    assert not torch.equal(before, engine.student_module.weight.detach())
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return _batch((float(self.cursor),), prefix=f"cursor-{self.cursor}")

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack(seed: int):
    engine, _, student_scheduler, fake_scheduler, ema = _engine(
        seed=seed,
        accumulation=2,
        use_generator=True,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(211)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "teacher": engine.teacher_module,
            "fake_score": engine.fake_score_module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.fake_score_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
            identity={
                "algorithm": "sgmd",
                "gradient_accumulation_steps": engine.gradient_accumulation_steps,
            },
        lr_scheduler=NamedStatefulCollection(
            {"student": student_scheduler, "fake_score": fake_scheduler}
        ),
        ema=NamedStatefulCollection({"student": ema}),
    )
    return (
        engine,
        loader,
        progress,
        generator,
        model,
        state,
        student_scheduler,
        fake_scheduler,
        ema,
    )


def test_dcp_split_resume_restores_rng_optimizers_schedulers_ema_and_data_cursor(
    tmp_path: Path,
) -> None:
    baseline = _checkpointable_stack(41)
    engine, loader, progress, generator, model, state, student_scheduler, fake_scheduler, ema = baseline
    session = NativeSGMDTrainingSession(engine, loader, progress)
    session.run(max_steps=1, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1, generator=generator)
    expected_parameters = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    expected_ema = ema.shadow.clone()
    expected_schedulers = (student_scheduler.steps, fake_scheduler.steps)

    restored = _checkpointable_stack(47)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_fake_scheduler,
        restored_ema,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = NativeSGMDTrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1, generator=restored_generator)
    assert restored_loader.cursor == 8
    assert restored_progress.optimizer_steps == 2
    assert actual.final_student_loss == expected.final_student_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    assert (
        restored_student_scheduler.steps,
        restored_fake_scheduler.steps,
    ) == expected_schedulers
    torch.testing.assert_close(restored_ema.shadow, expected_ema)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])


def test_recipe_supports_arbitrary_accumulation_without_card_count_assumptions() -> None:
    student = _Adapter(0.2, "same-base")
    teacher = _Adapter(0.6, "same-base", frozen=True)
    fake = _Adapter(0.4, "same-base")
    stack = build_native_sgmd_training_stack(
        _recipe(accumulation=7),
        student=student,
        teacher=teacher,
        fake_score=fake,
        fused_adamw=False,
    )
    assert stack.engine.gradient_accumulation_steps == 7
