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
from worldfoundry.training.post_training.distillation.dfd import (  # noqa: E402
    DFDLossResult,
    DFDTrainingBatch,
    NativeDFDTrainEngine,
    NativeDFDTrainingSession,
    build_native_dfd_training_stack,
)
from worldfoundry.training.recipes import DFDAlgorithmSpec, PostTrainingRecipe  # noqa: E402


def _batch(value: float, *, count: int = 2, weights=None) -> DFDTrainingBatch:
    return DFDTrainingBatch(
        sample_ids=tuple(f"sample-{value}-{index}" for index in range(count)),
        real_latents=torch.full((count, 1), value, dtype=torch.float32),
        conditioning={},
        unconditional_conditioning={},
        sample_weights=None if weights is None else torch.tensor(weights, dtype=torch.float32),
    )


class _EngineLosses:
    data_forcing_probability = 1.0
    student_update_frequency = 5

    def __init__(self, student, fake_score, discriminator, *, use_generator=False) -> None:
        self.student = student
        self.fake_score = fake_score
        self.discriminator = discriminator
        self.use_generator = use_generator
        self.calls: list[tuple[str, bool | None, tuple[str, ...]]] = []
        self.student_modes_in_guidance: list[bool] = []

    def loss_denominator(self, batch, *, role):
        del role
        if batch.sample_weights is None:
            return torch.tensor(float(batch.batch_size))
        return batch.sample_weights.sum()

    def _weights(self, batch, reference):
        if batch.sample_weights is None:
            return torch.ones((batch.batch_size,), device=reference.device)
        return batch.sample_weights.to(reference)

    def _jitter(self, generator, reference):
        if not self.use_generator:
            return torch.zeros((), device=reference.device)
        return torch.rand((), device=reference.device, generator=generator) * 0.03

    def _result(self, per_sample, batch):
        weights = self._weights(batch, per_sample)
        numerator = (per_sample * weights).sum()
        denominator = weights.sum()
        return DFDLossResult(
            loss=numerator / denominator,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
            },
        )

    def student_loss(self, batch, *, data_forcing, generator=None):
        self.calls.append(("student", data_forcing, batch.sample_ids))
        values = batch.real_latents.float()
        prediction = self.student(values)
        target = values * (
            (0.2 if data_forcing else 0.4) + self._jitter(generator, values)
        )
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return self._result(per_sample, batch)

    def guidance_loss(self, batch, *, generator=None):
        self.calls.append(("guidance", None, batch.sample_ids))
        self.student_modes_in_guidance.append(self.student.training)
        values = batch.real_latents.float()
        target = self.student(values).detach() + self._jitter(generator, values)
        fake = self.fake_score(values)
        discriminator = self.discriminator(values)
        per_sample = (fake - target).square().flatten(1).mean(1)
        per_sample = per_sample + (discriminator - values * 0.1).square().flatten(1).mean(1)
        return self._result(per_sample, batch)


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


def _engine(
    seed: int,
    *,
    accumulation: int = 1,
    use_generator: bool = False,
    student_scheduler=None,
    fake_score_scheduler=None,
    discriminator_scheduler=None,
    ema=None,
):
    torch.manual_seed(seed)
    student = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    fake_score = torch.nn.Linear(1, 1, bias=False)
    discriminator = torch.nn.Linear(1, 1, bias=False)
    losses = _EngineLosses(
        student,
        fake_score,
        discriminator,
        use_generator=use_generator,
    )
    engine = NativeDFDTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake_score,
        discriminator_module=discriminator,
        loss_adapter=losses,
        student_optimizer=torch.optim.AdamW(student.parameters(), lr=0.02),
        fake_score_optimizer=torch.optim.AdamW(fake_score.parameters(), lr=0.02),
        discriminator_optimizer=torch.optim.AdamW(discriminator.parameters(), lr=0.02),
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        student_ema=ema,
    )
    return engine, losses


def test_engine_follows_one_to_four_cadence_and_consumes_one_batch_set_per_iteration() -> None:
    engine, losses = _engine(7, accumulation=2)
    phases = []
    for step in range(6):
        result = engine.train_step(
            (_batch(float(step + 1)), _batch(float(step + 2))),
            generator=torch.Generator().manual_seed(step),
        )
        phases.append(result.phase)
        if result.phase == "student":
            assert result.data_forcing_decisions == (True, True)
        else:
            assert result.data_forcing_decisions == ()
    assert phases == ["student", "guidance", "guidance", "guidance", "guidance", "student"]
    assert engine.student_optimizer_steps == 2
    assert engine.fake_score_optimizer_steps == 4
    assert engine.discriminator_optimizer_steps == 4
    assert all(losses.student_modes_in_guidance)
    assert len(losses.calls) == 12


def test_engine_restores_and_rejects_cadence_inconsistent_state() -> None:
    engine, _ = _engine(11)
    for index in range(3):
        engine.train_step(_batch(float(index + 1)))
    state = engine.state_dict()
    restored, _ = _engine(13)
    restored.load_state_dict(state)
    assert restored.global_step == 3
    assert restored.next_phase == "guidance"
    invalid = dict(state)
    invalid["student_optimizer_steps"] = 2
    with pytest.raises(ValueError, match="cadence"):
        restored.load_state_dict(invalid)


class _FailingCounter(_Counter):
    def step(self) -> None:
        super().step()
        raise RuntimeError("scheduler failure after optimizer commit")


def test_engine_poisoned_after_post_optimizer_failure() -> None:
    engine, _ = _engine(17, student_scheduler=_FailingCounter())
    with pytest.raises(RuntimeError, match="scheduler failure"):
        engine.train_step(_batch(1.0))
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="restore"):
        engine.train_step(_batch(2.0))


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


def _expand(levels, reference):
    return levels.to(reference).reshape((levels.shape[0],) + (1,) * (reference.ndim - 1))


class _PredictionAdapter:
    noise_process_kind = "flow-matching"

    def __init__(self, value: float, identity: str, *, frozen=False) -> None:
        self.module = _Scale(value)
        if frozen:
            self.module.requires_grad_(False)
        self.checkpoint_identity = identity

    def add_noise(self, clean_latents, noise, timesteps):
        levels = _expand(timesteps, clean_latents)
        return clean_latents + levels * (noise - clean_latents)

    def predict_clean(
        self,
        noisy_latents,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del timesteps, sample_ids, conditioning, training, branch
        return noisy_latents * self.module.weight


class _FakeAdapter(_PredictionAdapter):
    def denoising_loss_per_sample(
        self,
        clean_latents,
        noisy_latents,
        noise,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del noise, timesteps, sample_ids, conditioning, training
        return (
            (noisy_latents * self.module.weight - clean_latents)
            .float()
            .square()
            .flatten(1)
            .mean(1)
        )


class _DiscriminatorAdapter:
    def __init__(self) -> None:
        self.module = _Scale(0.3)
        self.checkpoint_identity = "disc"

    def discriminator_logits(
        self,
        noisy_latents,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del timesteps, sample_ids, conditioning, training
        return noisy_latents.float().flatten(1).mean(1) * self.module.weight


def _dfd_recipe(*, accumulation: int = 7) -> PostTrainingRecipe:
    optimizer = {
        "type": "adamw",
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-2,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "max_grad_norm": 10.0,
        "gradient_accumulation_steps": accumulation,
    }
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "dfd", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "dmd2-student",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "paired-real.jsonl", "shuffle_seed": 17},
            "algorithm": {
                "type": "dfd",
                "teacher_checkpoint": "teacher",
                "fake_score_checkpoint": "dmd2-fake",
                "discriminator_checkpoint": "disc",
            },
            "optimizer": optimizer,
            "fake_score_optimizer": dict(optimizer),
            "discriminator_optimizer": dict(optimizer),
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def test_builder_uses_released_optimizer_defaults_and_arbitrary_accumulation() -> None:
    student = _PredictionAdapter(0.8, "dmd2-student")
    teacher = _PredictionAdapter(0.6, "teacher", frozen=True)
    fake_score = _FakeAdapter(0.7, "dmd2-fake")
    discriminator = _DiscriminatorAdapter()
    recipe = _dfd_recipe()
    assert isinstance(recipe.algorithm, DFDAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    stack = build_native_dfd_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=fake_score,
        discriminator=discriminator,
        fused_adamw=False,
    )
    assert stack.student_optimizer.param_groups[0]["lr"] == 1.0e-5
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 1.0e-5
    assert stack.discriminator_optimizer is not None
    assert stack.discriminator_optimizer.param_groups[0]["lr"] == 1.0e-5
    assert stack.student_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert stack.engine.gradient_accumulation_steps == 7

    fake_score.module = student.module
    with pytest.raises(ValueError, match="independently materialized"):
        build_native_dfd_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            discriminator=discriminator,
            fused_adamw=False,
        )


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = float(self.cursor % 5 + 1)
        self.cursor += 1
        return _batch(value)

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack(seed: int):
    student_scheduler = _Counter()
    fake_score_scheduler = _Counter()
    discriminator_scheduler = _Counter()
    torch.manual_seed(seed)
    preview_student = torch.nn.Linear(1, 1, bias=False)
    ema = _EMA(preview_student)
    # Re-seed so the engine's student matches the EMA initialization exactly.
    torch.manual_seed(seed)
    engine, _ = _engine(
        seed,
        accumulation=2,
        use_generator=True,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        ema=ema,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(101)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "teacher": engine.teacher_module,
            "fake_score": engine.fake_score_module,
            "discriminator": engine.discriminator_module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(
            engine.student_optimizer,
            engine.fake_score_optimizer,
            engine.discriminator_optimizer,
        ),
        engine=engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
            identity={
                "algorithm": "dfd",
                "gradient_accumulation_steps": engine.gradient_accumulation_steps,
            },
        lr_scheduler=NamedStatefulCollection(
            {
                "student": student_scheduler,
                "fake_score": fake_score_scheduler,
                "discriminator": discriminator_scheduler,
            }
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
        fake_score_scheduler,
        discriminator_scheduler,
        ema,
    )


def test_dcp_split_resume_restores_cadence_rng_optimizers_schedulers_ema_and_cursor(
    tmp_path: Path,
) -> None:
    baseline = _checkpointable_stack(31)
    (
        engine,
        loader,
        progress,
        generator,
        model,
        state,
        student_scheduler,
        fake_score_scheduler,
        discriminator_scheduler,
        ema,
    ) = baseline
    session = NativeDFDTrainingSession(engine, loader, progress)
    session.run(max_steps=4, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=3, generator=generator)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_schedulers = (
        student_scheduler.steps,
        fake_score_scheduler.steps,
        discriminator_scheduler.steps,
    )
    expected_ema = ema.shadow.clone()

    restored = _checkpointable_stack(37)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_fake_score_scheduler,
        restored_discriminator_scheduler,
        restored_ema,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = NativeDFDTrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=3, generator=restored_generator)
    assert restored_loader.cursor == 14
    assert restored_progress.optimizer_steps == 7
    assert actual.final_phase == expected.final_phase
    assert actual.final_loss == expected.final_loss
    assert (
        restored_student_scheduler.steps,
        restored_fake_score_scheduler.steps,
        restored_discriminator_scheduler.steps,
    ) == expected_schedulers
    torch.testing.assert_close(restored_ema.shadow, expected_ema)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
