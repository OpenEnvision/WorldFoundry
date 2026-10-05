from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection  # noqa: E402
from worldfoundry.training.post_training import (  # noqa: E402
    DMD2Config,
    DMD2LossResult,
    DMD2TrainingBatch,
    NativeDMD2LossAdapter,
    NativeDMD2TrainEngine,
    NativeDMD2TrainingSession,
    build_native_dmd2_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd.objective import (  # noqa: E402
    FewStepSchedule,
)
from worldfoundry.training.recipes import (  # noqa: E402
    DMD2AlgorithmSpec,
    PostTrainingRecipe,
)


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class _GuidanceModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score_weight = torch.nn.Parameter(torch.tensor(0.7))
        self.discriminator_weight = torch.nn.Parameter(torch.tensor(0.4))


def _expand_levels(levels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return levels.reshape((levels.shape[0],) + (1,) * (reference.ndim - 1))


class _PredictionAdapter:
    noise_process_kind = "flow-matching"

    def __init__(self, module: _Scale, checkpoint_identity: str) -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity
        self.grad_enabled: list[bool] = []
        self.noises: list[torch.Tensor] = []
        self.predict_calls = 0

    def add_noise(self, clean_latents, noise, noise_levels):
        self.noises.append(noise.detach().clone())
        levels = _expand_levels(noise_levels, clean_latents)
        return clean_latents + levels * (noise - clean_latents)

    def predict_clean(
        self,
        noisy_latents,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del noise_levels, sample_ids, conditioning, training, branch
        self.predict_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return noisy_latents * self.module.weight


class _GuidanceAdapter:
    noise_process_kind = "flow-matching"

    def __init__(self, checkpoint_identity: str = "guidance-init") -> None:
        self.module = _GuidanceModule()
        self.checkpoint_identity = checkpoint_identity
        self.score_calls = 0
        self.denoising_calls = 0
        self.discriminator_calls = 0
        self.noises: list[torch.Tensor] = []
        self.discriminator_sample_ids: list[tuple[str, ...]] = []

    def add_noise(self, clean_latents, noise, noise_levels):
        self.noises.append(noise.detach().clone())
        levels = _expand_levels(noise_levels, clean_latents)
        return clean_latents + levels * (noise - clean_latents)

    def predict_clean(
        self,
        noisy_latents,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del noise_levels, sample_ids, conditioning, training, branch
        self.score_calls += 1
        return noisy_latents * self.module.score_weight

    def denoising_loss_per_sample(
        self,
        clean_latents,
        noisy_latents,
        noise,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del noise, noise_levels, sample_ids, conditioning, training
        self.denoising_calls += 1
        prediction = noisy_latents * self.module.score_weight
        return (prediction - clean_latents).float().square().reshape(clean_latents.shape[0], -1).mean(1)

    def discriminator_logits(
        self,
        latents,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del noise_levels, conditioning, training
        self.discriminator_calls += 1
        self.discriminator_sample_ids.append(sample_ids)
        return latents.float().reshape(latents.shape[0], -1).mean(1) * self.module.discriminator_weight


def _batch(values: list[float], *, prefix: str = "sample", weights=None) -> DMD2TrainingBatch:
    count = len(values)
    return DMD2TrainingBatch(
        sample_ids=tuple(f"{prefix}-generated-{index}" for index in range(count)),
        real_sample_ids=tuple(f"{prefix}-real-{index}" for index in range(count)),
        real_latents=torch.tensor(values, dtype=torch.float32).reshape(count, 1),
        conditioning={},
        unconditional_conditioning={},
        real_conditioning={},
        sample_weights=None if weights is None else torch.tensor(weights, dtype=torch.float32),
    )


def _config(**changes) -> DMD2Config:
    values = {
        "schedule": FewStepSchedule((1000.0,), (1.0,)),
        "normalization_axes": (1,),
        "distribution_matching_weight": 1.0,
        "generator_adversarial_weight": 1.0,
        "guidance_denoising_weight": 1.0,
        "guidance_adversarial_weight": 1.0,
    }
    values.update(changes)
    return DMD2Config(**values)


def _native_losses(config: DMD2Config):
    student = _PredictionAdapter(_Scale(0.8), "student-init")
    teacher_module = _Scale(0.6).requires_grad_(False)
    teacher = _PredictionAdapter(teacher_module, "teacher-init")
    guidance = _GuidanceAdapter()
    losses = NativeDMD2LossAdapter(student, teacher, guidance, config)
    return losses, student, teacher, guidance


def test_dmd2_few_step_prefix_is_no_grad_and_independently_renoised() -> None:
    from worldfoundry.training.post_training import simulate_dmd2_student

    student = _PredictionAdapter(_Scale(0.8), "student-init")
    schedule = FewStepSchedule((1000.0, 700.0, 300.0), (1.0, 0.7, 0.3))
    simulate_dmd2_student(
        student,
        _batch([1.0, 2.0]),
        schedule,
        target_index=2,
        generator=torch.Generator().manual_seed(7),
    )
    assert student.grad_enabled == [False, False, True]
    assert len(student.noises) == 2
    assert not torch.equal(student.noises[0], student.noises[1])


def test_dmd2_few_step_prefix_can_reuse_the_initial_noise() -> None:
    from worldfoundry.training.post_training import simulate_dmd2_student

    student = _PredictionAdapter(_Scale(0.8), "student-init")
    simulate_dmd2_student(
        student,
        _batch([1.0, 2.0]),
        FewStepSchedule((1000.0, 700.0, 300.0), (1.0, 0.7, 0.3)),
        target_index=2,
        generator=torch.Generator().manual_seed(7),
        rollout_noise_mode="shared-initial",
    )
    assert len(student.noises) == 2
    torch.testing.assert_close(student.noises[0], student.noises[1], rtol=0, atol=0)


class _FusedGuidanceAdapter(_GuidanceAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.fused_calls = 0

    def predict_clean_and_logits(
        self,
        noisy_latents,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del noise_levels, conditioning, training
        self.fused_calls += 1
        clean = noisy_latents * self.module.score_weight
        logits = noisy_latents.float().reshape(noisy_latents.shape[0], -1).mean(1) * self.module.discriminator_weight
        self.discriminator_sample_ids.append(sample_ids)
        return clean, logits

    def denoising_loss_from_clean_per_sample(
        self,
        clean_latents,
        predicted_clean,
        noise_levels,
        *,
        conditioning,
    ):
        del noise_levels, conditioning
        return (predicted_clean - clean_latents).float().square().flatten(1).mean(1)


def test_dmd2_fused_guidance_reuses_one_fake_score_forward() -> None:
    student = _PredictionAdapter(_Scale(0.8), "student-init")
    teacher = _PredictionAdapter(_Scale(0.6).requires_grad_(False), "teacher-init")
    guidance = _FusedGuidanceAdapter()
    losses = NativeDMD2LossAdapter(
        student,
        teacher,
        guidance,
        _config(shared_adversarial_score_input=True),
    )

    losses.generator_loss(_batch([1.0, 2.0]), generator=torch.Generator().manual_seed(5))
    assert guidance.fused_calls == 1
    assert guidance.score_calls == 0
    assert guidance.discriminator_calls == 0

    losses.guidance_loss(_batch([1.0, 2.0]), generator=torch.Generator().manual_seed(7))
    assert guidance.fused_calls == 2
    assert guidance.denoising_calls == 0
    assert guidance.discriminator_calls == 1


def test_dmd2_generator_adversarial_gradient_only_updates_student_graph() -> None:
    losses, student, teacher, guidance = _native_losses(_config())
    result = losses.generator_loss(_batch([1.0, 2.0]), generator=torch.Generator().manual_seed(5))
    result.loss.backward()

    assert student.module.weight.grad is not None
    assert bool(student.module.weight.grad.abs() > 0)
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in guidance.module.parameters())


def test_dmd2_zero_weights_skip_inactive_model_paths_and_real_gan_is_renoised() -> None:
    losses, _, teacher, guidance = _native_losses(
        _config(
            distribution_matching_weight=0.0,
            guidance_denoising_weight=0.0,
            diffusion_gan_max_sigma=0.5,
        )
    )
    generator = torch.Generator().manual_seed(11)
    losses.generator_loss(_batch([1.0, 2.0]), generator=generator)
    assert teacher.predict_calls == 0
    assert guidance.score_calls == 0
    assert guidance.discriminator_calls == 1

    guidance.noises.clear()
    losses.guidance_loss(_batch([1.0, 2.0]), generator=generator).loss.backward()
    assert guidance.denoising_calls == 0
    assert guidance.discriminator_calls == 3
    assert len(guidance.noises) == 2
    assert not torch.equal(guidance.noises[0], guidance.noises[1])
    assert guidance.discriminator_sample_ids[-1][0].endswith("real-0")

    dm_losses, _, dm_teacher, dm_guidance = _native_losses(
        _config(generator_adversarial_weight=0.0, guidance_adversarial_weight=0.0)
    )
    dm_losses.generator_loss(_batch([1.0]), generator=torch.Generator().manual_seed(3))
    assert dm_teacher.predict_calls == 2
    assert dm_guidance.score_calls == 1
    assert dm_guidance.discriminator_calls == 0


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
    def __init__(self, module: torch.nn.Module) -> None:
        self.shadow = module.weight.detach().clone()

    def update(self, module: torch.nn.Module) -> None:
        self.shadow.mul_(0.5).add_(module.weight.detach(), alpha=0.5)

    def state_dict(self):
        return {"shadow": self.shadow.clone()}

    def load_state_dict(self, state_dict) -> None:
        self.shadow.copy_(state_dict["shadow"])


class _EngineLosses:
    def __init__(self, student, guidance, *, use_generator=False, fail_guidance_call=None) -> None:
        self.student = student
        self.guidance = guidance
        self.use_generator = use_generator
        self.fail_guidance_call = fail_guidance_call
        self.guidance_calls = 0
        self.seen_student_weights: list[float] = []

    def loss_denominator(self, batch, *, role):
        del role
        if batch.sample_weights is None:
            return torch.tensor(float(batch.batch_size))
        return batch.sample_weights.sum()

    def _jitter(self, generator, device):
        if not self.use_generator:
            return torch.zeros((), device=device)
        return torch.rand((), generator=generator, device=device) * 0.05

    def _result(self, per_sample, batch):
        weights = torch.ones_like(per_sample) if batch.sample_weights is None else batch.sample_weights.to(per_sample)
        denominator = weights.sum()
        numerator = (per_sample * weights).sum()
        return DMD2LossResult(
            numerator / denominator,
            {"loss_numerator": numerator.detach(), "loss_denominator": denominator.detach()},
        )

    def generator_loss(self, batch, *, generator=None):
        values = batch.real_latents.float()
        prediction = self.student(values)
        target = values * (0.25 + self._jitter(generator, values.device))
        return self._result((prediction - target).square().reshape(values.shape[0], -1).mean(1), batch)

    def guidance_loss(self, batch, *, generator=None):
        self.guidance_calls += 1
        if self.guidance_calls == self.fail_guidance_call:
            raise RuntimeError("intentional guidance failure")
        self.seen_student_weights.append(float(self.student.weight.detach().item()))
        values = batch.real_latents.float()
        target = self.student(values).detach() + self._jitter(generator, values.device)
        prediction = self.guidance(values)
        return self._result((prediction - target).square().reshape(values.shape[0], -1).mean(1), batch)


def _engine(
    *,
    seed: int,
    accumulation_steps: int,
    interval: int = 2,
    cadence: str = "generator-update",
    use_generator: bool = False,
    fail_guidance_call=None,
    update_mode: str = "generator-then-guidance",
):
    torch.manual_seed(seed)
    student = torch.nn.Linear(1, 1, bias=False)
    guidance = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    losses = _EngineLosses(
        student,
        guidance,
        use_generator=use_generator,
        fail_guidance_call=fail_guidance_call,
    )
    student_scheduler = _Counter()
    guidance_scheduler = _Counter()
    ema = _EMA(student)
    engine = NativeDMD2TrainEngine(
        student_module=student,
        teacher_module=teacher,
        guidance_module=guidance,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        guidance_optimizer=torch.optim.SGD(guidance.parameters(), lr=0.05),
        generator_update_interval=interval,
        gradient_accumulation_steps=accumulation_steps,
        student_max_grad_norm=1000.0,
        guidance_max_grad_norm=1000.0,
        student_scheduler=student_scheduler,
        guidance_scheduler=guidance_scheduler,
        student_scheduler_cadence=cadence,
        update_mode=update_mode,
        student_ema=ema,
    )
    return engine, losses, student_scheduler, guidance_scheduler, ema


def test_dmd2_uneven_accumulation_matches_combined_batch_and_guidance_sees_post_g() -> None:
    accumulated, losses, student_scheduler, guidance_scheduler, _ = _engine(seed=13, accumulation_steps=2)
    combined, _, _, _, _ = _engine(seed=13, accumulation_steps=1)
    initial_student = float(accumulated.student_module.weight.detach())
    first = _batch([1.0], prefix="first", weights=[0.5])
    second = _batch([2.0, 3.0, 4.0], prefix="second", weights=[1.0, 2.0, 0.5])
    merged = _batch([1.0, 2.0, 3.0, 4.0], prefix="merged", weights=[0.5, 1.0, 2.0, 0.5])

    accumulated_result = accumulated.train_step((first, second))
    combined_result = combined.train_step(merged)

    torch.testing.assert_close(accumulated.student_module.weight, combined.student_module.weight)
    torch.testing.assert_close(accumulated.guidance_module.weight, combined.guidance_module.weight)
    torch.testing.assert_close(accumulated_result.generator_loss, combined_result.generator_loss)
    torch.testing.assert_close(accumulated_result.guidance_loss, combined_result.guidance_loss)
    assert losses.seen_student_weights[0] != initial_student
    assert all(value == losses.seen_student_weights[0] for value in losses.seen_student_weights)
    assert all(parameter.grad is None for parameter in accumulated.student_parameters)
    assert student_scheduler.steps == 1
    assert guidance_scheduler.steps == 1


def test_dmd2_cadence_and_poison_after_generator_commit() -> None:
    engine, _, student_scheduler, guidance_scheduler, _ = _engine(
        seed=17,
        accumulation_steps=1,
        interval=3,
        cadence="iteration",
    )
    due = [engine.train_step(_batch([1.0])).generator_updated for _ in range(7)]
    assert due == [True, False, False, True, False, False, True]
    assert engine.student_optimizer_steps == 3
    assert engine.guidance_optimizer_steps == 7
    assert student_scheduler.steps == 7
    assert guidance_scheduler.steps == 7

    poisoned, _, _, _, _ = _engine(
        seed=19,
        accumulation_steps=2,
        fail_guidance_call=2,
    )
    before = poisoned.student_module.weight.detach().clone()
    with pytest.raises(RuntimeError, match="intentional guidance failure"):
        poisoned.train_step((_batch([1.0]), _batch([2.0])))
    assert not torch.equal(poisoned.student_module.weight.detach(), before)
    with pytest.raises(RuntimeError, match="partially committed"):
        poisoned.state_dict()


def test_dmd2_alternating_cadence_updates_exactly_one_role() -> None:
    engine, _, student_scheduler, guidance_scheduler, _ = _engine(
        seed=31,
        accumulation_steps=1,
        interval=5,
        cadence="generator-update",
        update_mode="alternating",
    )
    results = [engine.train_step(_batch([1.0])) for _ in range(11)]
    assert [result.generator_updated for result in results] == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [result.guidance_updated for result in results] == [
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert student_scheduler.steps == 3
    assert guidance_scheduler.steps == 8


def _recipe_mapping() -> dict[str, object]:
    return {
        "run": {"id": "dmd2-test", "output_dir": "runs/dmd2-test"},
        "model": {"recipe": "toy-flow", "checkpoint": "student-init"},
        "tuning": {"mode": "lora", "preset": "toy", "rank": 2, "alpha": 2},
        "data": {"manifest": "data/latents.jsonl"},
        "algorithm": {
            "type": "dmd2",
            "student_timesteps": [1000.0, 500.0],
            "student_sigmas": [1.0, 0.5],
            "real_score_checkpoint": "teacher-init",
            "guidance_checkpoint": "guidance-init",
            "normalization_axes": [1],
        },
        "optimizer": {"type": "adamw", "learning_rate": 1.0e-4},
        "guidance_optimizer": {"type": "adamw", "learning_rate": 2.0e-4},
    }


def test_dmd2_recipe_is_strict_and_builder_checks_loaded_checkpoint_identities() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert isinstance(recipe.algorithm, DMD2AlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    assert recipe.guidance_optimizer is not None

    unknown = deepcopy(_recipe_mapping())
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)
    wrong_optimizer = deepcopy(_recipe_mapping())
    wrong_optimizer["fake_score_optimizer"] = {"type": "adamw", "learning_rate": 1.0e-4}
    with pytest.raises(ValueError, match="only accepts guidance_optimizer"):
        PostTrainingRecipe.from_mapping(wrong_optimizer)

    student = _PredictionAdapter(_Scale(0.8), "student-init")
    teacher = _PredictionAdapter(_Scale(0.6).requires_grad_(False), "wrong-teacher")
    guidance = _GuidanceAdapter()
    with pytest.raises(ValueError, match="loaded checkpoint identity"):
        build_native_dmd2_training_stack(
            recipe,
            student=student,
            real_score=teacher,
            guidance=guidance,
            fused_adamw=False,
        )
    teacher.checkpoint_identity = "teacher-init"
    stack = build_native_dmd2_training_stack(
        recipe,
        student=student,
        real_score=teacher,
        guidance=guidance,
        fused_adamw=False,
    )
    assert stack.student_optimizer.param_groups[0]["lr"] == 1.0e-4
    assert stack.guidance_optimizer.param_groups[0]["lr"] == 2.0e-4


def test_dmd2_runtime_is_flow_matching_only_not_a_ddpm_materializer() -> None:
    student = _PredictionAdapter(_Scale(0.8), "student-init")
    teacher = _PredictionAdapter(_Scale(0.6).requires_grad_(False), "teacher-init")
    guidance = _GuidanceAdapter()
    teacher.noise_process_kind = "ddpm"
    with pytest.raises(ValueError, match="flow-matching noise process"):
        NativeDMD2LossAdapter(student, teacher, guidance, _config())


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return _batch([float(self.cursor)], prefix=f"cursor-{self.cursor}")

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack(seed: int):
    engine, _, student_scheduler, guidance_scheduler, ema = _engine(
        seed=seed,
        accumulation_steps=2,
        use_generator=True,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    objective_generator = torch.Generator().manual_seed(101)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "teacher": engine.teacher_module,
            "guidance": engine.guidance_module,
        }
    )
    scheduler_state = NamedStatefulCollection({"student": student_scheduler, "guidance": guidance_scheduler})
    ema_state = NamedStatefulCollection({"student": ema})
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.guidance_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=objective_generator,
        progress=progress,
        identity={
            "algorithm": "dmd2",
            "gradient_accumulation_steps": engine.gradient_accumulation_steps,
        },
        lr_scheduler=scheduler_state,
        ema=ema_state,
    )
    return engine, loader, progress, objective_generator, model, state, student_scheduler, guidance_scheduler, ema


def test_dmd2_dcp_split_resume_restores_rng_roles_optimizers_schedulers_and_ema(tmp_path: Path) -> None:
    baseline = _checkpointable_stack(23)
    engine, loader, progress, generator, model, state, student_scheduler, guidance_scheduler, ema = baseline
    session = NativeDMD2TrainingSession(engine, loader, progress)
    session.run(max_steps=1, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)

    expected = session.run(max_steps=1, generator=generator)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = ema.shadow.clone()
    expected_scheduler_steps = (student_scheduler.steps, guidance_scheduler.steps)

    restored = _checkpointable_stack(29)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_guidance_scheduler,
        restored_ema,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = NativeDMD2TrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1, generator=restored_generator)

    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    assert actual.student_optimizer_steps == expected.student_optimizer_steps
    assert actual.guidance_optimizer_steps == expected.guidance_optimizer_steps
    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_guidance_loss == expected.final_guidance_loss
    assert (restored_student_scheduler.steps, restored_guidance_scheduler.steps) == expected_scheduler_steps
    torch.testing.assert_close(restored_ema.shadow, expected_ema, rtol=0, atol=0)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


def test_dmd2_immediate_checkpoint_resumes_into_critic_phase(tmp_path: Path) -> None:
    def stack(seed: int):
        engine, _, student_scheduler, guidance_scheduler, _ = _engine(
            seed=seed,
            accumulation_steps=1,
            interval=5,
            cadence="generator-update",
            use_generator=True,
            update_mode="alternating",
        )
        loader = _StatefulLoader()
        progress = TrainingProgress()
        generator = torch.Generator().manual_seed(211)
        model = torch.nn.ModuleDict({"student": engine.student_module, "guidance": engine.guidance_module})
        state = TrainingState(
            model=model,
            optimizer=(engine.student_optimizer, engine.guidance_optimizer),
            engine=engine,
            dataloader=loader,
            objective_generator=generator,
            progress=progress,
            identity={"algorithm": "dmd2", "update_mode": "alternating"},
            lr_scheduler=NamedStatefulCollection({"student": student_scheduler, "guidance": guidance_scheduler}),
        )
        return engine, loader, progress, generator, model, state, student_scheduler, guidance_scheduler

    baseline = stack(37)
    engine, loader, progress, generator, model, state, student_scheduler, guidance_scheduler = baseline
    session = NativeDMD2TrainingSession(engine, loader, progress)
    first = session.run(max_steps=1, generator=generator)
    assert first.student_optimizer_steps == 1
    assert first.guidance_optimizer_steps == 0
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1, generator=generator)
    expected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_schedulers = (student_scheduler.steps, guidance_scheduler.steps)

    restored = stack(41)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_guidance_scheduler,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = NativeDMD2TrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1, generator=restored_generator)

    assert actual.student_optimizer_steps == expected.student_optimizer_steps == 1
    assert actual.guidance_optimizer_steps == expected.guidance_optimizer_steps == 1
    assert restored_loader.cursor == loader.cursor == 2
    assert (restored_student_scheduler.steps, restored_guidance_scheduler.steps) == expected_schedulers
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)


def test_dmd2_public_exports_are_lazy_and_resolvable() -> None:
    import worldfoundry.training.post_training as public

    for name in (
        "DMD2Config",
        "DMD2TrainingBatch",
        "NativeDMD2TrainEngine",
        "NativeDMD2TrainingSession",
        "build_native_dmd2_training_stack",
    ):
        assert name in public.__all__
        assert getattr(public, name) is not None
