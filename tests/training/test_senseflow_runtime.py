from __future__ import annotations

import copy
import multiprocessing as multiprocessing_module
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer  # noqa: E402
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState  # noqa: E402
from worldfoundry.training.post_training.distillation.senseflow import (  # noqa: E402
    NativeSenseFlowLossAdapter,
    NativeSenseFlowTrainEngine,
    NativeSenseFlowTrainingSession,
    SenseFlowConfig,
    SenseFlowGeneratorPhase,
    SenseFlowLossResult,
    SenseFlowPreparedBatch,
    SenseFlowSchedule,
    SenseFlowTrainingBatch,
    build_native_senseflow_training_stack,
    simulate_senseflow_anchor,
)
from worldfoundry.training.recipes.post_training.algorithms.senseflow import (  # noqa: E402
    SenseFlowAlgorithmSpec,
    SenseFlowScheduleSpec,
)
from worldfoundry.training.recipes.post_training.common import plain_data  # noqa: E402
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe  # noqa: E402


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.weight


class _ToyVFMDiscriminator(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.feature_backbone = torch.nn.Linear(1, 1, bias=False)
        self.head = _Scale(value)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_backbone(values))


def _expand(levels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return levels.reshape((levels.shape[0],) + (1,) * (reference.ndim - 1))


class _PredictionAdapter:
    noise_process_kind = "flow-matching"

    def __init__(self, module: torch.nn.Module, identity: str) -> None:
        self.module = module
        self.checkpoint_identity = identity
        self.grad_enabled: list[bool] = []
        self.predictions: list[torch.Tensor] = []

    def add_noise(self, clean_latents, noise, noise_levels):
        levels = _expand(noise_levels, clean_latents)
        return (1.0 - levels) * clean_latents + levels * noise

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
        self.grad_enabled.append(torch.is_grad_enabled())
        prediction = self.module(noisy_latents)
        self.predictions.append(prediction.detach().clone())
        return prediction


class _TeacherAdapter(_PredictionAdapter):
    def predict_guided_clean(
        self,
        noisy_latents,
        noise_levels,
        *,
        sample_ids,
        conditioning,
        unconditional_conditioning,
        guidance_scale,
    ):
        del noise_levels, sample_ids, conditioning, unconditional_conditioning
        return self.module(noisy_latents) + float(guidance_scale) * 0.01


class _FakeScoreAdapter(_PredictionAdapter):
    def __init__(self, module: torch.nn.Module, identity: str) -> None:
        super().__init__(module, identity)
        self.denoising_inputs: list[torch.Tensor] = []
        self.weights_seen_before_loss: list[float] = []

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
        self.denoising_inputs.append(clean_latents.detach().clone())
        self.weights_seen_before_loss.append(float(self.module.weight.detach()))
        prediction = self.module(noisy_latents)
        return (prediction - clean_latents).float().square().flatten(1).mean(1)


class _DiscriminatorAdapter:
    def __init__(
        self,
        module: torch.nn.Module,
        identity: str = "discriminator",
        *,
        frozen_feature_modules: tuple[torch.nn.Module, ...] = (),
        trainable_head_modules: tuple[torch.nn.Module, ...] = (),
    ) -> None:
        self.module = module
        self.checkpoint_identity = identity
        self.frozen_feature_modules = frozen_feature_modules
        self.trainable_head_modules = trainable_head_modules
        self.fake_inputs: list[torch.Tensor] = []

    def logits(
        self,
        latents,
        *,
        sample_ids,
        conditioning,
        reference_latents,
        training,
    ):
        del sample_ids, conditioning, reference_latents
        if training and not latents.requires_grad:
            self.fake_inputs.append(latents.detach().clone())
        reduced = latents.float().flatten(1).mean(1, keepdim=True)
        return self.module(reduced)


def _batch(
    values: list[list[float]],
    *,
    prefix: str = "sample",
    weights: list[float] | None = None,
    device: torch.device | str = "cpu",
) -> SenseFlowTrainingBatch:
    count = len(values)
    return SenseFlowTrainingBatch(
        sample_ids=tuple(f"{prefix}-generated-{index}" for index in range(count)),
        real_sample_ids=tuple(f"{prefix}-real-{index}" for index in range(count)),
        real_latents=torch.tensor(values, dtype=torch.float32, device=device),
        conditioning={},
        unconditional_conditioning={},
        real_conditioning={},
        sample_weights=(
            None
            if weights is None
            else torch.tensor(weights, dtype=torch.float32, device=device)
        ),
    )


def _small_config(**changes) -> SenseFlowConfig:
    values = {
        "schedule": SenseFlowSchedule(
            timesteps=(10,),
            sigmas=(0.5,),
            isg_margin=1,
            num_train_timesteps=20,
        ),
        "generator_update_interval": 1,
        "backward_simulation_probability": 0.0,
        "ida_decay": 0.5,
        "isg_teacher_guidance": (2.0, 2.0),
        "dmd_teacher_guidance": (3.0, 3.0),
        "score_sampling": "uniform-schedule-index",
        "fake_score_sampling": "uniform-schedule-index",
        "score_flow_shift": 1.0,
        "distribution_matching_weight": 1.0,
        "generator_adversarial_weight": 0.2,
        "isg_weight": 1.0,
    }
    values.update(changes)
    return SenseFlowConfig(**values)


def _native_objective(config: SenseFlowConfig, *, device: torch.device | str = "cpu"):
    student = _PredictionAdapter(_Scale(0.8).to(device), "student")
    teacher_module = _Scale(0.55).to(device).requires_grad_(False)
    teacher = _TeacherAdapter(teacher_module, "teacher")
    fake = _FakeScoreAdapter(_Scale(0.4).to(device), "fake")
    discriminator = _DiscriminatorAdapter(_Scale(0.3).to(device))
    objective = NativeSenseFlowLossAdapter(
        student,
        teacher,
        fake,
        discriminator,
        config,
    )
    return objective, student, teacher, fake, discriminator


def _clear_gradients(*modules: torch.nn.Module) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.grad = None


def test_native_objective_isolates_generator_fake_score_and_discriminator_gradients() -> None:
    objective, student, teacher, fake, discriminator = _native_objective(_small_config())
    phase = objective.generator_phase(
        _batch([[1.0, 2.0], [2.0, 4.0]]),
        update=True,
        generator=torch.Generator().manual_seed(7),
    )
    assert phase.loss_result is not None
    phase.loss_result.loss.backward()
    assert student.module.weight.grad is not None
    assert bool(student.module.weight.grad.abs() > 0)
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in fake.module.parameters())
    assert all(parameter.grad is None for parameter in discriminator.module.parameters())

    _clear_gradients(student.module, teacher.module, fake.module, discriminator.module)
    fake_result = objective.fake_score_loss(
        phase.prepared,
        generator=torch.Generator().manual_seed(11),
    )
    fake_result.loss.backward()
    assert fake.module.weight.grad is not None
    assert all(parameter.grad is None for parameter in student.module.parameters())
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in discriminator.module.parameters())

    _clear_gradients(student.module, teacher.module, fake.module, discriminator.module)
    discriminator_result = objective.discriminator_loss(phase.prepared)
    discriminator_result.loss.backward()
    assert discriminator.module.weight.grad is not None
    assert all(parameter.grad is None for parameter in student.module.parameters())
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in fake.module.parameters())


def test_rollout_supports_batch_shared_forward_and_backward_anchor_simulation() -> None:
    schedule = SenseFlowSchedule(
        timesteps=(10, 6, 3),
        sigmas=(1.0, 0.6, 0.3),
        isg_margin=1,
        num_train_timesteps=10,
    )
    config = _small_config(schedule=schedule)
    batch = _batch([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    forward = _PredictionAdapter(_Scale(0.8), "student")
    forward_rollout = simulate_senseflow_anchor(
        forward,
        batch,
        config,
        generator=torch.Generator().manual_seed(13),
        training=True,
        anchor_index=2,
        backward_simulation=False,
    )
    assert forward_rollout.anchor_index == 2
    assert forward_rollout.anchor_sigmas.shape == (3,)
    assert forward.grad_enabled == [True]

    backward = _PredictionAdapter(_Scale(0.8), "student")
    backward_rollout = simulate_senseflow_anchor(
        backward,
        batch,
        config,
        generator=torch.Generator().manual_seed(13),
        training=True,
        anchor_index=2,
        backward_simulation=True,
    )
    assert backward_rollout.anchor_index == 2
    assert backward_rollout.backward_simulation
    assert backward.grad_enabled == [False, False, True]
    backward_rollout.generated_clean.sum().backward()
    assert backward.module.weight.grad is not None


class _DeterministicLosses:

    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        fake_score: torch.nn.Module,
        discriminator: torch.nn.Module,
        *,
        interval: int,
        ida_decay: float,
        use_rng: bool = False,
        fail_fake_call: int | None = None,
    ) -> None:
        self.student = SimpleNamespace(module=student)
        self.teacher = SimpleNamespace(module=teacher)
        self.fake_score = SimpleNamespace(module=fake_score)
        self.discriminator = SimpleNamespace(module=discriminator)
        self.generator_update_interval = interval
        self.ida_decay = ida_decay
        self.ida_enabled = ida_decay < 1.0
        self.use_rng = use_rng
        self.fail_fake_call = fail_fake_call
        self.fake_calls = 0
        self.prepared_values: list[torch.Tensor] = []
        self.fake_inputs: list[torch.Tensor] = []
        self.discriminator_inputs: list[torch.Tensor] = []
        self.fake_weights_seen: list[torch.Tensor] = []
        self.student_modes: list[bool] = []

    def loss_denominator(self, batch, *, role):
        del role
        if batch.sample_weights is None:
            return torch.tensor(float(batch.batch_size), device=batch.real_latents.device)
        return batch.sample_weights.sum()

    def _weights(self, batch) -> torch.Tensor:
        if batch.sample_weights is None:
            return torch.ones(batch.batch_size, device=batch.real_latents.device)
        return batch.sample_weights

    def _result(self, per_sample: torch.Tensor, batch) -> SenseFlowLossResult:
        weights = self._weights(batch).to(per_sample)
        numerator = (per_sample * weights).sum()
        denominator = weights.sum()
        return SenseFlowLossResult(
            loss=numerator / denominator,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
            },
        )

    def _jitter(self, reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.use_rng:
            return torch.zeros((), device=reference.device)
        return torch.rand((), device=reference.device, generator=generator) * 0.05

    def generator_phase(self, batch, *, update, generator) -> SenseFlowGeneratorPhase:
        values = batch.real_latents.float()
        self.student_modes.append(self.student.module.training)
        prediction = self.student.module(values)
        jitter = self._jitter(values, generator)
        prepared_value = prediction.detach() + jitter
        self.prepared_values.append(prepared_value.clone())
        prepared = SenseFlowPreparedBatch(
            batch=batch,
            generated_clean=prepared_value,
            anchor_sigmas=torch.full(
                (batch.batch_size,),
                0.5,
                device=values.device,
            ),
            anchor_index=0,
            anchor_timestep=10,
            backward_simulation=False,
        )
        if not update:
            return SenseFlowGeneratorPhase(prepared=prepared, loss_result=None)
        target = values * (0.25 + jitter)
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return SenseFlowGeneratorPhase(
            prepared=prepared,
            loss_result=self._result(per_sample, batch),
        )

    def fake_score_loss(self, prepared, *, generator) -> SenseFlowLossResult:
        self.fake_calls += 1
        generated = prepared.generated_clean
        jitter = self._jitter(generated, generator)
        if self.fake_calls == self.fail_fake_call:
            raise RuntimeError("intentional fake-score failure")
        self.fake_inputs.append(generated.clone())
        self.fake_weights_seen.append(
            next(self.fake_score.module.parameters()).detach().clone()
        )
        prediction = self.fake_score.module(generated)
        target = generated * (0.5 + jitter)
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return self._result(per_sample, prepared.batch)

    def discriminator_loss(self, prepared) -> SenseFlowLossResult:
        generated = prepared.generated_clean
        self.discriminator_inputs.append(generated.clone())
        prediction = self.discriminator.module(generated)
        target = prepared.batch.real_latents.float() * 0.1
        per_sample = (prediction - target).square().flatten(1).mean(1)
        return self._result(per_sample, prepared.batch)


def _deterministic_engine(
    *,
    seed: int,
    accumulation_steps: int = 1,
    interval: int = 1,
    ida_decay: float = 1.0,
    use_rng: bool = False,
    fail_fake_call: int | None = None,
    student_scheduler: object | None = None,
    device: torch.device | str = "cpu",
):
    torch.manual_seed(seed)
    student = torch.nn.Linear(2, 2, bias=False, device=device)
    fake = torch.nn.Linear(2, 2, bias=False, device=device)
    discriminator = torch.nn.Linear(2, 2, bias=False, device=device)
    teacher = torch.nn.Linear(2, 2, bias=False, device=device).requires_grad_(False)
    losses = _DeterministicLosses(
        student,
        teacher,
        fake,
        discriminator,
        interval=interval,
        ida_decay=ida_decay,
        use_rng=use_rng,
        fail_fake_call=fail_fake_call,
    )
    engine = NativeSenseFlowTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake,
        discriminator_module=discriminator,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
        discriminator_optimizer=torch.optim.SGD(discriminator.parameters(), lr=0.05),
        student_max_grad_norm=1000.0,
        fake_score_max_grad_norm=1000.0,
        discriminator_max_grad_norm=1000.0,
        gradient_accumulation_steps=accumulation_steps,
        seed=seed + 100,
        student_scheduler=student_scheduler,
    )
    return engine, losses


def test_engine_matches_official_ttur_and_applies_ida_before_fake_score_update() -> None:
    engine, losses = _deterministic_engine(
        seed=19,
        interval=5,
        ida_decay=0.5,
    )
    batch = _batch([[1.0, 2.0], [3.0, 4.0]])
    initial_student = engine.student_module.weight.detach().clone()
    for _ in range(4):
        result = engine.train_step(batch)
        assert not result.generator_updated
    torch.testing.assert_close(engine.student_module.weight, initial_student)
    assert engine.fake_score_optimizer_steps == 4
    assert engine.discriminator_optimizer_steps == 4
    assert engine.student_optimizer_steps == 0
    assert engine.ida_updates == 0
    assert losses.student_modes == [True, True, True, True]

    fake_before = engine.fake_score_module.weight.detach().clone()
    result = engine.train_step(batch)
    student_after = engine.student_module.weight.detach().clone()
    assert result.generator_updated
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 5
    assert engine.discriminator_optimizer_steps == 5
    assert engine.ida_updates == 1
    expected_before_fake_step = 0.5 * fake_before + 0.5 * student_after
    torch.testing.assert_close(
        losses.fake_weights_seen[-1],
        expected_before_fake_step,
    )
    torch.testing.assert_close(losses.fake_inputs[-1], losses.prepared_values[-1])
    torch.testing.assert_close(losses.discriminator_inputs[-1], losses.prepared_values[-1])


def test_uneven_accumulation_matches_one_combined_batch() -> None:
    accumulated, _ = _deterministic_engine(seed=29, accumulation_steps=2)
    combined, _ = _deterministic_engine(seed=29, accumulation_steps=1)
    first = _batch([[1.0, 2.0]], prefix="first", weights=[0.5])
    second = _batch(
        [[2.0, 3.0], [3.0, 5.0], [4.0, 7.0]],
        prefix="second",
        weights=[1.0, 2.0, 0.5],
    )
    merged = _batch(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]],
        prefix="merged",
        weights=[0.5, 1.0, 2.0, 0.5],
    )

    accumulated_result = accumulated.train_step((first, second))
    combined_result = combined.train_step(merged)
    torch.testing.assert_close(accumulated.student_module.weight, combined.student_module.weight)
    torch.testing.assert_close(accumulated.fake_score_module.weight, combined.fake_score_module.weight)
    torch.testing.assert_close(
        accumulated.discriminator_module.weight,
        combined.discriminator_module.weight,
    )
    torch.testing.assert_close(accumulated_result.generator_loss, combined_result.generator_loss)
    torch.testing.assert_close(accumulated_result.fake_score_loss, combined_result.fake_score_loss)
    torch.testing.assert_close(
        accumulated_result.discriminator_loss,
        combined_result.discriminator_loss,
    )
    assert accumulated_result.metrics["accumulated_microbatches"] == 2


def test_precommit_failure_restores_rng_and_postcommit_failure_poisons_engine() -> None:
    recoverable, _ = _deterministic_engine(
        seed=31,
        interval=2,
        use_rng=True,
        fail_fake_call=1,
    )
    rng_before = recoverable.generator.get_state().clone()
    with pytest.raises(RuntimeError, match="intentional fake-score failure"):
        recoverable.train_step(_batch([[1.0, 2.0]]))
    torch.testing.assert_close(recoverable.generator.get_state(), rng_before, rtol=0, atol=0)
    assert recoverable.state_dict()["global_step"] == 0

    poisoned, _ = _deterministic_engine(
        seed=37,
        interval=1,
        use_rng=True,
        fail_fake_call=1,
    )
    with pytest.raises(RuntimeError, match="intentional fake-score failure"):
        poisoned.train_step(_batch([[1.0, 2.0]]))
    with pytest.raises(RuntimeError, match="partially committed"):
        poisoned.state_dict()
    with pytest.raises(RuntimeError, match="restore the last checkpoint"):
        poisoned.train_step(_batch([[1.0, 2.0]]))


class _FailingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        raise RuntimeError("intentional scheduler failure")

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


def test_scheduler_mutation_marks_the_iteration_as_partially_committed() -> None:
    scheduler = _FailingScheduler()
    engine, _ = _deterministic_engine(
        seed=39,
        interval=2,
        student_scheduler=scheduler,
    )
    with pytest.raises(RuntimeError, match="intentional scheduler failure"):
        engine.train_step(_batch([[1.0, 2.0]]))
    assert scheduler.steps == 1
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()


def test_engine_state_is_strict_and_round_trips_through_torch_dcp(tmp_path: Path) -> None:
    import torch.distributed.checkpoint as dcp

    engine, _ = _deterministic_engine(seed=41, interval=2, use_rng=True)
    batch = _batch([[1.0, 2.0], [3.0, 4.0]])
    engine.train_step(batch)
    engine.train_step(batch)
    expected = copy.deepcopy(engine.state_dict())
    checkpoint_dir = tmp_path / "senseflow-dcp"
    dcp.save({"engine": engine}, checkpoint_id=checkpoint_dir)

    engine.train_step(batch)
    assert engine.global_step == 3
    dcp.load({"engine": engine}, checkpoint_id=checkpoint_dir)
    assert engine.global_step == 2
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 2
    assert engine.discriminator_optimizer_steps == 2
    torch.testing.assert_close(
        engine.generator.get_state(),
        expected["rng_state"],
        rtol=0,
        atol=0,
    )

    invalid = copy.deepcopy(expected)
    invalid["student_optimizer_steps"] = 0
    with pytest.raises(ValueError, match="TTUR cadence"):
        engine.load_state_dict(invalid)
    extra = copy.deepcopy(expected)
    extra["unused"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        engine.load_state_dict(extra)


def _native_recipe(*, accumulation_steps: int = 2) -> PostTrainingRecipe:
    algorithm = SenseFlowAlgorithmSpec(
        schedule=SenseFlowScheduleSpec(
            timesteps=(10,),
            sigmas=(0.5,),
            isg_margin=1,
            num_train_timesteps=20,
        ),
        teacher_checkpoint="teacher-checkpoint",
        fake_score_checkpoint="fake-checkpoint",
        discriminator_checkpoint="discriminator-checkpoint",
        generator_update_interval=2,
        backward_simulation_probability=0.0,
        ida_decay=0.5,
        isg_teacher_guidance=(2.0, 2.0),
        dmd_teacher_guidance=(3.0, 3.0),
        score_sampling="uniform-schedule-index",
        fake_score_sampling="uniform-schedule-index",
        score_flow_shift=1.0,
        generator_adversarial_weight=0.2,
        seed=271,
        lr_warmup_steps=4,
        lr_warmup_start_ratio=0.5,
    )
    optimizer = {
        "type": "adamw",
        "learning_rate": 1.0e-2,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "max_grad_norm": 1000.0,
        "gradient_accumulation_steps": accumulation_steps,
    }
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "senseflow-native", "output_dir": "/tmp/senseflow"},
            "model": {"recipe": "toy-flow", "checkpoint": "student-checkpoint"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "/tmp/manifest.json"},
            "algorithm": plain_data(algorithm),
            "optimizer": optimizer,
            "fake_score_optimizer": optimizer,
            "discriminator_optimizer": optimizer,
            "export": {"format": "safetensors"},
        }
    )


def _native_stack(seed: int, *, accumulation_steps: int = 2):
    torch.manual_seed(seed)
    student = _PredictionAdapter(_Scale(0.8), "student-checkpoint")
    teacher = _TeacherAdapter(_Scale(0.55), "teacher-checkpoint")
    fake = _FakeScoreAdapter(_Scale(0.4), "fake-checkpoint")
    discriminator_module = _ToyVFMDiscriminator(0.3)
    discriminator = _DiscriminatorAdapter(
        discriminator_module,
        "discriminator-checkpoint",
        frozen_feature_modules=(discriminator_module.feature_backbone,),
        trainable_head_modules=(discriminator_module.head,),
    )
    stack = build_native_senseflow_training_stack(
        _native_recipe(accumulation_steps=accumulation_steps),
        student=student,
        teacher=teacher,
        fake_score=fake,
        discriminator=discriminator,
        fused_adamw=False,
    )
    return stack, teacher


def test_builder_freezes_teacher_and_session_consumes_released_scheduler_cadence() -> None:
    stack, teacher = _native_stack(71)
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not teacher.module.training
    feature_backbone = stack.model["discriminator"].feature_backbone
    discriminator_head = stack.model["discriminator"].head
    assert not any(parameter.requires_grad for parameter in feature_backbone.parameters())
    assert not feature_backbone.training
    assert {
        id(parameter)
        for group in stack.discriminator_optimizer.param_groups
        for parameter in group["params"]
    } == {id(parameter) for parameter in discriminator_head.parameters()}
    assert len(stack.optimizers) == 3
    assert stack.scheduler_state.component_names == (
        "discriminator",
        "fake-score",
        "student",
    )
    for optimizer in stack.optimizers:
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-3)

    progress = TrainingProgress()
    events: list[dict[str, object]] = []
    batches = [
        _batch([[1.0, 2.0]], prefix="first"),
        _batch([[2.0, 3.0]], prefix="second"),
    ]
    summary = NativeSenseFlowTrainingSession(
        stack.engine,
        batches,
        progress,
        event_sink=lambda event: events.append(dict(event)),
    ).run(max_steps=1)

    assert summary.final_step == 1
    assert summary.student_optimizer_steps == 0
    assert summary.fake_score_optimizer_steps == 1
    assert summary.discriminator_optimizer_steps == 1
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 2
    assert events[0]["generator_updated"] is False
    assert not feature_backbone.training
    assert all(parameter.grad is None for parameter in feature_backbone.parameters())
    assert stack.student_scheduler.step_count == 1
    assert stack.fake_score_scheduler.step_count == 1
    assert stack.discriminator_scheduler.step_count == 1
    for optimizer in stack.optimizers:
        assert optimizer.param_groups[0]["lr"] == pytest.approx(6.25e-3)


def test_builder_rejects_loaded_checkpoint_identity_mismatch_before_training() -> None:
    torch.manual_seed(73)
    student = _PredictionAdapter(_Scale(0.8), "wrong-student")
    teacher = _TeacherAdapter(_Scale(0.55), "teacher-checkpoint")
    fake = _FakeScoreAdapter(_Scale(0.4), "fake-checkpoint")
    discriminator_module = _ToyVFMDiscriminator(0.3)
    discriminator = _DiscriminatorAdapter(
        discriminator_module,
        "discriminator-checkpoint",
        frozen_feature_modules=(discriminator_module.feature_backbone,),
        trainable_head_modules=(discriminator_module.head,),
    )
    with pytest.raises(ValueError, match="differs"):
        build_native_senseflow_training_stack(
            _native_recipe(),
            student=student,
            teacher=teacher,
            fake_score=fake,
            discriminator=discriminator,
            fused_adamw=False,
        )
    assert any(parameter.requires_grad for parameter in teacher.module.parameters())


def test_builder_consumes_unified_recipe_without_optimizer_or_checkpoint_overrides() -> None:
    algorithm = SenseFlowAlgorithmSpec(
        schedule=SenseFlowScheduleSpec(
            timesteps=(10,),
            sigmas=(0.5,),
            isg_margin=1,
            num_train_timesteps=20,
        ),
        teacher_checkpoint="teacher-checkpoint",
        fake_score_checkpoint="fake-checkpoint",
        discriminator_checkpoint="discriminator-checkpoint",
        generator_update_interval=2,
        backward_simulation_probability=0.0,
        ida_decay=0.5,
        isg_teacher_guidance=(2.0, 2.0),
        dmd_teacher_guidance=(3.0, 3.0),
        score_sampling="uniform-schedule-index",
        fake_score_sampling="uniform-schedule-index",
        score_flow_shift=1.0,
        generator_adversarial_weight=0.2,
        seed=281,
        lr_warmup_steps=4,
        lr_warmup_start_ratio=0.5,
    )
    optimizer = {
        "type": "adamw",
        "learning_rate": 1.0e-2,
        "weight_decay": 0.01,
        "betas": [0.8, 0.9],
        "epsilon": 1.0e-7,
        "max_grad_norm": 1000.0,
        "gradient_accumulation_steps": 2,
    }
    recipe = PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "senseflow-runtime", "output_dir": "/tmp/senseflow"},
            "model": {"recipe": "toy-flow", "checkpoint": "student-checkpoint"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "/tmp/manifest.json"},
            "algorithm": plain_data(algorithm),
            "optimizer": optimizer,
            "fake_score_optimizer": {
                **optimizer,
                "learning_rate": 2.0e-2,
                "max_grad_norm": 900.0,
            },
            "discriminator_optimizer": {
                **optimizer,
                "learning_rate": 3.0e-2,
                "max_grad_norm": 800.0,
            },
            "export": {"format": "safetensors"},
        }
    )
    torch.manual_seed(75)
    student = _PredictionAdapter(_Scale(0.8), "student-checkpoint")
    teacher = _TeacherAdapter(_Scale(0.55), "teacher-checkpoint")
    fake = _FakeScoreAdapter(_Scale(0.4), "fake-checkpoint")
    discriminator_module = _ToyVFMDiscriminator(0.3)
    discriminator = _DiscriminatorAdapter(
        discriminator_module,
        "discriminator-checkpoint",
        frozen_feature_modules=(discriminator_module.feature_backbone,),
        trainable_head_modules=(discriminator_module.head,),
    )
    stack = build_native_senseflow_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=fake,
        discriminator=discriminator,
        fused_adamw=False,
    )

    assert stack.recipe is recipe
    assert stack.config.generator_update_interval == 2
    assert stack.optimizer_config.student_learning_rate == pytest.approx(1.0e-2)
    assert stack.optimizer_config.fake_score_learning_rate == pytest.approx(2.0e-2)
    assert stack.optimizer_config.discriminator_learning_rate == pytest.approx(3.0e-2)
    assert stack.optimizer_config.student_max_grad_norm == pytest.approx(1000.0)
    assert stack.optimizer_config.fake_score_max_grad_norm == pytest.approx(900.0)
    assert stack.optimizer_config.discriminator_max_grad_norm == pytest.approx(800.0)
    assert stack.optimizer_config.gradient_accumulation_steps == 2
    assert stack.student_scheduler.start_ratio == pytest.approx(0.5)
    progress = TrainingProgress()
    result = stack.create_session(
        [
            _batch([[1.0, 2.0]], prefix="recipe-first"),
            _batch([[2.0, 3.0]], prefix="recipe-second"),
        ],
        progress,
    ).run(max_steps=1)
    assert result.final_step == 1
    assert result.student_optimizer_steps == 0
    assert result.fake_score_optimizer_steps == 1
    assert result.discriminator_optimizer_steps == 1


class _SenseFlowStatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        value = float(self.cursor)
        return _batch(
            [[value, value + 1.0]],
            prefix=f"cursor-{self.cursor}",
        )

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_native_stack(seed: int):
    stack, _ = _native_stack(seed)
    loader = _SenseFlowStatefulLoader()
    progress = TrainingProgress()
    state = TrainingState(
        model=stack.model,
        optimizer=stack.optimizers,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=stack.engine.generator,
        progress=progress,
        identity={
            "algorithm": "senseflow",
            "gradient_accumulation_steps": stack.engine.gradient_accumulation_steps,
        },
        lr_scheduler=stack.scheduler_state,
    )
    return stack, loader, progress, state


def test_full_dcp_resume_restores_all_roles_optimizers_schedulers_loader_and_rng(
    tmp_path: Path,
) -> None:
    stack, loader, progress, state = _checkpointable_native_stack(79)
    session = NativeSenseFlowTrainingSession(stack.engine, loader, progress)
    session.run(max_steps=2)
    manager = TrainingCheckpointer(tmp_path / "senseflow-checkpoints")
    artifact = manager.save(state)

    expected = session.run(max_steps=2)
    expected_parameters = {
        name: value.detach().clone()
        for name, value in stack.model.state_dict().items()
    }
    expected_optimizer_steps = tuple(
        int(next(iter(optimizer.state.values()))["step"].item())
        for optimizer in stack.optimizers
    )
    expected_scheduler_steps = (
        stack.student_scheduler.step_count,
        stack.fake_score_scheduler.step_count,
        stack.discriminator_scheduler.step_count,
    )
    expected_rng = stack.engine.generator.get_state().clone()

    restored_stack, restored_loader, restored_progress, restored_state = (
        _checkpointable_native_stack(83)
    )
    manager.load(restored_state, artifact.path)
    actual = NativeSenseFlowTrainingSession(
        restored_stack.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=2)

    assert restored_loader.cursor == 8
    assert restored_progress.optimizer_steps == 4
    assert actual == expected
    for name, value in restored_stack.model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
    actual_optimizer_steps = tuple(
        int(next(iter(optimizer.state.values()))["step"].item())
        for optimizer in restored_stack.optimizers
    )
    assert actual_optimizer_steps == expected_optimizer_steps
    assert (
        restored_stack.student_scheduler.step_count,
        restored_stack.fake_score_scheduler.step_count,
        restored_stack.discriminator_scheduler.step_count,
    ) == expected_scheduler_steps
    torch.testing.assert_close(
        restored_stack.engine.generator.get_state(),
        expected_rng,
        rtol=0,
        atol=0,
    )


def _distributed_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    queue: multiprocessing_module.Queue,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(53)
        student = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2, bias=False))
        fake = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2, bias=False))
        discriminator = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(2, 2, bias=False)
        )
        teacher = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
        losses = _DeterministicLosses(
            student,
            teacher,
            fake,
            discriminator,
            interval=1,
            ida_decay=1.0,
        )
        engine = NativeSenseFlowTrainEngine(
            student_module=student,
            teacher_module=teacher,
            fake_score_module=fake,
            discriminator_module=discriminator,
            loss_adapter=losses,
            student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
            fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
            discriminator_optimizer=torch.optim.SGD(discriminator.parameters(), lr=0.05),
            student_max_grad_norm=1000.0,
            fake_score_max_grad_norm=1000.0,
            discriminator_max_grad_norm=1000.0,
            seed=153,
        )
        local_probe = torch.rand((), generator=engine.generator)
        probes = [torch.empty_like(local_probe) for _ in range(world_size)]
        dist.all_gather(probes, local_probe)
        assert len({float(value) for value in probes}) == world_size
        batch = (
            _batch([[1.0, 2.0]], prefix="rank-zero", weights=[0.5])
            if rank == 0
            else _batch(
                [[2.0, 3.0], [3.0, 5.0]],
                prefix="rank-one",
                weights=[1.0, 2.0],
            )
        )
        engine.train_step(batch)
        parameters = tuple(
            parameter.detach().clone()
            for module in (student, fake, discriminator)
            for parameter in module.parameters()
        )
        for parameter in parameters:
            peers = [torch.empty_like(parameter) for _ in range(world_size)]
            dist.all_gather(peers, parameter)
            for peer in peers[1:]:
                torch.testing.assert_close(peer, peers[0], rtol=0, atol=0)
        if rank == 0:
            queue.put(tuple(parameter.cpu().tolist() for parameter in parameters))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_rank_gloo_execution_is_synchronized_without_fixed_card_count(tmp_path: Path) -> None:
    context = multiprocessing_module.get_context("spawn")
    queue = context.Queue()
    rendezvous = str(tmp_path / "senseflow-gloo")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, 2, rendezvous, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        assert process.exitcode == 0
    assert queue.get(timeout=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_senseflow_objective_and_engine_run_a_cuda_optimizer_step() -> None:
    device = torch.device("cuda")
    objective, student, teacher, fake, discriminator = _native_objective(
        _small_config(),
        device=device,
    )
    engine = NativeSenseFlowTrainEngine(
        student_module=student.module,
        teacher_module=teacher.module,
        fake_score_module=fake.module,
        discriminator_module=discriminator.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=1.0e-3),
        fake_score_optimizer=torch.optim.SGD(fake.module.parameters(), lr=1.0e-3),
        discriminator_optimizer=torch.optim.SGD(discriminator.module.parameters(), lr=1.0e-3),
        student_max_grad_norm=1000.0,
        fake_score_max_grad_norm=1000.0,
        discriminator_max_grad_norm=1000.0,
        seed=61,
    )
    before = student.module.weight.detach().clone()
    result = engine.train_step(_batch([[1.0, 2.0], [2.0, 4.0]], device=device))
    torch.cuda.synchronize(device)
    assert result.generator_updated
    assert result.generator_loss.is_cuda
    assert result.fake_score_loss.is_cuda
    assert result.discriminator_loss.is_cuda
    assert not torch.equal(student.module.weight.detach(), before)
