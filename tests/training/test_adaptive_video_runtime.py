from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api.contracts import (  # noqa: E402
    PreparedBatch,
    TrainingBatch,
)
from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.objectives.flow_matching import (  # noqa: E402
    flow_clean_from_velocity,
)
from worldfoundry.training.post_training.distillation.adaptive_video import (  # noqa: E402
    AdaptiveVideoConfig,
    AdaptiveVideoTrainingBatch,
    FlowAdaptiveVideoLossAdapter,
    NativeAdaptiveVideoDataLoader,
    NativeAdaptiveVideoTrainEngine,
    NativeAdaptiveVideoTrainingSession,
    NativeAdaptiveVideoTrainingStack,
    build_native_adaptive_video_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDConfig,
    DMDTrainingBatch,
    FewStepSchedule,
)
from worldfoundry.training.recipes import (  # noqa: E402
    AdaptiveVideoAlgorithmSpec,
    PostTrainingRecipe,
)


class _ScaleFlow:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool,
        checkpoint_identity: str = "student-checkpoint",
    ) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(value)
        self.module.requires_grad_(trainable)
        self.checkpoint_identity = checkpoint_identity

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
        del sigmas, conditioning, training
        assert len(sample_ids) == noisy_latents.shape[0]
        sign = -1.0 if branch == "negative" else 1.0
        return noisy_latents * self.module.weight.reshape(()) * sign

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
        return flow_clean_from_velocity(noisy_latents, velocity, sigmas)


class _StatefulSequence:
    def __init__(self, values) -> None:
        self.values = tuple(values)
        self.cursor = 0

    def __iter__(self):
        while self.cursor < len(self.values):
            value = self.values[self.cursor]
            self.cursor += 1
            yield value

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


class _RealAdapter:
    prediction_type = "flow_velocity"
    trainable_module = torch.nn.Identity()
    lora_target_preset = None
    fsdp_block_classes = ()

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.conditions["latents"],
            conditioning={"prompt": batch.prompts},
        )

    def forward_train(self, batch):
        raise AssertionError("real-data preparation must not call forward_train")


def _config(*, temporal_cutoff: float = 0.8) -> AdaptiveVideoConfig:
    return AdaptiveVideoConfig(
        dmd=DMDConfig(
            schedule=FewStepSchedule((1000.0, 500.0), (1.0, 0.5)),
            score_flow_shift=5.0,
            teacher_guidance_scale=5.0,
            shared_score_timestep=False,
            per_sample_normalization=True,
        ),
        regression_ema_decay=0.95,
        regression_sensitivity=3.0,
        regression_loss_weight=1.0,
        temporal_regularization_weight=0.05,
        temporal_loss_cutoff=temporal_cutoff,
    )


def _batch(prefix: str, value: float = 0.25) -> AdaptiveVideoTrainingBatch:
    template = torch.full((1, 2, 1), value)
    return AdaptiveVideoTrainingBatch(
        sample_ids=(f"{prefix}-generated",),
        clean_latents=template,
        conditioning={"prompt": prefix},
        unconditional_conditioning={"prompt": ""},
        real_sample_ids=(f"{prefix}-real",),
        real_latents=torch.tensor([[[0.1], [0.7]]]),
        real_conditioning={"prompt": f"real-{prefix}"},
    )


def _generated_batch(prefix: str, value: float = 0.25) -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=(f"{prefix}-generated",),
        clean_latents=torch.full((1, 2, 1), value),
        conditioning={"prompt": prefix},
        unconditional_conditioning={"prompt": ""},
    )


def _engine(*, config: AdaptiveVideoConfig | None = None, interval: int = 2):
    student = _ScaleFlow(0.2, trainable=True)
    teacher = _ScaleFlow(
        0.4,
        trainable=False,
        checkpoint_identity="teacher-checkpoint",
    )
    fake = _ScaleFlow(
        -0.1,
        trainable=True,
        checkpoint_identity="fake-score-checkpoint",
    )
    resolved = config or _config()
    losses = FlowAdaptiveVideoLossAdapter(
        student,
        teacher,
        fake,
        resolved,
    )
    engine = NativeAdaptiveVideoTrainEngine(
        student_module=student.module,
        real_score_module=teacher.module,
        fake_score_module=fake.module,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=0.01),
        fake_score_optimizer=torch.optim.SGD(fake.module.parameters(), lr=0.01),
        generator_update_interval=interval,
        student_max_grad_norm=10.0,
        fake_score_max_grad_norm=10.0,
    )
    return engine, losses


def _recipe_mapping() -> dict[str, object]:
    return {
        "run": {"id": "adaptive-video-test", "output_dir": "unused"},
        "model": {
            "recipe": "wan2.1-t2v-1.3b",
            "checkpoint": "student-checkpoint",
        },
        "tuning": {"mode": "full"},
        "data": {"manifest": "generated-prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "adaptive-video-distillation",
            "student_timesteps": [1000, 900, 757, 522],
            "student_sigmas": [1.0, 0.9, 0.757, 0.522],
            "real_score_checkpoint": "teacher-checkpoint",
            "fake_score_checkpoint": "fake-score-checkpoint",
            "score_flow_shift": 5.0,
            "teacher_guidance_scale": 5.0,
            "generator_update_interval": 5,
            "regression_ema_decay": 0.95,
            "regression_sensitivity": 3.0,
            "regression_loss_weight": 1.0,
            "temporal_regularization_weight": 0.05,
            "temporal_loss_cutoff": 0.8,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 2.0e-6,
            "weight_decay": 0.0,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 2.0e-6,
            "weight_decay": 0.0,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def test_adaptive_video_combines_three_generator_losses_and_commits_state() -> None:
    engine, losses = _engine()
    generator = torch.Generator().manual_seed(11)
    before = engine.student_module.weight.detach().clone()

    result = engine.train_step(_batch("first"), generator=generator)

    assert result.generator_updated is True
    assert not torch.equal(engine.student_module.weight.detach(), before)
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 1
    assert losses.regression_ema.update_counts.sum().item() == 1
    metrics = result.metrics["student"]
    assert set(
        (
            "dmd_loss",
            "regression_loss",
            "temporal_raw_loss",
            "temporal_applied_loss",
        )
    ).issubset(metrics)
    assert torch.isfinite(result.generator_loss)
    assert torch.isfinite(result.fake_score_loss)

    state_after_generator = losses.regression_ema.state_dict()
    skipped = engine.train_step(_batch("second"), generator=generator)
    assert skipped.generator_updated is False
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 2
    assert torch.equal(
        losses.regression_ema.update_counts,
        state_after_generator["update_counts"],
    )


def test_adaptive_video_engine_restores_counters_and_regression_ema() -> None:
    engine, losses = _engine()
    engine.train_step(_batch("save"), generator=torch.Generator().manual_seed(7))
    saved = engine.state_dict()

    restored, restored_losses = _engine()
    restored.load_state_dict(saved)

    assert restored.global_step == engine.global_step
    assert restored.student_optimizer_steps == engine.student_optimizer_steps
    assert restored.fake_score_optimizer_steps == engine.fake_score_optimizer_steps
    torch.testing.assert_close(
        restored_losses.regression_ema.values,
        losses.regression_ema.values,
    )
    assert torch.equal(
        restored_losses.regression_ema.update_counts,
        losses.regression_ema.update_counts,
    )


def test_adaptive_video_batch_requires_fresh_real_video_geometry() -> None:
    with pytest.raises(ValueError, match="must match"):
        AdaptiveVideoTrainingBatch(
            sample_ids=("generated",),
            clean_latents=torch.zeros(1, 2, 1),
            conditioning={},
            unconditional_conditioning={},
            real_sample_ids=("real",),
            real_latents=torch.zeros(1, 3, 1),
            real_conditioning={},
        )


def test_adaptive_video_state_has_only_behavioral_fields() -> None:
    engine, _ = _engine()
    state = engine.state_dict()
    assert set(state) == {
        "schema",
        "global_step",
        "dmd_engine",
        "objective",
    }
    assert not any(
        name in str(state).lower()
        for name in ("provenance", "source_revision", "paper_url", "repository")
    )


def test_adaptive_video_recipe_builder_consumes_algorithm_and_optimizer_fields() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert isinstance(recipe.algorithm, AdaptiveVideoAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    student = _ScaleFlow(0.2, trainable=True)
    teacher = _ScaleFlow(
        0.4,
        trainable=False,
        checkpoint_identity="teacher-checkpoint",
    )
    fake = _ScaleFlow(
        -0.1,
        trainable=True,
        checkpoint_identity="fake-score-checkpoint",
    )

    stack = build_native_adaptive_video_training_stack(
        recipe,
        student=student,
        real_score=teacher,
        fake_score=fake,
        fused_adamw=False,
    )

    assert isinstance(stack, NativeAdaptiveVideoTrainingStack)
    assert stack.config.dmd.schedule.timesteps == (1000.0, 900.0, 757.0, 522.0)
    assert stack.config.dmd.schedule.sigmas == (1.0, 0.9, 0.757, 0.522)
    assert stack.config.dmd.shared_score_timestep is False
    assert stack.config.dmd.per_sample_normalization is True
    assert stack.config.regression_ema_decay == 0.95
    assert stack.config.regression_sensitivity == 3.0
    assert stack.config.temporal_regularization_weight == 0.05
    assert stack.engine.generator_update_interval == 5
    assert stack.engine.student_max_grad_norm == 10.0
    assert stack.engine.fake_score_max_grad_norm == 10.0
    assert stack.student_optimizer.param_groups[0]["lr"] == 2.0e-6
    assert stack.student_optimizer.param_groups[0]["betas"] == (0.0, 0.999)
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 2.0e-6

    teacher.checkpoint_identity = "different-teacher"
    with pytest.raises(ValueError, match="differs from recipe"):
        build_native_adaptive_video_training_stack(
            recipe,
            student=student,
            real_score=teacher,
            fake_score=fake,
            fused_adamw=False,
        )


def test_adaptive_video_session_advances_real_stream_only_on_generator_steps() -> None:
    engine, _ = _engine(interval=2)
    generated_source = _StatefulSequence(
        [_generated_batch(f"generated-{index}") for index in range(3)]
    )
    real_source = _StatefulSequence(
        [
            TrainingBatch(
                sample_ids=(f"real-{index}",),
                prompts=(f"real prompt {index}",),
                conditions={
                    "latents": torch.tensor([[[0.1], [0.6]]]) + index * 0.01
                },
            )
            for index in range(2)
        ]
    )
    loader = NativeAdaptiveVideoDataLoader(
        generated_source,
        real_source,
        _RealAdapter(),
    )
    events = []
    progress = TrainingProgress()
    session = NativeAdaptiveVideoTrainingSession(
        engine,
        loader,
        progress,
        event_sink=events.append,
    )

    summary = session.run(max_steps=3, generator=torch.Generator().manual_seed(23))

    assert summary.final_step == 3
    assert generated_source.cursor == 3
    assert real_source.cursor == 2
    assert progress.microbatches_seen == 5
    assert progress.samples_seen == 5
    assert [event["real_microbatches"] for event in events] == [1, 0, 1]
    saved = loader.state_dict()
    assert saved["generated_source"] == {"cursor": 3}
    assert saved["real_source"] == {"cursor": 2}


def _checkpointable_adaptive_video_stack():
    engine, losses = _engine(interval=2)
    generated_source = _StatefulSequence(
        [_generated_batch(f"generated-{index}") for index in range(4)]
    )
    real_source = _StatefulSequence(
        [
            TrainingBatch(
                sample_ids=(f"real-{index}",),
                prompts=(f"real prompt {index}",),
                conditions={
                    "latents": torch.tensor([[[0.1], [0.6]]]) + index * 0.01
                },
            )
            for index in range(3)
        ]
    )
    loader = NativeAdaptiveVideoDataLoader(
        generated_source,
        real_source,
        _RealAdapter(),
    )
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
        identity={"algorithm": "adaptive-video-distillation"},
    )
    return engine, losses, loader, progress, objective_generator, model, state


def test_adaptive_video_dcp_resume_restores_both_streams_rng_and_ema(
    tmp_path: Path,
) -> None:
    (
        baseline_engine,
        baseline_losses,
        baseline_loader,
        baseline_progress,
        baseline_generator,
        baseline_model,
        baseline_state,
    ) = _checkpointable_adaptive_video_stack()
    baseline_session = NativeAdaptiveVideoTrainingSession(
        baseline_engine,
        baseline_loader,
        baseline_progress,
    )
    baseline_session.run(max_steps=2, generator=baseline_generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(baseline_state)

    expected = baseline_session.run(max_steps=1, generator=baseline_generator)
    expected_parameters = {
        name: value.detach().clone()
        for name, value in baseline_model.state_dict().items()
    }
    expected_ema = baseline_losses.regression_ema.state_dict()

    (
        restored_engine,
        restored_losses,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
    ) = _checkpointable_adaptive_video_stack()
    manager.load(restored_state, artifact.path)
    restored_session = NativeAdaptiveVideoTrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    )
    actual = restored_session.run(max_steps=1, generator=restored_generator)

    assert restored_progress.optimizer_steps == 3
    assert restored_progress.microbatches_seen == 5
    assert restored_loader.generated_source.cursor == 3
    assert restored_loader.real_source.cursor == 2
    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
    restored_ema = restored_losses.regression_ema.state_dict()
    for name in ("values", "initialized", "update_counts"):
        assert isinstance(restored_ema[name], torch.Tensor)
        assert isinstance(expected_ema[name], torch.Tensor)
        torch.testing.assert_close(restored_ema[name], expected_ema[name], rtol=0, atol=0)
