from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.objectives.flow_matching import (  # noqa: E402
    flow_clean_from_velocity,
)
from worldfoundry.training.post_training.distillation.diagonal import (  # noqa: E402
    NativeDiagonalTrainingSession,
    SpatialMotionHead,
    build_native_diagonal_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDTrainingBatch,
)
from worldfoundry.training.recipes import (  # noqa: E402
    DiagonalAlgorithmSpec,
    PostTrainingRecipe,
)


class _ScalarModule(torch.nn.Module):
    def __init__(self, value: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value), requires_grad=trainable)


class _CausalAdapter:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool,
        checkpoint_identity: str,
    ) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
        self.checkpoint_identity = checkpoint_identity

    def initialize_cache(self, reference, *, sample_ids, conditioning):
        del reference, sample_ids, conditioning
        return {"blocks": 0}

    def predict_clean_chunk(
        self,
        noisy_chunk,
        timesteps,
        sigmas,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
        training,
    ):
        del timesteps, sigmas, start_frame, sample_ids, conditioning, training
        assert cache["blocks"] == block_index
        return noisy_chunk * self.module.scale

    def commit_clean_chunk(
        self,
        clean_chunk,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
    ):
        return self.commit_context_chunk(
            clean_chunk,
            torch.zeros(clean_chunk.shape[0], device=clean_chunk.device),
            block_index=block_index,
            start_frame=start_frame,
            sample_ids=sample_ids,
            conditioning=conditioning,
            cache=cache,
        )

    def commit_context_chunk(
        self,
        context_chunk,
        context_timesteps,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
    ):
        del context_chunk, context_timesteps, start_frame, sample_ids, conditioning
        assert cache["blocks"] == block_index
        cache["blocks"] += 1
        return cache


class _ScoreAdapter:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool,
        checkpoint_identity: str,
    ) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
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
        del sigmas, sample_ids, conditioning, training
        branch_scale = 0.5 if branch == "negative" else 1.0
        return noisy_latents * self.module.scale * branch_scale

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


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> DMDTrainingBatch:
        sample = f"video-{self.cursor}"
        value = 0.1 + 0.01 * self.cursor
        self.cursor += 1
        return DMDTrainingBatch(
            sample_ids=(sample,),
            clean_latents=torch.full((1, 1, 4, 1, 1), value),
            conditioning={},
            unconditional_conditioning={},
        )

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _recipe_mapping(*, stage: str = "stage-two") -> dict[str, object]:
    return {
        "run": {"id": "diagonal-test", "output_dir": "unused"},
        "model": {
            "recipe": "wan2.1-t2v-1.3b-causal",
            "checkpoint": "student-checkpoint",
        },
        "tuning": {"mode": "full"},
        "data": {"manifest": "prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "diagonal-distillation",
            "stage": stage,
            "real_score_checkpoint": "real-score-checkpoint",
            "fake_score_checkpoint": "fake-score-checkpoint",
            "fixed_teacher_checkpoint": "fixed-teacher-checkpoint",
            "frames_per_block": 2,
            "frame_dim": 2,
            "latent_channels": 1,
            "generator_update_interval": 1,
            "student_scheduler_cadence": "generator-update",
            "ema_decay": 0.99,
            "ema_start_step": 0,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 2.0e-6,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 4.0e-7,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


def _roles():
    return {
        "student": _CausalAdapter(
            0.2,
            trainable=True,
            checkpoint_identity="student-checkpoint",
        ),
        "real_score": _ScoreAdapter(
            0.7,
            trainable=False,
            checkpoint_identity="real-score-checkpoint",
        ),
        "fake_score": _ScoreAdapter(
            0.3,
            trainable=True,
            checkpoint_identity="fake-score-checkpoint",
        ),
        "fixed_teacher": _CausalAdapter(
            0.4,
            trainable=False,
            checkpoint_identity="fixed-teacher-checkpoint",
        ),
    }


def _checkpointable_stack():
    torch.manual_seed(71)
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    roles = _roles()
    stack = build_native_diagonal_training_stack(
        recipe,
        **roles,
        fused_adamw=False,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    model = torch.nn.ModuleDict(
        {
            "student": roles["student"].module,
            "fake_score": roles["fake_score"].module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
        engine=stack.engine,
        dataloader=loader,
        objective_generator=stack.sampler.generator,
        progress=progress,
        identity={"algorithm": "diagonal-distillation"},
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, model, state


def test_diagonal_recipe_selects_released_stage_and_consumes_all_fields() -> None:
    first = PostTrainingRecipe.from_mapping(_recipe_mapping(stage="stage-one"))
    second = PostTrainingRecipe.from_mapping(_recipe_mapping(stage="stage-two"))

    assert isinstance(first.algorithm, DiagonalAlgorithmSpec)
    assert first.algorithm.stage == "stage-one"
    assert second.algorithm.stage == "stage-two"
    assert first.algorithm != second.algorithm
    with pytest.raises(ValueError, match="fake_score_optimizer"):
        payload = _recipe_mapping()
        payload.pop("fake_score_optimizer")
        PostTrainingRecipe.from_mapping(payload)


def test_diagonal_builder_registers_roles_schedule_motion_head_and_cadence() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    roles = _roles()
    stack = build_native_diagonal_training_stack(
        recipe,
        **roles,
        fused_adamw=False,
    )

    assert stack.recipe is recipe
    assert stack.schedule.base_schedule.timesteps == pytest.approx(
        (1000.0, 357.14285714285717)
    )
    assert stack.schedule.block_schedule(0).timesteps == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 357.14285714285717)
    )
    assert stack.fixed_teacher_schedule.last_step_only
    assert isinstance(stack.motion_head_student, SpatialMotionHead)
    assert roles["student"].module.diagonal_motion_head is stack.motion_head_student
    assert not any(
        parameter.requires_grad for parameter in stack.motion_head_teacher.parameters()
    )
    assert stack.engine.gradient_accumulation_steps == 2
    assert stack.engine.generator_update_interval == 1
    assert stack.engine.dmd_engine.student_ema_start_step == 0
    assert stack.student_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-6)
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == pytest.approx(4.0e-7)


def test_diagonal_builder_rejects_checkpoint_role_mismatch() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    roles = _roles()
    roles["fixed_teacher"].checkpoint_identity = "wrong"
    with pytest.raises(ValueError, match="fixed teacher loaded checkpoint identity"):
        build_native_diagonal_training_stack(
            recipe,
            **roles,
            fused_adamw=False,
        )


def test_diagonal_session_reuses_scalable_dmd_accumulation() -> None:
    stack, loader, progress, _, _ = _checkpointable_stack()
    events: list[dict[str, object]] = []
    summary = NativeDiagonalTrainingSession(
        stack.engine,
        loader,
        progress,
        event_sink=lambda value: events.append(dict(value)),
    ).run(max_steps=1)

    assert summary.final_step == 1
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 2
    assert loader.cursor == 2
    assert events[0]["schema"] == "worldfoundry-diagonal-step-event"
    assert stack.loss_adapter.motion_ema_updates == 1


def test_diagonal_dcp_restores_model_optimizer_rng_ema_motion_and_cursor(
    tmp_path: Path,
) -> None:
    stack, loader, progress, model, state = _checkpointable_stack()
    session = NativeDiagonalTrainingSession(stack.engine, loader, progress)
    session.run(max_steps=1)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)

    expected = session.run(max_steps=1)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_motion_teacher = {
        name: value.detach().clone()
        for name, value in stack.motion_head_teacher.state_dict().items()
    }
    expected_rng = stack.sampler.generator.get_state().clone()

    restored, restored_loader, restored_progress, restored_model, restored_state = (
        _checkpointable_stack()
    )
    manager.load(restored_state, artifact.path)
    actual = NativeDiagonalTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1)

    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    assert restored.loss_adapter.motion_ema_updates == 2
    torch.testing.assert_close(
        restored.sampler.generator.get_state(),
        expected_rng,
        rtol=0,
        atol=0,
    )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
    for name, value in restored.motion_head_teacher.state_dict().items():
        torch.testing.assert_close(value, expected_motion_teacher[name], rtol=0, atol=0)
