from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.progressive import (  # noqa: E402
    NativeProgressiveDistillationTrainingSession,
    ProgressiveDistillationBatch,
    ProgressiveDistillationConfig,
    ProgressiveRandomInputs,
    build_native_progressive_distillation_training_stack,
)
from worldfoundry.training.post_training.distillation.progressive.math import (  # noqa: E402
    alpha_sigma,
    cosine_logsnr,
    implied_clean_target,
    prediction_to_clean_epsilon_velocity,
)
from worldfoundry.training.recipes import (  # noqa: E402
    PostTrainingRecipe,
    ProgressiveDistillationAlgorithmSpec,
)


class _ToyModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))
        self.register_buffer("forward_calls", torch.tensor(0, dtype=torch.int64))


class _ToyAdapter:
    def __init__(self, value: float, identity: str = "teacher") -> None:
        self.module = _ToyModule(value)
        self.checkpoint_identity = identity
        self.calls: list[dict[str, object]] = []

    def predict_model_output(
        self,
        noisy_latents,
        logsnr,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        self.calls.append(
            {
                "sample_ids": sample_ids,
                "training": training,
                "logsnr": logsnr.detach().clone(),
            }
        )
        if training:
            with torch.no_grad():
                self.module.forward_calls.add_(1)
        bias = float(conditioning["bias"])
        expanded_logsnr = logsnr.reshape(
            (logsnr.shape[0],) + (1,) * (noisy_latents.ndim - 1)
        )
        return (
            noisy_latents * self.module.scale
            + expanded_logsnr.to(noisy_latents.dtype) * 0.002
            + bias
        )


def _batch(prefix: str, *, batch_size: int = 2) -> ProgressiveDistillationBatch:
    return ProgressiveDistillationBatch(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(batch_size)),
        clean_latents=torch.linspace(
            -0.75,
            0.8,
            batch_size * 4,
        ).reshape(batch_size, 1, 2, 2),
        conditioning={"bias": 0.03},
    )


def _recipe(
    *,
    accumulation: int = 2,
    seed: int = 17,
    stage_steps: int = 2,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "progressive", "output_dir": "unused"},
            "model": {
                "recipe": "toy-diffusion",
                "checkpoint": "teacher",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "latents.jsonl", "shuffle_seed": seed},
            "algorithm": {
                "type": "progressive-distillation",
                "teacher_checkpoint": "teacher",
                "start_num_steps": 8,
                "end_num_steps": 2,
                "optimizer_steps_per_stage": stage_steps,
                "prediction_type": "sample",
                "loss_weight": "snr_trunc",
                "logsnr_min": -13.0,
                "logsnr_max": 15.0,
                "ema_decay": 0.5,
                "learning_rate_anneal": "linear",
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.02,
                "weight_decay": 0.0,
                "betas": [0.8, 0.9],
                "epsilon": 1.0e-7,
                "max_grad_norm": 100.0,
                "gradient_accumulation_steps": accumulation,
            },
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
            },
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def _stack(*, stage_steps: int = 2, seed: int = 17):
    student = _ToyAdapter(-4.0)
    teacher = _ToyAdapter(0.35)
    target = _ToyAdapter(9.0)
    stack = build_native_progressive_distillation_training_stack(
        _recipe(stage_steps=stage_steps, seed=seed),
        student=student,
        teacher=teacher,
        ema_target=target,
        fused_adamw=False,
    )
    return stack, student, teacher, target


def test_config_halving_schedule_and_official_cosine_endpoints() -> None:
    config = ProgressiveDistillationConfig(
        start_num_steps=16,
        end_num_steps=2,
        optimizer_steps_per_stage=7,
    )
    assert config.teacher_steps == (16, 8, 4)
    assert config.student_steps == (8, 4, 2)
    assert config.stage_count == 3
    values = cosine_logsnr(
        torch.tensor([0.0, 1.0]),
        logsnr_min=-12.0,
        logsnr_max=14.0,
    )
    torch.testing.assert_close(values, torch.tensor([14.0, -12.0]), atol=2e-4, rtol=0)

    with pytest.raises(ValueError, match="halve exactly|reachable"):
        ProgressiveDistillationConfig(start_num_steps=12, end_num_steps=2)
    with pytest.raises(ValueError, match="learning_rate_anneal"):
        ProgressiveDistillationConfig(
            start_num_steps=8,
            end_num_steps=2,
            learning_rate_anneal="cosine",  # type: ignore[arg-type]
        )


def test_prediction_parameterizations_and_implied_target_match_equations() -> None:
    noisy = torch.tensor([[[0.3]], [[-0.2]]])
    logsnr = torch.tensor([1.2, -0.7])
    alpha, sigma = alpha_sigma(logsnr, noisy)
    clean = torch.tensor([[[0.1]], [[0.4]]])
    epsilon = (noisy - alpha * clean) / sigma
    velocity = alpha * epsilon - sigma * clean
    for output, prediction_type in (
        (clean, "sample"),
        (epsilon, "epsilon"),
        (velocity, "v_prediction"),
    ):
        actual = prediction_to_clean_epsilon_velocity(
            output,
            noisy,
            alpha,
            sigma,
            prediction_type=prediction_type,
        )
        torch.testing.assert_close(actual[0], clean)
        torch.testing.assert_close(actual[1], epsilon)
        torch.testing.assert_close(actual[2], velocity)

    end = torch.tensor([[[0.05]], [[0.25]]])
    final_clean = torch.tensor([[[0.9]], [[-0.4]]])
    end_logsnr = torch.tensor([5.0, 2.0])
    indices = torch.tensor([0, 2], dtype=torch.int64)
    target = implied_clean_target(
        noisy,
        end,
        final_clean,
        logsnr,
        end_logsnr,
        indices,
    )
    end_alpha, _ = alpha_sigma(end_logsnr, noisy)
    ratio = torch.exp(
        0.5
        * (
            torch.nn.functional.softplus(logsnr)
            - torch.nn.functional.softplus(end_logsnr)
        )
    ).reshape(2, 1, 1)
    expected_second = (
        end[1:] - ratio[1:] * noisy[1:]
    ) / (end_alpha[1:] - ratio[1:] * alpha[1:])
    torch.testing.assert_close(target[:1], final_clean[:1])
    torch.testing.assert_close(target[1:], expected_second)


def test_recipe_builder_consumes_every_field_and_initializes_all_roles() -> None:
    recipe = _recipe(accumulation=3, seed=23, stage_steps=5)
    assert isinstance(recipe.algorithm, ProgressiveDistillationAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    student = _ToyAdapter(-4.0)
    teacher = _ToyAdapter(0.35)
    target = _ToyAdapter(9.0)
    stack = build_native_progressive_distillation_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        ema_target=target,
        fused_adamw=False,
    )
    algorithm = recipe.algorithm
    assert stack.config.start_num_steps == algorithm.start_num_steps
    assert stack.config.end_num_steps == algorithm.end_num_steps
    assert stack.config.optimizer_steps_per_stage == algorithm.optimizer_steps_per_stage
    assert stack.config.prediction_type == algorithm.prediction_type
    assert stack.config.loss_weight == algorithm.loss_weight
    assert stack.config.logsnr_min == algorithm.logsnr_min
    assert stack.config.logsnr_max == algorithm.logsnr_max
    assert stack.config.ema_decay == algorithm.ema_decay
    assert stack.config.learning_rate_anneal == algorithm.learning_rate_anneal
    assert stack.engine.gradient_accumulation_steps == 3
    assert stack.engine.max_grad_norm == recipe.optimizer.max_grad_norm
    assert stack.optimizer.param_groups[0]["lr"] == recipe.optimizer.learning_rate
    assert stack.optimizer.param_groups[0]["betas"] == recipe.optimizer.betas
    assert stack.optimizer.param_groups[0]["eps"] == recipe.optimizer.epsilon
    for name, value in student.module.state_dict().items():
        torch.testing.assert_close(value, teacher.module.state_dict()[name])
        torch.testing.assert_close(value, target.module.state_dict()[name])
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in target.module.parameters())

    payload = recipe.to_dict()
    payload["algorithm"]["metadata"] = {"unused": True}  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields.*metadata"):
        PostTrainingRecipe.from_mapping(payload)


def test_builder_rejects_wrong_identity_initialization_and_optimizer() -> None:
    recipe = _recipe(accumulation=1)
    for position in range(3):
        roles = [_ToyAdapter(0.1), _ToyAdapter(0.2), _ToyAdapter(0.3)]
        roles[position].checkpoint_identity = "wrong"
        with pytest.raises(ValueError, match="loaded checkpoint identity"):
            build_native_progressive_distillation_training_stack(
                recipe,
                student=roles[0],
                teacher=roles[1],
                ema_target=roles[2],
                fused_adamw=False,
            )

    payload = recipe.to_dict()
    payload["model"]["checkpoint"] = "student"  # type: ignore[index]
    mismatched = PostTrainingRecipe.from_mapping(payload)
    with pytest.raises(ValueError, match="initialize from teacher_checkpoint"):
        build_native_progressive_distillation_training_stack(
            mismatched,
            student=_ToyAdapter(0.1, "student"),
            teacher=_ToyAdapter(0.2),
            ema_target=_ToyAdapter(0.3),
            fused_adamw=False,
        )

    payload = recipe.to_dict()
    payload["optimizer"]["type"] = "came"  # type: ignore[index]
    payload["optimizer"]["betas"] = [0.9, 0.999, 0.9999]  # type: ignore[index]
    payload["optimizer"]["epsilon"] = [1.0e-30, 1.0e-16]  # type: ignore[index]
    payload["optimizer"]["update_clip_threshold"] = 1.0  # type: ignore[index]
    came = PostTrainingRecipe.from_mapping(payload)
    with pytest.raises(ValueError, match="requires optimizer.type='adamw'"):
        build_native_progressive_distillation_training_stack(
            came,
            student=_ToyAdapter(0.1),
            teacher=_ToyAdapter(0.2),
            ema_target=_ToyAdapter(0.3),
        )


def test_objective_calls_two_frozen_teacher_steps_and_one_student_step() -> None:
    stack, student, teacher, target = _stack()
    batch = _batch("objective")
    random_inputs = ProgressiveRandomInputs(
        noise=torch.full_like(batch.clean_latents, 0.25),
        timestep_indices=torch.tensor([0, 3], dtype=torch.int64),
    )
    result = stack.objective.loss(
        batch,
        random_inputs=random_inputs,
        student_num_steps=4,
    )
    assert result.loss.ndim == 0 and torch.isfinite(result.loss)
    assert len(teacher.calls) == 2
    assert [call["training"] for call in teacher.calls] == [False, False]
    assert len(student.calls) == 1 and student.calls[0]["training"] is True
    assert not target.calls
    result.loss.backward()
    assert student.module.scale.grad is not None
    assert teacher.module.scale.grad is None
    assert target.module.scale.grad is None


def test_engine_commits_halving_stages_lr_reset_ema_buffers_and_final_export() -> None:
    stack, student, teacher, target = _stack(stage_steps=2)
    first = stack.engine.train_step((_batch("a"), _batch("b")))
    assert first.metrics["trained_teacher_num_steps"] == 8
    assert first.metrics["trained_student_num_steps"] == 4
    assert first.metrics["learning_rates"] == (0.02,)
    assert stack.engine.stage_step == 1
    assert target.module.forward_calls.item() == student.module.forward_calls.item()

    second = stack.engine.train_step((_batch("c"), _batch("d")))
    assert second.metrics["learning_rates"] == (0.01,)
    assert second.metrics["stage_finished"] is True
    assert stack.engine.stage_index == 1
    assert stack.engine.stage_step == 0
    assert stack.optimizer.param_groups[0]["lr"] == 0.02
    assert not stack.optimizer.state
    for name, value in target.module.state_dict().items():
        torch.testing.assert_close(value, student.module.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(value, teacher.module.state_dict()[name], rtol=0, atol=0)

    third = stack.engine.train_step((_batch("e"), _batch("f")))
    assert third.metrics["learning_rates"] == (0.02,)
    final = stack.engine.train_step((_batch("g"), _batch("h")))
    assert final.metrics["training_complete"] is True
    assert stack.engine.is_complete
    assert stack.engine.remaining_optimizer_steps == 0
    assert not stack.optimizer.state
    for name, value in target.module.state_dict().items():
        torch.testing.assert_close(value, student.module.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="reached end_num_steps"):
        stack.engine.train_step((_batch("i"), _batch("j")))


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> ProgressiveDistillationBatch:
        value = _batch(f"loader-{self.cursor}", batch_size=1)
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack():
    stack, _, _, _ = _stack(stage_steps=2, seed=101)
    loader = _StatefulLoader()
    progress = TrainingProgress()
    state = TrainingState(
        model=stack.model,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=torch.Generator().manual_seed(303),
        progress=progress,
        identity={"algorithm": "progressive-distillation"},
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, state


def test_session_stops_at_exact_final_boundary() -> None:
    stack, loader, progress, _ = _checkpointable_stack()
    summary = NativeProgressiveDistillationTrainingSession(
        stack.engine,
        loader,
        progress,
    ).run(max_steps=99)
    assert summary.iterations == 4
    assert summary.final_step == 4
    assert loader.cursor == 8
    assert progress.optimizer_steps == 4
    assert stack.engine.is_complete


def test_dcp_resume_restores_stage_rng_optimizer_models_and_loader(
    tmp_path: Path,
) -> None:
    stack, loader, progress, state = _checkpointable_stack()
    session = NativeProgressiveDistillationTrainingSession(
        stack.engine,
        loader,
        progress,
    )
    session.run(max_steps=1)
    manager = TrainingCheckpointer(tmp_path / "progressive-checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1)
    expected_model = {
        name: value.detach().clone()
        for name, value in stack.model.state_dict().items()
    }

    restored, restored_loader, restored_progress, restored_state = (
        _checkpointable_stack()
    )
    manager.load(restored_state, artifact.path)
    actual = NativeProgressiveDistillationTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1)
    assert actual.final_loss == expected.final_loss
    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    assert restored.engine.global_step == 2
    assert restored.engine.stage_index == 1
    assert restored.engine.stage_step == 0
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)


def test_lazy_facades_resolve_progressive_runtime() -> None:
    import worldfoundry.training.post_training as post_training
    import worldfoundry.training.post_training.distillation as distillation

    assert (
        post_training.build_native_progressive_distillation_training_stack
        is build_native_progressive_distillation_training_stack
    )
    assert (
        distillation.build_native_progressive_distillation_training_stack
        is build_native_progressive_distillation_training_stack
    )
    assert post_training.ProgressiveDistillationBatch is ProgressiveDistillationBatch
    assert distillation.ProgressiveDistillationBatch is ProgressiveDistillationBatch
