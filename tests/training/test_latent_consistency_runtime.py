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
from worldfoundry.training.post_training.distillation.latent_consistency import (  # noqa: E402
    LatentConsistencyNoiseSchedule,
    LatentConsistencyRandomInputs,
    LatentConsistencyTrainingBatch,
    NativeLatentConsistencyTrainingSession,
    build_native_latent_consistency_training_stack,
)
from worldfoundry.training.post_training.distillation.latent_consistency.math import (  # noqa: E402
    add_forward_diffusion_noise,
    append_dims,
    classifier_free_guidance,
    deterministic_ddim_step,
    gather_schedule_coefficients,
    prediction_to_origin_and_epsilon,
)
from worldfoundry.training.recipes import (  # noqa: E402
    LatentConsistencyAlgorithmSpec,
    PostTrainingRecipe,
)


class _ScaleModule(torch.nn.Module):
    def __init__(self, scale: float, guidance_weight: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.guidance_weight = torch.nn.Parameter(torch.tensor(guidance_weight))
        self.register_buffer("fixed_offset", torch.tensor(0.125))


class _RecordingPredictionAdapter:
    def __init__(
        self,
        scale: float,
        guidance_weight: float,
        checkpoint_identity: str,
    ) -> None:
        self.module = _ScaleModule(scale, guidance_weight)
        self.checkpoint_identity = checkpoint_identity
        self.calls: list[dict[str, object]] = []

    def predict_model_output(
        self,
        noisy_latents,
        timesteps,
        *,
        guidance_embedding,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        self.calls.append(
            {
                "noisy_latents": noisy_latents.detach().clone(),
                "timesteps": timesteps.detach().clone(),
                "guidance_embedding": (None if guidance_embedding is None else guidance_embedding.detach().clone()),
                "sample_ids": sample_ids,
                "conditioning": conditioning,
                "training": training,
                "branch": branch,
            }
        )
        bias = torch.as_tensor(
            conditioning["bias"],
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        output = noisy_latents * self.module.scale + bias + self.module.fixed_offset
        if guidance_embedding is not None:
            embedded = guidance_embedding.float().mean(dim=1)
            embedded = append_dims(
                embedded.to(dtype=noisy_latents.dtype),
                noisy_latents.ndim,
            )
            output = output + embedded * self.module.guidance_weight
        return output


def _batch(
    sample_prefix: str,
    *,
    batch_size: int = 2,
    height: int = 2,
) -> LatentConsistencyTrainingBatch:
    return LatentConsistencyTrainingBatch(
        sample_ids=tuple(f"{sample_prefix}-{index}" for index in range(batch_size)),
        clean_latents=torch.linspace(
            -0.5,
            1.0,
            batch_size * height,
        ).reshape(batch_size, 1, height, 1),
        conditioning={"bias": 0.2, "name": "positive"},
        unconditional_conditioning={"bias": -0.1, "name": "negative"},
    )


def _recipe(
    *,
    accumulation: int,
    seed: int,
    algorithm: Mapping[str, object] | None = None,
    optimizer: Mapping[str, object] | None = None,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "latent-consistency", "output_dir": "unused"},
            "model": {
                "recipe": "test-diffusion-model",
                "checkpoint": "student",
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "latents.jsonl",
                "shuffle_seed": seed,
            },
            "algorithm": dict(
                algorithm
                or {
                    "type": "latent-consistency",
                    "teacher_checkpoint": "teacher",
                    "num_train_timesteps": 4,
                    "num_ddim_timesteps": 2,
                    "prediction_type": "epsilon",
                    "guidance_coefficient_min": 1.0,
                    "guidance_coefficient_max": 2.0,
                    "guidance_embedding_dim": 5,
                    "ema_decay": 0.5,
                }
            ),
            "optimizer": dict(
                optimizer
                or {
                    "type": "adamw",
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "max_grad_norm": 100.0,
                    "gradient_accumulation_steps": accumulation,
                }
            ),
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
            },
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def _stack(
    *,
    accumulation: int,
    seed: int,
    scheduler: bool = False,
):
    recipe = _recipe(accumulation=accumulation, seed=seed)
    student = _RecordingPredictionAdapter(0.2, 0.05, "student")
    teacher = _RecordingPredictionAdapter(0.35, 0.01, "teacher")
    target = _RecordingPredictionAdapter(-9.0, -3.0, "student")
    stack = build_native_latent_consistency_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        ema_target=target,
        noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
        scheduler_factory=(
            (lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)) if scheduler else None
        ),
        fused_adamw=False,
    )
    return stack, student, teacher, target


def test_recipe_round_trip_consumes_every_lcm_field_and_shared_optimizer() -> None:
    recipe = _recipe(
        accumulation=3,
        seed=73,
        algorithm={
            "type": "latent-consistency",
            "teacher_checkpoint": "teacher",
            "num_train_timesteps": 4,
            "num_ddim_timesteps": 2,
            "prediction_type": "v_prediction",
            "guidance_coefficient_min": 0.25,
            "guidance_coefficient_max": 0.75,
            "guidance_embedding_dim": 7,
            "guidance_embedding_scale": 321.0,
            "guidance_embedding_max_period": 1234.0,
            "sigma_data": 0.7,
            "timestep_scaling": 3.0,
            "loss_type": "pseudo_huber",
            "pseudo_huber_c": 0.02,
            "ema_decay": 0.8,
        },
        optimizer={
            "type": "adamw",
            "learning_rate": 0.004,
            "weight_decay": 0.03,
            "betas": [0.8, 0.91],
            "epsilon": 2.0e-7,
            "max_grad_norm": 4.0,
            "gradient_accumulation_steps": 3,
        },
    )
    assert isinstance(recipe.algorithm, LatentConsistencyAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    student = _RecordingPredictionAdapter(0.2, 0.05, "student")
    teacher = _RecordingPredictionAdapter(0.35, 0.01, "teacher")
    target = _RecordingPredictionAdapter(-9.0, -3.0, "student")
    stack = build_native_latent_consistency_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        ema_target=target,
        noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
        fused_adamw=False,
    )

    algorithm = recipe.algorithm
    assert stack.recipe is recipe
    assert stack.config.num_ddim_timesteps == algorithm.num_ddim_timesteps
    assert stack.config.prediction_type == algorithm.prediction_type
    assert stack.config.guidance_coefficient_min == algorithm.guidance_coefficient_min
    assert stack.config.guidance_coefficient_max == algorithm.guidance_coefficient_max
    assert stack.config.guidance_embedding_dim == algorithm.guidance_embedding_dim
    assert stack.config.guidance_embedding_scale == algorithm.guidance_embedding_scale
    assert stack.config.guidance_embedding_max_period == algorithm.guidance_embedding_max_period
    assert stack.config.sigma_data == algorithm.sigma_data
    assert stack.config.timestep_scaling == algorithm.timestep_scaling
    assert stack.config.loss_type == algorithm.loss_type
    assert stack.config.pseudo_huber_c == algorithm.pseudo_huber_c
    assert stack.config.ema_decay == algorithm.ema_decay
    assert stack.noise_schedule.num_train_timesteps == algorithm.num_train_timesteps
    assert stack.optimizer.param_groups[0]["lr"] == recipe.optimizer.learning_rate
    assert stack.optimizer.param_groups[0]["weight_decay"] == recipe.optimizer.weight_decay
    assert stack.optimizer.param_groups[0]["betas"] == recipe.optimizer.betas
    assert stack.optimizer.param_groups[0]["eps"] == recipe.optimizer.epsilon
    assert stack.engine.max_grad_norm == recipe.optimizer.max_grad_norm
    assert stack.engine.gradient_accumulation_steps == 3
    expected_rng = torch.Generator().manual_seed(73)
    torch.testing.assert_close(stack.engine._rng.get_state(), expected_rng.get_state())
    torch.testing.assert_close(target.module.scale, student.module.scale)
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in target.module.parameters())

    payload = recipe.to_dict()
    payload["algorithm"]["metadata"] = {"unused": True}  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields.*metadata"):
        PostTrainingRecipe.from_mapping(payload)


def test_builder_rejects_unverified_optimizer_identity_schedule_and_shared_roles() -> None:
    came_recipe = _recipe(
        accumulation=1,
        seed=5,
        optimizer={
            "type": "came",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999, 0.9999],
            "epsilon": [1.0e-30, 1.0e-16],
            "update_clip_threshold": 1.0,
        },
    )

    def adapters():
        return (
            _RecordingPredictionAdapter(0.2, 0.05, "student"),
            _RecordingPredictionAdapter(0.35, 0.01, "teacher"),
            _RecordingPredictionAdapter(-9.0, -3.0, "student"),
        )

    student, teacher, target = adapters()
    with pytest.raises(ValueError, match="requires optimizer.type='adamw'"):
        build_native_latent_consistency_training_stack(
            came_recipe,
            student=student,
            teacher=teacher,
            ema_target=target,
            noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
        )

    recipe = _recipe(accumulation=1, seed=5)
    for role, wrong_identity in (
        ("student", "wrong-student"),
        ("teacher", "wrong-teacher"),
        ("target", "wrong-target"),
    ):
        student, teacher, target = adapters()
        selected = {"student": student, "teacher": teacher, "target": target}[role]
        selected.checkpoint_identity = wrong_identity
        with pytest.raises(ValueError, match="loaded checkpoint identity"):
            build_native_latent_consistency_training_stack(
                recipe,
                student=student,
                teacher=teacher,
                ema_target=target,
                noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
                fused_adamw=False,
            )

    student, teacher, target = adapters()
    target.module = student.module
    with pytest.raises(ValueError, match="independently materialized|share parameters"):
        build_native_latent_consistency_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            ema_target=target,
            noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
            fused_adamw=False,
        )

    student, teacher, target = adapters()
    with pytest.raises(ValueError, match="noise schedule length"):
        build_native_latent_consistency_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            ema_target=target,
            noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.7, 0.25)),
            fused_adamw=False,
        )


def test_lazy_facades_resolve_canonical_lcm_runtime() -> None:
    import worldfoundry.training.post_training as post_training
    import worldfoundry.training.post_training.distillation as distillation

    assert (
        post_training.build_native_latent_consistency_training_stack
        is build_native_latent_consistency_training_stack
    )
    assert (
        distillation.build_native_latent_consistency_training_stack
        is build_native_latent_consistency_training_stack
    )
    assert post_training.LatentConsistencyTrainingBatch is LatentConsistencyTrainingBatch
    assert distillation.LatentConsistencyTrainingBatch is LatentConsistencyTrainingBatch


def test_objective_wires_teacher_ddim_target_boundary_and_guidance_embedding() -> None:
    stack, student, teacher, target = _stack(accumulation=1, seed=7)
    batch = _batch("objective")
    noise = torch.tensor(
        [
            [[[0.25], [-0.5]]],
            [[[1.0], [0.75]]],
        ]
    )
    indices = torch.tensor([0, 1], dtype=torch.int64)
    guidance = torch.tensor([1.25, 1.75])
    result = stack.objective.loss(
        batch,
        random_inputs=LatentConsistencyRandomInputs(
            noise=noise,
            timestep_indices=indices,
            guidance_coefficients=guidance,
        ),
    )

    assert len(student.calls) == 1
    assert len(teacher.calls) == 2
    assert len(target.calls) == 1
    assert student.calls[0]["training"] is True
    assert teacher.calls[0]["branch"] == "positive"
    assert teacher.calls[1]["branch"] == "negative"
    assert all(call["guidance_embedding"] is None for call in teacher.calls)
    student_embedding = student.calls[0]["guidance_embedding"]
    target_embedding = target.calls[0]["guidance_embedding"]
    assert isinstance(student_embedding, torch.Tensor)
    assert isinstance(target_embedding, torch.Tensor)
    assert student_embedding.shape == (2, 5)
    torch.testing.assert_close(student_embedding, target_embedding)

    alpha_schedule, sigma_schedule, starts, _, previous_alphas = stack.objective._schedule_tensors(
        batch.clean_latents.device
    )
    start_timesteps = starts.gather(0, indices)
    alpha = gather_schedule_coefficients(
        alpha_schedule,
        start_timesteps,
        batch.clean_latents,
    )
    sigma = gather_schedule_coefficients(
        sigma_schedule,
        start_timesteps,
        batch.clean_latents,
    )
    noisy = add_forward_diffusion_noise(batch.clean_latents, noise, alpha, sigma)
    conditional = noisy * teacher.module.scale.detach() + 0.2 + teacher.module.fixed_offset
    unconditional = noisy * teacher.module.scale.detach() - 0.1 + teacher.module.fixed_offset
    conditional_origin, conditional_epsilon = prediction_to_origin_and_epsilon(
        conditional,
        noisy,
        alpha,
        sigma,
        prediction_type="epsilon",
    )
    unconditional_origin, unconditional_epsilon = prediction_to_origin_and_epsilon(
        unconditional,
        noisy,
        alpha,
        sigma,
        prediction_type="epsilon",
    )
    expected_previous = deterministic_ddim_step(
        classifier_free_guidance(
            conditional_origin,
            unconditional_origin,
            guidance,
        ),
        classifier_free_guidance(
            conditional_epsilon,
            unconditional_epsilon,
            guidance,
        ),
        append_dims(previous_alphas.gather(0, indices), noisy.ndim),
    )
    torch.testing.assert_close(target.calls[0]["noisy_latents"], expected_previous)

    result.loss.backward()
    assert all(parameter.grad is not None for parameter in student.module.parameters())
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in target.module.parameters())
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in target.module.parameters())
    assert not teacher.module.training
    assert not target.module.training


def test_engine_accumulates_arbitrary_microbatches_and_updates_ema_once() -> None:
    actual, actual_student, actual_teacher, actual_target = _stack(
        accumulation=3,
        seed=29,
    )
    expected, expected_student, _, expected_target = _stack(
        accumulation=3,
        seed=29,
    )
    batches = (
        _batch("small", batch_size=1, height=1),
        _batch("medium", batch_size=2, height=2),
        _batch("large", batch_size=1, height=4),
    )
    initial_student = {name: value.detach().clone() for name, value in actual_student.module.state_dict().items()}
    teacher_before = {name: value.detach().clone() for name, value in actual_teacher.module.state_dict().items()}

    expected.optimizer.zero_grad(set_to_none=True)
    weights = [batch.clean_latents.numel() for batch in batches]
    total_weight = sum(weights)
    expected_losses = []
    for batch, weight in zip(batches, weights, strict=True):
        objective_result = expected.objective.loss(
            batch,
            random_inputs=expected.engine.sample_random_inputs(batch),
        )
        (objective_result.loss * (weight / total_weight)).backward()
        expected_losses.append(objective_result.loss.detach() * weight)
    torch.nn.utils.clip_grad_norm_(expected_student.module.parameters(), 100.0)
    expected.optimizer.step()
    expected.engine.ema_updater.update()

    result = actual.engine.train_step(batches)
    torch.testing.assert_close(
        result.loss,
        sum(expected_losses) / total_weight,
    )
    assert result.metrics["accumulated_microbatches"] == 3
    assert result.metrics["loss_denominator"] == total_weight
    for name, value in actual_student.module.state_dict().items():
        torch.testing.assert_close(
            value,
            expected_student.module.state_dict()[name],
            rtol=0,
            atol=0,
        )
    for name, value in actual_target.module.state_dict().items():
        torch.testing.assert_close(
            value,
            expected_target.module.state_dict()[name],
            rtol=0,
            atol=0,
        )
        if name in {"scale", "guidance_weight"}:
            torch.testing.assert_close(
                value,
                0.5 * initial_student[name] + 0.5 * actual_student.module.state_dict()[name],
            )
    for name, value in actual_teacher.module.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert actual.engine.global_step == 1
    assert actual.engine.optimizer_steps == 1


class _ForwardScaleModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))

    def forward(self, value):
        return value * self.scale


class _ForwardAdapter:
    def __init__(self, module: torch.nn.Module, checkpoint_identity: str) -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity

    def predict_model_output(
        self,
        noisy_latents,
        timesteps,
        *,
        guidance_embedding,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del timesteps, guidance_embedding, sample_ids, training, branch
        return self.module(noisy_latents) + float(conditioning["bias"])


def test_ddp_student_updates_an_independent_unwrapped_ema_target(tmp_path: Path) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    rendezvous = (tmp_path / "ddp-init").resolve()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        student_module = torch.nn.parallel.DistributedDataParallel(_ForwardScaleModule(0.2))
        student = _ForwardAdapter(student_module, "student")
        teacher = _ForwardAdapter(_ForwardScaleModule(0.35), "teacher")
        target = _ForwardAdapter(_ForwardScaleModule(-5.0), "student")
        stack = build_native_latent_consistency_training_stack(
            _recipe(accumulation=1, seed=11),
            student=student,
            teacher=teacher,
            ema_target=target,
            noise_schedule=LatentConsistencyNoiseSchedule((0.95, 0.8, 0.55, 0.25)),
            fused_adamw=False,
        )
        torch.testing.assert_close(
            target.module.scale,
            student_module.module.scale,
        )
        initial = target.module.scale.detach().clone()
        stack.engine.train_step(_batch("ddp", batch_size=1, height=2))
        torch.testing.assert_close(
            target.module.scale,
            0.5 * initial + 0.5 * student_module.module.scale.detach(),
        )
    finally:
        torch.distributed.destroy_process_group()


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> LatentConsistencyTrainingBatch:
        value = _batch(f"loader-{self.cursor}", batch_size=1, height=2)
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack():
    stack, _, _, _ = _stack(accumulation=2, seed=101, scheduler=True)
    loader = _StatefulLoader()
    progress = TrainingProgress()
    state = TrainingState(
        model=stack.model,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=torch.Generator().manual_seed(303),
        progress=progress,
        identity={"algorithm": "latent-consistency"},
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, state


def test_dcp_resume_restores_models_optimizer_scheduler_rng_and_loader(
    tmp_path: Path,
) -> None:
    stack, loader, progress, state = _checkpointable_stack()
    session = NativeLatentConsistencyTrainingSession(
        stack.engine,
        loader,
        progress,
    )
    session.run(max_steps=1)
    manager = TrainingCheckpointer(tmp_path / "latent-consistency-checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1)
    expected_model = {name: value.detach().clone() for name, value in stack.model.state_dict().items()}
    expected_lr = stack.optimizer.param_groups[0]["lr"]

    restored, restored_loader, restored_progress, restored_state = _checkpointable_stack()
    manager.load(restored_state, artifact.path)
    actual = NativeLatentConsistencyTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1)

    assert actual.final_loss == expected.final_loss
    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    assert restored.engine.global_step == 2
    assert restored.optimizer.param_groups[0]["lr"] == expected_lr
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
