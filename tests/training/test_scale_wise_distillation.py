from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

from worldfoundry.training.checkpoint import (
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.scale_wise.builder import (
    build_native_scale_wise_training_stack,
)
from worldfoundry.training.post_training.distillation.scale_wise.config import (
    ScaleWiseConfig,
    ScaleWiseSchedule,
)
from worldfoundry.training.post_training.distillation.scale_wise.contracts import (
    ScaleWiseTrainingBatch,
)
from worldfoundry.training.post_training.distillation.scale_wise.engine import (
    NativeScaleWiseTrainEngine,
)
from worldfoundry.training.post_training.distillation.scale_wise.math import (
    discriminator_logistic_loss,
    dmd_loss_per_sample,
    generator_logistic_loss,
    mmd_loss,
)
from worldfoundry.training.post_training.distillation.scale_wise.objective import (
    FlowScaleWiseLossAdapter,
)
from worldfoundry.training.post_training.distillation.scale_wise.sd3 import (
    SD3AdapterDisabledTeacherAdapter,
    SD3ScaleWiseCriticAdapter,
    SD3ScaleWiseCriticModule,
    SD3ScaleWisePredictionAdapter,
    sd3_velocity_and_features,
)
from worldfoundry.training.post_training.distillation.scale_wise.session import (
    NativeScaleWiseTrainingSession,
)
from worldfoundry.training.recipes import PostTrainingRecipe, ScaleWiseAlgorithmSpec


class _VelocityModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class _CriticModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.15))
        self.classifier = nn.Linear(1, 1)


class _PredictionAdapter:
    def __init__(self, module: _VelocityModule, identity: str) -> None:
        self.module = module
        self.checkpoint_identity = identity

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sigmas, sample_ids, conditioning, training
        offset = -0.05 if branch == "unconditional" else 0.0
        return self.module(noisy_latents) + offset


class _CriticAdapter:
    def __init__(self, module: _CriticModule, identity: str) -> None:
        self.module = module
        self.checkpoint_identity = identity

    @staticmethod
    def audit_scale_wise_critic(
        *,
        classifier_blocks: tuple[int, ...],
        mmd_blocks: tuple[int, ...],
        discriminator_layers: int,
    ) -> None:
        if not classifier_blocks or not mmd_blocks or discriminator_layers <= 0:
            raise ValueError("invalid toy critic configuration")

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sigmas, sample_ids, conditioning, training
        offset = -0.025 if branch == "unconditional" else 0.0
        return noisy_latents * self.module.velocity + offset

    @staticmethod
    def _features(
        noisy_latents: torch.Tensor,
        block_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        tokens = noisy_latents.flatten(2).transpose(1, 2)
        return tuple(tokens * float(index + 1) for index in block_indices)

    def predict_velocity_and_features(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
        )
        return velocity, self._features(noisy_latents, block_indices)

    def extract_features(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[torch.Tensor, ...]:
        del sigmas, sample_ids, conditioning, training
        return self._features(noisy_latents, block_indices)

    def classify_features(
        self,
        pooled_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.module.classifier(value) for value in pooled_features)


def _schedule() -> ScaleWiseSchedule:
    return ScaleWiseSchedule(
        scales=(2, 4),
        boundary_indices=(0, 1, 3),
        solver_sigmas=(1.0, 0.6, 0.25, 0.0),
    )


def _config(*, fake_updates: int = 2) -> ScaleWiseConfig:
    return ScaleWiseConfig(
        schedule=_schedule(),
        fake_updates_per_iteration=fake_updates,
        dmd_noise_start_index=1,
        dmd_noise_end_index=3,
        mmd_noise_start_index=1,
        mmd_noise_end_index=3,
        classifier_blocks=(0,),
        mmd_blocks=(0,),
    )


def _batch(interval: int, *, seed: int) -> ScaleWiseTrainingBatch:
    generator = torch.Generator().manual_seed(seed)
    current_scale = (2, 4)[interval]
    previous_scale = (2, 2)[interval]
    return ScaleWiseTrainingBatch(
        sample_ids=(f"sample-{seed}-a", f"sample-{seed}-b"),
        current_latents=torch.randn(
            2,
            1,
            current_scale,
            current_scale,
            generator=generator,
        ),
        previous_latents=torch.randn(
            2,
            1,
            previous_scale,
            previous_scale,
            generator=generator,
        ),
        conditioning={"prompt": ("a", "b")},
        unconditional_conditioning={"prompt": ("", "")},
        interval_index=interval,
    )


def _roles() -> tuple[_PredictionAdapter, _PredictionAdapter, _CriticAdapter]:
    student = _PredictionAdapter(_VelocityModule(0.25), "student")
    teacher = _PredictionAdapter(_VelocityModule(0.4), "teacher")
    teacher.module.requires_grad_(False)
    critic = _CriticAdapter(_CriticModule(), "fake")
    return student, teacher, critic


def _recipe_mapping() -> dict[str, object]:
    return {
        "run": {"id": "scale-wise-test", "output_dir": "unused"},
        "model": {"recipe": "sd35-medium", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "latents.jsonl", "shuffle": False},
        "algorithm": {
            "type": "scale-wise-distillation",
            "teacher_checkpoint": "teacher",
            "fake_score_checkpoint": "fake",
            "scales": [2, 4],
            "boundary_indices": [0, 1, 3],
            "solver_sigmas": [1.0, 0.6, 0.25, 0.0],
            "fake_updates_per_iteration": 2,
            "dmd_noise_start_index": 1,
            "dmd_noise_end_index": 3,
            "mmd_noise_start_index": 1,
            "mmd_noise_end_index": 3,
            "classifier_blocks": [0],
            "mmd_blocks": [0],
            "discriminator_layers": 2,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


class _StatefulScaleLoader:
    batches_per_iteration = 6

    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> ScaleWiseTrainingBatch:
        interval = (self.cursor // self.batches_per_iteration) % 2
        value = _batch(interval, seed=100 + self.cursor)
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack():
    torch.manual_seed(31)
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student, teacher, critic = _roles()
    stack = build_native_scale_wise_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=critic,
        fused_adamw=False,
    )
    loader = _StatefulScaleLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(37)
    model = nn.ModuleDict(
        {"student": student.module, "fake_score": critic.module}
    )
    state = TrainingState(
        model=model,
        optimizer=stack.optimizers,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={"algorithm": "scale-wise-distillation"},
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, generator, model, state


def test_released_schedule_matches_published_boundary_sigmas() -> None:
    schedule = ScaleWiseSchedule.released_sd35_four_step()
    assert schedule.scales == (64, 80, 96, 128)
    assert schedule.boundary_indices == (0, 7, 14, 18, 28)
    assert tuple(round(schedule.start_sigma(index), 4) for index in range(4)) == (
        1.0,
        0.8959,
        0.7371,
        0.6022,
    )


def test_dmd_pseudo_target_matches_released_formula() -> None:
    generated = torch.tensor([[[2.0, 4.0]]], requires_grad=True)
    real_clean = torch.tensor([[[1.0, 2.0]]])
    fake_clean = torch.tensor([[[1.5, 3.0]]])
    loss, gradient, normalizer = dmd_loss_per_sample(
        generated,
        real_clean,
        fake_clean,
    )
    expected_normalizer = torch.tensor([[[1.5]]])
    expected_gradient = torch.tensor([[[1.0 / 3.0, 2.0 / 3.0]]])
    assert torch.allclose(normalizer, expected_normalizer)
    assert torch.allclose(gradient, expected_gradient)
    assert torch.allclose(loss, 0.5 * expected_gradient.square().flatten(1).mean(1))
    loss.mean().backward()
    assert generated.grad is not None


def test_gan_and_mmd_formulas_match_reference_equations() -> None:
    fake_logits = (torch.tensor([[0.0], [1.0]]), torch.tensor([[2.0], [3.0]]))
    real_logits = (torch.tensor([[1.0], [0.0]]), torch.tensor([[3.0], [2.0]]))
    averaged_fake = torch.tensor([[1.0], [2.0]])
    averaged_real = torch.tensor([[2.0], [1.0]])
    assert torch.allclose(
        generator_logistic_loss(fake_logits),
        torch.nn.functional.softplus(-averaged_fake).flatten(),
    )
    assert torch.allclose(
        discriminator_logistic_loss(fake_logits, real_logits),
        (
            torch.nn.functional.softplus(averaged_fake)
            + torch.nn.functional.softplus(-averaged_real)
        ).flatten(),
    )
    real = torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]])
    fake = torch.tensor([[[1.0], [2.0]], [[0.0], [2.0]]])
    delta = (real.mean(1) - fake.mean(1)).square()
    expected = ((delta + 0.1**2).sqrt() - 0.1).mean()
    assert torch.allclose(
        mmd_loss(
            real,
            fake,
            kernel="linear",
            rbf_sigma=1.0,
            batch_mmd=False,
            huber_c=0.1,
        ),
        expected,
    )


def test_objective_executes_every_released_loss() -> None:
    student, teacher, critic = _roles()
    objective = FlowScaleWiseLossAdapter(
        student,
        teacher,
        critic,
        _config(),
    )
    result = objective.student_loss(
        _batch(0, seed=1),
        generator=torch.Generator().manual_seed(10),
    )
    assert torch.isfinite(result.loss)
    assert {
        "dmd_loss",
        "generator_gan_loss",
        "mmd_loss",
        "dmd_noise_indices",
        "mmd_noise_indices",
    } <= result.metrics.keys()
    fake_result = objective.fake_score_loss(
        _batch(0, seed=2),
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.isfinite(fake_result.loss)
    assert {"fake_diffusion_loss", "critic_gan_loss"} <= fake_result.metrics.keys()


def test_engine_runs_fresh_fake_updates_before_one_student_commit() -> None:
    student, teacher, critic = _roles()
    objective = FlowScaleWiseLossAdapter(student, teacher, critic, _config())
    student_optimizer = torch.optim.SGD(student.module.parameters(), lr=0.01)
    fake_optimizer = torch.optim.SGD(critic.module.parameters(), lr=0.01)
    engine = NativeScaleWiseTrainEngine(
        student_module=student.module,
        teacher_module=teacher.module,
        fake_score_module=critic.module,
        loss_adapter=objective,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        gradient_accumulation_steps=2,
    )
    initial_student = student.module.scale.detach().clone()
    initial_fake = critic.module.velocity.detach().clone()
    result = engine.train_step(
        (_batch(0, seed=1), _batch(0, seed=2)),
        fake_score_batches=tuple(_batch(0, seed=value) for value in range(3, 7)),
        generator=torch.Generator().manual_seed(12),
    )
    assert result.interval_index == 0
    assert engine.global_step == 1
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 2
    assert result.metrics["fake_score_updates"] == 2
    assert not torch.equal(student.module.scale.detach(), initial_student)
    assert not torch.equal(critic.module.velocity.detach(), initial_fake)
    assert engine.state_dict()["fake_score_optimizer_steps"] == 2

    with pytest.raises(ValueError, match="interval 1"):
        engine.train_step(
            (_batch(0, seed=20), _batch(0, seed=21)),
            fake_score_batches=tuple(
                _batch(0, seed=value) for value in range(22, 26)
            ),
        )


def test_engine_rejects_missing_fresh_fake_batches() -> None:
    student, teacher, critic = _roles()
    objective = FlowScaleWiseLossAdapter(student, teacher, critic, _config())
    engine = NativeScaleWiseTrainEngine(
        student_module=student.module,
        teacher_module=teacher.module,
        fake_score_module=critic.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=0.01),
        fake_score_optimizer=torch.optim.SGD(critic.module.parameters(), lr=0.01),
    )
    with pytest.raises(ValueError, match="2 fresh fake microbatches"):
        engine.train_step(
            _batch(0, seed=1),
            fake_score_batches=(_batch(0, seed=2),),
        )


def test_scale_wise_recipe_and_builder_consume_execution_fields() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert isinstance(recipe.algorithm, ScaleWiseAlgorithmSpec)
    assert recipe.algorithm.scales == (2, 4)
    student, teacher, critic = _roles()
    stack = build_native_scale_wise_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=critic,
        fused_adamw=False,
    )
    assert stack.recipe is recipe
    assert stack.config.schedule.boundary_indices == (0, 1, 3)
    assert stack.engine.gradient_accumulation_steps == 2
    assert stack.engine.fake_updates_per_iteration == 2
    assert stack.student_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert stack.fake_score_optimizer is not None

    payload = _recipe_mapping()
    payload.pop("fake_score_optimizer")
    with pytest.raises(ValueError, match="requires fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(payload)


def test_scale_wise_builder_rejects_checkpoint_mismatch() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student, teacher, critic = _roles()
    teacher.checkpoint_identity = "wrong"
    with pytest.raises(ValueError, match="teacher loaded checkpoint identity"):
        build_native_scale_wise_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            fake_score=critic,
            fused_adamw=False,
        )


def test_scale_wise_session_and_dcp_resume_exactly(tmp_path: Path) -> None:
    stack, loader, progress, generator, model, state = _checkpointable_stack()
    session = NativeScaleWiseTrainingSession(stack.engine, loader, progress)
    first = session.run(max_steps=1, generator=generator)
    assert first.final_step == 1
    assert first.fake_score_optimizer_steps == 2
    assert progress.microbatches_seen == 6
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)

    expected = session.run(max_steps=1, generator=generator)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_rng = generator.get_state().clone()

    restored, restored_loader, restored_progress, restored_generator, restored_model, restored_state = (
        _checkpointable_stack()
    )
    manager.load(restored_state, artifact.path)
    actual = NativeScaleWiseTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1, generator=restored_generator)

    assert actual.final_student_loss == expected.final_student_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    assert restored_loader.cursor == 12
    assert restored_progress.optimizer_steps == 2
    torch.testing.assert_close(
        restored_generator.get_state(),
        expected_rng,
        rtol=0,
        atol=0,
    )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


def _tiny_sd3_transformer() -> nn.Module:
    diffusers = pytest.importorskip("diffusers")
    return diffusers.SD3Transformer2DModel(
        sample_size=4,
        patch_size=2,
        in_channels=2,
        num_layers=2,
        attention_head_dim=4,
        num_attention_heads=2,
        joint_attention_dim=6,
        caption_projection_dim=8,
        pooled_projection_dim=5,
        out_channels=2,
        pos_embed_max_size=4,
    )


def _sd3_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(47)
    return (
        torch.randn(2, 2, 4, 4, generator=generator),
        torch.randn(2, 3, 6, generator=generator),
        torch.randn(2, 5, generator=generator),
        torch.tensor([0.2, 0.7]),
    )


def test_native_sd3_feature_path_matches_model_forward_and_backpropagates() -> None:
    transformer = _tiny_sd3_transformer()
    latents, prompt, pooled, sigmas = _sd3_inputs()
    timestep = sigmas * 1000.0
    expected = transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt,
        pooled_projections=pooled,
        timestep=timestep,
        return_dict=False,
    )[0]
    velocity, features = sd3_velocity_and_features(
        transformer,
        latents,
        prompt,
        pooled,
        timestep,
        block_indices=(0, 1),
        return_velocity=True,
    )
    assert velocity is not None
    torch.testing.assert_close(velocity, expected)
    assert tuple(feature.shape for feature in features) == ((2, 4, 8),) * 2

    critic_module = SD3ScaleWiseCriticModule(
        transformer,
        discriminator_layers=2,
    )
    critic = SD3ScaleWiseCriticAdapter(
        critic_module,
        checkpoint_identity="fake",
    )
    critic.audit_scale_wise_critic(
        classifier_blocks=(0,),
        mmd_blocks=(1,),
        discriminator_layers=2,
    )
    critic_velocity, critic_features = critic.predict_velocity_and_features(
        latents,
        sigmas,
        sample_ids=("a", "b"),
        conditioning={
            "prompt_embeds": prompt,
            "pooled_prompt_embeds": pooled,
        },
        block_indices=(0, 1),
        training=True,
    )
    logits = critic.classify_features(
        tuple(feature.mean(dim=1) for feature in critic_features)
    )
    assert tuple(value.shape for value in logits) == ((2, 1), (2, 1))
    (critic_velocity.square().mean() + sum(value.mean() for value in logits)).backward()
    assert any(
        parameter.grad is not None
        for parameter in critic_module.transformer.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in critic_module.classifier_head.parameters()
    )


class _AdapterAwareSD3(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.adapter_delta = nn.Parameter(torch.tensor(0.25))
        self._adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        previous = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = previous

    def forward(self, *args, **kwargs):
        velocity = self.transformer(*args, **kwargs)[0]
        if self._adapter_enabled:
            velocity = velocity + self.adapter_delta
        return (velocity,)


def test_shared_sd3_teacher_executes_base_without_exposing_trainable_parameters() -> None:
    module = _AdapterAwareSD3(_tiny_sd3_transformer())
    student = SD3ScaleWisePredictionAdapter(
        module,
        checkpoint_identity="student",
    )
    teacher = SD3AdapterDisabledTeacherAdapter(
        module,
        checkpoint_identity="teacher",
    )
    latents, prompt, pooled, sigmas = _sd3_inputs()
    kwargs = {
        "sample_ids": ("a", "b"),
        "conditioning": {
            "prompt_embeds": prompt,
            "pooled_prompt_embeds": pooled,
        },
        "training": True,
    }
    student_velocity = student.predict_velocity(latents, sigmas, **kwargs)
    teacher_velocity = teacher.predict_velocity(latents, sigmas, **kwargs)
    torch.testing.assert_close(
        student_velocity - teacher_velocity,
        torch.full_like(student_velocity, 0.25),
    )
    assert tuple(teacher.module.parameters()) == ()
    assert module._adapter_enabled is True
