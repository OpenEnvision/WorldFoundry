from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdata")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.adversarial_diffusion import (  # noqa: E402
    ADDConfig,
    ADDNoiseSchedule,
    ADDTrainingBatch,
    MultiScaleFeatureDiscriminator,
    NativeADDDiscriminatorAdapter,
    NativeADDLossAdapter,
    NativeADDTrainingSession,
    ProjectionFeatureHead,
    build_native_add_training_stack,
)
from worldfoundry.training.recipes.post_training.algorithms.adversarial_diffusion import (  # noqa: E402
    AdversarialDiffusionAlgorithmSpec,
    parse_adversarial_diffusion_algorithm,
)
from worldfoundry.training.recipes.post_training.recipe import (  # noqa: E402
    PostTrainingRecipe,
)


class _PredictionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = timesteps.to(dtype=noisy.dtype)[:, None, None, None] * 0.001
        return self.projection(noisy) + scale


class _PredictionAdapter:
    def __init__(
        self,
        module: torch.nn.Module,
        checkpoint_identity: str,
    ) -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity

    def predict_clean(
        self,
        noisy_latents,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del sample_ids, conditioning, training
        return self.module(noisy_latents, timesteps)


class _DecoderModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.projection(latents)


class _DecoderAdapter:
    def __init__(
        self,
        module: torch.nn.Module,
        checkpoint_identity: str,
    ) -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity

    def decode(self, clean_latents):
        return self.module(clean_latents)


class _FeatureNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(4, 4, kernel_size=1),
                torch.nn.Conv2d(4, 4, kernel_size=1),
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.silu(self.stem(images))
        for block in self.blocks:
            value = torch.nn.functional.silu(block(value))
        return value


def _algorithm_payload(*, discriminator_updates: int = 1) -> dict[str, object]:
    return {
        "type": "adversarial-diffusion-distillation",
        "teacher_checkpoint": "teacher",
        "decoder_checkpoint": "decoder",
        "feature_checkpoint": "feature",
        "student_alpha_cumprods": [1.0, 0.9, 0.65, 0.3, 0.0],
        "teacher_alpha_cumprods": [1.0, 0.85, 0.5, 0.2],
        "student_timesteps": [1, 2, 3, 4],
        "teacher_timestep_min": 1,
        "teacher_timestep_max": 3,
        "feature_resolutions": [4, 8],
        "feature_layers": ["blocks.0", "blocks.1"],
        "discriminator_conditioning_keys": ["text", "image"],
        "discriminator_updates_per_generator": discriminator_updates,
    }


def _recipe(
    *,
    accumulation: int = 1,
    discriminator_updates: int = 1,
    algorithm: dict[str, object] | None = None,
    optimizer: dict[str, object] | None = None,
    discriminator_optimizer: dict[str, object] | None = None,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "add-test", "output_dir": "unused"},
            "model": {
                "recipe": "test-diffusion-model",
                "checkpoint": "student",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "add.jsonl", "shuffle_seed": 17},
            "algorithm": (
                _algorithm_payload(discriminator_updates=discriminator_updates) if algorithm is None else algorithm
            ),
            "optimizer": (
                {
                    "type": "adamw",
                    "learning_rate": 2.0e-3,
                    "max_grad_norm": 1.0,
                    "gradient_accumulation_steps": accumulation,
                }
                if optimizer is None
                else optimizer
            ),
            "discriminator_optimizer": (
                {
                    "type": "adamw",
                    "learning_rate": 3.0e-3,
                    "max_grad_norm": 1.0,
                    "gradient_accumulation_steps": accumulation,
                }
                if discriminator_optimizer is None
                else discriminator_optimizer
            ),
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
            },
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def _batch(
    offset: float = 0.0,
    *,
    prefix: str = "sample",
    device: str | torch.device = "cpu",
) -> ADDTrainingBatch:
    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 8 * 8, device=device).reshape(2, 3, 8, 8) + offset
    real = torch.tanh(clean * 0.75)
    return ADDTrainingBatch(
        sample_ids=(f"{prefix}-0", f"{prefix}-1"),
        clean_latents=clean,
        real_images=real,
        conditioning={},
        discriminator_conditioning={
            "text": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device) + offset,
            "image": torch.tensor(
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
                device=device,
            )
            + offset,
        },
    )


def _build(
    *,
    seed: int,
    accumulation: int = 1,
    discriminator_updates: int = 1,
    device: str | torch.device = "cpu",
    algorithm: dict[str, object] | None = None,
    optimizer: dict[str, object] | None = None,
    discriminator_optimizer: dict[str, object] | None = None,
    student_scheduler_factory=None,
    discriminator_scheduler_factory=None,
):
    torch.manual_seed(seed)
    student = _PredictionAdapter(_PredictionModule().to(device), "student")
    teacher_module = _PredictionModule().to(device).requires_grad_(False)
    teacher = _PredictionAdapter(teacher_module, "teacher")
    decoder_module = _DecoderModule().to(device).requires_grad_(False)
    decoder = _DecoderAdapter(decoder_module, "decoder")
    feature_network = _FeatureNetwork().to(device).requires_grad_(False)
    recipe = _recipe(
        accumulation=accumulation,
        discriminator_updates=discriminator_updates,
        algorithm=algorithm,
        optimizer=optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    assert isinstance(recipe.algorithm, AdversarialDiffusionAlgorithmSpec)
    config = ADDConfig.from_recipe(recipe.algorithm)
    heads = {
        key: ProjectionFeatureHead(
            feature_dim=4,
            hidden_dim=5,
            feature_layout="channels-first",
            conditioning_dims={"text": 2, "image": 3},
        )
        for key in config.feature_keys
    }
    discriminator_graph = MultiScaleFeatureDiscriminator(
        feature_network=feature_network,
        heads=heads,
        feature_resolutions=config.feature_resolutions,
        feature_layers=config.feature_layers,
        conditioning_keys=config.discriminator_conditioning_keys,
    )
    discriminator = NativeADDDiscriminatorAdapter(
        discriminator_graph.to(device),
        checkpoint_identity="feature",
    )
    stack = build_native_add_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        decoder=decoder,
        discriminator=discriminator,
        student_scheduler_factory=student_scheduler_factory,
        discriminator_scheduler_factory=discriminator_scheduler_factory,
        fused_adamw=False,
    )
    return (
        stack,
        student,
        teacher,
        decoder,
        discriminator,
        stack.student_schedule,
        stack.teacher_schedule,
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], value) for name, value in module.state_dict().items())


def _unchanged(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return all(torch.equal(before[name], value) for name, value in module.state_dict().items())


def test_add_algorithm_parser_rejects_inert_fields_and_binds_every_schedule() -> None:
    payload = _algorithm_payload(discriminator_updates=2)
    payload.update(
        {
            "teacher_training_loss_weights": [1.0, 2.0, 3.0, 4.0],
            "teacher_timestep_probabilities": [1.0, 2.0, 1.0],
            "distillation_weight": 3.25,
            "distillation_weighting": "sds",
            "r1_weight": 2.0e-5,
        }
    )

    spec = parse_adversarial_diffusion_algorithm(payload)

    assert spec.type == "adversarial-diffusion-distillation"
    assert spec.student_alpha_cumprods == (1.0, 0.9, 0.65, 0.3, 0.0)
    assert spec.teacher_alpha_cumprods == (1.0, 0.85, 0.5, 0.2)
    assert spec.teacher_training_loss_weights == (1.0, 2.0, 3.0, 4.0)
    assert spec.teacher_timestep_probabilities == (0.25, 0.5, 0.25)
    assert spec.distillation_weight == 3.25
    assert spec.distillation_weighting == "sds"
    assert spec.r1_weight == 2.0e-5
    assert spec.discriminator_updates_per_generator == 2

    with pytest.raises(ValueError, match="unknown fields.*metadata"):
        parse_adversarial_diffusion_algorithm({**payload, "metadata": {"unused": True}})
    with pytest.raises(ValueError, match="unused by exponential ADD"):
        parse_adversarial_diffusion_algorithm({**payload, "distillation_weighting": "exponential"})
    with pytest.raises(ValueError, match="zero terminal SNR"):
        parse_adversarial_diffusion_algorithm(
            {
                **_algorithm_payload(),
                "student_alpha_cumprods": [1.0, 0.9, 0.65, 0.3, 0.1],
            }
        )


def test_add_facades_do_not_eager_load_the_execution_graph() -> None:
    leaf = "worldfoundry.training.post_training.distillation.adversarial_diffusion"
    root = Path(__file__).resolve().parents[2]
    for imported in (
        "worldfoundry.training.post_training",
        "worldfoundry.training.post_training.distillation",
        leaf,
    ):
        probe = f"""
import importlib
import sys

importlib.import_module({imported!r})
print("\\n".join(sorted(
    name for name in sys.modules if name.startswith({leaf!r} + ".")
)))
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""


def test_add_recipe_builds_all_objective_optimizer_and_scheduler_state() -> None:
    algorithm = _algorithm_payload(discriminator_updates=2)
    algorithm.update(
        {
            "teacher_training_loss_weights": [1.0, 1.5, 2.0, 2.5],
            "teacher_timestep_probabilities": [2.0, 3.0, 5.0],
            "distillation_weight": 3.5,
            "distillation_weighting": "sds",
            "r1_weight": 4.0e-5,
        }
    )
    student_optimizer = {
        "type": "adamw",
        "learning_rate": 4.0e-3,
        "weight_decay": 0.03,
        "betas": [0.8, 0.91],
        "epsilon": 2.0e-7,
        "max_grad_norm": 3.0,
        "gradient_accumulation_steps": 2,
    }
    discriminator_optimizer = {
        "type": "adamw",
        "learning_rate": 5.0e-3,
        "weight_decay": 0.04,
        "betas": [0.7, 0.92],
        "epsilon": 3.0e-7,
        "max_grad_norm": 4.0,
        "gradient_accumulation_steps": 2,
    }
    stack, *_ = _build(
        seed=5,
        accumulation=2,
        algorithm=algorithm,
        optimizer=student_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        student_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.8,
        ),
        discriminator_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.7,
        ),
    )

    assert isinstance(stack.recipe.algorithm, AdversarialDiffusionAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(stack.recipe.to_dict()) == stack.recipe
    assert stack.config == ADDConfig.from_recipe(stack.recipe.algorithm)
    assert stack.student_schedule.alpha_cumprods == (
        1.0,
        0.9,
        0.65,
        0.3,
        0.0,
    )
    assert stack.teacher_schedule.training_loss_weights == (1.0, 1.5, 2.0, 2.5)
    assert stack.student_optimizer.param_groups[0]["lr"] == 4.0e-3
    assert stack.student_optimizer.param_groups[0]["weight_decay"] == 0.03
    assert stack.student_optimizer.param_groups[0]["betas"] == (0.8, 0.91)
    assert stack.student_optimizer.param_groups[0]["eps"] == 2.0e-7
    assert stack.discriminator_optimizer.param_groups[0]["lr"] == 5.0e-3
    assert stack.discriminator_optimizer.param_groups[0]["weight_decay"] == 0.04
    assert stack.discriminator_optimizer.param_groups[0]["betas"] == (0.7, 0.92)
    assert stack.discriminator_optimizer.param_groups[0]["eps"] == 3.0e-7
    assert stack.engine.student_max_grad_norm == 3.0
    assert stack.engine.discriminator_max_grad_norm == 4.0
    assert stack.engine.gradient_accumulation_steps == 2
    assert stack.engine.discriminator_updates_per_generator == 2
    assert stack.scheduler_state is not None
    assert stack.scheduler_state.component_names == ("discriminator", "student")
    assert stack.scheduler_state.components["student"].optimizer is stack.student_optimizer
    assert stack.scheduler_state.components["discriminator"].optimizer is stack.discriminator_optimizer
    checkpoint_kwargs = stack.checkpoint_state_kwargs()
    assert checkpoint_kwargs["model"] is stack.checkpoint_model
    assert checkpoint_kwargs["optimizer"] == stack.optimizers
    assert checkpoint_kwargs["engine"] is stack.engine
    assert checkpoint_kwargs["lr_scheduler"] is stack.scheduler_state
    assert checkpoint_kwargs["ignore_frozen_parameters"] is True


def test_add_builder_gates_role_identities_and_optimizer_cadence() -> None:
    with pytest.raises(ValueError, match="accumulation steps must match"):
        _build(
            seed=29,
            accumulation=2,
            discriminator_optimizer={
                "type": "adamw",
                "learning_rate": 3.0e-3,
                "gradient_accumulation_steps": 1,
            },
        )

    foreign_parameter = torch.nn.Parameter(torch.tensor(1.0))
    foreign_optimizer = torch.optim.AdamW([foreign_parameter], lr=1.0e-3)
    with pytest.raises(ValueError, match="bound to a different optimizer"):
        _build(
            seed=30,
            student_scheduler_factory=lambda _optimizer: torch.optim.lr_scheduler.StepLR(
                foreign_optimizer,
                step_size=1,
            ),
        )

    for selected_index in range(4):
        built = _build(seed=31 + selected_index)
        stack, student, teacher, decoder, discriminator, *_ = built
        selected = (student, teacher, decoder, discriminator)[selected_index]
        selected.checkpoint_identity = "wrong"
        with pytest.raises(ValueError, match="loaded checkpoint identity"):
            build_native_add_training_stack(
                stack.recipe,
                student=student,
                teacher=teacher,
                decoder=decoder,
                discriminator=discriminator,
                fused_adamw=False,
            )


def test_native_feature_graph_executes_every_scale_layer_and_conditioning_path() -> None:
    stack, _, _, _, discriminator, _, _ = _build(seed=7)
    batch = _batch()
    output = discriminator.predict(
        batch.real_images,
        sample_ids=batch.sample_ids,
        conditioning=batch.discriminator_conditioning,
        track_image_grad=False,
        require_r1_inputs=True,
    )

    assert output.keys == stack.config.feature_keys
    assert len(output.heads) == 4
    assert all(head.features.is_leaf and head.features.requires_grad for head in output.heads)
    assert all(head.logits.shape[0] == batch.batch_size for head in output.heads)
    with pytest.raises(ValueError, match="conditioning inventory"):
        discriminator.predict(
            batch.real_images,
            sample_ids=batch.sample_ids,
            conditioning={"text": batch.discriminator_conditioning["text"]},
            track_image_grad=False,
            require_r1_inputs=False,
        )


def test_add_engine_accumulates_and_commits_n_discriminators_then_one_student() -> None:
    stack, student, teacher, decoder, discriminator, _, _ = _build(
        seed=11,
        accumulation=2,
        discriminator_updates=2,
    )
    student_before = _state(student.module)
    discriminator_before = _state(discriminator.module)
    teacher_before = _state(teacher.module)
    decoder_before = _state(decoder.module)
    feature_before = _state(discriminator.feature_module)
    batches = (_batch(0.0, prefix="a"), _batch(0.1, prefix="b"))
    phase_calls: list[str] = []
    discriminator_loss = stack.loss_adapter.discriminator_loss
    generator_loss = stack.loss_adapter.generator_loss

    def recorded_discriminator_loss(*args, **kwargs):
        phase_calls.append("discriminator")
        return discriminator_loss(*args, **kwargs)

    def recorded_generator_loss(*args, **kwargs):
        phase_calls.append("generator")
        return generator_loss(*args, **kwargs)

    stack.loss_adapter.discriminator_loss = recorded_discriminator_loss
    stack.loss_adapter.generator_loss = recorded_generator_loss

    result = stack.engine.train_step(
        batches,
        generator=torch.Generator().manual_seed(101),
    )

    assert result.generator_loss.isfinite()
    assert result.discriminator_loss.isfinite()
    assert stack.engine.global_step == 1
    assert stack.engine.student_optimizer_steps == 1
    assert stack.engine.discriminator_optimizer_steps == 2
    assert result.metrics["accumulated_microbatches"] == 2
    assert result.metrics["discriminator_updates"] == 2
    assert phase_calls == ["discriminator"] * 4 + ["generator"] * 2
    assert _changed(student_before, student.module)
    assert _changed(discriminator_before, discriminator.module)
    assert _unchanged(teacher_before, teacher.module)
    assert _unchanged(decoder_before, decoder.module)
    assert _unchanged(feature_before, discriminator.feature_module)
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in decoder.module.parameters())
    assert all(parameter.grad is None for parameter in discriminator.feature_module.parameters())
    assert all(parameter.grad is None for parameter in student.module.parameters())
    assert all(parameter.grad is None for parameter in discriminator.module.parameters())


def test_add_supports_the_report_unconditioned_discriminator_ablation() -> None:
    config = ADDConfig(
        student_timesteps=(1, 2, 3, 4),
        teacher_timestep_min=1,
        teacher_timestep_max=3,
        feature_resolutions=(8,),
        feature_layers=("blocks.0",),
        discriminator_conditioning_keys=(),
    )
    feature_network = _FeatureNetwork().requires_grad_(False)
    graph = MultiScaleFeatureDiscriminator(
        feature_network=feature_network,
        heads={
            config.feature_keys[0]: ProjectionFeatureHead(
                feature_dim=4,
                hidden_dim=5,
                feature_layout="channels-first",
                conditioning_dims={},
            )
        },
        feature_resolutions=config.feature_resolutions,
        feature_layers=config.feature_layers,
        conditioning_keys=config.discriminator_conditioning_keys,
    )
    adapter = NativeADDDiscriminatorAdapter(
        graph,
        checkpoint_identity="feature",
    )
    batch = _batch()

    output = adapter.predict(
        batch.real_images,
        sample_ids=batch.sample_ids,
        conditioning={},
        track_image_grad=False,
        require_r1_inputs=False,
    )

    assert output.keys == config.feature_keys
    assert output.heads[0].logits.shape[0] == batch.batch_size


def test_add_model_graph_fails_closed_on_unverifiable_roles_and_schedules() -> None:
    stack, student, teacher, decoder, discriminator, _, teacher_schedule = _build(seed=13)
    with pytest.raises(ValueError, match="zero terminal SNR"):
        NativeADDLossAdapter(
            student=student,
            teacher=teacher,
            decoder=decoder,
            discriminator=discriminator,
            student_schedule=ADDNoiseSchedule((1.0, 0.9, 0.7, 0.4, 0.1)),
            teacher_schedule=teacher_schedule,
            config=stack.config,
        )
    teacher.module.requires_grad_(True)
    with pytest.raises(ValueError, match="teacher must be frozen"):
        NativeADDLossAdapter(
            student=student,
            teacher=teacher,
            decoder=decoder,
            discriminator=discriminator,
            student_schedule=ADDNoiseSchedule((1.0, 0.9, 0.7, 0.4, 0.0)),
            teacher_schedule=teacher_schedule,
            config=stack.config,
        )
    with pytest.raises(TypeError, match="MultiScaleFeatureDiscriminator"):
        NativeADDDiscriminatorAdapter(
            torch.nn.Linear(2, 1),
            checkpoint_identity="feature",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_add_native_graph_executes_real_cuda_forward_and_backward() -> None:
    stack, student, teacher, decoder, discriminator, _, _ = _build(
        seed=29,
        device="cuda",
    )
    result = stack.engine.train_step(
        _batch(device="cuda"),
        generator=torch.Generator(device="cuda").manual_seed(31),
    )

    assert result.generator_loss.device.type == "cuda"
    assert result.discriminator_loss.device.type == "cuda"
    assert result.generator_loss.isfinite()
    assert result.discriminator_loss.isfinite()
    assert all(parameter.grad is None for parameter in student.module.parameters())
    assert all(parameter.grad is None for parameter in discriminator.module.parameters())
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert all(parameter.grad is None for parameter in decoder.module.parameters())


def test_add_rejects_inert_student_schedule_weights_and_shared_role_parameters() -> None:
    stack, student, teacher, decoder, discriminator, _, teacher_schedule = _build(seed=19)
    with pytest.raises(ValueError, match="student training_loss_weights"):
        NativeADDLossAdapter(
            student=student,
            teacher=teacher,
            decoder=decoder,
            discriminator=discriminator,
            student_schedule=ADDNoiseSchedule(
                (1.0, 0.9, 0.65, 0.3, 0.0),
                training_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            teacher_schedule=teacher_schedule,
            config=stack.config,
        )

    teacher.module.projection.weight = student.module.projection.weight
    teacher.module.requires_grad_(False)
    with pytest.raises(ValueError, match="cannot share parameters"):
        NativeADDLossAdapter(
            student=student,
            teacher=teacher,
            decoder=decoder,
            discriminator=discriminator,
            student_schedule=ADDNoiseSchedule((1.0, 0.9, 0.65, 0.3, 0.0)),
            teacher_schedule=teacher_schedule,
            config=stack.config,
        )


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def next_batch(self) -> ADDTrainingBatch:
        self.cursor += 1
        return _batch(self.cursor * 0.05, prefix=f"cursor-{self.cursor}")

    def __iter__(self):
        return self

    def __next__(self) -> ADDTrainingBatch:
        return self.next_batch()

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable(seed: int):
    stack, *_ = _build(seed=seed)
    loader = _StatefulLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(307)
    state = TrainingState(
        **stack.checkpoint_state_kwargs(),
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={
            "algorithm": "adversarial-diffusion-distillation",
            "gradient_accumulation_steps": stack.engine.gradient_accumulation_steps,
        },
    )
    return stack, loader, progress, generator, state


def _step(stack, loader, progress, generator):
    batch = loader.next_batch()
    result = stack.engine.train_step(batch, generator=generator)
    progress.record_step(
        microbatches=1,
        samples=batch.batch_size,
        latent_tokens=batch.clean_latents.numel(),
    )
    return batch, result


def test_add_dcp_resume_restores_both_roles_optimizers_rng_loader_and_cadence(tmp_path: Path) -> None:
    baseline = _checkpointable(17)
    stack, loader, progress, generator, state = baseline
    _step(stack, loader, progress, generator)
    manager = TrainingCheckpointer(tmp_path / "add-checkpoints")
    artifact = manager.save(state)

    expected_batch, expected_result = _step(stack, loader, progress, generator)
    expected_parameters = _state(stack.checkpoint_model)
    expected_generator_state = generator.get_state().clone()

    restored = _checkpointable(17)
    restored_stack, restored_loader, restored_progress, restored_generator, restored_state = restored
    manager.load(restored_state, artifact.path)
    actual_batch, actual_result = _step(
        restored_stack,
        restored_loader,
        restored_progress,
        restored_generator,
    )

    assert actual_batch.sample_ids == expected_batch.sample_ids
    assert restored_loader.cursor == 2
    assert restored_progress.optimizer_steps == 2
    assert restored_stack.engine.global_step == 2
    assert restored_stack.engine.student_optimizer_steps == 2
    assert restored_stack.engine.discriminator_optimizer_steps == 2
    torch.testing.assert_close(actual_result.generator_loss, expected_result.generator_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_result.discriminator_loss,
        expected_result.discriminator_loss,
        rtol=0,
        atol=0,
    )
    assert torch.equal(restored_generator.get_state(), expected_generator_state)
    for name, value in restored_stack.checkpoint_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


def test_add_from_mapping_builder_session_commits_complete_iterations_and_checkpoints(
    tmp_path: Path,
) -> None:
    stack, loader, progress, generator, state = _checkpointable(23)
    manager = TrainingCheckpointer(tmp_path / "add-session-checkpoints")
    session = NativeADDTrainingSession(
        stack.engine,
        loader,
        progress,
        checkpoint_state=state,
        checkpointer=manager,
        save_every_steps=1,
    )

    summary = session.run(max_steps=2, generator=generator)

    assert summary.initial_step == 0
    assert summary.final_step == 2
    assert summary.iterations == 2
    assert summary.student_optimizer_steps == 2
    assert summary.discriminator_optimizer_steps == 2
    assert progress.optimizer_steps == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 4
    assert progress.latent_tokens_seen == 256
    assert manager.inspect(manager.root / "step-00000002").global_step == 2
