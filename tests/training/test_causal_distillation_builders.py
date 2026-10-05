from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.causal_consistency import (  # noqa: E402
    CausalConsistencyTrainingBatch,
    NativeCausalConsistencyTrainingSession,
    build_native_causal_consistency_training_stack,
)
from worldfoundry.training.post_training.distillation.causal_ode import (  # noqa: E402
    CausalODETrainingBatch,
    NativeCausalODETrainingSession,
    build_native_causal_ode_training_stack,
)
from worldfoundry.training.recipes import (  # noqa: E402
    CausalConsistencyAlgorithmSpec,
    CausalODEAlgorithmSpec,
    PostTrainingRecipe,
)


class _ScaleModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))


class _CleanAdapter:
    def __init__(self, value: float, checkpoint_identity: str) -> None:
        self.module = _ScaleModule(value)
        self.checkpoint_identity = checkpoint_identity

    def predict_clean(
        self,
        noisy_latents,
        timesteps,
        *,
        clean_context,
        sample_ids,
        conditioning,
        training,
    ):
        del timesteps, clean_context, sample_ids, conditioning, training
        return noisy_latents * self.module.scale


class _VelocityAdapter:
    def __init__(self, value: float, checkpoint_identity: str) -> None:
        self.module = _ScaleModule(value)
        self.checkpoint_identity = checkpoint_identity

    def predict_velocity(
        self,
        noisy_latents,
        timesteps,
        *,
        clean_context,
        sample_ids,
        conditioning,
        training,
    ):
        del timesteps, clean_context, sample_ids, training
        bias = torch.as_tensor(
            conditioning["bias"],
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        return noisy_latents * self.module.scale + bias


def _base_mapping(algorithm: dict[str, object]) -> dict[str, object]:
    return {
        "run": {"id": "causal-distillation", "output_dir": "unused"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {
            "manifest": "latents.jsonl",
            "shuffle": True,
            "shuffle_seed": 37,
        },
        "algorithm": algorithm,
        "optimizer": {
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


def _ode_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        _base_mapping(
            {
                "type": "causal-ode",
                "raw_denoising_steps": [1000, 500],
                "num_train_timesteps": 1000,
                "flow_shift": 5.0,
                "extra_terminal_step": True,
                "frame_dim": 2,
            }
        )
    )


def _consistency_recipe() -> PostTrainingRecipe:
    mapping = _base_mapping(
        {
            "type": "causal-consistency",
            "teacher_checkpoint": "teacher",
            "num_levels": 4,
            "num_train_timesteps": 1000,
            "flow_shift": 1.0,
            "extra_terminal_step": True,
            "guidance_scale": 2.0,
            "ema_decay": 0.5,
            "frame_dim": 2,
        }
    )
    mapping["optimizer"]["gradient_accumulation_steps"] = 1  # type: ignore[index]
    return PostTrainingRecipe.from_mapping(mapping)


def _ode_batch(sample_id: str = "ode") -> CausalODETrainingBatch:
    return CausalODETrainingBatch(
        sample_ids=(sample_id,),
        ode_trajectories=torch.stack(
            (
                torch.ones(1, 1, 2, 1, 1),
                torch.full((1, 1, 2, 1, 1), 2.0),
                torch.full((1, 1, 2, 1, 1), 3.0),
            ),
            dim=1,
        ),
        conditioning={"context": torch.ones(1, 2, 3)},
    )


def _consistency_batch(sample_id: str = "consistency") -> CausalConsistencyTrainingBatch:
    return CausalConsistencyTrainingBatch(
        sample_ids=(sample_id,),
        clean_latents=torch.ones(1, 1, 2, 1, 1),
        conditioning={"bias": 2.0},
        unconditional_conditioning={"bias": 1.0},
    )


def test_causal_recipe_builders_consume_schedule_seed_and_role_identities() -> None:
    ode_recipe = _ode_recipe()
    assert isinstance(ode_recipe.algorithm, CausalODEAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(ode_recipe.to_dict()) == ode_recipe
    ode_student = _CleanAdapter(0.5, "student")
    ode = build_native_causal_ode_training_stack(
        ode_recipe,
        student=ode_student,
        fused_adamw=False,
    )
    assert ode.config.trajectory_timesteps == pytest.approx(
        (1000.0, 833.3333129882812)
    )
    assert ode.engine.gradient_accumulation_steps == 2
    expected_rng = torch.Generator().manual_seed(37)
    torch.testing.assert_close(
        ode.engine._rng.get_state(),
        expected_rng.get_state(),
    )

    consistency_recipe = _consistency_recipe()
    assert isinstance(
        consistency_recipe.algorithm,
        CausalConsistencyAlgorithmSpec,
    )
    student = _CleanAdapter(0.5, "student")
    teacher = _VelocityAdapter(0.25, "teacher")
    ema = _CleanAdapter(-4.0, "student")
    consistency = build_native_causal_consistency_training_stack(
        consistency_recipe,
        student=student,
        teacher=teacher,
        ema_student=ema,
        fused_adamw=False,
    )
    assert consistency.config.guidance_scale == 2.0
    assert consistency.config.ema_decay == 0.5
    torch.testing.assert_close(ema.module.scale, student.module.scale)
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in ema.module.parameters())

    teacher.checkpoint_identity = "wrong"
    with pytest.raises(ValueError, match="differs from recipe"):
        build_native_causal_consistency_training_stack(
            consistency_recipe,
            student=_CleanAdapter(0.5, "student"),
            teacher=teacher,
            ema_student=_CleanAdapter(0.5, "student"),
            fused_adamw=False,
        )


def test_shared_single_optimizer_session_preserves_algorithm_batch_boundaries() -> None:
    ode = build_native_causal_ode_training_stack(
        _ode_recipe(),
        student=_CleanAdapter(0.5, "student"),
        fused_adamw=False,
    )
    progress = TrainingProgress()
    events: list[dict[str, object]] = []
    session = NativeCausalODETrainingSession(
        ode.engine,
        [_ode_batch("first"), _ode_batch("second")],
        progress,
        event_sink=events.append,
    )
    summary = session.run(max_steps=1)
    assert summary.final_step == 1
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 2
    assert progress.latent_tokens_seen == 4
    assert events[0]["schema"] == "worldfoundry-causal-ode-step-event"


class _StatefulConsistencyLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = _consistency_batch(f"sample-{self.cursor}")
        self.cursor += 1
        return value

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_consistency_stack():
    student = _CleanAdapter(0.5, "student")
    teacher = _VelocityAdapter(0.25, "teacher")
    ema = _CleanAdapter(-4.0, "student")
    stack = build_native_causal_consistency_training_stack(
        _consistency_recipe(),
        student=student,
        teacher=teacher,
        ema_student=ema,
        fused_adamw=False,
    )
    loader = _StatefulConsistencyLoader()
    progress = TrainingProgress()
    model = torch.nn.ModuleDict(
        {
            "student": student.module,
            "teacher": teacher.module,
            "ema_student": ema.module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=torch.Generator().manual_seed(91),
        progress=progress,
        identity={"algorithm": "causal-consistency"},
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, model, state


def test_causal_consistency_dcp_restores_model_ema_optimizer_rng_and_cursor(
    tmp_path: Path,
) -> None:
    stack, loader, progress, model, state = _checkpointable_consistency_stack()
    session = NativeCausalConsistencyTrainingSession(stack.engine, loader, progress)
    session.run(max_steps=1)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    restored, restored_loader, restored_progress, restored_model, restored_state = (
        _checkpointable_consistency_stack()
    )
    manager.load(restored_state, artifact.path)
    actual = NativeCausalConsistencyTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1)

    assert actual.final_loss == expected.final_loss
    assert restored_loader.cursor == 2
    assert restored_progress.optimizer_steps == 2
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
