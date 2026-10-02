from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.distillation.scm_ladd.builder import (
    build_native_scm_ladd_training_stack,
)
from worldfoundry.training.post_training.distillation.scm_ladd.contracts import (
    SCMLADDLossResult,
    SCMLADDTrainingBatch,
    SCMVelocityPrediction,
)
from worldfoundry.training.post_training.distillation.scm_ladd.engine import (
    NativeSCMLADDTrainEngine,
)
from worldfoundry.training.post_training.distillation.scm_ladd.objective import (
    NativeSCMLADDLossAdapter,
)
from worldfoundry.training.post_training.distillation.scm_ladd.session import (
    NativeSCMLADDTrainingSession,
)
from worldfoundry.training.recipes import (
    PostTrainingRecipe,
    SCMLADDAlgorithmSpec,
)


class _VelocityModule(nn.Module):
    def __init__(self, *, weight: float, time_weight: float, log_variance: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.time_weight = nn.Parameter(torch.tensor(time_weight))
        self.log_variance = nn.Parameter(torch.tensor(log_variance))

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        offset: torch.Tensor,
        guidance_embedding_scale: float,
    ) -> torch.Tensor:
        dimensions = (1,) * (noisy.ndim - 1)
        time = timesteps.reshape(timesteps.shape[0], *dimensions)
        condition = offset.reshape(offset.shape[0], *dimensions)
        return self.weight * noisy + self.time_weight * time + guidance_embedding_scale * condition


class _VelocityAdapter:
    def __init__(self, module: _VelocityModule, *, checkpoint_identity: str = "default") -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity
        self.guidance_embedding_scales: list[float] = []

    def predict_velocity(
        self,
        scaled_noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        guidance_embedding_scale: float,
        return_log_variance: bool = False,
        branch: str = "positive",
    ) -> SCMVelocityPrediction:
        del sample_ids, training, branch
        self.guidance_embedding_scales.append(float(guidance_embedding_scale))
        offset = conditioning["offset"]
        assert isinstance(offset, torch.Tensor)
        velocity = self.module(
            scaled_noisy_latents,
            trig_timesteps,
            offset,
            guidance_embedding_scale,
        )
        log_variance = (
            self.module.log_variance.expand(scaled_noisy_latents.shape[0])
            if return_log_variance
            else None
        )
        return SCMVelocityPrediction(velocity=velocity, log_variance=log_variance)


class _DiscriminatorModule(nn.Module):
    def __init__(self, feature_module: _VelocityModule) -> None:
        super().__init__()
        self.feature_module = feature_module
        self.head_scale = nn.Parameter(torch.tensor(0.25))


class _DiscriminatorAdapter:
    def __init__(self, module: _DiscriminatorModule) -> None:
        self.module = module
        self.feature_module = module.feature_module
        self.head_block_ids = (2, 8, 14, 19)
        self.calls: list[dict[str, object]] = []

    def predict_logits(
        self,
        scaled_noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        head_block_ids: tuple[int, ...],
    ) -> torch.Tensor:
        if head_block_ids != self.head_block_ids:
            raise ValueError("wrong head blocks")
        offset = conditioning["offset"]
        assert isinstance(offset, torch.Tensor)
        self.calls.append(
            {
                "latents": scaled_noisy_latents.detach().clone(),
                "timesteps": trig_timesteps.detach().clone(),
                "sample_ids": sample_ids,
                "offset": offset.detach().clone(),
                "training": training,
            }
        )
        features = self.feature_module(
            scaled_noisy_latents,
            trig_timesteps,
            offset,
            guidance_embedding_scale=0.1,
        )
        return self.module.head_scale * features.flatten(1).mean(dim=1, keepdim=True)


class _FailingScheduler:
    def step(self) -> None:
        raise RuntimeError("scheduler failure")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.steps = int(state_dict["steps"])


class _WeightedSCMLADDLosses:
    def __init__(self, student: nn.Linear, discriminator: nn.Linear) -> None:
        self.student = student
        self.discriminator = discriminator

    def loss_denominator(self, batch: SCMLADDTrainingBatch, *, role: str) -> torch.Tensor:
        if role not in {"generator", "discriminator"}:
            raise ValueError(role)
        values = batch.clean_latents
        assert isinstance(values, torch.Tensor)
        return torch.tensor(values.numel(), device=values.device, dtype=torch.float32)

    def generator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        training_iteration: int,
        generator: torch.Generator | None = None,
    ):
        del training_iteration, generator
        values = batch.clean_latents
        assert isinstance(values, torch.Tensor)
        loss = (self.student(values) - values * 0.25).square().mean()
        denominator = torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
        return SCMLADDLossResult(loss, {"loss_denominator": denominator})

    def discriminator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ):
        del generator
        values = batch.clean_latents
        assert isinstance(values, torch.Tensor)
        loss = (self.discriminator(values) + values * 0.2).square().mean()
        denominator = torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
        return SCMLADDLossResult(loss, {"loss_denominator": denominator})


def _recipe_mapping(
    *,
    accumulation_steps: int = 1,
    **algorithm_overrides: object,
) -> dict[str, object]:
    algorithm = {
        "type": "scm-ladd",
        "lr_scheduler": "constant-with-warmup",
        "lr_warmup_steps": 5000,
        "student_fp32_attention": True,
        "teacher_fp32_attention": False,
        **algorithm_overrides,
    }
    optimizer = {
        "type": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.99],
        "epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": accumulation_steps,
    }
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "sana-sprint-scm-ladd", "output_dir": "runs/sana-sprint-scm-ladd"},
        "model": {"recipe": "sana-sprint-1600m-1024px", "checkpoint": "default"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/post_training/sana-sprint-latents.jsonl"},
        "algorithm": algorithm,
        "optimizer": dict(optimizer),
        "discriminator_optimizer": {**optimizer, "learning_rate": 2.0e-3},
        "runtime": {"param_dtype": "bfloat16", "reduce_dtype": "float32", "compile": False},
        "export": {"format": "distributed-checkpoint"},
    }


def _roles() -> tuple[_VelocityAdapter, _VelocityAdapter, _DiscriminatorAdapter]:
    student = _VelocityAdapter(_VelocityModule(weight=0.2, time_weight=-0.1, log_variance=0.05))
    teacher_module = _VelocityModule(weight=0.35, time_weight=0.15, log_variance=0.0)
    teacher_module.requires_grad_(False)
    teacher = _VelocityAdapter(teacher_module)
    discriminator = _DiscriminatorAdapter(_DiscriminatorModule(teacher_module))
    return student, teacher, discriminator


def _batch() -> SCMLADDTrainingBatch:
    return SCMLADDTrainingBatch(
        sample_ids=("sample-a", "sample-b"),
        clean_latents=torch.tensor([[[[0.2, -0.4]]], [[[0.6, 0.1]]]], dtype=torch.float32),
        conditioning={"offset": torch.tensor([0.1, 0.2])},
        unconditional_conditioning={"offset": torch.tensor([-0.1, -0.2])},
    )


def _weighted_batch(values: list[float], *, prefix: str) -> SCMLADDTrainingBatch:
    return SCMLADDTrainingBatch(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(len(values))),
        clean_latents=torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
        conditioning={},
        unconditional_conditioning={},
    )


def _weighted_engine(*, seed: int, accumulation_steps: int):
    torch.manual_seed(seed)
    student = nn.Linear(1, 1, bias=False)
    discriminator = nn.Linear(1, 1, bias=False)
    teacher = nn.Linear(1, 1, bias=False).requires_grad_(False)
    student_scheduler = _Counter()
    discriminator_scheduler = _Counter()
    engine = NativeSCMLADDTrainEngine(
        student_module=student,
        teacher_module=teacher,
        discriminator_module=discriminator,
        discriminator_feature_module=teacher,
        loss_adapter=_WeightedSCMLADDLosses(student, discriminator),
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        discriminator_optimizer=torch.optim.SGD(discriminator.parameters(), lr=0.05),
        student_max_grad_norm=1000.0,
        discriminator_max_grad_norm=1000.0,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        discriminator_scheduler=discriminator_scheduler,
    )
    return engine, student_scheduler, discriminator_scheduler


def test_scm_ladd_recipe_is_strict_and_consumes_behavior_fields() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert recipe.algorithm.discriminator_head_block_ids == (2, 8, 14, 19)
    assert recipe.algorithm.lr_scheduler == "constant-with-warmup"
    assert recipe.algorithm.lr_warmup_steps == 5000
    assert recipe.algorithm.student_fp32_attention is True
    assert recipe.algorithm.teacher_fp32_attention is False
    serialized = recipe.to_dict()["algorithm"]
    assert isinstance(serialized, dict)
    assert serialized["lr_warmup_steps"] == 5000
    assert serialized["student_fp32_attention"] is True
    assert serialized["teacher_fp32_attention"] is False
    assert recipe.optimizer.learning_rate == 1.0e-3
    assert recipe.discriminator_optimizer is not None
    assert recipe.discriminator_optimizer.learning_rate == 2.0e-3
    changed = PostTrainingRecipe.from_mapping(_recipe_mapping(guidance_embedding_scale=0.2))
    assert changed.algorithm != recipe.algorithm
    unknown = _recipe_mapping()
    unknown["algorithm"]["unused_label"] = "dead"
    with pytest.raises(ValueError, match="unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)
    with pytest.raises(ValueError, match="adversarial_loss"):
        PostTrainingRecipe.from_mapping(_recipe_mapping(adversarial_loss="cross-entropy"))
    with pytest.raises(ValueError, match="lr_scheduler"):
        PostTrainingRecipe.from_mapping(_recipe_mapping(lr_scheduler="cosine"))


def test_scm_ladd_discoverable_stack_recipe_uses_public_parser() -> None:
    pytest.importorskip("yaml")
    recipe_path = (
        Path(__file__).parent
        / "fixtures"
        / "recipes"
        / "sana_sprint_scm_ladd_stack.yaml"
    )
    recipe = PostTrainingRecipe.from_file(recipe_path)

    assert isinstance(recipe.algorithm, SCMLADDAlgorithmSpec)
    assert recipe.optimizer.gradient_accumulation_steps == 2
    assert recipe.discriminator_optimizer is not None
    assert recipe.discriminator_optimizer.gradient_accumulation_steps == 2


@pytest.mark.parametrize(
    "name,model_id",
    (
        ("sana_sprint_600m_scm_ladd.yaml", "sana-sprint-600m-1024px"),
        ("sana_sprint_1600m_scm_ladd.yaml", "sana-sprint-1600m-1024px"),
    ),
)
def test_official_sana_sprint_profiles_parse_with_came(name: str, model_id: str) -> None:
    recipe = PostTrainingRecipe.from_file(Path(__file__).parents[2] / "configs" / "post_training" / name)
    assert recipe.model.recipe == model_id
    assert recipe.optimizer.type == "came"
    assert recipe.optimizer.betas == (0.9, 0.999, 0.9999)
    assert recipe.optimizer.epsilon == (1.0e-30, 1.0e-16)
    assert recipe.optimizer.update_clip_threshold == 1.0
    assert recipe.discriminator_optimizer is not None
    assert recipe.discriminator_optimizer.type == "came"
    assert recipe.algorithm.lr_scheduler == "constant-with-warmup"
    assert recipe.algorithm.lr_warmup_steps == 5000
    assert recipe.algorithm.student_fp32_attention is True
    assert recipe.algorithm.teacher_fp32_attention is False


def test_discriminator_execution_uses_independent_times_and_misaligned_pairs(monkeypatch) -> None:
    import worldfoundry.training.post_training.distillation.scm_ladd.objective as objective

    times = iter((0.1, 0.2, 0.3, 0.4))

    def fixed_timesteps(reference: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.full((reference.shape[0],), next(times), dtype=torch.float32)

    noise_calls: list[int] = []

    def zero_noise(
        reference: torch.Tensor,
        *,
        sigma_data: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del sigma_data, generator
        noise_calls.append(reference.shape[0])
        return torch.zeros_like(reference)

    monkeypatch.setattr(objective, "sample_trigflow_timesteps", fixed_timesteps)
    monkeypatch.setattr(objective, "_normal_like", zero_noise)
    student, teacher, discriminator = _roles()
    config = SCMLADDAlgorithmSpec()
    adapter = NativeSCMLADDLossAdapter(
        student,
        teacher,
        discriminator,
        config,
    )
    result = adapter.discriminator_loss(_batch())
    assert torch.isfinite(result.loss)
    fake_call, real_call = discriminator.calls
    assert fake_call["sample_ids"] == (
        "sample-a",
        "sample-b",
        "sample-a#misaligned",
        "sample-b#misaligned",
    )
    torch.testing.assert_close(fake_call["offset"], torch.tensor([0.1, 0.2, 0.1, 0.2]))
    torch.testing.assert_close(fake_call["timesteps"], torch.tensor([0.2, 0.2, 0.4, 0.4]))
    torch.testing.assert_close(real_call["timesteps"], torch.tensor([0.3, 0.3]))
    expected_shifted = torch.cos(torch.tensor(0.4)) * torch.roll(_batch().clean_latents, 1, 0) / 0.5
    torch.testing.assert_close(fake_call["latents"][2:], expected_shifted)
    assert int(result.metrics["misaligned_pairs"].item()) == 2
    assert noise_calls == [2, 2, 2, 2]


def test_native_objective_and_engine_isolate_roles_and_resume_exact_phase() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tangent_warmup_steps=2))
    student, teacher, discriminator = _roles()
    stack = build_native_scm_ladd_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        discriminator=discriminator,
        fused_adamw=False,
    )
    batch = _batch()
    student_before = student.module.weight.detach().clone()
    discriminator_before = discriminator.module.head_scale.detach().clone()
    first = stack.engine.train_step(batch, generator=torch.Generator().manual_seed(11))
    assert first.phase == "generator"
    assert stack.engine.next_phase == "discriminator"
    assert not torch.equal(student.module.weight, student_before)
    torch.testing.assert_close(discriminator.module.head_scale, discriminator_before)
    assert discriminator.module.head_scale.grad is None
    assert discriminator.calls[-1]["training"] is False
    assert all(scale == 0.1 for scale in student.guidance_embedding_scales)
    assert all(scale == 0.1 for scale in teacher.guidance_embedding_scales)

    student_after_generator = student.module.weight.detach().clone()
    second = stack.engine.train_step(batch, generator=torch.Generator().manual_seed(17))
    assert second.phase == "discriminator"
    torch.testing.assert_close(student.module.weight, student_after_generator)
    assert not torch.equal(discriminator.module.head_scale, discriminator_before)
    assert all(call["training"] is True for call in discriminator.calls[-2:])
    assert all(parameter.grad is None for parameter in teacher.module.parameters())
    assert stack.engine.student_optimizer_steps == 1
    assert stack.engine.discriminator_optimizer_steps == 1
    assert stack.engine.next_phase == "generator"

    state = stack.engine.state_dict()
    restored_student, restored_teacher, restored_discriminator = _roles()
    restored = build_native_scm_ladd_training_stack(
        recipe,
        student=restored_student,
        teacher=restored_teacher,
        discriminator=restored_discriminator,
        fused_adamw=False,
    )
    restored.engine.load_state_dict(state)
    assert restored.engine.global_step == 2
    assert restored.engine.next_phase == "generator"
    invalid = dict(state)
    invalid["next_phase"] = "discriminator"
    with pytest.raises(ValueError, match="phase"):
        restored.engine.load_state_dict(invalid)


def test_native_session_counts_each_alternating_optimizer_commit() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(accumulation_steps=2))
    student, teacher, discriminator = _roles()
    stack = build_native_scm_ladd_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        discriminator=discriminator,
        fused_adamw=False,
    )
    events: list[Mapping[str, object]] = []
    progress = TrainingProgress()
    session = NativeSCMLADDTrainingSession(
        stack.engine,
        [_batch()],
        progress,
        event_sink=events.append,
    )
    summary = session.run(max_steps=3, generator=torch.Generator().manual_seed(23))
    assert summary.final_step == 3
    assert summary.student_optimizer_steps == 2
    assert summary.discriminator_optimizer_steps == 1
    assert summary.final_generator_loss is not None
    assert summary.final_discriminator_loss is not None
    assert progress.optimizer_steps == 3
    assert progress.microbatches_seen == 6
    assert progress.samples_seen == 12
    assert [event["microbatches"] for event in events] == [2, 2, 2]
    assert [event["phase"] for event in events] == ["generator", "discriminator", "generator"]


def test_scm_ladd_uneven_microbatch_accumulation_matches_combined_updates() -> None:
    accumulated, accumulated_student_scheduler, accumulated_discriminator_scheduler = (
        _weighted_engine(seed=37, accumulation_steps=2)
    )
    combined, combined_student_scheduler, combined_discriminator_scheduler = _weighted_engine(
        seed=37,
        accumulation_steps=1,
    )
    first = _weighted_batch([1.0], prefix="first")
    second = _weighted_batch([2.0, 3.0, 4.0], prefix="second")
    merged = _weighted_batch([1.0, 2.0, 3.0, 4.0], prefix="merged")

    accumulated_generator = accumulated.train_step((first, second))
    combined_generator = combined.train_step(merged)
    torch.testing.assert_close(
        accumulated.student_module.weight,
        combined.student_module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(accumulated_generator.loss, combined_generator.loss)
    assert accumulated_generator.metrics["loss_denominator"].item() == 4
    assert accumulated_generator.metrics["accumulated_microbatches"] == 2

    accumulated_discriminator = accumulated.train_step((first, second))
    combined_discriminator = combined.train_step(merged)
    torch.testing.assert_close(
        accumulated.discriminator_module.weight,
        combined.discriminator_module.weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(accumulated_discriminator.loss, combined_discriminator.loss)
    assert accumulated_discriminator.metrics["loss_denominator"].item() == 4
    assert accumulated_student_scheduler.steps == combined_student_scheduler.steps == 1
    assert accumulated_discriminator_scheduler.steps == combined_discriminator_scheduler.steps == 1
    assert accumulated.student_optimizer_steps == 1
    assert accumulated.discriminator_optimizer_steps == 1
    assert accumulated.global_step == 2

    state = accumulated.state_dict()
    assert state["gradient_accumulation_steps"] == 2
    with pytest.raises(ValueError, match="accumulation cadence"):
        combined.load_state_dict(state)


def test_scm_ladd_builder_requires_matching_role_accumulation() -> None:
    mapping = _recipe_mapping(accumulation_steps=2)
    discriminator_optimizer = mapping["discriminator_optimizer"]
    assert isinstance(discriminator_optimizer, dict)
    discriminator_optimizer["gradient_accumulation_steps"] = 1
    recipe = PostTrainingRecipe.from_mapping(mapping)
    student, teacher, discriminator = _roles()

    with pytest.raises(ValueError, match="accumulation steps must match"):
        build_native_scm_ladd_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            discriminator=discriminator,
            fused_adamw=False,
        )


def test_builder_rejects_discriminator_head_layout_drift() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student, teacher, discriminator = _roles()
    discriminator.head_block_ids = (2, 8, 14)
    with pytest.raises(ValueError, match="head blocks"):
        build_native_scm_ladd_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            discriminator=discriminator,
            fused_adamw=False,
        )


def test_builder_binds_scm_roles_to_recipe_checkpoints_and_rejects_aliasing() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student, teacher, discriminator = _roles()
    teacher.checkpoint_identity = "different-teacher"
    with pytest.raises(ValueError, match="differs from recipe"):
        build_native_scm_ladd_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            discriminator=discriminator,
            fused_adamw=False,
        )

    shared_module = _VelocityModule(weight=0.2, time_weight=-0.1, log_variance=0.05)
    shared_module.requires_grad_(False)
    shared_student = _VelocityAdapter(shared_module)
    shared_teacher = _VelocityAdapter(shared_module)
    shared_discriminator = _DiscriminatorAdapter(_DiscriminatorModule(shared_module))
    with pytest.raises(ValueError, match="independently materialized"):
        build_native_scm_ladd_training_stack(
            recipe,
            student=shared_student,
            teacher=shared_teacher,
            discriminator=shared_discriminator,
            fused_adamw=False,
        )


def test_engine_refuses_checkpoint_after_generator_commit_side_effect_fails() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student, teacher, discriminator = _roles()
    stack = build_native_scm_ladd_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        discriminator=discriminator,
        student_scheduler=_FailingScheduler(),
        fused_adamw=False,
    )
    with pytest.raises(RuntimeError, match="scheduler failure"):
        stack.engine.train_step(_batch(), generator=torch.Generator().manual_seed(29))
    with pytest.raises(RuntimeError, match="partially committed"):
        stack.engine.state_dict()
    with pytest.raises(RuntimeError, match="partially committed"):
        stack.engine.train_step(_batch())
