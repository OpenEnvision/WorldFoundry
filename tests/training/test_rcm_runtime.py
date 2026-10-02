from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from worldfoundry.core.attention.block_pattern import AttnMaskSpec
from worldfoundry.training.post_training.distillation.consistency.math import (
    batch_coefficients,
)
from worldfoundry.training.post_training.distillation.rcm import (
    CausalRCMConfig,
    CausalRolloutRequest,
    NativeCausalRCMLossAdapter,
    NativeCausalSelfForcingRollout,
    NativeRCMLossAdapter,
    NativeRCMTrainEngine,
    RCMConfig,
    RCMLossResult,
    RCMPrediction,
    RCMTrainingBatch,
    build_native_causal_rcm_training_stack,
    build_native_rcm_training_stack,
    causal_block_pattern,
)
from worldfoundry.training.recipes import PostTrainingRecipe

_FIXTURE = Path(__file__).parent / "fixtures/source_formulas/rcm.json"


class _ScalarModule(nn.Module):
    def __init__(self, value: float, *, trainable: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)), requires_grad=trainable)


class _PredictionAdapter:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool = True,
        checkpoint_identity: str = "student-checkpoint",
    ) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
        self.checkpoint_identity = checkpoint_identity

    def predict(
        self,
        noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> RCMPrediction:
        del sample_ids, conditioning, training, branch
        time = batch_coefficients(trig_timesteps, noisy_latents)
        velocity = self.module.weight * noisy_latents + 0.1 * time
        clean = torch.cos(time) * noisy_latents - torch.sin(time) * velocity
        return RCMPrediction(clean_latents=clean, velocity=velocity)


class _ExactPredictionAdapter(_PredictionAdapter):
    supports_exact_jvp = True

    def predict_with_directional_derivative(
        self,
        noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        tangent_latents: torch.Tensor,
        tangent_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> tuple[RCMPrediction, torch.Tensor]:
        primal = self.predict(
            noisy_latents,
            trig_timesteps,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=False,
        )
        tangent = self.module.weight * tangent_latents + 0.1 * batch_coefficients(
            tangent_timesteps,
            noisy_latents,
        )
        return primal, tangent


class _RecordingSynchronizer:
    def __init__(self) -> None:
        self.calls = 0

    def synchronize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value


def _batch(*, video: bool = False, offset: float = 0.0) -> RCMTrainingBatch:
    shape = (2, 1, 3, 2, 2) if video else (2, 3)
    clean = torch.linspace(-0.7 + offset, 0.9 + offset, steps=torch.tensor(shape).prod().item()).reshape(shape)
    return RCMTrainingBatch(
        sample_ids=("a", "b"),
        clean_latents=clean,
        conditioning={"embedding": torch.ones(2, 1)},
        unconditional_conditioning={"embedding": torch.zeros(2, 1)},
    )


def test_continuous_rcm_fails_closed_without_verified_exact_jvp() -> None:
    with pytest.raises(RuntimeError, match="verified exact JVP"):
        NativeRCMLossAdapter(
            _PredictionAdapter(0.2),
            _PredictionAdapter(0.4, trainable=False),
            None,
            RCMConfig(
                consistency_mode="continuous",
                dmd_loss_scale=0,
            ),
        )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_bidirectional_consistency_paths_execute_and_only_train_student(mode: str) -> None:
    student = _ExactPredictionAdapter(0.2)
    teacher = _PredictionAdapter(0.4, trainable=False)
    config = RCMConfig(
        consistency_mode=mode,
        tangent_warmup_steps=2,
        dmd_loss_scale=0,
    )
    objective = NativeRCMLossAdapter(student, teacher, None, config)
    result = objective.student_loss(
        _batch(),
        iteration=1,
        effective_student_iteration=1,
        include_dmd=False,
        generator=torch.Generator().manual_seed(3),
    )
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert teacher.module.weight.grad is None


def test_joint_bidirectional_cm_and_dmd_share_one_student_graph_and_isolate_scores() -> None:
    student = _ExactPredictionAdapter(0.2)
    teacher = _PredictionAdapter(0.4, trainable=False)
    fake = _PredictionAdapter(-0.1)
    objective = NativeRCMLossAdapter(
        student,
        teacher,
        fake,
        RCMConfig(
            consistency_mode="discrete",
            tangent_warmup_steps=0,
            max_rollout_steps=2,
        ),
    )
    result = objective.student_loss(
        _batch(),
        iteration=0,
        effective_student_iteration=0,
        include_dmd=True,
        generator=torch.Generator().manual_seed(11),
    )
    assert bool(result.metrics["joint_dmd"])
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert fake.module.weight.grad is None
    assert teacher.module.weight.grad is None

    student.module.weight.grad = None
    fake_result = objective.fake_score_loss(
        _batch(),
        effective_fake_iteration=0,
        generator=torch.Generator().manual_seed(13),
    )
    fake_result.loss.backward()
    assert student.module.weight.grad is None
    assert fake.module.weight.grad is not None


def test_native_builder_constructs_both_optimizer_roles_and_accumulates() -> None:
    student = _PredictionAdapter(0.2)
    teacher = _PredictionAdapter(
        0.4,
        trainable=False,
        checkpoint_identity="teacher-checkpoint",
    )
    fake = _PredictionAdapter(-0.1, checkpoint_identity="fake-score-checkpoint")
    recipe = PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "rcm", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "student-checkpoint",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "latents.jsonl"},
            "algorithm": {
                "type": "rcm",
                "teacher_checkpoint": "teacher-checkpoint",
                "fake_score_checkpoint": "fake-score-checkpoint",
                "consistency_mode": "discrete",
                "tangent_warmup_steps": 0,
                "student_update_frequency": 2,
                "max_rollout_steps": 2,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1e-3,
                "max_grad_norm": 10,
                "gradient_accumulation_steps": 2,
            },
            "fake_score_optimizer": {
                "type": "adamw",
                "learning_rate": 1e-3,
                "max_grad_norm": 10,
                "gradient_accumulation_steps": 2,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )
    synchronizer = _RecordingSynchronizer()
    stack = build_native_rcm_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=fake,
        fused_adamw=False,
        tensor_synchronizer=synchronizer,
    )
    assert len(stack.optimizers) == 2
    assert stack.engine.gradient_accumulation_steps == 2
    student_result = stack.engine.train_step(
        (_batch(offset=0.0), _batch(offset=0.2)),
        generator=torch.Generator().manual_seed(19),
    )
    assert student_result.phase == "student"
    assert stack.engine.student_optimizer_steps == 1
    assert stack.engine.fake_score_optimizer_steps == 0
    fake_result = stack.engine.train_step(
        (_batch(offset=0.1), _batch(offset=0.3)),
        generator=torch.Generator().manual_seed(23),
    )
    assert fake_result.phase == "fake-score"
    assert stack.engine.student_optimizer_steps == 1
    assert stack.engine.fake_score_optimizer_steps == 1
    assert synchronizer.calls > 0


class _EngineLoss:
    def __init__(self, student: _ScalarModule, fake: _ScalarModule) -> None:
        self.student = student
        self.fake = fake
        self.student_calls: list[tuple[int, int, bool]] = []
        self.fake_calls: list[int] = []

    def loss_denominator(self, batch: RCMTrainingBatch, *, role: str) -> int:
        del role
        return batch.batch_size

    def student_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        iteration: int,
        effective_student_iteration: int,
        include_dmd: bool,
        generator: object | None = None,
    ) -> RCMLossResult:
        del generator
        self.student_calls.append((iteration, effective_student_iteration, include_dmd))
        target = batch.clean_latents.float().flatten(start_dim=1).mean(dim=1)
        loss = (self.student.weight - target).square().mean()
        if include_dmd:
            loss = loss + (self.student.weight + 0.25).square()
        return RCMLossResult(
            loss=loss,
            metrics={"loss_denominator": torch.tensor(batch.batch_size)},
        )

    def fake_score_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        effective_fake_iteration: int,
        generator: object | None = None,
    ) -> RCMLossResult:
        del generator
        self.fake_calls.append(effective_fake_iteration)
        target = batch.clean_latents.float().flatten(start_dim=1).mean(dim=1)
        loss = (self.fake.weight - target).square().mean()
        return RCMLossResult(
            loss=loss,
            metrics={"loss_denominator": torch.tensor(batch.batch_size)},
        )


def _engine(*, accumulation: int = 1) -> tuple[NativeRCMTrainEngine, _EngineLoss, _ScalarModule, _ScalarModule]:
    student = _ScalarModule(0.5)
    teacher = _ScalarModule(0.0, trainable=False)
    fake = _ScalarModule(-0.5)
    objective = _EngineLoss(student, fake)
    engine = NativeRCMTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
        tangent_warmup_steps=2,
        student_update_frequency=3,
        dmd_enabled=True,
        student_max_grad_norm=10,
        fake_score_max_grad_norm=10,
        gradient_accumulation_steps=accumulation,
    )
    return engine, objective, student, fake


def test_engine_matches_official_warmup_cadence_and_joint_commit_counts() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    expected = fixture["expected"]["cadence"]
    engine, objective, student, fake = _engine()
    phases = []
    for _ in fixture["inputs"]["cadence"]["iterations"]:
        phases.append(engine.train_step(_batch()).phase)
    assert phases == expected["phases"]
    assert [value[1] for value in objective.student_calls] == [0, 1, 2, 3]
    assert [value[2] for value in objective.student_calls] == [False, False, True, True]
    assert objective.fake_calls == [0, 1, 2, 3]
    assert engine.student_optimizer_steps == 4
    assert engine.fake_score_optimizer_steps == 4
    assert engine.global_step == 8
    assert student.weight.grad is None
    assert fake.weight.grad is not None


def test_engine_state_resume_derives_and_rejects_impossible_counters() -> None:
    engine, _, _, _ = _engine()
    for _ in range(6):
        engine.train_step(_batch())
    state = engine.state_dict()
    restored, _, _, _ = _engine()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    invalid = dict(state)
    invalid["student_optimizer_steps"] = int(invalid["student_optimizer_steps"]) + 1
    with pytest.raises(ValueError, match="violate the active cadence"):
        restored.load_state_dict(invalid)


def test_engine_weighted_accumulation_matches_one_full_batch_update() -> None:
    values = torch.tensor(
        [
            [-0.8, -0.2, 0.1],
            [0.3, 0.7, 0.9],
            [1.1, 1.4, 1.8],
            [-1.2, -0.6, 0.2],
        ]
    )

    def make_batch(latents: torch.Tensor, prefix: str) -> RCMTrainingBatch:
        count = latents.shape[0]
        return RCMTrainingBatch(
            sample_ids=tuple(f"{prefix}-{index}" for index in range(count)),
            clean_latents=latents,
            conditioning={},
            unconditional_conditioning={},
        )

    full_engine, _, full_student, _ = _engine(accumulation=1)
    split_engine, _, split_student, _ = _engine(accumulation=2)
    full_engine.train_step(make_batch(values, "full"))
    split_engine.train_step(
        (
            make_batch(values[:1], "left"),
            make_batch(values[1:], "right"),
        )
    )
    torch.testing.assert_close(split_student.weight, full_student.weight, rtol=0, atol=1e-7)


class _FailingScheduler:
    def step(self) -> None:
        raise RuntimeError("scheduler failure")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


def test_engine_poison_after_optimizer_commit_cannot_continue_or_checkpoint() -> None:
    student = _ScalarModule(0.5)
    teacher = _ScalarModule(0.0, trainable=False)
    fake = _ScalarModule(-0.5)
    objective = _EngineLoss(student, fake)
    engine = NativeRCMTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
        tangent_warmup_steps=0,
        student_update_frequency=3,
        dmd_enabled=True,
        student_max_grad_norm=10,
        fake_score_max_grad_norm=10,
        student_scheduler=_FailingScheduler(),
    )
    with pytest.raises(RuntimeError, match="scheduler failure"):
        engine.train_step(_batch())
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.train_step(_batch())


class _CausalAdapter:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool = True,
        checkpoint_identity: str = "student-checkpoint",
    ) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
        self.checkpoint_identity = checkpoint_identity
        self.masks: list[AttnMaskSpec] = []

    def predict_velocity(
        self,
        packed_latents: torch.Tensor,
        packed_rf_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sample_ids, conditioning, training, branch
        self.masks.append(attention_mask)
        time = packed_rf_timesteps[:, None, :, None, None]
        return self.module.weight * packed_latents + 0.05 * time


class _CausalExactAdapter(_CausalAdapter):
    supports_exact_jvp = True

    def predict_velocity_with_directional_derivative(
        self,
        packed_latents: torch.Tensor,
        packed_rf_timesteps: torch.Tensor,
        tangent_latents: torch.Tensor,
        tangent_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primal = self.predict_velocity(
            packed_latents,
            packed_rf_timesteps,
            sample_ids=sample_ids,
            conditioning=conditioning,
            attention_mask=attention_mask,
            training=False,
        )
        tangent = self.module.weight * tangent_latents + 0.05 * tangent_timesteps[
            :,
            None,
            :,
            None,
            None,
        ]
        return primal, tangent


class _ScoreAdapter:
    def __init__(
        self,
        value: float,
        *,
        trainable: bool = True,
        checkpoint_identity: str = "fake-score-checkpoint",
    ) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
        self.checkpoint_identity = checkpoint_identity

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        rf_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sample_ids, conditioning, training, branch
        return self.module.weight * noisy_latents + 0.05 * batch_coefficients(
            rf_timesteps,
            noisy_latents,
        )


class _RolloutAdapter:
    def __init__(self, module: _ScalarModule) -> None:
        self.module = module
        self.requests: list[CausalRolloutRequest] = []

    def rollout(
        self,
        batch: RCMTrainingBatch,
        request: CausalRolloutRequest,
        *,
        training: bool,
        generator: object | None,
    ) -> torch.Tensor:
        del generator
        self.requests.append(request)
        clean = batch.clean_latents
        generated = torch.ones_like(clean) * self.module.weight
        return generated if training else generated.detach()


def test_causal_discrete_path_reuses_core_block_pattern_and_executes() -> None:
    batch = _batch(video=True)
    config = CausalRCMConfig(
        consistency_mode="discrete",
        dmd_loss_scale=0,
        first_chunk_frames=1,
        chunk_frames=1,
        spatial_patch_area=4,
    )
    pattern, blocks = causal_block_pattern(batch.clean_latents, config)
    assert pattern.frame_tokens == 1
    assert blocks == 3
    student = _CausalAdapter(0.2)
    teacher = _CausalAdapter(0.4, trainable=False)
    objective = NativeCausalRCMLossAdapter(
        student,
        teacher,
        None,
        None,
        None,
        config,
    )
    result = objective.student_loss(
        batch,
        iteration=0,
        effective_student_iteration=0,
        include_dmd=False,
        generator=torch.Generator().manual_seed(5),
    )
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert teacher.module.weight.grad is None
    assert student.masks
    assert all(mask.mode == "teacher_forcing" for mask in student.masks)
    assert all(mask.pattern == pattern and mask.clean_blocks == blocks for mask in student.masks)


def test_causal_continuous_path_fails_closed_without_exact_jvp_kernel() -> None:
    with pytest.raises(RuntimeError, match="verified native exact-JVP"):
        NativeCausalRCMLossAdapter(
            _CausalAdapter(0.2),
            _CausalAdapter(0.4, trainable=False),
            None,
            None,
            None,
            CausalRCMConfig(
                consistency_mode="continuous",
                dmd_loss_scale=0,
            ),
        )


def test_causal_joint_tf_scm_and_sf_dmd_and_fake_phase_use_fresh_rollouts() -> None:
    batch = _batch(video=True)
    student = _CausalExactAdapter(0.2)
    causal_teacher = _CausalAdapter(0.4, trainable=False)
    rollout = _RolloutAdapter(student.module)
    teacher = _ScoreAdapter(0.5, trainable=False)
    fake = _ScoreAdapter(-0.1)
    objective = NativeCausalRCMLossAdapter(
        student,
        causal_teacher,
        rollout,
        teacher,
        fake,
        CausalRCMConfig(
            consistency_mode="continuous",
            tangent_warmup_steps=0,
            max_rollout_steps=2,
            rollout_timesteps=(0.8,),
        ),
    )
    student_result = objective.student_loss(
        batch,
        iteration=0,
        effective_student_iteration=0,
        include_dmd=True,
        generator=torch.Generator().manual_seed(7),
    )
    student_result.loss.backward()
    assert student.module.weight.grad is not None
    assert causal_teacher.module.weight.grad is None
    assert teacher.module.weight.grad is None
    assert fake.module.weight.grad is None
    assert len(rollout.requests) == 1
    assert rollout.requests[0].steps_per_block == (1, 1, 1)

    student.module.weight.grad = None
    fake_result = objective.fake_score_loss(
        batch,
        effective_fake_iteration=1,
        generator=torch.Generator().manual_seed(9),
    )
    fake_result.loss.backward()
    assert student.module.weight.grad is None
    assert fake.module.weight.grad is not None
    assert len(rollout.requests) == 2
    assert rollout.requests[1].steps_per_block == (2, 2, 2)
    assert rollout.requests[1].timesteps_per_block == ((1.0, 0.8),) * 3


def test_causal_recipe_builder_binds_every_model_role_and_optimizer() -> None:
    student = _CausalAdapter(0.2)
    causal_teacher = _CausalAdapter(
        0.4,
        trainable=False,
        checkpoint_identity="causal-teacher-checkpoint",
    )
    rollout = _RolloutAdapter(student.module)
    bidirectional_teacher = _ScoreAdapter(
        0.5,
        trainable=False,
        checkpoint_identity="bidirectional-teacher-checkpoint",
    )
    fake_score = _ScoreAdapter(-0.1)
    recipe = PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "causal-rcm", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "student-checkpoint",
            },
            "tuning": {"mode": "full"},
            "data": {"manifest": "latents.jsonl"},
            "algorithm": {
                "type": "causal-rcm",
                "causal_teacher_checkpoint": "causal-teacher-checkpoint",
                "bidirectional_teacher_checkpoint": "bidirectional-teacher-checkpoint",
                "fake_score_checkpoint": "fake-score-checkpoint",
                "consistency_mode": "discrete",
                "tangent_warmup_steps": 0,
                "student_update_frequency": 2,
                "max_rollout_steps": 2,
                "rollout_timesteps": [0.8],
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1.0e-3,
                "max_grad_norm": 10.0,
            },
            "fake_score_optimizer": {
                "type": "adamw",
                "learning_rate": 1.0e-3,
                "max_grad_norm": 10.0,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )
    stack = build_native_causal_rcm_training_stack(
        recipe,
        student=student,
        causal_teacher=causal_teacher,
        rollout=rollout,
        bidirectional_teacher=bidirectional_teacher,
        fake_score=fake_score,
        fused_adamw=False,
    )

    assert stack.recipe is recipe
    assert len(stack.optimizers) == 2
    assert stack.engine.train_step(
        _batch(video=True),
        generator=torch.Generator().manual_seed(41),
    ).phase == "student"
    assert stack.engine.train_step(
        _batch(video=True),
        generator=torch.Generator().manual_seed(43),
    ).phase == "fake-score"


class _BlockModelAdapter:
    def __init__(self) -> None:
        self.module = _ScalarModule(0.25)
        self.events: list[tuple[str, bool, bool, int]] = []
        self.synchronizations = 0
        self.cache_allocations: list[tuple[int, int]] = []

    def allocate_cache(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        del device, dtype
        self.cache_allocations.append((batch_size, max_tokens))
        return {"appended": []}

    def synchronize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        self.synchronizations += 1
        return value

    def predict_block_velocity(
        self,
        block_latents: torch.Tensor,
        rf_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
        cache: dict[str, object],
        cache_mode: str,
        training: bool,
    ) -> torch.Tensor:
        del sample_ids, conditioning, rf_timesteps
        self.events.append(
            (
                cache_mode,
                training,
                torch.is_grad_enabled(),
                attention_mask.q_block_offset,
            )
        )
        if cache_mode == "append":
            cache["appended"].append(attention_mask.q_block_offset)
        return self.module.weight * block_latents


def test_native_causal_rollout_owns_block_cache_and_final_step_gradient_boundaries() -> None:
    batch = _batch(video=True)
    config = CausalRCMConfig(
        consistency_mode="discrete",
        dmd_loss_scale=0,
        max_rollout_steps=2,
        rollout_timesteps=(0.8,),
    )
    pattern, blocks = causal_block_pattern(batch.clean_latents, config)
    request = CausalRolloutRequest(
        pattern=pattern,
        steps_per_block=(2,) * blocks,
        timesteps_per_block=((1.0, 0.8),) * blocks,
    )
    model = _BlockModelAdapter()
    rollout = NativeCausalSelfForcingRollout(model)
    generated = rollout.rollout(
        batch,
        request,
        training=True,
        generator=torch.Generator().manual_seed(17),
    )
    generated.sum().backward()
    assert model.module.weight.grad is not None
    assert model.synchronizations == 2
    assert model.cache_allocations == [(2, 3)]
    reads = [event for event in model.events if event[0] == "read"]
    appends = [event for event in model.events if event[0] == "append"]
    assert reads == [
        ("read", False, False, 0),
        ("read", True, True, 0),
        ("read", False, False, 1),
        ("read", True, True, 1),
        ("read", False, False, 2),
        ("read", True, True, 2),
    ]
    assert appends == [
        ("append", False, False, 0),
        ("append", False, False, 1),
        ("append", False, False, 2),
    ]
