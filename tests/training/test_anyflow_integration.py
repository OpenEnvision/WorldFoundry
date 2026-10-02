from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training import (
    AnyFlowTrainingBatch,
    NativeAnyFlowBidirectionalAdapter,
    NativeAnyFlowFARAdapter,
    NativeAnyFlowOnPolicyTrainingSession,
    NativeAnyFlowScoreAdapter,
    build_native_anyflow_on_policy_training_stack,
    build_native_anyflow_pretraining_stack,
)
from worldfoundry.training.recipes import PostTrainingRecipe


class _AnyFlowModule(nn.Module):
    def __init__(self, value: float = 0.2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.config = SimpleNamespace(
            patch_size=(1, 1, 1),
            compressed_patch_size=(1, 1, 1),
            full_chunk_limit=1,
            chunk_partition=[1, 1],
            num_layers=2,
            num_attention_heads=1,
            attention_head_dim=2,
            deltatime_type="r",
            gate_value=0.25,
        )
        self.far_patch_embedding = nn.Conv3d(1, 1, 1)
        self.condition_embedder = SimpleNamespace(delta_embedder=nn.Linear(1, 1))

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> object:
        if kwargs.get("is_causal") and kwargs.get("kv_cache") is not None:
            flags = kwargs["kv_cache_flag"]
            assert isinstance(flags, dict)
            if flags["is_cache_step"]:
                return None, kwargs["kv_cache"]
            return self.weight * hidden_states, kwargs["kv_cache"]
        if kwargs.get("is_causal"):
            clean = kwargs.get("clean_hidden_states")
            assert isinstance(clean, torch.Tensor)
            return self.weight * clean
        return (self.weight * hidden_states,)


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self) -> _StatefulLoader:
        return self

    def __next__(self) -> AnyFlowTrainingBatch:
        value = _batch(offset=float(self.cursor) / 10.0, prefix=str(self.cursor))
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _batch(*, offset: float = 0.0, prefix: str = "sample") -> AnyFlowTrainingBatch:
    clean = torch.linspace(-0.8 + offset, 0.9 + offset, 16).reshape(2, 1, 2, 2, 2)
    return AnyFlowTrainingBatch(
        sample_ids=(f"{prefix}-a", f"{prefix}-b"),
        clean_latents=clean,
        conditioning={"context": torch.ones(2, 2, 3)},
        unconditional_conditioning={"context": torch.zeros(2, 2, 3)},
    )


def _algorithm(algorithm_type: str) -> dict[str, object]:
    common: dict[str, object] = {
        "type": algorithm_type,
        "lr_scheduler": "constant-with-warmup",
        "flow_map": {
            "num_train_timesteps": 10,
            "timestep_shift": 1.0,
            "central_difference_epsilon": 1.0,
            "diffusion_ratio": 1.0,
            "consistency_ratio": 0.0,
            "fused_guidance_scale": 1.0,
        },
    }
    if "far" in algorithm_type:
        common["far"] = {
            "chunk_partition": [1, 1],
            "full_chunk_limit": 1,
            "patch_size": [1, 1, 1],
            "compressed_patch_size": [1, 1, 1],
            "long_context_training_ratio": 0.0,
        }
    if "on-policy" in algorithm_type:
        common.update(
            {
                "lr_warmup_steps": 0,
                "real_score_checkpoint": "real-score",
                "fake_score_checkpoint": "fake-score",
                "inference_steps": [2],
                "dmd_batch_size": 1,
                "cotrain_flowmap": False,
                "discriminator_update_ratio": 1,
                "ema_decay": 0.9,
                "ema_warmup_steps": 0,
                "synchronized_seed": 13,
            }
        )
    else:
        common.update(
            {
                "lr_warmup_steps": 1000,
                "ema_decay": 0.999,
                "ema_warmup_steps": 1000,
            }
        )
    return common


def _recipe(algorithm_type: str) -> PostTrainingRecipe:
    payload: dict[str, object] = {
        "run": {"id": f"test-{algorithm_type}", "output_dir": "outputs/test"},
        "model": {"recipe": "wan", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "manifest.jsonl", "shuffle_seed": 7},
        "algorithm": _algorithm(algorithm_type),
        "optimizer": {
            "type": "adamw",
            "learning_rate": 1.0e-3,
            "gradient_accumulation_steps": 1,
        },
        "export": {"format": "safetensors"},
    }
    if "on-policy" in algorithm_type:
        payload["fake_score_optimizer"] = {
            "type": "adamw",
            "learning_rate": 2.0e-3,
            "gradient_accumulation_steps": 1,
        }
    return PostTrainingRecipe.from_mapping(payload)


@pytest.mark.parametrize(
    "algorithm_type",
    (
        "anyflow-far-pretrain",
        "anyflow-bidirectional-pretrain",
        "anyflow-far-on-policy",
        "anyflow-bidirectional-on-policy",
    ),
)
def test_anyflow_recipe_roundtrip_preserves_every_behavior_field(
    algorithm_type: str,
) -> None:
    recipe = _recipe(algorithm_type)
    restored = PostTrainingRecipe.from_mapping(recipe.to_dict())
    assert restored.algorithm.type == algorithm_type
    assert restored == recipe
    assert restored.algorithm.flow_map.num_train_timesteps == 10
    assert restored.algorithm.lr_scheduler == "constant-with-warmup"
    assert restored.algorithm.lr_warmup_steps == (
        0 if "on-policy" in algorithm_type else 1000
    )


def test_anyflow_recipe_enforces_optimizer_topology() -> None:
    on_policy = _recipe("anyflow-far-on-policy").to_dict()
    del on_policy["fake_score_optimizer"]
    with pytest.raises(ValueError, match="requires fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(on_policy)

    pretrain = _recipe("anyflow-far-pretrain").to_dict()
    pretrain["fake_score_optimizer"] = {
        "type": "adamw",
        "learning_rate": 1.0e-3,
    }
    with pytest.raises(ValueError, match="only the primary optimizer"):
        PostTrainingRecipe.from_mapping(pretrain)


def test_anyflow_recipe_limits_released_discriminator_cadence_to_one() -> None:
    payload = _recipe("anyflow-far-on-policy").to_dict()
    payload["algorithm"]["discriminator_update_ratio"] = 2
    with pytest.raises(ValueError, match="discriminator_update_ratio=1"):
        PostTrainingRecipe.from_mapping(payload)


def test_anyflow_pretrain_builder_binds_recipe_and_checkpoint_identity() -> None:
    recipe = _recipe("anyflow-far-pretrain")
    student = NativeAnyFlowFARAdapter(
        _AnyFlowModule(),
        checkpoint_identity="student",
    )
    stack = build_native_anyflow_pretraining_stack(
        recipe,
        student=student,
        fused_adamw=False,
    )
    assert stack.config.far.chunk_partition == (1, 1)
    assert stack.config.flow_map.timestep_shift == 1.0
    assert stack.config.lr_scheduler == "constant-with-warmup"
    assert stack.config.lr_warmup_steps == 1000
    assert stack.config.ema_decay == pytest.approx(0.999)
    assert stack.config.ema_warmup_steps == 1000
    assert stack.engine.gradient_accumulation_steps == 1
    assert tuple(stack.model) == ("student",)
    checkpoint_state = stack.checkpoint_state_kwargs()
    assert checkpoint_state["lr_scheduler"] is stack.scheduler_state
    assert checkpoint_state["ema"] is stack.ema

    initial = student.module.weight.detach().clone()
    generator = torch.Generator().manual_seed(5)
    stack.engine.train_step(_batch(), generator=generator)
    stack.engine.train_step(_batch(), generator=generator)
    assert stack.ema.optimizer_steps.item() == 2
    assert not torch.equal(student.module.weight.detach(), initial)
    live = student.module.weight.detach().clone()
    student.module.weight.data.add_(1.0)
    with stack.use_ema_weights() as ema_student:
        assert ema_student is student.module
        torch.testing.assert_close(student.module.weight, live)
    torch.testing.assert_close(student.module.weight, live + 1.0)

    anonymous = NativeAnyFlowFARAdapter(_AnyFlowModule())
    with pytest.raises(ValueError, match="checkpoint_identity"):
        build_native_anyflow_pretraining_stack(
            recipe,
            student=anonymous,
            fused_adamw=False,
        )


def test_anyflow_bidirectional_pretrain_builder_selects_full_video_objective() -> None:
    recipe = _recipe("anyflow-bidirectional-pretrain")
    student = NativeAnyFlowBidirectionalAdapter(
        _AnyFlowModule(),
        checkpoint_identity="student",
    )
    stack = build_native_anyflow_pretraining_stack(
        recipe,
        student=student,
        fused_adamw=False,
    )
    assert stack.config.image_conditioning_probability == 0.0
    assert "Bidirectional" in type(stack.loss_adapter).__name__


def test_anyflow_on_policy_builder_and_session_consume_fresh_role_batches() -> None:
    recipe = _recipe("anyflow-far-on-policy")
    student = NativeAnyFlowFARAdapter(
        _AnyFlowModule(0.2),
        checkpoint_identity="student",
    )
    real = NativeAnyFlowScoreAdapter(
        _AnyFlowModule(0.5),
        checkpoint_identity="real-score",
    )
    fake = NativeAnyFlowScoreAdapter(
        _AnyFlowModule(-0.1),
        checkpoint_identity="fake-score",
    )
    stack = build_native_anyflow_on_policy_training_stack(
        recipe,
        student=student,
        real_score=real,
        fake_score=fake,
        fused_adamw=False,
    )
    assert not any(parameter.requires_grad for parameter in real.module.parameters())
    assert stack.engine.discriminator_update_ratio == 1
    assert stack.config.lr_warmup_steps == 0
    assert stack.checkpoint_state_kwargs()["ema"] is stack.ema

    loader = _StatefulLoader()
    progress = TrainingProgress()
    session = NativeAnyFlowOnPolicyTrainingSession(stack.engine, loader, progress)
    summary = session.run(
        max_steps=1,
        generator=torch.Generator().manual_seed(17),
    )
    assert loader.cursor == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 4
    assert summary.student_optimizer_steps == 1
    assert summary.fake_score_optimizer_steps == 1
    assert summary.final_step == 1

    with pytest.raises(TypeError, match="fresh AnyFlow fake-score batches"):
        stack.engine.train_step(_batch())


def test_anyflow_on_policy_builder_rejects_role_identity_mismatch() -> None:
    recipe = _recipe("anyflow-bidirectional-on-policy")
    student = NativeAnyFlowBidirectionalAdapter(
        _AnyFlowModule(),
        checkpoint_identity="student",
    )
    real = NativeAnyFlowScoreAdapter(
        _AnyFlowModule(),
        checkpoint_identity="wrong-real",
    )
    fake = NativeAnyFlowScoreAdapter(
        _AnyFlowModule(),
        checkpoint_identity="fake-score",
    )
    with pytest.raises(ValueError, match="differs from recipe"):
        build_native_anyflow_on_policy_training_stack(
            recipe,
            student=student,
            real_score=real,
            fake_score=fake,
            fused_adamw=False,
        )


def test_anyflow_recipe_rejects_unknown_nested_fields() -> None:
    payload = deepcopy(_recipe("anyflow-far-pretrain").to_dict())
    payload["algorithm"]["flow_map"]["unused_metadata"] = "decorative"
    with pytest.raises(ValueError, match="unknown fields"):
        PostTrainingRecipe.from_mapping(payload)


def test_anyflow_recipe_rejects_incompatible_temporal_patch_geometry() -> None:
    payload = deepcopy(_recipe("anyflow-far-pretrain").to_dict())
    payload["algorithm"]["far"]["compressed_patch_size"] = [2, 1, 1]
    with pytest.raises(ValueError, match="temporal patch sizes must agree"):
        PostTrainingRecipe.from_mapping(payload)
