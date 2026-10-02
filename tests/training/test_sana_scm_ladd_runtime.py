from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from copy import deepcopy
from functools import partial

import torch
from transformers import get_constant_schedule_with_warmup

from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.sana.ladd import (
    LADDBatchNormLocal,
    LADDResidualBlock,
    LADDSpectralConv1d,
    SANAFeatureDiscriminatorHeads,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale import (
    SanaMSCM,
)
from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.engine.sana.scm_ladd_roles import SanaSCMLADDTrainableRoles
from worldfoundry.training.models.sana_scm_ladd import (
    SanaLADDDiscriminatorAdapter,
    SanaSCMVelocityAdapter,
)
from worldfoundry.training.optimizers import CAME
from worldfoundry.training.post_training.distillation.scm_ladd.builder import (
    build_native_scm_ladd_training_stack,
)
from worldfoundry.training.post_training.distillation.scm_ladd.contracts import (
    SCMLADDTrainingBatch,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe


def _tiny_sana_scm(*, student: bool) -> SanaMSCM:
    return SanaMSCM(
        input_size=2,
        patch_size=1,
        in_channels=4,
        hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        pred_sigma=False,
        caption_channels=8,
        model_max_length=3,
        qk_norm=True,
        y_norm=True,
        y_norm_scale_factor=0.01,
        attn_type="vanilla",
        ffn_type="mlp",
        use_pe=False,
        linear_head_dim=4,
        cross_norm=True,
        cross_attn_type="vanilla",
        logvar=student,
        cfg_embed=student,
        cfg_embed_scale=0.1,
        timestep_norm_scale_factor=1000.0,
    )


def _conditioning() -> dict[str, torch.Tensor]:
    return {
        "context": torch.randn(2, 1, 3, 8),
        "context_mask": torch.ones(2, 3, dtype=torch.long),
        "cfg_scale": torch.tensor([4.0, 5.0]),
        "img_hw": torch.tensor([[64.0, 64.0], [64.0, 64.0]]),
        "aspect_ratio": torch.ones(2, 1),
    }


def test_real_sana_scm_student_and_ladd_heads_backpropagate_with_role_isolation() -> None:
    student = SanaSCMVelocityAdapter(
        SanaDenoiser(_tiny_sana_scm(student=True), output_scale=0.5),
        role="student",
        checkpoint_identity="default",
        fp32_attention=True,
        expected_latent_channels=4,
    )
    teacher = SanaSCMVelocityAdapter(
        SanaDenoiser(_tiny_sana_scm(student=False), output_scale=0.5),
        role="teacher",
        checkpoint_identity="default",
        fp32_attention=False,
        expected_latent_channels=4,
    )
    assert student.fp32_attention is True
    assert teacher.fp32_attention is False
    assert all(module.fp32_attention is True for module in student.module.modules())
    assert all(module.fp32_attention is False for module in teacher.module.modules())
    heads = SANAFeatureDiscriminatorHeads(hidden_size=16, block_ids=(0, 1))
    discriminator = SanaLADDDiscriminatorAdapter(teacher, heads)
    latents = torch.randn(2, 4, 2, 2, requires_grad=True)
    timesteps = torch.tensor([0.2, 0.8])
    conditioning = _conditioning()

    prediction = student.predict_velocity(
        latents,
        timesteps,
        sample_ids=("a", "b"),
        conditioning=conditioning,
        training=True,
        guidance_embedding_scale=0.1,
        return_log_variance=True,
    )
    assert prediction.velocity.shape == latents.shape
    assert prediction.log_variance is not None
    assert prediction.log_variance.shape == (2, 1)
    (prediction.velocity.square().mean() + prediction.log_variance.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in student.module.parameters())

    student.module.zero_grad(set_to_none=True)
    latents.grad = None
    logits = discriminator.predict_logits(
        latents,
        timesteps,
        sample_ids=("a", "b"),
        conditioning=conditioning,
        training=True,
        head_block_ids=(0, 1),
    )
    assert logits.shape == (2, 8)
    logits.mean().backward()
    assert latents.grad is not None
    assert any(parameter.grad is not None for parameter in heads.parameters())
    assert not any(parameter.grad is not None for parameter in teacher.module.parameters())
    assert not any(parameter.requires_grad for parameter in teacher.module.parameters())


def test_ladd_head_preserves_official_layer_order_and_spectral_state() -> None:
    heads = SANAFeatureDiscriminatorHeads(hidden_size=16, block_ids=(0,))
    head = heads.heads[0]
    first = head.main[0]
    residual = head.main[1]

    assert isinstance(first[0], LADDSpectralConv1d)
    assert isinstance(first[1], LADDBatchNormLocal)
    assert isinstance(residual, LADDResidualBlock)
    assert isinstance(residual.function[0], LADDSpectralConv1d)
    assert isinstance(head.classifier, LADDSpectralConv1d)
    state = head.state_dict()
    assert "classifier.weight_orig" in state
    assert "classifier.weight_u" in state
    assert "classifier.weight_v" in state


class _CheckpointableLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if set(state_dict) != {"cursor"}:
            raise ValueError("loader state fields differ")
        self.cursor = int(state_dict["cursor"])


def _tiny_recipe() -> PostTrainingRecipe:
    optimizer = {
        "type": "came",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999, 0.9999],
        "epsilon": [1.0e-30, 1.0e-16],
        "update_clip_threshold": 1.0,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
    }
    return PostTrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-post-training",
            "execution_owner": "worldfoundry-native",
            "run": {"id": "tiny-sana-scm", "output_dir": "runs/tiny-sana-scm"},
            "model": {"recipe": "sana-sprint-1600m-1024px", "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "unused.jsonl"},
            "algorithm": {
                "type": "scm-ladd",
                "discriminator_head_block_ids": [0, 1],
                "tangent_warmup_steps": 4,
            },
            "optimizer": optimizer,
            "discriminator_optimizer": optimizer,
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "export": {"format": "distributed-checkpoint"},
        }
    )


def test_real_sana_scm_compound_dcp_restores_both_mutable_roles_and_phase(tmp_path) -> None:
    student = SanaSCMVelocityAdapter(
        SanaDenoiser(_tiny_sana_scm(student=True), output_scale=0.5),
        role="student",
        checkpoint_identity="default",
        fp32_attention=True,
        expected_latent_channels=4,
    )
    teacher = SanaSCMVelocityAdapter(
        SanaDenoiser(_tiny_sana_scm(student=False), output_scale=0.5),
        role="teacher",
        checkpoint_identity="default",
        fp32_attention=False,
        expected_latent_channels=4,
    )
    discriminator = SanaLADDDiscriminatorAdapter(
        teacher,
        SANAFeatureDiscriminatorHeads(hidden_size=16, block_ids=(0, 1)),
    )
    stack = build_native_scm_ladd_training_stack(
        _tiny_recipe(),
        student=student,
        teacher=teacher,
        discriminator=discriminator,
        student_scheduler_factory=partial(
            get_constant_schedule_with_warmup,
            num_warmup_steps=4,
        ),
        fused_adamw=False,
    )
    assert isinstance(stack.student_optimizer, CAME)
    assert isinstance(stack.discriminator_optimizer, CAME)
    assert stack.scheduler_state is not None
    assert stack.scheduler_state.component_names == ("student",)
    assert stack.engine.discriminator_scheduler is None
    assert stack.student_optimizer.param_groups[0]["lr"] == 0.0
    conditioning = _conditioning()
    batch = SCMLADDTrainingBatch(
        sample_ids=("a", "b"),
        clean_latents=torch.randn(2, 4, 2, 2),
        conditioning=conditioning,
        unconditional_conditioning=conditioning,
    )
    generator = torch.Generator().manual_seed(17)
    progress = TrainingProgress()
    loader = _CheckpointableLoader()
    learning_rates: list[float] = []
    for _ in range(2):
        stack.engine.train_step(batch, generator=generator)
        learning_rates.append(float(stack.student_optimizer.param_groups[0]["lr"]))
        loader.cursor += 1
        progress.record_step(microbatches=1, samples=2, latent_tokens=8)
    assert learning_rates == [2.5e-5, 2.5e-5]

    model = SanaSCMLADDTrainableRoles(student.module, discriminator.module)
    state = TrainingState(
        model=model,
        optimizer=(stack.student_optimizer, stack.discriminator_optimizer),
        engine=stack.engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={"algorithm": "scm-ladd", "roles": ["student", "discriminator-heads"]},
        **stack.checkpoint_state_kwargs(),
    )
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_generator = generator.get_state().clone()
    expected_optimizer_state = tuple(
        deepcopy(optimizer.state_dict())
        for optimizer in (stack.student_optimizer, stack.discriminator_optimizer)
    )
    expected_scheduler_step = stack.engine.student_scheduler.last_epoch
    checkpointer = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = checkpointer.save(state, asynchronous=False)
    assert isinstance(artifact, TrainingCheckpointArtifact)
    assert not any(name.startswith("teacher") for name in expected)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    stack.engine.global_step = 0
    stack.engine.student_optimizer_steps = 0
    stack.engine.discriminator_optimizer_steps = 0
    stack.engine.next_phase = "generator"
    progress.optimizer_steps = 0
    loader.cursor = 0
    generator.manual_seed(999)
    stack.engine.student_scheduler.last_epoch = 999
    for optimizer in (stack.student_optimizer, stack.discriminator_optimizer):
        for optimizer_state in optimizer.state.values():
            optimizer_state["step"] = -1
            for value in optimizer_state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

    checkpointer.load(state, artifact.path)

    assert stack.engine.global_step == 2
    assert stack.engine.next_phase == "generator"
    assert progress.optimizer_steps == 2
    assert loader.cursor == 2
    assert torch.equal(generator.get_state(), expected_generator)
    assert stack.engine.student_scheduler.last_epoch == expected_scheduler_step == 1
    restored = model.state_dict()
    assert restored.keys() == expected.keys()
    assert all(torch.equal(restored[name], value) for name, value in expected.items())
    for optimizer, expected_state in zip(
        (stack.student_optimizer, stack.discriminator_optimizer),
        expected_optimizer_state,
        strict=True,
    ):
        restored_state = optimizer.state_dict()
        assert restored_state["param_groups"] == expected_state["param_groups"]
        for parameter_id, state_values in expected_state["state"].items():
            restored_values = restored_state["state"][parameter_id]
            for name, value in state_values.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(restored_values[name], value)
                else:
                    assert restored_values[name] == value
