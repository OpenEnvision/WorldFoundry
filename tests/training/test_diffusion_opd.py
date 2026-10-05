from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from worldfoundry.training.post_training.distillation.diffusion_opd import (
    DiffusionOPDRolloutBatch,
    DiffusionOPDTrajectorySampler,
    NativeDiffusionOPDDataLoader,
    NativeDiffusionOPDEngine,
    NativeDiffusionOPDTrajectoryReplay,
    build_native_diffusion_opd_training_run,
    build_native_diffusion_opd_training_stack,
    diffusion_opd_loss,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (
    VariancePreservingFlowTransition,
)
from worldfoundry.training.recipes.post_training import (
    DiffusionOPDAlgorithmSpec,
    PostTrainingRecipe,
)
from worldfoundry.training.tuning.full_model import FullModelArtifact


class _Scale(nn.Module):
    def __init__(self, value: float, *, trainable: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value), requires_grad=trainable)


class _FlowAdapter:
    def __init__(self, value: float, checkpoint: str, *, trainable: bool) -> None:
        self.module = _Scale(value, trainable=trainable)
        self.checkpoint_identity = checkpoint
        self.branch_calls: list[str] = []

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
        del sigmas, sample_ids, conditioning
        self.module.train(training)
        self.branch_calls.append(branch)
        branch_offset = 0.05 if branch == "positive" else -0.05
        return noisy_latents * self.module.weight + branch_offset

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
        shape = (noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1)
        return noisy_latents - sigmas.reshape(shape) * velocity


def _recipe_mapping(output_dir: Path, *, save_every_steps: int = 0) -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "diffusion-opd-test", "output_dir": str(output_dir)},
        "model": {"recipe": "toy-flow", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {
            "manifest": "unused.jsonl",
            "shuffle": False,
            "tail_policy": "uneven",
        },
        "algorithm": {
            "type": "diffusion-opd",
            "teachers": [
                {"name": "aesthetic", "checkpoint": "teacher-a", "guidance_scale": 0.0},
                {"name": "ocr", "checkpoint": "teacher-b", "guidance_scale": 1.0},
            ],
            "sigmas": [1.0, 0.5, 0.0],
            "sde_step_indices": [0],
            "eta": 0.2,
            "guidance_scale": 1.0,
            "add_kl_coefficient": False,
            "trajectory_dtype": "float32",
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 2,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "checkpoint": {"save_every_steps": save_every_steps, "async": False},
        "export": {"format": "safetensors"},
    }


def _batches() -> tuple[DiffusionOPDRolloutBatch, ...]:
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    return (
        DiffusionOPDRolloutBatch(
            sample_ids=("a-0", "a-1"),
            domain="aesthetic",
            initial_latents=torch.tensor([[0.2, -0.3], [0.7, 0.4]]),
            sigmas=sigmas,
        ),
        DiffusionOPDRolloutBatch(
            sample_ids=("b-0", "b-1"),
            domain="ocr",
            initial_latents=torch.tensor([[-0.4, 0.1], [0.5, -0.8]]),
            sigmas=sigmas,
        ),
    )


def _roles():
    student = _FlowAdapter(0.4, "student", trainable=True)
    teachers = {
        "aesthetic": _FlowAdapter(0.0, "teacher-a", trainable=False),
        "ocr": _FlowAdapter(0.55, "teacher-b", trainable=False),
    }
    return student, teachers


def test_recipe_is_strict_and_round_trips(tmp_path: Path) -> None:
    payload = _recipe_mapping(tmp_path / "run")
    recipe = PostTrainingRecipe.from_mapping(payload)
    assert isinstance(recipe.algorithm, DiffusionOPDAlgorithmSpec)
    assert recipe.algorithm.sde_step_indices == (0,)
    assert recipe.algorithm.teachers[0].guidance_scale == 0.0
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    payload["algorithm"]["reward_weights"] = {"unused": 1.0}  # type: ignore[index]
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(payload)


def test_loss_matches_official_expression_and_detaches_teacher() -> None:
    student = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    scales = torch.tensor([[[2.0], [4.0]]])

    ode = diffusion_opd_loss(
        student,
        teacher,
        scales,
        add_kl_coefficient=False,
    )
    assert torch.allclose(ode.per_sample_step, torch.tensor([[2.5, 18.5]]))
    assert ode.loss.item() == pytest.approx(10.5)

    kl = diffusion_opd_loss(
        student,
        teacher,
        scales,
        add_kl_coefficient=True,
    )
    assert torch.allclose(
        kl.per_sample_step,
        torch.tensor([[0.625, 1.15625]]),
    )
    assert kl.loss.item() == pytest.approx(0.890625)
    kl.loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_on_policy_rollout_and_teacher_replay_use_same_transition_math() -> None:
    student = _FlowAdapter(0.4, "student", trainable=True)
    teacher = _FlowAdapter(0.1, "teacher-a", trainable=False)
    strategy = VariancePreservingFlowTransition(eta=0.2, sigma_max=0.5)
    sampler = DiffusionOPDTrajectorySampler(
        student,
        transition_strategy=strategy,
        sigmas=(1.0, 0.5, 0.0),
        step_indices=(0,),
        trajectory_dtype=torch.float32,
    )
    trajectory = sampler.sample(
        _batches()[0],
        generator=torch.Generator().manual_seed(9),
    )
    replay = NativeDiffusionOPDTrajectoryReplay(
        teacher,
        transition_strategy=strategy,
    ).replay(trajectory, training=False)
    current = trajectory.latents[:, 0]
    sigma = trajectory.sigmas[0].expand(trajectory.batch_size)
    sigma_next = trajectory.sigmas[1].expand(trajectory.batch_size)
    velocity = teacher.predict_velocity(
        current,
        sigma,
        sample_ids=trajectory.sample_ids,
        conditioning=trajectory.conditioning,
        training=False,
    )
    expected = strategy.step(
        velocity,
        current,
        sigma,
        sigma_next,
        next_sample=trajectory.latents[:, 1],
        trajectory_dtype=trajectory.latents.dtype,
    )
    assert torch.equal(replay.transition_means[:, 0], expected.mean)
    assert torch.equal(replay.transition_scales[:, 0], expected.scale)
    assert torch.equal(trajectory.transition_scales[:, 0], expected.scale)


def test_builder_selects_one_teacher_per_domain_and_updates_only_student(tmp_path: Path) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "run"))
    student, teachers = _roles()
    stack = build_native_diffusion_opd_training_stack(
        recipe,
        student=student,
        teachers=teachers,
        fused_adamw=False,
    )
    assert isinstance(stack.engine, NativeDiffusionOPDEngine)
    optimizer_parameters = {id(parameter) for group in stack.optimizer.param_groups for parameter in group["params"]}
    assert optimizer_parameters == {id(student.module.weight)}
    before = student.module.weight.detach().clone()
    trajectories = tuple(
        stack.sampler.sample(batch, generator=torch.Generator().manual_seed(20 + index))
        for index, batch in enumerate(_batches())
    )
    with pytest.raises(ValueError, match="complete balanced teacher-domain cycles"):
        stack.engine.train_step((trajectories[0], trajectories[0]))
    result = stack.engine.train_step(trajectories)
    assert set(result.domain_losses) == {"aesthetic", "ocr"}
    assert not torch.equal(student.module.weight.detach(), before)
    assert teachers["aesthetic"].module.weight.grad is None
    assert teachers["ocr"].module.weight.grad is None
    assert set(teachers["aesthetic"].branch_calls) == {"negative"}
    assert set(teachers["ocr"].branch_calls) == {"positive"}


def _training_run(
    recipe: PostTrainingRecipe,
    *,
    output_dir: Path,
    resume_checkpoint: str | None = None,
):
    student, teachers = _roles()
    stack = build_native_diffusion_opd_training_stack(
        recipe,
        student=student,
        teachers=teachers,
        fused_adamw=False,
    )
    loader = NativeDiffusionOPDDataLoader(
        _batches(),
        shuffle=recipe.data.shuffle,
        shuffle_seed=recipe.data.shuffle_seed,
    )
    run = build_native_diffusion_opd_training_run(
        recipe,
        stack=stack,
        dataloader=loader,
        student_module=student.module,
        student_tuning=None,
        objective_generator=torch.Generator().manual_seed(17),
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
    )
    return run, student


def test_update_dcp_split_resume_and_export_match_uninterrupted(tmp_path: Path) -> None:
    full_recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "full", save_every_steps=1))
    full_run, full_student = _training_run(
        full_recipe,
        output_dir=tmp_path / "full",
    )
    full_summary = full_run.run(max_iterations=2)
    assert full_summary.final_optimizer_step == 2
    expected = full_student.module.weight.detach().clone()

    split_recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path / "split", save_every_steps=1))
    first_run, _ = _training_run(split_recipe, output_dir=tmp_path / "split")
    assert first_run.run(max_iterations=1).final_optimizer_step == 1

    resumed_run, resumed_student = _training_run(
        split_recipe,
        output_dir=tmp_path / "split",
        resume_checkpoint="latest",
    )
    assert resumed_run.resume_artifact is not None
    summary = resumed_run.run(max_iterations=1)
    assert summary.initial_optimizer_step == 1
    assert summary.final_optimizer_step == 2
    assert torch.equal(resumed_student.module.weight.detach(), expected)

    artifact = resumed_run.export_student()
    assert isinstance(artifact, FullModelArtifact)
    assert artifact.path.is_dir()
