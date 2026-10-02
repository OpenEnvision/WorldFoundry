"""Formula gates against Ji4chenLi/t2v-turbo commit eaae323f10d136796a33e8f5304ed50e40def570.

Compared source paths:
  released LoRA trainer
  ode_solver/ddim_solver.py
  utils/common_utils.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api.contracts import TrainingBatch  # noqa: E402
from worldfoundry.training.engine.single_device import SingleDeviceTrainEngine  # noqa: E402
from worldfoundry.training.post_training.distillation.t2v_turbo import (  # noqa: E402
    LVDMEpsilonPredictor,
    T2VTurboConfig,
    T2VTurboObjective,
    T2VTurboTrainAdapter,
    apply_t2v_turbo_lora,
    audit_t2v_turbo_lora_targets,
    t2v_turbo_scaled_linear_beta_schedule,
)


class _TinyVideoUNet(torch.nn.Module):
    def __init__(self, *, student: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.13 if student else 0.29), requires_grad=student)
        self.context_scale = torch.nn.Parameter(torch.tensor(0.07), requires_grad=student)
        self.time_cond_proj = torch.nn.Linear(4, 1, bias=False) if student else None
        if self.time_cond_proj is not None:
            torch.nn.init.constant_(self.time_cond_proj.weight, 0.02)
        self.input_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(1, 1))])

    def forward(self, noisy, timesteps, *, context, fps=None, timestep_cond=None):
        context_value = context.float().mean(dim=tuple(range(1, context.ndim))).reshape(-1, 1, 1, 1, 1)
        guidance = noisy.new_zeros((noisy.shape[0], 1, 1, 1, 1))
        if timestep_cond is not None:
            guidance = self.time_cond_proj(timestep_cond).reshape(-1, 1, 1, 1, 1)
        return (
            noisy * self.scale
            + context_value * self.context_scale
            + guidance
            + timesteps.reshape(-1, 1, 1, 1, 1) * 0.0
            + (0.0 if fps is None else fps.reshape(-1, 1, 1, 1, 1) * 0.0)
        )


class _IdentityVideoCodec(torch.nn.Module):
    def decode_video(self, latents):
        return latents[:, :3]


class _ImageMeanReward(torch.nn.Module):
    def forward(self, images, prompts):
        del prompts
        return images.flatten(1).mean(dim=1)


class _TinyLoraGraph(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
        self.spatial = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.temporal = torch.nn.Conv3d(2, 3, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, vector, image, video):
        return self.linear(vector), self.spatial(image), self.temporal(video)


def _stack(*, codec=None, config=None, image_reward=None):
    student = _TinyVideoUNet(student=True)
    teacher = _TinyVideoUNet(student=False)
    adapter = T2VTurboTrainAdapter(
        student=LVDMEpsilonPredictor(student),
        teacher=LVDMEpsilonPredictor(teacher),
        codec=codec,
    )
    objective = T2VTurboObjective(
        adapter=adapter,
        config=config or T2VTurboConfig(guidance_embedding_dim=4),
        image_reward=image_reward,
    )
    return adapter, objective, student


def _batch() -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("clip-a", "clip-b"),
        prompts=("a running horse", "ocean waves"),
        conditions={
            "clean_latents": torch.linspace(-0.5, 0.7, 2 * 4 * 4 * 2 * 2).reshape(2, 4, 4, 2, 2),
            "context": torch.tensor([[[0.1], [0.3]], [[-0.2], [0.4]]]),
            "unconditional_context": torch.zeros(1, 2, 1),
            "fps": 16,
        },
    )


def test_t2v_turbo_ddim_grid_matches_released_solver() -> None:
    adapter, objective, _ = _stack()
    del adapter
    expected_starts = np.arange(1, 51, dtype=np.int64) * (1000 // 50) - 1
    np.testing.assert_array_equal(objective.start_timesteps.numpy(), expected_starts)
    alpha_cumprods = objective.alpha_cumprods.numpy()
    expected_previous = np.asarray([alpha_cumprods[0], *alpha_cumprods[expected_starts[:-1]]])
    np.testing.assert_allclose(objective.previous_alpha_cumprods.numpy(), expected_previous)

    prepared = objective.adapter.prepare_batch(_batch())
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(13))
    np.testing.assert_array_equal(
        corrupted.conditioning["end_timesteps"].numpy(),
        np.maximum(corrupted.timesteps.numpy() - 20, 0),
    )
    guidance = corrupted.conditioning["guidance_coefficients"]
    assert torch.all((guidance >= 5.0) & (guidance <= 15.0))


def test_t2v_turbo_schedule_matches_released_float32_formula_exactly() -> None:
    _, objective, _ = _stack()
    expected_betas = torch.linspace(
        0.00085**0.5,
        0.012**0.5,
        1000,
        dtype=torch.float32,
    ).square()
    expected_alpha_cumprods = torch.cumprod(1.0 - expected_betas, dim=0)

    assert t2v_turbo_scaled_linear_beta_schedule(1000).dtype is torch.float32
    torch.testing.assert_close(objective.betas, expected_betas, rtol=0.0, atol=0.0)
    torch.testing.assert_close(objective.alpha_cumprods, expected_alpha_cumprods, rtol=0.0, atol=0.0)


def test_t2v_turbo_uses_provenance_fps_without_a_duplicate_condition_tensor() -> None:
    adapter, _, _ = _stack()
    source = _batch()
    conditions = dict(source.conditions)
    conditions.pop("fps")
    batch = TrainingBatch(
        sample_ids=source.sample_ids,
        prompts=source.prompts,
        conditions=conditions,
        metadata={"target_fps": 16.0},
    )

    prepared = adapter.prepare_batch(batch)
    torch.testing.assert_close(prepared.conditioning["fps"], torch.full((2,), 16, dtype=torch.long))

    with pytest.raises(ValueError, match="requires 16 FPS"):
        adapter.prepare_batch(
            TrainingBatch(
                sample_ids=source.sample_ids,
                prompts=source.prompts,
                conditions=conditions,
                metadata={"target_fps": 15.0},
            )
        )


def test_t2v_turbo_target_matches_teacher_ddim_then_same_student_formula() -> None:
    adapter, objective, _ = _stack()
    prepared = adapter.prepare_batch(_batch())
    objective_batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(19))
    actual = objective._distillation_target(objective_batch)

    noisy = objective_batch.model_input
    starts = objective_batch.timesteps
    ends = objective_batch.conditioning["end_timesteps"]
    pair_indices = objective_batch.conditioning["pair_indices"]
    guidance = objective_batch.conditioning["guidance_coefficients"].reshape(-1, 1, 1, 1, 1)
    context = objective_batch.conditioning["context"]
    unconditional = objective_batch.conditioning["unconditional_context"]
    fps = objective_batch.conditioning["fps"]
    embedding = objective_batch.conditioning["guidance_embedding"]
    alpha = objective.alphas[starts].reshape(-1, 1, 1, 1, 1)
    sigma = objective.sigmas[starts].reshape(-1, 1, 1, 1, 1)
    conditional_epsilon = adapter.teacher.module(noisy, starts, context=context, fps=fps)
    unconditional_epsilon = adapter.teacher.module(noisy, starts, context=unconditional, fps=None)
    conditional_origin = (noisy - sigma * conditional_epsilon) / alpha
    unconditional_origin = (noisy - sigma * unconditional_epsilon) / alpha
    guided_origin = conditional_origin + guidance * (conditional_origin - unconditional_origin)
    guided_epsilon = conditional_epsilon + guidance * (conditional_epsilon - unconditional_epsilon)
    previous_alpha = objective.previous_alpha_cumprods[pair_indices].reshape(-1, 1, 1, 1, 1)
    previous = previous_alpha.sqrt() * guided_origin + (1.0 - previous_alpha).sqrt() * guided_epsilon
    target_epsilon = adapter.student.module(
        previous,
        ends,
        context=context,
        fps=fps,
        timestep_cond=embedding,
    )
    end_alpha = objective.alphas[ends].reshape(-1, 1, 1, 1, 1)
    end_sigma = objective.sigmas[ends].reshape(-1, 1, 1, 1, 1)
    target_origin = (previous - end_sigma * target_epsilon) / end_alpha
    scaled = ends.float() * 10.0
    c_skip = (0.5**2 / (scaled.square() + 0.5**2)).reshape(-1, 1, 1, 1, 1)
    c_out = (scaled / (scaled.square() + 0.5**2).sqrt()).reshape(-1, 1, 1, 1, 1)
    expected = c_skip * previous + c_out * target_origin
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_t2v_turbo_runs_generic_engine_and_lora_audit_matches_released_target_types() -> None:
    adapter, objective, student = _stack()
    audit = audit_t2v_turbo_lora_targets(student)
    expected = tuple(
        name
        for name, module in student.named_modules()
        if name and module.__class__ in (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)
    )
    assert audit.module_names == expected
    optimizer = torch.optim.AdamW(adapter.trainable_module.parameters(), lr=0.01, weight_decay=0.0)
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer, max_grad_norm=100.0)
    before = student.scale.detach().clone()
    result = engine.train_step(_batch(), generator=torch.Generator().manual_seed(23))
    assert result.diagnostics["target_role"] == "current_student_stop_gradient"
    assert not torch.equal(before, student.scale.detach())


def test_t2v_turbo_native_lora_matches_released_extended_injection_and_export(tmp_path: Path) -> None:
    torch.manual_seed(41)
    model = _TinyLoraGraph().eval()
    inputs = (
        torch.randn(2, 3),
        torch.randn(2, 2, 5, 5),
        torch.randn(2, 2, 4, 3, 3),
    )
    before = model(*inputs)

    application = apply_t2v_turbo_lora(model, rank=2, dropout=0.1)
    application.model.eval()
    after = application.model(*inputs)
    for actual, expected in zip(after, before, strict=True):
        torch.testing.assert_close(actual, expected)

    assert application.targeted_module_names == ("linear", "spatial", "temporal")
    assert all("lora_" in name for name in application.trainable_parameter_names)
    assert application.model.get_submodule("linear").dropout.p == pytest.approx(0.1)
    assert application.model.get_submodule("spatial").dropout.p == pytest.approx(0.1)
    assert application.model.get_submodule("temporal").dropout.p == pytest.approx(0.0)

    artifact = application.export_adapter(tmp_path / "adapter")
    tensors = torch.load(artifact.path / "unet_lora.pt", map_location="cpu", weights_only=True)
    assert len(tensors) == 2 * len(application.targeted_module_names)
    for index, name in enumerate(application.targeted_module_names):
        injected = application.model.get_submodule(name)
        torch.testing.assert_close(tensors[2 * index], injected.lora_up.weight.float())
        torch.testing.assert_close(tensors[2 * index + 1], injected.lora_down.weight.float())


def test_t2v_turbo_image_reward_keeps_gradient_through_video_decode() -> None:
    config = T2VTurboConfig(
        guidance_embedding_dim=4,
        distillation_weight=0.0,
        image_reward_weight=1.0,
        image_reward_frames=2,
        image_reward_batch_size=1,
    )
    adapter, objective, student = _stack(
        codec=_IdentityVideoCodec(),
        config=config,
        image_reward=_ImageMeanReward(),
    )
    prepared = adapter.prepare_batch(_batch())
    objective_batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(29))
    prediction = adapter.forward_train(objective_batch)
    teacher_calls = 0

    def count_teacher_calls(_module, _inputs, _output):
        nonlocal teacher_calls
        teacher_calls += 1

    handle = adapter.teacher.module.register_forward_hook(count_teacher_calls)
    try:
        result = objective.compute_loss(prediction, objective_batch)
    finally:
        handle.remove()
    result.loss.backward()
    assert teacher_calls == 0
    assert student.scale.grad is not None
    assert torch.count_nonzero(student.scale.grad).item() == 1
    assert result.losses["t2v_turbo/image_reward"].item() != 0.0
