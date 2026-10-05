from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.diagonal import (  # noqa: E402
    DiagonalDMDLossAdapter,
    DiagonalDMDLossResult,
    DiagonalFixedTeacherSampler,
    DiagonalObjectiveConfig,
    DiagonalRolloutSampler,
    DiagonalScheduleConfig,
    NativeDiagonalTrainEngine,
    SpatialMotionHead,
    diagonal_distribution_gradients,
    diagonal_proxy_losses,
    dynamic_motion_weights,
    exponential_motion_weights,
    hybrid_motion_weights,
    load_diagonal_stage_weights,
    register_motion_head,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDConfig,
    DMDTrainingBatch,
    NativeDMDTrainEngine,
)


class _StudentModule(torch.nn.Module):
    def __init__(self, value: float = 0.2) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))


class _RecordingAdapter:
    def __init__(self, value: float = 0.2, *, frozen: bool = False) -> None:
        self.module = _StudentModule(value)
        if frozen:
            self.module.requires_grad_(False)
        self.calls: list[dict[str, object]] = []
        self.commits: list[dict[str, object]] = []

    def initialize_cache(self, reference, *, sample_ids, conditioning):
        del sample_ids, conditioning
        return {
            "active_block": -1,
            "committed_blocks": 0,
            "embedding": torch.zeros(
                reference.shape[0],
                1,
                1,
                1,
                1,
                device=reference.device,
                dtype=reference.dtype,
            ),
        }

    def predict_clean_chunk(
        self,
        noisy_chunk,
        timesteps,
        sigmas,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
        training,
    ):
        del sigmas, start_frame, sample_ids, conditioning
        assert cache["committed_blocks"] == block_index
        cache["active_block"] = block_index
        self.calls.append(
            {
                "block": block_index,
                "timestep": float(timesteps[0].item()),
                "training": training,
                "grad_enabled": torch.is_grad_enabled(),
                "input": noisy_chunk.detach().clone(),
            }
        )
        step_value = timesteps.to(noisy_chunk.dtype).reshape(-1, 1, 1, 1, 1) / 1000.0
        return noisy_chunk * self.module.scale + step_value + cache["embedding"]

    def commit_clean_chunk(
        self,
        clean_chunk,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
    ):
        return self.commit_context_chunk(
            clean_chunk,
            torch.zeros(clean_chunk.shape[0], device=clean_chunk.device),
            block_index=block_index,
            start_frame=start_frame,
            sample_ids=sample_ids,
            conditioning=conditioning,
            cache=cache,
        )

    def commit_context_chunk(
        self,
        context_chunk,
        context_timesteps,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
    ):
        del start_frame, sample_ids, conditioning
        assert not torch.is_grad_enabled()
        assert not context_chunk.requires_grad
        assert cache["active_block"] == block_index
        self.commits.append(
            {
                "block": block_index,
                "context": context_chunk.detach().clone(),
                "timestep": context_timesteps.detach().clone(),
            }
        )
        cache["embedding"] = context_chunk.mean(dim=(1, 2, 3, 4), keepdim=True)
        cache["committed_blocks"] = block_index + 1
        return cache


class _ScoreAdapter:
    def __init__(self, value: float, *, frozen: bool = False) -> None:
        self.module = _StudentModule(value)
        if frozen:
            self.module.requires_grad_(False)

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
        del sigmas, sample_ids, conditioning, training
        branch_scale = 0.5 if branch == "negative" else 1.0
        return noisy_latents * self.module.scale * branch_scale

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
        del sigmas, sample_ids, conditioning, training, branch
        return noisy_latents * self.module.scale


def _batch(*, frames: int = 6, device: str | torch.device = "cpu") -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=("video",),
        clean_latents=torch.zeros(1, 1, frames, 1, 1, device=device),
        conditioning={},
        unconditional_conditioning={},
    )


def _small_schedule(*, diagonal: bool = True, last_step_only: bool = False) -> DiagonalScheduleConfig:
    return DiagonalScheduleConfig.from_raw_timesteps(
        (1000.0, 500.0, 250.0, 100.0),
        frames_per_block=2,
        frame_dim=2,
        warmup_mid_timesteps=(750.0, 500.0),
        use_diagonal_denoising=diagonal,
        context_timestep=100.0,
        exit_step_mode="block",
        last_step_only=last_step_only,
    )


def test_released_schedule_fixture_and_progressive_432_steps() -> None:
    stage_one = DiagonalScheduleConfig.stage_one()
    stage_two = DiagonalScheduleConfig.stage_two()
    teacher = DiagonalScheduleConfig.fixed_teacher()

    assert stage_one.base_schedule.timesteps == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 357.14285714285717)
    )
    assert stage_two.base_schedule.timesteps == pytest.approx((1000.0, 357.14285714285717))
    assert stage_two.warmup_mid_schedule is not None
    assert stage_two.warmup_mid_schedule.timesteps == pytest.approx(
        (937.5, 833.3333333333334)
    )
    assert stage_two.context_timestep == pytest.approx(100.0)
    assert stage_two.context_sigma == pytest.approx(0.1)
    assert stage_one.exit_step_mode == "sequence"
    assert stage_two.exit_step_mode == "sequence"
    assert teacher.base_schedule.timesteps == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 625.0)
    )
    assert [len(stage_two.block_schedule(index).timesteps) for index in range(5)] == [4, 3, 2, 2, 2]
    assert stage_two.block_schedule(0).timesteps == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 357.14285714285717)
    )
    assert stage_two.block_schedule(1).timesteps == pytest.approx(
        (1000.0, 937.5, 357.14285714285717)
    )
    assert stage_two.block_schedule(2).timesteps == pytest.approx(
        (1000.0, 357.14285714285717)
    )


def test_released_objective_fixture_matches_author_training_config() -> None:
    schedule = DiagonalScheduleConfig.stage_two()
    objective = DiagonalObjectiveConfig.released(schedule)

    assert objective.dmd.schedule is schedule.base_schedule
    assert objective.dmd.num_train_timesteps == 1000
    assert objective.dmd.score_min_sigma == pytest.approx(0.02)
    assert objective.dmd.score_max_sigma == pytest.approx(0.98)
    assert objective.dmd.score_flow_shift == pytest.approx(5.0)
    assert objective.dmd.teacher_guidance_scale == pytest.approx(3.0)
    assert objective.dmd.normalization_epsilon == pytest.approx(0.0)
    assert objective.dmd.shared_score_timestep is False
    assert objective.dmd.per_sample_normalization is True
    assert objective.frame_dim == schedule.frame_dim
    assert objective.flow_reg_ema_decay == pytest.approx(0.95)
    assert objective.lambda_spatial_dmd == pytest.approx(4.0)
    assert objective.lambda_flow_dmd == pytest.approx(4.0)
    assert objective.gamma_temporal == pytest.approx(1.0)
    assert objective.lambda_reg == pytest.approx(0.0)
    assert objective.regression_loss_type == "mse"
    assert objective.regression_epsilon == pytest.approx(1.0e-3)
    assert objective.regression_cauchy_scale == pytest.approx(1.0e-2)
    assert objective.use_motion_loss
    assert objective.use_flow_reg_loss
    assert objective.use_teacher_regression


def test_rollout_clips_base_exit_and_keeps_only_selected_calls_live() -> None:
    adapter = _RecordingAdapter()
    sampler = DiagonalRolloutSampler(adapter, _small_schedule(), seed=7)
    batch = _batch()
    result = sampler.rollout(
        batch,
        torch.ones_like(batch.clean_latents),
        base_exit_indices=(3, 3, 3),
        training=True,
    )

    assert result.base_exit_indices == (3, 3, 3)
    assert result.block_exit_indices == (3, 2, 1)
    assert [len(steps) for steps in result.block_timesteps] == [4, 3, 2]
    assert [call["grad_enabled"] for call in adapter.calls] == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    assert [call["training"] for call in adapter.calls] == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    assert bool(result.gradient_mask.all())
    assert result.clean_latents.requires_grad
    assert len(adapter.commits) == 3
    assert all(
        torch.allclose(commit["timestep"], torch.full((1,), 100.0))
        for commit in adapter.commits
    )
    result.clean_latents.sum().backward()
    assert adapter.module.scale.grad is not None
    assert torch.isfinite(adapter.module.scale.grad)
    assert [block.frame_start for block in result.cache_state.blocks] == [0, 2, 4]


def test_context_refresh_uses_fresh_noise_and_detached_cache_commit() -> None:
    adapter = _RecordingAdapter()
    sampler = DiagonalRolloutSampler(adapter, _small_schedule(), seed=19)
    batch = _batch(frames=2)
    result = sampler.rollout(
        batch,
        torch.zeros_like(batch.clean_latents),
        base_exit_indices=(0,),
        training=True,
    )
    clean = result.clean_latents.detach()
    context = adapter.commits[0]["context"]
    assert not torch.equal(context, clean)
    assert not context.requires_grad
    expected_sigma = sampler.config.context_sigma
    recovered_noise = (context - (1.0 - expected_sigma) * clean) / expected_sigma
    assert bool(torch.isfinite(recovered_noise).all())


def test_rank_zero_exit_sampling_is_broadcast_to_every_rank(monkeypatch) -> None:
    adapter = _RecordingAdapter()
    context = SimpleNamespace(rank=1, world_size=2, process_group=None)
    sampler = DiagonalRolloutSampler(
        adapter,
        _small_schedule(),
        parallel_context=context,
        seed=23,
    )
    calls: list[int] = []

    def broadcast(value, *, src, group):
        del group
        calls.append(src)
        value.copy_(torch.tensor([3, 2, 1], device=value.device))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    exits = sampler.sample_base_exit_indices(_batch().clean_latents)
    assert exits == (3, 2, 1)
    assert calls == [0]


def test_released_sequence_exit_consumes_all_block_draws_then_reuses_first(monkeypatch) -> None:
    config = DiagonalScheduleConfig.stage_two(frames_per_block=2)
    sampler = DiagonalRolloutSampler(_RecordingAdapter(), config, seed=23)
    observed_sizes: list[tuple[int, ...]] = []
    original_randint = torch.randint

    def randint(low, high, size, **kwargs):
        del low, high
        observed_sizes.append(size)
        kwargs.pop("generator")
        return torch.tensor([1, 0, 1], **kwargs)

    monkeypatch.setattr(torch, "randint", randint)
    try:
        exits = sampler.sample_base_exit_indices(_batch().clean_latents)
    finally:
        monkeypatch.setattr(torch, "randint", original_randint)

    assert observed_sizes == [(3,)]
    assert exits == (1, 1, 1)


def test_inference_executes_complete_released_432_schedules() -> None:
    adapter = _RecordingAdapter()
    sampler = DiagonalRolloutSampler(
        adapter,
        DiagonalScheduleConfig.stage_two(frames_per_block=2),
        seed=23,
    )
    result = sampler.inference(
        _batch(),
        torch.ones_like(_batch().clean_latents),
    )

    assert result.base_exit_indices == (1, 1, 1)
    assert result.block_exit_indices == (3, 2, 1)
    assert [len(steps) for steps in result.block_timesteps] == [4, 3, 2]
    assert len(adapter.calls) == 9
    assert all(call["grad_enabled"] is False for call in adapter.calls)
    assert not bool(result.gradient_mask.any())


def test_sampler_rng_and_diagonal_exit_state_restore_exactly() -> None:
    batch = _batch()
    first = DiagonalRolloutSampler(_RecordingAdapter(), _small_schedule(), seed=29)
    first.sample(batch, first.config.base_schedule, generator=first.generator, training=False)
    state = first.state_dict()
    expected = first.sample(
        batch,
        first.config.base_schedule,
        generator=first.generator,
        training=False,
    )

    restored = DiagonalRolloutSampler(_RecordingAdapter(), _small_schedule(), seed=999)
    restored.load_state_dict(state)
    actual = restored.sample(
        batch,
        restored.config.base_schedule,
        generator=restored.generator,
        training=False,
    )
    assert restored.last_base_exit_indices == first.last_base_exit_indices
    assert restored.last_block_exit_indices == first.last_block_exit_indices
    torch.testing.assert_close(actual.clean_latents, expected.clean_latents, rtol=0, atol=0)


def test_distribution_and_motion_proxy_match_released_formulas() -> None:
    generated = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64).reshape(1, 1, 3, 1, 1)
    generated.requires_grad_(True)
    fake = torch.tensor([0.5, 2.0, 5.0]).reshape_as(generated)
    real = torch.tensor([-0.5, 0.0, 1.0]).reshape_as(generated)
    gradients = diagonal_distribution_gradients(
        generated,
        fake,
        real,
        frame_dim=2,
    )

    spatial_normalizer = (generated.detach() - real).abs().mean(dim=(1, 2, 3, 4), keepdim=True)
    expected_spatial = (fake - real) / spatial_normalizer
    fake_motion = fake[:, :, 1:] - fake[:, :, :-1]
    real_motion = real[:, :, 1:] - real[:, :, :-1]
    generated_motion = generated.detach()[:, :, 1:] - generated.detach()[:, :, :-1]
    motion_normalizer = (generated_motion - real_motion).abs().mean(dim=(1, 2, 3, 4), keepdim=True)
    expected_motion = (fake_motion - real_motion) / motion_normalizer
    torch.testing.assert_close(gradients.spatial, expected_spatial.float())
    torch.testing.assert_close(gradients.motion, expected_motion.float())

    dynamic = dynamic_motion_weights(gradients.motion, generated, frame_dim=2)
    per_frame = (gradients.motion.movedim(2, 1) - generated.movedim(2, 1)[:, 1:]).square().mean(
        dim=(2, 3, 4)
    )
    cumulative = per_frame.cumsum(dim=1)
    expected_dynamic = 1.0 + cumulative / cumulative[:, -1:].clamp_min(1.0e-6)
    torch.testing.assert_close(dynamic, expected_dynamic)
    exponential = exponential_motion_weights(2, device=generated.device, dtype=dynamic.dtype)
    torch.testing.assert_close(
        exponential,
        torch.tensor([1.0, 1.2], dtype=dynamic.dtype) / 1.1,
    )
    hybrid = hybrid_motion_weights(gradients.motion, generated, frame_dim=2)
    torch.testing.assert_close(hybrid, 0.7 * dynamic + 0.3 * exponential.view(1, -1))

    mask = torch.ones_like(generated, dtype=torch.bool)
    proxy = diagonal_proxy_losses(generated, gradients, gradient_mask=mask, frame_dim=2)
    expected_spatial_loss = 0.5 * expected_spatial.double().square().mean()
    expected_motion_loss = (expected_motion.double().square() * hybrid.view(1, 1, 2, 1, 1)).mean()
    torch.testing.assert_close(proxy.spatial, expected_spatial_loss)
    torch.testing.assert_close(proxy.motion, expected_motion_loss)
    (proxy.spatial + proxy.motion).backward()
    assert generated.grad is not None
    assert bool(torch.isfinite(generated.grad).all())


def test_spatial_motion_head_starts_as_exact_identity() -> None:
    head = SpatialMotionHead(
        2,
        num_layers=2,
        kernel_size=1,
        hidden_dim=4,
        norm_num_groups=1,
    )
    value = torch.randn(2, 3, 2, 4, 4)
    torch.testing.assert_close(head(value), value, rtol=0, atol=0)


def _objective_stack(*, teacher_regression: bool, flow_regression: bool):
    schedule = DiagonalScheduleConfig.from_raw_timesteps(
        (1000.0, 100.0),
        frames_per_block=2,
        frame_dim=2,
        warmup_mid_timesteps=(750.0, 500.0),
        use_diagonal_denoising=True,
        context_timestep=100.0,
        exit_step_mode="block",
    )
    student_adapter = _RecordingAdapter()
    student_head = teacher_head = None
    if flow_regression:
        student_head = register_motion_head(
            student_adapter.module,
            SpatialMotionHead(
                1,
                num_layers=2,
                kernel_size=1,
                hidden_dim=4,
                norm_num_groups=1,
            ),
        )
        teacher_head = copy.deepcopy(student_head)
        teacher_head.requires_grad_(False)
    sampler = DiagonalRolloutSampler(student_adapter, schedule, seed=31)
    real = _ScoreAdapter(0.7, frozen=True)
    fake = _ScoreAdapter(0.3)
    fixed_sampler = None
    teacher_adapter = None
    if teacher_regression:
        teacher_adapter = _RecordingAdapter(0.4, frozen=True)
        fixed_sampler = DiagonalFixedTeacherSampler(
            teacher_adapter,
            DiagonalScheduleConfig.from_raw_timesteps(
                (1000.0, 750.0, 500.0, 250.0),
                frames_per_block=2,
                frame_dim=2,
                use_diagonal_denoising=False,
                context_timestep=100.0,
                exit_step_mode="block",
                last_step_only=True,
            ),
        )
    objective = DiagonalDMDLossAdapter(
        real,
        fake,
        DiagonalObjectiveConfig(
            dmd=DMDConfig(
                schedule=schedule.base_schedule,
                score_min_sigma=0.1,
                score_max_sigma=0.9,
                score_flow_shift=5.0,
                teacher_guidance_scale=3.5,
                per_sample_normalization=True,
            ),
            frame_dim=2,
            use_motion_loss=True,
            use_flow_reg_loss=flow_regression,
            use_teacher_regression=teacher_regression,
        ),
        student_sampler=sampler,
        fixed_teacher_sampler=fixed_sampler,
        motion_head_student=student_head,
        motion_head_teacher=teacher_head,
    )
    return objective, sampler, student_adapter, real, fake, teacher_adapter


def test_compound_objective_shares_rng_with_fixed_teacher_and_exact_total() -> None:
    objective, sampler, student, _, _, teacher = _objective_stack(
        teacher_regression=True,
        flow_regression=True,
    )
    assert teacher is not None
    result = objective.generator_loss(_batch(frames=4), generator=sampler.generator)
    assert isinstance(result, DiagonalDMDLossResult)
    metrics = result.metrics
    expected = (
        4.0 * metrics["spatial_dmd_loss"]
        + 4.0 * metrics["motion_dmd_loss"]
        + metrics["flow_regression_loss"]
    )
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(student.calls[0]["input"], teacher.calls[0]["input"], rtol=0, atol=0)
    result.loss.backward()
    assert student.module.scale.grad is not None
    assert getattr(student.module, "diagonal_motion_head").output.weight.grad is not None


def test_motion_teacher_ema_state_is_checkpointable_and_strict() -> None:
    objective, _, student, _, _, _ = _objective_stack(
        teacher_regression=False,
        flow_regression=True,
    )
    assert objective.motion_head_student is not None
    assert objective.motion_head_teacher is not None
    with torch.no_grad():
        objective.motion_head_student.output.bias.fill_(2.0)
    result = DiagonalDMDLossResult(loss=torch.tensor(0.0), metrics={})
    objective.commit_generator_step((result,))
    expected = torch.full_like(objective.motion_head_teacher.output.bias, 0.1)
    torch.testing.assert_close(objective.motion_head_teacher.output.bias, expected)
    state = objective.state_dict()

    restored, _, _, _, _, _ = _objective_stack(
        teacher_regression=False,
        flow_regression=True,
    )
    restored.load_state_dict(state)
    assert restored.motion_ema_updates == 1
    assert restored.motion_head_teacher is not None
    torch.testing.assert_close(restored.motion_head_teacher.output.bias, expected)


def test_native_engine_checkpoints_dmd_sampler_and_objective_together() -> None:
    objective, sampler, student, real, fake, _ = _objective_stack(
        teacher_regression=False,
        flow_regression=False,
    )
    dmd_engine = NativeDMDTrainEngine(
        student_module=student.module,
        real_score_module=real.module,
        fake_score_module=fake.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=1.0e-3),
        fake_score_optimizer=torch.optim.SGD(fake.module.parameters(), lr=1.0e-3),
        generator_update_interval=1,
    )
    engine = NativeDiagonalTrainEngine(dmd_engine, sampler, objective)
    result = engine.train_step(_batch(frames=4))
    assert result.generator_updated
    assert engine.global_step == 1
    state = engine.state_dict()

    restored_objective, restored_sampler, restored_student, restored_real, restored_fake, _ = _objective_stack(
        teacher_regression=False,
        flow_regression=False,
    )
    restored_dmd = NativeDMDTrainEngine(
        student_module=restored_student.module,
        real_score_module=restored_real.module,
        fake_score_module=restored_fake.module,
        loss_adapter=restored_objective,
        student_optimizer=torch.optim.SGD(restored_student.module.parameters(), lr=1.0e-3),
        fake_score_optimizer=torch.optim.SGD(restored_fake.module.parameters(), lr=1.0e-3),
        generator_update_interval=1,
    )
    restored = NativeDiagonalTrainEngine(
        restored_dmd,
        restored_sampler,
        restored_objective,
    )
    restored.load_state_dict(state)
    assert restored.global_step == 1
    assert restored.sampler.rollout_count == engine.sampler.rollout_count
    torch.testing.assert_close(
        restored.sampler.generator.get_state(),
        engine.sampler.generator.get_state(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_diagonal_engine_runs_a_cuda_optimizer_step() -> None:
    device = torch.device("cuda")
    schedule = _small_schedule()
    student = _RecordingAdapter()
    real = _ScoreAdapter(0.7, frozen=True)
    fake = _ScoreAdapter(0.3)
    student.module.to(device)
    real.module.to(device)
    fake.module.to(device)
    sampler = DiagonalRolloutSampler(student, schedule, seed=37)
    objective = DiagonalDMDLossAdapter(
        real,
        fake,
        DiagonalObjectiveConfig(
            dmd=DMDConfig(
                schedule=schedule.base_schedule,
                score_flow_shift=5.0,
                teacher_guidance_scale=3.0,
                shared_score_timestep=False,
                per_sample_normalization=True,
            ),
            frame_dim=2,
            use_flow_reg_loss=False,
            use_teacher_regression=False,
        ),
        student_sampler=sampler,
    )
    engine = NativeDiagonalTrainEngine(
        NativeDMDTrainEngine(
            student_module=student.module,
            real_score_module=real.module,
            fake_score_module=fake.module,
            loss_adapter=objective,
            student_optimizer=torch.optim.SGD(student.module.parameters(), lr=1.0e-3),
            fake_score_optimizer=torch.optim.SGD(fake.module.parameters(), lr=1.0e-3),
            generator_update_interval=1,
        ),
        sampler,
        objective,
    )
    before = student.module.scale.detach().clone()
    result = engine.train_step(_batch(frames=4, device=device))
    torch.cuda.synchronize(device)

    assert result.generator_updated
    assert result.generator_loss.is_cuda
    assert result.fake_score_loss.is_cuda
    assert not torch.equal(student.module.scale.detach(), before)


def test_diagonal_runtime_state_round_trips_through_torch_dcp(tmp_path) -> None:
    import torch.distributed.checkpoint as dcp

    objective, sampler, student, real, fake, _ = _objective_stack(
        teacher_regression=False,
        flow_regression=True,
    )
    dmd_engine = NativeDMDTrainEngine(
        student_module=student.module,
        real_score_module=real.module,
        fake_score_module=fake.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=1.0e-3),
        fake_score_optimizer=torch.optim.SGD(fake.module.parameters(), lr=1.0e-3),
        generator_update_interval=1,
    )
    engine = NativeDiagonalTrainEngine(dmd_engine, sampler, objective)
    engine.train_step(_batch(frames=4))
    expected = engine.state_dict()
    checkpoint_dir = tmp_path / "diagonal-dcp"
    dcp.save({"engine": engine}, checkpoint_id=checkpoint_dir)

    engine.train_step(_batch(frames=4))
    assert engine.global_step == 2
    dcp.load({"engine": engine}, checkpoint_id=checkpoint_dir)
    assert engine.global_step == 1
    assert engine.sampler.rollout_count == expected["sampler"]["rollout_count"]
    torch.testing.assert_close(
        engine.sampler.generator.get_state(),
        expected["sampler"]["rng_state"],
        rtol=0,
        atol=0,
    )
    assert objective.motion_ema_updates == expected["objective"]["motion_ema_updates"]
    assert objective.motion_head_teacher is not None
    for name, value in objective.motion_head_teacher.state_dict().items():
        torch.testing.assert_close(
            value,
            expected["objective"]["motion_head_teacher"][name],
            rtol=0,
            atol=0,
        )


def test_stage_one_weights_load_strictly_into_stage_two_topology() -> None:
    stage_one = _StudentModule(0.75)
    register_motion_head(
        stage_one,
        SpatialMotionHead(
            1,
            num_layers=2,
            kernel_size=1,
            hidden_dim=4,
            norm_num_groups=1,
        ),
    )
    stage_two = copy.deepcopy(stage_one)
    with torch.no_grad():
        for parameter in stage_two.parameters():
            parameter.zero_()
    wrapped = {
        "generator": {
            f"model.{name}": value.detach().clone()
            for name, value in stage_one.state_dict().items()
        }
    }
    load_diagonal_stage_weights(stage_two, wrapped)
    for name, value in stage_one.state_dict().items():
        torch.testing.assert_close(stage_two.state_dict()[name], value, rtol=0, atol=0)

    broken = copy.deepcopy(wrapped)
    broken["generator"].pop("model.scale")
    with pytest.raises(RuntimeError, match="missing"):
        load_diagonal_stage_weights(stage_two, broken)
