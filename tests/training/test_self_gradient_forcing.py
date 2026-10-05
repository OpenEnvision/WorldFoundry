from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDConfig,
    DMDTrainingBatch,
    FlowDMDLossAdapter,
    NativeDMDTrainEngine,
)
from worldfoundry.training.post_training.distillation.dmd.objective import (  # noqa: E402
    FewStepSchedule,
)
from worldfoundry.training.post_training.distillation.self_gradient_forcing import (  # noqa: E402
    NativeSelfGradientForcingTrainEngine,
    NativeSelfGradientForcingTrainingSession,
    SelfGradientForcingConfig,
    SelfGradientForcingSampler,
    WanSelfGradientForcingAdapter,
    build_native_self_gradient_forcing_training_stack,
)
from worldfoundry.training.recipes import (  # noqa: E402
    PostTrainingRecipe,
    SelfGradientForcingAlgorithmSpec,
)


class _StudentModule(torch.nn.Module):
    def __init__(self, value: float = 0.1) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))


class _RecordingAdapter:
    def __init__(
        self,
        value: float = 0.1,
        *,
        frozen: bool = False,
        checkpoint_identity: str = "student-checkpoint",
    ) -> None:
        self.module = _StudentModule(value)
        self.module.requires_grad_(not frozen)
        self.checkpoint_identity = checkpoint_identity
        self.chunk_calls: list[dict[str, object]] = []
        self.commits: list[dict[str, object]] = []
        self.teacher_calls: list[dict[str, object]] = []

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
        self.chunk_calls.append(
            {
                "block": block_index,
                "timestep": float(timesteps[0].item()),
                "training": training,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        value = timesteps.to(noisy_chunk.dtype).reshape(-1, 1, 1, 1, 1) / 10.0
        return torch.ones_like(noisy_chunk) * value * self.module.scale / 0.1 + cache["embedding"]

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
        zeros = torch.zeros(clean_chunk.shape[0], device=clean_chunk.device)
        return self.commit_context_chunk(
            clean_chunk,
            zeros,
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
        assert cache["active_block"] == block_index
        self.commits.append(
            {
                "block": block_index,
                "context": context_chunk.detach().clone(),
                "timesteps": context_timesteps.detach().clone(),
            }
        )
        cache["embedding"] = context_chunk.mean(dim=(1, 2, 3, 4), keepdim=True)
        cache["committed_blocks"] = block_index + 1
        return cache

    def predict_clean_teacher_forced(
        self,
        noisy_latents,
        timesteps,
        sigmas,
        *,
        clean_context,
        context_timesteps,
        sample_ids,
        conditioning,
        training,
    ):
        del sigmas, sample_ids, conditioning
        self.teacher_calls.append(
            {
                "timesteps": timesteps.detach().clone(),
                "context": clean_context,
                "context_timesteps": context_timesteps.detach().clone(),
                "training": training,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        temporal_context = clean_context.cumsum(dim=2)
        return (noisy_latents + temporal_context) * self.module.scale


class _ScoreAdapter:
    def __init__(
        self,
        value: float,
        *,
        frozen: bool = False,
        checkpoint_identity: str | None = None,
    ) -> None:
        self.module = _StudentModule(value)
        if frozen:
            self.module.requires_grad_(False)
        self.checkpoint_identity = checkpoint_identity or (
            "real-score-checkpoint" if frozen else "fake-score-checkpoint"
        )

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
        del sigmas, sample_ids, conditioning, training, branch
        return noisy_latents * self.module.scale

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


def _schedule() -> FewStepSchedule:
    return FewStepSchedule(timesteps=(10.0, 5.0), sigmas=(0.8, 0.4))


def _config(**overrides) -> SelfGradientForcingConfig:
    values = {
        "schedule": _schedule(),
        "frames_per_block": 2,
        "frame_dim": 2,
        "context_timestep": 2.0,
        "context_sigma": 0.2,
        "cache_target_mode": "exit",
        "exit_step_rank_mode": "local",
        "match_context": True,
    }
    values.update(overrides)
    return SelfGradientForcingConfig(**values)


def _batch(prefix: str = "video") -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=(prefix,),
        clean_latents=torch.zeros(1, 1, 4, 1, 1),
        conditioning={},
        unconditional_conditioning={},
    )


def test_released_schedule_and_context_noise_fixture() -> None:
    chunkwise = SelfGradientForcingConfig.from_raw_timesteps(frames_per_block=3)
    framewise = SelfGradientForcingConfig.from_raw_timesteps(frames_per_block=1)
    assert chunkwise.schedule.timesteps == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 625.0)
    )
    assert chunkwise.schedule.sigmas == pytest.approx((1.0, 0.9375, 5.0 / 6.0, 0.625))
    assert chunkwise.context_timestep == chunkwise.context_sigma == 0.0
    assert chunkwise.cache_target_mode == "exit"
    assert chunkwise.exit_step_rank_mode == "local"
    assert chunkwise.match_context is True
    assert chunkwise.frames_per_block == 3
    assert framewise.frames_per_block == 1


def test_two_pass_replay_is_no_grad_then_one_live_parallel_forward() -> None:
    adapter = _RecordingAdapter()
    sampler = SelfGradientForcingSampler(adapter, _config(), seed=7)
    replay = sampler.replay(
        _batch(),
        torch.zeros(1, 1, 4, 1, 1),
        exit_index=0,
        training=True,
    )

    assert len(adapter.chunk_calls) == 4
    assert all(call["training"] is False for call in adapter.chunk_calls)
    assert all(call["grad_enabled"] is False for call in adapter.chunk_calls)
    assert len(adapter.commits) == 2
    assert all(torch.all(call["timesteps"] == 2.0) for call in adapter.commits)
    assert len(adapter.teacher_calls) == 1
    teacher = adapter.teacher_calls[0]
    assert teacher["training"] is True
    assert teacher["grad_enabled"] is True
    torch.testing.assert_close(teacher["context"], replay.context_latents)
    assert not teacher["context"].requires_grad
    torch.testing.assert_close(teacher["context_timesteps"], torch.full((1, 4), 2.0))
    assert replay.clean_latents.requires_grad
    assert not replay.noisy_at_exit.requires_grad
    assert not replay.cache_targets.requires_grad
    assert not replay.context_latents.requires_grad
    assert [block.frame_start for block in replay.cache_state.blocks] == [0, 2]

    replay.clean_latents.sum().backward()
    assert adapter.module.scale.grad is not None
    assert torch.isfinite(adapter.module.scale.grad)


def test_cache_target_modes_and_synchronized_exit_short_circuit() -> None:
    batch = _batch()
    noise = torch.zeros_like(batch.clean_latents)

    exit_adapter = _RecordingAdapter()
    exit_sampler = SelfGradientForcingSampler(
        exit_adapter,
        _config(context_sigma=0.0, cache_target_mode="exit"),
        seed=11,
    )
    exit_replay = exit_sampler.replay(batch, noise, exit_index=0, training=False)

    final_adapter = _RecordingAdapter()
    final_sampler = SelfGradientForcingSampler(
        final_adapter,
        _config(context_sigma=0.0, cache_target_mode="final-clean"),
        seed=11,
    )
    final_replay = final_sampler.replay(batch, noise, exit_index=0, training=False)

    assert len(exit_adapter.chunk_calls) == len(final_adapter.chunk_calls) == 4
    assert not torch.equal(exit_replay.cache_targets, final_replay.cache_targets)
    torch.testing.assert_close(exit_replay.cache_targets[:, :, :2], torch.ones(1, 1, 2, 1, 1))
    torch.testing.assert_close(final_replay.cache_targets[:, :, :2], torch.full((1, 1, 2, 1, 1), 0.5))

    synchronized_adapter = _RecordingAdapter()
    synchronized = SelfGradientForcingSampler(
        synchronized_adapter,
        _config(
            context_sigma=0.0,
            cache_target_mode="exit",
            exit_step_rank_mode="synchronized",
        ),
        seed=11,
    )
    synchronized.replay(batch, noise, exit_index=0, training=False)
    assert len(synchronized_adapter.chunk_calls) == 2


def test_unmatched_context_replays_clean_cache_targets_at_zero_timestep() -> None:
    adapter = _RecordingAdapter()
    sampler = SelfGradientForcingSampler(
        adapter,
        _config(match_context=False, context_sigma=0.4),
        seed=13,
    )
    replay = sampler.replay(
        _batch(),
        torch.zeros(1, 1, 4, 1, 1),
        exit_index=1,
        training=True,
    )
    call = adapter.teacher_calls[0]
    torch.testing.assert_close(call["context"], replay.cache_targets)
    torch.testing.assert_close(call["context_timesteps"], torch.zeros(1, 4))


def test_sampler_rng_and_exit_state_restore_exactly() -> None:
    batch = _batch()
    first_adapter = _RecordingAdapter()
    first = SelfGradientForcingSampler(first_adapter, _config(), seed=17)
    first.sample(batch, first.config.schedule, generator=None, training=False)
    state = first.state_dict()
    expected = first.sample(batch, first.config.schedule, generator=first.generator, training=False)
    expected_exit = first.last_exit_index

    restored_adapter = _RecordingAdapter()
    restored = SelfGradientForcingSampler(restored_adapter, _config(), seed=999)
    restored.load_state_dict(state)
    actual = restored.sample(batch, restored.config.schedule, generator=restored.generator, training=False)
    assert restored.last_exit_index == expected_exit
    torch.testing.assert_close(actual.clean_latents, expected.clean_latents, rtol=0, atol=0)


def test_rank_modes_use_local_or_broadcast_exit_sampling(monkeypatch) -> None:
    batch = _batch()
    calls: list[tuple[int, object]] = []

    def broadcast(value, *, src, group):
        calls.append((src, group))
        value.fill_(1)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    parallel = SimpleNamespace(rank=1, world_size=2, process_group=None)
    synchronized = SelfGradientForcingSampler(
        _RecordingAdapter(),
        _config(exit_step_rank_mode="synchronized"),
        parallel_context=parallel,
        seed=19,
    )
    assert synchronized.sample_exit_index(batch.clean_latents) == 1
    assert calls == [(0, None)]

    calls.clear()
    local = SelfGradientForcingSampler(
        _RecordingAdapter(),
        _config(exit_step_rank_mode="local"),
        parallel_context=parallel,
        seed=19,
    )
    assert local.sample_exit_index(batch.clean_latents) in {0, 1}
    assert calls == []


class _TinyCausalWan(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.25))
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.dim = 4
        self.local_attn_size = -1
        self.model_type = "t2v"
        self.num_heads = 1
        self.num_layers = 2
        self.patch_size = (1, 1, 1)
        self.text_len = 3
        self.teacher_context: torch.Tensor | None = None
        self.teacher_aug_t: torch.Tensor | None = None

    def forward(self, *, x, t, kv_cache=None, clean_x=None, aug_t=None, current_start=0, **kwargs):
        del kwargs, t
        if kv_cache is not None:
            for cache in kv_cache:
                end = current_start + x.shape[2] * x.shape[3] * x.shape[4]
                cache["global_end_index"].fill_(end)
                cache["local_end_index"].fill_(end)
            return x * self.gain
        self.teacher_context = clean_x
        self.teacher_aug_t = aug_t
        assert clean_x is not None
        return (x + clean_x) * self.gain


def test_native_wan_bridge_executes_noisy_commit_and_parallel_teacher_forcing() -> None:
    graph = _TinyCausalWan()
    adapter = WanSelfGradientForcingAdapter(graph, frames_per_block=2)
    config = _config(context_timestep=2.0, context_sigma=0.2)
    sampler = SelfGradientForcingSampler(adapter, config, seed=23)
    batch = DMDTrainingBatch(
        sample_ids=("wan",),
        clean_latents=torch.zeros(1, 1, 4, 1, 1),
        conditioning={"context": torch.ones(1, 3, 4)},
        unconditional_conditioning={},
    )
    replay = sampler.replay(
        batch,
        torch.ones_like(batch.clean_latents),
        exit_index=0,
        training=True,
    )
    assert graph.teacher_context is not None
    assert graph.teacher_aug_t is not None
    torch.testing.assert_close(graph.teacher_context, replay.context_latents)
    torch.testing.assert_close(graph.teacher_aug_t, torch.full((1, 4), 2.0))
    replay.clean_latents.sum().backward()
    assert graph.gain.grad is not None


def _engine_stack(*, seed: int, accumulation: int = 2):
    student = _RecordingAdapter()
    sampler = SelfGradientForcingSampler(student, _config(), seed=seed)
    real_score = _ScoreAdapter(0.3, frozen=True)
    fake_score = _ScoreAdapter(0.2)
    dmd_config = DMDConfig(
        schedule=sampler.config.schedule,
        shared_score_timestep=False,
        per_sample_normalization=True,
    )
    objective = FlowDMDLossAdapter(
        None,
        real_score,
        fake_score,
        dmd_config,
        student_sampler=sampler,
    )
    dmd_engine = NativeDMDTrainEngine(
        student_module=student.module,
        real_score_module=real_score.module,
        fake_score_module=fake_score.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.SGD(student.module.parameters(), lr=0.01),
        fake_score_optimizer=torch.optim.SGD(fake_score.module.parameters(), lr=0.01),
        generator_update_interval=1,
        gradient_accumulation_steps=accumulation,
        student_max_grad_norm=100.0,
        fake_score_max_grad_norm=100.0,
    )
    return NativeSelfGradientForcingTrainEngine(dmd_engine, sampler)


def test_dmd_engine_composition_accumulates_and_checkpoints_replay_rng() -> None:
    engine = _engine_stack(seed=29)
    result = engine.train_step((_batch("first"), _batch("second")))
    assert result.generator_loss.isfinite()
    assert result.fake_score_loss.isfinite()
    assert engine.global_step == 1
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 1
    assert engine.sampler.rollout_count == 4
    state = engine.state_dict()
    assert set(state) == {"schema", "global_step", "dmd_engine", "sampler"}

    restored = _engine_stack(seed=999)
    restored.load_state_dict(state)
    assert restored.global_step == engine.global_step
    assert restored.sampler.rollout_count == engine.sampler.rollout_count
    assert torch.equal(
        restored.sampler.generator.get_state(),
        engine.sampler.generator.get_state(),
    )
    with pytest.raises(ValueError, match="owns its checkpointed generator"):
        restored.train_step(
            (_batch("third"), _batch("fourth")),
            generator=torch.Generator().manual_seed(1),
        )


def _recipe_mapping() -> dict[str, object]:
    return {
        "run": {"id": "self-gradient-forcing-test", "output_dir": "unused"},
        "model": {
            "recipe": "wan2.1-t2v-1.3b-causal",
            "checkpoint": "student-checkpoint",
        },
        "tuning": {"mode": "full"},
        "data": {"manifest": "prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "self-gradient-forcing",
            "real_score_checkpoint": "real-score-checkpoint",
            "fake_score_checkpoint": "fake-score-checkpoint",
            "denoising_timesteps": [10.0, 5.0],
            "denoising_flow_shift": 1.0,
            "num_train_timesteps": 10,
            "frames_per_block": 2,
            "frame_dim": 2,
            "context_timestep": 2.0,
            "cache_target_mode": "exit",
            "exit_step_rank_mode": "local",
            "match_context": True,
            "last_step_only": False,
            "score_min_sigma": 0.1,
            "score_max_sigma": 0.9,
            "score_flow_shift": 1.0,
            "teacher_guidance_scale": 3.0,
            "normalization_epsilon": 0.0,
            "generator_update_interval": 1,
            "student_scheduler_cadence": "generator-update",
            "ema_decay": 0.99,
            "ema_start_step": 0,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 2.0e-6,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 4.0e-7,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": 2,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "export": {"format": "safetensors"},
    }


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> DMDTrainingBatch:
        value = _batch(f"video-{self.cursor}")
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _native_stack():
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    student = _RecordingAdapter()
    real_score = _ScoreAdapter(0.3, frozen=True)
    fake_score = _ScoreAdapter(0.2)
    stack = build_native_self_gradient_forcing_training_stack(
        recipe,
        student=student,
        real_score=real_score,
        fake_score=fake_score,
        fused_adamw=False,
    )
    return stack, student, fake_score


def _checkpointable_native_stack():
    torch.manual_seed(101)
    stack, student, fake_score = _native_stack()
    loader = _StatefulLoader()
    progress = TrainingProgress()
    model = torch.nn.ModuleDict(
        {"student": student.module, "fake_score": fake_score.module}
    )
    state = TrainingState(
        model=model,
        optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
        engine=stack.engine,
        dataloader=loader,
        objective_generator=stack.sampler.generator,
        progress=progress,
        identity={
            "algorithm": "self-gradient-forcing",
        },
        **stack.checkpoint_state_kwargs(),
    )
    return stack, loader, progress, model, state


def test_self_gradient_forcing_recipe_builder_and_session_are_behavior_bound() -> None:
    stack, _, _ = _native_stack()
    assert isinstance(stack.recipe.algorithm, SelfGradientForcingAlgorithmSpec)
    assert stack.rollout_config.schedule.timesteps == pytest.approx((10.0, 5.0))
    assert stack.rollout_config.context_timestep == pytest.approx(2.0)
    assert stack.rollout_config.context_sigma == pytest.approx(0.2)
    assert stack.dmd_config.teacher_guidance_scale == pytest.approx(3.0)
    assert stack.engine.gradient_accumulation_steps == 2
    assert stack.engine.generator_update_interval == 1
    assert stack.student_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-6)
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == pytest.approx(4.0e-7)

    loader = _StatefulLoader()
    progress = TrainingProgress()
    events: list[dict[str, object]] = []
    summary = NativeSelfGradientForcingTrainingSession(
        stack.engine,
        loader,
        progress,
        event_sink=lambda value: events.append(dict(value)),
    ).run(max_steps=1)
    assert summary.final_step == 1
    assert progress.microbatches_seen == 2
    assert loader.cursor == 2
    assert events[0]["schema"] == "worldfoundry-self-gradient-forcing-step-event"


def test_self_gradient_forcing_builder_rejects_role_checkpoint_mismatch() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    with pytest.raises(ValueError, match="fake-score critic loaded checkpoint identity"):
        build_native_self_gradient_forcing_training_stack(
            recipe,
            student=_RecordingAdapter(),
            real_score=_ScoreAdapter(0.3, frozen=True),
            fake_score=_ScoreAdapter(
                0.2,
                checkpoint_identity="wrong-checkpoint",
            ),
            fused_adamw=False,
        )


def test_self_gradient_forcing_dcp_exact_resume(
    tmp_path: Path,
) -> None:
    stack, loader, progress, model, state = _checkpointable_native_stack()
    session = NativeSelfGradientForcingTrainingSession(stack.engine, loader, progress)
    session.run(max_steps=1)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)

    expected = session.run(max_steps=1)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_rng = stack.sampler.generator.get_state().clone()

    restored, restored_loader, restored_progress, restored_model, restored_state = (
        _checkpointable_native_stack()
    )
    manager.load(restored_state, artifact.path)
    actual = NativeSelfGradientForcingTrainingSession(
        restored.engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1)

    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    torch.testing.assert_close(
        restored.sampler.generator.get_state(),
        expected_rng,
        rtol=0,
        atol=0,
    )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
