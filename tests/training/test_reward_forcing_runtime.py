from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.data.rollout_cache import (  # noqa: E402
    RolloutConditionedPrompt,
)
from worldfoundry.training.data.rollout_manifest import (  # noqa: E402
    RolloutPromptRecord,
)
from worldfoundry.training.data.shared_conditioning import (  # noqa: E402
    CacheTensorDescriptor,
    SharedConditioningArtifact,
    SharedConditioningIdentity,
)
from worldfoundry.training.post_training.distillation.reward_forcing import (  # noqa: E402
    NativeRewardForcingDataLoader,
    NativeRewardForcingTrainingSession,
    RewardForcingAlgorithmSpec,
    RewardForcingConfig,
    RewardForcingTrainingBatch,
    WanRewardForcingChunkAdapter,
    build_native_reward_forcing_training_stack,
)
from worldfoundry.training.recipes.post_training.recipe import (  # noqa: E402
    PostTrainingRecipe,
)
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
    PromptSafetyAudit,
)


class _StudentModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.35))


class _Student:
    def __init__(self, *, checkpoint_identity: str = "student-checkpoint") -> None:
        self.module = _StudentModule()
        self.checkpoint_identity = checkpoint_identity

    def initialize_cache(self, reference, *, sample_ids, conditioning):
        del reference, sample_ids, conditioning
        return {}

    def audit_reward_forcing_cache(
        self,
        *,
        frames_per_block,
        local_attention_frames,
        ema_sink_frames,
        ema_sink_decay,
    ):
        assert frames_per_block == 1
        assert local_attention_frames == 2
        assert ema_sink_frames == 1
        assert ema_sink_decay == 0.999

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
        del timesteps, sigmas, block_index, start_frame, sample_ids, conditioning, cache, training
        return noisy_chunk * self.module.gain

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
        del clean_chunk, block_index, start_frame, sample_ids, conditioning
        return cache


class _ScoreModule(torch.nn.Module):
    def __init__(self, value: float, *, frozen: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        if frozen:
            self.requires_grad_(False)


class _Score:
    def __init__(
        self,
        value: float,
        *,
        frozen: bool = False,
        checkpoint_identity: str | None = None,
    ) -> None:
        self.module = _ScoreModule(value, frozen=frozen)
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
        del sigmas, sample_ids, conditioning, training
        branch_offset = -0.1 if branch == "negative" else 0.1
        return noisy_latents * 0.1 + self.module.weight + branch_offset

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
        return noisy_latents * self.module.weight


class _Decoder:
    def __init__(
        self,
        *,
        checkpoint_identity: str = "reward-decoder-checkpoint",
        trainable: bool = False,
    ) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.module.requires_grad_(trainable)
        self.checkpoint_identity = checkpoint_identity

    def decode_reward_videos(self, clean_latents, *, sample_ids, conditioning):
        del sample_ids, conditioning
        return clean_latents.expand(-1, 3, -1, -1, -1).tanh()


class _Reward:
    def __init__(
        self,
        *,
        checkpoint_identity: str = "motion-reward-checkpoint",
        owned_module: torch.nn.Module | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.checkpoint_identity = checkpoint_identity
        self.owned_module = owned_module
        self.calibration_mean = 1.1646
        self.calibration_std = 1.3811
        self.normalization_epsilon = 0.0

    def score_motion_quality(self, videos, batch):
        assert not videos.requires_grad
        self.calls.append(batch.sample_ids)
        return torch.linspace(
            -0.1,
            0.2,
            batch.batch_size,
            device=videos.device,
        )


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


def _recipe_mapping(
    *,
    accumulation: int = 1,
    interval: int = 5,
    ema_start: int = 0,
    student_learning_rate: float = 2.0e-6,
) -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {
            "id": "reward-forcing-test",
            "output_dir": "runs/reward-forcing-test",
        },
        "model": {
            "recipe": "wan2.1-t2v-1.3b",
            "checkpoint": "student-checkpoint",
        },
        "tuning": {"mode": "full"},
        "data": {"manifest": "data/prompts.jsonl", "shuffle": False},
        "algorithm": {
            "type": "reward-forcing",
            "real_score_checkpoint": "real-score-checkpoint",
            "fake_score_checkpoint": "fake-score-checkpoint",
            "reward_decoder_checkpoint": "reward-decoder-checkpoint",
            "motion_reward_checkpoint": "motion-reward-checkpoint",
            "denoising_timesteps": [1000.0, 500.0],
            "frames_per_block": 1,
            "training_frames": 2,
            "local_attention_frames": 2,
            "ema_sink_frames": 1,
            "generator_update_interval": interval,
            "ema_start_step": ema_start,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": student_learning_rate,
            "weight_decay": 1.0e-2,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": accumulation,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 4.0e-7,
            "weight_decay": 1.0e-2,
            "betas": [0.0, 0.999],
            "max_grad_norm": 10.0,
            "gradient_accumulation_steps": accumulation,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
        "checkpoint": {"save_every_steps": 0},
        "export": {"format": "safetensors"},
    }


def _recipe(
    *,
    accumulation: int = 1,
    interval: int = 5,
    ema_start: int = 0,
    student_learning_rate: float = 2.0e-6,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        _recipe_mapping(
            accumulation=accumulation,
            interval=interval,
            ema_start=ema_start,
            student_learning_rate=student_learning_rate,
        )
    )


def _batch(value: int, *, count: int = 1) -> RewardForcingTrainingBatch:
    return RewardForcingTrainingBatch(
        sample_ids=tuple(f"sample-{value}-{index}" for index in range(count)),
        clean_latents=torch.full((count, 1, 2, 1, 1), float(value)),
        conditioning={},
        unconditional_conditioning={},
        prompts=tuple(f"prompt {value}-{index}" for index in range(count)),
    )


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return _batch(self.cursor)

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _conditioned_prompt() -> RolloutConditionedPrompt:
    prompt = "a running horse"
    record = RolloutPromptRecord(
        prompt_id="horse",
        prompt=prompt,
        safety_audit=PromptSafetyAudit(
            prompt=prompt,
            unsafe_probabilities={name: 0.01 for name in SHIELDGEMMA_PROMPT_POLICIES},
            threshold=0.5,
        ),
    )
    identity = SharedConditioningIdentity(
        branch="rollout-horse",
        prompt=prompt,
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repository": "test/conditioner", "revision": "test"},
        tokenizer={"repository": "test/tokenizer", "revision": "test"},
        tensors={
            "context": CacheTensorDescriptor(
                dtype="float32",
                shape=(3, 4),
                layout="sequence-features",
            )
        },
    )
    return RolloutConditionedPrompt(
        record=record,
        conditioning={"context": torch.ones(3, 4)},
        artifact=SharedConditioningArtifact(
            identity=identity,
            object_size_bytes=100,
            object_path="shared-objects/rollout-horse.safetensors",
        ),
    )


class _PromptSource:
    def __init__(self) -> None:
        self.position = 0

    def __iter__(self):
        self.position += 1
        yield (_conditioned_prompt(),)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict) -> None:
        self.position = int(state_dict["position"])


def test_reward_forcing_loader_reuses_prompt_conditioning_and_keeps_reward_text() -> None:
    source = _PromptSource()
    loader = NativeRewardForcingDataLoader(
        source,
        latent_shape=(1, 2, 1, 1),
        device="cpu",
        dtype=torch.float32,
        shared_unconditional_conditioning={"context": torch.zeros(3, 4)},
    )
    batch = next(iter(loader))
    assert batch.sample_ids == ("horse",)
    assert batch.prompts == ("a running horse",)
    assert tuple(batch.clean_latents.shape) == (1, 1, 2, 1, 1)
    assert tuple(batch.conditioning["context"].shape) == (1, 3, 4)
    state = loader.state_dict()
    source.position = 9
    loader.load_state_dict(state)
    assert source.position == 1


def test_reward_forcing_recipe_is_strict_round_trippable_and_optimizer_owned() -> None:
    mapping = _recipe_mapping(accumulation=3)
    recipe = PostTrainingRecipe.from_mapping(mapping)

    assert isinstance(recipe.algorithm, RewardForcingAlgorithmSpec)
    assert recipe.algorithm.type == "reward-forcing"
    assert recipe.algorithm.motion_reward_calibration_mean == 1.1646
    assert recipe.optimizer.gradient_accumulation_steps == 3
    assert recipe.fake_score_optimizer is not None
    assert recipe.fake_score_optimizer.gradient_accumulation_steps == 3
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    unknown = copy.deepcopy(mapping)
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="algorithm contains unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)

    missing_fake = copy.deepcopy(mapping)
    missing_fake.pop("fake_score_optimizer")
    with pytest.raises(ValueError, match="requires fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(missing_fake)

    for optimizer_name in ("guidance_optimizer", "discriminator_optimizer"):
        extra = copy.deepcopy(mapping)
        extra[optimizer_name] = copy.deepcopy(mapping["optimizer"])
        with pytest.raises(ValueError, match="only accepts fake_score_optimizer"):
            PostTrainingRecipe.from_mapping(extra)


def _build(
    seed: int,
    *,
    accumulation: int,
    interval: int = 5,
    ema_start: int = 0,
    student_scheduler=None,
    fake_score_scheduler=None,
):
    torch.manual_seed(seed)
    student = _Student()
    real = _Score(0.75, frozen=True)
    fake = _Score(0.45)
    decoder = _Decoder()
    reward = _Reward()
    stack = build_native_reward_forcing_training_stack(
        _recipe(
            accumulation=accumulation,
            interval=interval,
            ema_start=ema_start,
        ),
        student=student,
        real_score=real,
        fake_score=fake,
        reward_decoder=decoder,
        motion_reward=reward,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        fused_adamw=False,
    )
    return stack, student, real, fake, decoder, reward


def test_builder_uses_released_optimizers_and_accepts_arbitrary_accumulation() -> None:
    stack, student, real, fake, decoder, reward = _build(3, accumulation=7)
    assert stack.student_optimizer.param_groups[0]["lr"] == 2.0e-6
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 4.0e-7
    assert stack.student_optimizer.param_groups[0]["betas"] == (0.0, 0.999)
    assert stack.fake_score_optimizer.param_groups[0]["betas"] == (0.0, 0.999)
    assert stack.student_optimizer.param_groups[0]["weight_decay"] == 1.0e-2
    assert stack.engine.gradient_accumulation_steps == 7
    assert stack.engine.generator_update_interval == 5
    assert isinstance(stack.recipe.algorithm, RewardForcingAlgorithmSpec)
    assert not student.module.training
    assert not real.module.training
    assert not fake.module.training
    assert not decoder.module.training

    fake.module = stack.engine.student_module
    with pytest.raises(ValueError, match="independently materialized"):
        build_native_reward_forcing_training_stack(
            _recipe(),
            student=student,
            real_score=real,
            fake_score=fake,
            reward_decoder=decoder,
            motion_reward=reward,
            fused_adamw=False,
        )


def test_builder_gates_every_checkpoint_and_frozen_module_role() -> None:
    for role in (
        "student",
        "real_score",
        "fake_score",
        "reward_decoder",
        "motion_reward",
    ):
        broken = {
            "student": _Student(),
            "real_score": _Score(0.75, frozen=True),
            "fake_score": _Score(0.45),
            "reward_decoder": _Decoder(),
            "motion_reward": _Reward(),
        }
        broken[role].checkpoint_identity = "wrong-checkpoint"
        with pytest.raises(ValueError, match="loaded checkpoint identity"):
            build_native_reward_forcing_training_stack(
                _recipe(),
                **broken,
                fused_adamw=False,
            )

    with pytest.raises(ValueError, match="real-score teacher must be frozen"):
        build_native_reward_forcing_training_stack(
            _recipe(),
            student=_Student(),
            real_score=_Score(0.75, frozen=False, checkpoint_identity="real-score-checkpoint"),
            fake_score=_Score(0.45),
            reward_decoder=_Decoder(),
            motion_reward=_Reward(),
            fused_adamw=False,
        )
    with pytest.raises(ValueError, match="reward decoder must be frozen"):
        build_native_reward_forcing_training_stack(
            _recipe(),
            student=_Student(),
            real_score=_Score(0.75, frozen=True),
            fake_score=_Score(0.45),
            reward_decoder=_Decoder(trainable=True),
            motion_reward=_Reward(),
            fused_adamw=False,
        )
    trainable_reward_module = torch.nn.Linear(1, 1, bias=False)
    with pytest.raises(ValueError, match="motion reward must be frozen"):
        build_native_reward_forcing_training_stack(
            _recipe(),
            student=_Student(),
            real_score=_Score(0.75, frozen=True),
            fake_score=_Score(0.45),
            reward_decoder=_Decoder(),
            motion_reward=_Reward(owned_module=trainable_reward_module),
            fused_adamw=False,
        )
    decoder = _Decoder()
    with pytest.raises(ValueError, match="independently materialized"):
        build_native_reward_forcing_training_stack(
            _recipe(),
            student=_Student(),
            real_score=_Score(0.75, frozen=True),
            fake_score=_Score(0.45),
            reward_decoder=decoder,
            motion_reward=_Reward(owned_module=decoder.module),
            fused_adamw=False,
        )


def test_wan_adapter_reuses_in_tree_ema_sink_and_audits_its_exact_behavior() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.magic_world.causal import (
        CausalWanModel,
    )

    model = CausalWanModel(
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=16,
        num_heads=1,
        num_layers=1,
        local_attn_size=9,
        sink_size=3,
    )
    adapter = WanRewardForcingChunkAdapter(model, RewardForcingConfig())
    assert adapter.frames_per_block == 3
    assert model.num_frame_per_block == 3
    attention = model.blocks[0].self_attn
    current_k = torch.full((1, 3, 1, 12), 2.0)
    current_v = torch.full((1, 3, 1, 12), 4.0)
    evicted_k = torch.full_like(current_k, 12.0)
    evicted_v = torch.full_like(current_v, 14.0)
    updated_k, updated_v = attention.incremental_update(
        evicted_k,
        evicted_v,
        current_k,
        current_v,
    )
    torch.testing.assert_close(updated_k, current_k * 0.999 + evicted_k * 0.001)
    torch.testing.assert_close(updated_v, current_v * 0.999 + evicted_v * 0.001)

    with pytest.raises(ValueError, match="decay differs"):
        WanRewardForcingChunkAdapter(
            model,
            RewardForcingConfig(ema_sink_decay=0.9),
        )


def test_session_implements_one_generator_per_five_fake_updates_with_fresh_batches() -> None:
    student_scheduler = _Counter()
    fake_score_scheduler = _Counter()
    stack, _, _, _, _, reward = _build(
        7,
        accumulation=2,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    events = []
    session = stack.build_session(
        loader,
        progress,
        event_sink=events.append,
    )
    assert isinstance(session, NativeRewardForcingTrainingSession)
    summary = session.run(
        max_steps=6,
        generator=torch.Generator().manual_seed(101),
    )
    assert summary.student_optimizer_steps == 2
    assert summary.fake_score_optimizer_steps == 6
    assert loader.cursor == 16
    assert progress.optimizer_steps == 6
    assert len(reward.calls) == 4
    assert student_scheduler.steps == 2
    assert fake_score_scheduler.steps == 6
    assert [event["generator_updated"] for event in events] == [
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    # Generator and critic each receive two independent microbatches when due.
    assert events[0]["microbatches"] == 4
    assert events[1]["microbatches"] == 2
    assert set(reward.calls[0]).isdisjoint(reward.calls[2])


def _checkpointable(seed: int):
    student_scheduler = _Counter()
    fake_score_scheduler = _Counter()
    stack, _, _, _, _, _ = _build(
        seed,
        accumulation=2,
        interval=2,
        ema_start=1,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(211)
    model = torch.nn.ModuleDict(
        {
            "student": stack.engine.student_module,
            "real_score": stack.engine.real_score_module,
            "fake_score": stack.engine.fake_score_module,
            "reward_decoder": stack.loss_adapter.reward_decoder.module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=stack.optimizers,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={"algorithm": "reward-forcing"},
        lr_scheduler=stack.scheduler_state,
        ema=stack.ema_state,
    )
    return (
        stack,
        loader,
        progress,
        generator,
        model,
        state,
        student_scheduler,
        fake_score_scheduler,
    )


def _clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensors(item) for key, item in value.items()}
    return value


def _assert_nested_close(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
        return
    assert actual == expected


def test_dcp_split_resume_restores_cadence_rng_optimizers_schedulers_ema_and_cursor(
    tmp_path: Path,
) -> None:
    baseline = _checkpointable(13)
    (
        stack,
        loader,
        progress,
        generator,
        model,
        state,
        student_scheduler,
        fake_score_scheduler,
    ) = baseline
    session = stack.build_session(loader, progress)
    session.run(max_steps=3, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected_summary = session.run(max_steps=3, generator=generator)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = _clone_tensors(stack.ema_state.state_dict())
    expected_schedulers = (student_scheduler.steps, fake_score_scheduler.steps)

    restored = _checkpointable(29)
    (
        restored_stack,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_fake_score_scheduler,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual_summary = restored_stack.build_session(
        restored_loader,
        restored_progress,
    ).run(max_steps=3, generator=restored_generator)

    assert restored_loader.cursor == 18
    assert restored_progress.optimizer_steps == 6
    assert actual_summary.final_generator_loss == expected_summary.final_generator_loss
    assert actual_summary.final_fake_score_loss == expected_summary.final_fake_score_loss
    assert (
        restored_student_scheduler.steps,
        restored_fake_score_scheduler.steps,
    ) == expected_schedulers
    _assert_nested_close(restored_stack.ema_state.state_dict(), expected_ema)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
