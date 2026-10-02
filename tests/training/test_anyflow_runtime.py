from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.post_training.distillation.anyflow import (
    AnyFlowDecisionRNG,
    AnyFlowEMA,
    AnyFlowFARConfig,
    AnyFlowMapConfig,
    AnyFlowOnPolicyConfig,
    AnyFlowPretrainConfig,
    AnyFlowRolloutChoice,
    AnyFlowTrainingBatch,
    NativeAnyFlowBidirectionalAdapter,
    NativeAnyFlowFARAdapter,
    NativeAnyFlowOnPolicyEngine,
    NativeAnyFlowOnPolicyLossAdapter,
    NativeAnyFlowPretrainEngine,
    NativeAnyFlowPretrainLossAdapter,
    NativeAnyFlowScoreAdapter,
    anyflow_rollout,
)


class _ScalarModule(nn.Module):
    def __init__(self, value: float, *, trainable: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value), requires_grad=trainable)


class _FARAdapter:
    def __init__(self, value: float = 0.2) -> None:
        self.module = _ScalarModule(value)
        self.calls: list[tuple[int, bool, bool]] = []
        self.transitions: list[tuple[int, float, float]] = []
        self.commits: list[int] = []
        self.bidirectional_calls: list[tuple[bool, bool, int]] = []

    def create_rollout_state(
        self,
        *,
        partition: TemporalChunkPartition,
        reference: torch.Tensor,
    ) -> object:
        assert int(reference.shape[2]) == partition.frame_count
        return {"next": 0}

    def predict_flow_map(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        destination_timesteps: torch.Tensor,
        *,
        clean_latents: torch.Tensor,
        context_latents: torch.Tensor,
        partition: TemporalChunkPartition,
        sampled_chunk_count: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del (
            timesteps,
            destination_timesteps,
            clean_latents,
            context_latents,
            partition,
            sampled_chunk_count,
            sample_ids,
            conditioning,
            training,
        )
        branch_offset = 0.05 if branch == "negative" else 0.0
        return self.module.weight * noisy_latents + branch_offset

    def predict_bidirectional_velocity(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del timesteps, sample_ids, conditioning
        self.bidirectional_calls.append(
            (training, torch.is_grad_enabled(), int(noisy_latents.shape[2]))
        )
        branch_offset = 0.05 if branch == "negative" else 0.0
        return self.module.weight * noisy_latents + branch_offset

    def rollout_velocity(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        destination_timesteps: torch.Tensor,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> torch.Tensor:
        del partition, sample_ids, conditioning
        assert isinstance(rollout_state, dict)
        assert rollout_state["next"] == chunk_index
        self.calls.append((chunk_index, training, torch.is_grad_enabled()))
        self.transitions.append(
            (
                chunk_index,
                float(timesteps[0, 0]),
                float(destination_timesteps[0, 0]),
            )
        )
        return self.module.weight * noisy_latents

    def commit_rollout_chunk(
        self,
        clean_prefix: torch.Tensor,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> None:
        del clean_prefix, partition, sample_ids, conditioning
        assert not torch.is_grad_enabled()
        assert isinstance(rollout_state, dict)
        assert rollout_state["next"] == chunk_index
        rollout_state["next"] += 1
        self.commits.append(chunk_index)


class _ScoreAdapter:
    def __init__(self, value: float, *, trainable: bool) -> None:
        self.module = _ScalarModule(value, trainable=trainable)
        self.calls: list[tuple[tuple[str, ...], float, float]] = []

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del conditioning, training
        self.calls.append(
            (
                sample_ids,
                float(timesteps.min().item()),
                float(timesteps.max().item()),
            )
        )
        offset = -0.1 if branch == "negative" else 0.0
        return self.module.weight * noisy_latents + offset


def _far_config() -> AnyFlowFARConfig:
    return AnyFlowFARConfig(
        chunk_partition=(1, 1),
        full_chunk_limit=1,
        patch_size=(1, 1, 1),
        compressed_patch_size=(1, 1, 1),
        long_context_training_ratio=0.0,
    )


def _map_config() -> AnyFlowMapConfig:
    return AnyFlowMapConfig(
        num_train_timesteps=10,
        timestep_shift=1.0,
        central_difference_epsilon=1.0,
        diffusion_ratio=1.0,
        consistency_ratio=0.0,
        fused_guidance_scale=1.0,
    )


def _batch(offset: float = 0.0) -> AnyFlowTrainingBatch:
    clean = torch.linspace(-0.8 + offset, 0.9 + offset, 16).reshape(2, 1, 2, 2, 2)
    return AnyFlowTrainingBatch(
        sample_ids=("a", "b"),
        clean_latents=clean,
        conditioning={"context": torch.ones(2, 2, 3)},
        unconditional_conditioning={"context": torch.zeros(2, 2, 3)},
    )


def test_rollout_is_chunk_autoregressive_and_uses_official_compressed_schedule() -> None:
    student = _FARAdapter()
    config = AnyFlowOnPolicyConfig(
        flow_map=_map_config(),
        far=_far_config(),
        inference_steps=(8,),
        ema_warmup_steps=0,
    )
    result = anyflow_rollout(
        student,
        _batch(),
        torch.ones_like(_batch().clean_latents),
        AnyFlowRolloutChoice(step_count=8, gradient_interval=3),
        config,
        differentiable=True,
    )
    assert student.commits == [0]
    assert student.calls == [
        (0, True, True),
        (0, True, True),
        (0, True, True),
        (1, True, True),
        (1, True, True),
        (1, True, True),
    ]
    assert student.transitions == [
        (0, 10.0, 6.25),
        (0, 6.25, 5.0),
        (0, 5.0, 0.0),
        (1, 10.0, 6.25),
        (1, 6.25, 5.0),
        (1, 5.0, 0.0),
    ]
    result.square().mean().backward()
    assert student.module.weight.grad is not None


def test_pretrain_objective_executes_flowmap_target_and_only_trains_student() -> None:
    student = _FARAdapter()
    decisions = AnyFlowDecisionRNG(4)
    objective = NativeAnyFlowPretrainLossAdapter(
        student,
        AnyFlowPretrainConfig(flow_map=_map_config(), far=_far_config()),
        decisions,
    )
    result = objective.student_loss(
        _batch(),
        generator=torch.Generator().manual_seed(7),
    )
    assert result.loss.ndim == 0
    assert result.metrics["diffusion_samples"].item() == 2
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert decisions.draw_count == 3


def test_far_pretrain_executes_released_full_video_bidirectional_auxiliary() -> None:
    student = _FARAdapter()
    decisions = AnyFlowDecisionRNG(4)
    objective = NativeAnyFlowPretrainLossAdapter(
        student,
        AnyFlowPretrainConfig(
            flow_map=_map_config(),
            far=_far_config(),
            bidirectional_modeling_probability=1.0,
        ),
        decisions,
    )
    result = objective.student_loss(
        _batch(),
        generator=torch.Generator().manual_seed(7),
    )
    assert result.metrics["bidirectional_applied"] is True
    assert student.bidirectional_calls == [(True, True, 2)]
    assert result.metrics["bidirectional_loss"].item() > 0
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert decisions.draw_count == 3


def test_pretrain_engine_accumulates_and_restores_synchronized_decisions() -> None:
    student = _FARAdapter()
    decisions = AnyFlowDecisionRNG(9)
    objective = NativeAnyFlowPretrainLossAdapter(
        student,
        AnyFlowPretrainConfig(flow_map=_map_config(), far=_far_config()),
        decisions,
    )
    optimizer = torch.optim.AdamW(student.module.parameters(), lr=1e-3)
    engine = NativeAnyFlowPretrainEngine(
        student_module=student.module,
        loss_adapter=objective,
        optimizer=optimizer,
        decisions=decisions,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )
    result = engine.train_step(
        (_batch(), _batch(0.1)),
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.isfinite(result.loss)
    assert engine.global_step == 1
    state = engine.state_dict()
    assert state["decisions"]["draw_count"] == 6

    restored_student = _FARAdapter()
    restored_decisions = AnyFlowDecisionRNG(9)
    restored_objective = NativeAnyFlowPretrainLossAdapter(
        restored_student,
        AnyFlowPretrainConfig(flow_map=_map_config(), far=_far_config()),
        restored_decisions,
    )
    restored = NativeAnyFlowPretrainEngine(
        student_module=restored_student.module,
        loss_adapter=restored_objective,
        optimizer=torch.optim.AdamW(restored_student.module.parameters(), lr=1e-3),
        decisions=restored_decisions,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )
    restored.load_state_dict(state)
    assert restored.global_step == 1
    assert torch.equal(
        restored.decisions.generator.get_state(),
        decisions.generator.get_state(),
    )


def test_on_policy_engine_commits_generator_then_fresh_fake_score_updates() -> None:
    student = _FARAdapter(0.2)
    real = _ScoreAdapter(0.5, trainable=False)
    fake = _ScoreAdapter(-0.1, trainable=True)
    config = AnyFlowOnPolicyConfig(
        flow_map=_map_config(),
        far=_far_config(),
        inference_steps=(2,),
        dmd_batch_size=1,
        dmd_min_timestep=2.0,
        dmd_max_timestep=4.0,
        discriminator_update_ratio=2,
        ema_decay=0.9,
        ema_warmup_steps=0,
        synchronized_seed=13,
    )
    decisions = AnyFlowDecisionRNG(config.synchronized_seed)
    objective = NativeAnyFlowOnPolicyLossAdapter(
        student,
        real,
        fake,
        config,
        decisions,
    )
    ema = AnyFlowEMA(student.module, decay=config.ema_decay, warmup_steps=0)
    engine = NativeAnyFlowOnPolicyEngine(
        student_module=student.module,
        real_score_module=real.module,
        fake_score_module=fake.module,
        loss_adapter=objective,
        student_optimizer=torch.optim.AdamW(student.module.parameters(), lr=1e-3),
        fake_score_optimizer=torch.optim.AdamW(fake.module.parameters(), lr=1e-3),
        decisions=decisions,
        discriminator_update_ratio=config.discriminator_update_ratio,
        student_max_grad_norm=1.0,
        fake_score_max_grad_norm=1.0,
        student_ema=ema,
    )
    result = engine.train_step(
        _batch(),
        fake_score_batches=(_batch(0.1), _batch(0.2)),
        generator=torch.Generator().manual_seed(17),
    )
    assert torch.isfinite(result.generator_loss)
    assert torch.isfinite(result.fake_score_loss)
    assert engine.student_optimizer_steps == 1
    assert engine.fake_score_optimizer_steps == 2
    assert decisions.draw_count == 9
    assert all(call[0] == ("a",) for call in real.calls + fake.calls)
    assert all(2.0 <= call[1] <= call[2] <= 4.0 for call in real.calls + fake.calls)
    assert int(ema.optimizer_steps.item()) == 1
    assert real.module.weight.grad is None
    state = engine.state_dict()
    engine.load_state_dict(state)
    bad = dict(state)
    bad["fake_score_optimizer_steps"] = 1
    with pytest.raises(ValueError, match="update ratio"):
        engine.load_state_dict(bad)


class _OfficialShapeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
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
        self.calls: list[dict[str, object]] = []

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> object:
        self.calls.append({"shape": tuple(hidden_states.shape), **kwargs})
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


def test_native_module_adapters_enforce_layout_and_use_core_dual_cache() -> None:
    module = _OfficialShapeModule()
    far = NativeAnyFlowFARAdapter(module)
    score = NativeAnyFlowScoreAdapter(_OfficialShapeModule())
    bidirectional_module = _OfficialShapeModule()
    bidirectional_student = NativeAnyFlowBidirectionalAdapter(
        bidirectional_module
    )
    partition = _far_config().partition
    batch = _batch()
    state = far.create_rollout_state(
        partition=partition,
        reference=batch.clean_latents,
    )
    cache = state.cache
    assert cache.full.shape == (2, 2, 2, 1, 4, 2)
    current = batch.clean_latents[:, :, :1]
    times = torch.ones(2, 1)
    velocity = far.rollout_velocity(
        current,
        times,
        torch.zeros_like(times),
        partition=partition,
        chunk_index=0,
        rollout_state=state,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=True,
    )
    torch.testing.assert_close(velocity, current * 0.5)
    far.commit_rollout_chunk(
        current,
        partition=partition,
        chunk_index=0,
        rollout_state=state,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
    )
    assert cache.num_cached_chunks == 1
    bidirectional = score.predict_velocity(
        batch.clean_latents,
        torch.ones(2, 2),
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=True,
    )
    torch.testing.assert_close(bidirectional, batch.clean_latents * 0.5)
    destinations = torch.full((2, 2), 0.25)
    full_video = bidirectional_student.predict_flow_map(
        batch.clean_latents,
        torch.ones(2, 2),
        destinations,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=True,
    )
    torch.testing.assert_close(full_video, batch.clean_latents * 0.5)
    call = bidirectional_module.calls[-1]
    assert call["shape"] == (2, 2, 1, 2, 2)
    assert call["is_causal"] is False
    torch.testing.assert_close(call["r_timestep"], destinations)


def test_native_far_adapter_rejects_standard_non_flowmap_module() -> None:
    module = _OfficialShapeModule()
    del module.condition_embedder.delta_embedder
    adapter = NativeAnyFlowFARAdapter(module)
    with pytest.raises(TypeError, match="destination-time"):
        adapter.create_rollout_state(
            partition=_far_config().partition,
            reference=_batch().clean_latents,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("deltatime_type", "delta", "destination time r"),
        ("gate_value", 1.0, "0.25 time gate"),
    ),
)
def test_native_bidirectional_adapter_rejects_wrong_flowmap_parameterization(
    field: str,
    value: object,
    message: str,
) -> None:
    module = _OfficialShapeModule()
    setattr(module.config, field, value)
    with pytest.raises(ValueError, match=message):
        NativeAnyFlowBidirectionalAdapter(module)


def test_anyflow_ema_snapshots_during_warmup_then_decays() -> None:
    module = _ScalarModule(1.0)
    ema = AnyFlowEMA(module, decay=0.5, warmup_steps=2)
    module.weight.data.fill_(2.0)
    ema.update(module)
    module.weight.data.zero_()
    ema.copy_to(module)
    assert module.weight.item() == pytest.approx(2.0)
    module.weight.data.fill_(3.0)
    ema.update(module)
    module.weight.data.fill_(5.0)
    ema.update(module)
    module.weight.data.zero_()
    ema.copy_to(module)
    assert module.weight.item() == pytest.approx(4.0)
