from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import multiprocessing as multiprocessing_module
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers import get_constant_schedule_with_warmup

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.post_training.distillation.anyflow import (
    AnyFlowBidirectionalOnPolicyConfig,
    AnyFlowBidirectionalPretrainConfig,
    AnyFlowDecisionRNG,
    AnyFlowEMA,
    AnyFlowMapConfig,
    AnyFlowRolloutChoice,
    AnyFlowTrainingBatch,
    NativeAnyFlowBidirectionalOnPolicyLossAdapter,
    NativeAnyFlowBidirectionalPretrainLossAdapter,
    NativeAnyFlowOnPolicyEngine,
    NativeAnyFlowPretrainEngine,
    ProcessGroupAnyFlowTensorSynchronizer,
    anyflow_bidirectional_rollout,
)


class _ScalarModule(nn.Module):
    def __init__(self, value: float, *, trainable: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value), requires_grad=trainable)


class _BidirectionalAdapter:
    def __init__(self, value: float = 0.2) -> None:
        self.module = _ScalarModule(value)
        self.flowmap_calls: list[dict[str, object]] = []
        self.rollout_calls: list[tuple[bool, bool, float, float]] = []

    def predict_flow_map(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        destination_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sample_ids
        context = conditioning["context"]
        assert isinstance(context, torch.Tensor)
        self.flowmap_calls.append(
            {
                "noisy": noisy_latents.detach().clone(),
                "timesteps": timesteps.detach().clone(),
                "destinations": destination_timesteps.detach().clone(),
                "training": training,
                "grad_enabled": torch.is_grad_enabled(),
                "branch": branch,
                "context": context.detach().clone(),
            }
        )
        delta = (timesteps - destination_timesteps).reshape(
            int(noisy_latents.shape[0]),
            1,
            int(noisy_latents.shape[2]),
            1,
            1,
        )
        offset = 0.01 if branch == "negative" else 0.0
        return self.module.weight * noisy_latents + delta.to(noisy_latents) * 0.001 + offset

    def rollout_velocity(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        destination_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> torch.Tensor:
        del sample_ids, conditioning
        self.rollout_calls.append(
            (
                training,
                torch.is_grad_enabled(),
                float(timesteps[0, 0]),
                float(destination_timesteps[0, 0]),
            )
        )
        return self.module.weight * noisy_latents


class _ScoreAdapter:
    def __init__(self, value: float, *, trainable: bool) -> None:
        self.module = _ScalarModule(value, trainable=trainable)

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
        del timesteps, sample_ids, conditioning, training
        offset = -0.05 if branch == "negative" else 0.0
        return self.module.weight * noisy_latents + offset


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
    clean = torch.linspace(-0.8 + offset, 0.9 + offset, 24).reshape(2, 1, 3, 2, 2)
    return AnyFlowTrainingBatch(
        sample_ids=("a", "b"),
        clean_latents=clean,
        conditioning={"context": torch.ones(2, 2, 3)},
        unconditional_conditioning={"context": torch.zeros(2, 2, 3)},
    )


def test_bidirectional_pretrain_matches_full_video_i2v_corruption_formula() -> None:
    student = _BidirectionalAdapter()
    decisions = AnyFlowDecisionRNG(3)
    objective = NativeAnyFlowBidirectionalPretrainLossAdapter(
        student,
        AnyFlowBidirectionalPretrainConfig(
            flow_map=_map_config(),
            image_conditioning_probability=1.0,
            conditioning_dropout_probability=0.0,
        ),
        decisions,
    )
    seed = 19
    result = objective.student_loss(
        _batch(),
        generator=torch.Generator().manual_seed(seed),
    )

    replay = torch.Generator().manual_seed(seed)
    clean = _batch().clean_latents
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=replay)
    first = torch.rand((2,), dtype=clean.dtype, generator=replay)
    second = torch.rand((2,), dtype=clean.dtype, generator=replay)
    expected_time = torch.maximum(first, second)[:, None].expand(2, 3).clone()
    expected_time[:, 0] = 0
    expected_noisy = expected_time[:, None, :, None, None] * noise + (
        1 - expected_time[:, None, :, None, None]
    ) * clean

    first_call = student.flowmap_calls[0]
    torch.testing.assert_close(first_call["noisy"], expected_noisy, rtol=0, atol=0)
    torch.testing.assert_close(
        first_call["timesteps"][:, 0],
        torch.zeros(2),
        rtol=0,
        atol=0,
    )
    assert bool(torch.all(first_call["destinations"][:, 0] > 0))
    assert result.metrics["first_frame_conditioned"] is True
    assert decisions.draw_count == 1
    result.loss.backward()
    assert student.module.weight.grad is not None


def test_bidirectional_pretrain_applies_released_text_conditioning_dropout() -> None:
    student = _BidirectionalAdapter()
    decisions = AnyFlowDecisionRNG(5)
    objective = NativeAnyFlowBidirectionalPretrainLossAdapter(
        student,
        AnyFlowBidirectionalPretrainConfig(
            flow_map=_map_config(),
            conditioning_dropout_probability=1.0,
        ),
        decisions,
    )
    batch = _batch()
    result = objective.student_loss(
        batch,
        generator=torch.Generator().manual_seed(31),
    )

    torch.testing.assert_close(
        student.flowmap_calls[0]["context"],
        torch.zeros(2, 2, 3),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(batch.conditioning["context"], torch.ones(2, 2, 3))
    assert result.metrics["conditioning_dropped_samples"].item() == 2
    assert decisions.draw_count == 1


def test_bidirectional_rollout_has_no_far_cache_and_uses_compressed_schedule() -> None:
    student = _BidirectionalAdapter()
    config = AnyFlowBidirectionalOnPolicyConfig(
        flow_map=_map_config(),
        inference_steps=(8,),
        dmd_batch_size=2,
    )
    batch = _batch()
    result = anyflow_bidirectional_rollout(
        student,
        batch,
        torch.ones_like(batch.clean_latents),
        AnyFlowRolloutChoice(step_count=8, gradient_interval=3),
        config,
        differentiable=True,
    )
    assert student.rollout_calls == [
        (True, True, 10.0, 6.25),
        (True, True, 6.25, 5.0),
        (True, True, 5.0, 0.0),
    ]
    assert not hasattr(student, "create_rollout_state")
    result.square().mean().backward()
    assert student.module.weight.grad is not None

    student.rollout_calls.clear()
    detached = anyflow_bidirectional_rollout(
        student,
        batch,
        torch.ones_like(batch.clean_latents),
        AnyFlowRolloutChoice(step_count=8, gradient_interval=0),
        config,
        differentiable=False,
    )
    assert student.rollout_calls == [
        (False, False, 10.0, 8.75),
        (False, False, 8.75, 0.0),
    ]
    assert not detached.requires_grad


def test_bidirectional_pretrain_engine_resume_uses_one_decision_per_microbatch() -> None:
    student = _BidirectionalAdapter()
    decisions = AnyFlowDecisionRNG(23)
    objective = NativeAnyFlowBidirectionalPretrainLossAdapter(
        student,
        AnyFlowBidirectionalPretrainConfig(flow_map=_map_config()),
        decisions,
    )
    engine = NativeAnyFlowPretrainEngine(
        student_module=student.module,
        loss_adapter=objective,
        optimizer=torch.optim.AdamW(student.module.parameters(), lr=1.0e-3),
        decisions=decisions,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )
    engine.train_step(
        (_batch(), _batch(0.1)),
        generator=torch.Generator().manual_seed(29),
    )
    state = engine.state_dict()
    assert state["decisions"]["draw_count"] == 2

    restored_student = _BidirectionalAdapter()
    restored_decisions = AnyFlowDecisionRNG(23)
    restored_objective = NativeAnyFlowBidirectionalPretrainLossAdapter(
        restored_student,
        AnyFlowBidirectionalPretrainConfig(flow_map=_map_config()),
        restored_decisions,
    )
    restored = NativeAnyFlowPretrainEngine(
        student_module=restored_student.module,
        loss_adapter=restored_objective,
        optimizer=torch.optim.AdamW(restored_student.module.parameters(), lr=1.0e-3),
        decisions=restored_decisions,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )
    restored.load_state_dict(state)
    assert torch.equal(
        restored_decisions.generator.get_state(),
        decisions.generator.get_state(),
    )


def _distributed_decision_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    queue: multiprocessing_module.Queue,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        decisions = AnyFlowDecisionRNG(
            41 + rank,
            synchronizer=ProcessGroupAnyFlowTensorSynchronizer(),
        )
        reference = torch.zeros(1)
        values = (
            decisions.choice((2, 4, 8), reference=reference),
            decisions.randrange(7, reference=reference),
            decisions.bernoulli(0.5, reference=reference),
            decisions.draw_count,
        )
        queue.put((rank, values))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_bidirectional_decisions_broadcast_across_arbitrary_world_size(tmp_path: Path) -> None:
    context = multiprocessing_module.get_context("spawn")
    queue = context.Queue()
    rendezvous = str(tmp_path / "anyflow-gloo")
    processes = [
        context.Process(
            target=_distributed_decision_worker,
            args=(rank, 2, rendezvous, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=5) for _ in processes)
    assert results[0][1] == results[1][1]
    assert results[0][1][-1] == 3


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self) -> _StatefulLoader:
        return self

    def __next__(self) -> AnyFlowTrainingBatch:
        batch = _batch(float(self.cursor) / 20.0)
        self.cursor += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_bidirectional_stack(objective_seed: int):
    student = _BidirectionalAdapter(0.2)
    real = _ScoreAdapter(0.5, trainable=False)
    fake = _ScoreAdapter(-0.1, trainable=True)
    config = AnyFlowBidirectionalOnPolicyConfig(
        flow_map=_map_config(),
        inference_steps=(2,),
        dmd_batch_size=2,
        image_conditioning_probability=0.0,
        discriminator_update_ratio=1,
        ema_decay=0.9,
        ema_warmup_steps=0,
        synchronized_seed=37,
    )
    decisions = AnyFlowDecisionRNG(config.synchronized_seed)
    objective = NativeAnyFlowBidirectionalOnPolicyLossAdapter(
        student,
        real,
        fake,
        config,
        decisions,
    )
    ema = AnyFlowEMA(student.module, decay=config.ema_decay, warmup_steps=0)
    student_optimizer = torch.optim.AdamW(student.module.parameters(), lr=1.0e-3)
    fake_optimizer = torch.optim.AdamW(fake.module.parameters(), lr=1.0e-3)
    student_scheduler = get_constant_schedule_with_warmup(
        student_optimizer,
        num_warmup_steps=3,
    )
    fake_scheduler = get_constant_schedule_with_warmup(
        fake_optimizer,
        num_warmup_steps=3,
    )
    engine = NativeAnyFlowOnPolicyEngine(
        student_module=student.module,
        real_score_module=real.module,
        fake_score_module=fake.module,
        loss_adapter=objective,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        decisions=decisions,
        discriminator_update_ratio=config.discriminator_update_ratio,
        student_max_grad_norm=1.0,
        fake_score_max_grad_norm=1.0,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_ema=ema,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(objective_seed)
    model = nn.ModuleDict(
        {
            "student": student.module,
            "real_score": real.module,
            "fake_score": fake.module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(student_optimizer, fake_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={
            "algorithm": "anyflow-bidirectional",
            "configuration": engine.config_state,
            "gradient_accumulation_steps": engine.gradient_accumulation_steps,
        },
        lr_scheduler=NamedStatefulCollection(
            {
                "student": student_scheduler,
                "fake_score": fake_scheduler,
            }
        ),
        ema=ema,
    )
    return engine, loader, progress, generator, model, ema, state


def _checkpointed_step(engine, loader, progress, generator):
    batch = next(loader)
    fake_batch = next(loader)
    result = engine.train_step(
        batch,
        fake_score_batches=(fake_batch,),
        generator=generator,
    )
    progress.record_step(
        microbatches=2,
        samples=batch.batch_size + fake_batch.batch_size,
        latent_tokens=batch.clean_latents.numel() + fake_batch.clean_latents.numel(),
    )
    return result


def test_bidirectional_dcp_resume_reproduces_exact_next_update(tmp_path: Path) -> None:
    baseline = _checkpointable_bidirectional_stack(47)
    engine, loader, progress, generator, model, ema, state = baseline
    _checkpointed_step(engine, loader, progress, generator)
    assert engine.decisions.draw_count == 5
    manager = TrainingCheckpointer(tmp_path / "anyflow-bidirectional-checkpoints")
    artifact = manager.save(state)

    expected = _checkpointed_step(engine, loader, progress, generator)
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_ema = {
        name: value.detach().clone() for name, value in ema.state_dict().items()
    }
    expected_generator = generator.get_state().clone()
    expected_decisions = engine.decisions.generator.get_state().clone()
    expected_student_scheduler = engine.student_scheduler.state_dict()
    expected_fake_scheduler = engine.fake_score_scheduler.state_dict()

    restored = _checkpointable_bidirectional_stack(999)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_ema,
        restored_state,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = _checkpointed_step(
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
    )

    torch.testing.assert_close(actual.generator_loss, expected.generator_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.fake_score_loss, expected.fake_score_loss, rtol=0, atol=0)
    assert restored_engine.decisions.draw_count == 10
    assert restored_loader.cursor == 4
    assert restored_progress.optimizer_steps == 2
    assert torch.equal(restored_generator.get_state(), expected_generator)
    assert torch.equal(
        restored_engine.decisions.generator.get_state(),
        expected_decisions,
    )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
    for name, value in restored_ema.state_dict().items():
        torch.testing.assert_close(value, expected_ema[name], rtol=0, atol=0)
    assert restored_engine.student_scheduler.state_dict() == expected_student_scheduler
    assert restored_engine.fake_score_scheduler.state_dict() == expected_fake_scheduler
