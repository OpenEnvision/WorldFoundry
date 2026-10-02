from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_dpo import (  # noqa: E402
    DIFFUSION_DPO_ENGINE_STATE_SCHEMA,
    DiffusionDPOBatch,
    NativeDiffusionDPOEngine,
)


class _ToyFlowPolicy:
    def __init__(self, gain: float) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)

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
        del sigmas, sample_ids, conditioning, branch
        self.module.train(training)
        return noisy_latents * self.module.weight.reshape(())

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
        expanded = sigmas.reshape((int(noisy_latents.shape[0]),) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - expanded * velocity


def _batch(batch_id: str, *, shift: float = 0.0) -> DiffusionDPOBatch:
    return DiffusionDPOBatch(
        batch_id=batch_id,
        sample_ids=(f"{batch_id}-cw", f"{batch_id}-cl", f"{batch_id}-dw", f"{batch_id}-dl"),
        pair_ids=(f"{batch_id}-c", f"{batch_id}-c", f"{batch_id}-d", f"{batch_id}-d"),
        clean_latents=torch.tensor([[0.2 + shift], [0.9 + shift], [-0.4 + shift], [1.2 + shift]]),
        conditioning={"context": torch.ones(4, 1)},
    )


class _RecordingParallelContext:
    world_size = 5

    def __init__(self) -> None:
        self.weights: list[float] = []

    def audit_synchronized_module(self, module, *, role) -> None:
        del module, role

    def scale_local_mean(self, local_mean, local_weight):
        self.weights.append(float(local_weight))
        return local_mean


def _engine(*, parallel_context=None):
    policy = _ToyFlowPolicy(0.25)
    reference = _ToyFlowPolicy(-0.1)
    optimizer = torch.optim.SGD(policy.module.parameters(), lr=0.05, momentum=0.2)
    engine = NativeDiffusionDPOEngine(
        policy,
        reference,
        optimizer,
        beta=0.5,
        parallel_context=parallel_context,
    )
    return engine, policy, reference, optimizer


def test_engine_freezes_reference_and_scales_dp_mean_by_pair_count() -> None:
    parallel = _RecordingParallelContext()
    engine, _, reference, _ = _engine(parallel_context=parallel)
    initial_reference = reference.module.weight.detach().clone()

    result = engine.train_step(
        _batch("batch-1"),
        generator=torch.Generator().manual_seed(31),
    )

    assert parallel.weights == [2.0]
    assert torch.isfinite(result.loss)
    assert result.times[0] == result.times[1]
    assert result.times[2] == result.times[3]
    assert reference.module.weight.grad is None
    assert not reference.module.training
    torch.testing.assert_close(reference.module.weight, initial_reference, rtol=0, atol=0)
    assert engine.state_dict() == {
        "schema": DIFFUSION_DPO_ENGINE_STATE_SCHEMA,
        "global_step": 1,
        "optimizer_steps": 1,
        "last_batch_id": "batch-1",
        "beta": 0.5,
        "max_grad_norm": 1.0,
        "data_parallel_size": 5,
    }


def test_engine_state_restores_the_exact_next_update() -> None:
    engine, policy, reference, optimizer = _engine()
    engine.train_step(_batch("batch-1"), generator=torch.Generator().manual_seed(37))
    engine_state = copy.deepcopy(engine.state_dict())
    policy_state = copy.deepcopy(policy.module.state_dict())
    reference_state = copy.deepcopy(reference.module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    expected = engine.train_step(
        _batch("batch-2", shift=0.1),
        generator=torch.Generator().manual_seed(41),
    )

    restored, restored_policy, restored_reference, restored_optimizer = _engine()
    restored_policy.module.load_state_dict(policy_state)
    restored_reference.module.load_state_dict(reference_state)
    restored_optimizer.load_state_dict(optimizer_state)
    restored.load_state_dict(engine_state)
    actual = restored.train_step(
        _batch("batch-2", shift=0.1),
        generator=torch.Generator().manual_seed(41),
    )

    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(restored_policy.module.weight, policy.module.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored_reference.module.weight, reference.module.weight, rtol=0, atol=0)
    assert restored.global_step == engine.global_step == 2


def test_engine_rejects_policy_reference_parameter_aliasing() -> None:
    policy = _ToyFlowPolicy(0.2)
    reference = _ToyFlowPolicy(0.2)
    reference.module = policy.module

    with pytest.raises(ValueError, match="distinct modules"):
        NativeDiffusionDPOEngine(
            policy,
            reference,
            torch.optim.SGD(policy.module.parameters(), lr=0.1),
            beta=0.1,
        )
