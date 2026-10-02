from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_nft import (  # noqa: E402
    DiffusionNFTRollout,
    NativeDiffusionNFTEngine,
    OldPolicyRefresh,
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


def _rollout(
    collection_id: str,
    *,
    shift: float = 0.0,
    policy_revision: str = "old-policy-initial",
) -> DiffusionNFTRollout:
    return DiffusionNFTRollout(
        collection_id=collection_id,
        policy_revision=policy_revision,
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        clean_latents=torch.tensor([[0.2 + shift], [0.9 + shift], [-0.4 + shift], [1.2 + shift]]),
        rewards=torch.tensor([0.0, 2.0, 1.0, 5.0]),
        conditioning={"context": torch.ones(4, 1)},
    )


def _engine(
    *,
    schedule: OldPolicyRefresh | None = None,
    parallel_context=None,
):
    policy = _ToyFlowPolicy(0.25)
    old_policy = _ToyFlowPolicy(-3.0)
    optimizer = torch.optim.SGD(policy.module.parameters(), lr=0.05, momentum=0.2)
    engine = NativeDiffusionNFTEngine(
        policy,
        old_policy,
        optimizer,
        initial_old_policy_revision="old-policy-initial",
        beta=0.1,
        advantage_clip_max=1.0,
        old_policy_refresh=schedule,
        parallel_context=parallel_context,
    )
    return engine, policy, old_policy, optimizer


def test_old_policy_schedules_match_pinned_trainer_retention() -> None:
    assert OldPolicyRefresh("copy").retention(999) == 0.0
    assert OldPolicyRefresh("linear_to_0_5").retention(1) == 0.001
    assert OldPolicyRefresh("linear_to_0_5").retention(500) == 0.5
    assert OldPolicyRefresh("linear_to_0_5").retention(900) == 0.5
    delayed = OldPolicyRefresh("delayed_linear_to_0_999")
    assert delayed.retention(74) == 0.0
    assert delayed.retention(75) == 0.0
    assert delayed.retention(76) == pytest.approx(0.0075)
    assert delayed.retention(1000) == 0.999


def test_engine_keeps_old_policy_frozen_and_refreshes_only_on_cadence() -> None:
    refresh = OldPolicyRefresh("linear_to_0_5", update_interval=2)
    engine, policy, old_policy, _ = _engine(schedule=refresh)
    initial_anchor = old_policy.module.weight.detach().clone()
    torch.testing.assert_close(initial_anchor, policy.module.weight.detach(), rtol=0, atol=0)
    assert not old_policy.module.training
    assert all(not parameter.requires_grad for parameter in old_policy.module.parameters())

    first = engine.train_step(_rollout("rollout-1"), generator=torch.Generator().manual_seed(41))
    anchor_after_first = old_policy.module.weight.detach().clone()
    torch.testing.assert_close(anchor_after_first, initial_anchor, rtol=0, atol=0)
    assert first.old_policy_refreshed is False
    assert not torch.equal(policy.module.weight.detach(), initial_anchor)

    second = engine.train_step(
        _rollout("rollout-2", shift=0.1),
        generator=torch.Generator().manual_seed(43),
    )
    expected = 0.002 * anchor_after_first + 0.998 * policy.module.weight.detach()
    torch.testing.assert_close(old_policy.module.weight.detach(), expected, rtol=1e-6, atol=1e-7)
    assert second.old_policy_refreshed is True
    assert second.old_policy_retention == pytest.approx(0.002)
    assert old_policy.module.weight.grad is None
    assert not old_policy.module.training


class _RecordingParallelContext:
    world_size = 7

    def __init__(self) -> None:
        self.weights: list[float] = []

    def audit_synchronized_module(self, module, *, role) -> None:
        del module, role

    def audit_local_group_ownership(self, group_ids) -> None:
        assert group_ids == ("first", "first", "second", "second")

    def scale_local_mean(self, local_mean, local_weight):
        self.weights.append(float(local_weight))
        return local_mean


def test_engine_routes_local_batch_weight_through_arbitrary_dp_mean_scaling() -> None:
    parallel = _RecordingParallelContext()
    engine, _, _, _ = _engine(parallel_context=parallel)

    result = engine.train_step(
        _rollout("distributed-rollout"),
        generator=torch.Generator().manual_seed(47),
    )

    assert parallel.weights == [4.0]
    assert torch.isfinite(result.loss)
    assert engine.state_dict()["data_parallel_size"] == 7


def test_engine_state_and_old_anchor_resume_exact_next_update() -> None:
    refresh = OldPolicyRefresh("delayed_linear_to_0_999", update_interval=2)
    engine, policy, old_policy, optimizer = _engine(schedule=refresh)
    engine.train_step(_rollout("rollout-1"), generator=torch.Generator().manual_seed(51))
    engine.train_step(
        _rollout("rollout-2", shift=0.1),
        generator=torch.Generator().manual_seed(53),
    )
    engine_state = copy.deepcopy(engine.state_dict())
    assert engine_state["collection_policy_revision"] == (engine.current_collection_policy_revision)
    policy_state = copy.deepcopy(policy.module.state_dict())
    old_policy_state = copy.deepcopy(old_policy.module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    expected = engine.train_step(
        _rollout(
            "rollout-3",
            shift=-0.2,
            policy_revision=engine.current_collection_policy_revision,
        ),
        generator=torch.Generator().manual_seed(59),
    )

    restored, restored_policy, restored_old_policy, restored_optimizer = _engine(schedule=refresh)
    restored_policy.module.load_state_dict(policy_state)
    restored_old_policy.module.load_state_dict(old_policy_state)
    restored_optimizer.load_state_dict(optimizer_state)
    restored.load_state_dict(engine_state)
    actual = restored.train_step(
        _rollout(
            "rollout-3",
            shift=-0.2,
            policy_revision=restored.current_collection_policy_revision,
        ),
        generator=torch.Generator().manual_seed(59),
    )

    assert restored.global_step == engine.global_step == 3
    assert restored.old_policy_refreshes == engine.old_policy_refreshes == 1
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_policy.module.weight,
        policy.module.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored_old_policy.module.weight,
        old_policy.module.weight,
        rtol=0,
        atol=0,
    )

    invalid_state = copy.deepcopy(engine_state)
    invalid_state["collection_policy_revision"] = "stale-revision"
    with pytest.raises(ValueError, match="collection-policy revision"):
        restored.load_state_dict(invalid_state)


def test_engine_rejects_multi_update_and_duplicate_collection() -> None:
    policy = _ToyFlowPolicy(0.2)
    old_policy = _ToyFlowPolicy(0.2)
    with pytest.raises(ValueError, match="exactly one optimizer update"):
        NativeDiffusionNFTEngine(
            policy,
            old_policy,
            torch.optim.SGD(policy.module.parameters(), lr=0.1),
            initial_old_policy_revision="old-policy-initial",
            beta=0.1,
            advantage_clip_max=1.0,
            updates_per_rollout=2,
        )

    engine, _, _, _ = _engine()
    batch = _rollout("single-use")
    engine.train_step(batch, generator=torch.Generator().manual_seed(61))
    with pytest.raises(ValueError, match="only once"):
        engine.train_step(batch, generator=torch.Generator().manual_seed(63))


def test_engine_rejects_stale_collection_revision_before_forward() -> None:
    engine, _, _, _ = _engine()
    engine.train_step(_rollout("fresh"), generator=torch.Generator().manual_seed(67))

    with pytest.raises(ValueError, match="stale behavior policy"):
        engine.train_step(
            _rollout("stale", policy_revision="old-policy-initial"),
            generator=torch.Generator().manual_seed(69),
        )


class _FailAfterMutationSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("failed after parameter mutation")


def test_engine_poisoned_when_optimizer_mutates_then_raises() -> None:
    policy = _ToyFlowPolicy(0.25)
    old_policy = _ToyFlowPolicy(0.25)
    optimizer = _FailAfterMutationSGD(policy.module.parameters(), lr=0.05)
    engine = NativeDiffusionNFTEngine(
        policy,
        old_policy,
        optimizer,
        initial_old_policy_revision="old-policy-initial",
        beta=0.1,
        advantage_clip_max=1.0,
    )
    before = policy.module.weight.detach().clone()
    engine_state = copy.deepcopy(engine.state_dict())
    policy_state = copy.deepcopy(policy.module.state_dict())
    old_policy_state = copy.deepcopy(old_policy.module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(RuntimeError, match="after parameter mutation"):
        engine.train_step(
            _rollout("partial-commit"),
            generator=torch.Generator().manual_seed(71),
        )

    assert not torch.equal(policy.module.weight.detach(), before)
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.train_step(
            _rollout("must-restore"),
            generator=torch.Generator().manual_seed(73),
        )

    policy.module.load_state_dict(policy_state)
    old_policy.module.load_state_dict(old_policy_state)
    optimizer.load_state_dict(optimizer_state)
    engine.load_state_dict(engine_state)
    assert engine.state_dict() == engine_state
