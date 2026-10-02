from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft import (  # noqa: E402
    DiffusionNFTRollout,
    NativeDiffusionNFTEngine,
    NativeDiffusionNFTTrainingSession,
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


def _engine() -> NativeDiffusionNFTEngine:
    policy = _ToyFlowPolicy(0.2)
    old_policy = _ToyFlowPolicy(-1.0)
    return NativeDiffusionNFTEngine(
        policy,
        old_policy,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        initial_old_policy_revision="old-policy-initial",
        beta=0.1,
        advantage_clip_max=1.0,
    )


def _rollout(
    collection_id: str,
    offset: float,
    *,
    policy_revision: str = "old-policy-initial",
) -> DiffusionNFTRollout:
    values = torch.arange(24, dtype=torch.float32).reshape(4, 1, 2, 3) / 24 + offset
    return DiffusionNFTRollout(
        collection_id=collection_id,
        policy_revision=policy_revision,
        sample_ids=(f"{collection_id}-a", f"{collection_id}-b", f"{collection_id}-c", f"{collection_id}-d"),
        group_ids=("first", "first", "second", "second"),
        clean_latents=values,
        rewards=torch.tensor([0.0, 2.0, 1.0, 4.0]),
        conditioning={"context": torch.ones(4, 1)},
    )


def test_toy_session_consumes_terminal_rollouts_once_and_commits_progress() -> None:
    engine = _engine()
    progress = TrainingProgress()
    events: list[dict[str, object]] = []

    def rollouts():
        yield _rollout("collection-1", 0.0)
        yield _rollout(
            "collection-2",
            0.1,
            policy_revision=engine.current_collection_policy_revision,
        )

    session = NativeDiffusionNFTTrainingSession(
        engine,
        rollouts(),
        progress,
        event_sink=events.append,
    )

    summary = session.run(max_steps=2, generator=torch.Generator().manual_seed(71))

    assert summary.initial_step == 0
    assert summary.final_step == 2
    assert summary.iterations == 2
    assert summary.old_policy_refreshes == 2
    assert progress.optimizer_steps == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 8
    assert progress.latent_tokens_seen == 48
    assert [event["collection_id"] for event in events] == ["collection-1", "collection-2"]
    assert all(event["old_policy_refreshed"] for event in events)


def test_session_refuses_to_cycle_and_reuse_collected_rollouts() -> None:
    session = NativeDiffusionNFTTrainingSession(
        _engine(),
        [_rollout("only-collection", 0.0)],
        TrainingProgress(),
    )

    with pytest.raises(RuntimeError, match="cannot be replayed"):
        session.run(max_steps=2, generator=torch.Generator().manual_seed(73))
