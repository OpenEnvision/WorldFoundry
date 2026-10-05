from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rl.algorithms.diffusion_dpo import (  # noqa: E402
    DiffusionDPOBatch,
    NativeDiffusionDPOEngine,
    NativeDiffusionDPOTrainingSession,
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


def _engine() -> NativeDiffusionDPOEngine:
    policy = _ToyFlowPolicy(0.2)
    reference = _ToyFlowPolicy(0.1)
    return NativeDiffusionDPOEngine(
        policy,
        reference,
        torch.optim.SGD(policy.module.parameters(), lr=0.05),
        beta=0.5,
    )


def _batch(batch_id: str, offset: float) -> DiffusionDPOBatch:
    values = torch.arange(24, dtype=torch.float32).reshape(4, 1, 2, 3) / 24 + offset
    return DiffusionDPOBatch(
        batch_id=batch_id,
        sample_ids=(f"{batch_id}-aw", f"{batch_id}-al", f"{batch_id}-bw", f"{batch_id}-bl"),
        pair_ids=(f"{batch_id}-a", f"{batch_id}-a", f"{batch_id}-b", f"{batch_id}-b"),
        clean_latents=values,
        conditioning={"context": torch.ones(4, 1)},
    )


class _BoundaryCheckpointer:
    def __init__(self, engine: NativeDiffusionDPOEngine) -> None:
        self.engine = engine
        self.saved_steps: list[int] = []

    def save(self, state, *, asynchronous: bool):
        del state, asynchronous
        self.engine.state_dict()
        self.saved_steps.append(self.engine.global_step)
        return object()


def test_session_commits_progress_events_and_safe_step_checkpoints() -> None:
    engine = _engine()
    progress = TrainingProgress()
    events: list[dict[str, object]] = []
    checkpointer = _BoundaryCheckpointer(engine)
    session = NativeDiffusionDPOTrainingSession(
        engine,
        [_batch("batch-1", 0.0), _batch("batch-2", 0.1)],
        progress,
        checkpoint_state=object(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=1,
        event_sink=events.append,
    )

    summary = session.run(max_steps=2, generator=torch.Generator().manual_seed(53))

    assert summary.initial_step == 0
    assert summary.final_step == 2
    assert summary.iterations == 2
    assert progress.optimizer_steps == 2
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 8
    assert progress.latent_tokens_seen == 48
    assert checkpointer.saved_steps == [1, 2]
    assert [event["batch_id"] for event in events] == ["batch-1", "batch-2"]
    assert all(event["schema"] == "worldfoundry-diffusion-dpo-step-event" for event in events)


def test_session_reports_loader_exhaustion_without_recycling() -> None:
    session = NativeDiffusionDPOTrainingSession(
        _engine(),
        [_batch("batch-1", 0.0)],
        TrainingProgress(),
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        session.run(max_steps=2, generator=torch.Generator().manual_seed(59))
