from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.ddrl import (  # noqa: E402
    DDRLRolloutBatch,
    DDRLTrajectory,
    NativeDDRLEngine,
    NativeDDRLTrainingSession,
)


class _ReplayAdapter:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(0.2)

    def replay_mean(self, trajectory, train_on_position, *, training):
        self.module.train(training)
        return trajectory.replay_inputs["noisy"][:, train_on_position] * self.module.weight.reshape(())


class _RolloutAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def collect(self, batch, *, generator=None):
        del generator
        self.calls.append(batch.batch_id)
        noisy = batch.model_inputs["noisy"]
        return DDRLTrajectory(
            trajectory_id=f"trajectory-{batch.batch_id}",
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            train_on=(1, 3),
            next_latents=noisy * 0.4,
            old_means=noisy * 0.1,
            terminal_latents=batch.model_inputs["terminal"],
            replay_inputs={"noisy": noisy},
        )


class _RewardAdapter:
    reward_ids = ("quality",)

    def score(self, trajectory):
        return {"quality": trajectory.terminal_latents.flatten(1).mean(dim=1)}


class _BoundaryCheckpointer:
    def __init__(self, engine: NativeDDRLEngine) -> None:
        self.engine = engine
        self.saved_steps: list[int] = []

    def save(self, state, *, asynchronous):
        del state, asynchronous
        self.engine.state_dict()
        self.saved_steps.append(self.engine.global_step)
        return object()


def test_session_collects_rewards_updates_progress_and_checkpoints_at_boundary() -> None:
    replay = _ReplayAdapter()
    engine = NativeDDRLEngine(
        replay,
        torch.optim.SGD(replay.module.parameters(), lr=0.05),
        clip_range=0.2,
    )
    rollout = _RolloutAdapter()
    progress = TrainingProgress()
    events: list[dict[str, object]] = []
    checkpointer = _BoundaryCheckpointer(engine)
    session = NativeDDRLTrainingSession(
        rollout_adapter=rollout,
        reward_adapter=_RewardAdapter(),
        scalarizer=WeightedRewardScalarizer({"quality": 1.0}),
        engine=engine,
        progress=progress,
        checkpoint_state=object(),
        checkpointer=checkpointer,  # type: ignore[arg-type]
        save_every_steps=1,
        event_sink=events.append,
    )
    noisy = torch.linspace(-0.5, 0.5, 48).reshape(4, 2, 1, 2, 3)
    batch = DDRLRolloutBatch(
        batch_id="batch-1",
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        model_inputs={
            "noisy": noisy,
            "terminal": torch.tensor([[0.0], [2.0], [1.0], [5.0]]),
        },
    )

    result = session.train_iteration(batch)

    assert result.trajectory.train_on == (1, 3)
    assert rollout.calls == ["batch-1"]
    assert progress.optimizer_steps == 1
    assert progress.microbatches_seen == 2
    assert progress.samples_seen == 4
    assert progress.latent_tokens_seen == 48
    assert checkpointer.saved_steps == [1]
    assert events == [
        {
            "schema": "worldfoundry-ddrl-step-event",
            "global_step": 1,
            "trajectory_id": "trajectory-batch-1",
            "train_on": [1, 3],
            "loss": float(result.update.loss.item()),
            "policy_loss": float(result.update.policy_loss.item()),
            "ratio_mean": float(result.update.ratios.mean().item()),
        }
    ]
