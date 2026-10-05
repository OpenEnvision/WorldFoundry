from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import TrainingProgress  # noqa: E402
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.bagel_flow_unigrpo import (  # noqa: E402
    BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA,
    BagelFlowUniGRPOIterationResult,
    NativeBagelFlowUniGRPOEngine,
    NativeBagelFlowUniGRPOTrainingSession,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch  # noqa: E402
from worldfoundry.training.post_training.rl.trajectory import (  # noqa: E402
    FlowTrajectorySampler,
    NativeFlowTrajectoryReplay,
)


class _ToyPolicy:
    def __init__(self, gain: float, *, trainable: bool = True) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)
        self.module.requires_grad_(trainable)

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
        sigma = sigmas.reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - sigma * velocity


class _TerminalReward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float().flatten(1)
        return {"reward": terminal.mean(dim=1)}


def test_native_bagel_flow_unigrpo_runs_reference_velocity_regularized_updates() -> None:
    policy = _ToyPolicy(0.25)
    reference = _ToyPolicy(0.1, trainable=False)
    engine = NativeBagelFlowUniGRPOEngine(
        NativeFlowTrajectoryReplay(policy),
        torch.optim.SGD(policy.module.parameters(), lr=0.02),
        initial_policy_revision="policy-root",
        reference_replay_adapter=NativeFlowTrajectoryReplay(reference),
        clip_range=0.2,
        velocity_mse_weight=0.5,
        ratio_norm=True,
        grad_reweight=True,
        updates_per_trajectory=2,
        replay_microbatch_size=1,
    )
    events: list[object] = []
    session = NativeBagelFlowUniGRPOTrainingSession(
        sampler=FlowTrajectorySampler(policy, eta=0.6),
        reward_adapter=_TerminalReward(),
        scalarizer=WeightedRewardScalarizer({"reward": 1.0}),
        engine=engine,
        progress=TrainingProgress(),
        sde_step_indices=(0, 2),
        old_log_prob_source="replay",
        event_sink=events.append,
    )
    batch = FlowRolloutBatch(
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        policy_revision="policy-root",
        initial_latents=torch.randn(
            4,
            3,
            generator=torch.Generator().manual_seed(59),
        ),
        sigmas=torch.tensor([1.0, 0.7, 0.2, 0.0]),
    )

    result = session.train_iteration(
        batch,
        generator=torch.Generator().manual_seed(61),
    )

    assert isinstance(result, BagelFlowUniGRPOIterationResult)
    assert len(result.updates) == 2
    assert float(result.updates[0].metrics["velocity_mse"]) > 0
    torch.testing.assert_close(
        result.updates[0].metrics["ratio_mean_bias"],
        torch.tensor(0.0),
        rtol=0,
        atol=0,
    )
    assert float(result.updates[1].metrics["ratio_mean_bias"]) > 0
    assert all(parameter.grad is None for parameter in reference.module.parameters())
    assert [event["schema"] for event in events] == [
        "worldfoundry-bagel-flow-unigrpo-step-event",
        "worldfoundry-bagel-flow-unigrpo-step-event",
    ]
    state = engine.state_dict()
    assert state["schema"] == BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA
    assert state["velocity_mse_weight"] == 0.5
    assert state["ratio_norm"] is True
    assert state["grad_reweight"] is True
