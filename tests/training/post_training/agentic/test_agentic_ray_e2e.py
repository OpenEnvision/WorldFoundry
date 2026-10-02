from __future__ import annotations

import pytest
import torch

from tests.training.fixtures.ray_agentic_e2e import (
    RayToyPolicyModule,
    RayToyReplayAdapter,
    ray_tool_executor_factory,
    ray_toy_policy_factory,
)
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.distributed import (
    RayDevicePoolConfig,
    RayPostTrainingRuntime,
    RayPostTrainingRuntimeConfig,
    RolloutPlacement,
    TrainerBinding,
    WeightKind,
)
from worldfoundry.training.post_training.agentic import (
    AgenticRewardComponent,
    AgenticRolloutRequest,
    AgenticSampleRequest,
    AgenticTrajectoryRewardAdapter,
    AgentMessage,
    NativeAgenticTrainingSession,
    setup_ray_agentic_rollout,
)
from worldfoundry.training.post_training.rewards.scalarization import (
    WeightedRewardScalarizer,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy import (
    NativeTokenPolicyEngine,
    NativeTokenPolicyTrainingSession,
    TokenGRPOStage,
)


def _request(policy_revision: str) -> AgenticRolloutRequest:
    return AgenticRolloutRequest(
        samples=tuple(
            AgenticSampleRequest(
                sample_id=f"sample-{suffix}",
                group_id="prompt",
                messages=(AgentMessage(role="user", content="add 2 and 3"),),
                conditioning={"answer_token": token},
            )
            for suffix, token in zip(("a", "b", "c"), (2, 3, 4), strict=True)
        ),
        policy_revision=policy_revision,
        sampling_temperature=0.7,
        max_turns=2,
    )


def _reward() -> AgenticTrajectoryRewardAdapter:
    return AgenticTrajectoryRewardAdapter(
        (
            AgenticRewardComponent(
                "correctness",
                lambda sample: 1.0 if sample.request.sample_id == "sample-a" else -1.0,
            ),
        )
    )


def _assert_remote_trajectory(result, expected_scale: float, rollout_index: int) -> None:
    trajectory = result.trajectory
    assert tuple(sample.request.sample_id for sample in trajectory.samples) == (
        "sample-a",
        "sample-c",
    )
    assert trajectory.failed_sample_ids == ("sample-b",)
    assert trajectory.rollout_index == rollout_index
    assert [message.role for message in trajectory.samples[0].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert trajectory.samples[0].turns[0].tool_results[0].content == "5"
    torch.testing.assert_close(
        trajectory.samples[0].old_log_probs,
        -torch.tensor([2.0, 3.0]) * expected_scale,
    )


@pytest.mark.parametrize("weight_kind", [WeightKind.FULL, WeightKind.LORA])
def test_real_cpu_ray_agentic_rollout_sync_and_training(weight_kind: WeightKind) -> None:
    pytest.importorskip("ray")
    source = RayToyPolicyModule()
    with torch.no_grad():
        source.lora_A.fill_(0.325 if weight_kind is WeightKind.LORA else 0.075)
        if weight_kind is WeightKind.FULL:
            source.base.fill_(0.375)
    runtime = RayPostTrainingRuntime(
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=2,
                devices_per_node=2,
                workers_per_device=1,
                cpus_per_worker=0.25,
                accelerator_resource="CPU",
            ),
            rollout_devices=2,
            rollout_placement=RolloutPlacement.SEPARATE,
            trainer_binding=TrainerBinding.EXTERNAL,
            weight_bucket_bytes=8,
        )
    )
    try:
        rollout = setup_ray_agentic_rollout(
            runtime,
            source,
            rollout_policy_factory=ray_toy_policy_factory,
            tool_executor_factory=ray_tool_executor_factory,
            weight_kind=weight_kind,
        )
        replay = RayToyReplayAdapter(source)
        engine = NativeTokenPolicyEngine(
            replay,
            torch.optim.SGD(source.parameters(), lr=0.01),
            algorithm=TokenGRPOStage(clip_range=0.2),
            initial_policy_revision="policy-root",
            old_log_prob_source="rollout",
            replay_microbatch_size=1,
        )
        progress = TrainingProgress()
        token_session = NativeTokenPolicyTrainingSession(
            rollout_adapter=rollout,
            reward_adapter=_reward(),
            scalarizer=WeightedRewardScalarizer({"correctness": 1.0}),
            engine=engine,
            progress=progress,
            group_size=3,
            sampling_temperature=0.7,
        )
        session = NativeAgenticTrainingSession(rollout, token_session)

        initial_lora = source.lora_A.detach().clone()
        first_scale = float((source.base + source.lora_A).detach())
        first = session.train_iteration(_request(engine.current_policy_revision))
        _assert_remote_trajectory(first, first_scale, rollout_index=0)
        assert first.trajectory.policy_revision == "policy-root"
        assert first.token_policy.updates[0].optimizer_committed
        torch.testing.assert_close(
            first.token_policy.updates[0].metrics["ratio_mean"],
            torch.tensor(1.0),
        )
        assert rollout.last_sync_report is not None
        assert rollout.last_sync_report.revision == 0
        assert rollout.last_sync_report.kind is weight_kind
        assert rollout.last_sync_report.receiver_count == 2
        after_first_update = source.lora_A.detach().clone()
        assert not torch.equal(after_first_update, initial_lora)

        second_scale = float((source.base + source.lora_A).detach())
        second = session.train_iteration(_request(engine.current_policy_revision))
        _assert_remote_trajectory(second, second_scale, rollout_index=1)
        assert second.trajectory.policy_revision == "policy-root:step-1"
        torch.testing.assert_close(
            second.token_policy.updates[0].metrics["ratio_mean"],
            torch.tensor(1.0),
        )
        assert rollout.last_sync_report is not None
        assert rollout.last_sync_report.revision == 1
        assert rollout.last_sync_report.kind is weight_kind
        assert replay.sample_batches == [
            ("sample-a",),
            ("sample-c",),
            ("sample-a",),
            ("sample-c",),
        ]
        assert progress.optimizer_steps == 2
        assert progress.samples_seen == 4
        assert progress.latent_tokens_seen == 8
        assert not torch.equal(source.lora_A.detach(), after_first_update)
    finally:
        runtime.shutdown()
