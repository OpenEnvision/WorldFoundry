from __future__ import annotations

import pytest
import torch

from worldfoundry.training.checkpoint import TrainingProgress
from worldfoundry.training.distributed.flow_rollout import (
    RayFlowTrajectorySampler,
    attach_ray_flow_policy_rollout,
    partition_complete_flow_groups,
)
from worldfoundry.training.distributed.ray_runtime import (
    RayDevicePoolConfig,
    RolloutPlacement,
)
from worldfoundry.training.distributed.rollout_runtime import (
    RayPostTrainingRuntime,
    RayPostTrainingRuntimeConfig,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch
from worldfoundry.training.recipes import PostTrainingRecipe


class _ToyFlowModule(torch.nn.Module):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.eye(2) * gain)

    def forward(self, value):
        return torch.nn.functional.linear(value, self.lora_weight)


class _RayToyFlowPolicy:
    def __init__(self, gain: float) -> None:
        self.module = _ToyFlowModule(gain)

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
        return self.module(noisy_latents)

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
        sigma = torch.as_tensor(
            sigmas,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        ).reshape(noisy_latents.shape[0], 1)
        return noisy_latents - sigma * velocity


class _TerminalRewards:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float()
        return {
            "video_quality": terminal.mean(dim=1),
            "motion_quality": terminal.square().mean(dim=1),
            "text_alignment": -terminal.abs().mean(dim=1),
        }


def _actor_policy_factory():
    class ActorModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.zeros(2, 2))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.lora_weight)

    class ActorPolicy:
        def __init__(self) -> None:
            self.module = ActorModule()

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
            return self.module(noisy_latents)

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
            sigma = torch.as_tensor(
                sigmas,
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            ).reshape(noisy_latents.shape[0], 1)
            return noisy_latents - sigma * velocity

    def factory(*, context):
        del context
        return ActorPolicy()

    return factory


def _flow_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-post-training",
            "execution_owner": "worldfoundry-native",
            "run": {"id": "ray-flow-test", "output_dir": "runs/ray-flow-test"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "policy-root",
            },
            "tuning": {"mode": "full"},
            "export": {"format": "safetensors"},
            "data": {"manifest": "data/train.jsonl", "shuffle": False},
            "algorithm": {
                "type": "flow-grpo",
                "sigmas": [1.0, 0.6, 0.0],
                "sde_step_indices": [0, 1],
                "eta": 0.7,
                "sigma_max": 0.99,
                "updates_per_trajectory": 1,
                "group_size": 2,
                "trajectory_dtype": "float32",
                "reward_weights": {
                    "video_quality": 1.0,
                    "motion_quality": 0.25,
                    "text_alignment": 0.5,
                },
                "reward_model": {"type": "videoalign"},
                "clip_range": 0.0002,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
        }
    )


def _stack(policy: _RayToyFlowPolicy):
    return build_native_flow_policy_training_stack(
        _flow_recipe(),
        policy=policy,
        initial_policy_revision="policy-root",
        fused_adamw=False,
    )


def _runtime(
    *,
    rollout_devices: int,
    placement: RolloutPlacement,
) -> RayPostTrainingRuntime:
    return RayPostTrainingRuntime(
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=rollout_devices,
                devices_per_node=1,
                workers_per_device=1,
                cpus_per_worker=0.25,
                accelerator_resource="CPU",
            ),
            rollout_devices=rollout_devices,
            rollout_placement=placement,
        )
    )


def test_complete_group_partition_never_splits_interleaved_prompt_groups() -> None:
    group_ids = ("a", "b", "c", "a", "b", "c")
    shards = partition_complete_flow_groups(group_ids, 2)

    assert shards == ((0, 3, 1, 4), (2, 5))
    for group_id in set(group_ids):
        owning_shards = [shard for shard in shards if any(group_ids[position] == group_id for position in shard)]
        assert len(owning_shards) == 1


def test_external_trainer_with_separate_ray_rollout_returns_exact_replay_and_backward() -> None:
    pytest.importorskip("ray")
    policy = _RayToyFlowPolicy(0.2)
    stack = _stack(policy)
    runtime = _runtime(
        rollout_devices=1,
        placement=RolloutPlacement.SEPARATE,
    )
    try:
        stack = attach_ray_flow_policy_rollout(
            stack,
            runtime,
            rollout_policy_factory=_actor_policy_factory(),
        )
        assert isinstance(stack.sampler, RayFlowTrajectorySampler)
        initial = torch.randn(4, 2, generator=torch.Generator().manual_seed(11))
        trajectory = stack.sampler.sample(
            initial,
            torch.tensor([1.0, 0.6, 0.0]),
            sample_ids=("a-0", "a-1", "b-0", "b-1"),
            group_ids=("a", "a", "b", "b"),
            conditioning={"marker": torch.arange(4).reshape(4, 1)},
            policy_revision="policy-root",
            sde_step_indices=(0, 1),
            generator=torch.Generator().manual_seed(17),
        )

        replay = stack.replay.replay(trajectory, training=True)
        torch.testing.assert_close(
            replay.log_probs.detach(),
            trajectory.old_log_probs,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        (-replay.log_probs.mean()).backward()
        gradient = policy.module.lora_weight.grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        assert stack.sampler.last_sync_report is not None
        assert stack.sampler.last_sync_report.revision == 0

        stack.optimizer.zero_grad(set_to_none=True)
        before = policy.module.lora_weight.detach().clone()
        session = stack.session_type(
            sampler=stack.sampler,
            reward_adapter=_TerminalRewards(),
            scalarizer=stack.scalarizer,
            engine=stack.engine,
            progress=TrainingProgress(),
            sde_index_schedule=stack.sde_index_schedule,
        )
        iteration = session.train_iteration(
            FlowRolloutBatch(
                sample_ids=("a-0", "a-1", "b-0", "b-1"),
                group_ids=("a", "a", "b", "b"),
                policy_revision="policy-root",
                initial_latents=initial,
                sigmas=torch.tensor([1.0, 0.6, 0.0]),
            ),
            generator=torch.Generator().manual_seed(18),
        )
        assert len(iteration.updates) == 1
        assert not torch.equal(policy.module.lora_weight.detach(), before)

        stack.sampler.sample(
            initial,
            torch.tensor([1.0, 0.6, 0.0]),
            sample_ids=("a-0", "a-1", "b-0", "b-1"),
            group_ids=("a", "a", "b", "b"),
            conditioning={},
            policy_revision=stack.engine.current_policy_revision,
            sde_step_indices=(0, 1),
            generator=torch.Generator().manual_seed(19),
        )
        assert stack.sampler.last_sync_report is not None
        assert stack.sampler.last_sync_report.revision == 1
    finally:
        runtime.shutdown()


def test_separate_ray_workers_restore_interleaved_multi_group_sample_order() -> None:
    pytest.importorskip("ray")
    policy = _RayToyFlowPolicy(0.15)
    runtime = _runtime(
        rollout_devices=2,
        placement=RolloutPlacement.SEPARATE,
    )
    try:
        stack = attach_ray_flow_policy_rollout(
            _stack(policy),
            runtime,
            rollout_policy_factory=_actor_policy_factory(),
            weight_kind="lora",
        )
        initial = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10
        sample_ids = ("a-0", "b-0", "c-0", "a-1", "b-1", "c-1")
        group_ids = ("a", "b", "c", "a", "b", "c")
        marker = torch.arange(6).reshape(6, 1)

        trajectory = stack.sampler.sample(
            initial,
            torch.tensor([1.0, 0.6, 0.0]),
            sample_ids=sample_ids,
            group_ids=group_ids,
            conditioning={"marker": marker},
            policy_revision="policy-root",
            sde_step_indices=(0, 1),
            generator=torch.Generator().manual_seed(23),
        )

        assert trajectory.sample_ids == sample_ids
        assert trajectory.group_ids == group_ids
        torch.testing.assert_close(trajectory.latents[:, 0], initial)
        assert trajectory.conditioning["marker"] is marker
        replay = stack.replay.replay(trajectory, training=True)
        torch.testing.assert_close(
            replay.log_probs.detach(),
            trajectory.old_log_probs,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert isinstance(stack.sampler, RayFlowTrajectorySampler)
        assert stack.sampler.last_sync_report is not None
        assert stack.sampler.last_sync_report.kind.value == "lora"
    finally:
        runtime.shutdown()
