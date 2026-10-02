from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.training.fixtures.ray_ltx_e2e import (
    RayTinyLTXPolicy,
    ray_tiny_ltx_policy_factory,
)
from worldfoundry.training.distributed.flow_rollout import (
    RayFlowTrajectorySampler,
    attach_ray_flow_policy_rollout,
)
from worldfoundry.training.distributed.ray_runtime import (
    RayDevicePoolConfig,
    RolloutPlacement,
)
from worldfoundry.training.distributed.rollout_runtime import (
    RayPostTrainingRuntime,
    RayPostTrainingRuntimeConfig,
)
from worldfoundry.training.engine.ltx.trajectory import (
    LTX_AUDIO_TRAJECTORY,
    LTX_AUDIO_TRANSITION_MEANS,
    LTX_AUDIO_TRANSITION_SCALES,
    LTXAudioConditionedTrajectoryReplay,
    LTXAudioConditionedTrajectorySampler,
    build_ltx_ray_trajectory_sampler,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.recipes import PostTrainingRecipe


def _recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-ltx-av-ray", "output_dir": "unused"},
            "model": {"recipe": "tiny-ltx-av", "checkpoint": "tiny-policy"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "unused.jsonl", "shuffle": False},
            "algorithm": {
                "type": "flow-grpo",
                "sigmas": [1.0, 0.5, 0.0],
                "sde_step_indices": [0, 1],
                "eta": 0.7,
                "group_size": 2,
                "trajectory_dtype": "float32",
                "advantage_normalization": "group-sample-std",
                "reward_weights": {
                    "video_quality": 1.0,
                    "motion_quality": 1.0,
                    "text_alignment": 1.0,
                },
                "reward_model": {"type": "videoalign"},
            },
            "optimizer": {"type": "adamw", "learning_rate": 0.01},
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def test_ltx_joint_av_ray_sampler_preserves_audio_trajectory_and_replays() -> None:
    pytest.importorskip("ray")
    policy = RayTinyLTXPolicy(0.2)
    stack = build_native_flow_policy_training_stack(
        _recipe(),
        policy=policy,
        initial_policy_revision="tiny-policy",
        fused_adamw=False,
    )
    local_sampler = LTXAudioConditionedTrajectorySampler(
        policy,
        transition_strategy=stack.transition_strategy,
        trajectory_dtype=torch.float32,
        audio_joint_sde=True,
        init_same_noise=False,
    )
    replay = LTXAudioConditionedTrajectoryReplay(
        policy,
        audio_joint_sde=True,
    )
    stack.engine.replay_adapter = replay
    stack = replace(stack, sampler=local_sampler, replay=replay)
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
            weight_bucket_bytes=16,
        )
    )
    try:
        stack = attach_ray_flow_policy_rollout(
            stack,
            runtime,
            rollout_policy_factory=ray_tiny_ltx_policy_factory,
            rollout_sampler_factory=build_ltx_ray_trajectory_sampler,
            rollout_sampler_factory_kwargs={
                "audio_joint_sde": True,
                "init_same_noise": False,
            },
        )
        marker = torch.arange(4).reshape(4, 1)
        initial = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1, 1)
        trajectory = stack.sampler.sample(
            initial,
            torch.tensor(stack.sigmas),
            sample_ids=("a-0", "b-0", "a-1", "b-1"),
            group_ids=("a", "b", "a", "b"),
            conditioning={"marker": marker},
            policy_revision="tiny-policy",
            sde_step_indices=(0, 1),
            generator=torch.Generator().manual_seed(17),
        )

        assert isinstance(stack.sampler, RayFlowTrajectorySampler)
        assert trajectory.conditioning["marker"] is marker
        assert trajectory.conditioning[LTX_AUDIO_TRAJECTORY].shape == (4, 3, 2, 1)
        torch.testing.assert_close(
            trajectory.conditioning[LTX_AUDIO_TRAJECTORY][:, 0, :, 0],
            initial.flatten(1).mean(dim=1, keepdim=True).expand(-1, 2),
        )
        assert trajectory.conditioning[LTX_AUDIO_TRANSITION_MEANS].shape == (
            4,
            2,
            2,
            1,
        )
        assert trajectory.conditioning[LTX_AUDIO_TRANSITION_SCALES].shape == (
            4,
            2,
            1,
            1,
        )
        replayed = stack.replay.replay(trajectory, training=True)
        torch.testing.assert_close(
            replayed.log_probs.detach(),
            trajectory.old_log_probs,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        replayed.log_probs.mean().backward()
        assert policy.module.weight.grad is not None
        assert stack.sampler.last_sync_report is not None
        assert stack.sampler.last_sync_report.revision == 0
    finally:
        runtime.close()
