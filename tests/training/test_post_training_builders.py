from __future__ import annotations

from collections.abc import Mapping

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training import (  # noqa: E402
    NativeDMDTrainingStack,
    NativeFlowDPPOEngine,
    NativeFlowDPPOTrainingSession,
    NativeFlowGRPOTrainingSession,
    NativeFlowPolicyTrainingStack,
    build_native_dmd_training_stack,
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (  # noqa: E402
    ConstantDiffusionFlowTransition,
    VariancePreservingFlowTransition,
)
from worldfoundry.training.post_training.rl.rollout_strategies.window_sde_steps import (  # noqa: E402
    FlowSDEWindowSchedule,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402


class _FlowPredictor:
    def __init__(
        self,
        *,
        trainable: bool = True,
        checkpoint_identity: str = "model-checkpoint",
    ) -> None:
        self.module = torch.nn.Linear(2, 2, bias=False)
        self.module.requires_grad_(trainable)
        self.checkpoint_identity = checkpoint_identity

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
        sigma = torch.as_tensor(sigmas, device=noisy_latents.device, dtype=noisy_latents.dtype)
        sigma = sigma.reshape((noisy_latents.shape[0],) + (1,) * (noisy_latents.ndim - 1))
        return noisy_latents - sigma * velocity


class _StatefulCounter:
    def __init__(self) -> None:
        self.value = 0

    def step(self) -> None:
        self.value += 1

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.value = int(state_dict["value"])


def _dmd_recipe_mapping() -> dict[str, object]:
    return {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "dmd-test", "output_dir": "runs/dmd-test"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "model-checkpoint"},
        "tuning": {"mode": "full"},
        "export": {"format": "safetensors"},
        "data": {"manifest": "data/train.jsonl", "shuffle": False},
        "algorithm": {
            "type": "dmd",
            "student_timesteps": [1000, 750, 500],
            "student_sigmas": [1.0, 0.75, 0.5],
            "real_score_checkpoint": "teacher-checkpoint",
            "fake_score_checkpoint": "critic-checkpoint",
            "score_flow_shift": 5.0,
            "teacher_guidance_scale": 3.5,
            "generator_update_interval": 3,
            "student_scheduler_cadence": "iteration",
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 2.0e-6,
            "weight_decay": 0.01,
            "max_grad_norm": 0.7,
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 4.0e-6,
            "max_grad_norm": 0.9,
        },
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
    }


def _flow_recipe_mapping(
    *,
    algorithm_type: str = "flow-grpo",
    transition_strategy: str = "variance-preserving",
) -> dict[str, object]:
    value = _dmd_recipe_mapping()
    value["run"] = {"id": "flow-grpo-test", "output_dir": "runs/flow-grpo-test"}
    algorithm = {
        "type": algorithm_type,
        "sigmas": [1.0, 0.7, 0.3, 0.0],
        "sde_step_indices": [1, 2],
        "eta": 0.6,
        "updates_per_trajectory": 2,
        "group_size": 4,
        "old_log_prob_source": "replay",
        "reference_kl_weight": 0.0,
        "advantage_epsilon": 1.0e-7,
        "advantage_clip_max": 4.0,
        "trajectory_dtype": "float32",
        "transition_strategy": transition_strategy,
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 0.25,
            "text_alignment": 0.5,
        },
        "reward_model": {"type": "videoalign"},
    }
    if transition_strategy == "variance-preserving":
        algorithm["sigma_max"] = 0.98
    if algorithm_type == "flow-grpo":
        algorithm["clip_range"] = 0.0002
        algorithm["clip_schedule"] = "linear_decay"
        algorithm["clip_schedule_steps"] = 40
    elif algorithm_type == "flow-dppo":
        algorithm["kl_mask_threshold"] = 0.025
        algorithm["add_kl_coefficient"] = False
    else:
        raise ValueError(f"unsupported test algorithm: {algorithm_type}")
    value["algorithm"] = algorithm
    value["optimizer"] = {
        "type": "adamw",
        "learning_rate": 3.0e-4,
        "weight_decay": 0.02,
        "max_grad_norm": 0.5,
    }
    del value["fake_score_optimizer"]
    return value


def test_dmd_builder_materializes_recipe_math_optimizers_and_checkpoint_inventory() -> None:
    recipe = PostTrainingRecipe.from_mapping(_dmd_recipe_mapping())
    student = _FlowPredictor()
    teacher = _FlowPredictor(
        trainable=False,
        checkpoint_identity="teacher-checkpoint",
    )
    fake_score = _FlowPredictor(checkpoint_identity="critic-checkpoint")
    student_scheduler = _StatefulCounter()
    fake_score_scheduler = _StatefulCounter()
    ema = _StatefulCounter()

    stack = build_native_dmd_training_stack(
        recipe,
        student=student,
        real_score=teacher,
        fake_score=fake_score,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=ema,
        fused_adamw=False,
    )

    assert isinstance(stack, NativeDMDTrainingStack)
    assert stack.config.schedule.timesteps == (1000.0, 750.0, 500.0)
    assert stack.config.score_flow_shift == 5.0
    assert stack.engine.generator_update_interval == 3
    assert stack.engine.student_max_grad_norm == 0.7
    assert stack.engine.fake_score_max_grad_norm == 0.9
    assert stack.student_optimizer.param_groups[0]["lr"] == 2.0e-6
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 4.0e-6
    assert stack.scheduler_state is not None
    assert stack.scheduler_state.component_names == ("fake_score", "student")
    assert stack.ema_state is not None
    assert stack.checkpoint_state_kwargs() == {
        "lr_scheduler": stack.scheduler_state,
        "ema": stack.ema_state,
        "algorithm_state": None,
    }
    teacher.checkpoint_identity = "wrong-teacher"
    with pytest.raises(ValueError, match="differs from recipe"):
        build_native_dmd_training_stack(
            recipe,
            student=student,
            real_score=teacher,
            fake_score=fake_score,
            fused_adamw=False,
        )


def test_flow_grpo_builder_materializes_rollout_replay_scalarizer_and_update_contract() -> None:
    recipe = PostTrainingRecipe.from_mapping(_flow_recipe_mapping())
    policy = _FlowPredictor()

    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=policy,
        initial_policy_revision="policy-checkpoint",
        fused_adamw=False,
    )

    assert isinstance(stack, NativeFlowPolicyTrainingStack)
    assert stack.sampler.eta == 0.6
    assert stack.sampler.sigma_max == 0.98
    assert isinstance(stack.transition_strategy, VariancePreservingFlowTransition)
    assert stack.sampler.trajectory_dtype is torch.float32
    assert stack.scalarizer.calibration_mean == {
        "video_quality": 3.6757,
        "motion_quality": 1.1646,
        "text_alignment": 2.8105,
    }
    assert stack.scalarizer.normalization_epsilon == 0.0
    assert stack.sigmas == (1.0, 0.7, 0.3, 0.0)
    assert stack.sde_step_indices == (1, 2)
    assert stack.group_size == 4
    assert stack.old_log_prob_source == "replay"
    assert stack.advantage_epsilon == 1.0e-7
    assert stack.advantage_clip_max == 4.0
    assert dict(stack.scalarizer.weights) == {
        "video_quality": 1.0,
        "motion_quality": 0.25,
        "text_alignment": 0.5,
    }
    assert stack.engine.updates_per_trajectory == 2
    assert stack.engine.clip_range == 0.0002
    assert stack.engine.algorithm.clip_schedule == "linear-decay"
    assert stack.engine.algorithm.clip_schedule_steps == 40
    assert stack.session_type is NativeFlowGRPOTrainingSession
    assert stack.optimizer.param_groups[0]["lr"] == 3.0e-4
    assert stack.checkpoint_state_kwargs()["algorithm_state"] is stack.scalarizer


def test_flow_policy_builder_dispatches_dppo_without_copying_shared_stack() -> None:
    recipe = PostTrainingRecipe.from_mapping(
        _flow_recipe_mapping(
            algorithm_type="flow-dppo",
            transition_strategy="constant-diffusion",
        )
    )

    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=_FlowPredictor(),
        initial_policy_revision="policy-checkpoint",
        fused_adamw=False,
    )

    assert isinstance(stack, NativeFlowPolicyTrainingStack)
    assert isinstance(stack.engine, NativeFlowDPPOEngine)
    assert stack.engine.kl_mask_threshold == 0.025
    assert stack.engine.add_kl_coefficient is False
    assert stack.session_type is NativeFlowDPPOTrainingSession
    assert isinstance(stack.transition_strategy, ConstantDiffusionFlowTransition)
    assert stack.sampler.sigma_max is None
    assert "sigma_max" not in dict(stack.transition_strategy.identity)
    assert stack.sampler.eta == 0.6
    assert stack.sde_step_indices == (1, 2)
    assert stack.optimizer.param_groups[0]["lr"] == 3.0e-4


def test_flow_policy_builder_materializes_sliding_sde_window_without_mutable_state() -> None:
    mapping = _flow_recipe_mapping()
    algorithm = mapping["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm.pop("sde_step_indices")
    algorithm["sde_window"] = {
        "window_size": 2,
        "iterations_per_window": 3,
        "stride": 1,
        "rollback": True,
    }
    recipe = PostTrainingRecipe.from_mapping(mapping)

    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=_FlowPredictor(),
        initial_policy_revision="policy-checkpoint",
        fused_adamw=False,
    )

    assert isinstance(stack.sde_index_schedule, FlowSDEWindowSchedule)
    assert stack.sde_step_indices is None
    assert stack.sde_index_schedule.resolve(0) == (0, 1)
    assert stack.sde_index_schedule.resolve(3) == (1, 2)
    assert stack.sde_index_schedule.resolve(6) == (0, 1)


def test_builders_materialize_dmd_accumulation_and_reject_unused_reference_semantics() -> None:
    dmd_mapping = _dmd_recipe_mapping()
    dmd_mapping["optimizer"]["gradient_accumulation_steps"] = 2
    dmd_mapping["fake_score_optimizer"]["gradient_accumulation_steps"] = 2
    dmd = PostTrainingRecipe.from_mapping(dmd_mapping)
    dmd_stack = build_native_dmd_training_stack(
        dmd,
        student=_FlowPredictor(),
        real_score=_FlowPredictor(
            trainable=False,
            checkpoint_identity="teacher-checkpoint",
        ),
        fake_score=_FlowPredictor(checkpoint_identity="critic-checkpoint"),
        fused_adamw=False,
    )
    assert dmd_stack.engine.gradient_accumulation_steps == 2

    flow = PostTrainingRecipe.from_mapping(_flow_recipe_mapping())
    with pytest.raises(ValueError, match="unused"):
        build_native_flow_policy_training_stack(
            flow,
            policy=_FlowPredictor(),
            reference_policy=_FlowPredictor(trainable=False),
            initial_policy_revision="policy-checkpoint",
            fused_adamw=False,
        )
