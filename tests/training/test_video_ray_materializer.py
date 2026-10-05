from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.training.fixtures.ray_video_e2e import ray_tiny_video_policy_factory
from worldfoundry.training.distributed.flow_rollout import RayFlowTrajectorySampler
from worldfoundry.training.distributed.rollout_runtime import RayPostTrainingRuntime
from worldfoundry.training.engine.video_policy import (
    VideoFlowPolicyMaterialization,
    materialize_video_flow_policy_training_run,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch
from worldfoundry.training.recipes import PostTrainingRecipe


class _TinyPrediction:
    def __init__(self, value: float = 0.2) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.module.weight, value)

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
        return noisy_latents * self.module.weight.reshape(1, 1, 1, 1, 1)

    def predict_clean(self, noisy_latents, sigmas, **kwargs):
        return noisy_latents - self.predict_velocity(
            noisy_latents,
            sigmas,
            **kwargs,
        )


class _Reward:
    def score(self, trajectory):
        terminal = trajectory.latents[:, -1].float().flatten(1).mean(dim=1)
        return {
            "video_quality": terminal,
            "motion_quality": terminal.square(),
            "text_alignment": -terminal.abs(),
        }


class _RolloutLoader:
    def __init__(self, stack) -> None:
        self.stack = stack
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        offset = self.cursor / 10
        return FlowRolloutBatch(
            sample_ids=(f"sample-{self.cursor}-a", f"sample-{self.cursor}-b"),
            group_ids=(f"group-{self.cursor}", f"group-{self.cursor}"),
            policy_revision=self.stack.engine.current_policy_revision,
            initial_latents=torch.tensor([1.0 + offset, -0.5 - offset]).reshape(
                2,
                1,
                1,
                1,
                1,
            ),
            sigmas=torch.tensor(self.stack.sigmas),
            conditioning={"context": torch.ones(2, 1, 1)},
        )

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict):
        self.cursor = int(state_dict["cursor"])


class _Conditioning:
    index = SimpleNamespace(
        to_dict=lambda: {
            "model_recipe": "wan2.2-t2v-a14b",
            "samples": 1,
        }
    )

    def __len__(self) -> int:
        return 1


def _recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-wan22-ray", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.2-t2v-a14b",
                "checkpoint": "tiny-policy",
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "shuffle": False,
                "tail_policy": "pad",
                "options": {
                    "global_prompt_batch_size": 1,
                    "generation": {"height": 1, "width": 1, "num_frames": 1},
                },
            },
            "algorithm": {
                "type": "flow-grpo",
                "sigmas": [1.0, 0.5, 0.0],
                "sde_step_indices": [0],
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
            "rollout": {
                "backend": "ray",
                "pool": {
                    "num_devices": 1,
                    "devices_per_node": 1,
                    "workers_per_device": 1,
                    "cpus_per_worker": 0.25,
                    "accelerator_resource": "CPU",
                },
                "rollout_devices": 1,
                "trainer_binding": "external",
                "placement": "separate",
                "weight_kind": "full",
                "weight_bucket_bytes": 16,
            },
            "checkpoint": {
                "save_every_steps": 1,
                "async": False,
                "export_every_steps": 1,
            },
            "export": {"format": "safetensors"},
        }
    )


def _patch_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.training.engine import video_policy

    conditioning = _Conditioning()

    def materialize_roles(recipe, **kwargs):
        del kwargs
        prediction = _TinyPrediction()
        stack = build_native_flow_policy_training_stack(
            recipe,
            policy=prediction,
            initial_policy_revision=recipe.model.checkpoint,
            fused_adamw=False,
        )
        return VideoFlowPolicyMaterialization(
            policy=SimpleNamespace(trainable_module=prediction.module),
            reference_policy=None,
            stack=stack,
            generation={"height": 1, "width": 1, "num_frames": 1},
            latent_shape=(1, 1, 1, 1),
            policy_tuning=None,
        )

    monkeypatch.setattr(
        video_policy,
        "_load_conditioning_dataset",
        lambda *args, **kwargs: (object(), conditioning),
    )
    monkeypatch.setattr(
        video_policy,
        "_build_conditioned_source",
        lambda *args, **kwargs: (
            object(),
            torch.Generator().manual_seed(123),
            conditioning,
        ),
    )
    monkeypatch.setattr(
        video_policy,
        "materialize_video_flow_policy_roles",
        materialize_roles,
    )
    monkeypatch.setattr(
        VideoFlowPolicyMaterialization,
        "build_rollout_loader",
        lambda self, *args, **kwargs: _RolloutLoader(self.stack),
    )
    monkeypatch.setattr(video_policy, "_build_native_decoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        video_policy,
        "DecodedTerminalRewardAdapter",
        lambda *args, **kwargs: _Reward(),
    )


def _materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    resume_checkpoint: Path | None = None,
):
    _patch_materializer(monkeypatch)
    return materialize_video_flow_policy_training_run(
        _recipe(),
        base_dir=tmp_path,
        device="cpu",
        output_dir=tmp_path / name,
        resume_checkpoint=resume_checkpoint,
        reward_evaluator=object(),
        fused_adamw=False,
        rollout_policy_factory=ray_tiny_video_policy_factory,
    )


def test_formal_wan22_ray_materializer_updates_resumes_and_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ray")
    first = _materialize(tmp_path, monkeypatch, name="first")
    assert isinstance(first.session.sampler, RayFlowTrajectorySampler)
    runtime = next(resource for resource in first.closeables if isinstance(resource, RayPostTrainingRuntime))
    assert runtime.rollout_group is not None
    assert runtime.rollout_group.lease.placement.value == "separate"

    summary = first.run(max_iterations=1)
    assert summary.final_optimizer_step == 1
    assert first.session.sampler.last_sync_report is not None
    assert first.session.sampler.last_sync_report.revision == 0
    assert first.session.sampler.last_sync_report.receiver_count == 1
    assert (first.output_dir / "exports/step-00000001/policy").is_dir()
    checkpoint = first.checkpointer.root / "step-00000001"
    saved_weight = first.policy_module.weight.detach().clone()
    first.close()

    resumed = _materialize(
        tmp_path,
        monkeypatch,
        name="resumed",
        resume_checkpoint=checkpoint,
    )
    torch.testing.assert_close(resumed.policy_module.weight, saved_weight)
    assert resumed.dataloader.cursor == 1
    resumed_summary = resumed.run(max_iterations=1)
    assert resumed_summary.initial_optimizer_step == 1
    assert resumed_summary.final_optimizer_step == 2
    assert resumed.session.sampler.last_sync_report is not None
    assert resumed.session.sampler.last_sync_report.revision == 0
    assert (resumed.output_dir / "exports/step-00000001/policy").is_dir()
    assert (resumed.output_dir / "exports/step-00000002/policy").is_dir()

    resumed.run(max_iterations=1)
    assert (resumed.output_dir / "exports/step-00000003/policy").is_dir()
    resumed.close()


def test_video_ray_materializer_rejects_per_rank_fsdp_runtime() -> None:
    mapping = _recipe().to_dict()
    mapping["distributed"] = {
        "backend": "fsdp2",
        "dp_replicate": 1,
        "dp_shard": "auto",
    }
    recipe = PostTrainingRecipe.from_mapping(mapping)
    from worldfoundry.training.engine.video_policy import _ray_video_runtime_config

    with pytest.raises(ValueError, match="per-rank FSDP2"):
        _ray_video_runtime_config(recipe)
