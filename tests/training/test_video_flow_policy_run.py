from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch
from worldfoundry.training.post_training.rl.run import (
    build_native_flow_policy_training_run,
)
from worldfoundry.training.recipes import PostTrainingRecipe


class _TinyPrediction:
    def __init__(self) -> None:
        self.module = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.module.weight, 0.2)

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
        return noisy_latents - self.predict_velocity(noisy_latents, sigmas, **kwargs)


class _Reward:
    def score(self, trajectory):
        values = trajectory.latents[:, -1].float().flatten(1).mean(dim=1)
        return {
            "video_quality": values,
            "motion_quality": values * 0.5,
            "text_alignment": -values,
        }


class _RolloutLoader:
    def __init__(self, stack) -> None:
        self.stack = stack
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        offset = float(self.cursor) * 0.1
        self.cursor += 1
        return FlowRolloutBatch(
            sample_ids=(f"sample-{self.cursor}-a", f"sample-{self.cursor}-b"),
            group_ids=(f"group-{self.cursor}", f"group-{self.cursor}"),
            policy_revision=self.stack.engine.current_policy_revision,
            initial_latents=torch.tensor([1.0 + offset, -0.5 - offset]).reshape(2, 1, 1, 1, 1),
            sigmas=torch.tensor(self.stack.sigmas),
            conditioning={"context": torch.ones(2, 1, 1)},
        )

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict):
        self.cursor = int(state_dict["cursor"])


def _recipe(output_dir: Path) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-video-policy", "output_dir": str(output_dir)},
            "model": {"recipe": "tiny-video", "checkpoint": "tiny-policy"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "shuffle": False,
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
            "checkpoint": {
                "save_every_steps": 1,
                "async": False,
                "export_every_steps": 2,
            },
            "export": {"format": "safetensors"},
        }
    )


def _run(output_dir: Path, *, resume_checkpoint: Path | None = None):
    recipe = _recipe(output_dir)
    prediction = _TinyPrediction()
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision="tiny-policy",
        fused_adamw=False,
    )
    generator = torch.Generator().manual_seed(123)
    loader = _RolloutLoader(stack)
    return build_native_flow_policy_training_run(
        recipe,
        stack=stack,
        dataloader=loader,
        reward_adapter=_Reward(),
        policy_module=prediction.module,
        policy_tuning=None,
        objective_generator=generator,
        output_dir=output_dir,
        resume_identity={"model": "tiny-video", "data": "two-sample-groups"},
        resume_checkpoint=resume_checkpoint,
    )


def test_flow_policy_run_exact_resume_and_full_model_export(tmp_path: Path) -> None:
    uninterrupted = _run(tmp_path / "uninterrupted")
    uninterrupted.run(max_iterations=2)
    uninterrupted_weight = uninterrupted.policy_module.weight.detach().clone()
    assert (uninterrupted.output_dir / "exports/step-00000002/policy").is_dir()
    artifact = uninterrupted.export_policy()
    assert artifact.path.is_dir()
    uninterrupted.close()

    first = _run(tmp_path / "first")
    first.run(max_iterations=1)
    checkpoint = first.checkpointer.root / "step-00000001"
    first.close()

    resumed = _run(tmp_path / "resumed", resume_checkpoint=checkpoint)
    summary = resumed.run(max_iterations=1)
    assert summary.initial_optimizer_step == 1
    assert summary.final_optimizer_step == 2
    torch.testing.assert_close(resumed.policy_module.weight, uninterrupted_weight)
    assert resumed.dataloader.cursor == 2
    assert (resumed.output_dir / "exports/step-00000002/policy").is_dir()
    resumed.export_policy()
    resumed.close()
