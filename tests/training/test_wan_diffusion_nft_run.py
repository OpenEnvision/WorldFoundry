from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.engine import (  # noqa: E402
    WanDiffusionNFTTrainingRun,
    validate_wan_diffusion_nft_recipe,
)
from worldfoundry.training.post_training.rewards.contracts import (  # noqa: E402
    RewardResult,
)
from worldfoundry.training.post_training.rewards.scalarization import (  # noqa: E402
    RewardScalarizationResult,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft import (  # noqa: E402
    DiffusionNFTIterationResult,
    DiffusionNFTRollout,
    DiffusionNFTStepResult,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402


def _recipe() -> PostTrainingRecipe:
    root = Path(__file__).resolve().parents[2]
    return PostTrainingRecipe.from_file(root / "configs/post_training/wan_1p3b_diffusion_nft.yaml")


def test_wan_diffusion_nft_recipe_resolves_single_device_data_plane() -> None:
    recipe = _recipe()
    algorithm, data_plan = validate_wan_diffusion_nft_recipe(recipe)

    assert recipe.distributed.backend == "single"
    assert algorithm.num_train_timesteps == 1000
    assert algorithm.collection.group_size == 4
    assert data_plan.generation == {
        "height": 256,
        "width": 416,
        "num_frames": 17,
    }


def test_wan_diffusion_nft_rejects_unimplemented_sharded_old_policy() -> None:
    recipe = _recipe()
    fsdp = replace(
        recipe,
        distributed=replace(recipe.distributed, backend="fsdp2"),
    )

    with pytest.raises(ValueError, match="old-policy refresh.*sharding-aware"):
        validate_wan_diffusion_nft_recipe(fsdp)


class _Progress:
    optimizer_steps = 0

    def state_dict(self):
        return {"optimizer_steps": self.optimizer_steps}


class _Session:
    def __init__(self, result: DiffusionNFTIterationResult) -> None:
        self.result = result
        self.engine = SimpleNamespace(
            global_step=0,
            current_collection_policy_revision="old-policy-next",
        )
        self.progress = _Progress()

    def train_iteration(self, batch, *, generator):
        assert batch == "batch"
        assert isinstance(generator, torch.Generator)
        self.engine.global_step += 1
        self.progress.optimizer_steps += 1
        return self.result

    def wait_for_checkpoints(self):
        return None


class _Roles:
    policy = SimpleNamespace(trainable_module=torch.nn.Linear(1, 1))
    policy_peft = None
    policy_fsdp = None

    @staticmethod
    def runtime_identity():
        return {"policy": {"checkpoint": {"repository": "policy"}}}


def _iteration_result() -> DiffusionNFTIterationResult:
    rollout = DiffusionNFTRollout(
        collection_id="collection",
        policy_revision="old-policy-current",
        sample_ids=("a", "b"),
        group_ids=("prompt", "prompt"),
        clean_latents=torch.ones(2, 1),
        rewards=torch.tensor([0.25, 0.75]),
    )
    zero = torch.tensor(0.0)
    update = DiffusionNFTStepResult(
        loss=torch.tensor(1.25),
        policy_loss=torch.tensor(1.0),
        reference_mse=None,
        advantages=torch.tensor([-1.0, 1.0]),
        reward_probabilities=torch.tensor([0.25, 0.75]),
        times=torch.tensor([0.2, 0.8]),
        gradient_norm=zero,
        old_policy_refreshed=True,
        old_policy_retention=0.0,
        metrics={},
    )
    scalarization = RewardScalarizationResult(
        scalar_rewards=torch.tensor([0.25, 0.75]),
        normalized_components={"video_quality": torch.tensor([0.25, 0.75])},
        valid_mask=torch.ones(2, dtype=torch.bool),
    )
    return DiffusionNFTIterationResult(
        rollout=rollout,
        update=update,
        reward_components={"video_quality": torch.tensor([0.25, 0.75])},
        scalarization=scalarization,
    )


def test_wan_diffusion_nft_run_commits_metrics_and_summary(tmp_path: Path) -> None:
    recipe = replace(
        _recipe(),
        checkpoint=replace(_recipe().checkpoint, export_every_steps=0),
    )
    output = tmp_path / "run"
    output.mkdir()
    reward_adapter = SimpleNamespace(
        reward_ids=("video_quality",),
        last_results=(
            RewardResult(
                request_id="a",
                rollout_id="rollout:a",
                values={"video_quality": 0.25},
                valid={"video_quality": True},
                diagnostics={},
                latency_ms=1.0,
            ),
            RewardResult(
                request_id="b",
                rollout_id="rollout:b",
                values={"video_quality": 0.75},
                valid={"video_quality": True},
                diagnostics={},
                latency_ms=1.0,
            ),
        ),
    )
    session = _Session(_iteration_result())
    run = WanDiffusionNFTTrainingRun(
        recipe=recipe,
        session=session,
        dataloader=["batch"],
        checkpoint_state=SimpleNamespace(objective_generator=torch.Generator().manual_seed(3)),
        checkpointer=SimpleNamespace(),
        roles=_Roles(),
        reward_adapter=reward_adapter,
        output_dir=output,
        data_identity={"dataset": "toy"},
        reward_identity={"reward": "toy"},
        resume_artifact=None,
        distributed_context=None,
    )

    summary = run.run(max_iterations=1)

    assert summary.initial_optimizer_step == 0
    assert summary.final_optimizer_step == 1
    assert summary.final_scalar_reward_mean == pytest.approx(0.5)
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["schema"] == "worldfoundry-wan-diffusion-nft-run"
    assert manifest["status"] == "complete"
    metric = json.loads((output / "metrics.jsonl").read_text())
    assert metric["collection_policy_revision"] == "old-policy-current"
    assert metric["next_collection_policy_revision"] == "old-policy-next"
