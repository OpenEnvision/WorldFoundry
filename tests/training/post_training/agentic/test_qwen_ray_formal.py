from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.training.fixtures.qwen_ray_formal import (
    RayQwenPolicy,
    RayQwenTokenizer,
    ray_qwen_rollout_policy_factory,
    ray_qwen_tokenizer_factory,
    ray_qwen_trainer_policy_factory,
)
from worldfoundry.training.post_training.agentic import AgenticPrompt, AgentMessage
from worldfoundry.training.post_training.causal_lm.qwen3 import (
    materialize_qwen3_agentic_training_run,
)
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import FullModelArtifact

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "tests/training/fixtures/qwen3_agentic_ray_cpu.yaml"


def _prompt() -> tuple[AgenticPrompt, ...]:
    return (
        AgenticPrompt(
            prompt_id="math",
            messages=(AgentMessage(role="user", content="Use the calculator for 2+3."),),
            conditioning={"answer": "5"},
        ),
    )


def _materialize(recipe, output_dir, *, resume_checkpoint=None):
    return materialize_qwen3_agentic_training_run(
        recipe,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        device="cpu",
        prompts=_prompt(),
        ray_trainer_policy_factory=ray_qwen_trainer_policy_factory,
        ray_tokenizer_factory=ray_qwen_tokenizer_factory,
        ray_rollout_policy_factory=ray_qwen_rollout_policy_factory,
        fused_adamw=False,
    )


@pytest.mark.parametrize("placement", ("separate", "colocate"))
def test_actor_qwen_materializer_updates_and_uses_requested_placement(
    tmp_path: Path,
    placement: str,
) -> None:
    pytest.importorskip("ray")
    payload = PostTrainingRecipe.from_file(RECIPE).to_dict()
    rollout = payload["rollout"]
    if placement == "separate":
        rollout["placement"] = "separate"
        rollout["pool"]["num_devices"] = 2
        rollout["pool"]["devices_per_node"] = 2
        rollout["pool"]["workers_per_device"] = 1
    recipe = PostTrainingRecipe.from_mapping(payload)
    output_dir = tmp_path / f"actor-{placement}"
    first = _materialize(recipe, output_dir)
    actual_placement = first.placement()
    assert actual_placement["rollout_placement"] == placement
    if placement == "colocate":
        assert actual_placement["rollout_devices"] == (actual_placement["trainer_device"],)
        assert (actual_placement["trainer_slot"], actual_placement["rollout_slot"]) == (0, 1)
    else:
        assert actual_placement["rollout_devices"] != (actual_placement["trainer_device"],)
    initial = first.policy_state()["transitions"].clone()
    summary = first.run(max_iterations=1)
    assert summary.final_optimizer_step == 1
    assert first.rollout_state() == {
        "schema": "worldfoundry-ray-agentic-rollout",
        "completed_rollouts": 1,
        "weight_revision": 0,
    }
    assert not torch.equal(first.policy_state()["transitions"], initial)
    first.close()

    if placement == "separate":
        return

    uninterrupted = _materialize(recipe, tmp_path / "actor-colocate-full")
    assert uninterrupted.run(max_iterations=2).final_optimizer_step == 2
    expected_policy = uninterrupted.policy_state()
    expected_rollout = uninterrupted.rollout_state()
    uninterrupted.close()

    resumed = _materialize(recipe, output_dir, resume_checkpoint="latest")
    summary = resumed.run(max_iterations=1)
    assert (summary.initial_optimizer_step, summary.final_optimizer_step) == (1, 2)
    assert resumed.rollout_state()["completed_rollouts"] == 2
    assert resumed.rollout_state()["weight_revision"] == 1
    for name, expected in expected_policy.items():
        torch.testing.assert_close(resumed.policy_state()[name], expected, rtol=0, atol=0)
    assert resumed.rollout_state() == expected_rollout
    artifact = resumed.export_policy()
    assert isinstance(artifact, FullModelArtifact)
    assert artifact.path.is_dir()
    resumed.close()


def test_external_separate_qwen_materializer_owns_remote_rollout(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    payload = PostTrainingRecipe.from_file(RECIPE).to_dict()
    rollout = payload["rollout"]
    rollout["trainer_binding"] = "external"
    rollout["placement"] = "separate"
    rollout.pop("trainer_devices")
    rollout["pool"]["workers_per_device"] = 1
    recipe = PostTrainingRecipe.from_mapping(payload)
    run = materialize_qwen3_agentic_training_run(
        recipe,
        output_dir=tmp_path / "external-run",
        device="cpu",
        prompts=_prompt(),
        policy_module=RayQwenPolicy(),
        tokenizer=RayQwenTokenizer(),
        ray_rollout_policy_factory=ray_qwen_rollout_policy_factory,
        fused_adamw=False,
    )
    summary = run.run(max_iterations=1)
    assert summary.final_optimizer_step == 1
    assert run.rollout_adapter.runtime.rollout_group.lease.placement.value == "separate"
    assert run.rollout_adapter.last_sync_report.receiver_count == 1
    run.close()
