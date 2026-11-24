# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.le_wm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("local_path", [True, False])
def test_public_model_path_selects_the_learned_policy(tmp_path: Path, local_path: bool) -> None:
    from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import LeWorldModelPipeline

    cache = tmp_path / "cache"
    checkpoint = cache / "learned-policy_object.ckpt"
    selector = str(checkpoint) if local_path else "learned-policy"
    pipeline = LeWorldModelPipeline.from_pretrained(model_path=selector, cache_dir=str(cache), device="cpu")
    synthesis = pipeline.synthesis_model
    assert synthesis.options["policy"] == selector
    assert "repo_root" not in synthesis.options
    assert synthesis.device == "cpu"
    assert synthesis._checkpoint_paths() == [checkpoint]
    assert runtime.policy_checkpoint_path(synthesis.options) == checkpoint
    command = synthesis._command(output_path=tmp_path / "result.json", prompt="", plan_path=tmp_path / "plan.json")
    assert command[command.index("--policy") + 1] == selector
    assert command[command.index("--cache-dir") + 1] == str(cache)


def test_random_policy_has_no_object_checkpoint() -> None:
    assert runtime.policy_checkpoint_path({"policy": "random"}) is None


@pytest.mark.parametrize("suffix,artifact", [(".mp4", ".result.json"), (".json", ".json")])
@pytest.mark.parametrize("policy", ["random", "learned-policy"])
def test_package_command_preserves_policy_data_options_and_artifact_names(
    tmp_path: Path, suffix: str, artifact: str, policy: str
) -> None:
    options = {
        "config_dir": str(tmp_path / "config"),
        "config_name": "pusht",
        "policy": policy,
        "cache_dir": str(tmp_path / "cache"),
        "dataset_name": str(tmp_path / "data.h5"),
        "seed": 7,
        "num_eval": 2,
        "eval_budget": 32,
        "goal_offset_steps": 5,
        "img_size": 224,
    }
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": str(tmp_path / f"result{suffix}"),
            "output_dir": str(tmp_path),
            "device": "cuda:1",
            "options": options,
        }
    )
    assert command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.le_wm.infer"]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--config-dir": options["config_dir"],
        "--config-name": "pusht",
        "--policy": policy,
        "--output-dir": str(tmp_path),
        "--artifact-path": str(tmp_path / f"result{artifact}"),
        "--device": "cuda:1",
        "--cache-dir": options["cache_dir"],
        "--dataset-name": options["dataset_name"],
        "--seed": "7",
        "--num-eval": "2",
        "--eval-budget": "32",
        "--goal-offset-steps": "5",
        "--img-size": "224",
    }
