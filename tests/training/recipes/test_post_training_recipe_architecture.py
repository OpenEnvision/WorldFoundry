from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from worldfoundry.training import recipes
from worldfoundry.training.recipes.post_training import (
    DMDAlgorithmSpec,
    FlowDPPOAlgorithmSpec,
    FlowGRPOAlgorithmSpec,
    FlowPolicyAlgorithmSpec,
    PostTrainingRecipe,
    VideoAlignRewardSpec,
)
from worldfoundry.training.recipes.post_training.recipe import (
    DMDAlgorithmSpec as LegacyDMDAlgorithmSpec,
)


def _golden_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {
                "id": "architecture-golden",
                "output_dir": "runs/architecture-golden",
            },
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "default",
            },
            "tuning": {
                "mode": "lora",
                "preset": "wan-attention",
                "rank": 8,
                "alpha": 16,
            },
            "data": {"manifest": "data/latents.jsonl"},
            "algorithm": {
                "type": "dmd",
                "student_timesteps": [1000, 500],
                "student_sigmas": [1.0, 0.5],
                "real_score_checkpoint": "teacher",
                "fake_score_checkpoint": "critic",
            },
            "optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
            "fake_score_optimizer": {
                "type": "adamw",
                "learning_rate": 2.0e-6,
            },
        }
    )


def test_recipe_contracts_are_split_by_responsibility_and_reexported() -> None:
    assert DMDAlgorithmSpec.__module__.endswith(".algorithms.dmd")
    assert FlowDPPOAlgorithmSpec.__module__.endswith(".algorithms.flow_dppo")
    assert FlowGRPOAlgorithmSpec.__module__.endswith(".algorithms.flow_grpo")
    assert FlowPolicyAlgorithmSpec.__module__.endswith(".algorithms.flow_policy")
    assert VideoAlignRewardSpec.__module__.endswith(".rewards.videoalign")
    assert PostTrainingRecipe.__module__.endswith(".recipe")

    assert recipes.DMDAlgorithmSpec is DMDAlgorithmSpec
    assert recipes.FlowDPPOAlgorithmSpec is FlowDPPOAlgorithmSpec
    assert recipes.FlowGRPOAlgorithmSpec is FlowGRPOAlgorithmSpec
    assert recipes.FlowPolicyAlgorithmSpec is FlowPolicyAlgorithmSpec
    assert recipes.VideoAlignRewardSpec is VideoAlignRewardSpec
    assert recipes.PostTrainingRecipe is PostTrainingRecipe
    assert LegacyDMDAlgorithmSpec is DMDAlgorithmSpec


def test_clean_recipe_import_does_not_load_execution_packages() -> None:
    root = Path(__file__).resolve().parents[3]
    probe = """
import json
import sys

import worldfoundry.training.recipes

blocked = [
    name
    for name in sys.modules
    if name.startswith(
        (
            "worldfoundry.training.post_training",
            "worldfoundry.training.engine",
            "worldfoundry.training.models",
            "worldfoundry.training.tuning",
        )
    )
]
print(json.dumps(sorted(blocked)))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_recipe_modules_have_no_top_level_execution_plane_imports() -> None:
    package = Path(__file__).resolve().parents[3] / "worldfoundry" / "training" / "recipes" / "post_training"
    blocked = (
        "worldfoundry.training.post_training",
        "worldfoundry.training.engine",
        "worldfoundry.training.models",
        "worldfoundry.training.tuning",
    )

    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module_name in imported:
                if module_name.startswith(blocked):
                    violations.append(f"{path.relative_to(package)}:{node.lineno}: {module_name}")

    assert violations == []


def test_recipe_serialization_round_trip_is_stable_across_file_split() -> None:
    recipe = _golden_recipe()

    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
