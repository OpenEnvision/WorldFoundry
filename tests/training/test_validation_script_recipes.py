from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = _ROOT / "scripts" / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"worldfoundry_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_roundtrip(recipe: object) -> None:
    recipe_type = type(recipe)
    restored = recipe_type.from_mapping(recipe.to_dict())
    assert restored == recipe


def test_pretraining_validation_scripts_build_current_recipes(tmp_path: Path) -> None:
    sana = _load_script("validate_sana_training_roundtrip")
    _assert_roundtrip(
        sana._recipe(
            work_dir=tmp_path,
            manifest_path=tmp_path / "manifest.jsonl",
            cache_dir=tmp_path / "cache",
            sana_checkpoint=tmp_path / "sana.pth",
            learning_rate=1.0e-3,
        )
    )

    wan = _load_script("validate_wan_training_roundtrip")
    _assert_roundtrip(
        wan._recipe(
            work_dir=tmp_path,
            manifest_path=tmp_path / "manifest.jsonl",
            cache_dir=tmp_path / "cache",
            learning_rate=1.0e-3,
        )
    )


def test_post_training_validation_scripts_build_current_recipes(tmp_path: Path) -> None:
    dmd = _load_script("validate_wan_post_training_roundtrip")
    dmd_recipe = dmd._recipe(
        work_dir=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        cache_dir=tmp_path / "cache",
    )
    _assert_roundtrip(dmd_recipe)
    assert dmd_recipe.algorithm.generator_update_interval == 5
    assert dmd_recipe.optimizer.gradient_accumulation_steps == 8
    assert dmd_recipe.fake_score_optimizer.gradient_accumulation_steps == 8

    flow_grpo = _load_script("validate_wan_flow_grpo_roundtrip")
    _assert_roundtrip(
        flow_grpo._recipe(
            manifest_path=tmp_path / "prompts.jsonl",
            cache_path=tmp_path / "conditioning",
            work_dir=tmp_path,
        )
    )

    sana_sid = _load_script("validate_sana_sid_roundtrip")
    source = sana_sid.PostTrainingRecipe.from_file(
        _ROOT / "configs" / "post_training" / "sana_sprint_600m_sid.yaml"
    )
    _assert_roundtrip(
        sana_sid._gate_recipe(
            source,
            manifest=tmp_path / "manifest.jsonl",
            cache=tmp_path / "cache",
            work_dir=tmp_path,
            steps=1,
            height=None,
            width=None,
        )
    )
