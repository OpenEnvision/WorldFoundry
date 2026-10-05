from __future__ import annotations

import ast
from pathlib import Path

import pytest

from worldfoundry.evaluation.models import ModelResolutionError, resolve_model_zoo_config
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
VISUAL_GENERATION_ROOT = REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation"

# Models whose catalog entries still record a planned (not integrated)
# in-tree runner surface.  cameractrl/motionctrl/open-magvit2/show-o/
# splatt3r/step-video-t2v graduated to integrated runnable runners once the
# schema registered the catalog's verified/in-tree integration statuses, so
# they no longer belong in this list.
NON_RUNNABLE_VISUAL_MODEL_IDS = frozenset(
    {
        "dreamdojo",
        "irasim",
        "pandora",
        "pixelsplat",
    }
)

ALLOWED_RUNTIME_PROFILE_SYNTHESIS_FILES = frozenset(
    {
        "animatediff/animatediff_synthesis.py",
        "zeroscope/zeroscope_synthesis.py",
    }
)


def test_non_runnable_visual_models_do_not_claim_runnable_runner_status() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id in sorted(NON_RUNNABLE_VISUAL_MODEL_IDS):
        entry = registry.get(model_id)
        assert entry.is_runnable_runner_entry is False, model_id
        assert all(not variant.is_runnable_runner_entry for variant in entry.variants), model_id
        if entry.runner_entry_kind == "listed_only":
            assert entry.runner_target is None, model_id
            assert entry.pipeline_target is None, model_id
            assert all(variant.runner_target is None for variant in entry.variants), model_id
            assert all(variant.pipeline_target is None for variant in entry.variants), model_id
        else:
            assert entry.runner_entry_kind == "runner_candidate", model_id
            assert entry.runner_target is not None, model_id
            assert entry.pipeline_target is not None, model_id


@pytest.mark.parametrize("model_id", sorted(NON_RUNNABLE_VISUAL_MODEL_IDS))
def test_non_runnable_visual_model_config_resolution_stays_non_runnable(model_id: str) -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get(model_id)

    if entry.runner_entry_kind == "listed_only":
        with pytest.raises(ModelResolutionError, match="listed_only|no runner_target"):
            resolve_model_zoo_config(
                model_id,
                manifest_dir=MODEL_CATALOG_DIR,
                runtime={"device": "cpu"},
            )
        return

    resolved = resolve_model_zoo_config(
        model_id,
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )
    assert resolved.diagnostics["entry_runner_entry_kind"] == "runner_candidate"
    assert resolved.diagnostics["entry_integration_status"] != "integrated"


def test_visual_generation_runtime_profile_synthesis_is_limited_to_in_tree_runtimes() -> None:
    offenders: list[str] = []
    for path in sorted(VISUAL_GENERATION_ROOT.rglob("*_synthesis.py")):
        relative_path = path.relative_to(VISUAL_GENERATION_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(_is_runtime_profile_synthesis_base(base) for base in node.bases):
                if relative_path not in ALLOWED_RUNTIME_PROFILE_SYNTHESIS_FILES:
                    offenders.append(f"{relative_path}:{node.name}")

    assert offenders == [], "unexpected RuntimeProfileSynthesis wrappers: " + ", ".join(offenders)


def _is_runtime_profile_synthesis_base(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "RuntimeProfileSynthesis"
    if isinstance(node, ast.Attribute):
        return node.attr == "RuntimeProfileSynthesis"
    return False
