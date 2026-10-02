from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_WORLD_DIR = (
    REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world"
)
SYNTHESIS_SOURCE = (
    REPO_ROOT
    / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay_synthesis.py"
)
RUNTIME_SOURCE = HUNYUAN_WORLD_DIR / "hunyuan_worldplay/runtime.py"
COMMONS_INIT = HUNYUAN_WORLD_DIR / "hunyuan_worldplay/commons/__init__.py"


def _import_from_modules(source_path: Path) -> set[tuple[int, str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return {
        (node.level, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def _module_source_path(module_name: str) -> Path:
    parts = module_name.split(".")
    assert parts[0] == "worldfoundry"

    module_path = REPO_ROOT / Path(*parts)
    module_file = module_path.with_suffix(".py")
    package_file = module_path / "__init__.py"

    if module_file.exists():
        return module_file
    return package_file


def test_hunyuan_worldplay_lazy_imports_resolve_inside_worldplay_tree():
    runtime_modules = _import_from_modules(RUNTIME_SOURCE)
    synthesis_modules = _import_from_modules(SYNTHESIS_SOURCE)
    expected_modules = {
        (
            "worldfoundry.synthesis.visual_generation.hunyuan_world."
            "hunyuan_worldplay.utils.rewrite.rewrite_utils"
        ),
        (
            "worldfoundry.synthesis.visual_generation.hunyuan_world."
            "hunyuan_worldplay.pipelines.hunyuan_video_sr_pipeline"
        ),
    }

    for module_name in expected_modules:
        assert _module_source_path(module_name).exists()
        assert (0, module_name) in runtime_modules

    assert (1, "utils.rewrite.rewrite_utils") not in runtime_modules
    assert (1, "hunyuan_video_sr_pipeline") not in runtime_modules
    assert (0, "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.runtime") in synthesis_modules
    assert "DiffusionPipeline" not in SYNTHESIS_SOURCE.read_text(encoding="utf-8")


def test_hunyuan_worldplay_commons_uses_local_sparse_attention_helper():
    source = COMMONS_INIT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "def is_sparse_attn_available(" in source
    assert "from .commons import is_sparse_attn_available" not in source
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "commons"
        and any(alias.name == "is_sparse_attn_available" for alias in node.names)
        for node in ast.walk(tree)
    )
