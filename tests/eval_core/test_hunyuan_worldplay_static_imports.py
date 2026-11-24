from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_WORLD_DIR = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world"
COMMONS_INIT = HUNYUAN_WORLD_DIR / "hunyuan_worldplay/commons/__init__.py"


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
