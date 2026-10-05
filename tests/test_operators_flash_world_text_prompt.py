"""OR-10: flash_world text_prompt pass-through (no GPU / no plyfile import)."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional

import pytest

OPERATORS_DIR = Path(__file__).resolve().parents[1] / "worldfoundry" / "operators"
FLASH_WORLD_SRC = OPERATORS_DIR / "flash_world_operator.py"


def _load_flash_world_text_helpers():
    source = FLASH_WORLD_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_text_from_flash_world_interaction",
        "_split_flash_world_interactions",
    }
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunk = ast.get_source_segment(source, node)
            assert chunk, f"could not extract {node.name}"
            chunks.append(chunk)
    ns: dict[str, Any] = {
        "Any": Any,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
    }
    exec("\n\n".join(chunks), ns)
    missing = wanted - ns.keys()
    assert not missing, f"helpers missing from source: {missing}"
    return ns


@pytest.fixture(scope="module")
def helpers():
    return _load_flash_world_text_helpers()


def test_text_from_mapping_and_prompt_alias(helpers):
    fn = helpers["_text_from_flash_world_interaction"]
    assert fn({"text_prompt": "a cafe"}) == "a cafe"
    assert fn({"prompt": "a hall"}) == "a hall"
    assert fn({"text_prompt": "wins", "prompt": "ignored"}) == "wins"
    assert fn("text_prompt") is None
    assert fn("forward") is None
    assert fn({"cameras": []}) is None


def test_split_keeps_camera_order_and_drops_text_token(helpers):
    split = helpers["_split_flash_world_interactions"]
    text, cameras = split(["forward", "text_prompt", "left"])
    assert text == ""
    assert cameras == ["forward", "left"]


def test_split_collects_mapping_text(helpers):
    split = helpers["_split_flash_world_interactions"]
    text, cameras = split(
        ["forward", {"text_prompt": "a sunny room"}, "camera_l"]
    )
    assert text == "a sunny room"
    assert cameras == ["forward", "camera_l"]


def test_split_kwargs_override_collected_text(helpers):
    split = helpers["_split_flash_world_interactions"]
    text, cameras = split(
        [{"prompt": "from list"}],
        text_prompt="from kwarg",
    )
    assert text == "from kwarg"
    assert cameras == []
    text, cameras = split(["forward"], prompt="via prompt alias")
    assert text == "via prompt alias"
    assert cameras == ["forward"]


def test_split_joins_multiple_text_mappings(helpers):
    split = helpers["_split_flash_world_interactions"]
    text, cameras = split(
        [{"text_prompt": "red"}, "backward", {"prompt": "barn"}]
    )
    assert text == "red barn"
    assert cameras == ["backward"]


def test_empty_mapping_text_is_not_a_camera_action(helpers):
    split = helpers["_split_flash_world_interactions"]
    text, cameras = split(["forward", {"text_prompt": ""}])
    assert text == ""
    assert cameras == ["forward"]


def test_process_interaction_uses_splitter_not_hardcoded_empty():
    """process_interaction must call the splitter instead of `text_prompt = ""`."""
    source = FLASH_WORLD_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FlashWorldOperator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "process_interaction":
                    found = item
    assert found is not None
    calls = [
        n.func.id
        for n in ast.walk(found)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_split_flash_world_interactions" in calls
