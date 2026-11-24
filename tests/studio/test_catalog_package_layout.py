"""Structural contracts for the split Studio catalog package."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

CATALOG_ROOT = Path(__file__).resolve().parents[2] / "worldfoundry" / "studio" / "inference" / "catalog"
FAMILIES = ("video_generation", "world_models", "scene_3d", "embodied", "remote_api")
# A module may only import catalog modules on the same or a lower layer.
LAYER = {"entry": 0, "resolve": 0, "params": 1, "models": 2, "discovery": 3}


def test_family_overrides_do_not_share_model_ids():
    from worldfoundry.studio.inference.catalog import CURATED_OVERRIDES

    seen: dict[str, str] = {}
    for family in FAMILIES:
        module = importlib.import_module(f"worldfoundry.studio.inference.catalog.models.{family}")
        for model_id in module.OVERRIDES:
            assert model_id not in seen, f"{model_id} defined in both {seen[model_id]} and {family}"
            seen[model_id] = family
    assert set(seen) == set(CURATED_OVERRIDES)


def _layer(path: Path) -> int:
    rel = path.relative_to(CATALOG_ROOT)
    return LAYER[rel.parts[0] if len(rel.parts) > 1 else rel.stem]


@pytest.mark.parametrize(
    "path",
    [p for p in sorted(CATALOG_ROOT.rglob("*.py")) if p.name != "__init__.py" or p.parent.name == "models"],
    ids=lambda p: str(p.relative_to(CATALOG_ROOT)),
)
def test_catalog_modules_only_import_lower_layers(path: Path):
    own = _layer(path)
    package = "models" if path.parent.name == "models" else ""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        if node.level == 1 and package:
            target = "models"
        else:
            target = (node.module or "").split(".")[0] or node.names[0].name
        assert LAYER[target] <= own, f"{path.name} imports higher layer {target}"
