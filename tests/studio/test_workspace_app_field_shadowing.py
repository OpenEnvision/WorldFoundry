"""Regression tests for SA-5 (F402): the name ``field`` must not be rebound.

``workspace_app`` imports ``dataclasses.field`` at module level and uses it in
dataclass definitions.  Loop variables named ``field`` inside functions used to
shadow that import, which is a latent trap for anyone adding a ``field(...)``
call inside those functions.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

WORKSPACE_APP_PATH = (
    Path(__file__).resolve().parents[2] / "worldfoundry" / "studio" / "serving" / "workspace.py"
)


def test_no_python_binding_shadows_dataclasses_field() -> None:
    tree = ast.parse(WORKSPACE_APP_PATH.read_text())
    offenders: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "field":
            offenders.append(node.lineno)
        if isinstance(node, ast.arg) and node.arg == "field":
            offenders.append(node.lineno)
    assert offenders == [], (
        f"workspace_app.py binds the name 'field' at lines {sorted(offenders)}; "
        "this shadows dataclasses.field (F402/SA-5). Rename the binding "
        "(e.g. input_field)."
    )


def test_module_level_field_is_dataclasses_field() -> None:
    pytest.importorskip("fastapi")
    workspace_app = pytest.importorskip("worldfoundry.studio.serving.workspace")
    assert workspace_app.field is dataclasses.field
