# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free guards for canonical diffusion dependency direction."""

from __future__ import annotations

import ast
import re
import tokenize
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_APPLICATION_PACKAGES = (
    "worldfoundry.synthesis",
    "worldfoundry.pipelines",
    "worldfoundry.studio",
    "worldfoundry.evaluation",
    "worldfoundry.operators",
)
_MODULE_TARGET = re.compile(r"\.*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*)?")


def _module_references(source: str, package: str) -> Iterator[tuple[int, str]]:
    """Inspect ordinary imports, lazy imports, and literal registration targets."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            target = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            yield node.lineno, target
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{target}.{alias.name}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.startswith(("worldfoundry.", ".")) and _MODULE_TARGET.fullmatch(value):
                target = value.split(":", 1)[0]
                yield node.lineno, resolve_name(target, package) if target.startswith(".") else target


def test_canonical_diffusion_does_not_depend_on_applications() -> None:
    """Model components and framework runners must be usable without application layers."""
    root = Path(__file__).resolve().parents[3]
    violations = []
    for path in (root / "worldfoundry/base_models/diffusion_model").rglob("*.py"):
        package = ".".join(path.relative_to(root).parts[:-1])
        with tokenize.open(path) as stream:
            references = _module_references(stream.read(), package)
            violations.extend(
                (str(path.relative_to(root)), line, target)
                for line, target in references
                if any(target == boundary or target.startswith(boundary + ".") for boundary in _APPLICATION_PACKAGES)
            )
    assert not violations, f"canonical diffusion depends on applications: {violations}"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("import worldfoundry.synthesis.visual_generation", "worldfoundry.synthesis.visual_generation"),
        ("from worldfoundry import studio", "worldfoundry.studio"),
        ("from ... import synthesis", "worldfoundry.synthesis"),
        (
            "def load():\n    from worldfoundry.studio.visualization import render",
            "worldfoundry.studio.visualization",
        ),
        (
            'import_module("worldfoundry.synthesis.visual_generation.sana.sana_synthesis")',
            "worldfoundry.synthesis.visual_generation.sana.sana_synthesis",
        ),
        ('targets = {"model": "worldfoundry.pipelines.demo:Pipeline"}', "worldfoundry.pipelines.demo"),
        ('entries = (("model", "...synthesis.demo", "recipe"),)', "worldfoundry.synthesis.demo"),
    ],
)
def test_boundary_guard_covers_import_and_registration_forms(source: str, expected: str) -> None:
    targets = {target for _, target in _module_references(source, "worldfoundry.base_models.diffusion_model")}
    assert expected in targets


def test_boundary_guard_distinguishes_documentation_from_module_targets() -> None:
    source = '"""Callers in worldfoundry.synthesis consume native components."""\nimport torch'
    assert list(_module_references(source, "worldfoundry.base_models.diffusion_model")) == [(2, "torch")]
