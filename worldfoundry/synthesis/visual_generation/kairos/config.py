# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Read Kairos recipe assignments as data, without importing configuration files."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def load_kairos_config(path: str | Path, *, models_root: str | None = None) -> dict[str, Any]:
    """Load JSON or literal Python assignments with name references and plain f-strings.

    ``models_root`` replaces the shipped ``KAIROS_MODEL_DIR = "models"`` placeholder.
    Python imports, calls and executable statements are not configuration data.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(source)
    values: dict[str, Any] = {}

    def value(node: ast.AST | None) -> Any:
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.List):
            return [value(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(value(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {value(key): value(item) for key, item in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.JoinedStr):
            return "".join(str(value(item)) for item in node.values)
        if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
            return value(node.value)
        if node is None:
            raise ValueError(f"{path}: dictionary unpacking is not a Kairos config value")
        return ast.literal_eval(node)

    for statement in ast.parse(source, filename=str(path)).body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if not isinstance(statement, ast.Assign) or not all(
            isinstance(target, ast.Name) for target in statement.targets
        ):
            raise ValueError(f"{path}:{statement.lineno}: Kairos configs must contain data assignments only")
        resolved = value(statement.value)
        for target in statement.targets:
            assert isinstance(target, ast.Name)
            if target.id == "KAIROS_MODEL_DIR" and resolved == "models" and models_root is not None:
                resolved = models_root
            values[target.id] = resolved
    return values
