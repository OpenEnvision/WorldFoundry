from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "worldfoundry"

LAYER_ORDER = {
    "core": 0,
    "runtime": 1,
    "base_models": 2,
    "pipelines": 3,
    "synthesis": 4,
    "training": 4,
    "evaluation": 4,
    "studio": 4,
    "operators": 4,
    "memories": 4,
    "representations": 4,
    "cli": 5,
    "mcp": 5,
}

# TODO(infra): burn these down after runtime helpers no longer depend on
# evaluation.utils compatibility helpers.
KNOWN_VIOLATIONS = frozenset(
    {
        ("worldfoundry/runtime/assets.py", "worldfoundry.evaluation.utils"),
        ("worldfoundry/runtime/conda.py", "worldfoundry.evaluation.utils"),
    }
)


def test_core_and_runtime_do_not_add_upward_imports() -> None:
    violations = set()
    for package in ("core", "runtime"):
        for path in sorted((SRC_ROOT / package).rglob("*.py")):
            for module in _runtime_imports(path):
                if _is_upward_import(package, module):
                    violations.add((path.relative_to(REPO_ROOT).as_posix(), module))

    assert sorted(violations - KNOWN_VIOLATIONS) == []


def _runtime_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    modules: list[str] = []
    current_module = _module_name(path)
    for node in ast.walk(tree):
        if _inside_type_checking(node, parents):
            continue
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(current_module, node)
            if resolved:
                modules.append(resolved)
    return tuple(modules)


def _inside_type_checking(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if not isinstance(current, ast.If):
            continue
        test = current.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            return True
    return False


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_from(current_module: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = current_module.split(".")
    if package_parts[-1] != "__init__":
        package_parts = package_parts[:-1]
    if node.level > 1:
        package_parts = package_parts[: -(node.level - 1)]
    if node.module:
        package_parts = [*package_parts, *node.module.split(".")]
    return ".".join(package_parts) if package_parts else None


def _is_upward_import(current_layer: str, module: str) -> bool:
    if module == "worldfoundry":
        return False
    if not module.startswith("worldfoundry."):
        return False
    parts = module.split(".")
    if len(parts) < 2:
        return False
    imported_layer = parts[1]
    if imported_layer not in LAYER_ORDER:
        return False
    return LAYER_ORDER[imported_layer] > LAYER_ORDER[current_layer]
