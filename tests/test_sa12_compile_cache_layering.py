"""SA-12: compile cache lives in core; runtime only re-exports."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "worldfoundry" / "core"
RUNTIME_COMPILE_CACHE = REPO_ROOT / "worldfoundry" / "runtime" / "compile_cache.py"
CORE_COMPILE_CACHE = REPO_ROOT / "worldfoundry" / "core" / "compile_cache.py"

PUBLIC_NAMES = frozenset(
    {
        "CompileCacheLayout",
        "CompilePolicy",
        "compile_callable_cached",
        "compile_module_cached",
        "configure_persistent_compile_cache",
    }
)
FORBIDDEN_RUNTIME_COMPILE_CACHE = "worldfoundry.runtime.compile_cache"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            if node.module == "worldfoundry.runtime":
                for alias in node.names:
                    modules.add(f"worldfoundry.runtime.{alias.name}")
    return modules


def test_core_modules_do_not_import_runtime_compile_cache() -> None:
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        imported = _imported_modules(path)
        if FORBIDDEN_RUNTIME_COMPILE_CACHE in imported or any(
            name == FORBIDDEN_RUNTIME_COMPILE_CACHE or name.startswith(f"{FORBIDDEN_RUNTIME_COMPILE_CACHE}.")
            for name in imported
        ):
            violations.append(path.relative_to(REPO_ROOT).as_posix())
    assert violations == [], violations


def test_core_compile_cache_does_not_import_runtime_env() -> None:
    imported = _imported_modules(CORE_COMPILE_CACHE)
    assert "worldfoundry.runtime.env" not in imported
    assert not any(name == "worldfoundry.runtime" or name.startswith("worldfoundry.runtime.") for name in imported)
    assert "worldfoundry.core.io.paths" in imported


def test_runtime_compile_cache_reexports_public_names() -> None:
    tree = ast.parse(RUNTIME_COMPILE_CACHE.read_text(encoding="utf-8"), filename=str(RUNTIME_COMPILE_CACHE))
    reexported: set[str] = set()
    imported_from_core = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "worldfoundry.core.compile_cache":
            imported_from_core = True
            reexported.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("worldfoundry.runtime."):
            raise AssertionError(f"runtime compile_cache still imports {node.module}")
    assert imported_from_core
    assert PUBLIC_NAMES <= reexported or reexported == {"*"}
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assert not (defined & PUBLIC_NAMES)

    core_mod = importlib.import_module("worldfoundry.core.compile_cache")
    runtime_mod = importlib.import_module("worldfoundry.runtime.compile_cache")
    assert PUBLIC_NAMES <= set(core_mod.__all__)
    assert PUBLIC_NAMES <= set(runtime_mod.__all__)
    for name in PUBLIC_NAMES:
        assert getattr(runtime_mod, name) is getattr(core_mod, name)
