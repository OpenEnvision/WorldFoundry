"""XC-19 / XC-20 guard: first-party library code must not mutate the root logger.

Scan ``worldfoundry/{core,cli,mcp,operators,training,evaluation/api,studio}``
plus ``evaluation/tasks/execution/framework`` (the in-tree ``evaluation/framework``).
Skip vendor trees and ``evaluation/tasks/execution/runners/*/runtime``.

Allowlisted files are process entrypoints that are *supposed* to configure
logging, or trees this tick is forbidden to edit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOTS = (
    REPO_ROOT / "worldfoundry/core",
    REPO_ROOT / "worldfoundry/cli",
    REPO_ROOT / "worldfoundry/mcp",
    REPO_ROOT / "worldfoundry/operators",
    REPO_ROOT / "worldfoundry/training",
    REPO_ROOT / "worldfoundry/evaluation/api",
    REPO_ROOT / "worldfoundry/evaluation/tasks/execution/framework",
    REPO_ROOT / "worldfoundry/studio",
)

# WONTFIX: these either *are* the process logging entrypoint, or sit in a
# tree this tick must not edit (core/distributed).
ALLOWLIST = {
    "worldfoundry/core/observability/logging_setup.py": (
        "WONTFIX: configure_logging is the CLI/process entry that must "
        "install root handlers"
    ),
    "worldfoundry/core/distributed/sequence_parallel/logger.py": (
        "dictConfig remains for the trainer logger only; root key is stripped"
    ),
}

FIRST_ROUND_CLEAN = (
    "worldfoundry/base_models/diffusion_model/models/networks/wan/media_geometry.py",
    "worldfoundry/evaluation/tasks/metrics/jedi/V_JEPA.py",
    "worldfoundry/evaluation/tasks/execution/runners/memobench/runtime/memobench/evaluation/run_eval.py",
)

_ROOT_MUTATOR_ATTRS = frozenset({"setLevel", "addHandler", "removeHandler", "handlers"})
_LOGGING_FUNCS = frozenset({"basicConfig", "disable", "getLogger", "root", "dictConfig"})


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_skipped(path: Path) -> bool:
    parts = path.parts
    if "vendor" in parts or "__pycache__" in parts:
        return True
    if "runners" in parts:
        idx = parts.index("runners")
        if idx + 2 < len(parts) and parts[idx + 2] == "runtime":
            return True
    return False


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        files.extend(path for path in root.rglob("*.py") if not _is_skipped(path))
    return sorted(files)


def _is_empty_get_logger(call: ast.Call) -> bool:
    if call.args:
        return False
    return not any(keyword.arg == "name" for keyword in call.keywords)


def _attr_chain(node: ast.AST) -> list[str] | None:
    names: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.append(current.id)
        names.reverse()
        return names
    return None


class _RootMutationVisitor(ast.NodeVisitor):
    """Flag basicConfig / logging.disable / root-handler mutation / dictConfig."""

    def __init__(self) -> None:
        self.logging_aliases: set[str] = set()
        self.imported: dict[str, str] = {}
        self.root_vars: set[str] = set()
        self.hits: list[tuple[int, str]] = []

    def _hit(self, node: ast.AST, reason: str) -> None:
        self.hits.append((getattr(node, "lineno", 0), reason))

    def _is_logging_attr(self, node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == name
            and isinstance(node.value, ast.Name)
            and node.value.id in self.logging_aliases
        )

    def _is_root_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and (
            node.id in self.root_vars or self.imported.get(node.id) == "root"
        ):
            return True
        if self._is_logging_attr(node, "root"):
            return True
        if isinstance(node, ast.Call) and self._is_get_logger(node.func) and _is_empty_get_logger(node):
            return True
        return False

    def _is_get_logger(self, func: ast.AST) -> bool:
        if isinstance(func, ast.Name) and self.imported.get(func.id) == "getLogger":
            return True
        return self._is_logging_attr(func, "getLogger")

    def _is_dict_config(self, func: ast.AST) -> bool:
        if isinstance(func, ast.Name) and self.imported.get(func.id) == "dictConfig":
            return True
        chain = _attr_chain(func)
        if not chain or chain[-1] != "dictConfig":
            return False
        if len(chain) == 2 and chain[0] in self.logging_aliases:
            return True
        return len(chain) == 3 and chain[0] in self.logging_aliases and chain[1] == "config"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "logging":
                self.logging_aliases.add(alias.asname or "logging")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "logging":
            for alias in node.names:
                dest = alias.asname or alias.name
                if alias.name in _LOGGING_FUNCS:
                    self.imported[dest] = alias.name
                if alias.name == "root":
                    self.root_vars.add(dest)
        elif module == "logging.config":
            for alias in node.names:
                if alias.name == "dictConfig":
                    self.imported[alias.asname or "dictConfig"] = "dictConfig"
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_root_expr(node.value) or (
            isinstance(node.value, ast.Call)
            and self._is_get_logger(node.value.func)
            and _is_empty_get_logger(node.value)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.root_vars.add(target.id)
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "handlers"
                and self._is_root_expr(target.value)
            ):
                self._hit(node, "assign logging.root.handlers")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and (
            self._is_root_expr(node.value)
            or (
                isinstance(node.value, ast.Call)
                and self._is_get_logger(node.value.func)
                and _is_empty_get_logger(node.value)
            )
        ):
            if isinstance(node.target, ast.Name):
                self.root_vars.add(node.target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and self.imported.get(func.id) in {"basicConfig", "disable", "dictConfig"}:
            self._hit(node, self.imported[func.id])
        elif self._is_logging_attr(func, "basicConfig"):
            self._hit(node, "logging.basicConfig")
        elif self._is_logging_attr(func, "disable"):
            self._hit(node, "logging.disable")
        elif self._is_dict_config(func):
            self._hit(node, "logging.dictConfig")
        elif isinstance(func, ast.Attribute) and func.attr in _ROOT_MUTATOR_ATTRS:
            if self._is_root_expr(func.value):
                self._hit(node, f"root.{func.attr}(...)")
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "handlers"
            and self._is_root_expr(func.value.value)
        ):
            self._hit(node, f"root.handlers.{func.attr}(...)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # ``root.handlers.clear()`` is a Call on Attribute; handled in visit_Call.
        # Also catch ``root.handlers.append`` via Call. Nothing else here.
        self.generic_visit(node)


def _scan_source(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    visitor = _RootMutationVisitor()
    visitor.visit(tree)
    return visitor.hits


def _scan_file(path: Path) -> list[tuple[int, str]]:
    return _scan_source(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("relpath", FIRST_ROUND_CLEAN)
def test_xc19_first_round_files_no_longer_mutate_root(relpath: str) -> None:
    path = REPO_ROOT / relpath
    assert path.is_file(), relpath
    hits = _scan_file(path)
    assert hits == [], f"{relpath} still mutates root logger: {hits}"


def test_xc19_allowlisted_wontfix_files_exist() -> None:
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).is_file()]
    assert missing == []


def test_xc19_first_party_scan_has_no_unallowlisted_root_mutation() -> None:
    violations: list[str] = []
    scanned = 0
    for path in _iter_scan_files():
        scanned += 1
        rel = _rel(path)
        hits = _scan_file(path)
        if not hits:
            continue
        if rel in ALLOWLIST:
            continue
        formatted = ", ".join(f"L{lineno} {reason}" for lineno, reason in hits)
        violations.append(f"{rel}: {formatted}")
    assert scanned > 0
    assert violations == [], "first-party root-logger mutations:\n" + "\n".join(violations)
