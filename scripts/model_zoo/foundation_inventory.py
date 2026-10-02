"""Inventory helpers for reusable model-layer dedup audits.

Lists layer classes so a refactor can prove two DiTs share core nn
blocks instead of forking copies.

Not this module:
    The layers themselves live under :mod:`worldfoundry.core.nn` and
    :mod:`worldfoundry.core.attention`. This file only walks the tree
    and hashes sources.

Public surface:

- :class:`FoundationLayerInventoryEntry`
- :func:`iter_foundation_layer_files` / :func:`build_foundation_layer_inventory`
- :func:`file_duplicate_groups` / :func:`ast_duplicate_groups`
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

# ──────────────────────────────────────────────────────────────────────────
# Defaults — scopes, filename tokens, and directories the walker skips
# ──────────────────────────────────────────────────────────────────────────


DEFAULT_FOUNDATION_LAYER_SCOPES = (
    "worldfoundry/core",
    "worldfoundry/base_models",
    "worldfoundry/pipelines",
    "worldfoundry/synthesis",
    "worldfoundry/training",
)
DEFAULT_FOUNDATION_LAYER_TOKENS = (
    "attention",
    "attn",
    "rope",
    "prope",
    "transformer",
    "patch",
    "drop_path",
    "layer_scale",
    "rmsnorm",
    "rms_norm",
)
DEFAULT_SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
}
RUNTIME_DIR_MARKERS = {"runtime", "third_party", "thirdparty", "vendor"}


# ──────────────────────────────────────────────────────────────────────────
# Inventory entry — path, hashes, tokens, and re-export detection
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoundationLayerInventoryEntry:
    """One candidate file in the model-layer dedup inventory."""

    path: str
    owner_package: str
    size_bytes: int
    file_sha256: str
    ast_sha256: str | None
    ast_error: str | None
    filename_tokens: tuple[str, ...]
    imports: tuple[str, ...]
    public_symbols: tuple[str, ...]
    reexport_only: bool
    runtime_owned: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping for audit reports; tuples become lists via ``asdict``."""

        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Walk and build — filename-token match, then per-file AST metadata
# ──────────────────────────────────────────────────────────────────────────


def iter_foundation_layer_files(
    root: Path,
    *,
    scopes: Sequence[str] = DEFAULT_FOUNDATION_LAYER_SCOPES,
    tokens: Sequence[str] = DEFAULT_FOUNDATION_LAYER_TOKENS,
    include_runtime: bool = True,
    max_bytes: int | None = 2_000_000,
) -> tuple[Path, ...]:
    """Return Python files whose filenames match foundation-layer audit tokens."""

    root = Path(root)
    normalized_tokens = tuple(_normalize_token(token) for token in tokens if str(token).strip())
    files: list[Path] = []
    for scope in scopes:
        scope_path = root / scope
        if not scope_path.exists():
            continue
        for current_root, dir_names, file_names in os.walk(scope_path):
            dir_names[:] = [name for name in dir_names if not _should_skip_dir(name, include_runtime=include_runtime)]
            current = Path(current_root)
            for file_name in file_names:
                if not file_name.endswith(".py"):
                    continue
                stem = _normalize_token(Path(file_name).stem)
                matched = tuple(token for token in normalized_tokens if token in stem)
                if not matched:
                    continue
                path = current / file_name
                if path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if max_bytes is not None and size > int(max_bytes):
                    continue
                files.append(path)
    return tuple(sorted(files))


def build_foundation_layer_inventory(
    root: Path,
    *,
    scopes: Sequence[str] = DEFAULT_FOUNDATION_LAYER_SCOPES,
    tokens: Sequence[str] = DEFAULT_FOUNDATION_LAYER_TOKENS,
    include_runtime: bool = True,
    max_bytes: int | None = 2_000_000,
) -> tuple[FoundationLayerInventoryEntry, ...]:
    """Build a deterministic inventory for attention/RoPE/transformer dedup work."""

    root = Path(root)
    entries: list[FoundationLayerInventoryEntry] = []
    normalized_tokens = tuple(_normalize_token(token) for token in tokens if str(token).strip())
    for path in iter_foundation_layer_files(
        root,
        scopes=scopes,
        tokens=tokens,
        include_runtime=include_runtime,
        max_bytes=max_bytes,
    ):
        relative_path = _relative_path(path, root)
        try:
            content = path.read_bytes()
        except OSError:
            continue
        ast_hash, ast_error, imports, public_symbols, reexport_only = _ast_metadata(path, content)
        stem = _normalize_token(path.stem)
        entries.append(
            FoundationLayerInventoryEntry(
                path=relative_path,
                owner_package=_owner_package(relative_path),
                size_bytes=len(content),
                file_sha256=hashlib.sha256(content).hexdigest(),
                ast_sha256=ast_hash,
                ast_error=ast_error,
                filename_tokens=tuple(token for token in normalized_tokens if token in stem),
                imports=imports,
                public_symbols=public_symbols,
                reexport_only=reexport_only,
                runtime_owned=_is_runtime_owned(path),
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.path))


# ──────────────────────────────────────────────────────────────────────────
# Duplicate groups — identical bytes or identical parsed AST
# ──────────────────────────────────────────────────────────────────────────


def file_duplicate_groups(entries: Iterable[FoundationLayerInventoryEntry]) -> tuple[tuple[str, ...], ...]:
    """Group inventory entries with identical file bytes."""

    return _duplicate_groups(entries, key_name="file_sha256")


def ast_duplicate_groups(entries: Iterable[FoundationLayerInventoryEntry]) -> tuple[tuple[str, ...], ...]:
    """Group inventory entries with identical parsed ASTs."""

    return _duplicate_groups((entry for entry in entries if entry.ast_sha256), key_name="ast_sha256")


def _duplicate_groups(
    entries: Iterable[FoundationLayerInventoryEntry],
    *,
    key_name: str,
) -> tuple[tuple[str, ...], ...]:
    """Bucket paths by ``key_name``; skip re-export-only files and empty hashes."""

    grouped: dict[str, list[str]] = {}
    for entry in entries:
        if entry.reexport_only:
            continue
        value = getattr(entry, key_name)
        if not value:
            continue
        grouped.setdefault(str(value), []).append(entry.path)
    groups = tuple(tuple(sorted(paths)) for paths in grouped.values() if len(paths) > 1)
    return tuple(sorted(groups, key=lambda group: (-len(group), group[0])))


# ──────────────────────────────────────────────────────────────────────────
# AST helpers — hash, imports, public names, re-export-only detection
# ──────────────────────────────────────────────────────────────────────────


def _ast_metadata(
    path: Path,
    content: bytes,
) -> tuple[str | None, str | None, tuple[str, ...], tuple[str, ...], bool]:
    """Parse ``content``; on decode/syntax errors return ``ast_error`` and empty metadata."""

    try:
        source = content.decode("utf-8")
        tree = ast.parse(source, filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        return None, f"{type(exc).__name__}: {exc}", (), (), False
    ast_payload = ast.dump(tree, include_attributes=False).encode("utf-8")
    return (
        hashlib.sha256(ast_payload).hexdigest(),
        None,
        _imports_from_ast(tree),
        _public_symbols_from_ast(tree),
        _is_reexport_only_ast(tree),
    )


def _imports_from_ast(tree: ast.AST) -> tuple[str, ...]:
    """Collect absolute and relative import module names (not aliases)."""

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * int(node.level) + (node.module or "")
            imports.add(module)
    return tuple(sorted(imports))


def _public_symbols_from_ast(tree: ast.Module) -> tuple[str, ...]:
    """Top-level public classes, functions, and assigned names (no ``_`` prefix)."""

    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            else:
                targets.append(node.target)
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    symbols.add(target.id)
    return tuple(sorted(symbols))


def _is_reexport_only_ast(tree: ast.Module) -> bool:
    """True when the module is only imports plus a docstring / ``__all__``."""

    significant_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            continue
        significant_nodes.append(node)

    if not significant_nodes:
        return False
    return all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in significant_nodes)


# ──────────────────────────────────────────────────────────────────────────
# Path / token helpers — owner package, skip rules, runtime markers
# ──────────────────────────────────────────────────────────────────────────


def _owner_package(relative_path: str) -> str:
    """Derive a dotted owner from ``worldfoundry/...`` or ``src/worldfoundry/...``.

    Stops after the first package past the known layer roots so two files
    under the same model stay grouped without listing every submodule.
    """

    path = Path(relative_path)
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ("src", "worldfoundry"):
        package_parts = ["worldfoundry"]
        owner_parts = parts[2:-1]
    elif len(parts) >= 2 and parts[0] == "worldfoundry":
        package_parts = ["worldfoundry"]
        owner_parts = parts[1:-1]
    else:
        return ""
    for part in owner_parts:
        package_parts.append(part)
        if part in {"core", "base_models", "pipelines", "synthesis", "training", "evaluation"}:
            continue
        if len(package_parts) >= 4:
            break
    return ".".join(package_parts)


def _relative_path(path: Path, root: Path) -> str:
    """Prefer a root-relative path; fall back to the absolute string if not under ``root``."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _normalize_token(value: str) -> str:
    """Lowercase, strip, and treat hyphens as underscores for filename matching."""

    return str(value).strip().lower().replace("-", "_")


def _should_skip_dir(name: str, *, include_runtime: bool) -> bool:
    """Skip cache/venv dirs always; skip vendor/runtime dirs unless requested."""

    if name in DEFAULT_SKIP_DIR_NAMES:
        return True
    if include_runtime:
        return False
    lowered = name.lower()
    return lowered.endswith("_runtime") or lowered in RUNTIME_DIR_MARKERS


def _is_runtime_owned(path: Path) -> bool:
    """True if any path part is a vendor/runtime marker (dedup audits can exclude these)."""

    return any(part.lower().endswith("_runtime") or part.lower() in RUNTIME_DIR_MARKERS for part in path.parts)


__all__ = [
    "DEFAULT_FOUNDATION_LAYER_SCOPES",
    "DEFAULT_FOUNDATION_LAYER_TOKENS",
    "FoundationLayerInventoryEntry",
    "ast_duplicate_groups",
    "build_foundation_layer_inventory",
    "file_duplicate_groups",
    "iter_foundation_layer_files",
]
