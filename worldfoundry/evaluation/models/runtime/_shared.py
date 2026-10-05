"""Shared coercion and YAML-scanning helpers for runtime profile loaders.

profiles.py / environments.py / assets.py previously carried verbatim copies
of these helpers; they are consolidated here so schema rules and error
formatting stay in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any scalar value or sequence into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def schema_version_or_none(value: Any) -> int | None:
    """Coerce any schema version value to int or return ``None``."""
    if value in (None, ""):
        return None
    return int(value)


def yaml_manifest_paths(root: str | Path | None, *, default_root: Path) -> tuple[Path, ...]:
    """Retrieve all YAML manifest paths found under a root directory recursively."""
    path = Path(root) if root is not None else default_root
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*.y*ml") if item.is_file()))


def iter_manifest_mappings(
    path: Path,
    *,
    collection_keys: Sequence[str],
    id_keys: Sequence[str],
    kind: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load a YAML manifest and return its entry mappings.

    The payload may either be a collection (first truthy value among
    ``collection_keys``) or a single entry identified by one of ``id_keys``.
    Parse errors are re-raised with the file path in the message.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"failed to parse {kind} file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{kind} file must contain a mapping: {path}")
    # Mirrors the original `payload.get(a) or payload.get(b)` chain: the first
    # truthy collection wins; if none is truthy the last lookup result is kept
    # (an explicit empty list therefore does not fall back to single-entry mode).
    entries: Any = None
    for key in collection_keys:
        entries = payload.get(key)
        if entries:
            break
    if entries is None:
        return (payload,) if any(payload.get(key) for key in id_keys) else ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise TypeError(f"{kind} collection must be a list: {path}")
    return tuple(item for item in entries if isinstance(item, Mapping))


__all__ = [
    "iter_manifest_mappings",
    "schema_version_or_none",
    "tuple_of_str",
    "yaml_manifest_paths",
]
