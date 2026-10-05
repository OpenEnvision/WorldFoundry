"""Checked-in YAML manifest loading.

Single canonical implementation shared by the runtime and evaluation layers.
This lives in ``worldfoundry.core`` so lower layers (``worldfoundry.runtime``)
never have to import ``worldfoundry.evaluation`` for manifest IO (SA-10);
``worldfoundry.evaluation.utils`` re-exports these names as its public
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MANIFEST_SUFFIXES = (".yaml", ".yml")

# ──────────────────────────────────────────────────────────────────────────
# Canonical YAML loaders — runtime must not import evaluation for this
# ──────────────────────────────────────────────────────────────────────────


def load_manifest(path: str | Path) -> Any:
    """Load a checked-in YAML WorldFoundry manifest.

    Raises:
        ValueError: If the path suffix is not ``.yaml``/``.yml``.
        yaml.YAMLError: On parse failure, re-raised with the file path in the
            message so callers can locate the broken manifest.
    """

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix not in MANIFEST_SUFFIXES:
        raise ValueError(f"unsupported manifest suffix for {resolved}: expected .yaml or .yml")
    text = resolved.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"failed to parse manifest {resolved}: {exc}") from exc


def manifest_paths(root: str | Path) -> tuple[Path, ...]:
    """Return sorted ``.yaml`` / ``.yml`` paths under *root* (recursive).

    Raises:
        FileNotFoundError: *root* does not exist.
        NotADirectoryError: *root* is a file.
    """

    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"manifest directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"manifest path is not a directory: {path}")

    return tuple(sorted(candidate for suffix in MANIFEST_SUFFIXES for candidate in path.rglob(f"*{suffix}")))


def load_manifest_collection(root: str | Path, *, item_key: str) -> dict[str, Any]:
    """Load a manifest file or a directory of one-item YAML manifests.

    A file is returned as-is (or wrapped as ``{item_key: payload}``).
    A directory merges ``_manifest.yaml`` metadata (if present) with
    every other YAML file's *item_key* list. Non-dict entries are
    appended to that list.

    Args:
        root: File or directory of checked-in manifests.
        item_key: Collection key (for example ``"examples"``).
    """

    path = Path(root)
    if path.is_file():
        payload = load_manifest(path)
        return payload if isinstance(payload, dict) else {item_key: payload}
    if not path.is_dir():
        raise FileNotFoundError(f"manifest path does not exist: {path}")

    meta_path = path / "_manifest.yaml"
    payload: dict[str, Any] = {}
    if meta_path.is_file():
        meta = load_manifest(meta_path)
        if isinstance(meta, dict):
            payload.update(meta)

    items: list[Any] = []
    for manifest_path in manifest_paths(path):
        if manifest_path.name == "_manifest.yaml":
            continue
        entry = load_manifest(manifest_path)
        if isinstance(entry, dict) and item_key in entry:
            values = entry[item_key]
            if isinstance(values, list):
                items.extend(values)
            elif values is not None:
                items.append(values)
        elif isinstance(entry, list):
            items.extend(entry)
        elif entry is not None:
            items.append(entry)

    payload[item_key] = items
    return payload


__all__ = [
    "MANIFEST_SUFFIXES",
    "load_manifest",
    "load_manifest_collection",
    "manifest_paths",
]
