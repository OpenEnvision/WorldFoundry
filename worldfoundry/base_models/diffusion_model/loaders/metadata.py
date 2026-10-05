"""Framework-owned checkpoint metadata readers.

Recipes use these helpers as ``ModuleLoadSpec.config_resolver`` callbacks so
architecture JSON stays with the checkpoint, not in process-wide globals.  The
runner never reads config files; only the loader / factory does.

Both helpers reject path traversal (absolute or ``..``) and require a JSON
object.  They do not mutate the materialized checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .materialize import MaterializedCheckpoint


def checkpoint_json_config(
    checkpoint: MaterializedCheckpoint,
    relative_path: str | Path = "config.json",
) -> Mapping[str, object]:
    """Read component JSON from either a snapshot root or direct subtree override.

    Tries ``root / relative_path`` then ``root / relative_path.name`` so a
    flattened local directory override still works.

    Raises:
        ValueError: ``relative_path`` is absolute or contains ``..``.
        FileNotFoundError: Neither candidate exists.
        TypeError: The file does not contain a JSON object.
    """

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint config paths must be relative and cannot contain '..'")
    candidates = (checkpoint.root / relative, checkpoint.root / relative.name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"checkpoint JSON config does not exist; tried {[str(candidate) for candidate in candidates]}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint JSON config must contain an object: {path}")
    return value


def safetensors_json_metadata(
    checkpoint: MaterializedCheckpoint,
    *,
    key: str = "config",
) -> Mapping[str, object]:
    """Read one JSON object from the first materialized safetensors file.

    Raises:
        ValueError: No materialized paths, or the first path is not
            ``.safetensors``.
        RuntimeError: Optional ``safetensors`` package is missing.
        KeyError: Header metadata does not contain ``key``.
        TypeError: Decoded ``key`` is not a JSON object.
    """

    if not checkpoint.paths:
        raise ValueError("checkpoint has no materialized files")
    path = checkpoint.paths[0]
    if path.suffix != ".safetensors":
        raise ValueError(f"checkpoint metadata requires a safetensors file: {path}")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "reading safetensors metadata requires the optional 'safetensors' package"
        ) from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    try:
        raw = metadata[key]
    except KeyError as error:
        raise KeyError(f"safetensors metadata does not contain {key!r}: {path}") from error
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"safetensors metadata {key!r} must contain a JSON object")
    return value


__all__ = ["checkpoint_json_config", "safetensors_json_metadata"]
