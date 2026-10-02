"""Integrity primitives for local evidence and artifacts.

Atomic replace, canonical JSON hash, and path-escape checks. Use these
for eval ledgers; ``serialization.py`` is convenience I/O and may
leave a truncated file on crash.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from uuid import uuid4

# ──────────────────────────────────────────────────────────────────────────
# Canonical JSON — allow_nan=False so hashes stay IEEE-safe and portable
# ──────────────────────────────────────────────────────────────────────────


def canonical_json(value: object) -> str:
    """Serialize strict JSON deterministically for hashing and evidence files."""

    return json.dumps(
        value,
        allow_nan=False,  # NaN / Inf are not portable JSON and would break hashes
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 digest of a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_path_within(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
    *,
    field_name: str = "path",
) -> Path:
    """Resolve a path and reject targets outside the resolved root."""

    resolved_root = Path(root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{field_name} must stay within {resolved_root}; got {resolved}") from error
    return resolved


def safe_relative_path(value: str | os.PathLike[str], *, field_name: str = "path") -> Path:
    """Validate a non-empty relative path without parent traversal."""

    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path in {Path("."), Path("")}:
        raise ValueError(f"{field_name} must be a safe relative path: {os.fspath(value)!r}")
    return path


def sync_directory(path: str | os.PathLike[str]) -> None:
    """Persist a directory entry when the platform supports directory fsync."""

    if os.name != "posix":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# ──────────────────────────────────────────────────────────────────────────
# Exclusive / atomic replace — eval ledgers; crash must not leave a truncated file
# ──────────────────────────────────────────────────────────────────────────


def write_exclusive_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Create a confined file exclusively and flush its contents to storage."""

    destination = ensure_path_within(path, root, field_name="write destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def write_exclusive_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Create a confined UTF-8 text file exclusively and durably."""

    return write_exclusive_bytes(path, text.encode("utf-8"), root=root)


def write_exclusive_json(
    path: str | os.PathLike[str],
    value: object,
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Create a confined canonical JSON file exclusively and durably."""

    return write_exclusive_text(path, f"{canonical_json(value)}\n", root=root)


def replace_json_atomic(
    path: str | os.PathLike[str],
    value: object,
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Atomically replace a confined canonical JSON file and persist the rename."""

    destination = ensure_path_within(path, root, field_name="JSON destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        write_exclusive_json(temporary, value, root=root)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def replace_jsonl_atomic(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, object]],
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Atomically replace a confined JSONL file using strict canonical rows."""

    destination = ensure_path_within(path, root, field_name="JSONL destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def append_jsonl_durable(
    path: str | os.PathLike[str],
    row: Mapping[str, object],
    *,
    root: str | os.PathLike[str],
) -> Path:
    """Append one strict canonical JSONL row and flush it to storage."""

    destination = ensure_path_within(path, root, field_name="JSONL destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


__all__ = [
    "append_jsonl_durable",
    "canonical_json",
    "canonical_sha256",
    "ensure_path_within",
    "replace_json_atomic",
    "replace_jsonl_atomic",
    "safe_relative_path",
    "sync_directory",
    "text_sha256",
    "write_exclusive_bytes",
    "write_exclusive_json",
    "write_exclusive_text",
]
