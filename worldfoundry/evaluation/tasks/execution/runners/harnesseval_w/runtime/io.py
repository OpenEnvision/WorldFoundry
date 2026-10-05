"""Small, auditable filesystem primitives used by every command."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = file_digest(path)
    return result


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def exclusive_lock(
    path: Path, owner: dict[str, Any], *, wait: bool = False
) -> Iterator[None]:
    """Acquire a process lock and leave useful owner metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_writable = True
    try:
        handle = path.open("a+", encoding="utf-8")
    except PermissionError:
        if not path.is_file():
            raise
        # Shared runs can leave a root-owned lock file behind. Lock the same
        # inode read-only so mutual exclusion is preserved across users.
        handle = path.open("r", encoding="utf-8")
        metadata_writable = False
    with handle:
        try:
            flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            handle.seek(0)
            current = handle.read().strip()
            raise RuntimeError(f"another process owns {path}: {current}") from exc
        if metadata_writable:
            handle.seek(0)
            handle.truncate()
            handle.write(canonical_json({"pid": os.getpid(), **owner}) + "\n")
            handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
