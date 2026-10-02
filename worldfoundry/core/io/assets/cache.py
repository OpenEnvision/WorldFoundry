"""Rank-safe local caching for local and remote inference assets.

Rank 0 downloads under a lock; others wait. S3 and HTTP share this
path so a multi-GPU job does not stampede the object store. Hugging
Face snapshots still go through ``hf.py``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from worldfoundry.core.io.paths import cache_root_path
from worldfoundry.core.io.formats.serialization import load_serialized
from worldfoundry.core.io.filesystem.storage import copy_uri, parse_uri_scheme, uri_to_local_path

# ──────────────────────────────────────────────────────────────────────────
# Cache slot — S3/HTTP → local path; flock + rank-0 so workers do not stampede
# ──────────────────────────────────────────────────────────────────────────


def _storage_options(backend_args: dict[str, Any] | None) -> dict[str, Any]:
    """Strip WorldFoundry-only keys before passing options to :func:`copy_uri`."""

    options = dict(backend_args or {})
    for key in ("backend", "path_mapping", "s3_credential_path"):
        options.pop(key, None)
    return options


def _cache_path(source_path, cache_fp=None, cache_dir=None) -> Path:
    """Map a URI to a cache file; ``://`` becomes ``/`` so S3 keys stay unique."""

    if cache_dir is None:
        cache_dir = cache_root_path()
    root = Path(os.path.expanduser(str(cache_dir)))
    target = (
        Path(os.path.expanduser(str(cache_fp)))
        if cache_fp is not None
        else root / str(source_path).replace("://", "/").lstrip("/")
    )
    return target if target.is_absolute() else root / target


def _populate(source_path, cache_path: Path, backend_args=None) -> None:
    """Fill the cache slot atomically: copy into a temp file, then rename.

    Publishing via ``os.replace`` means an interrupted copy leaves only a
    ``.tmp`` sibling behind; a cache file that exists is always complete, so
    the size-based hit check below can never accept a truncated download.
    """
    if parse_uri_scheme(source_path) == "file" and uri_to_local_path(source_path).resolve() == cache_path.resolve():
        return
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    if cache_path.exists():
        cache_path.unlink(missing_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        copy_uri(source_path, tmp_path, **_storage_options(backend_args))
        os.replace(tmp_path, cache_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def _populate_lock(cache_path: Path):
    """Serialize cache population across local processes with a sibling flock.

    Without an initialized torch process group every process reports rank 0,
    so plain multi-process launches (e.g. parallel evaluation workers) would
    all download the same asset concurrently. The lock elects one populator;
    the waiters re-run the existence check inside :func:`_populate` and hit
    the cache. On filesystems without flock support this degrades to
    lock-free operation, which stays corruption-safe thanks to the atomic
    temp-file publish in :func:`_populate`.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        yield
        return
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError:  # pragma: no cover - flock unsupported (some NFS mounts)
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _distributed_rank_and_barrier():
    """Return ``(rank, barrier)``; missing process group is treated as single-process rank 0."""

    try:
        from worldfoundry.core.distributed.runtime import torch_process_group as distributed

        return distributed.get_rank(), distributed.barrier
    except Exception:
        return 0, lambda: None


def download_from_cache_or_uri(
    source_path,
    cache_fp=None,
    cache_dir=None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
) -> str:
    """Resolve a local/remote URI to a rank-synchronized local cache file."""

    del backend_key
    path = _cache_path(source_path, cache_fp, cache_dir)
    rank, barrier = _distributed_rank_and_barrier()
    if not rank_sync or rank == 0:
        with _populate_lock(path):
            _populate(source_path, path, backend_args)
    if rank_sync:
        barrier()
    return str(path)


def load_from_cache_or_uri(
    source_path,
    cache_fp=None,
    cache_dir=None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
    easy_io_kwargs: dict[str, Any] | None = None,
):
    """Cache an inference asset locally, then deserialize it."""

    path = download_from_cache_or_uri(source_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key)
    return load_serialized(path, **(easy_io_kwargs or {}))


download_from_s3_with_cache = download_from_cache_or_uri
load_from_s3_with_cache = load_from_cache_or_uri

__all__ = [
    "download_from_cache_or_uri",
    "download_from_s3_with_cache",
    "load_from_cache_or_uri",
    "load_from_s3_with_cache",
]
