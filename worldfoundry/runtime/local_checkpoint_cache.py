"""Optional checkpoint staging for latency-sensitive resident runtimes.

Deployments may explicitly stage immutable model files into a faster storage
tier before loading. Rank zero performs the copy once and publishes the
resolved path to its peers. The open-source runtime does not guess storage
topology from filesystem paths and staging is disabled by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

_READY_FILE = ".worldfoundry-local-cache.json"


def _validated_relative_paths(
    paths: Iterable[str],
    *,
    parameter: str,
) -> tuple[str, ...]:
    validated: list[str] = []
    for raw_path in paths:
        value = str(raw_path)
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{parameter} entries must be relative paths without '..': {value!r}"
            )
        validated.append(value)
    return tuple(validated)


def _resolved_within(root: Path, candidate: Path, *, description: str) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"{description} resolves outside {resolved_root}: {candidate}"
        ) from error
    return resolved_candidate


def _resolved_relative(root: Path, relative: str, *, parameter: str) -> Path:
    return _resolved_within(
        root,
        root / relative,
        description=f"{parameter} entry {relative!r}",
    )


@contextmanager
def _publish_lock(cache_root: Path, target_name: str) -> Iterator[None]:
    """Serialize the publish step across unrelated processes via ``flock``.

    Staging copies may run concurrently, but replacing the published target
    directory must be exclusive: without the lock, one process could delete a
    directory that a concurrent process just published and started reading.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        yield
        return
    lock_path = cache_root / f".{target_name}.lock"
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _enabled() -> bool:
    raw = os.getenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "0").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT must be a boolean value."
    )


def _selected_files(source: Path, include_paths: tuple[str, ...]) -> list[Path]:
    if not include_paths:
        selected_files: list[Path] = []
        for item in source.rglob("*"):
            if item.is_file() and not item.is_symlink():
                _resolved_within(source, item, description="checkpoint source file")
                selected_files.append(item)
        return selected_files
    selected: set[Path] = set()
    for relative in include_paths:
        item = _resolved_relative(source, relative, parameter="include_paths")
        if item.is_file() and not item.is_symlink():
            selected.add(item)
        elif item.is_dir():
            for child in item.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    _resolved_within(
                        source,
                        child,
                        description="checkpoint source file",
                    )
                    selected.add(child)
    return list(selected)


def _directory_size(path: Path, include_paths: tuple[str, ...] = ()) -> int:
    return sum(item.stat().st_size for item in _selected_files(path, include_paths))


def _cache_target(
    source: Path,
    cache_root: Path,
    include_paths: tuple[str, ...] = (),
) -> Path:
    identity = json.dumps([str(source), sorted(include_paths)], separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    safe_name = "-".join(part for part in source.name.split() if part) or "checkpoint"
    target = cache_root / f"{safe_name}-{digest}"
    _resolved_within(cache_root, target, description="local checkpoint target")
    return target


def _copy_tree_parallel(
    source: Path,
    target: Path,
    *,
    include_paths: tuple[str, ...] = (),
) -> None:
    """Copy independent checkpoint shards concurrently from distributed storage."""

    files = _selected_files(source, include_paths)
    directories = sorted(
        {
            parent
            for item in files
            for parent in item.parents
            if parent != source and source in parent.parents
        },
        key=lambda item: len(item.parts),
    )
    target.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        (target / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)

    links: list[tuple[Path, str]] = []
    if not include_paths:
        for link in (item for item in source.rglob("*") if item.is_symlink()):
            _resolved_within(source, link, description="checkpoint source symlink")
            destination = target / link.relative_to(source)
            link_target = os.readlink(link)
            resolved_link_target = Path(link_target)
            if not resolved_link_target.is_absolute():
                resolved_link_target = destination.parent / resolved_link_target
            _resolved_within(
                target,
                resolved_link_target,
                description="staged checkpoint symlink",
            )
            links.append((destination, link_target))
    for destination, link_target in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(link_target)

    files.sort(key=lambda item: item.stat().st_size, reverse=True)
    workers = max(
        int(os.getenv("WORLDFOUNDRY_REALTIME_CHECKPOINT_COPY_WORKERS", "4") or "4"),
        1,
    )

    def copy_file(item: Path) -> None:
        shutil.copy2(item, target / item.relative_to(source))

    with ThreadPoolExecutor(max_workers=min(workers, max(len(files), 1))) as executor:
        list(executor.map(copy_file, files))


def _is_ready(target: Path, source: Path, required_paths: tuple[str, ...]) -> bool:
    ready = target / _READY_FILE
    if not ready.is_file():
        return False
    try:
        payload = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("source") != str(source):
        return False
    return all(
        _resolved_relative(target, relative, parameter="required_paths").exists()
        for relative in required_paths
    )


def _stage_rank_zero(
    source: Path,
    *,
    cache_root: Path,
    required_paths: tuple[str, ...],
    include_paths: tuple[str, ...],
) -> Path:
    target = _cache_target(source, cache_root, include_paths)
    if _is_ready(target, source, required_paths):
        return target

    cache_root.mkdir(parents=True, exist_ok=True)
    source_bytes = _directory_size(source, include_paths)
    free_bytes = shutil.disk_usage(cache_root).free
    if source_bytes > int(free_bytes * 0.9):
        raise OSError(
            f"Not enough node-local space to stage {source} "
            f"({source_bytes / 1024**3:.1f} GiB required, {free_bytes / 1024**3:.1f} GiB free)."
        )

    temporary = _resolved_within(
        cache_root,
        Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=cache_root)),
        description="checkpoint staging directory",
    )
    logger.info(
        "staging %.1f GiB checkpoint from %s to %s",
        source_bytes / 1024**3,
        source,
        target,
    )
    try:
        _copy_tree_parallel(source, temporary, include_paths=include_paths)
        (temporary / _READY_FILE).write_text(
            json.dumps(
                {
                    "source": str(source),
                    "size_bytes": source_bytes,
                    "include_paths": list(include_paths),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with _publish_lock(cache_root, target.name):
            if _is_ready(target, source, required_paths):
                # A concurrent process published a valid tree while we were
                # copying. Reuse it and discard the duplicate; deleting the
                # published tree here would break readers already loading it.
                shutil.rmtree(temporary, ignore_errors=True)
                return target
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def stage_checkpoint_for_realtime(
    source: str | Path,
    *,
    required_paths: Iterable[str] = (),
    include_paths: Iterable[str] = (),
    distributed: Any = None,
) -> Path:
    """Return an immutable node-local view of ``source`` when configured.

    ``distributed`` may be ``torch.distributed``.  All ranks call this
    function in the same order; only rank zero performs I/O and broadcasts the
    resulting path or error.
    """

    resolved = Path(source).expanduser().resolve()
    if not _enabled():
        return resolved
    required = _validated_relative_paths(required_paths, parameter="required_paths")
    included = tuple(
        sorted(_validated_relative_paths(include_paths, parameter="include_paths"))
    )
    for relative in (*required, *included):
        parameter = "required_paths" if relative in required else "include_paths"
        _resolved_relative(resolved, relative, parameter=parameter)
    cache_root_value = os.getenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE")
    if not cache_root_value:
        raise ValueError(
            "Set WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE to an explicit "
            "staging directory when checkpoint staging is enabled."
        )
    cache_root = Path(cache_root_value).expanduser().resolve()
    missing_source = [
        relative
        for relative in required
        if not _resolved_relative(
            resolved,
            relative,
            parameter="required_paths",
        ).exists()
    ]
    if missing_source:
        raise FileNotFoundError(
            f"Checkpoint is incomplete at {resolved}; missing: {', '.join(missing_source)}"
        )
    is_distributed = bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    )
    rank = int(distributed.get_rank()) if is_distributed else 0
    status: list[dict[str, str] | None] = [None]
    if rank == 0:
        try:
            target = _stage_rank_zero(
                resolved,
                cache_root=cache_root,
                required_paths=required,
                include_paths=included,
            )
            status[0] = {"path": str(target), "error": ""}
        except Exception as exc:
            status[0] = {"path": "", "error": f"{type(exc).__name__}: {exc}"}
    if is_distributed:
        distributed.broadcast_object_list(status, src=0)
    payload = status[0] or {}
    if payload.get("error"):
        raise RuntimeError(f"Realtime checkpoint staging failed: {payload['error']}")
    published_path = payload.get("path")
    if not published_path:
        raise RuntimeError("Realtime checkpoint staging returned no local path")
    target = _resolved_within(
        cache_root,
        Path(published_path).expanduser(),
        description="published local checkpoint path",
    )
    if not all(
        _resolved_relative(target, relative, parameter="required_paths").exists()
        for relative in required
    ):
        raise FileNotFoundError(f"Staged checkpoint is incomplete: {target}")
    return target


__all__ = ["stage_checkpoint_for_realtime"]
