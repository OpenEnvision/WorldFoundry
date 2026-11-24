# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream source trees fetched into the WorldFoundry cache on first use.

Runners that execute official upstream code declare the repository, an exact
commit, and the directory they need instead of vendoring the tree.
:func:`ensure_source` materializes that commit once under
``${WORLDFOUNDRY_CACHE_DIR}/sources/<source_id>/<revision>`` and returns the
requested directory on later calls. The full commit SHA pins the content, so a
completed cache entry is reused as is. The module depends only on the runtime
layer.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from worldfoundry.runtime.env import EnvMapping, resolve_cache_dir

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_COMPLETE_MARKER = ".worldfoundry_source"


class SourceUnavailableError(RuntimeError):
    """Raised when a pinned source tree is not cached and cannot be fetched."""


@dataclass(frozen=True)
class SourceSpec:
    """One upstream directory pinned to a commit.

    Attributes:
        source_id: Cache key, usually the benchmark or model id.
        url: Git URL of the upstream repository.
        revision: Full 40-character commit SHA.
        subdir: Repository-relative directory the caller needs; empty for the root.
    """

    source_id: str
    url: str
    revision: str
    subdir: str = ""

    def __post_init__(self) -> None:
        if not _SOURCE_ID_RE.fullmatch(self.source_id):
            raise ValueError(f"source_id must be a single path segment, got {self.source_id!r}")
        if not self.url:
            raise ValueError(f"{self.source_id}: a source URL is required")
        if not _COMMIT_RE.fullmatch(self.revision):
            raise ValueError(f"{self.source_id}: revision must be a full commit SHA, got {self.revision!r}")
        subdir = PurePosixPath(self.subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError(f"{self.source_id}: subdir must stay inside the repository, got {self.subdir!r}")


def source_cache_root(env: EnvMapping | None = None) -> Path:
    """Return the directory that holds fetched source trees."""
    return resolve_cache_dir(env) / "sources"


def ensure_source(spec: SourceSpec, *, env: EnvMapping | None = None) -> Path:
    """Return ``spec.subdir`` inside the cached checkout of ``spec.revision``.

    The first call fetches only that commit (``git fetch --depth 1``) and stores
    its tree without Git metadata; later calls reuse the cache entry.

    Args:
        spec: Pinned upstream source.
        env: Optional environment mapping used to locate the cache; defaults to ``os.environ``.

    Returns:
        The absolute path of the requested directory inside the cache entry.

    Raises:
        SourceUnavailableError: The commit is not cached and could not be fetched,
            the cache entry exists without having been completed by a fetch, or the
            commit does not contain ``spec.subdir``.
    """
    root = source_cache_root(env).absolute() / spec.source_id / spec.revision
    if not (root / _COMPLETE_MARKER).is_file():
        if root.exists():
            raise SourceUnavailableError(
                f"{root} is not a completed fetch of {spec.url}@{spec.revision}; remove it to fetch again"
            )
        _fetch(spec, root)
    target = root / spec.subdir if spec.subdir else root
    if not target.is_dir():
        raise SourceUnavailableError(f"{spec.url}@{spec.revision} has no directory {spec.subdir!r}")
    return target


def _fetch(spec: SourceSpec, root: Path) -> None:
    """Fetch ``spec.revision`` into a staging directory and publish it as ``root``."""
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.revision[:12]}.", dir=root.parent))
    try:
        _git(spec, staging, "init", "-q")
        _git(spec, staging, "fetch", "-q", "--depth", "1", spec.url, spec.revision)
        _git(spec, staging, "checkout", "-q", "FETCH_HEAD")
        shutil.rmtree(staging / ".git")
        (staging / _COMPLETE_MARKER).write_text(f"{spec.url}\n{spec.revision}\n", encoding="utf-8")
        # mkdtemp creates the directory owner-only; a shared cache must stay readable.
        staging.chmod(0o755)
        try:
            staging.rename(root)
        except OSError as exc:
            # A concurrent fetch of the same revision may have published first.
            if not (root / _COMPLETE_MARKER).is_file():
                raise SourceUnavailableError(f"could not publish {spec.url}@{spec.revision} at {root}: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _git(spec: SourceSpec, cwd: Path, *args: str) -> None:
    try:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        raise SourceUnavailableError(
            f"could not fetch {spec.url}@{spec.revision} (git {args[0]}): {detail[-500:]}"
        ) from exc


__all__ = [
    "SourceSpec",
    "SourceUnavailableError",
    "ensure_source",
    "source_cache_root",
]
