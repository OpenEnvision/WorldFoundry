"""Immutable checkpoint sources used by diffusion recipes.

A recipe names one logical role (``weights``, ``vae``, ``tokenizer``, …) as a
:class:`CheckpointSpec`.  The resolver / loader consume that spec; the runner
and ``Denoiser`` Protocol never see Hub IDs or integrity maps.

Construction fails unless the spec is a complete, safe, self-consistent
declaration.  After ``__post_init__``, ``source`` is always a ``tuple[str, ...]``
and the mapping fields are frozen ``MappingProxyType`` views.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence, cast

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """A local or Hub-backed source for one logical model component.

    Exactly one of a local ``source`` or a Hub ``repo_id`` must be present.

    Attributes:
        source: Local file(s) or a single directory.  Normalized to
            ``tuple[str, ...]``.  Empty strings are rejected.
        repo_id: Hugging Face repository.  When there is no local source,
            ``files`` must list every weight file to materialize.
        revision: Optional Hub git revision (branch, tag, or commit).
        files: Relative weight filenames under the snapshot / directory.
            Duplicates and blank names are rejected.
        allow_patterns: Extra Hub download filters merged with ``files``.
        metadata: Caller-owned audit notes; never used for I/O.
        file_sha256 / file_size_bytes: Integrity for declared ``files``.
            Keys that are not in ``files`` fail construction.
        resource_sha256 / resource_size_bytes: Integrity for sidecar
            resources (tokenizer JSON, configs).  Keys must be relative,
            must not contain ``..`` or overlap ``files``, and must use a
            64-hex SHA-256 plus a positive byte size.

    Raises:
        ValueError: Missing source and ``repo_id``; Hub spec without
            ``files``; empty / duplicate file names; integrity keys that
            reference undeclared files; resource paths that collide with
            ``files`` or are unsafe; invalid SHA-256; non-positive sizes.
        TypeError: A size field is a ``bool`` (rejected so ``True`` cannot
            silently become ``1``).
    """

    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None
    repo_id: str | None = None
    revision: str | None = None
    files: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    file_sha256: Mapping[str, str] = field(default_factory=dict)
    file_size_bytes: Mapping[str, int] = field(default_factory=dict)
    resource_sha256: Mapping[str, str] = field(default_factory=dict)
    resource_size_bytes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is None:
            sources: tuple[str, ...] = ()
        elif isinstance(self.source, (str, os.PathLike)):
            sources = (str(self.source),)
        else:
            sources = tuple(str(value) for value in self.source)
        if not sources and not self.repo_id:
            raise ValueError("checkpoint requires a local source or repo_id")
        files = tuple(str(value) for value in self.files)
        if self.repo_id and not sources and not files:
            raise ValueError("Hub-backed module checkpoints must declare files")
        if any(not source.strip() for source in sources):
            raise ValueError("checkpoint sources cannot be empty")
        if any(not filename.strip() for filename in files):
            raise ValueError("checkpoint files cannot be empty")
        if len(files) != len(set(files)):
            raise ValueError("checkpoint files cannot contain duplicates")
        declared_files = set(files)
        file_sha256 = {str(name): str(digest).lower() for name, digest in self.file_sha256.items()}
        file_size_bytes: dict[str, int] = {}
        for name, size in self.file_size_bytes.items():
            if isinstance(size, bool):
                raise TypeError("checkpoint file sizes must be integers, not bool")
            file_size_bytes[str(name)] = int(size)
        resource_sha256 = {
            str(name): str(digest).lower()
            for name, digest in self.resource_sha256.items()
        }
        resource_size_bytes: dict[str, int] = {}
        for name, size in self.resource_size_bytes.items():
            if isinstance(size, bool):
                raise TypeError("checkpoint resource sizes must be integers, not bool")
            resource_size_bytes[str(name)] = int(size)
        unknown_audits = sorted((set(file_sha256) | set(file_size_bytes)) - declared_files)
        if unknown_audits:
            raise ValueError(f"checkpoint integrity metadata references undeclared files: {unknown_audits}")
        resources = set(resource_sha256) | set(resource_size_bytes)
        overlap = sorted(resources & declared_files)
        if overlap:
            raise ValueError(f"checkpoint resources duplicate loaded files: {overlap}")
        unsafe_resources = sorted(
            name
            for name in resources
            if Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or not name
        )
        if unsafe_resources:
            raise ValueError(f"checkpoint resources contain unsafe relative paths: {unsafe_resources}")
        invalid_digests = sorted(
            name
            for name, digest in {**file_sha256, **resource_sha256}.items()
            if _SHA256_PATTERN.fullmatch(digest) is None
        )
        if invalid_digests:
            raise ValueError(f"checkpoint files contain invalid SHA-256 digests: {invalid_digests}")
        invalid_sizes = sorted(
            name
            for name, size in {**file_size_bytes, **resource_size_bytes}.items()
            if size <= 0
        )
        if invalid_sizes:
            raise ValueError(f"checkpoint files contain non-positive byte sizes: {invalid_sizes}")
        object.__setattr__(self, "source", sources)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "allow_patterns", tuple(self.allow_patterns))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "file_sha256", MappingProxyType(file_sha256))
        object.__setattr__(self, "file_size_bytes", MappingProxyType(file_size_bytes))
        object.__setattr__(self, "resource_sha256", MappingProxyType(resource_sha256))
        object.__setattr__(self, "resource_size_bytes", MappingProxyType(resource_size_bytes))

    @property
    def sources(self) -> tuple[str, ...]:
        """Normalized local sources; empty when the spec is Hub-only."""
        return cast(tuple[str, ...], self.source)


__all__ = ["CheckpointSpec"]
