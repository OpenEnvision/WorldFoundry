"""Immutable checkpoint sources used by diffusion recipes.

A recipe names one logical role (``weights``, ``vae``, ``tokenizer``, …) as a
:class:`CheckpointSpec`.  The resolver / loader consume that spec; the runner
and ``Denoiser`` Protocol never see Hub IDs.

Construction requires a source and nonempty declared filenames. After
``__post_init__``, ``source`` is always a ``tuple[str, ...]`` and metadata is
a frozen ``MappingProxyType`` view.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence, cast


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """A local or Hub-backed source for one logical model component.

    A local ``source`` or a Hub ``repo_id`` must be present.

    Attributes:
        source: Local file(s) or a single directory.  Normalized to
            ``tuple[str, ...]``.  Empty strings are rejected.
        repo_id: Hugging Face repository.  When there is no local source,
            ``files`` must list every weight file to materialize.
        revision: Optional Hub git revision (branch, tag, or commit).
        files: Relative weight filenames under the snapshot / directory.
            Duplicates and blank names are rejected.
        resources: Required sidecar files, such as model configuration.
        allow_patterns: Extra Hub download filters merged with ``files``.
        metadata: Caller-owned audit notes; never used for I/O.

    Raises:
        ValueError: Missing source and ``repo_id``; Hub spec without
            ``files``; empty or duplicate file names.
    """

    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None
    repo_id: str | None = None
    revision: str | None = None
    files: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

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
        object.__setattr__(self, "source", sources)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "allow_patterns", tuple(self.allow_patterns))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sources(self) -> tuple[str, ...]:
        """Normalized local sources; empty when the spec is Hub-only."""
        return cast(tuple[str, ...], self.source)


__all__ = ["CheckpointSpec"]
