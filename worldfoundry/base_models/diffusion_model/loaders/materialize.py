"""Checkpoint materialization shared by modules and non-weight resources.

Turns a frozen :class:`CheckpointSpec` into local paths the module loader can
open.  Recipes and runners never call Hugging Face or hash files themselves;
this resolver is the only I/O seam between a declarative spec and a
:class:`MaterializedCheckpoint`.

Local sources keep the user-facing symlink name (``model.safetensors``) so the
lazy loader still sees the format.  Hub specs first try a cached snapshot, then
download a filtered subset.  Integrity maps on the spec are verified after
paths exist; missing maps skip the audit.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .checkpoints import CheckpointSpec


@dataclass(frozen=True, slots=True)
class MaterializedCheckpoint:
    """Resolved local view of one declarative checkpoint source.

    Attributes:
        root: Snapshot or directory that owns relative resource paths.
        paths: Materialized weight files in ``CheckpointSpec.files`` order
            (or the raw local sources when ``files`` is empty).

    :meth:`directory` fails with :exc:`ValueError` on absolute / ``..``
    resource paths and :exc:`FileNotFoundError` when no candidate directory
    exists (including the flattened local-override layout).
    """

    root: Path
    paths: tuple[Path, ...]

    def directory(self, relative_path: str | os.PathLike[str] | None = None) -> Path:
        """Resolve a resource directory below the checkpoint root.

        A direct local directory override is also accepted when it already
        contains every declared file basename. This lets callers override a
        Hub subtree without reproducing the repository's parent layout.
        """

        if relative_path is None:
            candidate = self.root
        else:
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkpoint resource paths must be relative and cannot contain '..'")
            candidate = self.root / relative
        if candidate.is_dir():
            return candidate

        if len(self.paths) == 1 and self.paths[0] == self.root and self.root.is_dir():
            return self.root

        declared_names = {path.name for path in self.paths}
        if self.root.is_dir() and declared_names and all((self.root / name).is_file() for name in declared_names):
            return self.root
        raise FileNotFoundError(f"checkpoint resource directory does not exist: {candidate}")


class NativeCheckpointResolver:
    """Resolve local paths or a filtered Hub snapshot through core I/O.

    :meth:`materialize` picks local vs Hub from ``checkpoint.sources``, then
    always runs :meth:`_verify_integrity` when any hash/size map is set.

    Raises:
        FileNotFoundError: A local source, declared file, or audited sidecar
            is missing.
        ValueError: Local ``files`` without a single directory source; size
            or SHA-256 mismatch after materialization.
        RuntimeError: Declared ``files`` count no longer matches materialized
            paths (identity lost).
    """

    def materialize(self, checkpoint: CheckpointSpec) -> MaterializedCheckpoint:
        if checkpoint.sources:
            materialized = self._materialize_local(checkpoint)
        else:
            materialized = self._materialize_hub(checkpoint)
        self._verify_integrity(checkpoint, materialized)
        return materialized

    @staticmethod
    def _verify_integrity(
        checkpoint: CheckpointSpec,
        materialized: MaterializedCheckpoint,
    ) -> None:
        if not any(
            (
                checkpoint.file_sha256,
                checkpoint.file_size_bytes,
                checkpoint.resource_sha256,
                checkpoint.resource_size_bytes,
            )
        ):
            return
        if len(checkpoint.files) != len(materialized.paths):
            raise RuntimeError("checkpoint materialization lost declared file identity")
        paths = dict(zip(checkpoint.files, materialized.paths, strict=True))
        resource_paths = {
            name: materialized.root / name
            for name in set(checkpoint.resource_sha256) | set(checkpoint.resource_size_bytes)
        }
        missing_resources = sorted(name for name, path in resource_paths.items() if not path.is_file())
        if missing_resources:
            raise FileNotFoundError(f"checkpoint audited resources do not exist: {missing_resources}")
        audited_paths = {**paths, **resource_paths}
        expected_sizes = {
            **checkpoint.file_size_bytes,
            **checkpoint.resource_size_bytes,
        }
        for name, expected_size in expected_sizes.items():
            actual_size = audited_paths[name].stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"checkpoint size audit failed for {name!r}: expected {expected_size}, got {actual_size}"
                )
        expected_digests = {
            **checkpoint.file_sha256,
            **checkpoint.resource_sha256,
        }
        for name, expected_digest in expected_digests.items():
            digest = hashlib.sha256()
            with audited_paths[name].open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_digest = digest.hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(
                    f"checkpoint SHA-256 audit failed for {name!r}: expected {expected_digest}, got {actual_digest}"
                )

    @staticmethod
    def _materialize_local(checkpoint: CheckpointSpec) -> MaterializedCheckpoint:
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        # Preserve the user-facing symlink filename. Hugging Face snapshots use
        # names such as ``model.safetensors`` that point at extensionless
        # content-addressed blobs; resolving the final symlink loses the format
        # information used by the shared lazy loader.
        sources = tuple(resolve_worldfoundry_path(source).absolute() for source in checkpoint.sources)
        missing = tuple(path for path in sources if not path.exists())
        if missing:
            raise FileNotFoundError(f"checkpoint paths do not exist: {[str(path) for path in missing]}")

        if checkpoint.files:
            if len(sources) != 1 or not sources[0].is_dir():
                raise ValueError("local checkpoints with files require one directory source")
            root = sources[0]
            paths = tuple(root / filename for filename in checkpoint.files)
        else:
            paths = sources
            root = (
                sources[0]
                if len(sources) == 1 and sources[0].is_dir()
                else Path(os.path.commonpath([str(path.parent) for path in sources]))
            )

        missing = tuple(path for path in paths if not path.exists())
        if missing:
            raise FileNotFoundError(f"checkpoint files do not exist: {[str(path) for path in missing]}")
        return MaterializedCheckpoint(root=root, paths=paths)

    @staticmethod
    def _materialize_hub(checkpoint: CheckpointSpec) -> MaterializedCheckpoint:
        if checkpoint.repo_id is None:
            raise ValueError("checkpoint requires a local source or Hub repository")

        audited_resources = tuple(dict.fromkeys((*checkpoint.resource_sha256, *checkpoint.resource_size_bytes)))
        required_files = tuple(dict.fromkeys((*checkpoint.files, *audited_resources)))
        from worldfoundry.core.io.paths import resolve_local_hf_model_path

        try:
            root = resolve_local_hf_model_path(
                checkpoint.repo_id,
                required_files=required_files,
                revision=checkpoint.revision,
            )
        except FileNotFoundError:
            root = None
        if root is not None:
            return MaterializedCheckpoint(
                root=root,
                paths=tuple(root / filename for filename in checkpoint.files),
            )

        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        root = materialize_hf_snapshot(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            allow_patterns=tuple(
                dict.fromkeys(
                    (*checkpoint.allow_patterns, *required_files) if checkpoint.allow_patterns else required_files
                )
            ),
            required_files=required_files,
        )
        paths = tuple(root / filename for filename in checkpoint.files)
        return MaterializedCheckpoint(root=root, paths=paths)


__all__ = ["MaterializedCheckpoint", "NativeCheckpointResolver"]
