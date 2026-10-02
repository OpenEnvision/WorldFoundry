"""Minimal local/Hugging Face serialization facade for inference assets.

Convenience load/save used by recipes. Production loaders should call
``hf`` / ``serialization`` / ``storage`` explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.core.io.formats.serialization import (
    _TORCH_FORMATS,
    dump_serialized,
    infer_serialization_format,
    load_serialized,
)
from worldfoundry.core.io.filesystem.storage import copy_uri, exists_uri, list_uri

# ──────────────────────────────────────────────────────────────────────────
# Path resolve — hf:// snapshots vs local expanduser; S3 stays a URI
# ──────────────────────────────────────────────────────────────────────────


def resolve_checkpoint_path(value: str | Path) -> str:
    """Materialize ``hf://`` via :func:`resolve_hf_path`; otherwise expand ``~`` only."""

    text = str(value)
    if text.startswith("hf://"):
        from worldfoundry.core.io.assets.hf import resolve_hf_path

        return str(resolve_hf_path(text))
    return str(Path(text).expanduser())


get_checkpoint_path = resolve_checkpoint_path
download_checkpoint = resolve_checkpoint_path


class EasyIO:
    """Thin facade over :mod:`serialization` / :mod:`storage` for recipe scripts."""

    def __init__(self) -> None:
        """Start with no per-backend fsspec options."""

        self._storage_options: dict[str, dict[str, Any]] = {}

    def _options(self, backend_key=None, **kwargs) -> dict[str, Any]:
        """Merge stored backend kwargs; drop EasyIO-only keys before fsspec."""

        options = dict(self._storage_options.get(str(backend_key), {})) if backend_key is not None else {}
        options.update(kwargs)
        for ignored in ("fast_backend", "backend_key"):
            options.pop(ignored, None)
        return options

    def exists(self, path, *, suppress_errors: bool = False, **kwargs) -> bool:
        """Return whether *path* resolves to an existing file or object.

        Missing paths return ``False``. Authentication, network, and argument
        errors propagate by default so callers do not misdiagnose an unhealthy
        backend as absent data. Legacy best-effort probes can opt in to the old
        behavior with ``suppress_errors=True``.
        """
        try:
            resolved = resolve_checkpoint_path(path)
            return exists_uri(resolved, **self._options(**kwargs))
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        except Exception as exc:
            if not suppress_errors:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "easy_io.exists(%r) failed with %s: %s; reporting non-existent", path, type(exc).__name__, exc
            )
            return False

    def load(self, path, *, map_location="cpu", weights_only=True, **kwargs):
        """Load serialized data, defaulting PyTorch formats to safe tensor-only mode.

        ``map_location``/``weights_only`` are torch-loader options; forwarding
        them to the JSON/YAML/pickle loaders would raise ``TypeError``, so they
        are attached only when the inferred format is a torch checkpoint.
        """

        options = {
            key: value
            for key, value in kwargs.items()
            if key in {"allow_pickle", "encoding", "file_format", "loader"}
        }
        resolved = resolve_checkpoint_path(path)
        try:
            inferred_format = infer_serialization_format(resolved, options.get("file_format"))
        except ValueError:
            inferred_format = None
        if inferred_format in _TORCH_FORMATS:
            options["map_location"] = map_location
            options["weights_only"] = weights_only
        return load_serialized(resolved, **options)

    def dump(self, value, path, **kwargs):
        """Serialize *value* to *path*; only ``encoding`` / ``file_format`` are forwarded."""

        options = {key: item for key, item in kwargs.items() if key in {"encoding", "file_format"}}
        return dump_serialized(value, path, **options)

    def copyfile(self, source, destination, **kwargs) -> str:
        """Copy any supported URI pair through :func:`copy_uri`."""

        return copy_uri(source, destination, **self._options(**kwargs))

    copyfile_from_local = copyfile
    copyfile_to_local = copyfile

    def list_dir_or_file(
        self,
        path,
        *,
        recursive: bool = False,
        list_dir: bool = False,
        list_file: bool = True,
        suffix=None,
        **kwargs,
    ) -> list[str]:
        """List files via :func:`list_uri`, or local dirs only when *list_file* is false."""

        if list_dir and not list_file:
            root = Path(path)
            iterator = root.rglob("*") if recursive else root.iterdir()
            return sorted(str(item) for item in iterator if item.is_dir())
        return list_uri(
            path,
            recursive=recursive,
            suffix=suffix,
            **self._options(**kwargs),
        )

    def set_s3_backend(self, *, backend_key="default", **kwargs) -> None:
        """Store fsspec-compatible options for subsequent calls using *backend_key*."""

        self._storage_options[str(backend_key)] = dict(kwargs)


easy_io = EasyIO()


__all__ = [
    "EasyIO",
    "download_checkpoint",
    "easy_io",
    "get_checkpoint_path",
    "resolve_checkpoint_path",
]
