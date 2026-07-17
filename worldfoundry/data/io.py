"""Data file I/O helpers used by bundled model runtimes.

This module intentionally stays thin: data-facing callers can import from
``worldfoundry.data.io``, while the actual storage and serialization behavior is
implemented in ``worldfoundry.core.io``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any
from urllib.parse import urlparse

from worldfoundry.core.io import (
    copy_uri,
    dump_serialized,
    exists_uri,
    load_serialized,
    save_image_or_video_tensor,
)
from worldfoundry.core.io.cache import (
    download_from_cache_or_uri as _download_from_cache_or_uri,
)
from worldfoundry.core.io.cache import (
    load_from_cache_or_uri as _load_from_cache_or_uri,
)


def _strip_storage_compat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs.pop("backend_args", None)
    kwargs.pop("backend_key", None)
    return kwargs


def load(file: str | os.PathLike[str] | IO[Any], *, file_format: str | None = None, **kwargs: Any) -> Any:
    """Load a local/URI data object, inferring format from suffix when needed."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    if "weights_only" not in kwargs:
        normalized = (file_format or str(file).rsplit(".", 1)[-1]).lower()
        if normalized in {"pt", "pth", "ckpt", "bin"}:
            kwargs["weights_only"] = False
    return load_serialized(file, file_format=file_format, **kwargs)


def dump(
    obj: Any,
    file: str | os.PathLike[str] | IO[Any] | None = None,
    *,
    file_format: str | None = None,
    **kwargs: Any,
) -> Any:
    """Dump a data object, inferring format from suffix when possible."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return dump_serialized(obj, file, file_format=file_format, **kwargs)


def exists(file: str | os.PathLike[str], **kwargs: Any) -> bool:
    """Return whether a local/URI data object exists."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return exists_uri(file, **kwargs)


def copyfile_to_local(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    dst_type: str = "file",
    **kwargs: Any,
) -> str:
    """Copy a local/URI data object to a local file or directory."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    destination = Path(dst).expanduser()
    if dst_type == "dir":
        destination = destination / (Path(urlparse(str(src)).path).name or Path(str(src)).name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return copy_uri(src, destination, **kwargs)


def copyfile(src: str | os.PathLike[str], dst: str | os.PathLike[str], **kwargs: Any) -> str:
    """Copy a local/URI data object to another local/URI destination."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return copy_uri(src, dst, **kwargs)


def download_from_cache_or_uri(
    source_path: str | os.PathLike[str],
    cache_fp: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
) -> str:
    """Resolve a local/URI source to a local cached file."""

    return _download_from_cache_or_uri(source_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key)


def load_from_cache_or_uri(
    source_path: str | os.PathLike[str],
    cache_fp: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
    easy_io_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a local/URI data object, caching remote sources on local disk first."""

    return _load_from_cache_or_uri(
        source_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key, easy_io_kwargs
    )


def load_from_s3_with_cache(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for bundled runtimes that used S3-specific naming."""

    return load_from_cache_or_uri(*args, **kwargs)


def set_s3_backend(*args: Any, **kwargs: Any) -> None:
    """Compatibility no-op; URI handling is centralized in ``worldfoundry.core.io``."""

    del args, kwargs


def save_img_or_video(
    sample_c_t_h_w_in01,
    save_fp_wo_ext: str | os.PathLike[str] | IO[Any],
    fps: int = 24,
    quality: int | None = None,
    ffmpeg_params: list[str] | None = None,
) -> None:
    """Save a ``[C,T,H,W]`` tensor as ``.jpg`` for one frame or ``.mp4`` otherwise."""

    save_image_or_video_tensor(
        sample_c_t_h_w_in01,
        save_fp_wo_ext,
        fps=fps,
        quality=quality,
        ffmpeg_params=ffmpeg_params,
        value_range="0,1",
    )


easy_io = SimpleNamespace(
    copyfile=copyfile,
    copyfile_to_local=copyfile_to_local,
    dump=dump,
    exists=exists,
    load=load,
    set_s3_backend=set_s3_backend,
)


__all__ = [
    "copyfile",
    "copyfile_to_local",
    "download_from_cache_or_uri",
    "dump",
    "easy_io",
    "exists",
    "load",
    "load_from_cache_or_uri",
    "load_from_s3_with_cache",
    "save_img_or_video",
    "set_s3_backend",
]
