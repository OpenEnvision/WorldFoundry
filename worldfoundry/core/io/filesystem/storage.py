"""URI storage primitives for local and remote file-like paths.

``open_uri`` / ``copy_uri`` / ``list_uri`` hide ``file://`` vs S3.
Model code should not import boto3. Suffix and media kind live in
``media.py``.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Generator, Iterable, TextIO
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

# ──────────────────────────────────────────────────────────────────────────
# Scheme dispatch — empty / file:// is local; anything else needs fsspec
# ──────────────────────────────────────────────────────────────────────────


def parse_uri_scheme(uri: str | os.PathLike[str]) -> str:
    """Return the lowercase URI scheme, using ``file`` for local paths."""

    parsed = urlparse(str(uri))
    return parsed.scheme.lower() or "file"


def is_remote_uri(uri: str | os.PathLike[str]) -> bool:
    """Return whether a path needs a non-local storage backend."""

    return parse_uri_scheme(uri) not in {"", "file"}


def uri_to_local_path(uri: str | os.PathLike[str]) -> Path:
    """Convert a local path or ``file://`` URI into a ``Path``."""

    text = str(uri)
    parsed = urlparse(text)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"URI is not local: {uri}")
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    return Path(text).expanduser()


@contextmanager
def open_uri(
    uri: str | os.PathLike[str],
    mode: str = "rb",
    *,
    encoding: str = "utf-8",
    **storage_options,
) -> Generator[BinaryIO | TextIO, None, None]:
    """Open a local or remote URI.

    ``fsspec`` is used when available. Without it, local files are supported for
    all modes and HTTP(S) URLs are supported for reads.
    """

    scheme = parse_uri_scheme(uri)
    if scheme == "file":
        path = uri_to_local_path(uri)
        if any(flag in mode for flag in ("w", "a", "x")):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding=None if "b" in mode else encoding) as handle:
            yield handle
        return

    fsspec = _optional_fsspec()
    if fsspec is not None:
        with fsspec.open(str(uri), mode=mode, encoding=None if "b" in mode else encoding, **storage_options) as handle:
            yield handle
        return

    if scheme in {"http", "https"} and "r" in mode and all(flag not in mode for flag in ("w", "a", "x")):
        data = _read_http_bytes(str(uri))
        if "b" in mode:
            with io.BytesIO(data) as handle:
                yield handle
        else:
            with io.StringIO(data.decode(encoding)) as handle:
                yield handle
        return

    raise RuntimeError(f"Remote URI {uri!r} requires fsspec or a supported read-only HTTP(S) fallback")


def read_binary_uri(uri: str | os.PathLike[str], **storage_options) -> bytes:
    """Read all bytes from a URI."""

    with open_uri(uri, "rb", **storage_options) as handle:
        return handle.read()


def read_text_uri(uri: str | os.PathLike[str], *, encoding: str = "utf-8", **storage_options) -> str:
    """Read all text from a URI."""

    with open_uri(uri, "r", encoding=encoding, **storage_options) as handle:
        return handle.read()


def write_binary_uri(
    uri: str | os.PathLike[str], data: bytes | bytearray | memoryview | io.BytesIO, **storage_options
) -> None:
    """Write bytes to a URI, creating local parent directories as needed."""

    if isinstance(data, io.BytesIO):
        data.seek(0)
        payload = data.read()
    else:
        payload = bytes(data)
    with open_uri(uri, "wb", **storage_options) as handle:
        handle.write(payload)


def write_text_uri(
    uri: str | os.PathLike[str],
    data: str,
    *,
    encoding: str = "utf-8",
    **storage_options,
) -> None:
    """Write text to a URI, creating local parent directories as needed."""

    with open_uri(uri, "w", encoding=encoding, **storage_options) as handle:
        handle.write(data)


# ──────────────────────────────────────────────────────────────────────────
# Exists / list / copy / remove — local Path vs fsspec; HTTP reads without fsspec
# ──────────────────────────────────────────────────────────────────────────


def exists_uri(uri: str | os.PathLike[str], **storage_options) -> bool:
    """Return whether a URI exists."""

    scheme = parse_uri_scheme(uri)
    if scheme == "file":
        return uri_to_local_path(uri).exists()
    fsspec = _optional_fsspec()
    if fsspec is None:
        if scheme in {"http", "https"}:
            try:
                request = Request(str(uri), method="HEAD")
                with urlopen(request, timeout=10):
                    return True
            except Exception:
                return False
        return False
    filesystem, path = fsspec.core.url_to_fs(str(uri), **storage_options)
    return filesystem.exists(path)


def is_file_uri(uri: str | os.PathLike[str], **storage_options) -> bool:
    """Return whether a URI points to a regular file."""

    scheme = parse_uri_scheme(uri)
    if scheme == "file":
        return uri_to_local_path(uri).is_file()
    fsspec = _optional_fsspec()
    if fsspec is None:
        return exists_uri(uri)
    filesystem, path = fsspec.core.url_to_fs(str(uri), **storage_options)
    return filesystem.isfile(path)


def is_dir_uri(uri: str | os.PathLike[str], **storage_options) -> bool:
    """Return whether a URI points to a directory."""

    scheme = parse_uri_scheme(uri)
    if scheme == "file":
        return uri_to_local_path(uri).is_dir()
    fsspec = _optional_fsspec()
    if fsspec is None:
        return False
    filesystem, path = fsspec.core.url_to_fs(str(uri), **storage_options)
    return filesystem.isdir(path)


def join_uri(base: str | os.PathLike[str], *parts: str | os.PathLike[str]) -> str:
    """Join path components without losing remote URI schemes."""

    text = str(base)
    if parse_uri_scheme(text) == "file":
        return str(uri_to_local_path(text).joinpath(*(str(part) for part in parts)))
    stripped = text.rstrip("/")
    suffix = "/".join(str(part).strip("/") for part in parts)
    return f"{stripped}/{suffix}" if suffix else stripped


def list_uri(
    uri: str | os.PathLike[str],
    *,
    recursive: bool = False,
    suffix: str | tuple[str, ...] | None = None,
    **storage_options,
) -> list[str]:
    """List files below a local or fsspec-backed URI."""

    if parse_uri_scheme(uri) == "file":
        root = uri_to_local_path(uri)
        if not root.exists():
            return []
        paths: Iterable[Path]
        paths = root.rglob("*") if recursive else root.iterdir()
        values = [str(path) for path in paths if path.is_file()]
    else:
        fsspec = _optional_fsspec()
        if fsspec is None:
            raise RuntimeError("Remote listing requires fsspec")
        filesystem, path = fsspec.core.url_to_fs(str(uri), **storage_options)
        pattern = f"{path.rstrip('/')}/**" if recursive else f"{path.rstrip('/')}/*"
        values = [
            f"{filesystem.protocol}://{item}" if isinstance(filesystem.protocol, str) else item
            for item in filesystem.glob(pattern)
        ]
    if suffix is None:
        return sorted(values)
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    return sorted(value for value in values if value.endswith(suffixes))


_COPY_CHUNK_BYTES = 16 * 1024 * 1024


@contextmanager
def local_path_for_uri(uri: str | os.PathLike[str], **storage_options) -> Generator[Path, None, None]:
    """Yield a local path, downloading remote bytes to a temporary file when needed."""

    if parse_uri_scheme(uri) == "file":
        yield uri_to_local_path(uri)
        return
    suffix = Path(urlparse(str(uri)).path).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        with open_uri(uri, "rb", **storage_options) as source_handle:
            shutil.copyfileobj(source_handle, handle, length=_COPY_CHUNK_BYTES)
        handle.flush()
        yield Path(handle.name)


def copy_uri(src: str | os.PathLike[str], dst: str | os.PathLike[str], **storage_options) -> str:
    """Copy one URI to another using storage-aware byte streams.

    Remote transfers stream in bounded chunks instead of buffering the whole
    object in memory; this is the byte mover underneath the download caches,
    where objects are multi-GB checkpoints.
    """

    if parse_uri_scheme(src) == "file" and parse_uri_scheme(dst) == "file":
        source = uri_to_local_path(src)
        target = uri_to_local_path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(shutil.copy(source, target))
    with open_uri(src, "rb", **storage_options) as source_handle:
        with open_uri(dst, "wb", **storage_options) as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=_COPY_CHUNK_BYTES)
    return str(dst)


def remove_uri(uri: str | os.PathLike[str], **storage_options) -> None:
    """Remove a file or directory URI."""

    if parse_uri_scheme(uri) == "file":
        path = uri_to_local_path(uri)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        return
    fsspec = _optional_fsspec()
    if fsspec is None:
        raise RuntimeError("Remote removal requires fsspec")
    filesystem, path = fsspec.core.url_to_fs(str(uri), **storage_options)
    filesystem.rm(path, recursive=True)


def _read_http_bytes(uri: str) -> bytes:
    """Fetch HTTP(S) without fsspec; identify as WorldFoundry so CDNs do not block urllib."""

    request = Request(uri, headers={"User-Agent": "worldfoundry"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _optional_fsspec():
    """Return the fsspec module or ``None`` so local-only jobs skip the S3 extra."""

    try:
        import fsspec  # type: ignore
    except ImportError:
        return None
    return fsspec


__all__ = [
    "copy_uri",
    "exists_uri",
    "is_dir_uri",
    "is_file_uri",
    "is_remote_uri",
    "join_uri",
    "list_uri",
    "local_path_for_uri",
    "open_uri",
    "parse_uri_scheme",
    "read_binary_uri",
    "read_text_uri",
    "remove_uri",
    "uri_to_local_path",
    "write_binary_uri",
    "write_text_uri",
]
