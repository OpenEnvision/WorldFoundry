"""Local reads needed by MolmoBot inference."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Union

PathOrStr = Union[str, os.PathLike]  # noqa: UP007 - Runtime type alias kept as typing.Union.


def normalize_path(path: PathOrStr) -> str:
    return str(path).rstrip("/").removeprefix("file://")


def is_url(path: PathOrStr) -> bool:
    value = str(path)
    return "://" in value and value.split("://", 1)[0].isalnum()


def join_path(path1: PathOrStr, path2: PathOrStr) -> PathOrStr:
    if is_url(path1):
        return f"{str(path1).rstrip('/')}/{str(path2).lstrip('/')}"
    return Path(path1) / path2


def _local_read_path(path: PathOrStr) -> Path:
    normalized = normalize_path(path)
    if not is_url(normalized):
        return Path(normalized)
    raise FileNotFoundError(f"Remote reads are disabled for MolmoBot inference; stage this asset locally: {normalized}")


def get_bytes_range(path: PathOrStr, bytes_start: int, num_bytes: int | None) -> bytes:
    if bytes_start < 0 or (num_bytes is not None and num_bytes < 0):
        raise ValueError("Byte ranges must be non-negative.")
    local_path = _local_read_path(path)
    with local_path.open("rb") as stream:
        stream.seek(bytes_start)
        return stream.read(-1 if num_bytes is None else num_bytes)


def read_file(path: PathOrStr, mode: str = "r"):
    data = get_bytes_range(path, 0, None)
    if mode == "rb":
        return data
    if mode == "r":
        return data.decode("utf-8")
    raise ValueError(f"Unsupported read mode: {mode!r}")


def file_exists(path: PathOrStr) -> bool:
    normalized = normalize_path(path)
    if not is_url(normalized):
        return Path(normalized).is_file()
    try:
        _local_read_path(normalized)
        return True
    except (FileNotFoundError, OSError):
        return False


def write_file(
    directory: PathOrStr,
    fname: str,
    contents: str | bytes | Callable,
    save_overwrite: bool = False,
) -> Path:
    if is_url(directory):
        raise ValueError("MolmoBot inference does not upload files to remote storage.")
    directory_path = Path(normalize_path(directory) or ".")
    directory_path.mkdir(parents=True, exist_ok=True)
    target = directory_path / fname
    if target.exists() and not save_overwrite:
        raise FileExistsError(target)
    mode = "w" if isinstance(contents, str) else "wb"
    with tempfile.NamedTemporaryFile(mode=mode, dir=directory_path, delete=False) as stream:
        temporary = Path(stream.name)
        if callable(contents):
            contents(stream)
        else:
            stream.write(contents)
    temporary.replace(target)
    return target


__all__ = [
    "PathOrStr",
    "file_exists",
    "get_bytes_range",
    "is_url",
    "join_path",
    "normalize_path",
    "read_file",
    "write_file",
]
