"""Small inference utilities shared by the in-tree MolmoBot modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, Union

from .torch_utils import barrier, get_global_rank

PathOrStr = Union[str, Path]  # noqa: UP007 - Runtime type alias kept as typing.Union.
T = TypeVar("T")


def is_url(path: PathOrStr) -> bool:
    value = str(path)
    return "://" in value and value.split("://", 1)[0].isalnum()


def ensure_multiple_of(value: int, multiple: int) -> int:
    return multiple * math.ceil(value / multiple)


def split_into_groups(values: Sequence[T], max_group_size: int | None):
    if max_group_size is None:
        return [values]
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive.")
    group_count = (len(values) + max_group_size - 1) // max_group_size
    per_group, remainder = divmod(len(values), group_count)
    output = []
    offset = 0
    for group_index in range(group_count):
        size = per_group + int(group_index < remainder)
        output.append(values[offset : offset + size])
        offset += size
    return output


def get_all_keys(dicts: Sequence[dict]):
    keys = set()
    for item in dicts:
        keys.update(item)
    return keys


def _looks_like_hf_repo_id(value: str) -> bool:
    if value.startswith(("/", ".", "~")) or "\\" in value:
        return False
    parts = value.strip("/").split("/")
    return len(parts) == 2 and all(parts)


def resource_path(
    folder: PathOrStr,
    fname: str | None = None,
    local_cache: PathOrStr | None = None,
    progress=None,
    cache_dir: PathOrStr | None = None,
    quiet: bool = False,
) -> Path:
    """Resolve a staged local path or hfd model asset without network access."""
    del progress, quiet, cache_dir
    folder_value = str(folder)
    if fname is None:
        candidate = Path(folder_value).expanduser()
        if candidate.is_file():
            return candidate
        if is_url(folder_value):
            raise FileNotFoundError(f"Remote URL access is disabled for MolmoBot inference: {folder_value}")
        raise FileNotFoundError(f"Resource does not exist: {folder_value}")

    if local_cache is not None:
        cached = Path(local_cache).expanduser() / fname
        if cached.is_file():
            return cached

    local_folder = Path(folder_value).expanduser()
    local_candidate = local_folder / fname
    if local_candidate.is_file():
        return local_candidate
    if local_folder.is_dir() or local_folder.is_absolute():
        raise FileNotFoundError(f"Resource does not exist: {local_candidate}")

    if is_url(folder_value):
        raise FileNotFoundError(
            "Remote URL access is disabled for MolmoBot inference; stage the asset locally: "
            f"{folder_value.rstrip('/')}/{fname}"
        )
    if _looks_like_hf_repo_id(folder_value):
        from worldfoundry.core.io.paths import resolve_local_hf_model_path

        return resolve_local_hf_model_path(folder_value, required_files=(fname,)) / fname
    raise FileNotFoundError(f"Cannot resolve {fname!r} from local folder or model repository {folder_value!r}.")


def rank0_resource_path(device, *args, **kwargs) -> Path | None:
    """Resolve once per distributed job from the shared local model store."""
    del device
    result = resource_path(*args, **kwargs) if get_global_rank() == 0 else None
    barrier()
    if result is None:
        result = resource_path(*args, **kwargs)
    return result


__all__ = [
    "ensure_multiple_of",
    "get_all_keys",
    "is_url",
    "rank0_resource_path",
    "resource_path",
    "split_into_groups",
]
