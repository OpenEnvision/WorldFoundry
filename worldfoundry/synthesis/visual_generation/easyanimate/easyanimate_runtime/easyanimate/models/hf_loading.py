from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.io.paths import resolve_local_hf_model_path


def looks_like_hf_repo_id(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return False
    return value.count("/") == 1


def resolve_pretrained_subfolder(pretrained_model_path: str, subfolder: str | None = None) -> str:
    """Return a local directory for EasyAnimate's file-based loaders.

    Local paths are used as-is. Hugging Face repo ids are resolved only from
    WorldFoundry-local storage; inference never downloads a snapshot.
    """

    model_path = str(pretrained_model_path)
    expanded = str(Path(model_path).expanduser())
    local_root = expanded if Path(expanded).exists() else model_path
    local_subfolder = os.path.join(local_root, subfolder) if subfolder else local_root
    if os.path.isdir(local_subfolder):
        return local_subfolder
    if not looks_like_hf_repo_id(model_path):
        return local_subfolder

    snapshot_path = resolve_local_hf_model_path(model_path)
    resolved = snapshot_path / subfolder if subfolder else snapshot_path
    if not resolved.is_dir():
        raise FileNotFoundError(f"EasyAnimate local checkpoint is missing subfolder {subfolder!r}: {resolved}")
    return str(resolved)
