"""Checkpoint discovery for the shared Video-Depth-Anything runtime."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.evaluation.utils import REPO_ROOT

VDA_FILENAMES = {
    "vits": "video_depth_anything_vits.pth",
    "vitl": "video_depth_anything_vitl.pth",
}


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _checkpoint_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("WORLDFOUNDRY_CKPT_DIR", "WORLDARENA_CHECKPOINT_ROOT"):
        checkpoint_root = os.environ.get(key, "").strip()
        if checkpoint_root:
            roots.append(Path(checkpoint_root).expanduser())
    roots.extend(
        [
            REPO_ROOT / "checkpoints",
            REPO_ROOT.parent / "ckpt",
            REPO_ROOT.parent / "WorldArena" / "ckpt",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def checkpoint_path(model: str = "vits") -> Path | None:
    filename = VDA_FILENAMES[model]
    env_key = (
        "WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT"
        if model == "vits"
        else "WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_VITL_CKPT"
    )
    explicit = os.environ.get(env_key, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() and path.stat().st_size > 0 else None
    for root in _checkpoint_roots():
        for relative in (
            Path("Video-Depth-Anything-Small") / filename,
            Path("Video-Depth-Anything") / filename,
            Path("video_depth_anything") / filename,
        ):
            candidate = root / relative
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def small_checkpoint_path() -> Path | None:
    return checkpoint_path("vits")


def resolve_video_depth_checkpoint(model: str = "vits", weights_path: str | None = None) -> Path | None:
    """Return a local VDA weight, or None when a network download is still allowed."""
    if weights_path:
        path = Path(weights_path).expanduser()
        if path.is_file() and path.stat().st_size > 0:
            return path
        raise FileNotFoundError(f"Video-Depth-Anything weight missing: {path}")
    found = checkpoint_path(model)
    if found is not None:
        return found
    if _offline():
        raise FileNotFoundError(
            f"Video-Depth-Anything {model} weight missing offline. "
            "Set WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT or place "
            f"{VDA_FILENAMES[model]} under WORLDFOUNDRY_CKPT_DIR."
        )
    return None
