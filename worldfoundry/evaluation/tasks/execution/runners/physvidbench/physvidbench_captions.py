"""PhysVidBench AuroraCap caption loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import resolve_env_path, bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_prompts import (
    resolve_physvidbench_root,
)

DEFAULT_CAPTIONS_DIR_REL = Path("captions")
DEFAULT_CAPTION_BASE_REL = Path("captions/cogvideo2b")
CAPTION_SUFFIXES = ("_FP", "_OP", "_SR", "_TD", "_AU", "_MT", "_FM", "")




def resolve_captions_dir(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"PhysVidBench captions directory not found: {path}")
        return path
    env_dir = resolve_env_path("WORLDFOUNDRY_PHYSVIDBENCH_CAPTIONS_DIR")
    if env_dir is not None:
        if not env_dir.is_dir():
            raise FileNotFoundError(f"PhysVidBench captions directory not found: {env_dir}")
        return env_dir
    bundled = bundled_benchmark_asset("physvidbench", DEFAULT_CAPTIONS_DIR_REL)
    if bundled.is_dir():
        return bundled
    root = repo_root or resolve_physvidbench_root()
    if root is None:
        raise FileNotFoundError(
            "PhysVidBench captions directory is missing. Set WORLDFOUNDRY_PHYSVIDBENCH_CAPTIONS_DIR "
            "or WORLDFOUNDRY_PHYSVIDBENCH_ROOT."
        )
    candidate = root / DEFAULT_CAPTIONS_DIR_REL
    if not candidate.is_dir():
        raise FileNotFoundError(f"PhysVidBench captions directory not found: {candidate}")
    return candidate


def resolve_caption_base(
    *,
    explicit: Path | None = None,
    captions_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_base = resolve_env_path("WORLDFOUNDRY_PHYSVIDBENCH_CAPTION_BASE")
    if env_base is not None:
        return env_base
    root = repo_root or resolve_physvidbench_root()
    if root is not None and (root / DEFAULT_CAPTION_BASE_REL.parent).is_dir():
        return (root / DEFAULT_CAPTION_BASE_REL).resolve()
    directory = captions_dir or resolve_captions_dir(repo_root=root)
    return (directory / "cogvideo2b").resolve()


def caption_file_paths(caption_base: Path, suffixes: tuple[str, ...] = CAPTION_SUFFIXES) -> list[Path]:
    return [Path(f"{caption_base}{suffix}.txt") for suffix in suffixes]


def load_caption_matrix(
    caption_base: Path,
    *,
    suffixes: tuple[str, ...] = CAPTION_SUFFIXES,
) -> list[list[str]]:
    """Load the 8 caption tracks indexed by PromptID line number."""
    matrix: list[list[str]] = []
    for path in caption_file_paths(caption_base, suffixes):
        if not path.is_file():
            raise FileNotFoundError(f"PhysVidBench caption file not found: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        if not lines and path.read_text(encoding="utf-8").strip():
            lines = [path.read_text(encoding="utf-8").strip()]
        matrix.append(lines)
    if not matrix:
        raise ValueError(f"No PhysVidBench caption tracks found for base: {caption_base}")
    expected = max(len(track) for track in matrix)
    for index, track in enumerate(matrix):
        if len(track) < expected:
            raise ValueError(
                f"Caption track {caption_file_paths(caption_base, suffixes)[index]} has "
                f"{len(track)} lines; expected at least {expected}."
            )
    return matrix


def captions_for_prompt_id(caption_matrix: list[list[str]], prompt_id: int) -> list[str]:
    if prompt_id < 0 or prompt_id >= len(caption_matrix[0]):
        raise IndexError(f"PromptID {prompt_id} is outside caption line range 0..{len(caption_matrix[0]) - 1}")
    return [track[prompt_id] for track in caption_matrix]
