"""RBench official-result discovery and layout resolution.

The upstream scripts write two parallel result trees:

    results/4_embodiments/<model>/<embodiment>/VQA/<vlm>/<question>/results.csv
    results/4_embodiments/<model>/<embodiment>/motion/results.json
    results/5_tasks/<model>/<task>/<vlm>/results.csv

WorldFoundry accepts a path at any level of that hierarchy — the ``results/`` root, one
track directory, or a single model directory — and resolves the two per-model track
directories from it. The VLM judge subdirectory is selected explicitly because ``gpt`` and
``qwen_local``/``qwen_api`` are separate judges whose scores are never mixed upstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_prompts import (
    EMBODIMENT_ORDER,
    EMBODIMENT_TRACK,
    TASK_ORDER,
    TASK_TRACK,
    VLM_BACKENDS,
)

OFFICIAL_REPO_URL = "https://github.com/DAGroup-PKU/ReVidgen"
OFFICIAL_DATASET_REPO = "DAGroup-PKU/RBench"


class RBenchLayoutError(ValueError):
    """Raised when an RBench results path cannot be resolved into track directories."""


@dataclass(frozen=True)
class RBenchResultLayout:
    """Resolved per-model track directories for one RBench run."""

    results_root: Path
    embodiment_track_dir: Path | None
    task_track_dir: Path | None
    model_id: str | None
    vlm_backend: str

    @property
    def available_tracks(self) -> tuple[str, ...]:
        tracks: list[str] = []
        if self.embodiment_track_dir is not None:
            tracks.append(EMBODIMENT_TRACK)
        if self.task_track_dir is not None:
            tracks.append(TASK_TRACK)
        return tuple(tracks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_root": str(self.results_root),
            "embodiment_track_dir": None if self.embodiment_track_dir is None else str(self.embodiment_track_dir),
            "task_track_dir": None if self.task_track_dir is None else str(self.task_track_dir),
            "model_id": self.model_id,
            "vlm_backend": self.vlm_backend,
            "available_tracks": list(self.available_tracks),
        }


def _has_any_child(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).is_dir() for name in names)


def _is_embodiment_model_dir(path: Path) -> bool:
    return _has_any_child(path, EMBODIMENT_ORDER)


def _is_task_model_dir(path: Path) -> bool:
    return _has_any_child(path, TASK_ORDER)


def _model_dirs(track_root: Path) -> list[Path]:
    if not track_root.is_dir():
        return []
    return sorted(child for child in track_root.iterdir() if child.is_dir())


def _select_model_dir(track_root: Path, model_id: str | None, predicate) -> Path | None:
    candidates = [child for child in _model_dirs(track_root) if predicate(child)]
    if not candidates:
        return None
    if model_id:
        for child in candidates:
            if child.name == model_id:
                return child
        return None
    if len(candidates) > 1:
        names = ", ".join(child.name for child in candidates)
        raise RBenchLayoutError(
            f"multiple RBench models found under {track_root} ({names}); pass --result-model-id"
        )
    return candidates[0]


def resolve_vlm_backend(explicit: str | None = None) -> str:
    """Resolve the VLM judge subdirectory to read.

    Raises:
        RBenchLayoutError: If the requested backend is not one the upstream scripts write.
    """
    backend = (explicit or os.environ.get("WORLDFOUNDRY_RBENCH_VLM_BACKEND") or "").strip()
    if not backend:
        return ""
    if backend not in VLM_BACKENDS:
        known = ", ".join(VLM_BACKENDS)
        raise RBenchLayoutError(f"unknown RBench VLM backend {backend!r}; known: {known}")
    return backend


def discover_vlm_backends(layout_dirs: list[Path]) -> list[str]:
    """Return the VLM judge directories actually present under the given split dirs."""
    found: set[str] = set()
    for split_dir in layout_dirs:
        if not split_dir.is_dir():
            continue
        vqa_root = split_dir / "VQA"
        search_root = vqa_root if vqa_root.is_dir() else split_dir
        for child in search_root.iterdir():
            if child.is_dir() and child.name in VLM_BACKENDS:
                found.add(child.name)
    return [backend for backend in VLM_BACKENDS if backend in found]


def _split_dirs_for(layout: tuple[Path | None, Path | None]) -> list[Path]:
    embodiment_dir, task_dir = layout
    dirs: list[Path] = []
    if embodiment_dir is not None:
        dirs.extend(embodiment_dir / split for split in EMBODIMENT_ORDER)
    if task_dir is not None:
        dirs.extend(task_dir / split for split in TASK_ORDER)
    return [path for path in dirs if path.is_dir()]


def resolve_layout(
    results_path: Path,
    *,
    model_id: str | None = None,
    vlm_backend: str | None = None,
) -> RBenchResultLayout:
    """Resolve an RBench results path into per-model track directories.

    Accepts the ``results/`` root, a single track directory, or one model directory.

    Raises:
        RBenchLayoutError: If neither track can be located under ``results_path``.
    """
    root = Path(results_path).expanduser().resolve()
    if not root.is_dir():
        raise RBenchLayoutError(f"RBench results path is not a directory: {root}")

    embodiment_dir: Path | None = None
    task_dir: Path | None = None
    resolved_model = model_id

    # A single model directory of either track.
    if _is_embodiment_model_dir(root):
        embodiment_dir, resolved_model = root, resolved_model or root.name
    if _is_task_model_dir(root):
        task_dir, resolved_model = root, resolved_model or root.name

    # A track directory holding model subdirectories.
    if embodiment_dir is None and task_dir is None:
        embodiment_dir = _select_model_dir(root, model_id, _is_embodiment_model_dir)
        task_dir = _select_model_dir(root, model_id, _is_task_model_dir)
        if embodiment_dir is not None:
            resolved_model = resolved_model or embodiment_dir.name
        if task_dir is not None:
            resolved_model = resolved_model or task_dir.name

    # The results/ root holding both track directories.
    if embodiment_dir is None and task_dir is None:
        for track_name, predicate in ((EMBODIMENT_TRACK, _is_embodiment_model_dir), (TASK_TRACK, _is_task_model_dir)):
            track_root = root / track_name
            if not track_root.is_dir():
                continue
            selected = _select_model_dir(track_root, model_id, predicate)
            if selected is None:
                continue
            resolved_model = resolved_model or selected.name
            if track_name == EMBODIMENT_TRACK:
                embodiment_dir = selected
            else:
                task_dir = selected

    if embodiment_dir is None and task_dir is None:
        raise RBenchLayoutError(
            f"no RBench track results found under {root}. Expected 4_embodiments/<model>/<embodiment>/ "
            "or 5_tasks/<model>/<task>/ directories."
        )

    backend = resolve_vlm_backend(vlm_backend)
    if not backend:
        available = discover_vlm_backends(_split_dirs_for((embodiment_dir, task_dir)))
        if not available:
            known = ", ".join(VLM_BACKENDS)
            raise RBenchLayoutError(
                f"no VLM judge directory found under {root}; expected one of: {known}"
            )
        if len(available) > 1:
            names = ", ".join(available)
            raise RBenchLayoutError(
                f"multiple RBench VLM judge outputs found ({names}); pass --vlm-backend to select one"
            )
        backend = available[0]

    return RBenchResultLayout(
        results_root=root,
        embodiment_track_dir=embodiment_dir,
        task_track_dir=task_dir,
        model_id=resolved_model,
        vlm_backend=backend,
    )


def env_results_path() -> Path | None:
    """Return the results path bound through the environment, when it exists."""
    value = os.environ.get("WORLDFOUNDRY_RBENCH_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def env_data_root() -> Path | None:
    """Return the RBench dataset root bound through the environment."""
    for name in (
        "WORLDFOUNDRY_RBENCH_DATA_ROOT",
        "WORLDFOUNDRY_RBENCH_DATASET_ROOT",
        "WORLDFOUNDRY_BENCHMARK_DATA_ROOT",
    ):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            if path.is_dir():
                return path
    return None


__all__ = [
    "OFFICIAL_DATASET_REPO",
    "OFFICIAL_REPO_URL",
    "RBenchLayoutError",
    "RBenchResultLayout",
    "discover_vlm_backends",
    "env_data_root",
    "env_results_path",
    "resolve_layout",
    "resolve_vlm_backend",
]
