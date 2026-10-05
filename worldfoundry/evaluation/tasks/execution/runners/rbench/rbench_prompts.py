"""RBench split tables, prompt manifests, and canonical counts.

RBench (released with the ReVidgen project) evaluates image-to-video robotics generation
along two independent tracks:

* **4 embodiments** — ``dual_arm``, ``humanoid``, ``single_arm``, ``quad`` (100 prompts each),
  scored by three VLM questions plus two motion operators.
* **5 tasks** — ``common_manipulation``, ``long-horizon_planning``,
  ``multi-entity_collaboration``, ``spatial_relationship``, ``visual_reasoning``
  (50 prompts each), scored by one rubric VLM call per video.

Split ids, prompt-file names, and the image-grid frame counts match ``scripts/
rbench_eval_4embodiments.sh`` and ``scripts/rbench_eval_5tasks.sh``. Prompt counts were
read from the ``DAGroup-PKU/RBench`` dataset manifests.
"""

from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCHMARK_ID = "rbench"
DISPLAY_NAME = "RBench"

EMBODIMENT_TRACK = "4_embodiments"
TASK_TRACK = "5_tasks"


@dataclass(frozen=True)
class RBenchSplit:
    """One RBench evaluation split (an embodiment or a task category)."""

    split_id: str
    track: str
    display_name: str
    prompt_count: int
    image_grid_frames: int

    @property
    def prompt_filename(self) -> str:
        return f"{self.split_id}_prompts.json"


SPLITS: tuple[RBenchSplit, ...] = (
    RBenchSplit("dual_arm", EMBODIMENT_TRACK, "Dual Arm", 100, 6),
    RBenchSplit("humanoid", EMBODIMENT_TRACK, "Humanoid", 100, 6),
    RBenchSplit("single_arm", EMBODIMENT_TRACK, "Single Arm", 100, 6),
    RBenchSplit("quad", EMBODIMENT_TRACK, "Quadruped", 100, 6),
    RBenchSplit("common_manipulation", TASK_TRACK, "Common Manipulation", 50, 6),
    RBenchSplit("long-horizon_planning", TASK_TRACK, "Long-Horizon Planning", 50, 6),
    RBenchSplit("multi-entity_collaboration", TASK_TRACK, "Multi-Entity Collaboration", 50, 6),
    RBenchSplit("spatial_relationship", TASK_TRACK, "Spatial Relationship", 50, 3),
    RBenchSplit("visual_reasoning", TASK_TRACK, "Visual Reasoning", 50, 6),
)

SPLITS_BY_ID: Mapping[str, RBenchSplit] = {split.split_id: split for split in SPLITS}

# ``robot_types`` in rbench_eval_4embodiments.sh — also the row order of the summary CSV.
EMBODIMENT_ORDER: tuple[str, ...] = ("dual_arm", "humanoid", "single_arm", "quad")

# ``task_cfg`` order in rbench_eval_5tasks.sh.
TASK_ORDER: tuple[str, ...] = (
    "common_manipulation",
    "long-horizon_planning",
    "multi-entity_collaboration",
    "spatial_relationship",
    "visual_reasoning",
)

# Task split id -> metric id (metric ids cannot carry '-').
TASK_METRIC_IDS: Mapping[str, str] = {
    "common_manipulation": "common_manipulation",
    "long-horizon_planning": "long_horizon_planning",
    "multi-entity_collaboration": "multi_entity_collaboration",
    "spatial_relationship": "spatial_relationship",
    "visual_reasoning": "visual_reasoning",
}

# Category-A fields each task rubric averages before clamping against ``total``.
TASK_CATEGORY_A_FIELDS: Mapping[str, tuple[str, str]] = {
    "common_manipulation": ("action_execution", "task_completion"),
    "long-horizon_planning": ("action_execution", "event_completion_ratio"),
    "multi-entity_collaboration": ("action_coordination", "task_completion"),
    "spatial_relationship": ("spatial_relation_accuracy", "manipulation_feasibility"),
    "visual_reasoning": ("action_execution", "visual_reasoning_accuracy"),
}

# VLM judge directory names accepted by the upstream scripts.
VLM_BACKENDS: tuple[str, ...] = ("gpt", "qwen_local", "qwen_api")
DEFAULT_VLM_BACKEND = "gpt"

# ``prefix_tag`` used in upstream summary filenames: qwen_api/qwen_local both write "qwen".
def summary_prefix_tag(vlm_backend: str) -> str:
    """Return the upstream ``score_summary_<tag>.csv`` tag for a VLM backend."""
    return "gpt" if vlm_backend == "gpt" else "qwen"


CANONICAL_EMBODIMENT_PROMPT_COUNT = sum(
    split.prompt_count for split in SPLITS if split.track == EMBODIMENT_TRACK
)
CANONICAL_TASK_PROMPT_COUNT = sum(split.prompt_count for split in SPLITS if split.track == TASK_TRACK)
CANONICAL_PROMPT_COUNT = CANONICAL_EMBODIMENT_PROMPT_COUNT + CANONICAL_TASK_PROMPT_COUNT


_PROMPT_DIR_NAMES = ("prompts", "data/prompts")


def split_for_id(split_id: str) -> RBenchSplit:
    """Return the split table entry for ``split_id``.

    Raises:
        KeyError: If ``split_id`` is not an official RBench split.
    """
    try:
        return SPLITS_BY_ID[split_id]
    except KeyError as exc:
        known = ", ".join(sorted(SPLITS_BY_ID))
        raise KeyError(f"unknown RBench split {split_id!r}; known: {known}") from exc


def splits_for_track(track: str) -> tuple[RBenchSplit, ...]:
    """Return every split belonging to ``track``."""
    order = EMBODIMENT_ORDER if track == EMBODIMENT_TRACK else TASK_ORDER
    return tuple(SPLITS_BY_ID[split_id] for split_id in order)


def resolve_prompt_dir(*, explicit: Path | None = None, data_root: Path | None = None) -> Path | None:
    """Locate the directory holding ``<split>_prompts.json`` manifests."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_dir():
            return path.resolve()
        if path.is_file():
            return path.parent.resolve()
    if data_root is None:
        return None
    root = Path(data_root).expanduser()
    for name in _PROMPT_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate.resolve()
    if any((root / split.prompt_filename).is_file() for split in SPLITS):
        return root.resolve()
    return None


def load_prompt_records(prompt_dir: Path, split_id: str) -> list[dict[str, Any]]:
    """Load one split's prompt manifest.

    Raises:
        FileNotFoundError: If the manifest is not present under ``prompt_dir``.
    """
    path = Path(prompt_dir) / split_for_id(split_id).prompt_filename
    if not path.is_file():
        raise FileNotFoundError(f"RBench prompt manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("prompts", payload.get("records", []))
    if not isinstance(payload, list):
        raise ValueError(f"unsupported RBench prompt manifest shape: {path}")
    return [dict(record) for record in payload if isinstance(record, Mapping)]


def prompt_index(record: Mapping[str, Any]) -> str | None:
    """Return the numeric video stem for a prompt record.

    Prompt records are named ``<split>_0001`` while generated videos and the upstream
    result CSVs use the bare numeric stem ``0001``.
    """
    name = record.get("name")
    if not isinstance(name, str) or "_" not in name:
        return None if not isinstance(name, str) else name
    return name.rsplit("_", 1)[-1]


def expected_video_stems(prompt_dir: Path, split_id: str) -> set[str]:
    """Return the numeric video stems a split expects under ``videos/``."""
    stems = {prompt_index(record) for record in load_prompt_records(prompt_dir, split_id)}
    return {stem for stem in stems if stem}


def video_coverage(*, video_dir: Path | None, expected: Sequence[str]) -> dict[str, Any]:
    """Compare generated video stems against the expected prompt indices."""
    actual: set[str] = set()
    if video_dir is not None and Path(video_dir).is_dir():
        actual = {
            path.stem
            for path in Path(video_dir).iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        }
    expected_set = {str(item) for item in expected}
    missing = sorted(expected_set - actual)
    return {
        "expected_count": len(expected_set),
        "actual_count": len(actual),
        "matched_count": len(expected_set & actual),
        "missing_count": len(missing),
        "unexpected_count": len(sorted(actual - expected_set)),
        "complete": bool(expected_set) and not missing,
        "missing_ids": missing[:50],
    }


__all__ = [
    "BENCHMARK_ID",
    "CANONICAL_EMBODIMENT_PROMPT_COUNT",
    "CANONICAL_PROMPT_COUNT",
    "CANONICAL_TASK_PROMPT_COUNT",
    "DEFAULT_VLM_BACKEND",
    "DISPLAY_NAME",
    "EMBODIMENT_ORDER",
    "EMBODIMENT_TRACK",
    "RBenchSplit",
    "SPLITS",
    "SPLITS_BY_ID",
    "TASK_CATEGORY_A_FIELDS",
    "TASK_METRIC_IDS",
    "TASK_ORDER",
    "TASK_TRACK",
    "VIDEO_SUFFIXES",
    "VLM_BACKENDS",
    "expected_video_stems",
    "load_prompt_records",
    "prompt_index",
    "resolve_prompt_dir",
    "split_for_id",
    "splits_for_track",
    "summary_prefix_tag",
    "video_coverage",
]
