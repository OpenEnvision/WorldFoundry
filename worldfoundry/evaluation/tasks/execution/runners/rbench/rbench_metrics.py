"""RBench metric formulas, reproduced from the official ReVidgen summary scripts.

Two independent aggregation chains are implemented, matching upstream exactly:

**4 embodiments** (``eval/4_embodiments/8_summarize_robot_results.py`` →
``summarize_i2v_results.py`` → ``summary_scores.py``)
    Per video, merge the three VQA scores with the two motion-operator scores, min-max
    normalize each to ``[0, 1]``, then derive ``Task_Completion`` and a penalized
    ``Visual_Quality``. Per-embodiment values are column means; the benchmark row is the
    mean over embodiments; ``embodiment_overall`` pools ``Task_Completion`` and
    ``Visual_Quality`` across the four embodiment rows.

**5 tasks** (``eval/5_tasks/summary_scores.py``)
    Per video, take the rubric ``score``, drop negatives (the upstream bad-reply
    sentinel), rescale ``(score - 1) / 4``, average per task, and pool *per video* — not
    per task — for ``task_track_overall``.

Upstream runs on pandas; the NaN semantics that behaviour depends on are reproduced here
explicitly, because they change the numbers:

* ``clean_dataframe`` keeps only rows whose ``name`` is all digits.
* Non-numeric scores ("bad reply") become missing, not zero.
* Row means skip missing components; a row with only ``TAC`` scores on ``TAC`` alone.
* ``df[df["MA"] >= 0.0]`` drops rows with a missing amplitude, because ``NaN >= 0`` is
  false — those videos leave the penalized aggregate entirely.
* ``Visual_Quality_Base`` propagates missing values, so one missing operator score voids
  the row's visual quality rather than deflating it.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_prompts import (
    EMBODIMENT_ORDER,
    EMBODIMENT_TRACK,
    TASK_METRIC_IDS,
    TASK_ORDER,
    TASK_TRACK,
)

JsonValue = Any

NUMERIC_NAME_RE = re.compile(r"^\d+$")

# ``scaling`` in 8_summarize_robot_results.py — (min, max) per raw component.
COMPONENT_SCALING: Mapping[str, tuple[float, float]] = {
    "PSS": (1.0, 5.0),
    "TAC": (1.0, 5.0),
    "RSS": (1.0, 15.0),
    "MS": (0.0, 1.0),
    "MA": (0.0, 1.0),
}

# ``consistent_penalty`` letter penalties.
CONSISTENCY_PENALTIES: Mapping[str, float] = {"B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8}

# ``pas_penalty`` thresholds.
PAS_THRESHOLD = 0.1
PAS_LOW_THRESHOLD = 0.05
PAS_DELTA = 0.1

# ``Visual_Quality_Base`` weights.
RSS_WEIGHT = 0.8
MS_WEIGHT = 0.2

# Upstream ``amplitude_threshold`` default in filter_and_aggregate.
AMPLITUDE_THRESHOLD = 0.0

COMPONENT_METRIC_IDS: Mapping[str, str] = {
    "PSS": "physical_plausibility",
    "TAC": "task_adherence_consistency",
    "RSS": "robot_subject_stability",
    "MS": "motion_smoothness",
    "MA": "motion_amplitude",
}

DERIVED_METRIC_IDS: Mapping[str, str] = {
    "Task_Completion": "task_completion",
    "Visual_Quality_Base": "visual_quality_base",
    "Visual_Quality": "visual_quality",
}

EMBODIMENT_ROW_COLUMNS: tuple[str, ...] = (
    "PSS",
    "TAC",
    "RSS",
    "MS",
    "MA",
    "Task_Completion",
    "Visual_Quality_Base",
    "Visual_Quality",
)

METRIC_ORDER = (
    "rbench_overall",
    "embodiment_overall",
    "task_completion",
    "visual_quality",
    "visual_quality_base",
    "physical_plausibility",
    "task_adherence_consistency",
    "robot_subject_stability",
    "motion_smoothness",
    "motion_amplitude",
    "task_track_overall",
    "common_manipulation",
    "long_horizon_planning",
    "multi_entity_collaboration",
    "spatial_relationship",
    "visual_reasoning",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "rbench_overall": {
        "name": "RBench Overall",
        "group": "aggregate",
        "higher_is_better": True,
        "description": (
            "WorldFoundry composite: unweighted mean of embodiment_overall and "
            "task_track_overall. RBench publishes the two tracks separately and does not "
            "define a single cross-track number; emitted only when both tracks are complete."
        ),
        "primary": True,
        "worldfoundry_defined": True,
    },
    "embodiment_overall": {
        "name": "Embodiment Track Overall",
        "group": "embodiment_aggregate",
        "higher_is_better": True,
        "description": "Mean of Task_Completion and Visual_Quality pooled across the four embodiment rows.",
    },
    "task_completion": {
        "name": "Task Completion",
        "group": "embodiment_aggregate",
        "higher_is_better": True,
        "description": "Mean of normalized physical plausibility and task adherence consistency.",
    },
    "visual_quality": {
        "name": "Visual Quality",
        "group": "embodiment_aggregate",
        "higher_is_better": True,
        "description": "Visual_Quality_Base minus amplitude and stability-consistency penalties, floored at 0.",
    },
    "visual_quality_base": {
        "name": "Visual Quality (Unpenalized)",
        "group": "embodiment_aggregate",
        "higher_is_better": True,
        "description": "0.8 x normalized robot subject stability + 0.2 x motion smoothness.",
    },
    "physical_plausibility": {
        "name": "Physical Plausibility (PSS)",
        "group": "embodiment_component",
        "higher_is_better": True,
        "description": "VLM physical plausibility score, normalized from the 1-5 scale.",
    },
    "task_adherence_consistency": {
        "name": "Task Adherence Consistency (TAC)",
        "group": "embodiment_component",
        "higher_is_better": True,
        "description": "VLM task adherence score, normalized from the 1-5 scale.",
    },
    "robot_subject_stability": {
        "name": "Robot Subject Stability (RSS)",
        "group": "embodiment_component",
        "higher_is_better": True,
        "description": "VLM robot/object stability score, normalized from the 1-15 option-mapped scale.",
    },
    "motion_smoothness": {
        "name": "Motion Smoothness (MS)",
        "group": "embodiment_component",
        "higher_is_better": True,
        "description": "Motion smoothness operator score.",
    },
    "motion_amplitude": {
        "name": "Perceptible Motion Amplitude (MA)",
        "group": "embodiment_component",
        "higher_is_better": True,
        "description": "Robotic-manipulator perceptible amplitude operator score; drives the PAS penalty.",
    },
    "task_track_overall": {
        "name": "Task Track Overall",
        "group": "task_aggregate",
        "higher_is_better": True,
        "description": "ALL_TASKS_MEAN: mean over every scored video pooled across the five task splits.",
    },
    "common_manipulation": {
        "name": "Common Manipulation",
        "group": "task_component",
        "higher_is_better": True,
        "description": "Normalized rubric score for the common manipulation split.",
    },
    "long_horizon_planning": {
        "name": "Long-Horizon Planning",
        "group": "task_component",
        "higher_is_better": True,
        "description": "Normalized rubric score for the long-horizon planning split.",
    },
    "multi_entity_collaboration": {
        "name": "Multi-Entity Collaboration",
        "group": "task_component",
        "higher_is_better": True,
        "description": "Normalized rubric score for the multi-entity collaboration split.",
    },
    "spatial_relationship": {
        "name": "Spatial Relationship",
        "group": "task_component",
        "higher_is_better": True,
        "description": "Normalized rubric score for the spatial relationship split.",
    },
    "visual_reasoning": {
        "name": "Visual Reasoning",
        "group": "task_component",
        "higher_is_better": True,
        "description": "Normalized rubric score for the visual reasoning split.",
    },
}

EMBODIMENT_METRIC_IDS: tuple[str, ...] = (
    "embodiment_overall",
    "task_completion",
    "visual_quality",
    "visual_quality_base",
    "physical_plausibility",
    "task_adherence_consistency",
    "robot_subject_stability",
    "motion_smoothness",
    "motion_amplitude",
)

TASK_TRACK_METRIC_IDS: tuple[str, ...] = ("task_track_overall", *TASK_METRIC_IDS.values())

# Upstream VQA subdirectory -> merged column name (csv_info in 8_summarize_robot_results.py).
VQA_CSV_COLUMNS: Mapping[str, str] = {
    "1_robot_subject_stability": "RSS",
    "2_physical_plausibility": "PSS",
    "3_task_adherence_consistency": "TAC",
}


# ---------------------------------------------------------------------------
# Missing-value helpers (pandas skipna semantics without the dependency)
# ---------------------------------------------------------------------------


def _to_number(value: JsonValue) -> float | None:
    """Coerce to float, mapping non-numeric upstream sentinels to missing."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number != number else number


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean over present values only; ``None`` when nothing is present."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _clean_option(value: JsonValue) -> str | None:
    """Normalize a stability option cell, dropping the upstream bad-reply marker."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "bad reply" in text.lower():
        return None
    return text


# ---------------------------------------------------------------------------
# Official penalty functions
# ---------------------------------------------------------------------------


def pas_penalty(
    amplitude: float,
    *,
    threshold: float = PAS_THRESHOLD,
    low_threshold: float = PAS_LOW_THRESHOLD,
    delta: float = PAS_DELTA,
) -> float:
    """Penalize near-static videos (``pas_penalty`` in 8_summarize_robot_results.py)."""
    if amplitude < low_threshold:
        return (threshold - amplitude) + delta
    if amplitude < threshold:
        return threshold - amplitude
    return 0.0


def _option_penalty(option: str | None) -> float:
    if not isinstance(option, str):
        return 0.0
    value = option.strip().upper()
    base = value[:-1] if value and value[-1].isdigit() else value
    return CONSISTENCY_PENALTIES.get(base, 0.0)


def consistency_penalty(robot_option: str | None = None, object_option: str | None = None) -> float:
    """Penalize inconsistent robot/object stability options.

    Mirrors ``consistent_penalty``: the two penalties are averaged only when *both* are
    non-zero; otherwise the robot penalty alone applies.
    """
    robot = _option_penalty(robot_option)
    obj = _option_penalty(object_option)
    if robot and obj:
        return (robot + obj) / 2
    return robot


def normalize_component(component: str, value: float | None) -> float | None:
    """Min-max normalize one raw component into ``[0, 1]``."""
    if value is None:
        return None
    minimum, maximum = COMPONENT_SCALING[component]
    scaled = (value - minimum) / (maximum - minimum)
    return min(max(scaled, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Embodiment track
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _video_key(value: JsonValue) -> str | None:
    """Return the numeric video stem upstream's ``clean_dataframe`` would keep."""
    if value is None:
        return None
    stem = Path(str(value)).stem
    return stem if NUMERIC_NAME_RE.match(stem) else None


def load_embodiment_rows(split_dir: Path, *, vlm_backend: str) -> dict[str, dict[str, Any]]:
    """Merge one embodiment's VQA CSVs and motion JSON into per-video raw rows.

    Args:
        split_dir: ``results/4_embodiments/<model>/<embodiment>/``.
        vlm_backend: VQA judge subdirectory to read (``gpt``, ``qwen_local``, ``qwen_api``).
    """
    rows: dict[str, dict[str, Any]] = {}
    vqa_root = Path(split_dir) / "VQA" / vlm_backend
    for subdir, column in VQA_CSV_COLUMNS.items():
        csv_path = vqa_root / subdir / "results.csv"
        if not csv_path.is_file():
            continue
        for record in _read_csv_rows(csv_path):
            key = _video_key(record.get("name"))
            if key is None:
                continue
            row = rows.setdefault(key, {"name": key})
            row[column] = _to_number(record.get("score"))
            if column == "RSS" and "option" in record:
                option = _clean_option(record.get("option"))
                parts = [part.strip() for part in option.split(",")] if option else []
                row["Stable_Robo"] = parts[0] if parts else None
                row["Stable_Object"] = parts[1] if len(parts) > 1 else None

    motion_path = Path(split_dir) / "motion" / "results.json"
    if motion_path.is_file():
        payload = json.loads(motion_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                key = _video_key(item.get("index"))
                if key is None:
                    continue
                row = rows.setdefault(key, {"name": key})
                row["MA"] = _to_number(item.get("perceptible_amplitude_robotic_manipulator"))
                row["MS"] = _to_number(item.get("motion_smoothness_score"))
    return rows


def normalize_embodiment_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize components and derive Task_Completion / Visual_Quality for one video."""
    normalized: dict[str, Any] = {
        "name": row.get("name"),
        "Stable_Robo": row.get("Stable_Robo"),
        "Stable_Object": row.get("Stable_Object"),
    }
    for component in COMPONENT_SCALING:
        normalized[component] = normalize_component(component, row.get(component))

    normalized["Task_Completion"] = _mean((normalized["PSS"], normalized["TAC"]))

    rss, motion_smoothness = normalized["RSS"], normalized["MS"]
    if rss is None or motion_smoothness is None:
        # Upstream arithmetic propagates NaN here; a partial row has no visual quality.
        normalized["Visual_Quality_Base"] = None
        normalized["Visual_Quality"] = None
        return normalized

    base = rss * RSS_WEIGHT + motion_smoothness * MS_WEIGHT
    normalized["Visual_Quality_Base"] = base
    amplitude = normalized["MA"]
    penalty = consistency_penalty(normalized["Stable_Robo"], normalized["Stable_Object"])
    if amplitude is not None:
        penalty += pas_penalty(amplitude)
    normalized["Visual_Quality"] = max(base - penalty, 0.0)
    return normalized


def aggregate_embodiment_split(
    split_dir: Path,
    *,
    vlm_backend: str,
    amplitude_threshold: float = AMPLITUDE_THRESHOLD,
) -> dict[str, Any] | None:
    """Aggregate one embodiment into the upstream MEAN row.

    Returns ``None`` when the split has no usable rows.
    """
    raw_rows = load_embodiment_rows(Path(split_dir), vlm_backend=vlm_backend)
    if not raw_rows:
        return None
    normalized = [normalize_embodiment_row(row) for _, row in sorted(raw_rows.items())]
    # ``df[df["MA"] >= threshold]``: a missing amplitude fails the comparison and is dropped.
    retained = [row for row in normalized if row["MA"] is not None and row["MA"] >= amplitude_threshold]
    if not retained:
        return None
    means = {column: _mean(row[column] for row in retained) for column in EMBODIMENT_ROW_COLUMNS}
    return {
        "means": means,
        "video_count": len(normalized),
        "retained_video_count": len(retained),
        "dropped_by_amplitude_filter": len(normalized) - len(retained),
        "per_video": normalized,
    }


def aggregate_embodiment_track(
    track_dir: Path,
    *,
    vlm_backend: str,
    embodiments: Sequence[str] = EMBODIMENT_ORDER,
) -> dict[str, Any]:
    """Aggregate the four embodiments into per-split rows, TOTAL_MEAN, and overall_mean."""
    per_split: dict[str, Any] = {}
    for split_id in embodiments:
        split_dir = Path(track_dir) / split_id
        if not split_dir.is_dir():
            continue
        summary = aggregate_embodiment_split(split_dir, vlm_backend=vlm_backend)
        if summary is not None:
            per_split[split_id] = summary

    total_mean = {
        column: _mean(summary["means"][column] for summary in per_split.values())
        for column in EMBODIMENT_ROW_COLUMNS
    }

    # ``overall_mean``: Task_Completion and Visual_Quality pooled over embodiment rows.
    pooled: list[float] = []
    for summary in per_split.values():
        for column in ("Task_Completion", "Visual_Quality"):
            value = summary["means"][column]
            if value is not None:
                pooled.append(value)

    metrics: dict[str, float | None] = {
        metric_id: total_mean[column] for column, metric_id in COMPONENT_METRIC_IDS.items()
    }
    metrics.update({metric_id: total_mean[column] for column, metric_id in DERIVED_METRIC_IDS.items()})
    metrics["embodiment_overall"] = _mean(pooled) if pooled else None

    return {
        "metrics": metrics,
        "total_mean": total_mean,
        "per_split": per_split,
        "scored_splits": sorted(per_split),
        "complete": len(per_split) == len(embodiments),
        "vlm_backend": vlm_backend,
    }


# ---------------------------------------------------------------------------
# Task track
# ---------------------------------------------------------------------------


def normalize_task_score(value: JsonValue) -> float | None:
    """Rescale a 1-5 rubric score to ``[0, 1]``, dropping the negative bad-reply sentinel."""
    number = _to_number(value)
    if number is None or number < 0:
        return None
    return (number - 1.0) / 4.0


def aggregate_task_split(split_dir: Path, *, vlm_backend: str) -> dict[str, Any] | None:
    """Aggregate one task split's ``results.csv`` into a mean and its per-video scores."""
    csv_path = Path(split_dir) / vlm_backend / "results.csv"
    if not csv_path.is_file():
        return None
    scores: list[float] = []
    dropped = 0
    for record in _read_csv_rows(csv_path):
        normalized = normalize_task_score(record.get("score"))
        if normalized is None:
            dropped += 1
            continue
        scores.append(normalized)
    if not scores:
        return None
    return {
        "mean_score": sum(scores) / len(scores),
        "video_count": len(scores),
        "dropped_video_count": dropped,
        "scores": scores,
    }


def aggregate_task_track(
    track_dir: Path,
    *,
    vlm_backend: str,
    tasks: Sequence[str] = TASK_ORDER,
) -> dict[str, Any]:
    """Aggregate the five task splits, pooling per-video scores for ALL_TASKS_MEAN."""
    per_split: dict[str, Any] = {}
    pooled: list[float] = []
    for split_id in tasks:
        split_dir = Path(track_dir) / split_id
        if not split_dir.is_dir():
            continue
        summary = aggregate_task_split(split_dir, vlm_backend=vlm_backend)
        if summary is None:
            continue
        per_split[split_id] = summary
        pooled.extend(summary["scores"])

    metrics: dict[str, float | None] = {
        metric_id: (per_split[split_id]["mean_score"] if split_id in per_split else None)
        for split_id, metric_id in TASK_METRIC_IDS.items()
    }
    metrics["task_track_overall"] = sum(pooled) / len(pooled) if pooled else None

    return {
        "metrics": metrics,
        "per_split": {
            split_id: {key: value for key, value in summary.items() if key != "scores"}
            for split_id, summary in per_split.items()
        },
        "scored_splits": sorted(per_split),
        "pooled_video_count": len(pooled),
        "complete": len(per_split) == len(tasks),
        "vlm_backend": vlm_backend,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def compute_rbench_metrics(
    *,
    embodiment_track_dir: Path | None,
    task_track_dir: Path | None,
    vlm_backend: str,
) -> dict[str, Any]:
    """Compute the full RBench metric table from one model's official results."""
    metrics: dict[str, float | None] = {metric_id: None for metric_id in METRIC_ORDER}
    embodiment = (
        aggregate_embodiment_track(embodiment_track_dir, vlm_backend=vlm_backend)
        if embodiment_track_dir is not None
        else None
    )
    task = (
        aggregate_task_track(task_track_dir, vlm_backend=vlm_backend)
        if task_track_dir is not None
        else None
    )
    if embodiment is not None:
        metrics.update(embodiment["metrics"])
    if task is not None:
        metrics.update(task["metrics"])

    # WorldFoundry composite: only when both official tracks are complete, so a
    # single-track run can never look like a whole-benchmark number.
    both_complete = bool(
        embodiment
        and task
        and embodiment["complete"]
        and task["complete"]
        and metrics.get("embodiment_overall") is not None
        and metrics.get("task_track_overall") is not None
    )
    metrics["rbench_overall"] = (
        (metrics["embodiment_overall"] + metrics["task_track_overall"]) / 2 if both_complete else None
    )

    return {
        "metrics": metrics,
        "embodiment_track": embodiment,
        "task_track": task,
        "components": {
            "vlm_backend": vlm_backend,
            "embodiment_track_complete": bool(embodiment and embodiment["complete"]),
            "task_track_complete": bool(task and task["complete"]),
            "scored_embodiments": list(embodiment["scored_splits"]) if embodiment else [],
            "scored_tasks": list(task["scored_splits"]) if task else [],
            "both_tracks_complete": both_complete,
        },
    }


def track_for_metric(metric_id: str) -> str | None:
    """Return the RBench track a metric belongs to, if it is track-specific."""
    if metric_id in EMBODIMENT_METRIC_IDS:
        return EMBODIMENT_TRACK
    if metric_id in TASK_TRACK_METRIC_IDS:
        return TASK_TRACK
    return None


__all__ = [
    "AMPLITUDE_THRESHOLD",
    "COMPONENT_METRIC_IDS",
    "COMPONENT_SCALING",
    "CONSISTENCY_PENALTIES",
    "DERIVED_METRIC_IDS",
    "EMBODIMENT_METRIC_IDS",
    "EMBODIMENT_ROW_COLUMNS",
    "METRIC_ORDER",
    "METRIC_SPECS",
    "TASK_TRACK_METRIC_IDS",
    "VQA_CSV_COLUMNS",
    "aggregate_embodiment_split",
    "aggregate_embodiment_track",
    "aggregate_task_split",
    "aggregate_task_track",
    "compute_rbench_metrics",
    "consistency_penalty",
    "load_embodiment_rows",
    "normalize_component",
    "normalize_embodiment_row",
    "normalize_task_score",
    "pas_penalty",
    "track_for_metric",
]
