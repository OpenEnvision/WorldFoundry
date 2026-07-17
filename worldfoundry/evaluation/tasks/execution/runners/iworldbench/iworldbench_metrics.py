"""iWorld-Bench metric normalization from official report CSV/JSON exports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

COMPONENT_METRIC_ORDER = (
    "image_quality",
    "brightness_consistency",
    "color_temperature_constraint",
    "sharpness_retention",
    "motion_smoothness",
    "trajectory_accuracy",
    "trajectory_tolerance",
    "memory_symmetry",
    "trajectory_alignment",
)

METRIC_ORDER = (
    *COMPONENT_METRIC_ORDER,
    "iworldbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "image_quality": {
        "name": "Image Quality",
        "group": "generation_quality",
        "higher_is_better": True,
    },
    "brightness_consistency": {
        "name": "Brightness Consistency",
        "group": "generation_quality",
        "higher_is_better": True,
    },
    "color_temperature_constraint": {
        "name": "Color Temperature Constraint",
        "group": "generation_quality",
        "higher_is_better": True,
    },
    "sharpness_retention": {
        "name": "Sharpness Retention",
        "group": "generation_quality",
        "higher_is_better": True,
    },
    "motion_smoothness": {
        "name": "Motion Smoothness",
        "group": "generation_quality",
        "higher_is_better": True,
    },
    "trajectory_accuracy": {
        "name": "Trajectory Accuracy",
        "group": "trajectory_following",
        "higher_is_better": True,
    },
    "trajectory_tolerance": {
        "name": "Trajectory Tolerance",
        "group": "trajectory_following",
        "higher_is_better": True,
    },
    "memory_symmetry": {
        "name": "Memory Symmetry",
        "group": "memory_ability",
        "higher_is_better": True,
    },
    "trajectory_alignment": {
        "name": "Trajectory Alignment",
        "group": "trajectory_following",
        "higher_is_better": True,
    },
    "iworldbench_average": {
        "name": "iWorld-Bench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "primary": True,
    },
}

UPSTREAM_METRIC_ALIASES = {
    "image_quality": "image_quality",
    "imaging_quality": "image_quality",
    "brightness_consistency": "brightness_consistency",
    "color_temperature": "color_temperature_constraint",
    "color_temperature_constraint": "color_temperature_constraint",
    "sharpness": "sharpness_retention",
    "sharpness_retention": "sharpness_retention",
    "motion_smoothness": "motion_smoothness",
    "trajectory_accuracy": "trajectory_accuracy",
    "trajectory_tolerance": "trajectory_tolerance",
    "memory_symmetry": "memory_symmetry",
    "memory_mse": "memory_symmetry",
    "trajectory_alignment": "trajectory_alignment",
    "overall": "iworldbench_average",
    "average": "iworldbench_average",
    "mean": "iworldbench_average",
    "iworldbench_average": "iworldbench_average",
}

SCORE_KEYS = (
    "score",
    "value",
    "mean",
    "average",
    "avg",
    "overall",
    "metric_value",
    "normalized_score",
    "accuracy",
)


def _canonical_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _metric_id_from_text(value: Any) -> str | None:
    key = _canonical_key(value)
    if key in UPSTREAM_METRIC_ALIASES:
        return UPSTREAM_METRIC_ALIASES[key]
    for alias, metric_id in UPSTREAM_METRIC_ALIASES.items():
        if alias in key:
            return metric_id
    return None


def _parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if percent else number


def _row_score(row: Mapping[str, Any]) -> float | None:
    for key in SCORE_KEYS:
        if key in row:
            number = _parse_number(row[key])
            if number is not None:
                return number
    values = []
    for key, value in row.items():
        if _canonical_key(key) in {"sample_id", "id", "index", "frame", "video_id"}:
            continue
        number = _parse_number(value)
        if number is not None:
            values.append(number)
    return None if not values else sum(values) / len(values)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "results", "metrics", "samples", "scores"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else csv.Sniffer().sniff(sample or ",").delimiter
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def collect_result_rows(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}
    )
    rows: list[dict[str, Any]] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".json":
                raw_rows = _read_json_rows(file_path)
            elif file_path.suffix.lower() == ".jsonl":
                raw_rows = _read_jsonl_rows(file_path)
            elif file_path.suffix.lower() in {".csv", ".tsv"}:
                raw_rows = _read_csv_rows(file_path)
            else:
                raw_rows = []
        except (OSError, ValueError, json.JSONDecodeError, csv.Error):
            continue
        file_metric_id = _metric_id_from_text(file_path.stem)
        for row in raw_rows:
            metric_id = _metric_id_from_text(row.get("metric_id") or row.get("metric") or row.get("name")) or file_metric_id
            if metric_id is None:
                continue
            score = _row_score(row)
            rows.append(
                {
                    "metric_id": metric_id,
                    "score": score,
                    "source_file": str(file_path),
                    "raw": dict(row),
                }
            )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values]
    return sum(clean) / len(clean) if clean else None


def compute_iworldbench_metrics(
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    parsed_rows = list(rows or [])
    if results_path is not None and not parsed_rows:
        parsed_rows = collect_result_rows(results_path)

    per_metric: dict[str, float | None] = {metric_id: None for metric_id in METRIC_ORDER}
    sample_counts: dict[str, int] = {}

    if parsed_rows and all("metric_id" in row and "score" in row for row in parsed_rows[: min(3, len(parsed_rows))]):
        buckets: dict[str, list[float]] = {}
        for row in parsed_rows:
            metric_id = str(row.get("metric_id") or "")
            score = _parse_number(row.get("score"))
            if metric_id not in METRIC_ORDER or score is None:
                continue
            buckets.setdefault(metric_id, []).append(score)
        for metric_id, values in buckets.items():
            per_metric[metric_id] = _mean(values)
            sample_counts[metric_id] = len(values)
    else:
        collected = collect_result_rows(results_path) if results_path is not None else []
        buckets: dict[str, list[float]] = {}
        for row in collected:
            metric_id = row["metric_id"]
            score = row["score"]
            if score is None:
                continue
            buckets.setdefault(metric_id, []).append(float(score))
        for metric_id, values in buckets.items():
            per_metric[metric_id] = _mean(values)
            sample_counts[metric_id] = len(values)
        parsed_rows = collected

    component_scores = [per_metric[metric_id] for metric_id in COMPONENT_METRIC_ORDER]
    components_complete = all(value is not None for value in component_scores)
    if not components_complete:
        # A component-level report may contain an upstream "average" column,
        # but that value is not the aggregate for the complete iWorld-Bench
        # metric suite.
        per_metric["iworldbench_average"] = None
        sample_counts.pop("iworldbench_average", None)
    elif per_metric.get("iworldbench_average") is None:
        complete_scores = [float(value) for value in component_scores if value is not None]
        per_metric["iworldbench_average"] = sum(complete_scores) / len(complete_scores)
        sample_counts["iworldbench_average"] = len(complete_scores)

    return {
        "metrics": {key: value for key, value in per_metric.items() if value is not None},
        "components": {
            "sample_counts": sample_counts,
            "row_count": len(parsed_rows),
        },
    }


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    if results_path.is_dir():
        return collect_result_rows(results_path)
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        return _read_json_rows(results_path)
    if suffix == ".jsonl":
        return _read_jsonl_rows(results_path)
    if suffix in {".csv", ".tsv"}:
        return _read_csv_rows(results_path)
    raise ValueError(f"Unsupported iWorld-Bench results format: {results_path}")
