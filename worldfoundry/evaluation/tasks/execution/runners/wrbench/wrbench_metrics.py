"""Normalize WRBench D1-D6 tables into WorldFoundry metric rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "d1_cam_prec": {"name": "Requested-camera precision (D1-CamPrec)", "group": "camera"},
    "d1_cam_align_common_yaw": {"name": "Prompt-camera alignment, common yaw", "group": "camera"},
    "d1_cam_align_static_hold": {"name": "Prompt-camera alignment, static hold", "group": "camera"},
    "d2_visual_integrity": {"name": "Visual integrity (D2)", "group": "visible_consistency"},
    "d3_visible_spatial_consistency": {"name": "Visible spatial consistency (D3)", "group": "visible_consistency"},
    "d4_visible_state_consistency": {"name": "Visible state consistency (D4)", "group": "visible_consistency"},
    "d5_reobservation_spatial_consistency": {"name": "Re-observation spatial consistency (D5)", "group": "reobservation"},
    "d6_reobservation_state_consistency": {"name": "Re-observation state consistency (D6)", "group": "reobservation"},
    "wrbench_average": {"name": "WRBench diagnostic average", "group": "aggregate"},
}
METRIC_ORDER = tuple(METRIC_SPECS)

VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "d1_cam_prec": ("D1_CamPrec", "D1_camera_pose", "d1_camera_accuracy", "d1_cam_prec"),
    "d1_cam_align_common_yaw": ("D1_CamAlign_common_yaw", "D1_camalign", "d1_camalign_score", "d1_cam_align_common_yaw"),
    "d1_cam_align_static_hold": ("D1_CamAlign_static_hold", "d1_cam_align_static_hold"),
    "d2_visual_integrity": ("D2", "D2_visual_integrity", "d2_selected_visual_integrity_score", "d2_visual_integrity"),
    "d3_visible_spatial_consistency": ("D3", "D3_spatial_in", "vlm_spatial_fidelity", "d3_visible_spatial_consistency"),
    "d4_visible_state_consistency": ("D4", "D4_state_in", "vlm_state_fidelity", "d4_visible_state_consistency"),
    "d5_reobservation_spatial_consistency": ("D5", "D5_spatial_oov", "vlm_spatial_reasoning", "d5_reobservation_spatial_consistency"),
    "d6_reobservation_state_consistency": ("D6", "D6_state_oov", "vlm_state_reasoning", "d6_reobservation_state_consistency"),
    "wrbench_average": ("Avg", "wrbench_average"),
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_number(row: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value, key
    return None, None


def _count_for(row: Mapping[str, Any], value_key: str | None) -> float:
    candidates = [f"{value_key}_n"] if value_key else []
    candidates.extend(("n_outputs", "n_records", "count"))
    for key in candidates:
        value = _number(row.get(key))
        if value is not None and value > 0:
            return value
    return 1.0


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "records", "models", "results", "scores", "table"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
        if any(key in payload for aliases in VALUE_ALIASES.values() for key in aliases):
            return [dict(payload)]
    return []


def load_wrbench_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Path]:
    """Load a WRBench CSV/JSON/JSONL result file or discover one in a directory."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = (
            path / "main_table.csv",
            path / "wrbench_23model_results.json",
            path / "wrbench_23model_results.csv",
            path / "main_table_summary.json",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if path.is_dir():
            matches = sorted(path.rglob("main_table.csv")) + sorted(path.rglob("wrbench*results*.json"))
            if not matches:
                raise FileNotFoundError(f"no WRBench result table found under {path}")
            path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(f"WRBench results do not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)], None, path
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [dict(row) for row in rows if isinstance(row, Mapping)], None, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload) if isinstance(payload, Mapping) else None
    return _json_rows(payload), metadata, path


def _published_aggregates(metadata: Mapping[str, Any] | None) -> dict[str, tuple[float, float | None]]:
    if not metadata:
        return {}
    checks = metadata.get("aggregate_checks")
    if not isinstance(checks, Mapping):
        return {}
    source_keys = {
        "d1_cam_prec": "D1_CamPrec",
        "d1_cam_align_common_yaw": "D1_CamAlign_common_yaw",
        "d1_cam_align_static_hold": "D1_CamAlign_static_hold",
        "d2_visual_integrity": "D2",
        "d3_visible_spatial_consistency": "D3",
        "d4_visible_state_consistency": "D4",
        "d5_reobservation_spatial_consistency": "D5",
        "d6_reobservation_state_consistency": "D6",
    }
    output: dict[str, tuple[float, float | None]] = {}
    for metric_id, source_key in source_keys.items():
        value = checks.get(source_key)
        if isinstance(value, Mapping):
            score = _number(value.get("weighted_mean_from_rows", value.get("weighted_mean")))
            count = _number(value.get("n"))
            if score is not None:
                output[metric_id] = (score, count)
    return output


def compute_wrbench_metrics(
    *, rows: list[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate official WRBench table columns with their published denominators."""
    published = _published_aggregates(metadata)
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    sources: dict[str, str] = {}

    for metric_id in METRIC_ORDER:
        if metric_id in published:
            score, count = published[metric_id]
            metrics[metric_id] = score
            if count is not None:
                counts[metric_id] = int(count)
            sources[metric_id] = "published_aggregate_checks"
            continue
        weighted: list[tuple[float, float]] = []
        for row in rows:
            score, value_key = _first_number(row, VALUE_ALIASES[metric_id])
            if score is not None:
                weighted.append((score, _count_for(row, value_key)))
        if weighted:
            denominator = sum(weight for _, weight in weighted)
            metrics[metric_id] = sum(score * weight for score, weight in weighted) / denominator
            counts[metric_id] = int(denominator)
            sources[metric_id] = "weighted_result_rows"

    if "wrbench_average" not in metrics:
        core = [
            metrics[key]
            for key in (
                "d1_cam_prec",
                "d2_visual_integrity",
                "d3_visible_spatial_consistency",
                "d4_visible_state_consistency",
                "d5_reobservation_spatial_consistency",
                "d6_reobservation_state_consistency",
            )
            if key in metrics
        ]
        if core:
            metrics["wrbench_average"] = mean(core)
            sources["wrbench_average"] = "mean_available_core_dimensions"

    return {"metrics": metrics, "counts": counts, "sources": sources, "row_count": len(rows)}


__all__ = ["METRIC_ORDER", "METRIC_SPECS", "compute_wrbench_metrics", "load_wrbench_results"]
