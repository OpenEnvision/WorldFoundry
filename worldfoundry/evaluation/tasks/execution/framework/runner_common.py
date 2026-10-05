"""Shared constants and helpers for per-benchmark official runners.

Bench-specific runners should import from here instead of redefining
``SCORECARD_SCHEMA_VERSION``, ``VIDEO_SUFFIXES``, ``resolve_env_path``, and
the common imported-results metric-row builder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
VIDEO_EXTENSIONS = VIDEO_SUFFIXES | {".m4v", ".gif"}


def resolve_env_path(name: str) -> Path | None:
    """Resolve a single environment variable to an absolute path, if set."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def iter_video_files(directory: Path) -> list[Path]:
    """Return sorted video files directly under ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def video_stems_in_directory(directory: Path | None) -> set[str]:
    """Return video file stems present in a directory (non-recursive)."""
    if directory is None or not directory.exists():
        return set()
    return {path.stem for path in iter_video_files(directory)}


def build_import_metric_rows(
    *,
    metric_order: Sequence[str],
    metric_specs: Mapping[str, Mapping[str, Any]],
    computed: Mapping[str, Any],
    source_path: Path,
    source_label: str,
    evidence_scope: str = "result_artifact_import_only",
    imported_flag_name: str = "imported_via_run_official",
    imported_flag_value: bool = False,
    reason_template: str = "score_not_available_in_results",
) -> list[dict[str, Any]]:
    """Build normalized per-metric rows from imported upstream results."""
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in metric_order:
        spec = metric_specs[metric_id]
        score = direct_metrics.get(metric_id)
        row = {
            "metric_id": metric_id,
            "name": spec["name"],
            "available": score is not None,
            "raw_score": score,
            "normalized_score": score,
            "score": score,
            "higher_is_better": spec["higher_is_better"],
            "group": spec["group"],
            "source": source_label,
            "source_path": str(source_path),
            "evidence_scope": evidence_scope,
            "components": components,
            "reason": None if score is not None else reason_template,
        }
        row[imported_flag_name] = imported_flag_value
        rows.append(row)
    return rows


def build_video_coverage(expected_prompt_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    """Compare expected prompt ids against video stems in a generated-artifact directory."""
    actual_names = video_stems_in_directory(generated_dir)
    missing = sorted(expected_prompt_ids - actual_names)
    unexpected = sorted(actual_names - expected_prompt_ids)
    matched = sorted(expected_prompt_ids & actual_names)
    return {
        "expected_count": len(expected_prompt_ids),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_prompt_ids) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


__all__ = [
    "SCORECARD_SCHEMA_VERSION",
    "VIDEO_EXTENSIONS",
    "VIDEO_SUFFIXES",
    "build_import_metric_rows",
    "build_video_coverage",
    "iter_video_files",
    "resolve_env_path",
    "video_stems_in_directory",
]
