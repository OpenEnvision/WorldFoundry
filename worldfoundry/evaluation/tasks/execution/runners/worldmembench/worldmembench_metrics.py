# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Import the official ZJU-REAL/WorldMemBench per-model metric summaries.

Protocol: https://github.com/ZJU-REAL/WorldMemBench
Scores retain upstream units and directions; there is no cross-metric average.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Metric directory -> (official score field, higher-is-better).
METRIC_SPECS: dict[str, tuple[str, bool]] = {
    "imaging_quality": ("score", True),
    "aesthetic_quality": ("score", True),
    "human_identity": ("score", True),
    "human_appearance": ("score", True),
    "object_geometry": ("score", True),
    "object_appearance": ("score", True),
    "environment_visible_memory": ("score", True),
    "environment_depth_fidelity": ("depth_spearman", True),
    "environment_scene_reconstruction": ("reconstruction_score", True),
    "dynamic_memory": ("score", True),
    "camera_rotation": ("camera_rotation_rpe_path", False),
    "camera_translation": ("camera_translation_rpe", False),
}


def load_official_results(
    path: Path,
    *,
    model_id: str | None = None,
    metrics: tuple[str, ...] = (),
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read compact official summaries without pooling generation models.

    Accept an evaluation output root, its ``metrics`` directory, or one
    ``metrics/<metric>/summary.json`` file. Official case records, when present,
    are copied with their status and detailed scores intact.
    """
    unknown = set(metrics) - METRIC_SPECS.keys()
    if unknown:
        raise ValueError(f"Unknown WorldMemBench metrics: {', '.join(sorted(unknown))}")
    if path.is_file():
        paths = [path]
    else:
        root = path / "metrics" if (path / "metrics").is_dir() else path
        paths = sorted(root.glob("*/summary.json"))
    summaries: dict[str, tuple[Path, dict[str, Any]]] = {}
    for summary_path in paths:
        metric_id = summary_path.parent.name
        if metric_id not in METRIC_SPECS or (metrics and metric_id not in metrics):
            continue
        summaries[metric_id] = (summary_path, json.loads(summary_path.read_text(encoding="utf-8")))
    missing = set(metrics) - summaries.keys()
    if missing:
        raise ValueError(f"Missing requested metric summaries: {', '.join(sorted(missing))}")
    if not summaries:
        raise ValueError(f"No selected WorldMemBench metric summaries under {path}")
    models = {key for _, summary in summaries.values() for key in summary.get("results", {})}
    if model_id is None:
        if len(models) != 1:
            raise ValueError("Pass --model-id to select one official model group; models: " + ", ".join(sorted(models)))
        model_id = next(iter(models))
    if model_id not in models:
        raise ValueError(f"Model {model_id!r} is not present in the selected summaries")

    extracted: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for metric_id, (summary_path, summary) in summaries.items():
        group = summary.get("results", {}).get(model_id)
        if group is None:
            raise ValueError(f"Model {model_id!r} is missing from {summary_path}")
        count = int(group["num_cases"])
        scored = int(group.get("num_scored", count))
        statuses = group.get("status_counts", {"ok": count})
        score_key, _ = METRIC_SPECS[metric_id]
        raw = group.get("scores", {}).get(score_key)
        if raw is not None:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise ValueError(f"Non-finite or non-numeric {metric_id} score in {summary_path}")
            raw = float(raw)
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw,
            "normalized_score": raw,
            "sample_count": scored,
            "source": str(summary_path),
            "model_id": model_id,
            "coverage": {"num_cases": count, "num_scored": scored, "status_counts": statuses},
            "official_scores": group.get("scores", {}),
            "official_groups": [g for g in summary.get("groups", []) if str(g.get("model_id") or "all") == model_id],
        }
        case_path = summary_path.with_name("cases.jsonl")
        if case_path.is_file():
            for line in case_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                if str(case.get("model_id") or "all") == model_id:
                    cases.append({**case, "metric_id": metric_id})
    return model_id, extracted, cases
