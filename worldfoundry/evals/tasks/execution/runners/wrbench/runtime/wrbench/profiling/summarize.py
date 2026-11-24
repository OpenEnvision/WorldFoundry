"""Summarize wrbench generation resource profile JSON files."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_profiles(paths: list[Path]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.resource_profile.json")):
                profiles.append(json.loads(child.read_text(encoding="utf-8")))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    profiles.append(json.loads(line))
        else:
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
    return profiles


def summarize(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for profile in profiles:
        ident = profile.get("run_identity", {})
        status = profile.get("status", {})
        stage = profile.get("stage_summary", {})
        gpu = profile.get("gpu_observation", {})
        derived = profile.get("derived_metrics", {})
        model = str(ident.get("model") or "unknown")
        run_profile = str(ident.get("profile") or "")
        key = (model, run_profile)
        row = groups.setdefault(
            key,
            {
                "model": model,
                "profile": run_profile,
                "rows_total": 0,
                "generated_rows": 0,
                "denominator_rows": 0,
                "sum_output_video_seconds": 0.0,
                "sum_benchmark_generation_seconds": 0.0,
                "sum_benchmark_gpu_seconds": 0.0,
                "sum_preprocess_seconds": 0.0,
                "sum_inference_seconds": 0.0,
                "sum_model_load_seconds": 0.0,
                "peak_memory_mib_max": None,
                "peak_memory_mib_sum_max": None,
                "stage_unavailable_rows": 0,
            },
        )
        row["rows_total"] += 1
        generation_status = str(status.get("generation_status") or ("generated" if status.get("ok") else "failed"))
        counters[key][generation_status] += 1
        if stage.get("stage_status") == "stage_unavailable":
            row["stage_unavailable_rows"] += 1
        row["sum_model_load_seconds"] += float(stage.get("model_load_seconds") or 0.0)
        row["peak_memory_mib_max"] = _max_optional(row["peak_memory_mib_max"], gpu.get("peak_memory_mib_max"))
        row["peak_memory_mib_sum_max"] = _max_optional(row["peak_memory_mib_sum_max"], gpu.get("peak_memory_mib_sum"))

        if generation_status != "generated":
            continue
        row["generated_rows"] += 1
        output_seconds = _float_or_none(derived.get("output_video_seconds"))
        benchmark = _float_or_none(derived.get("benchmark_generation_seconds"))
        gpu_seconds = _float_or_none(derived.get("benchmark_gpu_seconds"))
        if output_seconds is None or output_seconds <= 0 or benchmark is None or gpu_seconds is None:
            continue
        row["denominator_rows"] += 1
        row["sum_output_video_seconds"] += output_seconds
        row["sum_benchmark_generation_seconds"] += benchmark
        row["sum_benchmark_gpu_seconds"] += gpu_seconds
        row["sum_preprocess_seconds"] += float(stage.get("preprocess_seconds") or 0.0)
        row["sum_inference_seconds"] += float(stage.get("inference_seconds") or 0.0)

    out = []
    for key in sorted(groups):
        row = groups[key]
        denom = row["sum_output_video_seconds"]
        row["status_counts"] = dict(sorted(counters[key].items()))
        row["generation_seconds_per_output_second"] = _ratio(row["sum_benchmark_generation_seconds"], denom)
        row["gpu_seconds_per_output_second"] = _ratio(row["sum_benchmark_gpu_seconds"], denom)
        row["preprocess_seconds_per_output_second"] = _ratio(row["sum_preprocess_seconds"], denom)
        row["inference_seconds_per_output_second"] = _ratio(row["sum_inference_seconds"], denom)
        out.append(row)
    return out


def format_rows(rows: list[dict[str, Any]], fmt: str) -> str:
    fields = [
        "model",
        "profile",
        "rows_total",
        "generated_rows",
        "denominator_rows",
        "gpu_seconds_per_output_second",
        "generation_seconds_per_output_second",
        "preprocess_seconds_per_output_second",
        "inference_seconds_per_output_second",
        "sum_model_load_seconds",
        "peak_memory_mib_max",
        "peak_memory_mib_sum_max",
        "stage_unavailable_rows",
        "status_counts",
    ]
    if fmt == "json":
        return json.dumps(rows, indent=2, sort_keys=True)
    if fmt == "csv":
        from io import StringIO

        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["status_counts"] = json.dumps(copy.get("status_counts", {}), sort_keys=True)
            writer.writerow(copy)
        return buf.getvalue()
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif isinstance(value, dict):
                values.append("`" + json.dumps(value, sort_keys=True) + "`")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(value: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return value / denom


def _max_optional(current: Any, candidate: Any) -> float | None:
    value = _float_or_none(candidate)
    if value is None:
        return _float_or_none(current)
    old = _float_or_none(current)
    return value if old is None else max(old, value)
