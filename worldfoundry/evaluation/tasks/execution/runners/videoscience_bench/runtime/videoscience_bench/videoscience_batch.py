#!/usr/bin/env python3
"""Run the in-tree VideoScience VLM judge over generated videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from .judge.api_manager import judge_experiment
    from .judge.vlm_as_a_judge import WEIGHTS, _compute_overall_1to4, _parse_output_text
except ImportError:
    from judge.api_manager import judge_experiment
    from judge.vlm_as_a_judge import WEIGHTS, _compute_overall_1to4, _parse_output_text

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
METRIC_MAP = {
    "prompt_consistency": "prompt_consistency",
    "expected_phenomenon": "phenomenon_congruency",
    "dynamism": "correct_dynamism",
    "immutability": "immutability",
    "coherence": "spatio_temporal_coherence",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_")


def _video_id(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"vid[_-]?0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    if text.isdigit():
        return str(int(text))
    return _slug(text)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _candidate_video_names(video_id: str) -> list[str]:
    if video_id.isdigit():
        raw = int(video_id)
        return [
            f"vid_{raw}_run_1",
            f"vid_{raw:03d}_run_1",
            f"vid_{raw}",
            f"vid_{raw:03d}",
            str(raw),
        ]
    return [video_id]


def _find_video(root: Path, video_id: str) -> Path | None:
    names = set(_candidate_video_names(video_id))
    for suffix in VIDEO_SUFFIXES:
        for name in names:
            direct = root / f"{name}{suffix}"
            if direct.is_file():
                return direct
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        stem = path.stem
        if stem in names:
            return path
        for name in names:
            if stem.startswith(f"{name}_") or stem.endswith(f"_{name}"):
                return path
    return None


def _reference_video(root: Path | None, video_id: str) -> Path | None:
    if root is None:
        return None
    if video_id.isdigit():
        raw = int(video_id)
        candidates = [f"vid_{raw}_ref", f"vid_{raw:03d}_ref", f"vid_{raw}", f"vid_{raw:03d}"]
    else:
        candidates = [video_id]
    for suffix in VIDEO_SUFFIXES:
        for name in candidates:
            path = root / f"{name}{suffix}"
            if path.is_file():
                return path
    return None


def _normalize_1to4(value: Any) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    score = max(1.0, min(4.0, score))
    return (score - 1.0) / 3.0


def _metrics_from_rubric(
    rubric: dict[str, Any],
    *,
    overall_1to4: float | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_key, metric_id in METRIC_MAP.items():
        normalized = _normalize_1to4(rubric.get(source_key))
        if normalized is not None:
            metrics[metric_id] = normalized
    if overall_1to4 is None:
        overall_1to4 = _compute_overall_1to4(rubric, WEIGHTS)
    overall = _normalize_1to4(overall_1to4)
    if overall is not None:
        # Preserve the weighting from the vendored official judge.  A simple
        # mean would incorrectly discard the 30% phenomenon weight.
        metrics["videoscience_average"] = overall
    return metrics


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item in results:
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(str(key), []).append(float(value))
    return {key: avg for key, values in buckets.items() if (avg := _mean(values)) is not None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-tree VideoScience-Bench VLM judge.")
    parser.add_argument("--experiments-csv", type=Path, required=True)
    parser.add_argument("--generated-video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", default=os.environ.get("WORLDFOUNDRY_VIDEOSCIENCE_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.environ.get("WORLDFOUNDRY_VIDEOSCIENCE_MODEL", "gpt-4o"))
    parser.add_argument("--reference-videos-dir", type=Path)
    parser.add_argument("--authors", default=None)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--timeout-s", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_rows(args.experiments_csv)
    authors = {item.strip() for item in args.authors.split(",")} if args.authors else set()
    selected: list[dict[str, str]] = []
    for row in rows:
        if authors and str(row.get("Author", "")).strip() not in authors:
            continue
        if args.ready_only and str(row.get("finalized?", "")).strip().lower() != "done":
            continue
        if not row.get("Prompts") or not row.get("Expected phenomenon"):
            continue
        if _video_id(row.get("Unique ID", "")) is None:
            continue
        selected.append(row)
        if args.max_runs and len(selected) >= args.max_runs:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = args.output_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in selected:
        video_id = _video_id(row.get("Unique ID", ""))
        if video_id is None:
            continue
        video_path = _find_video(args.generated_video_dir, video_id)
        if video_path is None:
            missing.append({"video_id": video_id, "reason": "generated_video_not_found"})
            continue
        ref_path = _reference_video(args.reference_videos_dir, video_id)
        raw = judge_experiment(
            provider=args.provider,
            model=args.model,
            video_path=str(video_path),
            phenomenon=str(row.get("Expected phenomenon", "")),
            gt_description=str(row.get("Prompts", "")),
            ref_video_path=None if ref_path is None else str(ref_path),
            max_frames=args.max_frames,
            fps=args.fps,
            timeout_s=args.timeout_s,
            extra={},
        )
        rubric, explanations = _parse_output_text(str(raw.get("output_text", "")))
        overall_1to4 = _compute_overall_1to4(rubric, WEIGHTS)
        metrics = _metrics_from_rubric(rubric, overall_1to4=overall_1to4)
        item = {
            "video_id": video_id,
            "video": str(video_path),
            "reference_video": None if ref_path is None else str(ref_path),
            "phenomenon": row.get("Expected phenomenon", ""),
            "prompt": row.get("Prompts", ""),
            "provider": raw.get("provider", args.provider),
            "model": raw.get("model", args.model),
            "rubric": {**rubric, "overall_weighted": overall_1to4},
            "metrics": metrics,
            "explanations": explanations,
            "evidence": raw.get("evidence", {}),
            "raw": raw.get("raw", {}),
        }
        (per_sample_dir / f"{video_id}.json").write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(item)

    output = {
        "benchmark": "videoscience-bench",
        "provider": args.provider,
        "model": args.model,
        "experiments_csv": str(args.experiments_csv),
        "generated_video_dir": str(args.generated_video_dir),
        "selected_count": len(selected),
        "evaluated_count": len(results),
        "missing_count": len(missing),
        "missing": missing,
        "metrics": _aggregate(results),
        "results": results,
    }
    out_path = args.output_dir / "videoscience_judge_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
