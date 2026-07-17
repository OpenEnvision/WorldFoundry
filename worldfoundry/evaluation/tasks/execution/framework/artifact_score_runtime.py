"""In-tree artifact-score runtime for official benchmark runners.

This module materializes official-style metric rows from score artifacts already
produced inside WorldFoundry. It is intentionally strict: it does not invent
scores from videos, and it fails when no supported metric artifacts are present.
Heavy benchmark-specific metric models should live under ``worldfoundry.base_models``
or the benchmark runner, then write JSON/JSONL/CSV rows consumed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from worldfoundry.evaluation.tasks.execution.framework.io import scalar_number, write_json

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
COMMON_SCORE_FILENAMES = (
    "results.json",
    "scores.json",
    "metrics.json",
    "scorecard.json",
    "raw_metric_table.jsonl",
    "scores.csv",
    "metrics.csv",
)
METRIC_ALIASES: dict[str, dict[str, str]] = {
    "devil-dynamics": {
        "dynamics_range": "dynamics_range",
        "dynamics_controllability": "dynamics_controllability",
        "dynamics_quality": "dynamics_quality",
        "dynamics_based_quality": "dynamics_quality",
        "naturalness": "dynamics_quality",
        "motion_smoothness": "dynamics_controllability",
        "subject_consistency": "dynamics_range",
        "background_consistency": "dynamics_range",
        "overall": "devil_dynamics_average",
        "average": "devil_dynamics_average",
        "devil_dynamics_average": "devil_dynamics_average",
    },
    "fetv": {
        "static_quality": "static_quality",
        "temporal_quality": "temporal_quality",
        "overall_alignment": "overall_alignment",
        "fine_grained_alignment": "fine_grained_alignment",
        "clip_score": "clip_score",
        "blip_score": "blip_score",
        "fid": "fid",
        "fvd": "fvd",
        "overall": "fetv_average",
        "average": "fetv_average",
        "fetv_average": "fetv_average",
    },
    "vmbench": {
        "pas": "perceptible_amplitude_score",
        "perceptible_amplitude_score": "perceptible_amplitude_score",
        "perceptible_amplitude_socre": "perceptible_amplitude_score",
        "ois": "object_integrity_score",
        "object_integrity_score": "object_integrity_score",
        "tcs": "temporal_coherence_score",
        "temporal_coherence_score": "temporal_coherence_score",
        "cas": "commonsense_adherence_score",
        "commonsense_adherence_score": "commonsense_adherence_score",
        "mss": "motion_smoothness_score",
        "motion_smoothness_score": "motion_smoothness_score",
        "avg": "vmbench_average",
        "average": "vmbench_average",
        "overall": "vmbench_average",
        "total_score": "vmbench_average",
        "vmbench_average": "vmbench_average",
    },
    "worldbench": {
        "foreground_miou": "foreground_miou",
        "foreground_iou": "foreground_miou",
        "miou": "foreground_dice",
        "foreground_dice": "foreground_dice",
        "background_rmse": "background_rmse",
        "bg_rmse": "background_rmse",
        "text_accuracy": "text_based_accuracy",
        "text_based_accuracy": "text_based_accuracy",
        "multiple_choice_accuracy": "multiple_choice_accuracy",
        "binary_accuracy": "binary_accuracy",
    },
    "physical-ai-bench": {
        "aesthetic_quality": "aesthetic_quality",
        "background_consistency": "background_consistency",
        "imaging_quality": "imaging_quality",
        "motion_smoothness": "motion_smoothness",
        "overall_consistency": "overall_consistency",
        "subject_consistency": "subject_consistency",
        "i2v_background": "i2v_background",
        "i2v_subject": "i2v_subject",
        "vqa_accuracy": "vqa_accuracy",
        "overall_accuracy": "vqa_accuracy",
        "dover_tech_score": "dover_tech_score",
        "blur_ssim": "blur_ssim",
        "canny_f1_score": "canny_f1_score",
        "canny_precision": "canny_precision",
        "canny_recall": "canny_recall",
        "depth_si_rmse": "depth_si_rmse",
        "seg_m_iou": "seg_m_iou",
        "seg_recall": "seg_recall",
        "lpips_diversity": "lpips_diversity",
    },
    "camerabench": {
        "camera_motion_average_precision": "camera_motion_average_precision",
        "camera_motion_roc_auc": "camera_motion_roc_auc",
        "vqa_accuracy": "vqa_accuracy",
        "retrieval_accuracy": "retrieval_accuracy",
        "caption_quality": "caption_quality",
        "overall": "camerabench_average",
        "average": "camerabench_average",
        "camerabench_average": "camerabench_average",
    },
}


def canonical_key(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")


def canonical_metric_id(benchmark_id: str, value: Any) -> str | None:
    aliases = METRIC_ALIASES.get(benchmark_id, {})
    key = canonical_key(value)
    if key in aliases:
        return aliases[key]
    return key if key else None


def load_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    raise ValueError(f"unsupported artifact score format: {path}")


def discover_score_files(score_dir: Path) -> list[Path]:
    if score_dir.is_file():
        return [score_dir]
    if not score_dir.is_dir():
        raise FileNotFoundError(f"artifact score directory not found: {score_dir}")
    preferred = [score_dir / name for name in COMMON_SCORE_FILENAMES if (score_dir / name).is_file()]
    if preferred:
        return preferred
    return sorted(path for path in score_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _score_from_mapping(row: dict[str, Any]) -> float | None:
    for key in ("score", "value", "raw_score", "normalized_score", "mean", "average", "Overall Mean", "overall_mean"):
        if key in row:
            score = scalar_number(row.get(key))
            if score is not None:
                return score
    return None


def _rows_from_metric_mapping(benchmark_id: str, payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"schema_version", "run", "benchmark", "artifacts", "eligibility", "validation"}:
            continue
        if isinstance(value, dict):
            metric_id = canonical_metric_id(benchmark_id, value.get("metric_id") or key)
            score = _score_from_mapping(value)
        else:
            metric_id = canonical_metric_id(benchmark_id, key)
            score = scalar_number(value)
        if metric_id and score is not None:
            rows.append({"metric_id": metric_id, "score": score, "source": str(source)})
    return rows


def metric_rows_from_payload(benchmark_id: str, payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            metric_key = item.get("metric_id") or item.get("metric") or item.get("name") or item.get("Key") or item.get("key")
            metric_id = canonical_metric_id(benchmark_id, metric_key)
            score = _score_from_mapping(item)
            if metric_id and score is not None:
                rows.append({**item, "metric_id": metric_id, "score": score, "source": item.get("source") or str(source)})
                continue
            rows.extend(_rows_from_metric_mapping(benchmark_id, item, source))
        return rows

    if isinstance(payload, dict):
        for container_key in ("per_metric", "leaderboard", "metrics", "scores", "results", "summary"):
            nested = payload.get(container_key)
            if isinstance(nested, dict):
                rows = _rows_from_metric_mapping(benchmark_id, nested, source)
                if rows:
                    return rows
            if isinstance(nested, list):
                rows = metric_rows_from_payload(benchmark_id, nested, source)
                if rows:
                    return rows
        return _rows_from_metric_mapping(benchmark_id, payload, source)
    return []


def collect_metric_rows(benchmark_id: str, score_files: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in score_files:
        try:
            payload = load_payload(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows.extend(metric_rows_from_payload(benchmark_id, payload, path))
    return rows


def materialize_artifact_scores(
    *,
    benchmark_id: str,
    score_dir: Path,
    output_path: Path,
    generated_video_dir: Path | None = None,
) -> dict[str, Any]:
    score_files = discover_score_files(score_dir.expanduser().resolve())
    rows = collect_metric_rows(benchmark_id, score_files)
    if not rows:
        raise ValueError(
            "no supported metric rows found. Expected JSON/JSONL/CSV score artifacts with metric_id/name plus "
            "score/value/raw_score fields"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, rows)
    summary = {
        "benchmark_id": benchmark_id,
        "score_dir": str(score_dir),
        "generated_video_dir": None if generated_video_dir is None else str(generated_video_dir),
        "score_files": [str(path) for path in score_files],
        "row_count": len(rows),
        "results_path": str(output_path.resolve()),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize official-style metric rows from in-tree score artifacts.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--generated-video-dir", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = materialize_artifact_scores(
            benchmark_id=args.benchmark_id,
            score_dir=args.score_dir,
            generated_video_dir=args.generated_video_dir,
            output_path=args.output_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(summary["results_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
