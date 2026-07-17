#!/usr/bin/env python3
"""CLI and scorecard normalization for the in-tree SHI-Labs PAI-Bench runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from worldfoundry.evaluation.tasks.execution.framework.io import env_path, utc_now_iso, write_json, write_jsonl

from .runtime.physical_ai_bench.conditional_generation import (
    CONDITIONAL_METRICS,
    ConditionalGenerationRequest,
    evaluate_conditional_generation,
)
from .runtime.physical_ai_bench.generation import GENERATION_METRICS, GenerationRequest, evaluate_generation

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
TRACKS = ("generation", "conditional-generation")
METRIC_SPECS: dict[str, tuple[str, bool, str]] = {
    "aesthetic_quality": ("Aesthetic quality", True, "generation"),
    "background_consistency": ("Background consistency", True, "generation"),
    "imaging_quality": ("Imaging quality", True, "generation"),
    "motion_smoothness": ("Motion smoothness", True, "generation"),
    "overall_consistency": ("Overall consistency", True, "generation"),
    "subject_consistency": ("Subject consistency", True, "generation"),
    "i2v_background": ("I2V background", True, "generation"),
    "i2v_subject": ("I2V subject", True, "generation"),
    "vqa_accuracy": ("Physical binary VQA accuracy", True, "generation"),
    "dover_tech_score": ("DOVER technical score", True, "conditional-generation"),
    "blur_ssim": ("Blur SSIM", True, "conditional-generation"),
    "canny_f1_score": ("Canny F1", True, "conditional-generation"),
    "canny_precision": ("Canny precision", True, "conditional-generation"),
    "canny_recall": ("Canny recall", True, "conditional-generation"),
    "depth_si_rmse": ("Depth SI-RMSE", False, "conditional-generation"),
    "seg_m_iou": ("Segmentation mIoU", True, "conditional-generation"),
    "seg_recall": ("Segmentation recall", True, "conditional-generation"),
    "lpips_diversity": ("LPIPS diversity", True, "conditional-generation"),
}
ALIASES = {
    "overall_accuracy": "vqa_accuracy",
    "vqa_overall_accuracy": "vqa_accuracy",
    "dover_technical_score": "dover_tech_score",
    "technical_score": "dover_tech_score",
    "depth_sirmse": "depth_si_rmse",
    "seg_miou": "seg_m_iou",
    "segmentation_miou": "seg_m_iou",
    "segmentation_recall": "seg_recall",
    "diversity": "lpips_diversity",
}


def canonical_metric(value: Any) -> str:
    key = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")
    return ALIASES.get(key, key)


def split_metrics(values: Iterable[str] | None, track: str) -> tuple[str, ...]:
    metrics: list[str] = []
    for value in values or ():
        for item in str(value).replace(",", " ").split():
            metric = canonical_metric(item)
            if metric and metric not in metrics:
                metrics.append(metric)
    if metrics:
        return tuple(metrics)
    if track == "generation":
        return tuple(GENERATION_METRICS)
    return tuple(CONDITIONAL_METRICS)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            return float(value.rstrip("%"))
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("score", "raw_score", "value", "mean", "average", "accuracy"):
            number = _number(value.get(key))
            if number is not None:
                return number
    return None


def _collect_metrics(payload: Any, result: dict[str, list[float]]) -> None:
    if isinstance(payload, list):
        for value in payload:
            _collect_metrics(value, result)
        return
    if not isinstance(payload, dict):
        return
    if payload.get("schema_version") == SCORECARD_SCHEMA_VERSION:
        per_metric = payload.get("metrics", {}).get("per_metric", {})
        rows = (
            per_metric.values() if isinstance(per_metric, dict) else per_metric if isinstance(per_metric, list) else ()
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric_id = canonical_metric(row.get("metric_id"))
            value = _number(row.get("raw_score", row.get("score", row.get("value"))))
            if metric_id in METRIC_SPECS and value is not None:
                result.setdefault(metric_id, []).append(value)
        return
    global_scores = payload.get("global")
    if isinstance(global_scores, dict):
        _collect_metrics(global_scores, result)
        return
    summary_scores = payload.get("summary")
    if isinstance(summary_scores, dict) and any(canonical_metric(key) in METRIC_SPECS for key in summary_scores):
        _collect_metrics(summary_scores, result)
        return
    metric_id = canonical_metric(payload.get("metric_id") or payload.get("metric") or "")
    if metric_id in METRIC_SPECS:
        value = _number(payload)
        if value is not None:
            result.setdefault(metric_id, []).append(value)
    for key, value in payload.items():
        canonical = canonical_metric(key)
        if canonical in METRIC_SPECS:
            number = _number(value)
            if number is not None:
                result.setdefault(canonical, []).append(number)
                continue
        if key in {
            "summary",
            "metrics",
            "leaderboard",
            "leaderboard_metrics",
            "per_metric",
            "category_scores",
            "results",
            "scores",
            "aggregate",
            "global",
            "per_video",
            "detailed_results",
            "samples",
        }:
            _collect_metrics(value, result)


def _load_results(path: Path) -> tuple[list[Any], list[Path]]:
    if path.is_file():
        files = [path]
    else:
        preferred = [
            path / name
            for name in (
                "official_results.json",
                "metrics.json",
                "results.json",
                "scores.json",
                "scorecard.json",
            )
            if (path / name).is_file()
        ]
        files = preferred[:1] or sorted(
            item for item in path.rglob("*") if item.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}
        )
    if not files:
        raise FileNotFoundError(f"no PAI-Bench result files found: {path}")
    payloads: list[Any] = []
    for file in files:
        if file.suffix.lower() == ".json":
            payloads.append(json.loads(file.read_text(encoding="utf-8")))
        elif file.suffix.lower() == ".jsonl":
            payloads.append(
                [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            )
        else:
            with file.open(encoding="utf-8-sig", newline="") as handle:
                payloads.append(list(csv.DictReader(handle, delimiter="\t" if file.suffix.lower() == ".tsv" else ",")))
    return payloads, files


def _sample_rows(payloads: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ("samples", "per_sample_scores", "detailed_results", "results"):
            values = payload.get(key)
            if isinstance(values, list):
                rows.extend(dict(value) for value in values if isinstance(value, dict))
    return rows


def _normalized_score(metric_id: str, raw_score: float) -> float | None:
    if metric_id == "dover_tech_score":
        return raw_score / 100.0
    if metric_id == "depth_si_rmse":
        return None
    if 1.0 < raw_score <= 100.0 and metric_id in {
        "vqa_accuracy",
        "aesthetic_quality",
        "background_consistency",
        "imaging_quality",
        "motion_smoothness",
        "overall_consistency",
        "subject_consistency",
        "i2v_background",
        "i2v_subject",
    }:
        return raw_score / 100.0
    return raw_score


def normalize_results(
    *,
    results_path: Path,
    output_dir: Path,
    track: str,
    command: list[str] | None,
    duration_seconds: float | None,
    runtime_executed: bool,
    dataset_root: Path | None,
) -> dict[str, Any]:
    payloads, files = _load_results(results_path)
    extracted: dict[str, list[float]] = {}
    for payload in payloads:
        _collect_metrics(payload, extracted)
    metric_ids = [metric for metric, spec in METRIC_SPECS.items() if spec[2] == track]
    rows: list[dict[str, Any]] = []
    leaderboard: dict[str, float] = {}
    per_metric: dict[str, Any] = {}
    summary: dict[str, float] = {}
    for metric_id in metric_ids:
        values = extracted.get(metric_id, [])
        raw_score = sum(values) / len(values) if values else None
        normalized = _normalized_score(metric_id, raw_score) if raw_score is not None else None
        name, higher_is_better, group = METRIC_SPECS[metric_id]
        row = {
            "metric_id": metric_id,
            "name": name,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalized,
            "higher_is_better": higher_is_better,
            "group": group,
            "source": "physical_ai_bench_in_tree_runtime" if runtime_executed else "official_result_import",
        }
        if raw_score is None:
            row["reason"] = "score_not_found"
        else:
            summary[metric_id] = raw_score
            if normalized is not None:
                leaderboard[metric_id] = normalized
        rows.append(row)
        per_metric[metric_id] = row
    samples = _sample_rows(payloads)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    metric_path = output_dir / "raw_metric_table.jsonl"
    sample_path = output_dir / "per_sample_scores.jsonl"
    write_jsonl(metric_path, rows)
    write_jsonl(sample_path, samples)
    available = sum(bool(row["available"]) for row in rows)
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": "completed" if available else "failed",
            "started_at": utc_now_iso(),
            "runner": "physical_ai_bench_in_tree_runner",
            "command": command,
            "returncode": 0 if available else 1,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": "physical-ai-bench",
            "name": "Physical AI Bench (PAI-Bench)",
            "track": track,
            "contract_only": False,
            "requires_upstream_runtime": False,
            "official_runtime_available": True,
        },
        "dataset": {
            "dataset_root": str(dataset_root.resolve()) if dataset_root else None,
            "upstream_results": str(results_path.resolve()),
            "result_file_count": len(files),
            "sample_count": len(samples),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": ["local evaluation is not an official SHI-Labs leaderboard submission"],
        },
        "generation": {"successful": len(samples), "failed": 0},
        "metrics": {
            "summary": summary,
            "leaderboard": leaderboard,
            "groups": {track: metric_ids},
            "per_metric": per_metric,
        },
        "evaluation": {
            "available": available > 0,
            "kind": "physical_ai_bench_in_tree" if runtime_executed else "physical_ai_bench_result_normalizer",
            "track": track,
            "leaderboard_metrics": leaderboard,
            "num_results": len(samples),
        },
        "validation": {
            "normalizer_only": not runtime_executed,
            "official_runtime_executed": runtime_executed,
            "official_result_shape": {
                "checked": True,
                "ok": available > 0,
                "available_metric_count": available,
                "files": [str(file) for file in files],
            },
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(metric_path.resolve()),
            "per_sample_scores": str(sample_path.resolve()),
            "upstream_results": str(results_path.resolve()),
        },
        "official_benchmark_verified": False,
        "integration_evidence": runtime_executed and available > 0,
        "normalization_ok": available > 0,
        "official_results_imported": not runtime_executed and available > 0,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize the in-tree SHI-Labs PAI-Bench evaluator.")
    parser.add_argument("--benchmark-id", default="physical-ai-bench")
    parser.add_argument(
        "--track", choices=TRACKS, default=os.environ.get("WORLDFOUNDRY_PHYSICAL_AI_BENCH_TRACK", "generation")
    )
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-results-path", "--results-path", dest="results_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--dataset-root", type=Path, default=env_path("WORLDFOUNDRY_PHYSICAL_AI_BENCH_DATASET_ROOT"))
    parser.add_argument("--generated-video-dir", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--reference-image-dir", type=Path)
    parser.add_argument("--vqa-questions-dir", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--metrics", action="append")
    parser.add_argument("--pred-depth-dir", type=Path)
    parser.add_argument("--pred-segmentation-dir", type=Path)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--grounding-checkpoint", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--dover-checkpoint", type=Path)
    parser.add_argument("--allow-trusted-pickle", action="store_true")
    parser.add_argument("--judge-backend", choices=("local-qwen", "openai-compatible"))
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--max-frames", type=int, default=121)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true")
    return parser


def _run_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for raw PAI-Bench evaluation")
    if args.generated_video_dir is None:
        raise ValueError("--generated-video-dir is required for PAI-Bench G/C evaluation")
    metrics = split_metrics(args.metrics, args.track)
    runtime_dir = args.output_dir / "runtime"
    if args.track == "generation":
        return evaluate_generation(
            GenerationRequest(
                dataset_root=args.dataset_root,
                generated_video_dir=args.generated_video_dir,
                output_dir=runtime_dir,
                prompt_file=args.prompt_file,
                reference_image_dir=args.reference_image_dir,
                vqa_questions_dir=args.vqa_questions_dir,
                prediction_manifest=args.prediction_manifest,
                metrics=metrics,
                judge_backend=args.judge_backend,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
                max_frames=args.max_frames,
                limit=args.limit,
            )
        )
    return evaluate_conditional_generation(
        ConditionalGenerationRequest(
            dataset_root=args.dataset_root,
            generated_video_dir=args.generated_video_dir,
            output_dir=runtime_dir,
            metadata_path=args.metadata_path,
            metrics=metrics,
            pred_depth_dir=args.pred_depth_dir,
            pred_segmentation_dir=args.pred_segmentation_dir,
            depth_checkpoint=args.depth_checkpoint,
            grounding_checkpoint=args.grounding_checkpoint,
            sam2_checkpoint=args.sam2_checkpoint,
            dover_checkpoint=args.dover_checkpoint,
            allow_trusted_pickle=args.allow_trusted_pickle,
            max_frames=args.max_frames,
            limit=args.limit,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.run_physical_ai_bench_official_runner",
        *(argv or sys.argv[1:]),
    ]
    started = time.monotonic()
    try:
        runtime_executed = bool(args.run_official or args.results_path is None)
        if runtime_executed:
            payload = _run_runtime(args)
            args.results_path = args.output_dir / "official_results.json"
            write_json(args.results_path, payload)
        if args.results_path is None:
            raise ValueError("--official-results-path is required when runtime execution is disabled")
        scorecard = normalize_results(
            results_path=args.results_path,
            output_dir=args.output_dir,
            track=args.track,
            command=command,
            duration_seconds=time.monotonic() - started,
            runtime_executed=runtime_executed,
            dataset_root=args.dataset_root,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        AssertionError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": "physical-ai-bench",
        "track": args.track,
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"physical-ai-bench/{args.track}: {'ok' if result['ok'] else 'failed'}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
