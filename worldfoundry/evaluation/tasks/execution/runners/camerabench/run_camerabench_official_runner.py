#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import (  # noqa: E402
    env_path,
    load_json,
    mean_numeric,
    normalize_unit_score,
    scalar_number,
    score_item,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.runners.camerabench.camerabench_metrics import (
    evaluate_camerabench_from_score_dir,
)

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
METRIC_ORDER = (
    "camera_motion_average_precision",
    "camera_motion_roc_auc",
    "camera_vqa_accuracy",
    "camera_retrieval_accuracy",
    "camera_caption_score",
    "camerabench_average",
)
CAMERABENCH_VIDEO_SUFFIXES = frozenset({".gif", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


scalar = partial(scalar_number, dict_keys=("score", "raw_score", "value", "mean", "average", "accuracy"))
mean = mean_numeric


def _strict_enabled(explicit: bool) -> bool:
    if explicit:
        return True
    return str(os.environ.get("WORLDFOUNDRY_CAMERABENCH_STRICT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _candidate_dataset_root(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    return env_path(
        "WORLDFOUNDRY_CAMERABENCH_DATA_ROOT",
        "WORLDFOUNDRY_BENCHMARK_DATA_ROOT",
        "WORLDFOUNDRY_CAMERABENCH_DATASET_ROOT",
    )


def _dataset_coverage(dataset_root: Path | None) -> dict[str, Any]:
    if dataset_root is None:
        return {
            "provided": False,
            "dataset_root": None,
            "complete": False,
            "reason": "dataset_root_not_provided",
        }
    dataset_root = dataset_root.expanduser().resolve()
    test_path = dataset_root / "test.jsonl"
    video_dir = dataset_root / "videos_gif"
    if not test_path.is_file():
        return {
            "provided": True,
            "dataset_root": str(dataset_root),
            "complete": False,
            "reason": "missing_test_jsonl",
        }
    rows = _load_jsonl_objects(test_path)
    expected_stems: set[str] = set()
    for row in rows:
        path_value = row.get("path") or row.get("Video") or row.get("video")
        if path_value:
            expected_stems.add(Path(str(path_value).split("/")[-1]).stem)
    video_paths: list[Path] = []
    if video_dir.is_dir():
        video_paths = [
            path
            for path in sorted(video_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in CAMERABENCH_VIDEO_SUFFIXES
        ]
    video_stems = {path.stem for path in video_paths}
    missing = sorted(expected_stems - video_stems)
    unexpected = sorted(video_stems - expected_stems)
    return {
        "provided": True,
        "dataset_root": str(dataset_root),
        "test_jsonl": str(test_path),
        "video_dir": str(video_dir),
        "test_row_count": len(rows),
        "expected_video_count": len(expected_stems),
        "video_count": len(video_paths),
        "matched_video_count": len(expected_stems & video_stems),
        "missing_video_count": len(missing),
        "unexpected_video_count": len(unexpected),
        "missing_video_stems": missing[:50],
        "unexpected_video_stems": unexpected[:50],
        "complete": bool(rows) and not missing and len(expected_stems) == len(video_stems),
    }


def _score_dir_report(score_dir: Path | None) -> dict[str, Any]:
    if score_dir is None:
        return {"provided": False, "score_dir": None}
    score_dir = score_dir.expanduser().resolve()
    files = [path for path in sorted(score_dir.rglob("*.json")) if path.is_file()] if score_dir.is_dir() else []
    patterns = {
        "binary": sum(1 for path in files if "classification" in path.name or "vqa_scores" in path.name),
        "vqa_retrieval": sum(1 for path in files if "vqa_retrieval" in path.name),
        "caption": sum(1 for path in files if "caption" in path.name),
    }
    return {
        "provided": True,
        "score_dir": str(score_dir),
        "exists": score_dir.is_dir(),
        "json_file_count": len(files),
        "patterns": patterns,
        "sample_files": [str(path) for path in files[:20]],
    }


def _namespaced_results_by_split(task: str, raw_results: Mapping[str, Any]) -> dict[str, Any]:
    results_by_split = raw_results.get("results_by_split")
    if not isinstance(results_by_split, Mapping):
        return {}
    return {f"{task}:{split_name}": metrics for split_name, metrics in results_by_split.items()}


def merge_task_results(task_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"task_results": {task: dict(result) for task, result in task_results.items()}}
    results_by_split: dict[str, Any] = {}
    caption_rows: list[Any] = []
    for task, raw_results in task_results.items():
        for key, value in raw_results.items():
            if key == "results_by_split":
                results_by_split.update(_namespaced_results_by_split(task, raw_results))
            elif key == "results":
                rows = list(value.values()) if isinstance(value, Mapping) else value
                if isinstance(rows, list):
                    caption_rows.extend(rows)
            elif key not in merged or merged.get(key) in (None, {}, []):
                merged[key] = value
        if task == "binary":
            for key in ("overall_average_precision", "overall_roc_auc", "evaluated_splits"):
                if key in raw_results:
                    merged[key] = raw_results[key]
        elif task == "vqa_retrieval":
            for key in (
                "overall_binary_acc",
                "overall_question_acc",
                "overall_retrieval_text",
                "overall_retrieval_image",
                "overall_retrieval_group",
            ):
                if key in raw_results:
                    merged[key] = raw_results[key]
    if results_by_split:
        merged["results_by_split"] = results_by_split
    if caption_rows:
        merged["results"] = caption_rows
    return merged


def caption_metric_from_results(raw_results: dict[str, Any]) -> tuple[float | None, int]:
    rows = raw_results.get("results")
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return None, 0

    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "gen_match" in row:
            value = scalar(row.get("gen_match"))
            if value is not None:
                values.append(value)
        metric_values = [
            scalar(row.get("spice")),
            scalar(row.get("cider")),
            scalar(row.get("bleu2")),
            scalar(row.get("rouge_l")),
            scalar(row.get("meteor")),
        ]
        metric_score = mean(metric_values)
        if metric_score is not None:
            values.append(metric_score)
    return mean(values), len(rows)


def extract_scores(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}

    ap = scalar(raw_results.get("overall_average_precision"))
    auc = scalar(raw_results.get("overall_roc_auc"))
    if ap is not None:
        extracted["camera_motion_average_precision"] = score_item(
            ap,
            "overall_average_precision",
            scalar(raw_results.get("evaluated_splits")),
        )
    if auc is not None:
        extracted["camera_motion_roc_auc"] = score_item(auc, "overall_roc_auc", scalar(raw_results.get("evaluated_splits")))

    vqa_score = mean([scalar(raw_results.get("overall_binary_acc")), scalar(raw_results.get("overall_question_acc"))])
    if vqa_score is not None:
        extracted["camera_vqa_accuracy"] = score_item(vqa_score, "mean_overall_binary_and_question_accuracy")

    retrieval_score = mean(
        [
            scalar(raw_results.get("overall_retrieval_text")),
            scalar(raw_results.get("overall_retrieval_image")),
            scalar(raw_results.get("overall_retrieval_group")),
        ]
    )
    if retrieval_score is not None:
        extracted["camera_retrieval_accuracy"] = score_item(retrieval_score, "mean_overall_retrieval_scores")

    caption_score, caption_count = caption_metric_from_results(raw_results)
    if caption_score is not None:
        extracted["camera_caption_score"] = score_item(caption_score, "mean_caption_metrics", caption_count)

    component_scores = [item["raw_score"] for metric_id, item in extracted.items() if metric_id != "camerabench_average"]
    average_score = mean(component_scores)
    if average_score is not None:
        extracted["camerabench_average"] = score_item(average_score, "computed_from_available_camerabench_metrics")
    return extracted


def prediction_rows(raw_results: dict[str, Any], upstream_results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_split = raw_results.get("results_by_split")
    if isinstance(by_split, dict):
        for split_name, metrics in by_split.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "task": "binary_or_vqa_retrieval",
                    "split": split_name,
                    "metrics": metrics,
                    "source": str(upstream_results_path.resolve()),
                }
            )

    caption_rows = raw_results.get("results")
    if isinstance(caption_rows, dict):
        caption_rows = list(caption_rows.values())
    if isinstance(caption_rows, list):
        for index, row in enumerate(caption_rows):
            if isinstance(row, dict):
                rows.append(
                    {
                        "task": "caption",
                        "sample_id": row.get("sample_id") or index,
                        "metrics": row,
                        "source": str(upstream_results_path.resolve()),
                    }
                )
    return rows


def normalize_camerabench_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    command: Any,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
    benchmark_data_root: Path | None = None,
    score_dir: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    camera_predictions_path = output_dir / "camera_predictions.jsonl"

    extracted_scores = extract_scores(raw_results)
    camera_rows = prediction_rows(raw_results, upstream_results_path)
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}

    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalize_unit_score(raw_score),
            "raw_score_range": [0.0, 1.0],
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_camerabench_results"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(camera_predictions_path, camera_rows)
    available_count = sum(1 for row in metric_rows if row["available"])
    all_metrics_available = available_count == len(METRIC_ORDER)
    dataset_coverage = _dataset_coverage(benchmark_data_root)
    score_inputs = _score_dir_report(score_dir)
    full_suite_valid = (
        returncode == 0
        and all_metrics_available
        and dataset_coverage.get("provided") is True
        and dataset_coverage.get("complete") is True
    )
    strict_failed = strict and not full_suite_valid
    normalization_ok = returncode == 0 and available_count > 0 and not strict_failed
    # Running the score-directory aggregator is not the complete CameraBench
    # evaluation protocol: the upstream per-sample score generation still has
    # to happen outside this runner.  Keep all imported/aggregated evidence
    # fail-closed until that end-to-end path is implemented and audited.
    normalizer_only = True
    official_verified = False
    integration_evidence = False

    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": "official_verified" if official_verified else "failed" if strict_failed else "official_results_normalized",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_camerabench_official_runner",
            "command": command,
            "returncode": 1 if strict_failed else returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "CameraBench",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_model_weights": True,
        },
        "dataset": {
            "sample_count": len(camera_rows),
            "coverage": dataset_coverage,
            "score_inputs": score_inputs,
            "official_results_path": str(upstream_results_path.resolve()),
        },
        "eligibility": {
            "full_suite_valid": full_suite_valid,
            "leaderboard_valid": False,
            "all_metrics_available": all_metrics_available,
            "reasons": [
                "WorldFoundry normalizes official CameraBench evaluator outputs for binary, VQA/retrieval, and caption metric families.",
                "leaderboard_valid remains false until the upstream score-generation method, model/checkpoint, and submission protocol are independently audited.",
            ],
        },
        "generation": {
            "successful": len(camera_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "binary_classification": ["camera_motion_average_precision", "camera_motion_roc_auc"],
                "vqa_retrieval": ["camera_vqa_accuracy", "camera_retrieval_accuracy"],
                "captioning": ["camera_caption_score"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(camera_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
                "all_metrics_available": all_metrics_available,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "camerabench_result_normalizer",
            "upstream_results": str(upstream_results_path.resolve()),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": False,
            "in_tree_metric_evaluator_executed": False,
            "upstream_aggregation_executed": command is not None,
            "official_results_imported": command is None and available_count > 0,
            "strict": strict,
            "strict_failed": strict_failed,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "camera_predictions": str(camera_predictions_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "normalization_ok": normalization_ok,
        "official_results_imported": command is None and available_count > 0,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def run_official_camerabench(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.from_upstream_results is not None:
        args.from_upstream_results = args.from_upstream_results.expanduser().resolve()
    if args.score_dir is not None:
        args.score_dir = args.score_dir.expanduser().resolve()
    if args.benchmark_data_root is not None:
        args.benchmark_data_root = args.benchmark_data_root.expanduser().resolve()
    strict = _strict_enabled(args.strict)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    if args.from_upstream_results:
        raw_results = load_json(args.from_upstream_results)
        if not isinstance(raw_results, dict):
            raise ValueError(f"CameraBench result JSON must be an object: {args.from_upstream_results}")
        return normalize_camerabench_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=args.from_upstream_results,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            benchmark_data_root=args.benchmark_data_root,
            score_dir=args.score_dir,
            strict=strict,
        )

    if args.score_dir is None:
        raise ValueError("--score-dir or --official-results-path is required")
    if not args.score_dir.is_dir():
        raise FileNotFoundError(f"CameraBench score directory not found: {args.score_dir}")
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    task_results, commands = evaluate_camerabench_from_score_dir(
        args.score_dir,
        output_dir=upstream_output_dir,
        task=args.task,
        mode=args.mode,
        no_gpt=args.no_gpt,
        openai_api_key=args.openai_api_key,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(
        "\n".join([f"===== {task} =====\n{task_results.get(task, {})}" for task in task_results]),
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    raw_results = merge_task_results(task_results)
    upstream_results_path = upstream_output_dir / f"camerabench_{args.task}_results.json"
    write_json(upstream_results_path, raw_results)
    return normalize_camerabench_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        command=commands[0] if len(commands) == 1 else commands,
        duration_seconds=duration_seconds,
        returncode=0,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        benchmark_data_root=args.benchmark_data_root,
        score_dir=args.score_dir,
        strict=strict,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute or normalize CameraBench metrics to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "camerabench"))
    parser.add_argument(
        "--official-results-path",
        dest="from_upstream_results",
        type=Path,
        default=env_path("WORLDFOUNDRY_CAMERABENCH_RESULTS_PATH"),
    )
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--score-dir", type=Path, default=env_path("WORLDFOUNDRY_CAMERABENCH_SCORE_DIR"))
    parser.add_argument("--benchmark-data-root", type=Path, default=_candidate_dataset_root(None))
    parser.add_argument("--task", choices=("binary", "vqa_retrieval", "caption", "all"), default=os.environ.get("WORLDFOUNDRY_CAMERABENCH_TASK", "binary"))
    parser.add_argument("--mode", choices=("vqa", "retrieval", "both"), default=os.environ.get("WORLDFOUNDRY_CAMERABENCH_MODE", "both"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--no-gpt", action="store_true")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--strict", action="store_true", help="Require complete local testset coverage and all CameraBench metric families.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_official_camerabench(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["integration_evidence"] is True,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "camera_predictions": scorecard["artifacts"]["camera_predictions"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official CameraBench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
