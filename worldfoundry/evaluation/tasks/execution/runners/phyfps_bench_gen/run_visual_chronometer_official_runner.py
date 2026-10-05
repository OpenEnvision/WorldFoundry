#!/usr/bin/env python3
"""Official runner for Visual Chronometer PhyFPS prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path, SCORECARD_SCHEMA_VERSION
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_predict import (
    PhyFPSPredictConfig,
    parse_results_csv,
    run_phyfps_predict,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.visual_chronometer_metrics import (
    METRIC_ORDER,
    compute_visual_chronometer_metrics,
    metric_rows_from_computed,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.visual_chronometer_runtime import (
    VIDEO_SUFFIXES,
    resolve_chronometer_root,
)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize Visual Chronometer official outputs.")
    parser.add_argument("--benchmark-id", default="visual-chronometer")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-official", action="store_true", help="Execute Visual Chronometer PhyFPS prediction in-tree.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--visual-chronometer-root", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=30)
    parser.add_argument("--device", default=os.environ.get("WORLDFOUNDRY_VISUAL_CHRONOMETER_DEVICE", "cuda:0"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)




def _resolve_predict_backend() -> str:
    for env_name in ("WORLDFOUNDRY_VISUAL_CHRONOMETER_PREDICT_BACKEND", "WORLDFOUNDRY_PHYFPS_PREDICT_BACKEND"):
        value = os.environ.get(env_name)
        if value:
            return value.lower()
    return "official"


def _video_index(generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None:
        return {
            "provided": False,
            "generated_artifact_dir": None,
            "video_count": 0,
            "by_stem": {},
            "duplicate_stems": [],
        }
    by_stem: dict[str, list[str]] = {}
    if generated_dir.exists():
        for path in sorted(generated_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_stem.setdefault(path.stem, []).append(str(path))
    return {
        "provided": True,
        "generated_artifact_dir": str(generated_dir),
        "video_count": sum(len(paths) for paths in by_stem.values()),
        "by_stem": dict(by_stem),
        "duplicate_stems": sorted(stem for stem, paths in by_stem.items() if len(paths) > 1),
    }


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    video_coverage: Mapping[str, Any],
    metric_rows: list[dict[str, Any]],
    predict_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("provided") is True
        and video_coverage.get("video_count", 0) > 0
        and not video_coverage.get("duplicate_stems")
    )
    full_suite_valid = video_complete and len(available_rows) == len(METRIC_ORDER)
    normalization_ok = bool(available_rows)
    predict_backend = str((predict_summary or {}).get("backend") or "").lower()
    official_runtime_real = official_runtime_executed and predict_backend != "mock"
    official_verified = official_runtime_real and normalization_ok
    integration_evidence = official_verified and full_suite_valid
    normalizer_only = not official_runtime_executed and not integration_evidence
    evaluation_kind = (
        "visual_chronometer_official_in_tree"
        if official_runtime_executed
        else "visual_chronometer_result_normalizer"
    )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": normalizer_only,
        "normalization_ok": normalization_ok,
        "eligibility": {
            "full_suite_valid": full_suite_valid,
            "video_coverage_complete": video_complete,
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_visual_chronometer_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "predict_summary": dict(predict_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "Visual Chronometer"},
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": evaluation_kind,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "results_csv": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "Visual Chronometer predicts PhyFPS from generated or caller-supplied videos.",
            "Use WORLDFOUNDRY_VISUAL_CHRONOMETER_PREDICT_BACKEND=mock for CI fixture validation.",
        ],
    }


def normalize_visual_chronometer_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    predict_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or resolve_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or resolve_env_path("WORLDFOUNDRY_VISUAL_CHRONOMETER_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_VISUAL_CHRONOMETER_RESULTS_PATH, "
            "or --run-official is required"
        )
    records = parse_results_csv(Path(official_results_path))
    computed = compute_visual_chronometer_metrics(video_records=records)
    metric_rows = metric_rows_from_computed(
        computed,
        source_path=Path(official_results_path),
        official_runtime_executed=official_runtime_executed,
    )
    video_coverage = _video_index(generated_dir)
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        video_coverage=video_coverage,
        metric_rows=metric_rows,
        predict_summary=predict_summary,
        official_runtime_executed=official_runtime_executed,
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_VISUAL_CHRONOMETER_STRICT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(
        output_dir / "per_sample_scores.jsonl",
        [
            {
                "video": record.video,
                "avg_phyfps": record.avg_phyfps,
                "segment_count": len(record.segment_phyfps),
            }
            for record in records
        ],
    )
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_visual_chronometer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_artifact_dir = args.generated_artifact_dir or resolve_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_artifact_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")

    config = PhyFPSPredictConfig(
        backend=_resolve_predict_backend(),
        stride=args.stride,
        clip_length=args.clip_length,
        device=args.device,
        python_executable=args.python,
        chronometer_root=resolve_chronometer_root(args.visual_chronometer_root),
        timeout_seconds=args.timeout,
    )
    results_csv = output_dir / "results.csv"
    predict_summary = run_phyfps_predict(
        config=config,
        video_dir=Path(generated_artifact_dir),
        output_csv=results_csv,
        stdout_path=output_dir / "predict_stdout.log",
        stderr_path=output_dir / "predict_stderr.log",
    )
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_csv,
        output_dir=output_dir,
        generated_artifact_dir=generated_artifact_dir,
        visual_chronometer_root=args.visual_chronometer_root,
        stride=args.stride,
        clip_length=args.clip_length,
        device=args.device,
        python=args.python,
        timeout=args.timeout,
        strict=args.strict,
        json=False,
    )
    return normalize_visual_chronometer_results(
        normalize_args,
        official_runtime_executed=True,
        predict_summary=predict_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_official:
            scorecard = run_official_visual_chronometer(args)
        else:
            scorecard = normalize_visual_chronometer_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_visual_chronometer_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "Visual Chronometer"},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "visual_chronometer_result_normalizer",
                "blocked_count": len(METRIC_ORDER),
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
        return 1
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
    return int(scorecard.get("run", {}).get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
