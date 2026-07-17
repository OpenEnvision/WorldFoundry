#!/usr/bin/env python3
"""Run or normalize the in-tree WRBench D1-D6 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.utils import benchmark_task_sample_path

from worldfoundry.evaluation.tasks.execution.runners.wrbench.wrbench_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_wrbench_metrics,
    load_wrbench_results,
)
from worldfoundry.evaluation.tasks.execution.runners.wrbench.wrbench_paths import (
    VENDORED_REPOSITORY,
    VENDORED_REVISION,
    published_results_path,
)
from worldfoundry.evaluation.tasks.execution.runners.wrbench.wrbench_prompts import (
    CANONICAL_REQUEST_COUNT,
    materialize_wrbench_generation_requests,
)
from worldfoundry.evaluation.tasks.execution.runners.wrbench.wrbench_runtime import run_wrbench_evaluator


SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize WRBench D1-D6 outputs.")
    parser.add_argument("--benchmark-id", default="wrbench")
    parser.add_argument("--official-results-path", "--from-upstream-results", dest="official_results_path", type=Path)
    parser.add_argument("--run-official", action="store_true", help="Execute the vendored WRBench D1-D6 runtime.")
    parser.add_argument("--run-fixture", action="store_true", help="Normalize the checked-in sample result table.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--wrbench-root", type=Path)
    parser.add_argument("--scorer-profile", default="wrbench_default", choices=("wrbench_default", "ablation_manifest_metadata", "custom"))
    parser.add_argument("--sidecar-profile-gate", default="main", choices=("main", "certified_opencv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _coverage(expected_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    actual: set[str] = set()
    if generated_dir is not None and generated_dir.is_dir():
        actual = {
            path.stem
            for path in generated_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        }
    missing = sorted(expected_ids - actual)
    unexpected = sorted(actual - expected_ids)
    return {
        "expected_count": len(expected_ids),
        "actual_count": len(actual),
        "matched_count": len(expected_ids & actual),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_ids) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _metric_rows(
    *, computed: Mapping[str, Any], source_path: Path, official_runtime_executed: bool
) -> list[dict[str, Any]]:
    metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    counts = computed.get("counts") if isinstance(computed.get("counts"), Mapping) else {}
    sources = computed.get("sources") if isinstance(computed.get("sources"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        score = metrics.get(metric_id)
        spec = METRIC_SPECS[metric_id]
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "group": spec["group"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": True,
                "count": counts.get(metric_id),
                "aggregation_source": sources.get(metric_id),
                "source": "wrbench_official_runtime" if official_runtime_executed else "wrbench_results_file",
                "source_path": str(source_path.resolve()),
                "reason": None if score is not None else "score_not_available_in_wrbench_results",
            }
        )
    return rows


def normalize_wrbench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is not None:
        generated_dir = generated_dir.expanduser().resolve()

    results_path = args.official_results_path or _env_path("WORLDFOUNDRY_WRBENCH_RESULTS_PATH")
    if results_path is None and args.run_fixture:
        results_path = benchmark_task_sample_path(args.benchmark_id)
    if results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_WRBENCH_RESULTS_PATH, --run-fixture, or --run-official is required"
        )

    result_rows, metadata, resolved_results_path = load_wrbench_results(Path(results_path))
    computed = compute_wrbench_metrics(rows=result_rows, metadata=metadata)
    metric_rows = _metric_rows(
        computed=computed,
        source_path=resolved_results_path,
        official_runtime_executed=official_runtime_executed,
    )
    requests = materialize_wrbench_generation_requests(limit=args.limit, repo_root=args.wrbench_root)
    expected_ids = {request.sample_id for request in requests}
    video_coverage = _coverage(expected_ids, generated_dir)
    available_rows = [row for row in metric_rows if row["available"]]
    leaderboard = {row["metric_id"]: row["score"] for row in available_rows}
    full_suite_valid = (
        len(requests) == CANONICAL_REQUEST_COUNT
        and video_coverage["complete"] is True
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    official_verified = official_runtime_executed and normalization_ok
    scorecard: dict[str, Any] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_verified and full_suite_valid,
        "leaderboard_valid": False,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "eligibility": {
            "full_suite_valid": full_suite_valid,
            "natural25_request_count": len(requests),
            "canonical_natural25_request_count": CANONICAL_REQUEST_COUNT,
            "video_coverage_complete": video_coverage["complete"],
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_wrbench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {
            "benchmark_id": args.benchmark_id,
            "name": "WRBench",
            "metric_contract": "wrbench_latest_d1_d6_metrics_v3",
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": {row["metric_id"]: row for row in metric_rows},
            "summary": {
                "available_metric_count": len(available_rows),
                "declared_metric_count": len(METRIC_ORDER),
                "normalized_result_rows": computed["row_count"],
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "wrbench_official_in_tree" if official_runtime_executed else "wrbench_result_normalizer",
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "coverage": {"videos": video_coverage},
        "artifacts": {
            "scorecard": str(output_dir / "scorecard.json"),
            "official_results_path": str(resolved_results_path),
            "raw_metric_table": str(output_dir / "raw_metric_table.jsonl"),
            "per_sample_scores": str(output_dir / "per_sample_scores.jsonl"),
        },
        "provenance": {
            "repository": VENDORED_REPOSITORY,
            "revision": VENDORED_REVISION,
            "runtime_location": "worldfoundry/evaluation/tasks/execution/runners/wrbench/runtime/wrbench",
        },
        "notes": [
            "D1-CamPrec and D1-CamAlign remain separate metrics.",
            "D5/D6 retain their conditional re-observation denominators.",
            "Full scoring needs explicitly configured VGGT, DINOv2, Qwen3.5, and Qwen3-VL assets.",
        ],
    }
    if args.strict and not full_suite_valid:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", result_rows)
    write_jsonl(output_dir / "generation_requests.jsonl", [request.to_dict() for request in requests])
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "schema_version": "wrbench_latest_d1_d6_metrics_v3",
            "metric_ids": list(METRIC_ORDER),
            "natural25_request_count": len(requests),
            "official_results_path": str(resolved_results_path),
            "vendored_revision": VENDORED_REVISION,
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_wrbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    summary = run_wrbench_evaluator(
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        video_manifest=args.video_manifest,
        runtime_config=args.runtime_config,
        scorer_profile=args.scorer_profile,
        sidecar_profile_gate=args.sidecar_profile_gate,
        repo_root=args.wrbench_root,
    )
    args.official_results_path = Path(str(summary["results_path"]))
    return normalize_wrbench_results(args, official_runtime_executed=True, runtime_summary=summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture and args.official_results_path is None:
        fixture = benchmark_task_sample_path(args.benchmark_id)
        if fixture is None:
            fixture = published_results_path(args.wrbench_root)
        args.official_results_path = fixture
    try:
        scorecard = run_official_wrbench(args) if args.run_official else normalize_wrbench_results(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    return int(scorecard["run"]["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
