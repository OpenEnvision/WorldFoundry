#!/usr/bin/env python3
"""Official runner for iWorld-Bench camera-controllable video evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_iworldbench_metrics,
    load_results_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    load_prompt_records,
    official_video_filename_for_record,
    resolve_iworldbench_root,
    resolve_metadata_csv_path,
    unique_prompt_records,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_runtime import (
    discover_report_results,
    run_iworldbench_evaluator,
    runtime_config_from_env,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
SPLIT_DEFAULT_METRICS = {
    "diff": "action_control",
    "mem": "memory_ability",
    "camera_following": "camera_following",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize iWorld-Bench official outputs.")
    parser.add_argument("--benchmark-id", default="iworld-bench")
    parser.add_argument(
        "--official-results-path",
        dest="official_results_path",
        type=Path,
    )
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Execute the in-tree iWorld-Bench evaluator.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-videos-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
    )
    parser.add_argument("--iworld-root", "--iworldbench-root", dest="iworld_root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root containing dataset/all_pack metadata and first frames; independent of --iworld-root.",
    )
    parser.add_argument("--prompt-manifest", "--meta-csv", dest="prompt_manifest", type=Path)
    parser.add_argument("--split", default="diff", choices=("diff", "mem", "camera_following"))
    parser.add_argument(
        "--metric",
        default=None,
        help="Upstream metric selector for --run-official (defaults from --split or WORLDFOUNDRY_IWORLD_BENCH_METRIC).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = direct_metrics.get(metric_id)
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": "iworldbench_official_runtime" if official_runtime_executed else "iworldbench_results_file",
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_iworldbench_results",
            }
        )
    return rows


def _coverage(expected_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                actual_names.add(path.stem)
    missing = sorted(expected_ids - actual_names)
    unexpected = sorted(actual_names - expected_ids)
    matched = sorted(expected_ids & actual_names)
    return {
        "expected_count": len(expected_ids),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_ids) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    prompt_manifest_path: Path | None,
    metric_rows: list[dict[str, Any]],
    video_coverage: Mapping[str, Any],
    runtime_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
    prompt_count: int,
    split: str,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = video_coverage.get("expected_count", 0) > 0 and video_coverage.get("complete") is True
    full_suite_valid = (
        prompt_count >= CANONICAL_PROMPT_COUNT and video_complete and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    runtime_backend = str((runtime_summary or {}).get("backend") or "")
    official_component_verified = (
        official_runtime_executed and runtime_backend == "official" and normalization_ok
    )
    official_verified = official_component_verified and full_suite_valid
    integration_evidence = official_component_verified
    normalizer_only = not official_runtime_executed and not integration_evidence
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
            "official_component_verified": official_component_verified,
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_iworldbench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "iWorld-Bench"},
        "dataset": {
            "dataset_id": benchmark_id,
            "split": split,
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "sample_count": prompt_count,
            "canonical_sample_count": CANONICAL_PROMPT_COUNT,
            "generated_video_count": int(video_coverage.get("actual_count", 0)),
            "coverage_complete": video_complete,
        },
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
            "kind": "iworldbench_official_in_tree" if official_runtime_executed else "iworldbench_result_normalizer",
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "iWorld-Bench in-tree runtime materializes dataset/all_pack metadata prompts and normalizes reports/*.csv exports.",
            "The mock backend is available only when explicitly selected for CI contract validation and never counts as official evidence.",
            "Full trajectory/VBench metrics use the WorldFoundry VIPe and VBench integrations with their checkpoint assets.",
        ],
        "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
        "prompt_count": prompt_count,
    }


def normalize_iworldbench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = (
        args.generated_artifact_dir
        or _env_path("WORLDFOUNDRY_IWORLD_BENCH_GENERATED_VIDEO_DIR")
        or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    )
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_IWORLD_BENCH_RESULTS_PATH")
    if official_results_path is None:
        repo_root = args.iworld_root or resolve_iworldbench_root()
        discovered = discover_report_results([output_dir, repo_root] if repo_root else [output_dir])
        official_results_path = discovered
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_IWORLD_BENCH_RESULTS_PATH, or --run-official is required"
        )

    prompt_manifest_path: Path | None = None
    prompt_records: list[dict[str, Any]] = []
    try:
        prompt_manifest_path = resolve_metadata_csv_path(
            explicit=args.prompt_manifest,
            dataset_root=getattr(args, "dataset_root", None),
            repo_root=args.iworld_root,
            split=args.split,
        )
        prompt_records = unique_prompt_records(
            load_prompt_records(meta_csv_path=prompt_manifest_path, split=args.split)
        )
        if args.limit is not None:
            prompt_records = prompt_records[: int(args.limit)]
    except FileNotFoundError:
        if official_runtime_executed:
            raise

    result_rows = load_results_rows(Path(official_results_path))
    computed = compute_iworldbench_metrics(rows=result_rows, results_path=Path(official_results_path))
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        official_runtime_executed=official_runtime_executed,
    )
    video_coverage = _coverage(
        {Path(official_video_filename_for_record(record)).stem for record in prompt_records},
        generated_dir,
    )
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        prompt_manifest_path=prompt_manifest_path,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        runtime_summary=runtime_summary,
        official_runtime_executed=official_runtime_executed,
        prompt_count=len(prompt_records),
        split=args.split,
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "prompt_count": len(prompt_records),
            "split": args.split,
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_iworldbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = (
        args.generated_artifact_dir
        or _env_path("WORLDFOUNDRY_IWORLD_BENCH_GENERATED_VIDEO_DIR")
        or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    )
    if generated_dir is None:
        raise ValueError(
            "--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official"
        )
    metric = args.metric or os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_METRIC") or SPLIT_DEFAULT_METRICS[args.split]
    runtime_summary = run_iworldbench_evaluator(
        generated_artifact_dir=Path(generated_dir),
        output_dir=output_dir,
        config=runtime_config_from_env(metric=metric, repo_root=args.iworld_root),
    )
    results_path = Path(str(runtime_summary.get("results_path") or (output_dir / "reports")))
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_path,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        iworld_root=args.iworld_root,
        dataset_root=getattr(args, "dataset_root", None),
        prompt_manifest=args.prompt_manifest,
        split=args.split,
        metric=metric,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_iworldbench_results(
        normalize_args,
        official_runtime_executed=True,
        runtime_summary=runtime_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("iworld-bench")
        if sample_path is None:
            print("error: no checked-in iWorld-Bench sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        if args.run_official:
            scorecard = run_official_iworldbench(args)
        else:
            scorecard = normalize_iworldbench_results(args)
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
                "runner": "benchmark_zoo_iworldbench_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "iWorld-Bench"},
            "dataset": {
                "dataset_id": args.benchmark_id,
                "split": getattr(args, "split", None),
                "prompt_manifest": None
                if getattr(args, "prompt_manifest", None) is None
                else str(args.prompt_manifest),
                "sample_count": 0,
                "canonical_sample_count": CANONICAL_PROMPT_COUNT,
                "generated_video_count": 0,
                "coverage_complete": False,
            },
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "iworldbench_result_normalizer",
                "blocked_count": len(METRIC_ORDER),
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"iworld-bench: failed ({exc})", file=sys.stderr)
        return 1

    payload = {
        "ok": scorecard.get("normalization_ok") is True,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        **scorecard,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"iworld-bench: {scorecard.get('evaluation', {}).get('kind', 'unknown')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
