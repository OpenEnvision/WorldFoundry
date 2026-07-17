#!/usr/bin/env python3
"""Official runner for World-in-World closed-loop visual world model evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_world_in_world_metrics,
    load_results_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
    CANONICAL_AEQA_PROMPT_COUNT,
    CANONICAL_TASKS,
    load_prompt_records,
    resolve_world_in_world_root,
    unique_prompt_records,
)
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_runtime import (
    discover_metrics_json,
    run_world_in_world_evaluator,
    runtime_config_from_env,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize World-in-World official outputs.")
    parser.add_argument("--benchmark-id", default="world-in-world")
    parser.add_argument(
        "--official-results-path",
        dest="official_results_path",
        type=Path,
    )
    parser.add_argument(
        "--run-official",
        action="store_true",
        help=(
            "Evaluate WorldFoundry-generated World-in-World artifacts. "
            "Requires a metrics.json/world_in_world_metrics.json under --generated-artifact-dir "
            "or WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH."
        ),
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--world-in-world-assets-root", dest="world_in_world_assets_root", type=Path)
    parser.add_argument("--exp-id", default=None, help="Experiment id recorded in the normalized scorecard.")
    parser.add_argument("--task", default=None, help="World-in-World task: AR / IGNav / AEQA / Manip")
    parser.add_argument("--prompt-manifest", type=Path)
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
    imported_via_run_official: bool,
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
                "source": "world_in_world_imported_results",
                "source_path": str(source_path),
                "evidence_scope": "result_artifact_import_only",
                "imported_via_run_official": imported_via_run_official,
                "components": components,
                "reason": None if score is not None else "score_not_available_in_world_in_world_results",
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
    task: str | None,
    metric_rows: list[dict[str, Any]],
    video_coverage: Mapping[str, Any],
    runtime_summary: Mapping[str, Any] | None,
    imported_via_run_official: bool,
    prompt_count: int,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("expected_count", 0) > 0
        and video_coverage.get("complete") is True
    )
    reported_artifact_coverage_complete = (
        prompt_count >= CANONICAL_AEQA_PROMPT_COUNT
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    evidence_scope = "result_artifact_import_only"
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": True,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalization_ok,
        "evidence_scope": evidence_scope,
        "eligibility": {
            "full_suite_valid": False,
            "leaderboard_valid": False,
            "reported_artifact_coverage_complete": reported_artifact_coverage_complete,
            "video_coverage_complete": video_complete,
            "task": task,
            "canonical_tasks": list(CANONICAL_TASKS),
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_world_in_world_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "imported_via_run_official": imported_via_run_official,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "World-in-World"},
        "dataset": {"upstream_results": str(official_results_path.resolve())},
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
            "kind": "world_in_world_result_importer",
            "evidence_scope": evidence_scope,
            "importer_only": True,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "validation": {
            "normalizer_only": True,
            "official_runtime_executed": False,
            "official_runtime_succeeded": False,
            "official_results_imported": normalization_ok,
            "full_suite_complete": False,
            "scope": evidence_scope,
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "This runner imports and normalizes a supplied World-in-World metrics summary.",
            "It does not execute closed-loop task evaluators, so imported results are not official-run, full-suite, or leaderboard evidence.",
            "Video-quality metrics (FVD/SSIM/PSNR/LPIPS) are normalized when present in official result exports.",
        ],
        "prompt_count": prompt_count,
    }


def normalize_world_in_world_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH")
    if official_results_path is None:
        repo_root = args.world_in_world_assets_root or resolve_world_in_world_root()
        discovered = discover_metrics_json([output_dir, repo_root] if repo_root else [output_dir])
        official_results_path = discovered
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH, or --run-official is required"
        )

    task = args.task or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_TASK")
    prompt_records: list[dict[str, Any]] = []
    try:
        prompt_records = unique_prompt_records(
            load_prompt_records(
                task=task or "AEQA",
                episodes_path=args.prompt_manifest,
                repo_root=args.world_in_world_assets_root,
            )
        )
        if args.limit is not None:
            prompt_records = prompt_records[: int(args.limit)]
    except FileNotFoundError:
        if official_runtime_executed:
            raise

    result_rows = load_results_rows(Path(official_results_path))
    computed = compute_world_in_world_metrics(
        rows=result_rows,
        results_path=Path(official_results_path),
        task=task,
    )
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        imported_via_run_official=official_runtime_executed,
    )
    video_coverage = _coverage({record["prompt_id"] for record in prompt_records}, generated_dir)
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        task=task,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        runtime_summary=runtime_summary,
        imported_via_run_official=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_STRICT", "").strip().lower() in {
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
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "prompt_count": len(prompt_records),
            "task": task,
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_world_in_world(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    task = args.task or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_TASK", "AEQA")
    exp_id = args.exp_id or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_EXP_ID") or "worldfoundry_artifact"
    runtime_summary = run_world_in_world_evaluator(
        generated_artifact_dir=Path(generated_dir) if generated_dir is not None else None,
        output_dir=output_dir,
        config=runtime_config_from_env(task=task, exp_id=exp_id, repo_root=args.world_in_world_assets_root),
        limit=args.limit,
    )
    results_path = Path(str(runtime_summary.get("results_path") or (output_dir / "world_in_world_metrics.json")))
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_path,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        world_in_world_assets_root=args.world_in_world_assets_root,
        exp_id=exp_id,
        task=task,
        prompt_manifest=args.prompt_manifest,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_world_in_world_results(
        normalize_args,
        official_runtime_executed=True,
        runtime_summary=runtime_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("world-in-world")
        if sample_path is None:
            print("error: no checked-in World-in-World sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        if args.run_official:
            scorecard = run_official_world_in_world(args)
        else:
            scorecard = normalize_world_in_world_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "official_results_imported": False,
            "evidence_scope": "result_artifact_import_only",
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_world_in_world_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "World-in-World"},
            "dataset": {},
            "eligibility": {"full_suite_valid": False, "leaderboard_valid": False},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "world_in_world_result_importer",
                "evidence_scope": "result_artifact_import_only",
                "importer_only": True,
                "blocked_count": len(METRIC_ORDER),
            },
            "validation": {
                "normalizer_only": True,
                "official_runtime_executed": False,
                "official_runtime_succeeded": False,
                "official_results_imported": False,
                "full_suite_complete": False,
                "scope": "result_artifact_import_only",
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"world-in-world: failed ({exc})", file=sys.stderr)
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
        print(f"world-in-world: {scorecard.get('evaluation', {}).get('kind', 'unknown')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
