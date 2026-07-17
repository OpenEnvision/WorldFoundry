#!/usr/bin/env python3
"""Official runner for PhyGround physical reasoning evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_phyground_metrics,
    load_results_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_prompts import (
    CANONICAL_PROMPT_COUNT,
    load_prompt_records,
    resolve_prompts_json_path,
    unique_generation_records,
)
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_runtime import (
    judge_config_from_env,
    run_phyground_judge,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize PhyGround official outputs.")
    parser.add_argument("--benchmark-id", default="phyground")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Import PhyGround scores for WorldFoundry-generated artifacts.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--phyground-root", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--benchmark-data-root", type=Path)
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
                "source": "phyground_imported_results",
                "source_path": str(source_path),
                "evidence_scope": "result_artifact_import_only",
                "imported_via_run_official": imported_via_run_official,
                "components": components,
                "reason": None if score is not None else "score_not_available_in_phyground_results",
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
    judge_summary: Mapping[str, Any] | None,
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
        prompt_count >= CANONICAL_PROMPT_COUNT
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
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_phyground_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "imported_via_run_official": imported_via_run_official,
            "judge_summary": dict(judge_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "PhyGround"},
        "dataset": {
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "upstream_results": str(official_results_path.resolve()),
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
            "kind": "phyground_result_importer",
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
            "This runner imports and normalizes a supplied PhyGround scores.json artifact.",
            "It does not execute PhyJudge, so imported results are not official-run, full-suite, or leaderboard evidence.",
        ],
        "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
        "prompt_count": prompt_count,
    }


def normalize_phyground_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    judge_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_PHYGROUND_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_PHYGROUND_RESULTS_PATH, or --run-official is required"
        )
    prompts_path: Path | None = None
    prompt_records: list[dict[str, Any]] = []
    try:
        prompts_path = resolve_prompts_json_path(
            explicit=args.prompt_manifest,
            repo_root=args.phyground_root,
            data_root=args.benchmark_data_root,
        )
        prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
        if args.limit is not None:
            prompt_records = prompt_records[: int(args.limit)]
    except FileNotFoundError:
        if official_runtime_executed:
            raise
    result_rows = load_results_rows(Path(official_results_path))
    computed = compute_phyground_metrics(rows=result_rows, results_path=Path(official_results_path))
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
        prompt_manifest_path=prompts_path,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        judge_summary=judge_summary,
        imported_via_run_official=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_PHYGROUND_STRICT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    per_sample_rows = [
        {
            "video": row.get("video"),
            "SA": row.get("SA"),
            "PTV": row.get("PTV"),
            "persistence": row.get("persistence"),
            "physical_laws": row.get("physical_laws"),
        }
        for row in result_rows
        if row.get("video") or row.get("SA") is not None
    ]
    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", per_sample_rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "prompt_manifest": None if prompts_path is None else str(prompts_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "prompt_count": len(prompt_records),
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_phyground(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")
    prompts_path = resolve_prompts_json_path(
        explicit=args.prompt_manifest,
        repo_root=args.phyground_root,
        data_root=args.benchmark_data_root,
    )
    judge_summary = run_phyground_judge(
        generated_artifact_dir=Path(generated_dir),
        output_dir=output_dir,
        config=judge_config_from_env(),
        prompts_json_path=prompts_path,
        limit=args.limit,
    )
    scores_json = Path(str(judge_summary.get("scores_json") or (output_dir / "scores.json")))
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=scores_json,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        phyground_root=args.phyground_root,
        prompt_manifest=prompts_path,
        benchmark_data_root=args.benchmark_data_root,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_phyground_results(
        normalize_args,
        official_runtime_executed=True,
        judge_summary=judge_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("phyground")
        if sample_path is None:
            print("error: no checked-in PhyGround sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        if args.run_official:
            scorecard = run_official_phyground(args)
        else:
            scorecard = normalize_phyground_results(args)
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
                "runner": "benchmark_zoo_phyground_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "PhyGround"},
            "dataset": {},
            "eligibility": {"full_suite_valid": False, "leaderboard_valid": False},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "phyground_result_importer",
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
            print(f"phyground: failed ({exc})", file=sys.stderr)
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
        print(f"phyground: {scorecard.get('evaluation', {}).get('kind', 'unknown')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
