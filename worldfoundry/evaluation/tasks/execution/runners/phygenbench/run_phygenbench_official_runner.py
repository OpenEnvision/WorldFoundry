#!/usr/bin/env python3
"""Official runner for PhyGenBench physical commonsense evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_phygenbench_metrics,
    load_results_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    load_prompt_records,
    resolve_prompts_json_path,
    unique_generation_records,
)
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_runtime import (
    judge_config_from_env,
    run_phygenbench_judge,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize PhyGenBench official outputs.")
    parser.add_argument("--benchmark-id", default="phygenbench")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Run the configured fixture/overall aggregation component; official stage judges are not in-tree.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--phygenbench-root", type=Path)
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
    official_component_executed: bool,
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
                "source": (
                    "phygenbench_official_overall_aggregation"
                    if official_component_executed
                    else "phygenbench_results_file"
                ),
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_phygenbench_results",
            }
        )
    return rows


def _coverage(expected_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    def _normalize(stem: str) -> str:
        if stem.startswith("output_video_"):
            return stem.removeprefix("output_video_")
        return stem.lstrip("0") if stem.isdigit() else stem

    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                actual_names.add(_normalize(path.stem))
    expected_normalized = {_normalize(prompt_id) for prompt_id in expected_ids}
    actual_normalized = actual_names
    missing = sorted(expected_normalized - actual_normalized)
    unexpected = sorted(actual_normalized - expected_normalized)
    matched = sorted(expected_normalized & actual_normalized)
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


def _official_overall_component_executed(
    *,
    official_runtime_executed: bool,
    judge_summary: Mapping[str, Any] | None,
) -> bool:
    backend = str((judge_summary or {}).get("backend") or "").strip().lower()
    return official_runtime_executed and backend == "official"


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    prompt_manifest_path: Path | None,
    metric_rows: list[dict[str, Any]],
    video_coverage: Mapping[str, Any],
    judge_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
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
    input_coverage_complete = (
        prompt_count >= CANONICAL_PROMPT_COUNT
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    component_executed = _official_overall_component_executed(
        official_runtime_executed=official_runtime_executed,
        judge_summary=judge_summary,
    )
    integration_evidence = component_executed and normalization_ok
    evidence_scope = (
        "official_overall_aggregation"
        if integration_evidence
        else "mock_or_result_import"
    )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": not integration_evidence,
        "normalization_ok": normalization_ok,
        "eligibility": {
            "full_suite_valid": False,
            "input_coverage_complete": input_coverage_complete,
            "video_coverage_complete": video_complete,
            "official_stage_judges_executed": False,
        },
        "run": {
            "status": (
                "official_overall_aggregation_completed"
                if integration_evidence
                else "mock_or_results_normalized"
                if normalization_ok
                else "failed"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_phygenbench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "judge_summary": dict(judge_summary or {}),
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "PhyGenBench",
            "evidence_scope": evidence_scope,
        },
        "dataset": {
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "prompt_count": prompt_count,
            "generated_video_coverage": dict(video_coverage),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "official_available_count": len(available_rows) if integration_evidence else 0,
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": (
                "phygenbench_official_overall_component"
                if integration_evidence
                else "phygenbench_mock_or_result_normalizer"
            ),
            "evidence_scope": evidence_scope,
            "official_stage_judges_executed": False,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "PhyGenBench in-tree runtime materializes prompts.json T2V requests and official output_video_{id}.mp4 names.",
            "The non-mock runtime executes upstream overall.py only after official stage-result files already exist.",
            "The mock backend is fixture validation and never benchmark evidence.",
        ],
        "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
        "prompt_count": prompt_count,
    }


def normalize_phygenbench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    judge_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_PHYGENBENCH_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_PHYGENBENCH_RESULTS_PATH, or --run-official is required"
        )
    prompts_path: Path | None = None
    prompt_records: list[dict[str, Any]] = []
    try:
        prompts_path = resolve_prompts_json_path(
            explicit=args.prompt_manifest,
            repo_root=args.phygenbench_root,
        )
        prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
        if args.limit is not None:
            prompt_records = prompt_records[: int(args.limit)]
    except FileNotFoundError:
        if official_runtime_executed:
            raise
    result_rows = load_results_rows(Path(official_results_path))
    computed = compute_phygenbench_metrics(rows=result_rows, results_path=Path(official_results_path))
    official_component_executed = _official_overall_component_executed(
        official_runtime_executed=official_runtime_executed,
        judge_summary=judge_summary,
    )
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        official_component_executed=official_component_executed,
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
        official_runtime_executed=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_PHYGENBENCH_STRICT", "").strip().lower() in {
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
            "prompt_id": row.get("prompt_id"),
            "caption": row.get("caption"),
            "single": row.get("single"),
            "multi_gpt": row.get("multi_gpt"),
            "video_gpt": row.get("video_gpt"),
            "semantic_score": row.get("semantic_score") or row.get("semantic_adherence"),
            "main_category": row.get("main_category"),
        }
        for row in result_rows
        if row.get("caption") or row.get("single") is not None
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


def run_official_phygenbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")
    prompts_path = resolve_prompts_json_path(
        explicit=args.prompt_manifest,
        repo_root=args.phygenbench_root,
    )
    judge_summary = run_phygenbench_judge(
        generated_artifact_dir=Path(generated_dir),
        output_dir=output_dir,
        config=judge_config_from_env(repo_root=args.phygenbench_root),
        prompts_json_path=prompts_path,
        limit=args.limit,
    )
    results_json = Path(str(judge_summary.get("results_json") or (output_dir / "phygenbench_results.json")))
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_json,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        phygenbench_root=args.phygenbench_root,
        prompt_manifest=prompts_path,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_phygenbench_results(
        normalize_args,
        official_runtime_executed=True,
        judge_summary=judge_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("phygenbench")
        if sample_path is None:
            print("error: no checked-in PhyGenBench sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        if args.run_official:
            scorecard = run_official_phygenbench(args)
        else:
            scorecard = normalize_phygenbench_results(args)
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
                "runner": "benchmark_zoo_phygenbench_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "PhyGenBench"},
            "dataset": {
                "prompt_manifest": None,
                "prompt_count": 0,
                "generated_video_coverage": {},
            },
            "eligibility": {
                "full_suite_valid": False,
                "official_stage_judges_executed": False,
            },
            "metrics": {
                "leaderboard": {},
                "per_metric": {},
                "summary": {"available_metric_count": 0, "official_available_count": 0},
            },
            "evaluation": {
                "available": False,
                "kind": "phygenbench_result_normalizer",
                "blocked_count": len(METRIC_ORDER),
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"phygenbench: failed ({exc})", file=sys.stderr)
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
        print(f"phygenbench: {scorecard.get('evaluation', {}).get('kind', 'unknown')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
