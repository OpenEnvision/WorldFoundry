#!/usr/bin/env python3
"""Official runner implementation for EvalCrafter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    load_upstream_results,
    normalized_score,
    scalar,
)
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
    CANONICAL_PROMPT_COUNT,
    load_prompt_records,
    resolve_evalcrafter_root,
    resolve_prompt700_path,
    unique_prompt_records,
)
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_raw_metrics import (
    SUPPORTED_RAW_METRICS,
)
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_runtime import (
    run_evalcrafter_scorer,
    validate_official_inputs,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
BENCHMARK_ID = "evalcrafter"
DISPLAY_NAME = "EvalCrafter"
INPUT_KEYS = ("generated_video_dir", "prompt700_txt", "official_results_path")
OUTPUT_KEYS = ("scorecard", "raw_metric_table", "per_sample_metrics", "benchmark_contract")


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize the official EvalCrafter benchmark runtime.")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--evalcrafter-root", type=Path)
    parser.add_argument("--videos-dir", "--generated-artifact-dir", "--generated-video-dir", dest="videos_dir", type=Path)
    parser.add_argument(
        "--results-dir",
        "--official-results-path",
        dest="results_dir",
        type=Path,
    )
    parser.add_argument("--from-upstream-results", dest="results_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Run the supported official raw-video metrics on WorldFoundry-generated artifacts.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate prompt/video layout without executing EvalCrafter.",
    )
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metrics",
        nargs="+",
        help=(
            "Raw-video metrics to execute (space- or comma-separated). "
            f"Supported: {', '.join(SUPPORTED_RAW_METRICS)}. Defaults to all supported metrics."
        ),
    )
    parser.add_argument("--clip-model", type=Path, help="Local openai/clip-vit-base-patch32 checkpoint directory.")
    parser.add_argument("--device", default="auto", help="Torch device for raw-video metrics (default: auto).")
    parser.add_argument("--batch-size", type=int, default=8, help="CLIP frame batch size (default: 8).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the complete 700-video, 17-metric, five-aggregate suite; bounded metric runs fail this gate.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _metric_rows(
    *,
    extracted_scores: dict[str, dict[str, Any]],
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        item = extracted_scores.get(metric_id, {})
        raw_score = scalar(item.get("raw_score"))
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": raw_score is not None,
                "raw_score": raw_score,
                "normalized_score": normalized_score(raw_score),
                "score": raw_score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": item.get("source")
                or ("evalcrafter_official_runtime" if official_runtime_executed else "evalcrafter_results_file"),
                "source_path": str(source_path),
                "sample_count": item.get("sample_count"),
                "reason": None if raw_score is not None else "score_not_found_in_evalcrafter_final_result",
            }
        )
    return rows


def _coverage(expected_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".mp4":
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


def write_scorecard(
    extracted_scores: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    upstream_results_path: Path,
    evalcrafter_root: Path | None,
    videos_dir: Path | None,
    prompt700_path: Path | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
    video_coverage: Mapping[str, Any] | None = None,
    scorer_summary: Mapping[str, Any] | None = None,
    official_runtime_executed: bool | None = None,
    prompt_count: int | None = None,
    official_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    scorecard_path = output_dir / "scorecard.json"
    contract_path = output_dir / "benchmark_contract.json"
    per_sample_path = output_dir / "per_sample_metrics.jsonl"

    if scorer_summary is None and official_run is not None:
        scorer_summary = {"official_run": dict(official_run)}
    if official_runtime_executed is None:
        official_runtime_executed = official_run is not None
    if prompt700_path is None:
        try:
            prompt700_path = resolve_prompt700_path(repo_root=evalcrafter_root)
        except FileNotFoundError:
            prompt700_path = None
    if prompt_count is None:
        if prompt700_path is None:
            prompt_count = 0
        else:
            prompt_count = len(unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path)))
    if metric_rows is None:
        metric_rows = _metric_rows(
            extracted_scores=extracted_scores,
            source_path=upstream_results_path,
            official_runtime_executed=official_runtime_executed,
        )
    if video_coverage is None:
        expected_ids = set()
        if prompt700_path is not None:
            expected_ids = {record["prompt_id"] for record in unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path))}
        video_coverage = _coverage(expected_ids, videos_dir)

    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["raw_score"]
        for row in metric_rows
        if row.get("available") and row.get("raw_score") is not None
    }
    available_count = sum(1 for row in metric_rows if row["available"])
    normalization_ok = available_count > 0
    official_run = scorer_summary.get("official_run") if isinstance(scorer_summary, Mapping) else None
    scorer_backend = str(scorer_summary.get("backend") or "") if isinstance(scorer_summary, Mapping) else ""
    executed_metrics = (
        [str(item) for item in scorer_summary.get("executed_metrics") or ()]
        if isinstance(scorer_summary, Mapping)
        else []
    )
    raw_video_component = scorer_backend == "raw_video" and normalization_ok and bool(executed_metrics)
    official_results_imported = normalization_ok and not raw_video_component
    video_complete = (
        video_coverage.get("expected_count", 0) > 0
        and video_coverage.get("complete") is True
    )
    full_suite_valid = (
        prompt_count >= CANONICAL_PROMPT_COUNT
        and video_complete
        and available_count == len(METRIC_ORDER)
    )
    official_verified = raw_video_component and full_suite_valid
    integration_evidence = raw_video_component
    normalizer_only = not raw_video_component
    evidence_level = (
        "official_raw_video_metric_subset"
        if raw_video_component and not full_suite_valid
        else "official_full_runtime"
        if official_verified
        else "official_results_normalized"
    )
    run_status = (
        "official_full_runtime"
        if official_verified
        else "official_raw_video_metric_subset"
        if raw_video_component
        else "official_results_imported"
        if normalization_ok
        else "failed"
    )
    eligibility_reasons: list[str] = []
    if not full_suite_valid:
        eligibility_reasons.append(
            "full EvalCrafter requires 700 prompt-matched videos and all 17 raw metrics plus five official aggregates"
        )
    if not raw_video_component:
        eligibility_reasons.append("this invocation did not execute raw-video metrics")
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_evalcrafter_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "official_run": official_run,
            "scorer_summary": dict(scorer_summary or {}),
        },
        "benchmark": {
            "benchmark_id": BENCHMARK_ID,
            "name": DISPLAY_NAME,
            "contract_only": False,
            "evidence_level": evidence_level,
            "evidence_scope": (
                "bounded_official_clip_metric_subset" if raw_video_component and not full_suite_valid else evidence_level
            ),
            "official_benchmark_verified": official_verified,
        },
        "dataset": {
            "generated_artifact_dir": None if videos_dir is None else str(videos_dir),
            "prompt_suite": None if prompt700_path is None else str(prompt700_path),
        },
        "eligibility": {
            "full_suite_valid": full_suite_valid,
            "video_coverage_complete": video_complete,
            "official_raw_video_component_verified": raw_video_component,
            "executed_metrics": executed_metrics,
            "leaderboard_valid": official_verified and full_suite_valid,
            "reasons": [] if official_verified else eligibility_reasons,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "available_metrics": available_count,
                "total_metrics": len(metric_rows),
                "executed_raw_video_metrics": len(executed_metrics),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": (
                "evalcrafter_official_raw_video_metrics" if raw_video_component else "evalcrafter_result_normalizer"
            ),
            "evidence_level": evidence_level,
            "leaderboard_metrics": leaderboard,
            "num_results": available_count,
            "blocked_count": len(METRIC_ORDER) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": official_runtime_executed,
            "official_raw_video_component_verified": raw_video_component,
            "official_results_imported": official_results_imported,
        },
        "coverage": {"videos": dict(video_coverage)},
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "benchmark_contract": str(contract_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_metrics": str(per_sample_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "leaderboard_valid": official_verified and full_suite_valid,
        "integration_evidence": integration_evidence,
        "normalization_ok": normalization_ok,
        "normalizer_only": normalizer_only,
        "official_results_imported": official_results_imported,
        "prompt_count": prompt_count,
        "notes": [
            (
                "This run executed a bounded subset of official EvalCrafter raw-video metrics; it is not a full "
                "17-metric benchmark or leaderboard result."
                if raw_video_component and not full_suite_valid
                else "This run normalized an existing EvalCrafter final_result.txt."
            ),
            "Unavailable metrics remain unavailable; no missing value or aggregate is synthesized.",
        ],
    }

    benchmark_contract = {
        "benchmark_id": BENCHMARK_ID,
        "display_name": DISPLAY_NAME,
        "input_keys": list(INPUT_KEYS),
        "output_keys": list(OUTPUT_KEYS),
        "metric_ids": list(METRIC_ORDER),
        "requires_upstream_runtime": not raw_video_component,
        "runner": "benchmark_zoo_evalcrafter_official_runner",
        "upstream_results": str(upstream_results_path),
    }
    write_json(scorecard_path, scorecard)
    write_json(contract_path, benchmark_contract)
    write_jsonl(raw_metric_table_path, metric_rows)
    if not per_sample_path.is_file():
        write_jsonl(per_sample_path, [])
    return scorecard


def normalize_evalcrafter_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    scorer_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    evalcrafter_root = args.evalcrafter_root or resolve_evalcrafter_root()
    videos_dir = args.videos_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    results_dir = args.results_dir or _env_path("WORLDFOUNDRY_EVALCRAFTER_RESULTS_PATH")
    if results_dir is None and scorer_summary is not None:
        results_dir = Path(str(scorer_summary.get("results_path") or output_dir / "final_result.txt"))
    if results_dir is None:
        raise ValueError("--results-dir, WORLDFOUNDRY_EVALCRAFTER_RESULTS_PATH, or --run-official is required")

    prompt700_path: Path | None = None
    prompt_records: list[dict[str, Any]] = []
    try:
        prompt700_path = resolve_prompt700_path(explicit=args.prompt_manifest, repo_root=evalcrafter_root)
        prompt_records = unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path))
        if args.limit is not None:
            prompt_records = prompt_records[: int(args.limit)]
    except FileNotFoundError:
        if official_runtime_executed:
            raise

    extracted_scores, upstream_results_path = load_upstream_results(Path(results_dir))
    metric_rows = _metric_rows(
        extracted_scores=extracted_scores,
        source_path=upstream_results_path,
        official_runtime_executed=official_runtime_executed,
    )
    if isinstance(scorer_summary, Mapping) and scorer_summary.get("backend") == "raw_video":
        sample_counts = scorer_summary.get("metric_sample_counts")
        sample_counts = sample_counts if isinstance(sample_counts, Mapping) else {}
        executed = {str(item) for item in scorer_summary.get("executed_metrics") or ()}
        for row in metric_rows:
            metric_id = str(row["metric_id"])
            if metric_id in executed:
                row["source"] = "evalcrafter_official_raw_video_metric"
                row["sample_count"] = int(sample_counts.get(metric_id) or 0)
                row["evidence_scope"] = "bounded_official_clip_metric_subset"
    video_coverage = _coverage({record["prompt_id"] for record in prompt_records}, videos_dir)
    scorecard = write_scorecard(
        extracted_scores,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        evalcrafter_root=evalcrafter_root,
        prompt700_path=prompt700_path,
        videos_dir=videos_dir,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        scorer_summary=scorer_summary,
        official_runtime_executed=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_EVALCRAFTER_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1
        write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_evalcrafter(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.videos_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if videos_dir is None:
        raise ValueError("--videos-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")
    evalcrafter_root = args.evalcrafter_root or resolve_evalcrafter_root()
    if evalcrafter_root is None:
        raise FileNotFoundError("EvalCrafter root not found; bundled assets should be available in-tree.")
    input_validation = validate_official_inputs(evalcrafter_root, Path(videos_dir))
    write_json(output_dir / "input_validation.json", input_validation)
    if args.strict and not input_validation["ok"]:
        raise ValueError("; ".join(input_validation["reasons"]))
    prompt700_path = resolve_prompt700_path(explicit=args.prompt_manifest, repo_root=evalcrafter_root)
    scorer_summary = run_evalcrafter_scorer(
        generated_artifact_dir=Path(videos_dir),
        output_dir=output_dir,
        evalcrafter_root=evalcrafter_root,
        prompt700_path=prompt700_path,
        limit=args.limit,
        metrics=args.metrics,
        clip_model_path=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
        use_official_metric_prompts=args.prompt_manifest is None,
    )
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        evalcrafter_root=evalcrafter_root,
        videos_dir=videos_dir,
        results_dir=Path(str(scorer_summary.get("results_path"))),
        output_dir=output_dir,
        prompt_manifest=prompt700_path,
        limit=args.limit,
        strict=args.strict,
        json=False,
        run_official=False,
        run_fixture=False,
        check_inputs=False,
        metrics=args.metrics,
        clip_model=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    return normalize_evalcrafter_results(
        normalize_args,
        official_runtime_executed=True,
        scorer_summary=scorer_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("evalcrafter")
        if sample_path is None:
            print("error: no checked-in EvalCrafter sample results found", file=sys.stderr)
            return 2
        args.results_dir = sample_path
    try:
        if args.check_inputs:
            evalcrafter_root = args.evalcrafter_root or resolve_evalcrafter_root()
            if evalcrafter_root is None:
                print("error: EvalCrafter root not found", file=sys.stderr)
                return 2
            if args.videos_dir is None:
                print(
                    "error: --videos-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required with --check-inputs",
                    file=sys.stderr,
                )
                return 2
            input_validation = validate_official_inputs(evalcrafter_root, args.videos_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(args.output_dir / "input_validation.json", input_validation)
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": input_validation["ok"],
                            "benchmark_id": BENCHMARK_ID,
                            "input_validation": input_validation,
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            else:
                status = "ok" if input_validation["ok"] else "failed"
                print(f"evalcrafter input validation: {status}")
                for reason in input_validation["reasons"]:
                    print(f"- {reason}")
            return 0 if input_validation["ok"] else 1
        if args.run_official:
            scorecard = run_official_evalcrafter(args)
        else:
            scorecard = normalize_evalcrafter_results(args)
    except (ImportError, OSError, RuntimeError, ValueError, SyntaxError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": not args.run_official,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_evalcrafter_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": BENCHMARK_ID, "name": DISPLAY_NAME},
            "dataset": {
                "generated_artifact_dir": None if args.videos_dir is None else str(args.videos_dir),
                "prompt_suite": None if args.prompt_manifest is None else str(args.prompt_manifest),
            },
            "eligibility": {
                "full_suite_valid": False,
                "video_coverage_complete": False,
                "leaderboard_valid": False,
                "reasons": [f"EvalCrafter execution failed: {exc}"],
            },
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metrics": 0}},
            "evaluation": {
                "available": False,
                "kind": (
                    "evalcrafter_official_raw_video_metrics"
                    if args.run_official
                    else "evalcrafter_result_normalizer"
                ),
                "blocked_count": len(METRIC_ORDER),
            },
            "validation": {
                "normalizer_only": not args.run_official,
                "official_runtime_executed": False,
                "official_results_imported": False,
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
            "official_results_imported": False,
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    run_failed = scorecard["run"].get("status") == "failed" or scorecard["run"].get("returncode") not in {
        None,
        0,
    }
    result = {
        "ok": bool(scorecard["evaluation"]["available"]) and not run_failed,
        "benchmark_id": BENCHMARK_ID,
        "output_dir": str(args.output_dir),
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "leaderboard_valid": scorecard["leaderboard_valid"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
        "artifacts": scorecard["artifacts"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = (
            "failed"
            if not result["ok"]
            else "full official benchmark verified"
            if result["official_benchmark_verified"]
            else "official raw-video metric subset completed"
            if result["integration_evidence"]
            else "official results normalized"
        )
        print(f"evalcrafter: {status}")
        print(f"output_dir: {result['output_dir']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
