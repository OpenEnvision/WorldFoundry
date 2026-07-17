#!/usr/bin/env python3
"""Official runner implementation for GenAI-Bench."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric, utc_now_iso, write_json
from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_metrics import (
    evaluate_genai_preference_rows,
)
from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_prompts import (
    resolve_genai_bench_assets_root,
    resolve_preference_pairs_path,
)
from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_runtime import (
    run_genai_bench_scorer,
    scorer_config_from_env,
)
from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_video_quality_contract import (
    GENAI_TASK_METRICS,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

BENCHMARK_ID = "genai-bench"
DISPLAY_NAME = "GenAI-Bench"

CONFIG = ors.build_runner_config_from_contract(
    BENCHMARK_ID,
    root_env="WORLDFOUNDRY_T2V_METRICS_ROOT",
    results_path_env="WORLDFOUNDRY_GENAI_BENCH_RESULTS_PATH",
    default_repo_subdir="",
    official_output_globs=("*genai*preference*.jsonl", "*genai*results*.jsonl", "genai_bench*.jsonl"),
    metric_groups={
        "pairwise_accuracy": "pairwise",
        "image_generation_preference_accuracy": "image_generation",
        "image_editing_preference_accuracy": "image_editing",
        "video_preference_accuracy": "video_generation",
        "genai_bench_average": "aggregate",
    },
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize the official GenAI-Bench benchmark runtime.")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--official-results-path", dest="results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Execute in-tree GenAI-Bench scorer (mock or VQAScore backend).",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--preference-pairs-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    default_task = next(
        (
            part
            for part in results_path.parts
            if part in {"image_generation", "image_edition", "image_editing", "video_generation"}
        ),
        None,
    )
    preference_metrics = evaluate_genai_preference_rows(rows, default_task=default_task)
    if preference_metrics["num_total"] <= 0:
        return ors.extract_tabular_official_metrics(payload, results_path, CONFIG)

    scores = {"pairwise_accuracy": preference_metrics["pairwise_accuracy"]}
    task_scores: list[float] = []
    for task, metric_id in GENAI_TASK_METRICS.items():
        task_result = preference_metrics["per_task"].get(task)
        if not task_result:
            continue
        score = float(task_result["accuracy"])
        scores[metric_id] = score
        task_scores.append(score)
    if task_scores:
        scores["genai_bench_average"] = mean_numeric(task_scores)

    extracted: dict[str, dict[str, Any]] = {}
    for metric_id, score in scores.items():
        if score is None:
            continue
        extracted[metric_id] = ors.metric_row(
            metric_id,
            score,
            source="genai_preference_pairs",
            sample_count=int(preference_metrics["num_total"]),
        )
    return extracted


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir]
    if repo_root is not None:
        search_roots.append(repo_root)
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)


def _resolve_results_path(args: argparse.Namespace, scorer_summary: dict[str, Any] | None = None) -> Path:
    if scorer_summary is not None and scorer_summary.get("results_path"):
        return Path(str(scorer_summary["results_path"]))
    if args.results_path is not None:
        return args.results_path.expanduser().resolve()
    env_path = _env_path("WORLDFOUNDRY_GENAI_BENCH_RESULTS_PATH")
    if env_path is not None:
        return env_path
    raise ValueError(
        "--official-results-path, WORLDFOUNDRY_GENAI_BENCH_RESULTS_PATH, or --run-official/--run-fixture is required"
    )


def normalize_genai_bench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    scorer_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results_path = _resolve_results_path(args, scorer_summary)
    payload, _fmt = ors.load_upstream_payload(results_path)
    extracted = extract_metrics(payload, results_path)
    extracted = ors.catalog_fallback(CONFIG, results_path, extracted)
    scorecard = ors.build_scorecard(
        config=CONFIG,
        output_dir=args.output_dir,
        results_path=results_path,
        extracted=extracted,
        command=None,
        duration_seconds=None,
        returncode=0,
        repo_root=resolve_genai_bench_assets_root(),
        generated_video_dir=args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
    )
    if official_runtime_executed and scorer_summary is not None:
        scorecard.setdefault("run", {})["scorer_summary"] = scorer_summary
        scorecard["run"]["official_runtime_executed"] = True
    write_json(args.output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_genai_bench(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer_summary = run_genai_bench_scorer(
        output_dir=args.output_dir / "upstream",
        config=scorer_config_from_env(),
        preference_pairs_path=args.preference_pairs_path,
        limit=args.limit,
    )
    return normalize_genai_bench_results(
        args,
        official_runtime_executed=True,
        scorer_summary=scorer_summary,
    )


def run_fixture_genai_bench(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_source = benchmark_task_sample_path(BENCHMARK_ID)
    if not fixture_source.is_file():
        fixture_source = resolve_preference_pairs_path()
    fixture_copy = args.output_dir / "upstream" / fixture_source.name
    fixture_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture_source, fixture_copy)
    args.results_path = fixture_copy
    return normalize_genai_bench_results(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.generated_artifact_dir is not None:
        args.generated_artifact_dir = args.generated_artifact_dir.expanduser().resolve()
    if args.preference_pairs_path is not None:
        args.preference_pairs_path = args.preference_pairs_path.expanduser().resolve()

    try:
        if args.run_official:
            scorecard = run_official_genai_bench(args)
        elif args.run_fixture:
            scorecard = run_fixture_genai_bench(args)
        else:
            scorecard = normalize_genai_bench_results(args)
    except (OSError, ValueError, RuntimeError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": ors.SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": not args.run_official,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_genai_bench_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": BENCHMARK_ID, "name": DISPLAY_NAME},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metrics": 0}},
            "evaluation": {"available": False, "kind": "genai_bench_result_normalizer", "blocked_count": 1},
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": bool(scorecard.get("evaluation", {}).get("available")),
        "benchmark_id": BENCHMARK_ID,
        "output_dir": str(args.output_dir),
        "official_benchmark_verified": scorecard.get("official_benchmark_verified"),
        "leaderboard_valid": scorecard.get("leaderboard_valid"),
        "integration_evidence": scorecard.get("integration_evidence"),
        "normalization_ok": scorecard.get("normalization_ok"),
        "official_results_imported": scorecard.get("official_results_imported"),
        "artifacts": scorecard.get("artifacts"),
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"genai-bench normalization ok={scorecard.get('normalization_ok')}")
    return 0 if scorecard.get("normalization_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
