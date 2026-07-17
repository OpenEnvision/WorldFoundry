#!/usr/bin/env python3
"""WorldFoundry runner for Physics-IQ Original and Physics-IQ Verified."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_metrics import (
    load_official_results,
    metric_specs,
    metric_values_from_scores,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
    CANONICAL_PROMPT_COUNT,
    resolve_descriptions_path,
    video_stem_for_record,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_runtime import (
    PhysicsIQRunConfig,
    discover_physics_iq_results,
    run_physics_iq_evaluation,
    select_complete_scenario_records,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.protocols import PhysicsIQProtocolSpec, resolve_protocol

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Physics-IQ Original or Verified in tree.")
    parser.add_argument("--benchmark-id", default="physics-iq")
    parser.add_argument("--protocol", choices=("original", "verified"))
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--raw-metrics-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--physics-iq-root", dest="dataset_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--descriptions-file", type=Path)
    parser.add_argument("--generated-mask-dir", type=Path)
    parser.add_argument("--n-processes", type=int, default=0)
    parser.add_argument("--mask-threshold", type=int, default=10)
    parser.add_argument("--skip-video-validation", action="store_true")
    parser.add_argument("--lazy-integrity", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=os.environ.get("WORLDFOUNDRY_BENCHMARK_LIMIT"),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _coverage(expected_stems: set[str], generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None or not generated_dir.is_dir():
        return {
            "provided": False,
            "expected_count": len(expected_stems),
            "actual_count": 0,
            "matched_count": 0,
            "complete": False,
            "missing_ids": sorted(expected_stems)[:50],
        }
    actual = {
        path.stem
        for path in generated_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    # Model runtimes may preserve only the official four-digit ID prefix.
    expected_by_prefix = {stem[:4]: stem for stem in expected_stems}
    actual_prefixes = {stem[:4] for stem in actual}
    matched = {stem for prefix, stem in expected_by_prefix.items() if prefix in actual_prefixes}
    missing = expected_stems - matched
    return {
        "provided": True,
        "expected_count": len(expected_stems),
        "actual_count": len(actual),
        "matched_count": len(matched),
        "complete": not missing and len(actual_prefixes) >= len(expected_stems),
        "missing_ids": sorted(missing)[:50],
    }


def _metric_rows(
    scores: Mapping[str, Any],
    spec: PhysicsIQProtocolSpec,
    *,
    source_path: Path,
    runtime_executed: bool,
) -> list[dict[str, Any]]:
    values = metric_values_from_scores(scores, spec)
    specs = metric_specs(spec)
    rows: list[dict[str, Any]] = []
    for metric_id, value in values.items():
        meta = specs[metric_id]
        normalized = value
        if value is not None and metric_id != "physics_iq_mse_penalty":
            normalized = max(0.0, min(1.0, value))
        rows.append(
            {
                "metric_id": metric_id,
                "name": meta["name"],
                "available": value is not None,
                "raw_score": value,
                "normalized_score": normalized,
                "score": value,
                "higher_is_better": meta["higher_is_better"],
                "group": meta["group"],
                "primary": meta["primary"],
                "source": "physics_iq_in_tree_runtime" if runtime_executed else "physics_iq_official_results",
                "source_path": str(source_path.resolve()),
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric_id", "score", "normalized_score"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric_id": row["metric_id"],
                    "score": row["score"],
                    "normalized_score": row["normalized_score"],
                }
            )


def normalize_physics_iq_results(
    args: argparse.Namespace,
    *,
    spec: PhysicsIQProtocolSpec | None = None,
    official_runtime_executed: bool = False,
    scorer_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec or resolve_protocol(benchmark_id=args.benchmark_id, protocol=args.protocol)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions_path = resolve_descriptions_path(explicit=args.descriptions_file, spec=spec)
    prompt_records, _selected_scenarios = select_complete_scenario_records(
        descriptions_path=descriptions_path,
        protocol=spec,
        limit=args.limit,
    )

    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    results_path = args.official_results_path or args.raw_metrics_path
    if results_path is None:
        env_result = _env_path("WORLDFOUNDRY_PHYSICS_IQ_RESULTS_PATH")
        results_path = env_result or discover_physics_iq_results([output_dir, generated_dir] if generated_dir else [output_dir])
    if results_path is None:
        raise ValueError("Provide --official-results-path/--raw-metrics-path or use --run-official.")

    scores, scenario_rows = load_official_results(
        results_path,
        spec,
        lazy_integrity=args.lazy_integrity,
    )
    rows = _metric_rows(
        scores,
        spec,
        source_path=results_path,
        runtime_executed=official_runtime_executed,
    )
    expected_stems = {video_stem_for_record(record) for record in prompt_records}
    coverage = _coverage(expected_stems, generated_dir)
    all_metrics = all(row["available"] for row in rows)
    full_suite = (
        len(prompt_records) == CANONICAL_PROMPT_COUNT
        and coverage["complete"]
        and all_metrics
        and (not scenario_rows or len(scenario_rows) == 66)
    )
    primary = next((row for row in rows if row["primary"]), None)
    normalization_ok = primary is not None and primary["available"]
    dataset_root = (scorer_summary or {}).get("dataset_root") or args.dataset_root
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_runtime_executed and full_suite,
        "integration_evidence": official_runtime_executed and normalization_ok,
        "leaderboard_valid": official_runtime_executed and full_suite,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_physics_iq_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "scorer_summary": dict(scorer_summary or {}),
        },
        "benchmark": {
            "benchmark_id": spec.benchmark_id,
            "name": spec.display_name,
            "protocol": spec.protocol,
        },
        "dataset": {
            "dataset_root": None if dataset_root is None else str(Path(dataset_root).expanduser().resolve()),
            "descriptions_path": str(descriptions_path.resolve()),
            "generated_artifact_dir": None if generated_dir is None else str(generated_dir.resolve()),
            "prompt_count": len(prompt_records),
        },
        "metrics": {
            "leaderboard": {
                row["metric_id"]: row["normalized_score"] for row in rows if row["available"]
            },
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {
                "available_metric_count": sum(row["available"] for row in rows),
                "declared_metric_count": len(rows),
                "primary_metric_id": spec.primary_metric_id,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "physics_iq_official_in_tree" if official_runtime_executed else "physics_iq_result_normalizer",
            "blocked_count": sum(not row["available"] for row in rows),
        },
        "eligibility": {"full_suite_valid": full_suite, "video_coverage_complete": coverage["complete"]},
        "coverage": {"videos": coverage, "scenario_count": len(scenario_rows)},
        "artifacts": {
            "scorecard": str(output_dir / "scorecard.json"),
            "official_results_path": str(results_path.resolve()),
            "raw_metrics_path": (scorer_summary or {}).get("raw_metrics_path"),
            "official_metrics_path": (scorer_summary or {}).get("results_path"),
        },
        "descriptions_path": str(descriptions_path),
        "prompt_count": len(prompt_records),
    }
    if args.strict and not full_suite:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", scenario_rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": spec.benchmark_id,
            "protocol": spec.protocol,
            "descriptions_path": str(descriptions_path),
            "official_results_path": str(results_path.resolve()),
            "metric_ids": list(metric_specs(spec)),
            "prompt_count": len(prompt_records),
        },
    )
    _write_summary_csv(output_dir / "results_summary.csv", rows)
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_physics_iq(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_protocol(benchmark_id=args.benchmark_id, protocol=args.protocol)
    descriptions_path = resolve_descriptions_path(explicit=args.descriptions_file, spec=spec)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None and args.raw_metrics_path is None:
        raise ValueError("--generated-artifact-dir is required unless --raw-metrics-path is supplied.")
    summary = run_physics_iq_evaluation(
        PhysicsIQRunConfig(
            protocol=spec,
            dataset_root=args.dataset_root,
            descriptions_path=descriptions_path,
            generated_video_dir=generated_dir or Path("."),
            output_dir=args.output_dir,
            generated_mask_dir=args.generated_mask_dir,
            raw_metrics_path=args.raw_metrics_path,
            n_processes=args.n_processes,
            mask_threshold=args.mask_threshold,
            validate_videos=not args.skip_video_validation,
            lazy_integrity=args.lazy_integrity,
            limit=args.limit,
        )
    )
    normalize_args = argparse.Namespace(**vars(args))
    normalize_args.official_results_path = Path(summary["raw_metrics_path"])
    normalize_args.descriptions_file = descriptions_path
    return normalize_physics_iq_results(
        normalize_args,
        spec=spec,
        official_runtime_executed=True,
        scorer_summary=summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scorecard = run_official_physics_iq(args) if args.run_official else normalize_physics_iq_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "normalization_ok": False,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "run": {
                "status": "failed",
                "returncode": 1,
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_physics_iq_official_runner",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {
                "benchmark_id": args.benchmark_id,
                "name": "Physics-IQ",
                "protocol": args.protocol,
            },
            "dataset": {
                "dataset_root": None if args.dataset_root is None else str(args.dataset_root),
                "generated_artifact_dir": (
                    None if args.generated_artifact_dir is None else str(args.generated_artifact_dir)
                ),
            },
            "metrics": {
                "leaderboard": {},
                "per_metric": {},
                "summary": {"available_metric_count": 0},
            },
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"physics-iq: failed ({exc})", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"ok": True, **scorecard}, indent=2, ensure_ascii=False))
    else:
        print(f"{scorecard['benchmark']['name']}: {scorecard['metrics']['leaderboard']}")
    return 0 if scorecard["run"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
