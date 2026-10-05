#!/usr/bin/env python3
"""Run the Apple-PI evaluator in WorldFoundry's tree.

``--run-official`` means the Apple-PI protocol is executed by the checked-in
WorldFoundry runtime.  An existing official-compatible JSON can still be
imported with ``--official-results-path`` for reproducibility and comparison.
"""

from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path, SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl

BENCHMARK_ID = "apple-pi"
SUBTRACKS = (
    "perception_text",
    "perception_graphic",
    "formulation_text",
    "formulation_graphic",
    "deduction",
)
METRIC_ORDER = (*SUBTRACKS, "apple_pi_average")
METRIC_SPECS = {
    "perception_text": ("Perception-Text", "perception"),
    "perception_graphic": ("Perception-Graphic", "perception"),
    "formulation_text": ("Formulation-Text", "formulation"),
    "formulation_graphic": ("Formulation-Graphic", "formulation"),
    "deduction": ("Deduction", "deduction"),
    "apple_pi_average": ("Apple-PI Average", "aggregate"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--pred-dir", "--prediction-dir", "--generated-artifact-dir", dest="pred_dir", type=Path)
    parser.add_argument("--protocol", choices=("video", "image"))
    parser.add_argument("--subtrack", choices=("all", *SUBTRACKS), default="all")
    parser.add_argument("--gemini-model", default=os.environ.get("APPLE_PI_GEMINI_MODEL", "gemini-3-flash-preview"))
    parser.add_argument("--judge-backend", default=os.environ.get("WORLDFOUNDRY_APPLE_PI_JUDGE_BACKEND", "gemini"))
    parser.add_argument("--no-foundation-models", action="store_true", help="Skip native SAM3/MoGe adapters; useful for protocol smoke tests.")
    parser.add_argument("--strict", action="store_true", help="Require all five subtracks and all three rollouts.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)




def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _result_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("apple_pi_results.json", "results.json", "evaluation.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        candidates = sorted(path.glob("*.json"))
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f"Apple-PI result JSON not found: {path}")


def _selected_subtracks(args: argparse.Namespace) -> tuple[str, ...]:
    return SUBTRACKS if args.subtrack == "all" else (args.subtrack,)


def _summary(payload: Mapping[str, Any], subtracks: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw_summary = payload.get("summary")
    cases = payload.get("cases") if isinstance(payload.get("cases"), Mapping) else {}
    summary: dict[str, dict[str, Any]] = {}
    for subtrack in subtracks:
        entry = raw_summary.get(subtrack) if isinstance(raw_summary, Mapping) else None
        if isinstance(entry, Mapping) and _number(entry.get("mean")) is not None:
            summary[subtrack] = {"mean": _number(entry.get("mean")), "num_cases": int(entry.get("num_cases") or 0)}
            continue
        values = []
        for case in cases.values():
            item = case.get(subtrack) if isinstance(case, Mapping) else None
            aggregate = item.get("aggregate") if isinstance(item, Mapping) else None
            score = _number(aggregate.get("mean")) if isinstance(aggregate, Mapping) else None
            if score is None and isinstance(item, Mapping):
                scores = [_number(row.get("score")) for row in item.get("rollouts", ()) if isinstance(row, Mapping)]
                scores = [score for score in scores if score is not None]
                score = statistics.mean(scores) if scores else None
            if score is not None:
                values.append(score)
        summary[subtrack] = {"mean": statistics.mean(values) if values else None, "num_cases": len(values)}
    values = [entry["mean"] for entry in summary.values() if entry["mean"] is not None]
    summary["apple_pi_average"] = {"mean": statistics.mean(values) if values else None, "num_subtracks": len(values)}
    return summary


def _rollout_coverage(payload: Mapping[str, Any], subtracks: tuple[str, ...]) -> dict[str, Any]:
    cases = payload.get("cases") if isinstance(payload.get("cases"), Mapping) else {}
    expected = len(cases) * len(subtracks) * 3
    successful = 0
    for case in cases.values():
        if not isinstance(case, Mapping):
            continue
        for subtrack in subtracks:
            item = case.get(subtrack)
            rows = item.get("rollouts", ()) if isinstance(item, Mapping) else ()
            successful += sum(1 for row in rows[:3] if isinstance(row, Mapping) and row.get("status") != "failed" and _number(row.get("score")) is not None)
    return {
        "case_count": len(cases), "expected_rollouts": expected, "successful_rollouts": successful,
        "failed_or_missing_rollouts": max(0, expected - successful), "complete": bool(expected) and successful == expected,
        "required_rollouts_per_case": 3,
    }


def _per_sample_rows(payload: Mapping[str, Any], subtracks: tuple[str, ...]) -> list[dict[str, Any]]:
    cases = payload.get("cases") if isinstance(payload.get("cases"), Mapping) else {}
    rows = []
    for case_id, case in cases.items():
        if not isinstance(case, Mapping):
            continue
        for subtrack in subtracks:
            item = case.get(subtrack)
            aggregate = item.get("aggregate") if isinstance(item, Mapping) else None
            if isinstance(item, Mapping):
                rows.append({
                    "case_id": str(case_id), "subtrack": subtrack,
                    "score": _number(aggregate.get("mean")) if isinstance(aggregate, Mapping) else None,
                    "num_successful_rollouts": aggregate.get("num_successful") if isinstance(aggregate, Mapping) else None,
                    "num_expected_rollouts": aggregate.get("num_expected", 3) if isinstance(aggregate, Mapping) else 3,
                })
    return rows


def _scorecard(args: argparse.Namespace, payload: Mapping[str, Any], results_path: Path, *, native_runtime: Mapping[str, Any] | None) -> dict[str, Any]:
    subtracks = _selected_subtracks(args)
    summary = _summary(payload, subtracks)
    coverage = _rollout_coverage(payload, subtracks)
    rows = []
    for metric_id in METRIC_ORDER:
        name, group = METRIC_SPECS[metric_id]
        score = _number(summary.get(metric_id, {}).get("mean"))
        rows.append({
            "metric_id": metric_id, "name": name, "available": score is not None,
            "raw_score": score, "normalized_score": score, "score": score,
            "higher_is_better": True, "group": group,
            "source": "apple_pi_in_tree_runtime" if native_runtime else "apple_pi_results_json",
            "source_path": str(results_path),
            "reason": None if score is not None else "score_not_available",
        })
    available = [row for row in rows if row["available"]]
    full_suite = set(subtracks) == set(SUBTRACKS) and all(summary[item]["mean"] is not None for item in SUBTRACKS) and coverage["complete"]
    ok = bool(available) and (not args.strict or full_suite)
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": bool(native_runtime) and ok,
        "integration_evidence": bool(native_runtime) and full_suite,
        "leaderboard_valid": False,
        "normalizer_only": native_runtime is None,
        "normalization_ok": ok,
        "official_results_imported": native_runtime is None and bool(available),
        "run": {
            "status": "succeeded" if ok else "failed", "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_apple_pi_in_tree_runner", "returncode": 0 if ok else 1,
            "native_runtime": dict(native_runtime or {}),
        },
        "benchmark": {
            "benchmark_id": args.benchmark_id, "name": "Apple-PI", "version": payload.get("benchmark_version", "1.0"),
            "protocol": payload.get("protocol"), "num_rollouts": payload.get("num_rollouts", 3),
        },
        "dataset": {"dataset_version": payload.get("dataset_version"), "official_results_path": str(results_path), "selected_subtracks": list(subtracks), "rollout_coverage": coverage},
        "eligibility": {"full_suite_valid": full_suite, "leaderboard_valid": False, "reasons": ["Leaderboard validity still requires the benchmark's complete public GT audit."]},
        "metrics": {"leaderboard": {row["metric_id"]: row["score"] for row in available}, "per_metric": {row["metric_id"]: row for row in rows}, "summary": {"available_metric_count": len(available), "declared_metric_count": len(METRIC_ORDER), "case_count": coverage["case_count"]}},
        "evaluation": {"available": ok, "kind": "apple_pi_in_tree_model_backed" if native_runtime else "apple_pi_result_normalizer", "subtracks": list(subtracks), "full_suite_valid": full_suite, "upstream_evaluator": "worldfoundry.apple_pi_runtime"},
        "validation": {"normalizer_only": native_runtime is None, "official_runtime_executed": native_runtime is not None, "official_runtime_succeeded": bool(native_runtime) and ok, "official_results_imported": native_runtime is None and bool(available), "full_suite_complete": full_suite},
        "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve()), "raw_metric_table": str((args.output_dir / "raw_metric_table.jsonl").resolve()), "per_sample_scores": str((args.output_dir / "per_sample_scores.jsonl").resolve()), "benchmark_contract": str((args.output_dir / "benchmark_contract.json").resolve()), "official_results_path": str(results_path)},
        "notes": [
            "Apple-PI is executed in-tree; no Apple-PI source checkout is imported at runtime.",
            "SAM3 and MoGe-2 resolve through WorldFoundry base_models capabilities.",
            "The Apple-PI protocol requires exactly three rollouts per case and subtrack.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "raw_metric_table.jsonl", rows)
    write_jsonl(args.output_dir / "per_sample_scores.jsonl", _per_sample_rows(payload, subtracks))
    write_json(args.output_dir / "benchmark_contract.json", {"benchmark_id": args.benchmark_id, "benchmark_version": payload.get("benchmark_version", "1.0"), "protocol": payload.get("protocol"), "num_rollouts": payload.get("num_rollouts", 3), "subtracks": list(subtracks), "metric_ids": list(METRIC_ORDER), "execution": "in_tree"})
    write_json(args.output_dir / "scorecard.json", scorecard)
    return scorecard


def _failure(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {"schema_version": SCORECARD_SCHEMA_VERSION, "official_benchmark_verified": False, "integration_evidence": False, "leaderboard_valid": False, "normalizer_only": not args.run_official, "normalization_ok": False, "official_results_imported": False, "run": {"status": "failed", "started_at": utc_now_iso(), "runner": "benchmark_zoo_apple_pi_in_tree_runner", "returncode": 1, "error": f"{type(exc).__name__}: {exc}"}, "benchmark": {"benchmark_id": args.benchmark_id, "name": "Apple-PI"}, "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}}, "evaluation": {"available": False, "kind": "apple_pi_in_tree_model_backed" if args.run_official else "apple_pi_result_normalizer"}, "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())}}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        native_runtime = None
        if args.run_official:
            # Catalog-documented env fallbacks for the GT and prediction roots.
            if args.gt_dir is None:
                args.gt_dir = resolve_env_path("WORLDFOUNDRY_APPLE_PI_GT_DIR")
            if args.pred_dir is None:
                args.pred_dir = resolve_env_path("WORLDFOUNDRY_APPLE_PI_PREDICTION_DIR")
            if args.gt_dir is None or args.pred_dir is None:
                raise ValueError("--run-official requires --gt-dir and --pred-dir")
            from worldfoundry.evaluation.tasks.execution.runners.apple_pi.apple_pi_runtime import evaluate_native_apple_pi
            result_path = args.output_dir / "apple_pi_results.json"
            evaluate_native_apple_pi(
                gt_root=args.gt_dir.expanduser().resolve(), prediction_root=args.pred_dir.expanduser().resolve(), output_path=result_path,
                protocol=args.protocol, subtracks=_selected_subtracks(args), judge_model=args.gemini_model,
                judge_backend=args.judge_backend, enable_foundation_models=not args.no_foundation_models,
            )
            native_runtime = {"kind": "worldfoundry_in_tree", "runner": "apple_pi_runtime", "judge_backend": args.judge_backend, "foundation_models": not args.no_foundation_models}
        else:
            result_path = args.official_results_path or resolve_env_path("WORLDFOUNDRY_APPLE_PI_RESULTS_PATH")
            if result_path is None:
                raise ValueError("--official-results-path or WORLDFOUNDRY_APPLE_PI_RESULTS_PATH is required")
        result_path = _result_path(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Apple-PI result must be a JSON object: {result_path}")
        scorecard = _scorecard(args, payload, result_path, native_runtime=native_runtime)
    except Exception as exc:  # noqa: BLE001
        scorecard = _failure(args, exc)
        write_json(args.output_dir / "scorecard.json", scorecard)
    output = {"ok": scorecard["normalization_ok"], "output_dir": str(args.output_dir), **scorecard}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif output["ok"]:
        print(f"apple-pi: {scorecard['evaluation']['kind']}")
    else:
        print(f"apple-pi: failed ({scorecard['run'].get('error', 'incomplete results')})", file=sys.stderr)
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
