#!/usr/bin/env python3
"""Official runner for the LikePhys intuitive-physics likelihood-preference benchmark.

Two surfaces are provided:

``--run-official``
    Drive the upstream ``evaluator.py`` probe over a caller-supplied LikePhys checkout and
    dataset, then normalize the ``results_<model>.json`` artifacts it writes. This is the
    only path that produces the benchmark's own evidence, and it needs a CUDA device plus
    the probed diffusion checkpoints.

default (normalizer)
    Recompute the official mis-rank metrics from existing ``results_<model>.json``
    artifacts supplied through ``--official-results-path`` or the environment.

Mis-rank is an error rate over (valid, impossible) clip pairs, so lower is better for
every metric this runner reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
    SCORECARD_SCHEMA_VERSION,
    resolve_env_path,
)
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_likephys_metrics,
    load_scenario_results,
)
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_runtime import (
    LikePhysRuntimeError,
    inspect_dataset,
    probe_config_from_env,
    resolve_checkout_root,
    resolve_dataset_root,
    run_likephys_probe,
)
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_scenarios import (
    CANONICAL_SCENARIO_COUNT,
    CANONICAL_SUBGROUP_COUNT,
    DEFAULT_SEED,
    DEFAULT_TIMESTEP_NUM,
    DISPLAY_NAME,
    OFFICIAL_SCENARIO_SWEEP,
    SCENARIO_ORDER,
)
from worldfoundry.evaluation.utils import REPO_ROOT, benchmark_task_sample_path

RUNNER_NAME = "benchmark_zoo_likephys_official_runner"
EVALUATION_KIND = "likephys_misrank_normalizer"
OFFICIAL_EVALUATION_KIND = "likephys_official_probe"
RESULT_IMPORT_SCOPE = "result_artifact_import_only"
OFFICIAL_RUN_SCOPE = "official_probe_execution"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize LikePhys official outputs.")
    parser.add_argument("--benchmark-id", default="likephys")
    parser.add_argument(
        "--official-results-path",
        "--results-root",
        dest="official_results_path",
        type=Path,
        help="results_<model>.json file, or an experiment directory holding <scenario>/results_<model>.json.",
    )
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Execute the upstream LikePhys ELBO probe before normalizing its results.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
        help="Optional extra search root for existing LikePhys result artifacts.",
    )
    parser.add_argument("--likephys-root", type=Path, help="Official LikePhys checkout used by --run-official.")
    parser.add_argument("--benchmark-data-root", type=Path, help="Directory holding <scenario>_videos clip folders.")
    parser.add_argument(
        "--probe-model",
        "--result-model-id",
        dest="probe_model",
        help="LikePhys probe backend key, for example wan2.1-T2V-1.3b.",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=sorted(SCENARIO_ORDER),
        help="Restrict the run to one scenario; repeatable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timestep-num", type=int, default=DEFAULT_TIMESTEP_NUM)
    parser.add_argument(
        "--no-guidance-scale",
        dest="guidance_scale",
        action="store_false",
        help="Probe without classifier-free guidance; the official sweep keeps it enabled.",
    )
    parser.add_argument("--tag-name", dest="tag_name")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --run-official, prepare the probe workspace and report commands without executing them.",
    )
    parser.add_argument(
        "--no-report-filter",
        dest="apply_reported_filter",
        action="store_false",
        help="Keep the variations the paper excludes from its reported aggregates.",
    )
    parser.add_argument("--limit", type=int, help="Score at most N scenarios.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(guidance_scale=True, apply_reported_filter=True)
    return parser.parse_args(argv)




def _strict_enabled(args: argparse.Namespace) -> bool:
    return args.strict or os.environ.get("WORLDFOUNDRY_LIKEPHYS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_results_path(args: argparse.Namespace) -> Path | None:
    candidates = [
        args.official_results_path,
        resolve_env_path("WORLDFOUNDRY_LIKEPHYS_RESULTS_PATH"),
        args.generated_artifact_dir,
        resolve_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
    ]
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = metrics.get(metric_id)
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
                "source": "likephys_official_probe" if official_runtime_executed else "likephys_imported_results",
                "source_path": str(source_path),
                "evidence_scope": OFFICIAL_RUN_SCOPE if official_runtime_executed else RESULT_IMPORT_SCOPE,
                "official_runtime_executed": official_runtime_executed,
                "components": dict(components),
                "reason": None if score is not None else "metric_not_computable_from_supplied_results",
            }
        )
    return rows


def _per_scenario_rows(computed: Mapping[str, Any]) -> list[dict[str, Any]]:
    per_scenario = computed.get("per_scenario")
    if not isinstance(per_scenario, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for scenario_id, summary in per_scenario.items():
        variations = summary.get("variations") if isinstance(summary.get("variations"), Mapping) else {}
        for variation, payload in variations.items():
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "variation": variation,
                    "misrank_rate": payload.get("misrank_rate"),
                    "subgroup_count": payload.get("subgroup_count"),
                    "pair_count": payload.get("pair_count"),
                    "source_path": summary.get("source_path"),
                }
            )
    return rows


def _scenario_coverage(computed: Mapping[str, Any]) -> dict[str, Any]:
    per_scenario = computed.get("per_scenario") if isinstance(computed.get("per_scenario"), Mapping) else {}
    scored = sorted(str(scenario_id) for scenario_id in per_scenario)
    missing = [scenario_id for scenario_id in SCENARIO_ORDER if scenario_id not in per_scenario]
    complete_subgroups = all(
        summary.get("subgroup_count", 0) >= CANONICAL_SUBGROUP_COUNT
        for summary in per_scenario.values()
        if isinstance(summary, Mapping)
    )
    return {
        "expected_scenario_count": CANONICAL_SCENARIO_COUNT,
        "scored_scenario_count": len(scored),
        "scored_scenarios": scored,
        "missing_scenarios": missing,
        "all_subgroups_present": bool(per_scenario) and complete_subgroups,
        "complete": not missing and bool(per_scenario) and complete_subgroups,
    }


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path,
    metric_rows: list[dict[str, Any]],
    scenario_coverage: Mapping[str, Any],
    probe_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    normalization_ok = bool(available_rows)
    coverage_complete = scenario_coverage.get("complete") is True
    full_suite_complete = (
        official_runtime_executed
        and coverage_complete
        and len(available_rows) == len(METRIC_ORDER)
        and bool(probe_summary and probe_summary.get("complete"))
    )
    evidence_scope = OFFICIAL_RUN_SCOPE if official_runtime_executed else RESULT_IMPORT_SCOPE
    notes = [
        "LikePhys probes a video diffusion model directly; mis-rank is an error rate, so lower is better.",
        "Reported aggregates drop the variations excluded by the official read_exp_final.py filter_config.",
    ]
    if not official_runtime_executed:
        notes.append(
            "This run normalized supplied results_<model>.json artifacts and did not execute the ELBO probe."
        )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalization_ok,
        "evidence_scope": evidence_scope,
        "eligibility": {
            "full_suite_valid": full_suite_complete,
            "leaderboard_valid": False,
            "scenario_coverage_complete": coverage_complete,
            "official_probe_executed": official_runtime_executed,
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0 if normalization_ok else 1,
            "official_runtime_executed": official_runtime_executed,
            "probe_summary": dict(probe_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": DISPLAY_NAME},
        "dataset": {
            "upstream_results": str(results_path.resolve()),
            "probe_models": list(components.get("probe_models") or []),
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
            "kind": OFFICIAL_EVALUATION_KIND if official_runtime_executed else EVALUATION_KIND,
            "evidence_scope": evidence_scope,
            "importer_only": not official_runtime_executed,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "validation": {
            "normalizer_only": not official_runtime_executed,
            "official_runtime_executed": official_runtime_executed,
            "official_runtime_succeeded": bool(probe_summary and probe_summary.get("complete")),
            "official_results_imported": normalization_ok,
            "full_suite_complete": full_suite_complete,
            "scope": evidence_scope,
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(results_path.resolve()),
        },
        "coverage": {"scenarios": dict(scenario_coverage)},
        "notes": notes,
    }


def normalize_likephys_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    probe_summary: Mapping[str, Any] | None = None,
    results_path_override: Path | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_path_override or _resolve_results_path(args)
    if results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_LIKEPHYS_RESULTS_PATH, or --run-official is required"
        )
    scenario_results = load_scenario_results(Path(results_path), model_key=args.probe_model)
    if args.scenarios:
        selected = set(args.scenarios)
        scenario_results = {key: value for key, value in scenario_results.items() if key in selected}
        if not scenario_results:
            raise ValueError(f"no LikePhys results found for the selected scenarios: {sorted(selected)}")
    if args.limit is not None:
        ordered = [key for key in SCENARIO_ORDER if key in scenario_results]
        ordered.extend(key for key in scenario_results if key not in SCENARIO_ORDER)
        scenario_results = {key: scenario_results[key] for key in ordered[: int(args.limit)]}

    computed = compute_likephys_metrics(
        scenario_results=scenario_results,
        apply_reported_filter=args.apply_reported_filter,
    )
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(results_path),
        official_runtime_executed=official_runtime_executed,
    )
    scenario_coverage = _scenario_coverage(computed)
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        results_path=Path(results_path),
        metric_rows=metric_rows,
        scenario_coverage=scenario_coverage,
        probe_summary=probe_summary,
        official_runtime_executed=official_runtime_executed,
        components=components,
    )
    if _strict_enabled(args) and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", _per_scenario_rows(computed))
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "official_results_path": str(results_path),
            "metric_ids": list(METRIC_ORDER),
            "scenario_ids": list(SCENARIO_ORDER),
            "scored_scenarios": scenario_coverage["scored_scenarios"],
            "probe_models": list(components.get("probe_models") or []),
            "reported_variation_filter_applied": args.apply_reported_filter,
        },
    )
    write_json(output_dir / "misrank_breakdown.json", computed.get("per_scenario", {}))
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_likephys(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.probe_model:
        raise LikePhysRuntimeError("--probe-model is required for --run-official")
    checkout_root = resolve_checkout_root(args.likephys_root, repo_root=REPO_ROOT)
    dataset_root = resolve_dataset_root(args.benchmark_data_root, checkout_root=checkout_root)
    if dataset_root is None:
        raise LikePhysRuntimeError(
            "LikePhys dataset root not found. Download JianhaoDYDY/LikePhys-Benchmark and pass "
            "--benchmark-data-root or set WORLDFOUNDRY_LIKEPHYS_DATA_ROOT."
        )
    scenarios = tuple(args.scenarios or OFFICIAL_SCENARIO_SWEEP)
    if args.limit is not None:
        scenarios = scenarios[: int(args.limit)]
    config = probe_config_from_env(
        probe_model=args.probe_model,
        checkout_root=checkout_root,
        dataset_root=dataset_root,
        scenarios=scenarios,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        tag_name=args.tag_name,
        timestep_num=args.timestep_num,
        timeout_seconds=args.timeout_seconds,
    )
    dataset_report = inspect_dataset(dataset_root, scenarios)
    write_json(output_dir / "dataset_inventory.json", dataset_report)
    probe_summary = run_likephys_probe(config=config, output_dir=output_dir, dry_run=args.dry_run)
    probe_summary["dataset_inventory"] = {
        "available_scenario_count": dataset_report["available_scenario_count"],
        "complete": dataset_report["complete"],
    }
    write_json(output_dir / "probe_summary.json", probe_summary)
    if args.dry_run:
        scorecard = _dry_run_scorecard(args, probe_summary=probe_summary, dataset_report=dataset_report)
        write_json(output_dir / "scorecard.json", scorecard)
        return scorecard
    return normalize_likephys_results(
        args,
        official_runtime_executed=True,
        probe_summary=probe_summary,
        results_path_override=Path(probe_summary["results_root"]),
    )


def _dry_run_scorecard(
    args: argparse.Namespace,
    *,
    probe_summary: Mapping[str, Any],
    dataset_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Scorecard for ``--run-official --dry-run``: a validated plan, deliberately score-free."""
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": False,
        "normalization_ok": False,
        "official_results_imported": False,
        "evidence_scope": "probe_plan_only",
        "eligibility": {"full_suite_valid": False, "leaderboard_valid": False, "official_probe_executed": False},
        "run": {
            "status": "dry_run",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0,
            "official_runtime_executed": False,
            "probe_summary": dict(probe_summary),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": DISPLAY_NAME},
        "dataset": {
            "dataset_root": dataset_report.get("dataset_root"),
            "available_scenarios": list(dataset_report.get("available_scenarios") or []),
            "probe_models": [args.probe_model] if args.probe_model else [],
        },
        "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
        "evaluation": {
            "available": False,
            "kind": OFFICIAL_EVALUATION_KIND,
            "evidence_scope": "probe_plan_only",
            "importer_only": False,
            "blocked_count": len(METRIC_ORDER),
        },
        "validation": {
            "normalizer_only": False,
            "official_runtime_executed": False,
            "official_runtime_succeeded": False,
            "official_results_imported": False,
            "full_suite_complete": False,
            "scope": "probe_plan_only",
        },
        "artifacts": {
            "scorecard": str((args.output_dir / "scorecard.json").resolve()),
            "probe_summary": str((args.output_dir / "probe_summary.json").resolve()),
            "dataset_inventory": str((args.output_dir / "dataset_inventory.json").resolve()),
        },
        "notes": [
            "--dry-run validated the dataset layout and prepared the probe workspace and command plan.",
            "No model was probed, so no metric was produced.",
        ],
    }


def _failure_scorecard(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": True,
        "normalization_ok": False,
        "official_results_imported": False,
        "evidence_scope": RESULT_IMPORT_SCOPE,
        "run": {
            "status": "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": DISPLAY_NAME},
        "dataset": {},
        "eligibility": {"full_suite_valid": False, "leaderboard_valid": False},
        "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
        "evaluation": {
            "available": False,
            "kind": EVALUATION_KIND,
            "evidence_scope": RESULT_IMPORT_SCOPE,
            "importer_only": True,
            "blocked_count": len(METRIC_ORDER),
        },
        "validation": {
            "normalizer_only": True,
            "official_runtime_executed": False,
            "official_runtime_succeeded": False,
            "official_results_imported": False,
            "full_suite_complete": False,
            "scope": RESULT_IMPORT_SCOPE,
        },
        "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("likephys")
        if sample_path is None:
            print("error: no checked-in LikePhys sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        if args.run_official:
            scorecard = run_official_likephys(args)
        else:
            scorecard = normalize_likephys_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = _failure_scorecard(args, exc)
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"likephys: failed ({exc})", file=sys.stderr)
        return 1

    dry_run = scorecard.get("run", {}).get("status") == "dry_run"
    payload = {
        "ok": scorecard.get("normalization_ok") is True or dry_run,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        **scorecard,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif dry_run:
        planned = scorecard.get("run", {}).get("probe_summary", {}).get("scenario_count", 0)
        print(f"likephys: dry run prepared {planned} probe command(s); no model was probed")
    else:
        primary = scorecard.get("metrics", {}).get("leaderboard", {}).get("likephys_misrank_rate")
        rendered = "n/a" if primary is None else f"{primary:.4f}"
        print(f"likephys: {scorecard.get('evaluation', {}).get('kind', 'unknown')} misrank={rendered} (lower is better)")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
