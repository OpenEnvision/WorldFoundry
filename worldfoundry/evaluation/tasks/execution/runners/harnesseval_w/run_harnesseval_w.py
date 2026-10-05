"""Run HarnessEval-W's Python evaluator or normalize official reports.

Generated inputs use outputs/<axis>/<family>/<model>/<case>/{output.mp4,metadata.json}.
Supply the official manifest and unmodified skill plans; generation stays in WorldFoundry.
Each skill runs in a bounded Python process so its GPU models are released before the next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from worldfoundry.core.io.paths import package_data_path, package_root
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.metrics import (
    METRIC_IDS,
    PRIMARY_METRIC,
    normalize_results,
)
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.io import read_json
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.protocols import SKILLS

MODULE = "worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.run_harnesseval_w"
REVISION = "ed4ccc6486b8271723ee8baea60d89b32d0a7518"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default="harnesseval-w", choices=["harnesseval-w"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-official", action="store_true")
    mode.add_argument("--run-fixture", action="store_true")
    mode.add_argument("--official-results-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--generated-artifact-dir", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--manifest", type=Path, default=env_path("WORLDFOUNDRY_HARNESSEVAL_MANIFEST"))
    parser.add_argument("--plan-root", type=Path, default=env_path("WORLDFOUNDRY_HARNESSEVAL_PLAN_ROOT"))
    parser.add_argument(
        "--assets-root", type=Path, help="Root used to resolve initial-observation paths in the manifest."
    )
    parser.add_argument("--model-id", default=os.environ.get("WORLDFOUNDRY_HARNESSEVAL_MODEL_ID"))
    parser.add_argument("--backend-config", type=Path, default=env_path("WORLDFOUNDRY_HARNESSEVAL_BACKEND_CONFIG"))
    parser.add_argument("--metric-cache-root", type=Path, help="Reuse official metric bundles and skill evidence.")
    parser.add_argument(
        "--score-cached", action="store_true", help="Score existing official bundles without model inference."
    )
    parser.add_argument("--dry-run", action="store_true", help="Audit inputs and plan skills without loading models.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Score the first N cases, retaining every selected skill per case."
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=86400, help="Timeout in seconds for each skill worker.")
    parser.add_argument(
        "--strict", action="store_true", help="Require a numeric overall score covering all six families."
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker-skill", choices=SKILLS, help=argparse.SUPPRESS)
    parser.add_argument("--drift-stage", choices=["physical", "render", "motion", "clip"], help=argparse.SUPPRESS)
    return parser


def _safe_id(value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid input identifier: {value!r}")


def skill_tasks(args):
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.planner import load_cases
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.runner import plan_skill_tasks

    cases = load_cases([args.manifest])
    case_ids = {case["case_id"] for case in (cases[: args.limit] if args.limit else cases)}
    return plan_skill_tasks(
        args.output_dir / "inventory",
        [args.manifest],
        args.plan_root,
        args.metric_cache_root,
        model_ids={args.model_id},
        case_ids=case_ids,
        skill_ids={args.worker_skill} if args.worker_skill else None,
    )


def run_worker(args) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.backends import build_backends
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.runner import (
        execute_skill_tasks,
        prepare_drift_stage_tasks,
    )

    tasks = skill_tasks(args)
    config = read_json(args.backend_config)
    backends = build_backends(
        {args.worker_skill},
        config,
        cache_root=args.metric_cache_root,
        config_root=args.backend_config.parent,
    )
    audit_path = (
        args.output_dir
        / "skill_audits"
        / f"{args.worker_skill}{'-' + args.drift_stage if args.drift_stage else ''}.json"
    )
    if args.drift_stage:
        audit = prepare_drift_stage_tasks(
            tasks,
            backends[args.worker_skill],
            args.drift_stage,
            workers=args.workers,
            audit_path=audit_path,
        )
    else:
        audit = execute_skill_tasks(
            tasks,
            backends,
            args.metric_cache_root,
            workers=args.workers,
            analyze_workers=args.workers,
            audit_path=audit_path,
        )
    if audit["status"] != "passed":
        raise RuntimeError(f"skill execution failed; inspect {audit_path}")


def run_official(args) -> Path | None:
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.inventory import (
        build_inventory,
        write_inventory,
    )
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.planner import load_cases
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.pipeline.runner import task_summary
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.score import (
        execute_score_tasks,
        plan_score_tasks,
    )
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.validation import validate_run
    from worldfoundry.runtime.jobs import run_bounded_command

    if args.generated_artifact_dir is None or args.plan_root is None or not args.model_id:
        raise ValueError("--run-official requires --generated-artifact-dir, --plan-root and --model-id")
    args.generated_artifact_dir = args.generated_artifact_dir.resolve(strict=True)
    args.plan_root = args.plan_root.resolve(strict=True)
    args.manifest = (args.manifest or args.generated_artifact_dir / "manifest.json").resolve(strict=True)
    args.metric_cache_root = (args.metric_cache_root or args.output_dir / "metric_cache").resolve()
    if args.metric_cache_root.is_relative_to(package_root()):
        raise ValueError("metric caches must live outside the package source tree")
    _safe_id(args.model_id)
    cases = load_cases([args.manifest])
    for case in cases:
        _safe_id(case["case_id"])
        if case["taxonomy"].get("primary_axis") not in {"transition_correctness", "world_persistence"}:
            raise ValueError("unsupported primary_axis")
    if not cases:
        raise ValueError("manifest has no cases")
    rows, audit = build_inventory(
        [args.manifest],
        args.generated_artifact_dir,
        [args.model_id],
        assets_root=args.assets_root or args.manifest.parent,
        workers=args.workers,
    )
    write_inventory(args.output_dir / "inventory", rows, audit)
    tasks = skill_tasks(args)
    if not tasks:
        raise ValueError("no selected skill tasks")
    write_json(args.output_dir / "skill_plan.json", task_summary(tasks, details=True))
    if args.dry_run:
        return None
    if not args.score_cached:
        if args.backend_config is None:
            raise ValueError("scored execution requires --backend-config (official skill backend JSON)")
        args.backend_config = args.backend_config.resolve(strict=True)
        config = read_json(args.backend_config)
        skills = {task.skill_id for task in tasks}
        for skill in SKILLS:
            if skill not in skills:
                continue
            stages = [None]
            if skill == "drift_degradation_analyzer":
                values = {**config.get("defaults", {}), **config.get("skills", {}).get(skill, {})}
                if values.get("mode") == "staged_local":
                    stages = ["physical", "render", "motion", "clip", None]
            for stage in stages:
                command = [
                    os.environ.get("WORLDFOUNDRY_HARNESSEVAL_PYTHON", sys.executable),
                    "-m",
                    MODULE,
                    "--worker-skill",
                    skill,
                    "--output-dir",
                    str(args.output_dir),
                    "--generated-artifact-dir",
                    str(args.generated_artifact_dir),
                    "--manifest",
                    str(args.manifest),
                    "--plan-root",
                    str(args.plan_root),
                    "--model-id",
                    args.model_id,
                    "--backend-config",
                    str(args.backend_config),
                    "--metric-cache-root",
                    str(args.metric_cache_root),
                    "--workers",
                    str(args.workers),
                    "--limit",
                    str(args.limit),
                ]
                if stage:
                    command.extend(["--drift-stage", stage])
                execution = run_bounded_command(
                    command,
                    cwd=args.output_dir,
                    env={
                        "PYTHONPATH": os.pathsep.join(
                            filter(None, [str(package_root().parent), os.environ.get("PYTHONPATH")])
                        )
                    },
                    timeout=args.timeout,
                )
                log = args.output_dir / "logs" / f"{skill}{'-' + stage if stage else ''}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(f"{execution.get('stdout', '')}\n{execution.get('stderr', '')}", encoding="utf-8")
                if execution.get("returncode") != 0 or execution.get("timed_out"):
                    raise RuntimeError(f"skill worker failed or timed out; inspect {log}")
    selected_cases = cases[: args.limit] if args.limit else cases
    selected_ids = {case["case_id"] for case in selected_cases}
    eval_root = args.output_dir / "evaluation"
    score_tasks = [
        task
        for task in plan_score_tasks(
            args.generated_artifact_dir,
            args.plan_root,
            args.metric_cache_root,
            eval_root,
            models={args.model_id},
        )
        if task.case_id in selected_ids
    ]
    if {task.case_id for task in score_tasks} != selected_ids:
        raise ValueError("metric bundles do not cover every selected case")
    execution = execute_score_tasks(score_tasks, eval_root, refresh_stale=True)
    write_json(args.output_dir / "score_execution.json", execution)
    if execution["blocked"]:
        raise ValueError("scoring blocked by missing or incompatible skill evidence")
    selected_manifest = args.output_dir / "selected_manifest.json"
    write_json(selected_manifest, {"cases": selected_cases})
    completion = validate_run(eval_root, [selected_manifest], expected_models=[args.model_id])
    write_json(args.output_dir / "completion_audit.json", completion)
    # Partial-family runs may be useful, but must not masquerade as a full benchmark.
    violations = [item for item in completion["violations"] if item["kind"] != "missing_overall_macro"]
    if violations:
        raise ValueError("completion audit failed; inspect completion_audit.json")
    return eval_root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.is_relative_to(package_root()):
        parser.error("--output-dir must be outside the package source tree")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_skill:
        run_worker(args)
        return 0
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "benchmark": {"benchmark_id": "harnesseval-w", "name": "HarnessEval-W"},
        "official_benchmark_verified": False,
        "leaderboard_valid": False,
        "integration_evidence": False,
        "leaderboard_blockers": ["Full official 100-case suite and judge/checkpoint parity have not been verified."],
        "normalizer_only": not args.run_official,
        "normalization_ok": False,
        "run": {"runner": MODULE, "started_at": utc_now_iso(), "status": "failed", "returncode": 1},
        "metrics": {
            "leaderboard": {},
            "per_metric": {},
            "primary_metric": PRIMARY_METRIC,
            "summary": {"sample_count": 0},
        },
        "artifacts": {"scorecard": str(args.output_dir / "scorecard.json")},
    }
    try:
        if args.limit < 0 or args.workers < 1 or args.timeout < 1:
            raise ValueError("limit must be nonnegative; workers and timeout must be positive")
        if (args.dry_run or args.score_cached) and not args.run_official:
            raise ValueError("--dry-run and --score-cached require --run-official")
        if args.run_official:
            path = run_official(args)
        elif args.run_fixture:
            path = package_data_path("benchmarks", "assets", "harnesseval-w", "scores.json")
        else:
            path = args.official_results_path or env_path("WORLDFOUNDRY_HARNESSEVAL_RESULTS_PATH")
            if path is None:
                raise ValueError("provide --official-results-path, --run-fixture or --run-official")
        scorecard["evaluation"] = {
            "kind": "plan"
            if args.dry_run
            else "fixture"
            if args.run_fixture
            else "official_python"
            if args.run_official
            else "result_normalizer",
            "upstream_revision": REVISION,
            "scored": False,
        }
        if path is not None:
            normalized = normalize_results(path, args.model_id)
            scores = normalized["scores"]
            if not any(value is not None for value in scores.values()):
                raise ValueError("no numeric metrics found")
            if args.strict and scores[PRIMARY_METRIC] is None:
                raise ValueError("--strict requires all six families and overall_macro")
            rows = [
                {
                    "metric_id": key,
                    "score": value,
                    "available": value is not None,
                    "higher_is_better": True,
                    "source_path": str(path),
                }
                for key, value in scores.items()
            ]
            scorecard["metrics"].update(
                {
                    "leaderboard": {k: v for k, v in scores.items() if v is not None},
                    "per_metric": {row["metric_id"]: row for row in rows},
                    "summary": {
                        "sample_count": normalized["case_count"],
                        "metric_count": len(rows),
                        "available_metrics": sum(row["available"] for row in rows),
                    },
                }
            )
            scorecard["evaluation"].update(
                {
                    "scored": True,
                    "model_id": normalized["model_id"],
                    "result_kind": normalized["kind"],
                    "common_case_counts": normalized["common_case_counts"],
                }
            )
            scorecard["normalization_ok"] = True
            scorecard["integration_evidence"] = args.run_official
            for filename, payload in (
                ("raw_metric_table.jsonl", rows),
                ("per_case_metrics.jsonl", normalized["case_records"]),
            ):
                write_jsonl(args.output_dir / filename, payload)
                scorecard["artifacts"][filename.removesuffix(".jsonl")] = str(args.output_dir / filename)
            write_json(args.output_dir / "official_report.json", normalized["report"])
            scorecard["artifacts"]["official_report"] = str(args.output_dir / "official_report.json")
        scorecard["run"].update(status="succeeded", returncode=0)
    except Exception as error:
        scorecard["run"]["error"] = f"{type(error).__name__}: {error}"
        scorecard["leaderboard_blockers"].append(str(error))
    write_json(
        args.output_dir / "benchmark_contract.json",
        {"benchmark_id": "harnesseval-w", "metric_ids": list(METRIC_IDS), "primary_metric": PRIMARY_METRIC},
    )
    scorecard["artifacts"]["benchmark_contract"] = str(args.output_dir / "benchmark_contract.json")
    write_json(args.output_dir / "scorecard.json", scorecard)
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    else:
        print(f"HarnessEval-W: {scorecard['run'].get('error', scorecard['run']['status'])}")
    return scorecard["run"]["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
