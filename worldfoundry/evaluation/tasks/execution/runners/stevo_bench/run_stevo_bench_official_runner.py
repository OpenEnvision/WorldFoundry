#!/usr/bin/env python3
"""Run the in-tree STEVO-Bench evaluator or normalize its official outputs.

The upstream evaluator is vendored under ``runtime/stevo_bench`` so
``--run-official`` executes the official judge code without an external
checkout.  The vendored tree is read-only: every mutable artifact (staged
videos, upstream ``runs/`` directories, logs, manifests) is written under
``--output-dir``.
"""

from __future__ import annotations


from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.io.paths import checkpoint_root_path  # noqa: E402
from worldfoundry.core.observability.time import utc_now_iso  # noqa: E402
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, write_json, write_jsonl  # noqa: E402
from worldfoundry.evaluation.tasks.execution.runners.stevo_bench.stevo_bench_metrics import (  # noqa: E402
    METRIC_ORDER,
    METRIC_SPECS,
    PRIMARY_METRIC_ID,
    StevoResultError,
    aggregate_metrics,
    load_sample_records,
    metric_rows,
    summarize_run,
)
from worldfoundry.runtime.jobs import run_bounded_command  # noqa: E402

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = RUNNER_ROOT / "runtime" / "stevo_bench"
FIXTURE_ROOT = REPO_ROOT / "worldfoundry/data/benchmarks/assets/stevo-bench/sample_run"
BENCHMARK_NAME = "STEVO-Bench"
RUNNER_NAME = "benchmark_zoo_stevo_bench_official_runner"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# STEVO-Bench ships 225 tasks across 6 world-change categories.
EXPECTED_TASK_COUNT = 225

# ``eval/eval_cli.py`` exposes one flag per criterion. The pinned CLI does not
# define ``--artifact`` or ``--coherence``, so those criteria are not executable.
CRITERIA = ("control", "physics", "state")
CRITERION_FLAGS = {"control": "--control", "physics": "--physics", "state": "--state"}
# Ensemble configuration from the paper (majority, ensemble size 3).
CRITERION_ENSEMBLE_MODE = {"control": "majority", "physics": "majority", "state": "majority"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize STEVO-Bench official outputs.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "stevo-bench"))
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"),
        help="Directory for every WorldFoundry and upstream mutable artifact (required).",
    )
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
        help="Directory holding generated .mp4 clips and an optional output-map JSON.",
    )
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--run-fixture", action="store_true")
    parser.add_argument(
        "--stevo-bench-root",
        dest="stevo_bench_root",
        type=Path,
        help="Override the checked-in read-only STEVO-Bench runtime root.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help=(
            "Optional local judge-asset root. STEVO-Bench's official judges are remote VLM APIs and "
            "need no weights; this never defaults inside the vendored source tree."
        ),
    )
    parser.add_argument("--task-root", type=Path, help="Official benchmark task YAML root (benchmark/tasks).")
    parser.add_argument("--output-map", type=Path, help="Explicit task_id -> video filename JSON map.")
    parser.add_argument("--criterion", choices=("all", *CRITERIA), default="all")
    parser.add_argument("--vlm-provider", default=os.environ.get("WORLDFOUNDRY_STEVO_BENCH_VLM_PROVIDER", "gemini"))
    parser.add_argument("--judge-model", default=os.environ.get("WORLDFOUNDRY_STEVO_BENCH_JUDGE_MODEL"))
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--ensemble-mode", choices=("majority", "unanimous", "unanimous_true"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--pattern", help="fnmatch filter over task ids.")
    parser.add_argument("--exclude-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Also run eval.summarize_results after judging.")
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    parser.add_argument("--json", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Path resolution and isolation guards
# ---------------------------------------------------------------------------


def resolve_runtime_root(args: argparse.Namespace) -> Path:
    explicit = args.stevo_bench_root or env_path("WORLDFOUNDRY_STEVO_BENCH_ROOT")
    return (explicit or DEFAULT_RUNTIME_ROOT).expanduser().resolve()


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_output_dir(args: argparse.Namespace, *, runtime_root: Path) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    if _is_inside(output_dir, runtime_root):
        raise ValueError(
            f"--output-dir must not resolve inside the read-only STEVO-Bench runtime tree: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_weights_dir(args: argparse.Namespace, *, runtime_root: Path) -> Path:
    weights_dir = args.weights_dir or env_path("WORLDFOUNDRY_STEVO_BENCH_WEIGHTS_DIR")
    if weights_dir is None:
        weights_dir = checkpoint_root_path(
            "stevo_bench", specific_env="WORLDFOUNDRY_STEVO_BENCH_WEIGHTS_DIR"
        )
    weights_dir = Path(weights_dir).expanduser().resolve()
    if _is_inside(weights_dir, runtime_root):
        raise ValueError(
            f"STEVO-Bench judge assets must not resolve inside the vendored runtime tree: {weights_dir}"
        )
    return weights_dir


def resolve_work_dir(output_dir: Path, *, runtime_root: Path) -> Path:
    work_dir = output_dir / "stevo_bench_work"
    if _is_inside(work_dir, runtime_root):
        raise ValueError(f"STEVO-Bench work directory must not modify the in-tree runtime: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def resolve_task_root(args: argparse.Namespace, *, runtime_root: Path) -> Path:
    candidate = args.task_root or env_path("WORLDFOUNDRY_STEVO_BENCH_TASK_ROOT")
    if candidate is None:
        raise FileNotFoundError(
            "STEVO-Bench task YAMLs are a separate Hugging Face dataset. Pass --task-root or set "
            "WORLDFOUNDRY_STEVO_BENCH_TASK_ROOT to the directory produced by "
            "`hf download JhanLiufu/StEvo-Bench --repo-type dataset --local-dir <dir>`."
        )
    task_root = Path(candidate).expanduser().resolve()
    if not task_root.is_dir():
        raise FileNotFoundError(f"STEVO-Bench task root is missing: {task_root}")
    if _is_inside(task_root, runtime_root):
        raise ValueError(f"STEVO-Bench task data must not be staged inside the vendored runtime: {task_root}")
    return task_root


def resolve_results_path(args: argparse.Namespace) -> Path | None:
    if args.official_results_path is not None:
        return args.official_results_path
    return env_path("WORLDFOUNDRY_STEVO_BENCH_RESULTS_PATH")


# ---------------------------------------------------------------------------
# Generated-video staging
# ---------------------------------------------------------------------------


def stage_generated_videos(
    *,
    generated_artifact_dir: Path,
    outputs_dir: Path,
    explicit_output_map: Path | None,
    staging_manifest_path: Path,
) -> dict[str, Any]:
    """Link generated clips plus an output map into an isolated outputs folder.

    Upstream requires exactly one ``.json`` map alongside the videos.  An
    existing map is copied verbatim; otherwise the map is derived from video
    stems, which is the naming convention documented in the upstream README.
    """
    source_root = generated_artifact_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"generated artifact directory is missing: {source_root}")
    videos = sorted(
        candidate
        for candidate in source_root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"generated artifact directory contains no video files: {source_root}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(outputs_dir.iterdir()):
        if stale.is_file() or stale.is_symlink():
            stale.unlink()

    rows: list[dict[str, Any]] = []
    for source in videos:
        target = outputs_dir / source.name
        if target.exists() or target.is_symlink():
            raise ValueError(f"duplicate generated video filename resolves to one task: {source.name}")
        try:
            target.symlink_to(source.resolve())
            method = "symlink"
        except OSError:
            target.write_bytes(source.read_bytes())
            method = "copy"
        rows.append({"task_id": source.stem, "source": str(source.resolve()), "destination": str(target), "method": method})

    map_candidates = sorted(path for path in source_root.glob("*.json"))
    if explicit_output_map is not None:
        map_source = explicit_output_map.expanduser().resolve()
        if not map_source.is_file():
            raise FileNotFoundError(f"--output-map is not a file: {map_source}")
        mapping = json.loads(map_source.read_text(encoding="utf-8"))
        map_origin = "explicit_output_map"
        map_name = map_source.name
    elif len(map_candidates) == 1:
        mapping = json.loads(map_candidates[0].read_text(encoding="utf-8"))
        map_origin = "generated_artifact_dir"
        map_name = map_candidates[0].name
    elif len(map_candidates) > 1:
        raise ValueError(
            f"generated artifact directory holds {len(map_candidates)} JSON files; STEVO-Bench requires "
            "exactly one output map. Pass --output-map to disambiguate."
        )
    else:
        mapping = {row["task_id"]: Path(row["destination"]).name for row in rows}
        map_origin = "derived_from_video_filenames"
        map_name = "output_map.json"
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("STEVO-Bench output map must be a non-empty {task_id: video_filename} object")

    map_path = outputs_dir / map_name
    map_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "outputs_dir": str(outputs_dir),
        "output_map": str(map_path),
        "output_map_origin": map_origin,
        "video_count": len(rows),
        "mapped_task_count": len(mapping),
        "videos": rows,
    }
    write_json(staging_manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Official runtime execution
# ---------------------------------------------------------------------------


def build_official_commands(
    args: argparse.Namespace,
    *,
    runtime_root: Path,
    outputs_dir: Path,
    task_root: Path,
    run_dir: Path,
) -> list[list[str]]:
    criteria = CRITERIA if args.criterion == "all" else (args.criterion,)
    python = os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON") or sys.executable
    commands: list[list[str]] = []
    for criterion in criteria:
        command = [
            python,
            "-m",
            "eval.eval_cli",
            "--outputs",
            str(outputs_dir),
            "--task_root",
            str(task_root),
            "--run_dir",
            str(run_dir),
            CRITERION_FLAGS[criterion],
            "--ensemble_size",
            str(args.ensemble_size),
            "--ensemble_mode",
            str(args.ensemble_mode or CRITERION_ENSEMBLE_MODE[criterion]),
            "--workers",
            str(args.workers),
            "--vlm_provider",
            str(args.vlm_provider),
        ]
        if args.judge_model:
            command.extend(
                [
                    "--control_judge_model",
                    args.judge_model,
                    "--physics_judge_model",
                    args.judge_model,
                    "--se_judge_model",
                    args.judge_model,
                ]
            )
        if args.pattern:
            command.extend(["--pattern", args.pattern])
        if args.exclude_baseline:
            command.append("--exclude_baseline")
        if args.overwrite:
            command.append("--overwrite")
        commands.append(command)
    return commands


def _runtime_env(*, runtime_root: Path, weights_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(runtime_root), str(REPO_ROOT), env.get("PYTHONPATH", "")) if path
    )
    env["WORLDFOUNDRY_STEVO_BENCH_WEIGHTS_DIR"] = str(weights_dir)
    return env


def run_official(args: argparse.Namespace, *, runtime_root: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    entrypoint = runtime_root / "eval" / "eval_cli.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing in-tree STEVO-Bench runtime: {entrypoint}")
    generated_dir = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_dir is None:
        raise ValueError(
            "--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official"
        )
    args.generated_artifact_dir = Path(generated_dir)
    weights_dir = resolve_weights_dir(args, runtime_root=runtime_root)
    task_root = resolve_task_root(args, runtime_root=runtime_root)
    work_dir = resolve_work_dir(output_dir, runtime_root=runtime_root)
    outputs_dir = work_dir / "outputs"
    run_dir = work_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    staging = stage_generated_videos(
        generated_artifact_dir=Path(generated_dir),
        outputs_dir=outputs_dir,
        explicit_output_map=args.output_map,
        staging_manifest_path=output_dir / "stevo_bench_staging.json",
    )

    env = _runtime_env(runtime_root=runtime_root, weights_dir=weights_dir)
    commands = build_official_commands(
        args,
        runtime_root=runtime_root,
        outputs_dir=outputs_dir,
        task_root=task_root,
        run_dir=run_dir,
    )
    started_at = utc_now_iso()
    log_path = output_dir / "official_runtime.log"
    log_chunks: list[str] = []
    executions: list[dict[str, Any]] = []
    for command in commands:
        result = run_bounded_command(command, cwd=runtime_root, env=env, timeout=args.timeout)
        log_chunks.append(
            f"$ {' '.join(command)}\n\n[stdout]\n{result.get('stdout', '')}\n\n[stderr]\n{result.get('stderr', '')}\n"
        )
        executions.append(
            {
                "command": command,
                "returncode": result.get("returncode"),
                "timed_out": result.get("timed_out"),
            }
        )
        if result.get("returncode") != 0:
            log_path.write_text("\n".join(log_chunks), encoding="utf-8")
            raise RuntimeError(
                f"STEVO-Bench official runtime exited with {result.get('returncode')}; see {log_path}"
            )

    run_root = run_dir / Path(staging["output_map"]).stem
    if args.summarize and run_root.is_dir():
        summarize_command = [
            os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON") or sys.executable,
            "-m",
            "eval.summarize_results",
            "--run_dir",
            str(run_root),
        ]
        result = run_bounded_command(summarize_command, cwd=runtime_root, env=env, timeout=args.timeout)
        log_chunks.append(
            f"$ {' '.join(summarize_command)}\n\n[stdout]\n{result.get('stdout', '')}\n\n"
            f"[stderr]\n{result.get('stderr', '')}\n"
        )
        executions.append(
            {
                "command": summarize_command,
                "returncode": result.get("returncode"),
                "timed_out": result.get("timed_out"),
            }
        )
    log_path.write_text("\n".join(log_chunks), encoding="utf-8")

    results_path = run_root if run_root.is_dir() else run_dir
    runtime_summary = {
        "started_at": started_at,
        "runtime_root": str(runtime_root),
        "work_dir": str(work_dir),
        "run_dir": str(run_dir),
        "run_root": str(run_root),
        "task_root": str(task_root),
        "weights_dir": str(weights_dir),
        "log_path": str(log_path),
        "staging_manifest": str(output_dir / "stevo_bench_staging.json"),
        "staged_video_count": staging["video_count"],
        "mapped_task_count": staging["mapped_task_count"],
        "output_map_origin": staging["output_map_origin"],
        "criterion": args.criterion,
        "ensemble_size": args.ensemble_size,
        "vlm_provider": args.vlm_provider,
        "executions": executions,
    }
    write_json(output_dir / "runtime_summary.json", runtime_summary)
    return results_path, runtime_summary


# ---------------------------------------------------------------------------
# Scorecard construction
# ---------------------------------------------------------------------------


def _comparability_blockers(
    *,
    official_runtime_executed: bool,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    runtime_summary: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if not official_runtime_executed:
        blockers.append("the official STEVO-Bench judge runtime was not executed by this run")
    missing = [row["metric_id"] for row in rows if not row["available"]]
    if missing:
        blockers.append(f"official metric coverage is partial ({len(missing)} of {len(METRIC_ORDER)} metrics missing)")
    task_count = len({str(record["task_id"]) for record in records})
    if task_count < EXPECTED_TASK_COUNT:
        blockers.append(f"official task coverage is partial ({task_count}/{EXPECTED_TASK_COUNT} tasks)")
    if runtime_summary is not None:
        if runtime_summary.get("output_map_origin") == "derived_from_video_filenames":
            blockers.append(
                "the task_id -> video map was derived from filenames rather than an official output map"
            )
        if int(runtime_summary.get("ensemble_size") or 0) < 3:
            blockers.append("the paper ensemble configuration (ensemble_size 3) was not used")
    return blockers


def build_scorecard(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    results_path: Path,
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    runtime_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    official_runtime_executed = runtime_summary is not None
    rows = metric_rows(metrics, official_runtime_executed=official_runtime_executed)
    available = [row for row in rows if row["available"]]
    breakdown = summarize_run(records)
    comparability_blockers = _comparability_blockers(
        official_runtime_executed=official_runtime_executed,
        records=records,
        rows=rows,
        runtime_summary=runtime_summary,
    )
    normalization_ok = bool(available)
    # Verification requires a real official run with no outstanding blockers.
    official_verified = official_runtime_executed and not comparability_blockers and normalization_ok
    leaderboard_blockers = list(comparability_blockers)
    leaderboard_blockers.append(
        "leaderboard submission requires the complete 225-task split judged with the official ensemble settings"
    )

    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_path = output_dir / "per_sample_scores.jsonl"
    contract_path = output_dir / "benchmark_contract.json"

    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_runtime_executed and normalization_ok,
        "leaderboard_valid": False,
        "leaderboard_blockers": leaderboard_blockers,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": (runtime_summary or {}).get("started_at") or utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0 if normalization_ok else 1,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": BENCHMARK_NAME},
        "metrics": {
            "leaderboard": {
                row["metric_id"]: row["normalized_score"]
                for row in available
                if row["normalized_score"] is not None
            },
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {
                "primary_metric_id": PRIMARY_METRIC_ID,
                "primary_metric_score": metrics.get(PRIMARY_METRIC_ID, {}).get("normalized_score"),
                "sample_count": breakdown["task_count"],
                "record_count": breakdown["record_count"],
                "metric_count": len(METRIC_ORDER),
                "available_metrics": len(available),
                "failed_metrics": len(METRIC_ORDER) - len(available),
                "judges": breakdown["judges"],
                "baseline": breakdown["baseline"],
                "occluded": breakdown["occluded"],
                "by_level": breakdown["by_level"],
            },
        },
        "evaluation": {
            "kind": "stevo_bench_official_in_tree" if official_runtime_executed else "stevo_bench_result_normalizer",
            "available_metric_count": len(available),
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": official_runtime_executed,
            "benchmark_comparable": official_verified,
            "comparability_blockers": comparability_blockers,
            "expected_task_count": EXPECTED_TASK_COUNT,
            "observed_task_count": breakdown["task_count"],
        },
        "dataset": {
            "results_path": str(results_path),
            "generated_artifact_dir": None
            if args.generated_artifact_dir is None
            else str(Path(args.generated_artifact_dir).expanduser().resolve()),
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_path.resolve()),
            "benchmark_contract": str(contract_path.resolve()),
            "official_results_path": str(results_path.resolve()),
            "official_runtime_log": (runtime_summary or {}).get("log_path"),
        },
        "provenance": {
            "runtime_root": str(resolve_runtime_root(args)),
            "upstream_repository": "https://github.com/jhanliufu-personal/STEVO-Bench",
            "upstream_revision": "680fb6ee2733894ebc8e5584c08146f4bf7e6415",
            "upstream_license": "MIT",
            "external_repository_checkout_required": False,
        },
    }

    write_jsonl(raw_metric_table_path, rows)
    write_jsonl(per_sample_path, records)
    write_json(
        contract_path,
        {
            "benchmark_id": args.benchmark_id,
            "name": BENCHMARK_NAME,
            "metric_ids": list(METRIC_ORDER),
            "primary_metric_id": PRIMARY_METRIC_ID,
            "metric_specs": {metric_id: METRIC_SPECS[metric_id] for metric_id in METRIC_ORDER},
            "runtime_root": str(resolve_runtime_root(args)),
            "external_repository_checkout_required": False,
        },
    )
    write_json(scorecard_path, scorecard)
    return scorecard


def failure_scorecard(args: argparse.Namespace, output_dir: Path, exc: BaseException) -> dict[str, Any]:
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "leaderboard_blockers": [f"{type(exc).__name__}: {exc}"],
        "normalizer_only": not bool(getattr(args, "run_official", False)),
        "normalization_ok": False,
        "run": {
            "status": "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "benchmark": {
            "benchmark_id": getattr(args, "benchmark_id", "stevo-bench"),
            "name": BENCHMARK_NAME,
        },
        "metrics": {
            "leaderboard": {},
            "per_metric": {},
            "summary": {
                "primary_metric_id": PRIMARY_METRIC_ID,
                "sample_count": 0,
                "metric_count": len(METRIC_ORDER),
                "available_metrics": 0,
                "failed_metrics": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "kind": "stevo_bench_result_normalizer",
            "available_metric_count": 0,
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": False,
            "comparability_blockers": [f"{type(exc).__name__}: {exc}"],
        },
        "artifacts": {"scorecard": str((output_dir / "scorecard.json").resolve())},
    }
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 1
    output_dir = args.output_dir.expanduser().resolve()
    try:
        runtime_root = resolve_runtime_root(args)
        output_dir = resolve_output_dir(args, runtime_root=runtime_root)
        runtime_summary: dict[str, Any] | None = None
        if args.run_fixture:
            results_path = FIXTURE_ROOT
        elif args.run_official:
            results_path, runtime_summary = run_official(args, runtime_root=runtime_root, output_dir=output_dir)
        else:
            resolved = resolve_results_path(args)
            if resolved is None:
                raise StevoResultError(
                    "--official-results-path, WORLDFOUNDRY_STEVO_BENCH_RESULTS_PATH, --run-fixture, "
                    "or --run-official is required"
                )
            results_path = resolved
        results_path = Path(results_path).expanduser().resolve()
        records = load_sample_records(results_path)
        metrics = aggregate_metrics(records)
        scorecard = build_scorecard(
            args,
            output_dir=output_dir,
            results_path=results_path,
            records=records,
            metrics=metrics,
            runtime_summary=runtime_summary,
        )
    except BaseException as exc:  # noqa: BLE001 - a failure must still produce a scorecard
        output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = failure_scorecard(args, output_dir, exc)
        if args.json:
            print(json.dumps({"ok": False, "output_dir": str(output_dir), **scorecard}, ensure_ascii=False, indent=2))
        else:
            print(f"stevo-bench: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ok = scorecard["normalization_ok"] is True
    if args.json:
        print(json.dumps({"ok": ok, "output_dir": str(output_dir), **scorecard}, ensure_ascii=False, indent=2))
    elif ok:
        print(
            f"stevo-bench: normalized {scorecard['evaluation']['available_metric_count']} metrics "
            f"from {scorecard['metrics']['summary']['sample_count']} tasks"
        )
    else:
        print("stevo-bench: no STEVO-Bench metrics were available", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
