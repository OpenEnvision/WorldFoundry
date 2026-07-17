#!/usr/bin/env python3
"""WorldFoundry runner for MemoBench.

The runner can normalize any combination of MemoBench stage outputs:

* ``eval_*.csv`` from ``evaluation/run_eval.py``
* ``ors_scores.csv`` from ``evaluation/compute_ors.py``
* VQA ``overall.csv`` or per-clip CSVs from ``evaluation/vqa/llm-vqa.py``
* ``leaderboard.csv`` from ``leaderboard/leaderboard.py``

``--run-official`` executes the vendored Step 1 automated metric script. ORS
and VQA are intentionally left as explicit user-controlled stages because they
require SAM-3 and Gemini credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.memobench.memobench_metrics import (
    METRIC_ORDER,
    compute_memobench_metrics,
    metric_rows,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
BENCHMARK_ID = "memobench"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _runtime_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_root = _env_path("WORLDFOUNDRY_MEMOBENCH_ROOT")
    if env_root is not None:
        return env_root
    return (Path(__file__).resolve().parent / "runtime" / "memobench").resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize MemoBench outputs.")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--official-results-path", "--results-path", dest="official_results_path", type=Path)
    parser.add_argument("--run-fixture", action="store_true")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--generated-synthetic-dir", type=Path)
    parser.add_argument("--generated-real-dir", type=Path)
    parser.add_argument("--memobench-root", type=Path)
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_MEMOBENCH_MODEL_NAME"))
    parser.add_argument("--mode", choices=["synthetic", "real", "both"], default="both")
    parser.add_argument("--device", default=os.environ.get("WORLDFOUNDRY_MEMOBENCH_DEVICE"))
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--sample-step", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "7200")))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _collect_results_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.official_results_path is not None:
        paths.append(args.official_results_path.expanduser().resolve())
    env_results = _env_path("WORLDFOUNDRY_MEMOBENCH_RESULTS_PATH")
    if env_results is not None and env_results not in paths:
        paths.append(env_results)
    for name in (
        "WORLDFOUNDRY_MEMOBENCH_EVAL_DIR",
        "WORLDFOUNDRY_MEMOBENCH_ORS_DIR",
        "WORLDFOUNDRY_MEMOBENCH_VQA_DIR",
        "WORLDFOUNDRY_MEMOBENCH_LEADERBOARD_PATH",
    ):
        path = _env_path(name)
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def _run_step1(args: argparse.Namespace, runtime_root: Path) -> tuple[Path | None, dict[str, Any]]:
    generated_root = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    generated_syn = args.generated_synthetic_dir or _env_path("WORLDFOUNDRY_MEMOBENCH_GENERATED_SYNTHETIC_DIR")
    generated_real = args.generated_real_dir or _env_path("WORLDFOUNDRY_MEMOBENCH_GENERATED_REAL_DIR")
    if args.mode in {"synthetic", "real"} and generated_root is None:
        raise ValueError("--generated-artifact-dir is required for --run-official with mode synthetic/real")
    if args.mode == "both" and (generated_syn is None or generated_real is None):
        if generated_root is not None:
            generated_syn = generated_syn or generated_root / "Synthetic"
            generated_real = generated_real or generated_root / "Real"
        if generated_syn is None or generated_real is None:
            raise ValueError(
                "--generated-synthetic-dir and --generated-real-dir are required for --run-official --mode both"
            )

    out_csv = args.output_dir / "upstream" / f"eval_{args.mode}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    script = runtime_root / "evaluation" / "run_eval.py"
    if not script.is_file():
        raise FileNotFoundError(f"MemoBench run_eval.py not found: {script}")
    command = [sys.executable, str(script), "--mode", args.mode, "--out_csv", str(out_csv)]
    if args.mode == "both":
        command.extend(["--gen_root_syn", str(generated_syn), "--gen_root_real", str(generated_real)])
    else:
        command.extend(["--gen_root", str(generated_root)])
    if args.device:
        command.extend(["--device", args.device])
    if args.max_side:
        command.extend(["--max_side", str(args.max_side)])
    if args.sample_step:
        command.extend(["--sample_step", str(args.sample_step)])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(runtime_root), env.get("PYTHONPATH")) if part
    )
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(runtime_root),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    duration = time.monotonic() - start
    (args.output_dir / "upstream_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.output_dir / "upstream_stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary = {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "stdout_path": str((args.output_dir / "upstream_stdout.log").resolve()),
        "stderr_path": str((args.output_dir / "upstream_stderr.log").resolve()),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"MemoBench Step 1 failed with exit code {completed.returncode}")
    return out_csv, summary


def _scorecard(
    *,
    args: argparse.Namespace,
    results_paths: list[Path],
    metrics: Mapping[str, Mapping[str, Any]],
    official_runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(metrics)
    raw_metric_path = args.output_dir / "raw_metric_table.jsonl"
    per_sample_path = args.output_dir / "per_sample_scores.jsonl"
    scorecard_path = args.output_dir / "scorecard.json"
    write_jsonl(raw_metric_path, rows)
    write_jsonl(per_sample_path, [])

    available_rows = [row for row in rows if row.get("available") is True]
    leaderboard = {
        row["metric_id"]: row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    official_runtime_executed = official_runtime is not None
    normalization_ok = bool(available_rows)
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        # ``--run-official`` executes only MemoBench Step 1.  The official
        # benchmark also requires ORS, VQA, and leaderboard aggregation, so a
        # successful Step-1 run must not be reported as full-benchmark proof.
        "official_benchmark_verified": False,
        "integration_evidence": normalization_ok,
        "leaderboard_valid": False,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_memobench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "official_runtime": dict(official_runtime or {}),
        },
        "benchmark": {
            "benchmark_id": args.benchmark_id,
            "name": "MemoBench",
            "contract_only": False,
            "requires_upstream_runtime": True,
        },
        "dataset": {
            "upstream_results": [str(path.resolve()) for path in results_paths],
            "generated_artifact_dir": None if args.generated_artifact_dir is None else str(args.generated_artifact_dir),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {
                "available_metric_count": len(available_rows),
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "memobench_official_step1" if official_runtime_executed else "memobench_result_normalizer",
            "step1_runtime_verified": official_runtime_executed and normalization_ok,
            "full_benchmark_verified": False,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_path.resolve()),
            "per_sample_scores": str(per_sample_path.resolve()),
            "upstream_results": [str(path.resolve()) for path in results_paths],
        },
        "notes": [
            "MemoBench full-suite evidence combines automated metrics, ORS, VQA, and leaderboard aggregation.",
            "WorldFoundry --run-official currently executes Step 1 automated metrics; ORS and VQA remain explicit external stages because they require SAM-3 and Gemini credentials.",
        ],
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = _runtime_root(args.memobench_root)

    results_paths = _collect_results_paths(args)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path(BENCHMARK_ID)
        if sample_path is None:
            print("error: no MemoBench checked-in sample results found", file=sys.stderr)
            return 2
        results_paths = [sample_path]

    official_runtime = None
    if args.run_official:
        try:
            step1_path, official_runtime = _run_step1(args, runtime_root)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if step1_path is not None:
            results_paths.append(step1_path)

    if not results_paths:
        print(
            "error: --official-results-path, WORLDFOUNDRY_MEMOBENCH_RESULTS_PATH, --run-fixture, or --run-official is required",
            file=sys.stderr,
        )
        return 2

    metrics = compute_memobench_metrics(results_paths, model_name=args.model_name)
    scorecard = _scorecard(
        args=args,
        results_paths=results_paths,
        metrics=metrics,
        official_runtime=official_runtime,
    )
    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"memobench: {'ok' if result['ok'] else 'failed'}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
