#!/usr/bin/env python3
"""Run an external WorldReasonBench checkout or normalize its official outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.worldreasonbench.worldreasonbench_metrics import (
    METRIC_ORDER,
    metric_rows,
    normalize_results,
)
from worldfoundry.runtime.jobs import run_bounded_command

REPO_ROOT = Path(__file__).resolve().parents[6]
FIXTURE_ROOT = REPO_ROOT / "worldfoundry/data/benchmarks/assets/worldreasonbench"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default="worldreasonbench")
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument(
        "--protocol",
        choices=("auto", "all", "qa", "pointwise", "pairwise"),
        default=os.environ.get("WORLDFOUNDRY_WORLDREASONBENCH_PROTOCOL", "auto"),
    )
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--run-fixture", action="store_true")
    parser.add_argument("--worldreasonbench-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", "--video-dir", dest="video_dir", type=Path)
    parser.add_argument("--qa-json", type=Path)
    parser.add_argument("--pairs-json", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "default"))
    parser.add_argument(
        "--judge-model", default=os.environ.get("WORLDFOUNDRY_WORLDREASONBENCH_JUDGE_MODEL", "qwen3.5-27b")
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _resolve_upstream_root(args: argparse.Namespace) -> Path:
    root = args.worldreasonbench_root or _env_path("WORLDFOUNDRY_WORLDREASONBENCH_ROOT")
    if root is None or not root.is_dir():
        raise FileNotFoundError(
            "WorldReasonBench checkout is required for --run-official; pass --worldreasonbench-root "
            "or WORLDFOUNDRY_WORLDREASONBENCH_ROOT"
        )
    return root


def _official_command(args: argparse.Namespace, root: Path) -> tuple[list[str], Path]:
    protocol = args.protocol
    if protocol not in {"qa", "pointwise", "pairwise"}:
        raise ValueError("--run-official requires one explicit --protocol")
    python = os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable)
    if protocol == "qa":
        qa_json = args.qa_json or _env_path("WORLDFOUNDRY_WORLDREASONBENCH_QA_JSON")
        video_dir = args.video_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
        if qa_json is None or video_dir is None:
            raise ValueError("QA execution requires --qa-json and --video-dir")
        result_dir = args.output_dir / "qa"
        command = [
            python,
            str(root / "evaluation/eval_qa.py"),
            "--qa_json",
            str(qa_json),
            "--video_dir",
            str(video_dir),
            "--output_dir",
            str(result_dir),
            "--base_url",
            str(args.base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")),
            "--model",
            args.model,
            "--video_fps",
            str(args.video_fps),
            "--no_progress",
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        return command, result_dir

    pairs = args.pairs_json or _env_path("WORLDFOUNDRY_WORLDREASONBENCH_PAIRS_JSON")
    if pairs is None:
        pairs = root / "data/statistics_model_pairs_by_task_stratified_balanced_tie_v2.json"
    output = args.output_dir / f"{protocol}_eval.jsonl"
    script = root / f"evaluation/reward_bench/run_{protocol}_eval.py"
    command = [
        python,
        str(script),
        "--pairs-json",
        str(pairs),
        "--judge-model",
        args.judge_model,
        "--judge-base-url",
        str(args.base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30002/v1")),
        "--output-jsonl",
        str(output),
        "--num-workers",
        str(args.num_workers),
        "--resume",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command, output


def _run_upstream(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = _resolve_upstream_root(args)
    command, results = _official_command(args, root)
    execution = run_bounded_command(
        command,
        cwd=root,
        env={"OPENAI_API_KEY": args.api_key},
        timeout=args.timeout,
    )
    log_path = args.output_dir / "official_runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"STDOUT\n{execution['stdout']}\n\nSTDERR\n{execution['stderr']}", encoding="utf-8")
    execution["log_path"] = str(log_path)
    execution.pop("stdout", None)
    execution.pop("stderr", None)
    if execution["returncode"] != 0:
        raise RuntimeError(f"WorldReasonBench official runtime exited with {execution['returncode']}; see {log_path}")
    return results, execution


def _scorecard(
    args: argparse.Namespace,
    results_path: Path,
    protocol_results: list[dict[str, Any]],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = metric_rows(protocol_results)
    available = [row for row in rows if row["available"]]
    protocols = sorted(str(result["protocol"]) for result in protocol_results)
    full_protocol_set = set(protocols) == {"pairwise", "pointwise", "qa"}
    official_runtime_executed = runtime is not None
    strict_ok = not args.strict or full_protocol_set
    leaderboard = {
        row["metric_id"]: row["normalized_score"] for row in available if row["normalized_score"] is not None
    }
    scorecard = {
        "schema_version": "worldfoundry-scorecard",
        "official_benchmark_verified": official_runtime_executed and bool(available),
        "integration_evidence": official_runtime_executed and full_protocol_set and bool(available),
        "leaderboard_valid": False,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": bool(available) and strict_ok,
        "run": {
            "status": "succeeded" if available and strict_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_worldreasonbench_official_runner",
            "returncode": 0 if available and strict_ok else 1,
            "official_runtime": runtime or {},
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": "WorldReasonBench"},
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {"available_metric_count": len(available), "declared_metric_count": len(METRIC_ORDER)},
        },
        "evaluation": {
            "available": bool(available),
            "kind": "worldreasonbench_official_external_runtime"
            if official_runtime_executed
            else "worldreasonbench_result_normalizer",
            "protocols": protocols,
            "full_protocol_set": full_protocol_set,
        },
        "artifacts": {
            "scorecard": str((args.output_dir / "scorecard.json").resolve()),
            "official_results_path": str(results_path.resolve()),
        },
        "notes": [
            "QA, pointwise, and pairwise protocols are normalized in-tree.",
            "Restricted upstream data, templates, and evaluator scripts remain in the caller-provided checkout.",
            "Leaderboard validity requires the complete official data and judge configuration.",
        ],
    }
    write_jsonl(args.output_dir / "raw_metric_table.jsonl", rows)
    write_json(args.output_dir / "protocol_results.json", protocol_results)
    write_json(
        args.output_dir / "benchmark_contract.json",
        {"benchmark_id": args.benchmark_id, "protocols": protocols, "metric_ids": list(METRIC_ORDER)},
    )
    write_json(args.output_dir / "scorecard.json", scorecard)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime = None
        if args.run_fixture:
            results_path = FIXTURE_ROOT
            protocol = "all"
        elif args.run_official:
            results_path, runtime = _run_upstream(args)
            protocol = args.protocol
        else:
            results_path = args.official_results_path or _env_path("WORLDFOUNDRY_WORLDREASONBENCH_RESULTS_PATH")
            if results_path is None:
                raise ValueError("--official-results-path or WORLDFOUNDRY_WORLDREASONBENCH_RESULTS_PATH is required")
            protocol = args.protocol
        results = normalize_results(results_path, protocol)
        scorecard = _scorecard(args, results_path, results, runtime)
    except Exception as exc:  # noqa: BLE001
        scorecard = {
            "schema_version": "worldfoundry-scorecard",
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": not args.run_official,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_worldreasonbench_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "WorldReasonBench"},
            "metrics": {"leaderboard": {}, "per_metric": {}},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
    payload = {"ok": scorecard["normalization_ok"], "output_dir": str(args.output_dir), **scorecard}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif payload["ok"]:
        print(f"worldreasonbench: {scorecard['evaluation']['kind']}")
    else:
        print(f"worldreasonbench: failed ({scorecard['run'].get('error', 'incomplete results')})", file=sys.stderr)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
