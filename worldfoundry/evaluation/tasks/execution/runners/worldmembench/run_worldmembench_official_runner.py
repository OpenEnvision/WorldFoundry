# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the in-tree WorldMemBench evaluator or import official per-model results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.framework.official_runner import (
    BenchRunnerConfig,
    build_scorecard,
    write_failed_scorecard,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path as _env_path
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.worldmembench_metrics import (
    METRIC_SPECS,
    load_official_results,
)

CONFIG = BenchRunnerConfig(
    benchmark_id="worldmembench",
    display_name="WorldMemBench",
    root_env="",
    results_path_env="WORLDFOUNDRY_WORLDMEMBENCH_RESULTS_PATH",
    default_repo_subdir="",
    metric_order=tuple(METRIC_SPECS),
    metric_specs={
        name: {"name": name.replace("_", " ").title(), "group": name.split("_")[0], "higher_is_better": higher}
        for name, (_, higher) in METRIC_SPECS.items()
    },
    metric_aliases={},
    average_metric_id=None,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose the standard benchmark runner flags and metric asset configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", choices=["worldmembench"], default="worldmembench")
    parser.add_argument(
        "--official-results-path", "--results-path", type=Path, default=_env_path(CONFIG.results_path_env)
    )
    parser.add_argument(
        "--generated-video-dir",
        "--generated-artifact-dir",
        "--input-path",
        type=Path,
        default=_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
        help="Official video tree, video file, or JSONL input manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--python", default=os.environ.get("WORLDFOUNDRY_WORLDMEMBENCH_PYTHON", sys.executable))
    parser.add_argument("--config", type=Path, default=_env_path("WORLDFOUNDRY_WORLDMEMBENCH_CONFIG"))
    parser.add_argument("--weights-dir", type=Path, default=_env_path("WORLDFOUNDRY_WORLDMEMBENCH_WEIGHTS_DIR"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", help="Official multi-GPU process sharding, e.g. 0,1")
    parser.add_argument("--model-id", help="Select one upstream model group; required for multi-model imports")
    parser.add_argument(
        "--metric",
        "--metrics",
        action="append",
        default=[],
        help="Official metric IDs; repeat the flag or separate IDs with commas",
    )
    parser.add_argument("--expected-revisits", type=int, choices=[1, 2, 3])
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "7200"))
    )
    parser.add_argument("--json", action="store_true")
    return parser


def build_official_command(args: argparse.Namespace, results: Path, metrics: tuple[str, ...]) -> list[str]:
    """Run the in-tree official protocol, including its multi-GPU sharding."""
    if args.generated_video_dir is None or not args.generated_video_dir.exists():
        raise ValueError("--run-official requires an existing --generated-video-dir or JSONL --input-path")
    command = [
        args.python,
        "-m",
        "worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime",
        "eval",
        str(args.generated_video_dir.resolve()),
        "--output",
        str(results),
    ]
    if metrics:
        command.extend(["--metrics", *metrics])
    if args.model_id and args.model_id != "all":
        command.extend(["--models", args.model_id])
    if args.gpus:
        command.extend(["--gpus", args.gpus])
    else:
        command.extend(["--device", args.device])
    for flag, value in [("--config", args.config), ("--weights-dir", args.weights_dir)]:
        if value is not None:
            command.extend([flag, str(value.expanduser().resolve())])
    if args.expected_revisits is not None:
        command.extend(["--expected-revisits", str(args.expected_revisits)])
    return command


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Write standard scorecard artifacts, retaining upstream coverage and failures."""
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    command: list[str] | None = None
    started = time.monotonic()
    try:
        metrics = tuple(
            dict.fromkeys(name.strip() for value in args.metric for name in value.split(",") if name.strip())
        )
        if set(metrics) - METRIC_SPECS.keys():
            raise ValueError("Unknown WorldMemBench metric: " + ", ".join(sorted(set(metrics) - METRIC_SPECS.keys())))
        results = args.official_results_path
        returncode = 0
        if args.run_official:
            if results is not None:
                raise ValueError("Use --run-official or --official-results-path, not both")
            results = output / "official"
            if results.exists():
                raise ValueError(f"Use a fresh output directory; official results already exist: {results}")
            command = build_official_command(args, results, metrics)
            environment = dict(os.environ)
            with (
                (output / "runtime_stdout.log").open("w", encoding="utf-8") as stdout,
                (output / "runtime_stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.timeout,
                    check=False,
                )
            returncode = completed.returncode
        elif results is None:
            raise ValueError("Pass --official-results-path or --run-official with generated videos")
        results = results.expanduser().resolve()
        model_id, extracted, cases = load_official_results(results, model_id=args.model_id, metrics=metrics)
        evaluation_failed = any(
            row["coverage"]["status_counts"].get(status, 0)
            for row in extracted.values()
            for status in ("eval_failed", "data_invalid")
        )
        if evaluation_failed or not any(row["raw_score"] is not None for row in extracted.values()):
            returncode = returncode or 1
        scorecard = build_scorecard(
            config=CONFIG,
            output_dir=output,
            results_path=results,
            extracted=extracted,
            command=command,
            duration_seconds=time.monotonic() - started,
            returncode=returncode,
            generated_video_dir=args.generated_video_dir,
            run_status_override="failed" if returncode else None,
        )
        scorecard["dataset"]["model_id"] = model_id
        scorecard["evaluation"]["kind"] = "worldmembench_official"
        scorecard["evaluation"]["available"] = returncode == 0
        if returncode:
            scorecard["metrics"]["leaderboard"] = {}
            scorecard["evaluation"]["leaderboard_metrics"] = {}
        scorecard["validation"]["upstream_case_failures"] = evaluation_failed
        for name, item in extracted.items():
            scorecard["metrics"]["per_metric"][name].update(
                {key: item[key] for key in ("coverage", "official_scores", "official_groups")}
            )
        write_jsonl(output / "raw_metric_table.jsonl", list(scorecard["metrics"]["per_metric"].values()))
        write_jsonl(output / "per_sample_scores.jsonl", cases)
        write_json(output / "scorecard.json", scorecard)
        return scorecard, 0 if returncode == 0 else 1
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        scorecard = write_failed_scorecard(
            benchmark_id=CONFIG.benchmark_id,
            display_name=CONFIG.display_name,
            output_dir=output,
            reason=str(error),
            command=command,
        )
        return scorecard, 1


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI without importing metric models in this process."""
    args = build_parser().parse_args(argv)
    scorecard, returncode = run(args)
    print(json.dumps(scorecard, ensure_ascii=False) if args.json else str(args.output_dir / "scorecard.json"))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
