# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""``wf embodied``: simulator-backed closed-loop run, serve, merge, and plan."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .presentation import print_details, print_table, terminal_enabled
from .utils import json_dump

_EMBODIED_ACCEPTED_LICENSES_ENV = "WORLDFOUNDRY_ACCEPTED_LICENSES"


def _accept_embodied_licenses(license_ids: list[str] | None) -> None:
    """Merge explicit CLI acknowledgements into the embodied license environment."""

    accepted = [item.strip() for item in os.environ.get(_EMBODIED_ACCEPTED_LICENSES_ENV, "").split(",") if item.strip()]
    for value in license_ids or ():
        for item in str(value).split(","):
            normalized = item.strip()
            if normalized and normalized not in accepted:
                accepted.append(normalized)
    if accepted:
        os.environ[_EMBODIED_ACCEPTED_LICENSES_ENV] = ",".join(accepted)


def _handle_embodied_run(args: argparse.Namespace) -> int:
    """Run an embodied closed-loop evaluation config."""
    _accept_embodied_licenses(getattr(args, "accept_license", None))
    from worldfoundry.evals.tasks.embodied.orchestrator import run_embodied_eval_config

    result = run_embodied_eval_config(
        args.config,
        output_dir=args.output_dir,
        server_url=args.server_url,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        eval_id=args.eval_id,
        no_docker=args.no_docker,
        no_save=args.no_save,
        pull_docker=args.pull_docker,
    )
    if args.json:
        json_dump(result.to_dict())
    else:
        print_details(
            "Embodied evaluation",
            {
                "status": result.evaluate_result.status,
                "samples": result.evaluate_result.sample_count,
                "scorecard": result.evaluate_result.scorecard_path,
            },
            plain=f"Embodied eval {result.evaluate_result.status}: samples={result.evaluate_result.sample_count}, scorecard={result.evaluate_result.scorecard_path}",
        )
    return result.evaluate_result.exit_code


def _handle_embodied_serve(args: argparse.Namespace) -> int:
    """Serve an embodied policy adapter over WebSocket."""
    from worldfoundry.evals.tasks.embodied.model_server.serve import serve_from_config

    serve_from_config(args.config, host=args.host, port=args.port)
    return 0


def _handle_embodied_merge(args: argparse.Namespace) -> int:
    """Merge embodied shard outputs into one scorecard."""
    from worldfoundry.evals.tasks.embodied.config_loader import load_canonical_embodied_config
    from worldfoundry.evals.tasks.embodied.merge_results import merge_embodied_results

    config = load_canonical_embodied_config(args.config, output_dir=args.output_dir) if args.config else {}
    output_dir = args.output_dir or config.get("output_dir") or "."
    result = merge_embodied_results(
        output_dir,
        eval_id=args.eval_id,
        metric_ids=tuple(args.metric or ("task_success", "success_rate")),
        config=config,
    )
    if args.json:
        json_dump(result.to_dict())
    else:
        print_details(
            "Embodied merge",
            {"status": result.status, "samples": result.sample_count, "scorecard": result.scorecard_path},
            plain=f"Embodied merge {result.status}: samples={result.sample_count}, scorecard={result.scorecard_path}",
        )
    return result.exit_code


def _handle_embodied_plan(args: argparse.Namespace) -> int:
    """Dry-run an embodied config and print materialized request counts."""
    from worldfoundry.evals.tasks.embodied.config_loader import load_canonical_embodied_config
    from worldfoundry.evals.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests

    config = load_canonical_embodied_config(args.config, output_dir=args.output_dir, server_url=args.server_url)
    benchmarks = []
    total = 0
    for bench_cfg in config.get("benchmarks") or ():
        requests = materialize_embodied_rollout_requests(bench_cfg)
        benchmarks.append(
            {
                "id": bench_cfg.get("id"),
                "benchmark_id": bench_cfg.get("benchmark_id"),
                "request_count": len(requests),
                "sample_ids": [request.sample_id for request in requests[:5]],
            }
        )
        total += len(requests)
    payload = {
        "id": config.get("id"),
        "output_dir": config.get("output_dir"),
        "model": config.get("model"),
        "server": config.get("server"),
        "docker": config.get("docker"),
        "total_requests": total,
        "benchmarks": benchmarks,
    }
    if args.json:
        json_dump(payload)
    else:
        if terminal_enabled():
            print_details("Embodied plan", {"requests": total, "output": config.get("output_dir")})
            print_table(
                "Benchmarks",
                ("Benchmark", "Requests"),
                [(item["benchmark_id"], item["request_count"]) for item in benchmarks],
            )
        else:
            print(f"Embodied plan: requests={total}, output_dir={config.get('output_dir')}")
            for item in benchmarks:
                print(f"  {item['benchmark_id']}: {item['request_count']} request(s)")
    return 0


def register_embodied_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``wf embodied`` and its run/serve/merge/plan subcommands."""
    embodied_parser = subparsers.add_parser(
        "embodied",
        help="Run simulator-backed embodied closed-loop evaluations",
        description="Run, serve, plan, and merge WorldFoundry embodied closed-loop benchmark evaluations.",
    )
    embodied_subparsers = embodied_parser.add_subparsers(dest="embodied_command", required=True)

    embodied_run_parser = embodied_subparsers.add_parser("run", help="Run an embodied eval config")
    embodied_run_parser.add_argument("--config", "-c", type=Path, required=True)
    embodied_run_parser.add_argument("--output-dir", type=Path)
    embodied_run_parser.add_argument("--server-url")
    embodied_run_parser.add_argument("--shard-id", type=int)
    embodied_run_parser.add_argument("--num-shards", type=int)
    embodied_run_parser.add_argument("--eval-id")
    embodied_run_parser.add_argument("--no-docker", action="store_true")
    embodied_run_parser.add_argument("--pull-docker", action="store_true")
    embodied_run_parser.add_argument("--no-save", action="store_true")
    embodied_run_parser.add_argument(
        "--accept-license",
        action="append",
        default=None,
        metavar="LICENSE_ID",
        help="Acknowledge an embodied simulator/data license; repeat for multiple licenses",
    )
    embodied_run_parser.add_argument("--json", action="store_true")
    embodied_run_parser.set_defaults(func=_handle_embodied_run)

    embodied_serve_parser = embodied_subparsers.add_parser("serve", help="Serve an embodied policy over WebSocket")
    embodied_serve_parser.add_argument("--config", "-c", type=Path, required=True)
    embodied_serve_parser.add_argument("--host")
    embodied_serve_parser.add_argument("--port", type=int)
    embodied_serve_parser.set_defaults(func=_handle_embodied_serve)

    embodied_merge_parser = embodied_subparsers.add_parser("merge", help="Merge sharded embodied outputs")
    embodied_merge_parser.add_argument("--config", "-c", type=Path)
    embodied_merge_parser.add_argument("--output-dir", type=Path)
    embodied_merge_parser.add_argument("--eval-id")
    embodied_merge_parser.add_argument("--metric", action="append", default=None)
    embodied_merge_parser.add_argument("--json", action="store_true")
    embodied_merge_parser.set_defaults(func=_handle_embodied_merge)

    embodied_plan_parser = embodied_subparsers.add_parser("plan", help="Dry-run an embodied eval config")
    embodied_plan_parser.add_argument("--config", "-c", type=Path, required=True)
    embodied_plan_parser.add_argument("--output-dir", type=Path)
    embodied_plan_parser.add_argument("--server-url")
    embodied_plan_parser.add_argument("--json", action="store_true")
    embodied_plan_parser.set_defaults(func=_handle_embodied_plan)
