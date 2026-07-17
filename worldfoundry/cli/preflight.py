"""CLI registration for benchmark runtime preflight commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.runtime_preflight import (
    DEFAULT_PROFILE_DIR,
    REPO_ROOT,
    run_preflight,
)


def register_preflight_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "preflight",
        help="Check benchmark runtime requirements without running a benchmark",
    )
    preflight_subparsers = parser.add_subparsers(dest="preflight_command", required=True)
    runtime_parser = preflight_subparsers.add_parser(
        "runtime",
        help="Check runtime-profile environment, paths, imports, and CUDA readiness",
    )
    runtime_parser.add_argument("--profile", required=True, help="Runtime profile id, or 'all'.")
    runtime_parser.add_argument("--manifest", type=Path, default=DEFAULT_PROFILE_DIR)
    runtime_parser.add_argument("--output-dir", type=Path, required=True)
    runtime_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    runtime_parser.add_argument("--import-timeout", type=float, default=30.0)
    runtime_parser.add_argument("--json", action="store_true")
    runtime_parser.set_defaults(func=_runtime_preflight)


def _runtime_preflight(args: argparse.Namespace) -> int:
    import json

    report = run_preflight(
        profile=args.profile,
        manifest=args.manifest,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        import_timeout=args.import_timeout,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        state = "ready" if report["ok"] else "not ready"
        print(f"runtime preflight: {state}; report: {report['report_path']}")
    return 0 if report["ok"] else 2
