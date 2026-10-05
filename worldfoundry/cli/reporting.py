"""CLI commands for comparing runs, indexing run directories, and validating contract artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .presentation import print_report
from .utils import CliUsageError, json_dump

# ── Compare runs ────────────────────────────────────────────────


def _handle_compare_runs(args: argparse.Namespace) -> int:
    """Compare WorldFoundry run results, optionally selecting runs from an index."""
    from worldfoundry.evaluation.reporting import (
        build_markdown_comparison,
        run_paths_from_index,
        write_run_comparison,
    )

    run_paths = list(args.run or ())
    positional_labels = list(args.label or ())
    if positional_labels and len(positional_labels) != len(run_paths):
        raise CliUsageError("--label count must match explicit run count")

    try:
        indexed_runs = [
            selected_run
            for index_path in args.index or ()
            for selected_run in run_paths_from_index(
                index_path,
                benchmarks=args.index_benchmark,
                models=args.index_model,
                datasets=args.index_dataset,
                run_ids=args.index_run_id,
                statuses=args.index_status,
                require_score_valid=args.require_score_valid,
                require_leaderboard_valid=args.require_leaderboard_valid,
                required_metrics=args.require_metric,
            )
        ]
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise CliUsageError(str(exc)) from exc

    index_labels = [selected_run["label"] for selected_run in indexed_runs]
    if positional_labels or index_labels:
        labels: list[str | None] | None = list(positional_labels)
        labels.extend([None] * (len(run_paths) - len(labels)))
        labels.extend(index_labels)
    else:
        labels = None
    run_paths.extend(Path(selected_run["path"]) for selected_run in indexed_runs)

    baseline: int | str | None = None
    if args.baseline is not None:
        if args.baseline_label is not None or labels is not None:
            labels = [args.baseline_label, *(labels or [None] * len(run_paths))]
        run_paths = [args.baseline, *run_paths]
        baseline = 0
    elif args.baseline_label is not None:
        raise CliUsageError("--baseline-label requires --baseline")
    elif args.baseline_run is not None:
        baseline = args.baseline_run

    if not run_paths:
        raise CliUsageError("at least one run path or --index selection is required")
    if labels is not None and len(labels) != len(run_paths):
        raise CliUsageError("--label count must match run count")

    try:
        comparison = write_run_comparison(
            run_paths,
            labels=labels,
            baseline=baseline,
            metric_ids=args.metric,
            output_json=args.output_json,
            output_md=args.output_md,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise CliUsageError(str(exc)) from exc

    if args.json:
        json_dump(comparison)
    else:
        print_report(build_markdown_comparison(comparison))
    return 1 if comparison.get("issues") and args.fail_on_issue else 0

# ── Index runs ──────────────────────────────────────────────────


def _handle_index_runs(args: argparse.Namespace) -> int:
    """Index WorldFoundry run directories below one or more roots."""
    from worldfoundry.evaluation.reporting import build_markdown_run_index, write_run_index

    try:
        index = write_run_index(
            list(args.root),
            include_invalid=args.include_invalid,
            output_dir=args.output_dir,
            output_json=args.output_json,
            output_jsonl=args.output_jsonl,
            output_html=args.output_html,
        )
    except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as exc:
        raise CliUsageError(str(exc)) from exc

    if args.json:
        json_dump(index)
    else:
        print_report(build_markdown_run_index(index))
    return 1 if index.get("issues") and args.fail_on_issue else 0


# ── Validate artifacts ──────────────────────────────────────────


def _validate_artifact_kind_choices() -> tuple[str, ...]:
    """Resolve the accepted artifact-kind choices from the reporting module."""
    from worldfoundry.evaluation.reporting import CONTRACT_ARTIFACT_KIND_CHOICES

    return CONTRACT_ARTIFACT_KIND_CHOICES


def _handle_validate_artifact(args: argparse.Namespace) -> int:
    """Validate WorldFoundry summary, scorecard, index, comparison, or suite artifacts."""
    from worldfoundry.evaluation.reporting import (
        build_markdown_contract_validation,
        normalize_contract_artifact_kind,
        validate_contract_paths,
    )

    kind = normalize_contract_artifact_kind(args.kind)
    paths = [*list(args.path or ()), *list(args.path_option or ())]
    if not paths:
        raise CliUsageError("at least one artifact path is required")
    report = validate_contract_paths(
        paths,
        kind=kind,
        check_artifacts=args.check_artifacts,
        strict=args.strict,
    )
    if args.json:
        json_dump(report)
    else:
        print_report(build_markdown_contract_validation(report))
    return 0 if report.get("ok") else 1


# ── Parser registration ─────────────────────────────────────────


def register_reporting_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``compare-runs``, ``index-runs``, and ``validate-artifact`` subparsers."""
    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare explicit WorldFoundry runs or runs selected from an index",
    )
    compare_runs_parser.add_argument("run", nargs="*", type=Path)
    compare_runs_parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline run directory or summary file. When set, it is prepended to compared runs.",
    )
    compare_runs_parser.add_argument("--baseline-label", help="Label to use for --baseline.")
    compare_runs_parser.add_argument(
        "--baseline-run",
        help="Existing run label, run_id, or model_id to use as the baseline for deltas.",
    )
    compare_runs_parser.add_argument("--label", action="append", default=None, help="Label for each explicit positional run.")
    compare_runs_parser.add_argument(
        "--index",
        action="append",
        type=Path,
        default=None,
        help="Select runs from an index.json or index.jsonl file. Repeatable.",
    )
    compare_runs_parser.add_argument("--index-benchmark", action="append", default=None, help="Only select index rows with this benchmark name. Repeatable.")
    compare_runs_parser.add_argument("--index-model", action="append", default=None, help="Only select index rows with this model_id or model_name. Repeatable.")
    compare_runs_parser.add_argument("--index-dataset", action="append", default=None, help="Only select index rows with this dataset_id. Repeatable.")
    compare_runs_parser.add_argument("--index-run-id", action="append", default=None, help="Only select index rows with this run_id. Repeatable.")
    compare_runs_parser.add_argument("--index-status", action="append", default=None, help="Only select index rows with this run status. Repeatable.")
    compare_runs_parser.add_argument("--require-score-valid", action="store_true", help="Only select index rows where score_valid is true.")
    compare_runs_parser.add_argument("--require-leaderboard-valid", action="store_true", help="Only select index rows where leaderboard_valid is true.")
    compare_runs_parser.add_argument("--require-metric", action="append", default=None, help="Only select index rows containing this metric id. Repeatable.")
    compare_runs_parser.add_argument("--metric", action="append", default=None, help="Metric id to include. Repeatable.")
    compare_runs_parser.add_argument("--output-json", type=Path, help="Write comparison payload to this JSON file.")
    compare_runs_parser.add_argument("--output-md", type=Path, help="Write comparison table to this Markdown file.")
    compare_runs_parser.add_argument("--fail-on-issue", action="store_true", help="Exit 1 when comparison issues are present.")
    compare_runs_parser.add_argument("--json", action="store_true")
    compare_runs_parser.set_defaults(func=_handle_compare_runs)

    index_runs_parser = subparsers.add_parser(
        "index-runs",
        help="Index WorldFoundry run directories below one or more roots",
    )
    index_runs_parser.add_argument("root", nargs="+", type=Path)
    index_runs_parser.add_argument("--include-invalid", action="store_true", help="Include rows for discovered summaries that cannot be loaded.")
    index_runs_parser.add_argument("--output-dir", type=Path, help="Write index.json and index.jsonl under this directory.")
    index_runs_parser.add_argument("--output-json", type=Path, help="Write index payload to this JSON file.")
    index_runs_parser.add_argument("--output-jsonl", type=Path, help="Write one indexed run row per JSONL line.")
    index_runs_parser.add_argument("--output-html", type=Path, help="Write a dependency-free HTML run browser.")
    index_runs_parser.add_argument("--fail-on-issue", action="store_true", help="Exit 1 when index issues are present.")
    index_runs_parser.add_argument("--json", action="store_true")
    index_runs_parser.set_defaults(func=_handle_index_runs)

    validate_artifact_parser = subparsers.add_parser(
        "validate-artifact",
        help="Validate WorldFoundry summary, scorecard, index, comparison, and suite artifacts",
    )
    validate_artifact_parser.add_argument("path", nargs="*", type=Path)
    validate_artifact_parser.add_argument(
        "--path",
        action="append",
        dest="path_option",
        type=Path,
        default=None,
        help="Artifact path to validate. May repeat. Positional paths are also accepted.",
    )
    validate_artifact_parser.add_argument(
        "--kind",
        choices=_validate_artifact_kind_choices(),
        default="auto",
        help="Expected artifact kind. auto reads schema_version.",
    )
    validate_artifact_parser.add_argument("--check-artifacts", action="store_true", help="Check local paths referenced by artifacts mappings.")
    validate_artifact_parser.add_argument("--strict", action="store_true", help="Treat warnings as validation failures.")
    validate_artifact_parser.add_argument("--json", action="store_true")
    validate_artifact_parser.set_defaults(func=_handle_validate_artifact)
