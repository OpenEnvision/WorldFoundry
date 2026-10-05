"""CLI sub-commands for creating, inspecting, and validating run plans and metric IDs.

This module wires up the ``plan`` and ``metric`` sub-commands under the
``worldfoundry-eval`` CLI.  The ``plan`` sub-command supports ``create``,
``show``, and ``validate`` operations on stable executable run plans.  The
``metric`` sub-command supports ``list``, ``show``, and ``validate``
operations on the metric registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldfoundry.evaluation.utils import BENCHMARK_TASK_ROOT, TMP_ROOT

from .presentation import print_details, print_notice, print_table, terminal_enabled
from .utils import (
    json_dump,
    load_json_mapping,
    parse_key_value_mapping,
)
from .utils import (
    task_roots_from_args as _task_roots_from_args,
)

DEFAULT_TASK_ROOT = BENCHMARK_TASK_ROOT
DEFAULT_BENCHMARK_TASK_ROOT = DEFAULT_TASK_ROOT


# ── Plan helpers ────────────────────────────────────────────


def _build_run_plan_from_cli_args(args: argparse.Namespace):
    """Build a :class:`RunPlan` from the parsed CLI argument namespace.

    Delegates to :func:`build_run_plan` or
    :func:`build_run_plan_from_task_registry` depending on whether
    ``--task-name`` was supplied.

    Args:
        args: Parsed CLI namespace with plan construction parameters.

    Returns:
        A fully constructed :class:`RunPlan` instance.
    """
    from worldfoundry.evaluation.runner import build_run_plan, build_run_plan_from_task_registry

    plan_kwargs = {
        "mode": args.mode,
        "dataset_root": args.data_path,
        "dataset_manifest": args.dataset_manifest,
        "dataset_id": args.dataset_id,
        "split": args.split,
        "requests_path": args.requests_path,
        "results_path": args.results_path,
        "model_id": args.model_id,
        "model_runner": args.model_runner,
        "model_manifest_dir": args.model_manifest_dir,
        "model_variant_id": args.model_variant,
        "model_parameters": parse_key_value_mapping(args.model_parameter),
        "model_runtime": parse_key_value_mapping(args.model_runtime),
        "model_config": load_json_mapping(args.model_config),
        "metrics": tuple(args.metric or ("artifact_count",)),
        "required_artifacts": tuple(args.required_artifact or ()),
        "limit": args.num_samples,
        "materialize_requests": args.materialize_requests,
        "run_id": args.run_id,
        "fail_on_sample_error": args.fail_on_sample_error,
        "write_artifacts_index": not args.no_artifacts_index,
    }
    if args.task_name:
        return build_run_plan_from_task_registry(
            task_name=args.task_name,
            task_roots=_task_roots_from_args(args),
            output_dir=args.output_dir,
            benchmark=args.benchmark,
            recursive=args.recursive,
            root_dir=args.root_dir,
            **plan_kwargs,
        )
    return build_run_plan(output_dir=args.output_dir, **plan_kwargs)

def _plan_result_payload(plan) -> dict:
    """Extract a human-readable summary payload from a :class:`RunPlan`.

    Includes schema version, fingerprint, runner/mode info, task name,
    request count, and best-effort validation results.

    Args:
        plan: A :class:`RunPlan` instance.

    Returns:
        Dict suitable for pretty-printing or JSON serialization.
    """
    validation = None
    try:
        from worldfoundry.evaluation.runner import validate_run_plan

        validation = validate_run_plan(plan)
    except Exception:  # pragma: no cover - validation is best-effort for summaries.
        validation = None
    return {
        "schema_version": plan.schema_version,
        "fingerprint": plan.fingerprint,
        "runner": plan.runner,
        "mode": plan.mode,
        "output_dir": plan.output_dir,
        "task_name": (plan.task or {}).get("task_name") if plan.task else None,
        "benchmark_name": (plan.task or {}).get("benchmark_name") if plan.task else None,
        "request_count": len(plan.requests),
        "validation": validation,
    }

def _handle_plan_create(args: argparse.Namespace) -> int:
    """Create a run plan and print or write it to disk.

    Args:
        args: Parsed ``plan create`` CLI namespace.

    Returns:
        ``0`` on success.
    """
    from worldfoundry.evaluation.runner import write_run_plan

    plan = _build_run_plan_from_cli_args(args)
    if args.output_json:
        write_run_plan(plan, args.output_json)
    if args.json:
        json_dump(plan.to_dict())
    else:
        if terminal_enabled():
            payload = _plan_result_payload(plan)
            print_details("Plan created", {**payload, "saved_to": args.output_json})
        else:
            payload = _plan_result_payload(plan)
            print(
                f"Plan {payload['fingerprint']}: mode={payload['mode']} "
                f"task={payload['task_name'] or '-'} requests={payload['request_count']}"
            )
            if args.output_json:
                print(f"wrote: {args.output_json}")
    return 0

def _handle_plan_show(args: argparse.Namespace) -> int:
    """Load and display an existing run plan.

    Args:
        args: Parsed ``plan show`` CLI namespace.

    Returns:
        ``0`` on success.
    """
    from worldfoundry.evaluation.runner import load_run_plan

    plan = load_run_plan(args.plan_path)
    if args.json:
        json_dump(plan.to_dict())
        return 0
    payload = _plan_result_payload(plan)
    print_details(
        "Run plan",
        {
            "schema_version": f"{payload['schema_version']}",
            "fingerprint": f"{payload['fingerprint']}",
            "runner": f"{payload['runner']}",
            "mode": f"{payload['mode']}",
            "output_dir": f"{payload['output_dir']}",
            "task_name": f"{payload['task_name'] or '-'}",
            "benchmark_name": f"{payload['benchmark_name'] or '-'}",
            "request_count": f"{payload['request_count']}",
        },
    )
    return 0

def _handle_plan_validate(args: argparse.Namespace) -> int:
    """Validate a run plan without executing it.

    Args:
        args: Parsed ``plan validate`` CLI namespace.

    Returns:
        ``0`` if the plan is valid, ``1`` if issues were found.
    """
    from worldfoundry.evaluation.runner import load_run_plan, validate_run_plan

    plan = load_run_plan(args.plan_path)
    payload = validate_run_plan(plan)
    if args.json:
        json_dump(payload)
    else:
        if terminal_enabled():
            print_details("Plan validation", {"ok": payload["ok"], "fingerprint": payload.get("fingerprint")})
            for issue in payload.get("issues", ()):
                print_notice(str(issue), level="error")
        else:
            print(f"ok: {payload['ok']}")
            print(f"fingerprint: {payload.get('fingerprint') or '-'}")
            for issue in payload.get("issues", ()):
                print(f"issue: {issue}")
    return 0 if payload["ok"] else 1

# ── Metric sub-command handlers ─────────────────────────────


def _handle_metric_list(args: argparse.Namespace) -> int:
    """List all entries in the metric registry.

    Args:
        args: Parsed ``metric list`` CLI namespace.

    Returns:
        ``0`` on success.
    """
    from worldfoundry.evaluation.tasks.metrics.registry import list_metric_registry_entries

    entries = list_metric_registry_entries()
    payload = [entry.to_dict() for entry in entries]
    if args.json:
        json_dump(payload)
        return 0

    if terminal_enabled():
        print_table(
            "Metrics",
            ("Metric", "Family", "Aliases", "Description"),
            [
                (
                    entry.id,
                    entry.family,
                    entry.aliases,
                    entry.description
                    + (f" Pattern: {entry.parameterized_prefix}<name>" if entry.parameterized_prefix else ""),
                )
                for entry in entries
            ],
        )
        return 0

    for entry in entries:
        aliases = ", ".join(entry.aliases) if entry.aliases else "-"
        parameterized = f" pattern={entry.parameterized_prefix}<name>" if entry.parameterized_prefix else ""
        print(f"{entry.id}: family={entry.family} aliases={aliases}{parameterized} {entry.description}")
    return 0

def _handle_metric_show(args: argparse.Namespace) -> int:
    """Display details for a single metric registry entry.

    Args:
        args: Parsed ``metric show`` CLI namespace with ``metric_id``.

    Returns:
        ``0`` on success.
    """
    from worldfoundry.evaluation.tasks.metrics.registry import default_metric_registry

    entry = default_metric_registry().get(args.metric_id)
    payload = entry.to_dict()
    payload["resolved_metric_id"] = args.metric_id
    if args.json:
        json_dump(payload)
        return 0

    print_details(
        "Metric details",
        {
            "id": f"{entry.id}",
            "resolved_metric_id": f"{args.metric_id}",
            "family": f"{entry.family}",
            "aliases": f"{(', '.join(entry.aliases) if entry.aliases else '-')}",
            "parameterized_prefix": f"{entry.parameterized_prefix or '-'}",
            "description": f"{entry.description}",
        },
    )
    return 0

def _handle_metric_validate(args: argparse.Namespace) -> int:
    """Validate one or more metric IDs against the registry.

    Args:
        args: Parsed ``metric validate`` CLI namespace with ``metric_id`` list.

    Returns:
        ``0`` if all IDs resolve, ``1`` if any are unknown.
    """
    from worldfoundry.evaluation.tasks.metrics.registry import validate_metric_ids

    payload = validate_metric_ids(args.metric_id)
    if args.json:
        json_dump(payload)
    else:
        if terminal_enabled():
            print_details("Metric validation", {"ok": payload["ok"]})
            print_table(
                "Resolved metrics",
                ("Metric", "Registry ID", "Parameterized"),
                [(item["metric_id"], item["registry_id"], item["parameterized"]) for item in payload["metrics"]],
            )
            for metric_id in payload["unknown_metrics"]:
                print_notice(f"Unknown metric: {metric_id}", level="error")
        else:
            print(f"ok: {payload['ok']}")
            for item in payload["metrics"]:
                print(
                    f"metric: {item['metric_id']} registry_id={item['registry_id']} "
                    f"parameterized={item['parameterized']}"
                )
            for metric_id in payload["unknown_metrics"]:
                print(f"unknown_metric: {metric_id}")
    return 0 if payload["ok"] else 1


# ── Sub-parser registration ─────────────────────────────────


def register_plan_metric_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``plan`` and ``metric`` sub-commands on the root CLI parser.

    Args:
        subparsers: The subparser collection from the root
            ``worldfoundry-eval`` command.
    """
    plan_parser = subparsers.add_parser(
        "plan",
        help="Create, inspect, and validate stable executable run plans",
    )
    plan_subparsers = plan_parser.add_subparsers(dest="plan_command", required=True)

    plan_create_parser = plan_subparsers.add_parser(
        "create",
        help="Create a worldfoundry-run-plan JSON file",
    )
    plan_create_parser.add_argument("--task-name")
    plan_create_parser.add_argument("--task-root", action="append", type=Path)
    plan_create_parser.add_argument("--include-path", "--benchmark-include-path", action="append", type=Path, dest="include_path")
    plan_create_parser.add_argument("--root-dir", type=Path)
    plan_create_parser.add_argument("--recursive", action="store_true")
    plan_create_parser.add_argument("--benchmark")
    plan_create_parser.add_argument("--mode", choices=["existing-results", "existing", "results", "model", "generate"], default="model")
    plan_create_parser.add_argument("--data-path", type=Path)
    plan_create_parser.add_argument("--dataset-manifest", type=Path)
    plan_create_parser.add_argument("--requests-path", type=Path)
    plan_create_parser.add_argument("--results-path", type=Path)
    plan_create_parser.add_argument("--output-dir", type=Path, default=TMP_ROOT / "worldfoundry_plan_run")
    plan_create_parser.add_argument("--output-json", type=Path)
    plan_create_parser.add_argument("--model-id")
    plan_create_parser.add_argument("--model-runner")
    plan_create_parser.add_argument("--model-manifest-dir", type=Path)
    plan_create_parser.add_argument("--model-variant")
    plan_create_parser.add_argument("--model-parameter", action="append", default=None, metavar="KEY=VALUE")
    plan_create_parser.add_argument("--model-runtime", action="append", default=None, metavar="KEY=VALUE")
    plan_create_parser.add_argument("--model-config", type=Path)
    plan_create_parser.add_argument("--dataset-id")
    plan_create_parser.add_argument("--split", default="default")
    plan_create_parser.add_argument("--metric", action="append", default=None)
    plan_create_parser.add_argument("--required-artifact", action="append", default=None)
    plan_create_parser.add_argument("--num-samples", type=int, default=None)
    plan_create_parser.add_argument("--materialize-requests", action="store_true")
    plan_create_parser.add_argument("--run-id")
    plan_create_parser.add_argument("--fail-on-sample-error", action="store_true")
    plan_create_parser.add_argument("--no-artifacts-index", action="store_true")
    plan_create_parser.add_argument("--json", action="store_true")
    plan_create_parser.set_defaults(func=_handle_plan_create)

    plan_show_parser = plan_subparsers.add_parser("show", help="Show one run plan")
    plan_show_parser.add_argument("plan_path", type=Path)
    plan_show_parser.add_argument("--json", action="store_true")
    plan_show_parser.set_defaults(func=_handle_plan_show)

    plan_validate_parser = plan_subparsers.add_parser("validate", help="Validate a run plan without executing it")
    plan_validate_parser.add_argument("plan_path", type=Path)
    plan_validate_parser.add_argument("--json", action="store_true")
    plan_validate_parser.set_defaults(func=_handle_plan_validate)

    metric_parser = subparsers.add_parser(
        "metric",
        help="Inspect and validate executable metric ids",
    )
    metric_subparsers = metric_parser.add_subparsers(dest="metric_command", required=True)

    metric_list_parser = metric_subparsers.add_parser("list", help="List executable metric registry entries")
    metric_list_parser.add_argument("--json", action="store_true")
    metric_list_parser.set_defaults(func=_handle_metric_list)

    metric_show_parser = metric_subparsers.add_parser("show", help="Show one metric registry entry")
    metric_show_parser.add_argument("metric_id")
    metric_show_parser.add_argument("--json", action="store_true")
    metric_show_parser.set_defaults(func=_handle_metric_show)

    metric_validate_parser = metric_subparsers.add_parser("validate", help="Validate metric ids")
    metric_validate_parser.add_argument("metric_id", nargs="+")
    metric_validate_parser.add_argument("--json", action="store_true")
    metric_validate_parser.set_defaults(func=_handle_metric_validate)
