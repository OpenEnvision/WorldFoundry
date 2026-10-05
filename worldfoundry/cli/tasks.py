"""Task-related CLI command handlers and parser registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from worldfoundry.evaluation.utils import BENCHMARK_TASK_ROOT, write_json, write_jsonl

from .presentation import print_details, print_table, terminal_enabled
from .utils import CliUsageError, json_dump
from .utils import task_roots_from_args as _task_roots_from_args

DEFAULT_TASK_ROOT = BENCHMARK_TASK_ROOT
DEFAULT_BENCHMARK_TASK_ROOT = DEFAULT_TASK_ROOT

# ── Task selector helpers ───────────────────────────────────────


def _add_task_selector(parser: argparse.ArgumentParser, *, require_data_path: bool) -> None:
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--benchmark-name")
    if require_data_path:
        parser.add_argument("--data-path", required=True)


# ── Built-in task registry commands ──────────────────────────────


def _handle_tasks_list(args: argparse.Namespace) -> int:
    """Print benchmark task index entries from the built-in task registry."""
    from worldfoundry.evaluation.tasks.catalog.specs import list_benchmark_zoo_cli_tasks

    items = list_benchmark_zoo_cli_tasks(
        task_type=args.task_type,
        suite=args.suite,
        backend=args.backend,
        source_kind=args.source_kind,
    )
    if args.json:
        json_dump(items)
        return 0

    if args.flat:
        for item in items:
            print(_format_flat_task_list_item(item))
        return 0

    _print_grouped_task_list(items)
    return 0

def _format_flat_task_list_item(item: dict) -> str:
    """Format one task entry for the compact one-line listing."""
    label = _task_list_label(item)
    return (
        f"{label} "
        f"[{item['suite']}/{item['backend']}] "
        f"[{item['evaluation_protocol']}] {item['description']}"
    )

def _task_list_group_key(item: dict) -> tuple[str, str, str, str]:
    """Derive the grouping key for a task list entry."""
    return (
        str(item.get("source_kind", "")),
        str(item.get("suite", "")),
        str(item.get("backend", "")),
        str(item.get("evaluation_protocol", "")),
    )

def _task_list_group_title(item: dict) -> str:
    """Build a human-readable group title from the grouping key."""
    source_kind, suite, backend, protocol = _task_list_group_key(item)
    if source_kind == "benchmark_zoo":
        return f"benchmark_zoo: {backend}"
    if protocol == backend:
        return f"{suite}: {backend}"
    return f"{suite}: {backend} [{protocol}]"

def _task_list_label(item: dict) -> str:
    """Render the primary label for a task entry, preferring ``benchmark_zoo_id`` when available."""
    if item.get("source_kind") == "benchmark_zoo":
        return str(item.get("benchmark_zoo_id") or item["task_type"])
    return str(item["task_type"])

def _task_list_detail(item: dict) -> str:
    """Render the detail line for a task entry, including validation surface or benchmark name."""
    if item.get("source_kind") == "benchmark_zoo":
        status = "validated" if item.get("official_runtime_validated") else "contract"
        return f"{status} surface | {item['description']}"
    return f"benchmark={item['benchmark_name']} | {item['description']}"

def _print_grouped_task_list(items: list[dict]) -> None:
    """Print task entries grouped by suite, backend, and evaluation protocol."""
    if terminal_enabled():
        print_table(
            "Benchmark tasks",
            ("Task", "Suite / backend", "Details"),
            [
                (_task_list_label(item), _task_list_group_title(item), _task_list_detail(item))
                for item in sorted(items, key=lambda row: (_task_list_group_key(row), _task_list_label(row)))
            ],
        )
        return

    if not items:
        print("No registered tasks matched the filters.")
        return

    current_group: tuple[str, str, str, str] | None = None
    for item in sorted(items, key=lambda row: (_task_list_group_key(row), _task_list_label(row))):
        group = _task_list_group_key(item)
        if group != current_group:
            if current_group is not None:
                print()
            print(_task_list_group_title(item))
            current_group = group

        print(f"  {_task_list_label(item):<34} {_task_list_detail(item)}")

def _handle_tasks_show(args: argparse.Namespace) -> int:
    """Show one built-in benchmark task entry by task type/benchmark pair."""
    from worldfoundry.evaluation.tasks.catalog.specs import get_benchmark_zoo_cli_task, list_benchmark_zoo_cli_tasks

    if args.benchmark_name:
        item = get_benchmark_zoo_cli_task(args.task_type, args.benchmark_name)
    else:
        matches = list_benchmark_zoo_cli_tasks(task_type=args.task_type)
        if not matches:
            raise CliUsageError(f"unknown task: {args.task_type!r}")
        if len(matches) > 1:
            benchmark_names = ", ".join(row["benchmark_name"] for row in matches)
            raise CliUsageError(
                f"task {args.task_type!r} exists in multiple benchmarks: {benchmark_names}; pass --benchmark-name"
            )
        item = matches[0]
    if args.json:
        json_dump(item)
        return 0

    if terminal_enabled():
        keys = (
            "task_type",
            "benchmark_name",
            "suite",
            "backend",
            "name",
            "protocol",
            "capability_track",
            "schema_type",
            "evaluation_protocol",
            "input_keys",
            "output_keys",
            "metric_groups",
            "task_yaml_path",
            "source_kind",
            "benchmark_zoo_id",
            "contract_only_surface",
            "requires_upstream_runtime",
            "official_runtime_validated",
            "description",
        )
        print_details("Benchmark task", {key: item[key] for key in keys if key in item})
        return 0
    print_details(
        "Benchmark task",
        {
            "task_type": f"{item['task_type']}",
            "benchmark_name": f"{item['benchmark_name']}",
            "suite": f"{item['suite']}",
            "backend": f"{item['backend']}",
            "name": f"{item['name']}",
            "protocol": f"{item['protocol']}",
            "capability_track": f"{item['capability_track']}",
            "schema_type": f"{item['schema_type']}",
            "evaluation_protocol": f"{item['evaluation_protocol']}",
            "input_keys": f"{', '.join(item['input_keys'])}",
            "output_keys": f"{', '.join(item['output_keys'])}",
            "metric_groups": f"{', '.join(item['metric_groups'])}",
            "task_yaml_path": f"{item['task_yaml_path']}",
            "source_kind": f"{item['source_kind']}",
        },
    )
    if item["benchmark_zoo_id"]:
        print(f"benchmark_zoo_id: {item['benchmark_zoo_id']}")
        print(f"contract_only_surface: {item['contract_only_surface']}")
        print(f"requires_upstream_runtime: {item['requires_upstream_runtime']}")
        print(f"official_runtime_validated: {item['official_runtime_validated']}")
    print(f"description: {item['description']}")
    return 0

def _handle_tasks_catalog(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.catalog.specs import build_benchmark_zoo_catalog_registry

    catalog = build_benchmark_zoo_catalog_registry(
        task_type=args.task_type,
        suite=args.suite,
        backend=args.backend,
        source_kind=args.source_kind,
    )
    items = [item.to_dict() for item in catalog.list_benchmarks()]
    if args.json:
        json_dump(items)
        return 0

    if terminal_enabled():
        print_table(
            "Task catalog",
            ("Benchmark", "Schema", "Tasks", "Tags"),
            [
                (item["name"], item["schema_version"], [task["name"] for task in item["tasks"]], item["tags"])
                for item in items
            ],
        )
        return 0

    for item in items:
        task_names = ", ".join(task["name"] for task in item["tasks"])
        print(f"{item['name']} [{item['schema_version']}] tasks={task_names} tags={', '.join(item['tags'])}")
    return 0

# ── Suite commands ───────────────────────────────────────────────


def _handle_suites_list(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.runner import list_model_benchmark_suite_presets

    suites = [dict(item) for item in list_model_benchmark_suite_presets(args.suite_preset_path)]
    if args.json:
        json_dump(suites)
        return 0
    if terminal_enabled():
        print_table(
            "Evaluation suites",
            ("Suite", "Name", "Models", "Benchmarks"),
            [
                (
                    suite["id"],
                    suite.get("name"),
                    len(suite.get("model_ids") or []),
                    len(suite.get("benchmark_ids") or []),
                )
                for suite in suites
            ],
        )
        return 0

    for suite in suites:
        print(
            f"{suite['id']}: models={len(suite.get('model_ids') or [])} "
            f"benchmarks={len(suite.get('benchmark_ids') or [])} {suite.get('name') or ''}".rstrip()
        )
    return 0

def _handle_suites_show(args: argparse.Namespace) -> int:
    """Show one model-benchmark suite preset with its models and benchmarks."""
    from worldfoundry.evaluation.runner import get_model_benchmark_suite_preset

    try:
        suite = dict(get_model_benchmark_suite_preset(args.suite, args.suite_preset_path))
    except KeyError as exc:
        raise CliUsageError(str(exc)) from exc
    if args.json:
        json_dump(suite)
        return 0
    print_details(
        "Evaluation suite",
        {
            "id": f"{suite['id']}",
            "name": f"{suite.get('name') or suite['id']}",
            "aliases": f"{', '.join(suite.get('aliases') or ()) or '-'}",
            "models": f"{', '.join(suite.get('model_ids') or ()) or '-'}",
            "benchmarks": f"{', '.join(suite.get('benchmark_ids') or ()) or '-'}",
        },
    )
    return 0

# ── Filesystem task catalog commands ─────────────────────────────


def _load_filesystem_task_registry(args: argparse.Namespace):
    """Load a task registry from filesystem YAML roots derived from CLI args."""
    from worldfoundry.evaluation.tasks.catalog.registry import load_task_registry_from_paths

    return load_task_registry_from_paths(
        _task_roots_from_args(args),
        recursive=args.recursive,
        root_dir=args.root_dir,
    )

def _handle_task_list(args: argparse.Namespace) -> int:
    """List filesystem task YAML entries from one or more root paths."""
    registry = _load_filesystem_task_registry(args)
    items = [
        item.to_dict()
        for item in registry.list(
            benchmark=args.benchmark,
            tag=args.tag,
            protocol=args.protocol,
        )
    ]
    if args.json:
        json_dump(items)
        return 0

    if terminal_enabled():
        print_table(
            "Task files",
            ("Task", "Benchmark", "Protocol", "Source"),
            [(item["task_name"], item["benchmark_name"], item["protocol"], item["source_path"]) for item in items],
        )
        return 0

    for item in items:
        eval_protocol = ", ".join(item["evaluation_protocol"]) or "-"
        print(
            f"{item['task_name']} [{item['benchmark_name']}] "
            f"protocol={item['protocol']} eval={eval_protocol} "
            f"source={item['source_path']}"
        )
    return 0

def _handle_task_show(args: argparse.Namespace) -> int:
    """Display one filesystem task entry in a normalized, user-readable form."""
    registry = _load_filesystem_task_registry(args)
    item = registry.get(args.task_name, benchmark=args.benchmark).to_dict()
    if args.json:
        json_dump(item)
        return 0

    print_details(
        "Task details",
        {
            "task_name": f"{item['task_name']}",
            "benchmark_name": f"{item['benchmark_name']}",
            "protocol": f"{item['protocol']}",
            "evaluation_protocol": f"{', '.join(item['evaluation_protocol']) or '-'}",
            "input_keys": f"{', '.join(item['input_keys']) or '-'}",
            "output_keys": f"{', '.join(item['output_keys']) or '-'}",
            "metric_ids": f"{', '.join(item['metric_ids']) or '-'}",
            "metric_groups": f"{', '.join(item['metric_groups']) or '-'}",
            "tags": f"{', '.join(item['tags']) or '-'}",
            "source_path": f"{item['source_path']}",
            "description": f"{item['description']}",
        },
    )
    return 0

def _task_validation_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    """Collect YAML paths for validation from explicit args or task roots."""
    from worldfoundry.evaluation.tasks.catalog.registry import iter_task_yaml_paths

    roots = tuple(Path(path) for path in (args.path or ()))
    if not roots:
        roots = _task_roots_from_args(args)

    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(iter_task_yaml_paths(root, recursive=args.recursive))
        else:
            paths.append(root)
    return tuple(paths)

def _handle_task_validate(args: argparse.Namespace) -> int:
    """Validate one or many task YAML paths and return non-zero when invalid."""
    from worldfoundry.evaluation.tasks.catalog.registry import validate_task_yaml_file

    items = [
        validate_task_yaml_file(
            path,
            root_dir=args.root_dir or (path.parent if path.is_file() else None),
        )
        for path in _task_validation_paths(args)
    ]
    invalid = [item for item in items if not item["ok"]]
    payload = {
        "ok": not invalid,
        "path_count": len(items),
        "valid_count": len(items) - len(invalid),
        "invalid_count": len(invalid),
        "items": items,
    }
    if args.json:
        json_dump(payload)
    else:
        if terminal_enabled():
            print_details(
                "Task validation",
                {
                    "ok": payload["ok"],
                    "paths": payload["path_count"],
                    "valid": payload["valid_count"],
                    "invalid": payload["invalid_count"],
                },
            )
            print_table(
                "Task files",
                ("Path", "OK", "Details"),
                [
                    (
                        item["path"],
                        item["ok"],
                        f"{item['task_count']} tasks · {item['kind']}"
                        if item["ok"]
                        else f"{item['error_type']}: {item['error']}",
                    )
                    for item in items
                ],
            )
        else:
            for item in items:
                if item["ok"]:
                    print(f"ok: {item['path']} tasks={item['task_count']} kind={item['kind']}")
                else:
                    print(f"error: {item['path']} {item['error_type']}: {item['error']}", file=sys.stderr)
            print(
                f"validated {payload['path_count']} paths: "
                f"{payload['valid_count']} ok, {payload['invalid_count']} invalid"
            )
    return 0 if payload["ok"] else 1


def _handle_task_materialize(args: argparse.Namespace) -> int:
    """Materialize GenerationRequest rows from one task YAML entry."""
    from worldfoundry.evaluation.api import GenerationRequest
    from worldfoundry.evaluation.tasks.execution.orchestration.materialize import (
        MATERIALIZED_REQUESTS_SCHEMA_VERSION,
        MaterializedRequests,
    )
    from worldfoundry.evaluation.tasks.execution.orchestration.plan import build_run_plan

    if args.dataset_root is None and args.dataset_manifest is None:
        raise CliUsageError("task materialize requires --dataset-root or --dataset-manifest")

    registry = _load_filesystem_task_registry(args)
    entry = registry.get(args.task_name, benchmark=args.benchmark)
    plan = build_run_plan(
        output_dir=args.output_dir,
        task_entry=entry,
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        split=args.split,
        limit=args.num_samples,
        materialize_requests=True,
    )
    requests = tuple(GenerationRequest.from_dict(item) for item in plan.requests)
    materialized = MaterializedRequests(
        schema_version=MATERIALIZED_REQUESTS_SCHEMA_VERSION,
        task_type=entry.task.name,
        benchmark_name=entry.benchmark_name,
        split=str(plan.dataset.get("split", args.split)),
        requests=requests,
    )
    payload = materialized.to_dict()
    if args.output_json:
        write_json(args.output_json, payload, atomic=False)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, [request.to_dict() for request in requests], atomic=False)
    if args.json:
        json_dump(payload)
    else:
        if terminal_enabled():
            print_details(
                "Requests materialized",
                {
                    "task": materialized.task_type,
                    "benchmark": materialized.benchmark_name,
                    "requests": materialized.sample_count,
                    "JSON": args.output_json,
                    "JSONL": args.output_jsonl,
                },
            )
        else:
            print(
                f"materialized: task={materialized.task_type} "
                f"benchmark={materialized.benchmark_name} requests={materialized.sample_count}"
            )
            if args.output_json:
                print(f"wrote_json: {args.output_json}")
            if args.output_jsonl:
                print(f"wrote_jsonl: {args.output_jsonl}")
    return 0


# ── Parser registration ─────────────────────────────────────────


def register_task_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register task commands and keep related argument groups contiguous."""
    tasks_parser = subparsers.add_parser("tasks", help="Browse benchmark tasks and evaluation suites")
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)

    tasks_list_parser = tasks_subparsers.add_parser("list", help="List WorldFoundry benchmark entries")
    tasks_list_parser.add_argument("--task-type")
    tasks_list_parser.add_argument("--suite")
    tasks_list_parser.add_argument("--backend")
    tasks_list_parser.add_argument("--source-kind", choices=["task_yaml", "benchmark_zoo"])
    tasks_list_parser.add_argument("--flat", action="store_true", help="Use the compact one-line task listing.")
    tasks_list_parser.add_argument("--json", action="store_true")
    tasks_list_parser.set_defaults(func=_handle_tasks_list)

    tasks_show_parser = tasks_subparsers.add_parser("show", help="Show one WorldFoundry benchmark definition")
    _add_task_selector(tasks_show_parser, require_data_path=False)
    tasks_show_parser.add_argument("--json", action="store_true")
    tasks_show_parser.set_defaults(func=_handle_tasks_show)

    tasks_catalog_parser = tasks_subparsers.add_parser("catalog", help="Export active task registry entries as BenchmarkSpec objects")
    tasks_catalog_parser.add_argument("--task-type")
    tasks_catalog_parser.add_argument("--suite")
    tasks_catalog_parser.add_argument("--backend")
    tasks_catalog_parser.add_argument("--source-kind", choices=["task_yaml", "benchmark_zoo"])
    tasks_catalog_parser.add_argument("--json", action="store_true")
    tasks_catalog_parser.set_defaults(func=_handle_tasks_catalog)

    # Filesystem task catalog commands.
    suites_parser = subparsers.add_parser("suites", help="Inspect model x benchmark suite presets")
    suites_subparsers = suites_parser.add_subparsers(dest="suites_command", required=True)

    suites_list_parser = suites_subparsers.add_parser("list", help="List named model-benchmark suite presets")
    suites_list_parser.add_argument("--suite-preset-path", type=Path)
    suites_list_parser.add_argument("--json", action="store_true")
    suites_list_parser.set_defaults(func=_handle_suites_list)

    suites_show_parser = suites_subparsers.add_parser("show", help="Show one model-benchmark suite preset")
    suites_show_parser.add_argument("suite")
    suites_show_parser.add_argument("--suite-preset-path", type=Path)
    suites_show_parser.add_argument("--json", action="store_true")
    suites_show_parser.set_defaults(func=_handle_suites_show)

    task_parser = subparsers.add_parser("task", help="Inspect and validate filesystem task YAML catalog entries")
    task_subparsers = task_parser.add_subparsers(dest="task_command", required=True)

    task_list_parser = task_subparsers.add_parser("list", help="List task YAML entries from a filesystem catalog")
    task_list_parser.add_argument("--task-root", action="append", type=Path)
    task_list_parser.add_argument("--include-path", "--benchmark-include-path", action="append", type=Path, dest="include_path")
    task_list_parser.add_argument("--root-dir", type=Path)
    task_list_parser.add_argument("--recursive", action="store_true")
    task_list_parser.add_argument("--benchmark")
    task_list_parser.add_argument("--tag")
    task_list_parser.add_argument("--protocol")
    task_list_parser.add_argument("--json", action="store_true")
    task_list_parser.set_defaults(func=_handle_task_list)

    task_show_parser = task_subparsers.add_parser("show", help="Show one task YAML entry by task name")
    task_show_parser.add_argument("task_name")
    task_show_parser.add_argument("--task-root", action="append", type=Path)
    task_show_parser.add_argument("--include-path", "--benchmark-include-path", action="append", type=Path, dest="include_path")
    task_show_parser.add_argument("--root-dir", type=Path)
    task_show_parser.add_argument("--recursive", action="store_true")
    task_show_parser.add_argument("--benchmark")
    task_show_parser.add_argument("--json", action="store_true")
    task_show_parser.set_defaults(func=_handle_task_show)

    task_validate_parser = task_subparsers.add_parser("validate", help="Validate task YAML files or directories without loading runtime backends")
    task_validate_parser.add_argument("path", nargs="*", type=Path)
    task_validate_parser.add_argument("--task-root", action="append", type=Path)
    task_validate_parser.add_argument("--include-path", "--benchmark-include-path", action="append", type=Path, dest="include_path")
    task_validate_parser.add_argument("--root-dir", type=Path)
    task_validate_parser.add_argument("--recursive", action="store_true")
    task_validate_parser.add_argument("--json", action="store_true")
    task_validate_parser.set_defaults(func=_handle_task_validate)

    task_materialize_parser = task_subparsers.add_parser(
        "materialize",
        help="Materialize GenerationRequest rows from a task YAML entry",
    )
    task_materialize_parser.add_argument("task_name")
    task_materialize_parser.add_argument("--task-root", action="append", type=Path)
    task_materialize_parser.add_argument("--include-path", "--benchmark-include-path", action="append", type=Path, dest="include_path")
    task_materialize_parser.add_argument("--root-dir", type=Path)
    task_materialize_parser.add_argument("--recursive", action="store_true")
    task_materialize_parser.add_argument("--benchmark")
    task_materialize_parser.add_argument("--dataset-root", type=Path)
    task_materialize_parser.add_argument("--dataset-manifest", type=Path)
    task_materialize_parser.add_argument("--split", default="default")
    task_materialize_parser.add_argument("--num-samples", type=int)
    task_materialize_parser.add_argument("--output-dir", type=Path, default=Path("tmp/worldfoundry_task_materialize"))
    task_materialize_parser.add_argument("--output-json", type=Path)
    task_materialize_parser.add_argument("--output-jsonl", type=Path)
    task_materialize_parser.add_argument("--json", action="store_true")
    task_materialize_parser.set_defaults(func=_handle_task_materialize)
