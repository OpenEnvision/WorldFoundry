"""Top-level CLI composition for WorldFoundry command execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from worldfoundry.evaluation.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_ZOO_DIR,
    REPO_ROOT,
    TMP_ROOT,
)

from .help import WorldFoundryArgumentParser
from .presentation import (
    BOLD,
    MUTED,
    Panel,
    Row,
    paint,
    print_details,
    print_hint,
    print_notice,
    print_table,
    render_panels,
    terminal_enabled,
    terminal_width,
)
from .utils import CliUsageError, cli_prog_name, json_dump, load_json_mapping, parse_key_value_mapping

_BENCHMARK_RUN_MODE_CHOICES = ("normalizer", "official-run", "official-validation", "contract")
_EMBODIED_ACCEPTED_LICENSES_ENV = "WORLDFOUNDRY_ACCEPTED_LICENSES"

# ── CLI banners and public command surface ──────────────────────

_FIRST_RUN_BANNER = """\
WorldFoundry evaluation CLI

Start with GPU-ready discovery and validation commands:
  {prog} zoo benchmarks --json
  {prog} zoo models
  {prog} tasks list

Then run a selected model x benchmark cell:
  {prog} run --benchmark <id> --model <model-id> --output-dir tmp/worldfoundry_run --json

Interactive and help:
  {prog} tui
  {prog} <command> --help
"""

_PUBLIC_ROOT_COMMANDS = (
    "tui",
    "tasks",
    "suites",
    "task",
    "dataset",
    "config",
    "plan",
    "metric",
    "compare-runs",
    "index-runs",
    "validate-artifact",
    "preflight",
    "models",
    "zoo",
    "mcp",
    "evaluate",
    "score",
    "generate-score",
    "reproduce",
    "embodied",
    "validate",
    "run",
)


def _print_first_run_banner() -> None:
    """Print the first-run discovery banner when no subcommand is selected."""
    if terminal_enabled():
        prog = cli_prog_name()
        print(paint("WorldFoundry", BOLD))
        print(paint("World models · Inference · Evaluation", MUTED))
        print()
        panels = (
            Panel(
                "Explore",
                (
                    Row(f"{prog} zoo models", "Browse models and their requirements."),
                    Row(f"{prog} zoo benchmarks", "Find benchmarks and evaluation surfaces."),
                    Row(f"{prog} tui", "Open the interactive terminal workspace."),
                ),
            ),
            Panel(
                "Run and evaluate",
                (
                    Row(f"{prog} run <model> --help", "Inspect typed inputs, pipeline settings and defaults."),
                    Row(
                        f"{prog} run <model> --prompt 'A quiet coastal town'",
                        "Generate with a configured model and its checkpoints.",
                    ),
                    Row(f"{prog} score --help", "Score generated artifacts or existing results."),
                ),
            ),
        )
        print(render_panels(panels, terminal_width(), min_column_width=48))
        print_hint(f"Workflow templates: {prog} config list\nAll commands: {prog} --help")
        return
    print(_FIRST_RUN_BANNER.format(prog=cli_prog_name()).rstrip())


# ── Module and data loading helpers ─────────────────────────────


def _curate_root_subparser_help(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Keep the root help focused on the supported public command surface."""

    subparsers.metavar = "{" + ",".join(_PUBLIC_ROOT_COMMANDS) + "}"
    subparsers.help_groups = {
        "Start here": ("run", "tui", "zoo", "models"),
        "Evaluate": ("evaluate", "score", "generate-score", "reproduce", "embodied"),
        "Catalogs and workflows": ("tasks", "suites", "task", "dataset", "config", "plan", "metric"),
        "Results and validation": ("compare-runs", "index-runs", "validate-artifact", "validate", "preflight"),
        "Services": ("mcp",),
    }

    def describe_children(action: argparse._SubParsersAction) -> None:
        for command in action._get_subactions():
            child = action.choices[command.dest]
            if not child.description and command.help != argparse.SUPPRESS:
                child.description = command.help
        for child in set(action.choices.values()):
            for nested in child._actions:
                if isinstance(nested, argparse._SubParsersAction):
                    describe_children(nested)

    describe_children(subparsers)


def _load_repo_script(relative_path: str) -> ModuleType:
    """Dynamically import a repo script as a module by its relative path.

    Args:
        relative_path: Path relative to ``REPO_ROOT`` pointing to a Python script.

    Raises:
        ImportError: If the script cannot be loaded.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"_worldfoundry_cli_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json_or_jsonl(path: Path | None):
    """Load a JSON or JSONL file, returning ``None`` when *path* is ``None``."""
    if path is None:
        return None
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def _samples_or_episodes_from_payload(payload, *, field_name: str) -> list:
    """Normalize a JSON payload into a flat list of samples or episodes.

    Handles dicts with ``samples``/``episodes`` keys, inline lists, and
    dicts keyed by sample id.

    Args:
        payload: Raw JSON structure (``None``, list, or dict).
        field_name: Label used in error messages.

    Raises:
        ValueError: If the payload structure is not recognized.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        samples = payload.get("samples", payload.get("episodes", []))
        if isinstance(samples, dict):
            return [
                {"sample_id": sample_id, **item}
                if isinstance(item, dict)
                else {"sample_id": sample_id, "value": item}
                for sample_id, item in samples.items()
            ]
        if isinstance(samples, list):
            return samples
        raise ValueError(f"{field_name} object must contain a samples or episodes list")
    raise ValueError(f"{field_name} must contain a JSON object/list or JSONL rows")


def _load_samples_or_episodes(path: Path | None, *, field_name: str) -> list:
    """Load samples or episodes from a JSON/JSONL path."""
    return _samples_or_episodes_from_payload(_load_json_or_jsonl(path), field_name=field_name)


def _load_json_mapping_or_inline(value: str | Path | None, *, field_name: str) -> dict | None:
    """Load a JSON object from a path or an inline JSON string.

    Args:
        value: Path to a JSON file or inline JSON text; ``None`` returns ``None``.
        field_name: Label used in error messages.

    Raises:
        ValueError: If the resolved payload is not a JSON object.
    """
    if value is None:
        return None
    text = str(value)
    path = Path(text)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload


# ── Argument-group helpers ──────────────────────────────────────


def _add_task_selector(parser: argparse.ArgumentParser, *, require_data_path: bool) -> None:
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--benchmark-name", required=True)
    if require_data_path:
        parser.add_argument("--data-path", required=True)


def _add_generation_cache_args(parser: argparse.ArgumentParser, *, namespace: str) -> None:
    """Add generation-cache ``--generation-cache-*`` arguments to a parser."""
    parser.add_argument(
        "--generation-cache-dir",
        type=Path,
        help="Optional SQLite generation-result cache directory for deterministic model outputs.",
    )
    parser.add_argument(
        "--generation-cache-mode",
        choices=["off", "read", "write", "read-write", "refresh"],
        default="off",
        help="Generation cache mode. read-write reuses hits and stores new successful outputs.",
    )
    parser.add_argument(
        "--generation-cache-namespace",
        default=namespace,
        help="Namespace inside the generation-result cache database.",
    )


# ── Dataset command bridges ──────────────────────────────────────


def _handle_dataset_create(args: argparse.Namespace) -> int:
    from .dataset import _handle_dataset_create as handle_dataset_create

    return handle_dataset_create(args)


def _handle_dataset_show(args: argparse.Namespace) -> int:
    from .dataset import _handle_dataset_show as handle_dataset_show

    return handle_dataset_show(args)


def _handle_dataset_validate(args: argparse.Namespace) -> int:
    from .dataset import _handle_dataset_validate as handle_dataset_validate

    return handle_dataset_validate(args)


def _handle_dataset_materialize(args: argparse.Namespace) -> int:
    from .dataset import _handle_dataset_materialize as handle_dataset_materialize

    return handle_dataset_materialize(args)


# ── TUI bridges ─────────────────────────────────────────────────


def _execute_run_plan_file(args: argparse.Namespace):
    """Load and execute a run-plan JSON file through the evaluation runner."""
    from worldfoundry.evaluation.runner import evaluate_request_from_run_plan, execute_evaluate_run, load_run_plan

    plan = load_run_plan(args.plan)
    return execute_evaluate_run(evaluate_request_from_run_plan(plan))


def _handle_tui(args: argparse.Namespace) -> int:
    """Translate root CLI arguments into the TUI command namespace."""
    from worldfoundry.cli.tui import main as tui_main

    # TUI accepts the same high-level options as the CLI entrypoint; this bridge
    # normalizes generation flags into one argument list.
    argv: list[str] = []
    if args.model_manifest_dir is not None:
        argv.extend(["--model-manifest-dir", str(args.model_manifest_dir)])
    if args.benchmark_manifest_dir is not None:
        argv.extend(["--benchmark-manifest-dir", str(args.benchmark_manifest_dir)])
    if args.runtime_profile_dir is not None:
        argv.extend(["--runtime-profile-dir", str(args.runtime_profile_dir)])
    for suite_id in args.suite_ids or ():
        argv.extend(["--suite", str(suite_id)])
    if args.model_id is not None:
        argv.extend(["--model-id", str(args.model_id)])
    if args.benchmark_id is not None:
        argv.extend(["--benchmark-id", str(args.benchmark_id)])
    for metric in getattr(args, "metric", None) or ():
        argv.extend(["--metric", str(metric)])
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if getattr(args, "input", None) is not None:
        argv.extend(["--input", str(args.input)])
    if getattr(args, "input_dir", None) is not None:
        argv.extend(["--input-dir", str(args.input_dir)])
    if getattr(args, "video", None) is not None:
        argv.extend(["--video", str(args.video)])
    if getattr(args, "trajectory_file", None) is not None:
        argv.extend(["--trajectory-file", str(args.trajectory_file)])
    for attr, flag in (
        ("prompt", "--prompt"),
        ("negative_prompt", "--negative-prompt"),
        ("task", "--task"),
        ("mode", "--mode"),
        ("resize_mode", "--resize-mode"),
        ("size", "--size"),
        ("frames", "--frames"),
        ("steps", "--steps"),
        ("frames_per_generation", "--frames-per-generation"),
        ("guidance_scale", "--guidance-scale"),
        ("seed", "--seed"),
        ("fps", "--fps"),
        ("dtype", "--dtype"),
        ("max_sequence_length", "--max-sequence-length"),
        ("cam_type", "--cam-type"),
        ("interactions", "--interactions"),
        ("output_formats", "--output-formats"),
        ("trajectory", "--trajectory"),
        ("angle", "--angle"),
        ("distance", "--distance"),
        ("orbit_radius", "--orbit-radius"),
        ("zoom_ratio", "--zoom-ratio"),
        ("alpha_threshold", "--alpha-threshold"),
    ):
        value = getattr(args, attr, None)
        if value:
            argv.extend([flag, str(value)])
    for attr, flag in (
        ("static_scene", "--static-scene"),
        ("low_vram", "--low-vram"),
        ("disable_lora", "--disable-lora"),
        ("vis_rendering", "--vis-rendering"),
        ("offload_t5", "--offload-t5"),
        ("offload_transformer_during_vae", "--offload-transformer-during-vae"),
        ("offload_vae", "--offload-vae"),
    ):
        if getattr(args, attr, False):
            argv.append(flag)
    if getattr(args, "output_path", None) is not None:
        argv.extend(["--output-path", str(args.output_path)])
    if getattr(args, "ckpt_type", None):
        argv.extend(["--ckpt-type", str(args.ckpt_type)])
    if getattr(args, "ckpt_root", None) is not None:
        argv.extend(["--ckpt-root", str(args.ckpt_root)])
    if getattr(args, "ckpt_path", None) is not None:
        argv.extend(["--ckpt-path", str(args.ckpt_path)])
    if getattr(args, "conda_envs_root", None) is not None:
        argv.extend(["--conda-envs-root", str(args.conda_envs_root)])
    if getattr(args, "gpu", None):
        argv.extend(["--gpu", str(args.gpu)])
    if args.fallback:
        argv.append("--fallback")
    if args.catalog_json:
        argv.append("--catalog-json")
    if args.print_command:
        argv.append("--print-command")
    if getattr(args, "write_suite_plan", False):
        argv.append("--write-suite-plan")
    return tui_main(argv)


# ── Evaluate command ─────────────────────────────────────────────


def _validate_evaluate_usage(args: argparse.Namespace) -> None:
    """Raise :class:`CliUsageError` for mutually exclusive evaluate flags (CM-08)."""

    task_args = (args.task_type, args.benchmark_name, args.data_path)
    if args.embodied_spec is not None and any(item is not None for item in task_args):
        raise CliUsageError(
            "--embodied-spec cannot be combined with --task-type/--benchmark-name/--data-path"
        )
    if args.samples_path is not None and args.embodied_spec is None:
        raise CliUsageError("--samples-path requires --embodied-spec")
    if args.embodied_spec is not None and args.samples_path is not None and args.requests_path is not None:
        raise CliUsageError("use either --samples-path or --requests-path with --embodied-spec, not both")
    if any(item is not None for item in task_args) and not all(item is not None for item in task_args):
        raise CliUsageError("--task-type, --benchmark-name, and --data-path must be provided together")


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Route evaluate workloads to either cached run-plan replay or runtime eval."""
    if args.plan is None:
        _validate_evaluate_usage(args)
    from worldfoundry.evaluation.runner import EvaluateRunRequest, execute_evaluate_run

    if args.plan is not None:
        result = _execute_run_plan_file(args)
        payload = result.to_dict()
        if args.json:
            json_dump(payload)
        else:
            print_details(
                "Evaluation result",
                {
                    "status": result.status,
                    "mode": result.mode,
                    "samples": result.sample_count,
                    "scorecard": result.scorecard_path,
                },
                plain=f"Evaluate {result.status}: mode={result.mode}, samples={result.sample_count}, scorecard={result.scorecard_path}",
            )
        return result.exit_code

    requests = None
    benchmark_metadata = None
    dataset_metadata = None
    metric_objects = None
    model_parameters = parse_key_value_mapping(args.model_parameter)
    model_runtime = parse_key_value_mapping(args.model_runtime)
    model_id = args.model_id
    model_runner = args.model_runner
    if args.embodied_spec is not None:
        from worldfoundry.evaluation.tasks.embodied.contracts import EmbodiedGenerationSpec
        from worldfoundry.evaluation.tasks.embodied.materialize import materialize_vla_va_wam_requests
        from worldfoundry.evaluation.tasks.embodied.metrics import metric_suite

        spec_payload = _load_json_mapping_or_inline(args.embodied_spec, field_name="--embodied-spec")
        if spec_payload is None:
            raise CliUsageError("--embodied-spec is required")
        embodied_spec = EmbodiedGenerationSpec.from_dict(spec_payload)
        if args.requests_path is None:
            samples = _load_samples_or_episodes(args.samples_path, field_name="--samples-path")
            if args.num_samples is not None:
                samples = samples[: args.num_samples]
            materialized = materialize_vla_va_wam_requests(
                samples or [{"sample_id": "sample-000000"}],
                spec=embodied_spec,
                split=args.split,
            )
            requests = list(materialized.requests)
        benchmark_metadata = {
            "suite": "vla_va_wam",
            "benchmark_name": args.benchmark_id or embodied_spec.task_name,
            "benchmark_id": args.benchmark_id or embodied_spec.task_name,
            "task_type": embodied_spec.task_name,
            "evaluation_protocol": "worldfoundry_evaluate_model",
            "track": embodied_spec.track.value,
            "request_kind": embodied_spec.kind.value,
            "action_space": embodied_spec.action_space.to_dict(),
            "required_capabilities": list(embodied_spec.required_capabilities),
        }
        dataset_metadata = {
            "dataset_id": args.dataset_id or "vla_va_wam_materialized",
            "name": args.dataset_id or "vla_va_wam_materialized",
            "split": args.split,
        }
        if requests is not None:
            dataset_metadata["sample_count"] = len(requests)
        metric_objects = metric_suite(tuple(args.metric or ()), track=embodied_spec.track.value)
        model_parameters = {
            "track": embodied_spec.track.value,
            "metadata_namespace": "vla_va_wam",
            "capabilities": list(embodied_spec.required_capabilities),
            **model_parameters,
        }
    if args.task_type and args.benchmark_name and args.data_path:
        from worldfoundry.cli.utils import resolve_cli_benchmark_for_materialize
        from worldfoundry.evaluation.runner import materialize_requests_from_benchmark

        benchmark = resolve_cli_benchmark_for_materialize(args.task_type, args.benchmark_name)
        materialized = materialize_requests_from_benchmark(
            benchmark,
            args.data_path,
            limit=args.num_samples,
            split=args.split,
        )
        requests = list(materialized.requests)
        benchmark_metadata = {
            "suite": benchmark.suite,
            "benchmark_name": benchmark.benchmark_name,
            "task_type": benchmark.task_type,
            "backend": benchmark.backend,
            "evaluation_protocol": benchmark.evaluation_protocol,
        }
        dataset_metadata = {
            "root": str(Path(args.data_path).resolve()),
            "split": args.split,
            "sample_count": materialized.sample_count,
        }

    evaluate_mode = args.mode
    if args.embodied_spec is not None and args.results_path is None:
        evaluate_mode = "model"

    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=args.output_dir,
            mode=evaluate_mode,
            requests=requests,
            requests_path=args.requests_path,
            results_path=args.results_path,
            metrics=metric_objects if metric_objects is not None else tuple(args.metric or ("artifact_count",)),
            required_artifacts=tuple(args.required_artifact or ()),
            benchmark=benchmark_metadata,
            dataset=dataset_metadata,
            benchmark_id=args.benchmark_id or args.benchmark_name,
            model_id=model_id,
            model_runner=model_runner,
            model_zoo_manifest_dir=args.model_manifest_dir,
            model_variant_id=args.model_variant,
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            model_config=load_json_mapping(args.model_config),
            dataset_id=args.dataset_id,
            run_id=args.run_id,
            fail_on_sample_error=args.fail_on_sample_error,
            write_artifacts_index=not args.no_artifacts_index,
            generation_cache_dir=args.generation_cache_dir,
            generation_cache_mode=args.generation_cache_mode,
            generation_cache_namespace=args.generation_cache_namespace,
        )
    )
    payload = result.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print_details(
            "Evaluation result",
            {
                "status": result.status,
                "mode": result.mode,
                "samples": result.sample_count,
                "scorecard": result.scorecard_path,
            },
            plain=f"Evaluate {result.status}: mode={result.mode}, samples={result.sample_count}, scorecard={result.scorecard_path}",
        )
    return result.exit_code


def _accept_embodied_licenses(license_ids: list[str] | None) -> None:
    """Merge explicit CLI acknowledgements into the embodied license environment."""

    accepted = [
        item.strip()
        for item in os.environ.get(_EMBODIED_ACCEPTED_LICENSES_ENV, "").split(",")
        if item.strip()
    ]
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
    from worldfoundry.evaluation.tasks.embodied.orchestrator import run_embodied_eval_config

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
    from worldfoundry.evaluation.tasks.embodied.model_server.serve import serve_from_config

    serve_from_config(args.config, host=args.host, port=args.port)
    return 0


def _handle_embodied_merge(args: argparse.Namespace) -> int:
    """Merge embodied shard outputs into one scorecard."""
    from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
    from worldfoundry.evaluation.tasks.embodied.merge_results import merge_embodied_results

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
    from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
    from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests

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


# ── Validate command ─────────────────────────────────────────────


def _handle_validate(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.catalog.specs import (
        get_benchmark_zoo_cli_task,
        validate_benchmark_zoo_cli_task,
    )

    item = get_benchmark_zoo_cli_task(args.task_type, args.benchmark_name)
    payload = validate_benchmark_zoo_cli_task(item, dataset_root=args.data_path, limit=args.num_samples)
    if args.json:
        json_dump(payload)
        return 0

    print_details(
        "Task validation",
        {
            "task": payload["task_type"],
            "benchmark": payload["benchmark_name"],
            "protocol": payload["evaluation_protocol"],
        },
        plain=(
            f"Validated {payload['task_type']}/{payload['benchmark_name']}: protocol={payload['evaluation_protocol']}"
        ),
    )
    return 0


# ── Run routing helpers ──────────────────────────────────────────


def _has_complete_task_args(args: argparse.Namespace) -> bool:
    """Check whether the required task-type/benchmark-name/data-path triplet is present."""
    return all(item is not None for item in (args.task_type, args.benchmark_name, args.data_path))


def _run_uses_unified_framework(args: argparse.Namespace) -> bool:
    """Decide whether the run args route to the unified WorldFoundry framework."""
    if args.all_benchmarks or args.suite_ids or args.benchmark_ids:
        return True
    if args.benchmark_id and not _has_complete_task_args(args):
        return True
    return args.results_path is not None and not _has_complete_task_args(args)


def _run_model_ids(args: argparse.Namespace) -> tuple[str, ...]:
    """Collect positional, repeated, and singular model ids without duplicates."""
    schema = getattr(args, "model_run_schema", None)
    positional_model = getattr(schema, "model_id", None) or getattr(args, "model_slug", None)
    values = (
        *((positional_model,) if positional_model else ()),
        *(args.model_ids or ()),
        *((args.model_id,) if args.model_id else ()),
    )
    return tuple(dict.fromkeys(str(value) for value in values if value))


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested CLI configuration mappings."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mapping(current, value)
        else:
            merged[key] = value
    return merged


def _resolved_model_run_options(args: argparse.Namespace):
    from .model_run import resolve_model_run_options

    return resolve_model_run_options(args)


def _run_model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Merge legacy KEY=VALUE parameters with typed model-specific load options."""
    from worldfoundry.evaluation.models.pipelines.loading import GENERATION_DEFAULTS_PARAMETER

    resolution = _resolved_model_run_options(args)
    parameters = _deep_merge_mapping(
        parse_key_value_mapping(args.model_parameter),
        resolution.model_parameters,
    )
    if resolution.generation_defaults:
        existing = parameters.get(GENERATION_DEFAULTS_PARAMETER)
        parameters[GENERATION_DEFAULTS_PARAMETER] = _deep_merge_mapping(
            existing if isinstance(existing, Mapping) else {},
            resolution.generation_defaults,
        )
    return parameters


def _run_model_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Merge legacy KEY=VALUE runtime settings with typed runtime options."""
    resolution = _resolved_model_run_options(args)
    return _deep_merge_mapping(
        parse_key_value_mapping(args.model_runtime),
        resolution.model_runtime,
    )


def _run_benchmark_ids(args: argparse.Namespace) -> tuple[str, ...]:
    """Collect benchmark ids from both ``--benchmark`` and singular ``--benchmark-id``."""
    return tuple(args.benchmark_ids or ()) + (((args.benchmark_id,) if args.benchmark_id else ()))


def _suite_presets_declare_models(args: argparse.Namespace) -> bool:
    """Check whether selected suite presets declare their own model ids."""
    suite_ids = tuple(args.suite_ids or ())
    if not suite_ids:
        return False
    try:
        from worldfoundry.evaluation.runner import get_model_benchmark_suite_preset

        return all(
            bool(get_model_benchmark_suite_preset(suite_id, args.suite_preset_path).get("model_ids"))
            for suite_id in suite_ids
        )
    except Exception:  # noqa: BLE001 - keep CLI validation conservative for missing/invalid presets.
        return False


def _run_output_artifact(args: argparse.Namespace) -> str | None:
    """Resolve the output artifact name from ``--output-artifact`` or ``--required-artifact``."""
    if args.output_artifact:
        return args.output_artifact
    if args.required_artifact:
        return args.required_artifact[0]
    return None


def _worldfoundry_run_request_from_args(args: argparse.Namespace):
    """Build a ``WorldFoundryRunRequest`` from CLI args for the unified facade."""
    from worldfoundry.evaluation.framework import WorldFoundryRunRequest

    return WorldFoundryRunRequest(
        output_dir=args.output_dir,
        model_ids=_run_model_ids(args),
        benchmark_ids=_run_benchmark_ids(args),
        suite_ids=tuple(args.suite_ids or ()),
        all_benchmarks=args.all_benchmarks,
        benchmark_id=args.benchmark_id,
        benchmark_manifest_dir=args.benchmark_manifest_dir,
        model_manifest_dir=args.model_manifest_dir or MODEL_ZOO_DIR,
        suite_preset_path=args.suite_preset_path,
        engine=args.engine,
        benchmark_mode=getattr(args, "benchmark_mode", "official-run"),
        execute=not args.plan_only,
        model_workers=getattr(args, "model_workers", 1),
        worker_cuda_devices=tuple(getattr(args, "model_worker_device", None) or ()),
        resume=args.resume,
        skip_incompatible=args.skip_incompatible,
        fail_on_skipped=args.fail_on_skipped,
        model_runner=args.model_runner,
        model_variant_id=args.model_variant,
        model_parameters=_run_model_parameters(args),
        model_runtime=_run_model_runtime(args),
        model_config=load_json_mapping(args.model_config),
        requests_path=args.requests_path,
        results_path=args.results_path,
        task_name=args.task_name,
        task_roots=tuple(args.task_root or ()) or None,
        task_benchmark=args.task_benchmark,
        task_recursive=args.task_recursive,
        task_root_dir=args.task_root_dir,
        dataset_root=args.data_path,
        dataset_id=args.dataset_id,
        split=args.split,
        num_samples=args.num_samples,
        generated_artifact_dir=args.generated_artifact_dir,
        output_artifact=_run_output_artifact(args),
        required_artifacts=tuple(args.required_artifact) if args.required_artifact is not None else None,
        metrics=tuple(args.metric) if args.metric is not None else ("artifact_count", "required_artifacts_present"),
        generation_cache_dir=args.generation_cache_dir,
        generation_cache_mode=args.generation_cache_mode,
        generation_cache_namespace=args.generation_cache_namespace,
        benchmark_timeout_seconds=args.timeout,
        benchmark_workdir=args.workdir,
        benchmark_env=parse_key_value_mapping(args.env),
        benchmark_parameters=parse_key_value_mapping(args.benchmark_parameter),
        materialize_placeholders=args.materialize_placeholders,
        contract_fixture=getattr(args, "contract_fixture", False),
        fail_on_generation_error=args.fail_on_generation_error,
        run_id=args.run_id,
        fail_on_sample_error=args.fail_on_sample_error,
        write_artifacts_index=not args.no_artifacts_index,
    )


def _handle_worldfoundry_run(args: argparse.Namespace) -> int:
    """Run the unified worldfoundry facade and return the framework exit code."""
    from worldfoundry.evaluation.framework import run_worldfoundry

    performance_profile = getattr(args, "performance_profile", "balanced")
    if performance_profile == "throughput":
        os.environ.setdefault("WORLDFOUNDRY_EVAL_TORCH_COMPILE", "1")
        os.environ.setdefault("WORLDFOUNDRY_KERNEL_AUTOTUNE", "1")
        os.environ.setdefault("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", "1")
        os.environ.setdefault("WORLDFOUNDRY_TORCH_COMPILE_MODE", "reduce-overhead")
        os.environ.setdefault("WORLDFOUNDRY_MATMUL_PRECISION", "high")
        os.environ.setdefault("WORLDFOUNDRY_ENABLE_TF32", "1")
    elif performance_profile == "latency":
        os.environ.setdefault("WORLDFOUNDRY_EVAL_TORCH_COMPILE", "0")
        os.environ.setdefault("WORLDFOUNDRY_KERNEL_AUTOTUNE", "0")

    result = run_worldfoundry(_worldfoundry_run_request_from_args(args))
    payload = result.to_dict()
    payload["engine"] = args.engine
    payload["performance_profile"] = performance_profile
    if args.json:
        json_dump(payload)
    else:
        print_details(
            "Run result",
            {"status": result.status, "kind": result.kind, "output": result.output_dir},
            plain=f"Run {result.status}: kind={result.kind}, output_dir={result.output_dir}",
        )
    return result.exit_code


def _model_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a side-effect-free report of the resolved model-specific configuration."""
    from worldfoundry.evaluation.models.pipelines.loading import GENERATION_DEFAULTS_PARAMETER

    schema = getattr(args, "model_run_schema", None)
    if schema is None:
        raise CliUsageError("model-specific configuration requires `run <model-id>`")
    resolution = _resolved_model_run_options(args)
    parameters = _run_model_parameters(args)
    generation_defaults = parameters.pop(GENERATION_DEFAULTS_PARAMETER, {})
    resolved = resolution.to_dict()
    resolved["model_parameters"] = parameters
    resolved["model_runtime"] = _run_model_runtime(args)
    resolved["generation_defaults"] = generation_defaults
    return {
        "schema_version": "worldfoundry-model-run-config-v1",
        "model": schema.to_dict(),
        "resolved": resolved,
        "execution": {
            "output_dir": str(args.output_dir),
            "requests_path": None if args.requests_path is None else str(args.requests_path),
            "plan_only": bool(args.plan_only),
        },
    }


def _print_model_run_plan(args: argparse.Namespace) -> int:
    payload = _model_run_plan(args)
    if args.json:
        json_dump(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _uses_direct_model_run(args: argparse.Namespace) -> bool:
    """Return whether positional model syntax should execute direct inference."""
    if getattr(args, "model_run_schema", None) is None:
        return False
    return not any(
        (
            args.plan is not None,
            args.all_benchmarks,
            bool(args.suite_ids),
            bool(_run_benchmark_ids(args)),
            args.results_path is not None,
            args.generated_artifact_dir is not None,
            _has_complete_task_args(args),
        )
    )


def _handle_direct_model_run(args: argparse.Namespace) -> int:
    """Execute one model directly from its typed inference contract."""
    schema = args.model_run_schema
    if not schema.runnable:
        raise CliUsageError(
            f"model {schema.model_id!r} is not runnable: {schema.blocked_reason}. "
            f"Use `{cli_prog_name()} run {schema.requested_model_id} --model-status` for its contract.",
        )
    resolution = _resolved_model_run_options(args)
    missing = [
        field.option
        for field in schema.fields
        if field.scope == "input"
        and field.required
        and not resolution.inputs.get(field.input_key or field.key_path[-1])
    ]
    missing.extend(
        field.option
        for field in schema.fields
        if field.scope == "call"
        and field.required
        and getattr(args, field.dest, None) in (None, "")
    )
    if missing and args.requests_path is None:
        raise CliUsageError(f"direct {schema.model_id} inference requires {', '.join(missing)}")

    from worldfoundry.evaluation.api import GenerationRequest
    from worldfoundry.evaluation.runner import EvaluateRunRequest, execute_evaluate_run

    requests = None
    if args.requests_path is None:
        requests = [
            GenerationRequest(
                sample_id="sample-0000",
                task_name=resolution.task_id or schema.task_id,
                split=args.split,
                inputs=resolution.inputs,
                generation_kwargs=resolution.generation_defaults,
            )
        ]
    runtime = _run_model_runtime(args)
    runtime.setdefault("output_dir", str(Path(args.output_dir) / "generated"))
    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=args.output_dir,
            mode="model",
            requests=requests,
            requests_path=args.requests_path,
            metrics=tuple(args.metric) if args.metric is not None else ("artifact_count",),
            required_artifacts=tuple(args.required_artifact or ()),
            benchmark={
                "suite": "model_direct",
                "benchmark_name": resolution.task_id or schema.task_id,
                "task_type": resolution.task_id or schema.task_id,
                "evaluation_protocol": "direct_model_inference",
            },
            model_id=schema.model_id,
            model_runner=args.model_runner,
            model_zoo_manifest_dir=args.model_manifest_dir or MODEL_ZOO_DIR,
            model_variant_id=args.model_variant or schema.catalog_variant_id,
            model_parameters=_run_model_parameters(args),
            model_runtime=runtime,
            model_config=load_json_mapping(args.model_config),
            dataset_id=args.dataset_id or f"{schema.model_id}:direct",
            run_id=args.run_id,
            fail_on_sample_error=bool(args.fail_on_sample_error or args.fail_on_generation_error),
            write_artifacts_index=not args.no_artifacts_index,
            generation_cache_dir=args.generation_cache_dir,
            generation_cache_mode=args.generation_cache_mode,
            generation_cache_namespace=args.generation_cache_namespace,
        )
    )
    payload = result.to_dict()
    payload["engine"] = "model-direct"
    payload["model_cli"] = {
        "model_id": schema.model_id,
        "variant_id": schema.variant_id,
        "task_id": resolution.task_id,
    }
    if args.json:
        json_dump(payload)
    else:
        print_details(
            "Model run",
            {
                "status": result.status,
                "model": schema.model_id,
                "samples": result.sample_count,
                "scorecard": result.scorecard_path,
            },
            plain=f"Run {result.status}: model={schema.model_id}, samples={result.sample_count}, scorecard={result.scorecard_path}",
        )
    return result.exit_code


def _handle_run(args: argparse.Namespace) -> int:
    """Route run into either plan replay or unified/in-process execution."""
    if getattr(args, "model_run_schema", None) is not None and len(_run_model_ids(args)) > 1:
        raise CliUsageError("positional model syntax accepts one model; remove conflicting --model/--model-id values")
    schema = getattr(args, "model_run_schema", None)
    if schema is not None and not schema.runnable and not args.print_config and not args.plan_only:
        raise CliUsageError(
            f"model {schema.model_id!r} is not runnable: {schema.blocked_reason}. "
            f"Use `{cli_prog_name()} run {schema.requested_model_id} --model-status` for its contract.",
        )
    if args.plan is not None:
        result = _execute_run_plan_file(args)
        payload = result.to_dict()
        payload["engine"] = "plan"
        if args.json:
            json_dump(payload)
        else:
            print_details(
                "Run result",
                {
                    "status": result.status,
                    "engine": "plan",
                    "samples": result.sample_count,
                    "scorecard": result.scorecard_path,
                },
                plain=f"Run {result.status}: engine=plan, samples={result.sample_count}, scorecard={result.scorecard_path}",
            )
        return result.exit_code

    if args.print_config:
        return _print_model_run_plan(args)

    if _uses_direct_model_run(args):
        if args.engine != "in-process":
            raise CliUsageError("direct model execution requires --engine in-process")
        if args.plan_only:
            return _print_model_run_plan(args)
        return _handle_direct_model_run(args)

    if _run_uses_unified_framework(args):
        if (
            (args.all_benchmarks or args.suite_ids or _run_benchmark_ids(args))
            and not _run_model_ids(args)
            and not _suite_presets_declare_models(args)
            and not args.contract_fixture
        ):
            raise CliUsageError("model-benchmark runs require --model for real evaluation")
        return _handle_worldfoundry_run(args)

    if not _has_complete_task_args(args):
        raise CliUsageError(
            "select a run target: `run --benchmark <id> --model <id>` for a benchmark-zoo cell, "
            "`run <model-id>` for direct model inference, or `run --plan <plan.json>` for replay. "
            "(The legacy --task-type/--benchmark-name/--data-path flow is retired.)"
        )

    return _handle_run_in_process(args)


# ── Run command ──────────────────────────────────────────────────


def _handle_run_in_process(args: argparse.Namespace) -> int:
    from worldfoundry.cli.utils import resolve_cli_benchmark_for_materialize
    from worldfoundry.evaluation.runner import (
        EvaluateRunRequest,
        execute_evaluate_run,
        materialize_requests_from_benchmark,
    )

    benchmark = resolve_cli_benchmark_for_materialize(args.task_type, args.benchmark_name)
    materialized = materialize_requests_from_benchmark(
        benchmark,
        args.data_path,
        limit=args.num_samples,
        split=args.split,
    )
    if args.engine == "existing-results":
        mode = "existing-results"
    else:
        mode = "model"
    if mode == "existing-results" and args.results_path is None:
        raise CliUsageError("run --engine existing-results requires --results-path")

    benchmark_metadata = {
        "suite": benchmark.suite,
        "benchmark_name": benchmark.benchmark_name,
        "task_type": benchmark.task_type,
        "backend": benchmark.backend,
        "evaluation_protocol": benchmark.evaluation_protocol,
    }
    dataset_metadata = {
        "root": str(Path(args.data_path).resolve()),
        "split": args.split,
        "sample_count": materialized.sample_count,
    }
    result = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=args.output_dir,
            mode=mode,
            requests=list(materialized.requests),
            requests_path=args.requests_path,
            results_path=args.results_path,
            metrics=tuple(args.metric) if args.metric is not None else ("artifact_count",),
            required_artifacts=tuple(args.required_artifact or ()),
            benchmark=benchmark_metadata,
            dataset=dataset_metadata,
            benchmark_id=args.benchmark_id or args.benchmark_name,
            model_id=(_run_model_ids(args)[0] if _run_model_ids(args) else args.model_type),
            model_runner=args.model_runner,
            model_zoo_manifest_dir=args.model_manifest_dir,
            model_variant_id=args.model_variant,
            model_parameters=_run_model_parameters(args),
            model_runtime=_run_model_runtime(args),
            model_config=load_json_mapping(args.model_config),
            dataset_id=args.dataset_id or args.benchmark_name,
            run_id=args.run_id,
            fail_on_sample_error=args.fail_on_sample_error,
            write_artifacts_index=not args.no_artifacts_index,
            generation_cache_dir=args.generation_cache_dir,
            generation_cache_mode=args.generation_cache_mode,
            generation_cache_namespace=args.generation_cache_namespace,
        )
    )
    payload = result.to_dict()
    payload["engine"] = args.engine
    if args.json:
        json_dump(payload)
    else:
        print_details(
            "Run result",
            {
                "status": result.status,
                "engine": args.engine,
                "samples": result.sample_count,
                "scorecard": result.scorecard_path,
            },
            plain=f"Run {result.status}: engine={args.engine}, samples={result.sample_count}, scorecard={result.scorecard_path}",
        )
    return result.exit_code


def _build_parser(model_run_schema: Any | None = None) -> argparse.ArgumentParser:
    """Build all CLI commands, keeping parser wiring near handler selection."""
    parser = WorldFoundryArgumentParser(
        prog=cli_prog_name(),
        description="Discover world models, run inference, and evaluate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Command areas:\n"
            "  Discovery:   zoo benchmarks, zoo models, tasks list\n"
            "  Run:         run\n"
            "  Score:       evaluate, compare-runs, index-runs, validate-artifact\n"
            "  Extend:      model and benchmark manifests, task YAML, runtime profiles\n"
            "  Maintain:    task validate, preflight, dataset, plan, metric\n"
        ),
    )
    # Global logging flags. They are pre-scanned and stripped from ``argv`` in
    # ``main`` (so they may appear before *or* after the subcommand) and applied
    # via ``configure_logging`` before the handler runs. Registered here so
    # ``--help`` documents them; argparse never enforces them.
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (env: WORLDFOUNDRY_LOG_LEVEL). Default INFO.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="Write a rotating log to this path (env: WORLDFOUNDRY_LOG_FILE).",
    )
    parser.add_argument(
        "--log-json",
        dest="log_json",
        action="store_true",
        default=False,
        help="Emit the file sink as JSON Lines (env: WORLDFOUNDRY_LOG_JSON).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Show the full traceback when a command fails.",
    )
    subparsers = parser.add_subparsers(dest="command")

    tui_parser = subparsers.add_parser(
        "tui",
        help="Browse models and launch runs in the terminal UI",
    )
    tui_parser.add_argument("--model-manifest-dir", type=Path)
    tui_parser.add_argument("--benchmark-manifest-dir", type=Path)
    tui_parser.add_argument("--runtime-profile-dir", type=Path)
    tui_parser.add_argument("--suite", action="append", dest="suite_ids", default=None)
    tui_parser.add_argument("--model-id")
    tui_parser.add_argument("--benchmark-id")
    tui_parser.add_argument("--metric", action="append", default=None)
    tui_parser.add_argument("--output-dir", type=Path)
    tui_parser.add_argument("--input", type=Path)
    tui_parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    tui_parser.add_argument("--video", type=Path)
    tui_parser.add_argument("--trajectory-file", type=Path)
    tui_parser.add_argument("--prompt")
    tui_parser.add_argument("--negative-prompt", dest="negative_prompt")
    tui_parser.add_argument("--task")
    tui_parser.add_argument("--mode")
    tui_parser.add_argument("--resize-mode")
    tui_parser.add_argument("--size")
    tui_parser.add_argument("--frames")
    tui_parser.add_argument("--steps")
    tui_parser.add_argument("--frames-per-generation", "--chunk-frames", dest="frames_per_generation")
    tui_parser.add_argument("--guidance-scale", "--cfg-scale", dest="guidance_scale")
    tui_parser.add_argument("--seed")
    tui_parser.add_argument("--fps")
    tui_parser.add_argument("--dtype")
    tui_parser.add_argument("--max-sequence-length", dest="max_sequence_length")
    tui_parser.add_argument("--cam-type", "--camera-type", dest="cam_type")
    tui_parser.add_argument("--interactions", "--actions", dest="interactions")
    tui_parser.add_argument("--output-formats", "--output_formats", dest="output_formats")
    tui_parser.add_argument("--trajectory", "--camera-trajectory", dest="trajectory")
    tui_parser.add_argument("--angle")
    tui_parser.add_argument("--distance")
    tui_parser.add_argument("--orbit-radius", dest="orbit_radius")
    tui_parser.add_argument("--zoom-ratio", dest="zoom_ratio")
    tui_parser.add_argument("--alpha-threshold", dest="alpha_threshold")
    tui_parser.add_argument("--static-scene", action="store_true", default=None)
    tui_parser.add_argument("--low-vram", action="store_true", default=None)
    tui_parser.add_argument("--disable-lora", action="store_true", default=None)
    tui_parser.add_argument("--vis-rendering", action="store_true", default=None)
    tui_parser.add_argument("--offload-t5", "--offload_t5", dest="offload_t5", action="store_true", default=None)
    tui_parser.add_argument(
        "--offload-transformer-during-vae",
        "--offload_transformer_during_vae",
        dest="offload_transformer_during_vae",
        action="store_true",
        default=None,
    )
    tui_parser.add_argument("--offload-vae", "--offload_vae", dest="offload_vae", action="store_true", default=None)
    tui_parser.add_argument("--output-path", type=Path)
    tui_parser.add_argument("--ckpt-type", "--weight-type", dest="ckpt_type")
    tui_parser.add_argument("--ckpt-root", type=Path)
    tui_parser.add_argument("--ckpt-path", type=Path)
    tui_parser.add_argument("--conda-envs-root", type=Path)
    tui_parser.add_argument("--gpu")
    tui_parser.add_argument("--fallback", action="store_true")
    tui_parser.add_argument("--catalog-json", action="store_true")
    tui_parser.add_argument("--print-command", action="store_true")
    tui_parser.add_argument("--write-suite-plan", action="store_true")
    tui_parser.set_defaults(func=_handle_tui)

    from .tasks import register_task_subparsers

    register_task_subparsers(subparsers)

    from .dataset import register_dataset_subparser

    register_dataset_subparser(subparsers)

    from .config import register_config_subparser

    register_config_subparser(subparsers)

    from .plan_metric import register_plan_metric_subparsers

    register_plan_metric_subparsers(subparsers)

    from .evaluation_intents import register_evaluation_intent_subparsers

    register_evaluation_intent_subparsers(subparsers)

    from .reporting import register_reporting_subparsers

    register_reporting_subparsers(subparsers)

    from .models import register_model_subparsers

    register_model_subparsers(subparsers)

    from .preflight import register_preflight_subparser

    register_preflight_subparser(subparsers)

    from .zoo import register_zoo_subparser

    register_zoo_subparser(subparsers)

    from .mcp import add_mcp_parser

    add_mcp_parser(subparsers)

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

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        aliases=["eval"],
        help="Evaluate materialized world-model outputs through the deterministic eval-core path",
        description=(
            "Evaluate materialized world-model outputs through the deterministic eval-core path. "
            "Normalize existing generation outputs into the standard WorldFoundry run artifact set. "
            "This connects model demos, benchmark runners, and WorldFoundry scorecards."
        ),
    )
    evaluate_parser.add_argument(
        "--plan",
        type=Path,
        help="worldfoundry-run-plan JSON file to execute through EvaluateRunRequest.",
    )
    evaluate_parser.add_argument(
        "--mode",
        choices=["existing-results", "existing", "results", "model", "generate"],
        default="existing-results",
        help=(
            "Execution mode. existing-results scores materialized outputs (--results-path); "
            "model resolves a runner and scores generated outputs. "
            "With --embodied-spec and no --results-path, mode is forced to model."
        ),
    )
    evaluate_parser.add_argument(
        "--requests-path",
        type=Path,
        help="JSON or JSONL GenerationRequest/sample file. If omitted in existing-results mode, requests are derived from results.",
    )
    evaluate_parser.add_argument(
        "--embodied-spec",
        "--vla-va-wam-spec",
        dest="embodied_spec",
        help=(
            "JSON file or inline EmbodiedGenerationSpec for VLA/VA/VAM/WAM tasks. "
            "Use with --samples-path or pre-materialized --requests-path."
        ),
    )
    evaluate_parser.add_argument(
        "--samples-path",
        type=Path,
        help="JSON/JSONL samples or episodes materialized with --embodied-spec.",
    )
    evaluate_parser.add_argument(
        "--results-path",
        type=Path,
        help="JSON or JSONL GenerationResult/output file. Required for existing-results mode.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=TMP_ROOT / "worldfoundry_evaluate",
        help="Directory for run_manifest.json, requests/results ledgers, metrics, artifacts index, and scorecard.json.",
    )
    evaluate_parser.add_argument(
        "--task-type",
        help=(
            "Retired legacy materialization flow; kept for compatibility and always fails with "
            "guidance. Use `run --benchmark <id> --model <id>` or `task materialize` instead."
        ),
    )
    evaluate_parser.add_argument(
        "--benchmark-name",
        help="Retired; see --task-type.",
    )
    evaluate_parser.add_argument(
        "--data-path",
        type=Path,
        help="Retired; see --task-type.",
    )
    evaluate_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit materialized benchmark samples before evaluation.",
    )
    evaluate_parser.add_argument(
        "--split",
        default="default",
        help="Split label written into materialized GenerationRequest rows.",
    )
    evaluate_parser.add_argument(
        "--benchmark-id",
        help="Benchmark id to record in the scorecard metadata.",
    )
    evaluate_parser.add_argument(
        "--model-id",
        help="Model id to record in the scorecard metadata, or resolve in --mode model.",
    )
    evaluate_parser.add_argument(
        "--model-runner",
        help="Optional 'module:Class' runner target for --mode model.",
    )
    evaluate_parser.add_argument(
        "--model-manifest-dir",
        type=Path,
        help="Optional model-zoo manifest directory used to resolve --model-id in --mode model.",
    )
    evaluate_parser.add_argument(
        "--model-variant",
        help="Optional model-zoo variant id used with --model-manifest-dir.",
    )
    evaluate_parser.add_argument(
        "--model-parameter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Runner parameter for --mode model. VALUE is parsed as JSON when possible; repeatable.",
    )
    evaluate_parser.add_argument(
        "--model-runtime",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Runner runtime setting for --mode model. VALUE is parsed as JSON when possible; repeatable.",
    )
    evaluate_parser.add_argument(
        "--model-config",
        type=Path,
        help="JSON WorldModelConfig object for --mode model. Ignored when --model-manifest-dir is set.",
    )
    evaluate_parser.add_argument(
        "--dataset-id",
        help="Dataset id to record in the scorecard metadata.",
    )
    evaluate_parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Built-in deterministic eval-core metric. Repeatable. Supported: artifact_count, "
            "required_artifacts_present, has_artifact:<name>, numeric, numeric:<name>. "
            "With --embodied-spec, metric ids are resolved as embodied result-field metrics."
        ),
    )
    evaluate_parser.add_argument(
        "--required-artifact",
        action="append",
        default=None,
        help="Artifact name that must be present for required_artifacts_present.",
    )
    evaluate_parser.add_argument("--run-id", help="Stable run id to write into run_manifest.json and scorecard.json.")
    evaluate_parser.add_argument(
        "--fail-on-sample-error",
        action="store_true",
        help="Return exit code 1 when any sample fails generation or metric evaluation.",
    )
    evaluate_parser.add_argument(
        "--no-artifacts-index",
        action="store_true",
        help="Do not write artifacts.jsonl even when output artifacts are present.",
    )
    _add_generation_cache_args(evaluate_parser, namespace="evaluate_model")
    evaluate_parser.add_argument("--json", action="store_true", help="Print the run result as JSON.")
    evaluate_parser.set_defaults(func=_handle_evaluate)

    validate_parser = subparsers.add_parser("validate", help="Validate WorldFoundry benchmark metadata loading")
    _add_task_selector(validate_parser, require_data_path=True)
    validate_parser.add_argument("--num-samples", type=int, default=None)
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(func=_handle_validate)

    run_parser = subparsers.add_parser(
        "run",
        help="Run model inference or a benchmark",
        description=(
            (
                f"{model_run_schema.display_name}: {model_run_schema.description}\n"
                f"Status: {model_run_schema.runner_entry_kind}; "
                f"integration={model_run_schema.integration_status}; "
                f"source={model_run_schema.source_status}."
            )
            if model_run_schema is not None
            else "Execute a model directly, or run and score a model through the unified benchmark facade."
        ),
    )
    run_parser.add_argument(
        "model_slug",
        nargs="?",
        metavar="MODEL",
        help="Model id or alias. Positional syntax enables typed model-specific pipeline options.",
    )
    run_parser.add_argument("--plan", type=Path, help="worldfoundry-run-plan JSON file to execute.")
    run_parser.add_argument("--suite", action="append", dest="suite_ids", default=None)
    run_parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run or plan the formal docs benchmark inventory suite.",
    )
    run_parser.add_argument("--suite-preset-path", type=Path)
    run_parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmark_ids",
        default=None,
        help="Benchmark-zoo id or alias. Repeat for a model x benchmark suite.",
    )
    run_parser.add_argument(
        "--model",
        action="append",
        dest="model_ids",
        default=None,
        help="Model-zoo id, alias, or custom model id. Repeat for a suite.",
    )
    run_parser.add_argument("--benchmark-manifest-dir", type=Path, default=BENCHMARK_ZOO_DIR)
    run_parser.add_argument(
        "--mode",
        dest="benchmark_mode",
        choices=_BENCHMARK_RUN_MODE_CHOICES,
        default="official-run",
        help="Benchmark runner mode for model x benchmark runs.",
    )
    run_parser.add_argument("--plan-only", action="store_true", help="Plan a suite without executing cells.")
    run_parser.add_argument(
        "--model-workers",
        type=int,
        default=1,
        help="Spawn up to this many model-level suite workers (default: 1).",
    )
    run_parser.add_argument(
        "--performance-profile",
        choices=["balanced", "throughput", "latency"],
        default="balanced",
        help=(
            "Runtime optimization policy: throughput enables shape-bucketed eval compilation "
            "and hardware-scoped persistent kernel autotuning; latency avoids first-run "
            "compile/tuning cost."
        ),
    )
    run_parser.add_argument(
        "--model-worker-device",
        action="append",
        default=None,
        metavar="CUDA_VISIBLE_DEVICES",
        help=(
            "CUDA device group pinned to one model worker, for example 0 or 4,5,6,7. "
            "Repeat for parallel workers; groups must not overlap and must have equal size."
        ),
    )
    run_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved model-specific configuration and exit without loading the model.",
    )
    run_parser.add_argument("--resume", action="store_true", help="Reuse completed generation, metric results, and suite cells when inputs match.")
    run_parser.add_argument("--no-skip-incompatible", dest="skip_incompatible", action="store_false", default=True)
    run_parser.add_argument("--fail-on-skipped", action="store_true")
    _RETIRED_TASK_FLAG_HELP = (
        "Retired legacy flow; kept for compatibility and always fails with guidance. "
        "Use --benchmark/--model or `task materialize` instead."
    )
    run_parser.add_argument("--task-type", help=_RETIRED_TASK_FLAG_HELP)
    run_parser.add_argument("--benchmark-name", help=_RETIRED_TASK_FLAG_HELP)
    run_parser.add_argument("--data-path", type=Path, help=_RETIRED_TASK_FLAG_HELP)
    run_parser.add_argument("--task-name", help="Task YAML name for benchmark-zoo model generation.")
    run_parser.add_argument("--task-root", action="append", type=Path, default=None)
    run_parser.add_argument("--task-benchmark")
    run_parser.add_argument("--task-recursive", action="store_true")
    run_parser.add_argument("--task-root-dir", type=Path)
    run_parser.add_argument(
        "--engine",
        choices=["in-process", "existing-results"],
        default="in-process",
        help=(
            "Execution engine: in-process materializes requests and runs the model runner; "
            "existing-results scores a materialized results file."
        ),
    )
    run_parser.add_argument("--model-type")
    run_parser.add_argument("--model-id", help="Model id written to scorecard metadata for in-process engines.")
    run_parser.add_argument("--model-runner", help="Optional 'module:Class' runner target for --engine in-process.")
    run_parser.add_argument(
        "--model-manifest-dir",
        type=Path,
        help="Optional model-zoo manifest directory used to resolve --model-id for --engine in-process.",
    )
    run_parser.add_argument("--model-variant", help="Optional model-zoo variant id for --engine in-process.")
    run_parser.add_argument(
        "--model-parameter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Runner parameter for --engine in-process. VALUE is parsed as JSON when possible; repeatable.",
    )
    run_parser.add_argument(
        "--model-runtime",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Runner runtime setting for --engine in-process. VALUE is parsed as JSON when possible; repeatable.",
    )
    run_parser.add_argument(
        "--model-config",
        type=Path,
        help="JSON WorldModelConfig object for --engine in-process. Ignored when --model-manifest-dir is set.",
    )
    run_parser.add_argument("--output-dir", default="./benchmark_results")
    run_parser.add_argument("--num-samples", type=int, default=None)
    run_parser.add_argument("--split", default="default")
    run_parser.add_argument("--requests-path", type=Path)
    run_parser.add_argument("--results-path", type=Path)
    run_parser.add_argument("--benchmark-id")
    run_parser.add_argument("--dataset-id")
    run_parser.add_argument("--generated-artifact-dir", type=Path)
    run_parser.add_argument("--output-artifact")
    run_parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Built-in deterministic metric for in-process engines. Repeatable. "
            "Supported: artifact_count, required_artifacts_present, has_artifact:<name>, numeric, numeric:<name>."
        ),
    )
    run_parser.add_argument("--required-artifact", action="append", default=None)
    run_parser.add_argument("--timeout", type=float)
    run_parser.add_argument("--workdir", type=Path)
    run_parser.add_argument("--env", action="append", default=None, metavar="KEY=VALUE")
    run_parser.add_argument(
        "--benchmark-parameter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Official benchmark-runner parameter. VALUE is parsed as JSON when possible; repeatable.",
    )
    run_parser.add_argument("--materialize-placeholders", action="store_true", default=None)
    run_parser.add_argument("--no-materialize-placeholders", dest="materialize_placeholders", action="store_false")
    run_parser.add_argument("--fail-on-generation-error", action="store_true")
    run_parser.add_argument(
        "--contract-fixture",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--fail-on-sample-error", action="store_true")
    run_parser.add_argument("--no-artifacts-index", action="store_true")
    _add_generation_cache_args(run_parser, namespace="worldfoundry_run")
    run_parser.add_argument("--json", action="store_true")
    if model_run_schema is not None:
        from .model_run import register_model_run_arguments

        register_model_run_arguments(run_parser, model_run_schema)
        run_parser.set_defaults(model_run_schema=model_run_schema)
    run_parser.set_defaults(func=_handle_run)

    _curate_root_subparser_help(subparsers)
    return parser


def _extract_logging_flags(
    argv: list[str],
) -> tuple[str | None, str | None, bool | None, bool, list[str]]:
    """Pre-scan ``argv`` for the global logging/verbosity flags, returning them
    plus the remaining argv with those flags stripped.

    Recognizes both ``--flag value`` and ``--flag=value`` forms; the flags are
    accepted in any position (before *or* after the subcommand) because the
    real application happens here, ahead of argparse.
    """
    level = log_file = log_json = None
    verbose = False
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--log-level" and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            level = argv[i + 1]
            i += 2
            continue
        if token.startswith("--log-level="):
            level = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--log-file" and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            log_file = argv[i + 1]
            i += 2
            continue
        if token.startswith("--log-file="):
            log_file = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--log-json":
            log_json = True
            i += 1
            continue
        if token in ("-v", "--verbose"):
            verbose = True
            i += 1
            continue
        out.append(token)
        i += 1
    return level, log_file, log_json, verbose, out


def _configure_cli_logging(
    level: str | None,
    log_file: str | None,
    log_json: bool | None,
) -> None:
    """Apply ``configure_logging`` for the CLI and export the resolved values to
    the environment so framework-owned child processes (which re-enter
    ``main``) inherit the same configuration. A bad level falls back to INFO
    rather than aborting the whole command."""
    from worldfoundry.core import configure_logging

    try:
        configure_logging(level=level, log_file=log_file, json=log_json)
    except ValueError as exc:
        sys.stderr.write(f"warning: {exc} Falling back to INFO.\n")
        configure_logging(level="INFO", log_file=log_file, json=log_json, force=True)

    if level is not None:
        os.environ["WORLDFOUNDRY_LOG_LEVEL"] = str(level)
    if log_file is not None:
        os.environ["WORLDFOUNDRY_LOG_FILE"] = str(log_file)
    if log_json:
        os.environ["WORLDFOUNDRY_LOG_JSON"] = "1"


def _bind_cli_log_context(args: argparse.Namespace, *, run_id: str | None = None) -> None:
    """Attach command identity to logs emitted by this CLI process."""

    from worldfoundry.core import bind_log_context

    benchmark_id = getattr(args, "benchmark_id", None) or getattr(args, "benchmark_name", None)
    bind_log_context(
        run_id=run_id or getattr(args, "run_id", None),
        benchmark_id=benchmark_id,
        model_id=getattr(args, "model_id", None),
        phase="cli",
        command=getattr(args, "command", None),
    )


def _new_run_id() -> str:
    """Return a sortable, collision-resistant ID for a CLI-owned output run."""

    return f"wf-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _safe_run_log_component(run_id: str) -> str:
    """Convert a caller-provided run ID into one safe path component."""

    normalized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in run_id)
    return normalized.strip(".-") or "run"


def _prepare_cli_run_observability(
    args: argparse.Namespace,
    *,
    explicit_log_file: str | None,
) -> tuple[Path, str, float] | None:
    """Create an isolated log directory for commands that own an output tree."""

    output_dir = getattr(args, "output_dir", None)
    if output_dir in (None, ""):
        _bind_cli_log_context(args)
        return None
    if getattr(args, "plan_only", False) or getattr(args, "print_config", False):
        # Dry-run flows must not create ``logs/`` under the output tree or
        # rewrite the process logging environment (CM-09).
        _bind_cli_log_context(args)
        return None

    run_id = str(getattr(args, "run_id", None) or _new_run_id())
    # Only some command parsers expose --run-id, but every output-owning CLI
    # operation should still carry a generated ID in its logs and child env.
    setattr(args, "run_id", run_id)
    _bind_cli_log_context(args, run_id=run_id)

    from worldfoundry.core import configure_logging, log_context_environment, write_jsonl_event

    root = Path(output_dir).expanduser().resolve()
    if bool(getattr(args, "_requires_exclusive_output_dir", False)):
        # Some commands atomically claim their output directory as a run
        # transaction.  Keep lifecycle logs in an isolated sibling so logging
        # cannot pre-create that directory and defeat the exclusivity check.
        event_path = (
            root.parent
            / ".worldfoundry-cli-logs"
            / _safe_run_log_component(root.name)
            / _safe_run_log_component(run_id)
            / "events.jsonl"
        )
    else:
        event_path = root / "logs" / _safe_run_log_component(run_id) / "events.jsonl"
    # A caller-selected --log-file / env file remains authoritative. Otherwise
    # make the per-run JSONL artifact the default sink for this command.
    if explicit_log_file is None and not os.environ.get("WORLDFOUNDRY_LOG_FILE"):
        configure_logging(log_file=event_path, json=True, force=True)
        os.environ["WORLDFOUNDRY_LOG_FILE"] = str(event_path)
        os.environ["WORLDFOUNDRY_LOG_JSON"] = "1"
    os.environ["WORLDFOUNDRY_RUN_ID"] = run_id
    os.environ.update(log_context_environment())
    write_jsonl_event(
        event_path,
        level="INFO",
        event="run.started",
        message="WorldFoundry command started",
        logger_name=__name__,
        output_dir=str(root),
        command=getattr(args, "command", None),
    )
    return event_path, run_id, time.monotonic()


def _write_cli_run_lifecycle(
    state: tuple[Path, str, float] | None,
    *,
    event: str,
    level: str,
    message: str,
    exit_code: int | None = None,
    exception: BaseException | None = None,
) -> None:
    """Persist a terminal CLI lifecycle event even with a custom log sink."""

    if state is None:
        return
    from worldfoundry.core import write_jsonl_event

    event_path, _, started = state
    fields: dict[str, Any] = {"duration_seconds": round(time.monotonic() - started, 6)}
    if exit_code is not None:
        fields["exit_code"] = exit_code
    write_jsonl_event(
        event_path,
        level=level,
        event=event,
        message=message,
        logger_name=__name__,
        exception=exception,
        **fields,
    )


# Long-lived or workload-owning commands whose in-process logging must run
# through the configured pipeline even when the parsed namespace does not
# carry an ``output_dir`` value.
_EAGER_LOGGING_COMMANDS = frozenset(
    {
        "tui",
        "mcp",
    }
)


def _logging_flags_requested(level: str | None, log_file: str | None, log_json: bool | None) -> bool:
    """Whether the caller explicitly asked for configured logging (flag or env)."""

    if level is not None or log_file is not None or log_json is not None:
        return True
    return any(
        os.environ.get(name)
        for name in ("WORLDFOUNDRY_LOG_LEVEL", "WORLDFOUNDRY_LOG_FILE", "WORLDFOUNDRY_LOG_JSON")
    )


def _command_wants_eager_logging(args: argparse.Namespace) -> bool:
    """Whether the selected command should configure logging before dispatch.

    Pure query commands skip ``configure_logging`` entirely: its
    ``core.distributed`` reparenting currently drags ``torch`` into every
    process (CM-01), which query paths must not pay for.  Workload commands
    (anything owning an output tree, TUI, MCP server) keep the
    configured pipeline.  Dry-run variants (``--plan-only`` /
    ``--print-config`` / ``--model-status``) stay unconfigured like queries.
    """

    if getattr(args, "plan_only", False) or getattr(args, "print_config", False):
        return False
    if getattr(args, "command", None) in _EAGER_LOGGING_COMMANDS:
        return True
    return getattr(args, "output_dir", None) not in (None, "")


def _format_cli_error(exc: BaseException) -> str:
    """Render one concise, typed error line for terminal users."""

    message = str(exc).strip()
    type_name = type(exc).__name__
    if not message:
        return type_name
    # ``str(KeyError("x"))`` is just ``"'x'"``; always carry the type so bare
    # messages stay interpretable.
    return f"{type_name}: {message}"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse args and dispatch to the selected command handler."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _log_level, _log_file, _log_json, _verbose, raw_argv = _extract_logging_flags(raw_argv)
    # CM-01: defer configure_logging so ``--help`` and pure query commands do
    # not pull the heavy core.distributed/torch import chain. Explicit flags
    # or WORLDFOUNDRY_LOG_* environment values keep today's eager behaviour.
    if _logging_flags_requested(_log_level, _log_file, _log_json):
        _configure_cli_logging(_log_level, _log_file, _log_json)
    if "--tui" in raw_argv:
        _configure_cli_logging(_log_level, _log_file, _log_json)
        from worldfoundry.cli.tui import main as tui_main

        return tui_main([item for item in raw_argv if item != "--tui"])

    model_run_schema = None
    if raw_argv and raw_argv[0] == "run":
        from .model_run import (
            ModelRunSchemaError,
            load_model_run_schema,
            model_id_from_run_argv,
            option_from_argv,
        )

        model_ref = model_id_from_run_argv(raw_argv)
        positional_model = len(raw_argv) > 1 and not raw_argv[1].startswith("-")
        wants_model_schema = positional_model or any(
            item in {"-h", "--help", "--print-config", "--model-status"}
            or item.startswith("--pipeline.")
            or item.startswith("--runtime.")
            for item in raw_argv[1:]
        )
        if model_ref and wants_model_schema:
            try:
                model_run_schema = load_model_run_schema(
                    model_ref,
                    variant_id=option_from_argv(raw_argv, "--model-variant"),
                    task_id=option_from_argv(raw_argv, "--pipeline.task-profile"),
                )
            except ModelRunSchemaError as exc:
                parser = _build_parser()
                parser.error(str(exc))

    parser = _build_parser(model_run_schema)
    args = parser.parse_args(raw_argv)
    if not hasattr(args, "func"):
        _print_first_run_banner()
        return 0
    if _command_wants_eager_logging(args):
        _configure_cli_logging(_log_level, _log_file, _log_json)
    run_observability = _prepare_cli_run_observability(args, explicit_log_file=_log_file)

    try:
        exit_code = args.func(args)
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "INFO" if exit_code == 0 else "ERROR",
                "cli.command.finished",
                "CLI command finished",
                command=getattr(args, "command", None),
                exit_code=exit_code,
            )
        _write_cli_run_lifecycle(
            run_observability,
            event="run.finished",
            level="INFO" if exit_code == 0 else "ERROR",
            message="WorldFoundry command finished",
            exit_code=exit_code,
        )
        return exit_code
    except KeyboardInterrupt:
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "WARNING",
                "cli.command.cancelled",
                "CLI command interrupted",
                command=getattr(args, "command", None),
                exit_code=130,
            )
        _write_cli_run_lifecycle(
            run_observability,
            event="run.cancelled",
            level="WARNING",
            message="WorldFoundry command interrupted",
            exit_code=130,
        )
        parser.exit(130, "Interrupted.\n")
    except CliUsageError as exc:
        # Usage mistakes (bad flag combinations, missing required values) keep
        # the argparse convention: exit 2, one concise stderr line, no stack
        # trace or stack-carrying log event (CM-08).  Runtime failures below
        # stay on exit 1.  Returning (not parser.exit) preserves the historic
        # ``return 2`` handler behaviour for in-process callers of ``main``.
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "ERROR",
                "cli.command.usage_error",
                "CLI command rejected: usage error",
                command=getattr(args, "command", None),
                error=str(exc),
            )
        _write_cli_run_lifecycle(
            run_observability,
            event="run.failed",
            level="ERROR",
            message="WorldFoundry command rejected: usage error",
            exit_code=2,
            exception=exc,
        )
        if getattr(args, "json", False):
            json_dump(
                {
                    "status": "error",
                    "command": getattr(args, "command", None),
                    "error": {"type": "usage", "message": str(exc)},
                    "exit_code": 2,
                }
            )
        print_notice(
            str(exc),
            level="error",
            stream=sys.stderr,
            hint=f"Run {cli_prog_name()} {getattr(args, 'command', '')} --help for usage.",
        )
        return 2
    except Exception as exc:
        from worldfoundry.core.observability.logging_setup import is_configured

        if is_configured():
            from worldfoundry.core import get_logger

            get_logger(__name__).event(
                "ERROR",
                "cli.command_failed",
                "CLI command failed",
                exc_info=_verbose,
                command=getattr(args, "command", None),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        _write_cli_run_lifecycle(
            run_observability,
            event="run.failed",
            level="ERROR",
            message="WorldFoundry command failed",
            exit_code=1,
            exception=exc,
        )
        # Machine consumers asked for JSON: keep stdout a parseable contract
        # even on failure (CM-07), with the runtime-failure exit code 1 kept
        # distinct from argparse usage errors (exit 2).
        if getattr(args, "json", False):
            json_dump(
                {
                    "status": "error",
                    "command": getattr(args, "command", None),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "exit_code": 1,
                }
            )
        if _verbose:
            import traceback

            traceback.print_exc()
        print_notice(_format_cli_error(exc), level="error", stream=sys.stderr)
        parser.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
