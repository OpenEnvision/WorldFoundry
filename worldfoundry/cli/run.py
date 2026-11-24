# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""``wf run``: direct model inference, model x benchmark cells, suites, and run-plan replay."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from worldfoundry.evals.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR

from .presentation import print_details
from .utils import (
    CliUsageError,
    add_generation_cache_args,
    cli_prog_name,
    json_dump,
    load_json_mapping,
    parse_key_value_mapping,
)

_BENCHMARK_RUN_MODE_CHOICES = ("normalizer", "official-run", "official-validation", "contract")


def execute_run_plan_file(args: argparse.Namespace):
    """Load and execute a run-plan JSON file through the evaluation runner."""
    from worldfoundry.evals.tasks.execution.orchestration.evaluate import execute_evaluate_run
    from worldfoundry.evals.tasks.execution.orchestration.plan import evaluate_request_from_run_plan, load_run_plan

    plan = load_run_plan(args.plan)
    request = replace(
        evaluate_request_from_run_plan(plan),
        resume=args.resume,
        metric_batch_size=args.metric_batch_size,
    )
    return execute_evaluate_run(request)


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
    from worldfoundry.evals.models.pipelines.loading import GENERATION_DEFAULTS_PARAMETER

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
    return tuple(args.benchmark_ids or ()) + ((args.benchmark_id,) if args.benchmark_id else ())


def _suite_presets_declare_models(args: argparse.Namespace) -> bool:
    """Check whether selected suite presets declare their own model ids."""
    suite_ids = tuple(args.suite_ids or ())
    if not suite_ids:
        return False
    try:
        from worldfoundry.evals.tasks.execution.orchestration.suite_presets import (
            get_model_benchmark_suite_preset,
        )

        return all(
            bool(get_model_benchmark_suite_preset(suite_id, args.suite_preset_path).get("model_ids"))
            for suite_id in suite_ids
        )
    except Exception:
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
    from worldfoundry.evals.tasks.execution.orchestration.run import WorldFoundryRunRequest

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
        metric_batch_size=args.metric_batch_size,
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
    from worldfoundry.evals.tasks.execution.orchestration.run import run_worldfoundry

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
    from worldfoundry.evals.models.pipelines.loading import GENERATION_DEFAULTS_PARAMETER

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


def _validate_model_run_required_fields(args: argparse.Namespace, resolution: Any) -> None:
    """Reject incomplete inline requests before direct execution or planning."""
    schema = args.model_run_schema
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
        if field.scope == "call" and field.required and getattr(args, field.dest, None) in (None, "")
    )
    if missing and args.requests_path is None:
        raise CliUsageError(f"direct {schema.model_id} inference requires {', '.join(missing)}")


def _handle_direct_model_run(args: argparse.Namespace) -> int:
    """Execute one model directly from its typed inference contract."""
    schema = args.model_run_schema
    if not schema.runnable:
        raise CliUsageError(
            f"model {schema.model_id!r} is not runnable: {schema.blocked_reason}. "
            f"Use `{cli_prog_name()} run {schema.requested_model_id} --model-status` for its contract.",
        )
    resolution = _resolved_model_run_options(args)
    _validate_model_run_required_fields(args, resolution)

    from worldfoundry.evals.api import GenerationRequest
    from worldfoundry.evals.tasks.execution.orchestration.evaluate import EvaluateRunRequest, execute_evaluate_run

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
            resume=args.resume,
            metric_batch_size=args.metric_batch_size,
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
        result = execute_run_plan_file(args)
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
            _validate_model_run_required_fields(args, _resolved_model_run_options(args))
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


def _handle_run_in_process(args: argparse.Namespace) -> int:
    from worldfoundry.cli.utils import resolve_cli_benchmark_for_materialize
    from worldfoundry.evals.tasks.execution.orchestration.evaluate import EvaluateRunRequest, execute_evaluate_run
    from worldfoundry.evals.tasks.execution.orchestration.materialize import materialize_requests_from_benchmark

    benchmark = resolve_cli_benchmark_for_materialize(args.task_type, args.benchmark_name)
    materialized = materialize_requests_from_benchmark(
        benchmark,
        args.data_path,
        limit=args.num_samples,
        split=args.split,
    )
    mode = "existing-results" if args.engine == "existing-results" else "model"
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
            resume=args.resume,
            metric_batch_size=args.metric_batch_size,
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


def register_run_subparser(subparsers: argparse._SubParsersAction, model_run_schema: Any | None = None) -> None:
    """Register ``wf run``; a resolved model schema adds its typed ``--pipeline.*`` options."""
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
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed generation, metric results, and suite cells when inputs match.",
    )
    run_parser.add_argument(
        "--metric-batch-size", type=int, default=32, help="Batch size for per-sample scoring (default: 32)."
    )
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
    add_generation_cache_args(run_parser, namespace="worldfoundry_run")
    run_parser.add_argument("--json", action="store_true")
    if model_run_schema is not None:
        from .model_run import register_model_run_arguments

        register_model_run_arguments(run_parser, model_run_schema)
        run_parser.set_defaults(model_run_schema=model_run_schema)
    run_parser.set_defaults(func=_handle_run)
