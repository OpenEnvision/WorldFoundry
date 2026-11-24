# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""``wf evaluate`` (alias ``eval``) and ``wf validate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldfoundry.evals.utils import TMP_ROOT

from .presentation import print_details
from .run import execute_run_plan_file
from .utils import CliUsageError, add_generation_cache_args, json_dump, load_json_mapping, parse_key_value_mapping


def _load_json_or_jsonl(path: Path | None):
    """Load a JSON or JSONL file, returning ``None`` when *path* is ``None``."""
    if path is None:
        return None
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
                {"sample_id": sample_id, **item} if isinstance(item, dict) else {"sample_id": sample_id, "value": item}
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
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload


def _add_task_selector(parser: argparse.ArgumentParser, *, require_data_path: bool) -> None:
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--benchmark-name", required=True)
    if require_data_path:
        parser.add_argument("--data-path", required=True)


def _validate_evaluate_usage(args: argparse.Namespace) -> None:
    """Raise :class:`CliUsageError` for mutually exclusive evaluate flags (CM-08)."""

    task_args = (args.task_type, args.benchmark_name, args.data_path)
    if args.embodied_spec is not None and any(item is not None for item in task_args):
        raise CliUsageError("--embodied-spec cannot be combined with --task-type/--benchmark-name/--data-path")
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
    from worldfoundry.evals.tasks.execution.orchestration.evaluate import EvaluateRunRequest, execute_evaluate_run

    if args.plan is not None:
        result = execute_run_plan_file(args)
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
        from worldfoundry.core.json_contract import to_plain
        from worldfoundry.evals.tasks.embodied.contracts import (
            EmbodiedGenerationSpec,
            EvaluationTrack,
            RequestKind,
        )
        from worldfoundry.evals.tasks.embodied.materialize import materialize_vla_va_wam_requests
        from worldfoundry.evals.tasks.embodied.metrics import metric_suite

        spec_payload = _load_json_mapping_or_inline(args.embodied_spec, field_name="--embodied-spec")
        if spec_payload is None:
            raise CliUsageError("--embodied-spec is required")
        embodied_spec = EmbodiedGenerationSpec.from_dict(spec_payload)
        track = EvaluationTrack(embodied_spec.track).value
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
            "track": track,
            "request_kind": RequestKind(embodied_spec.kind).value,
            "action_space": to_plain(embodied_spec.action_space),
            "required_capabilities": list(embodied_spec.required_capabilities),
        }
        dataset_metadata = {
            "dataset_id": args.dataset_id or "vla_va_wam_materialized",
            "name": args.dataset_id or "vla_va_wam_materialized",
            "split": args.split,
        }
        if requests is not None:
            dataset_metadata["sample_count"] = len(requests)
        metric_objects = metric_suite(tuple(args.metric or ()), track=track)
        model_parameters = {
            "track": track,
            "metadata_namespace": "vla_va_wam",
            "capabilities": list(embodied_spec.required_capabilities),
            **model_parameters,
        }
    if args.task_type and args.benchmark_name and args.data_path:
        from worldfoundry.cli.utils import resolve_cli_benchmark_for_materialize
        from worldfoundry.evals.tasks.execution.orchestration.materialize import materialize_requests_from_benchmark

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


def _handle_validate(args: argparse.Namespace) -> int:
    from worldfoundry.evals.tasks.catalog.specs import (
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


def register_evaluate_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``wf evaluate`` (alias ``eval``) and ``wf validate``."""
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
    evaluate_parser.add_argument(
        "--resume", action="store_true", help="Reuse matching completed generation and metric checkpoints."
    )
    evaluate_parser.add_argument(
        "--metric-batch-size", type=int, default=32, help="Batch size for per-sample scoring (default: 32)."
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
    add_generation_cache_args(evaluate_parser, namespace="evaluate_model")
    evaluate_parser.add_argument("--json", action="store_true", help="Print the run result as JSON.")
    evaluate_parser.set_defaults(func=_handle_evaluate)

    validate_parser = subparsers.add_parser("validate", help="Validate WorldFoundry benchmark metadata loading")
    _add_task_selector(validate_parser, require_data_path=True)
    validate_parser.add_argument("--num-samples", type=int, default=None)
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(func=_handle_validate)
