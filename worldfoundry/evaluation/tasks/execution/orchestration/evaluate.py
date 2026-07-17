"""Unified model evaluation facade and entry point for WorldFoundry.

This module orchestrates the complete lifecycle of benchmark evaluations. It handles both
offline results metric calculations ('existing-results' mode) and active model-based closed-loop or
open-loop simulation/inference ('model' mode). It automatically leverages local caching,
aggregates scores, compiles scorecard outputs, and integrates with registered metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, Metric
from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import BenchmarkZooInTreeEvaluator
from worldfoundry.evaluation.tasks.execution.framework.in_tree_registry import target_benchmark_metrics
from worldfoundry.evaluation.tasks.metrics.registry import BuiltinExistingResultsMetric, create_existing_results_metric
from worldfoundry.evaluation.utils import build_version_context, read_json_or_jsonl

from .cache import cache_paths_from_stats, run_generation_with_cache
from .contract import ContractRunRequest, execute_contract_run
from .existing_results import ExistingResultsMetric, ExistingResultsRunRequest, execute_existing_results

# Schema versions defining structure format of request/result payloads
EVALUATE_RUN_REQUEST_SCHEMA_VERSION = "worldfoundry-evaluate-run-request"
EVALUATE_RUN_RESULT_SCHEMA_VERSION = "worldfoundry-evaluate-run-result"


@dataclass(frozen=True)
class EvaluateRunRequest:
    """Unified evaluation facade for materialized generation outputs and model testing loops.

    Serves as the root configuration for orchestrating an entire evaluation lifecycle:
    1. Instantiating a specific `WorldModelRunner` (e.g. VlaEvalClosedLoopBridgeRunner or HuggingFace endpoints).
    2. Distributing batches of `GenerationRequest` instances to the model.
    3. Caching and resuming inference outputs safely.
    4. Automatically piping model outputs (`GenerationResult`s) through a specified metric suite.
    5. Formatting final deliverables, manifests, and Scorecards.
    """

    output_dir: str | Path
    mode: str = "existing-results"
    requests: Sequence[Any] | None = None
    requests_path: str | Path | None = None
    results: Any = None
    results_path: str | Path | None = None
    metrics: Sequence[Any] = ("artifact_count",)
    required_artifacts: Sequence[str] = ()
    benchmark: Mapping[str, Any] | Any | None = None
    model: Mapping[str, Any] | Any | None = None
    dataset: Mapping[str, Any] | Any | None = None
    runner: Any = None
    benchmark_id: str | None = None
    model_id: str | None = None
    model_runner: str | None = None
    model_zoo_manifest_dir: str | Path | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    dataset_id: str | None = None
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    cleanup_runner: bool = True
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "evaluate_model"
    run_metadata: Mapping[str, Any] | None = None
    schema_version: str = EVALUATE_RUN_REQUEST_SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluateRunResult:
    """Execution summary and manifest paths produced by an evaluation run.

    This dataclass encapsulates the outcomes of an evaluation run, including status,
    exit codes, paths to generated artifacts (manifest, execution plan, scorecard),
    and counts of processed samples and artifacts.
    """
    schema_version: str
    mode: str
    delegate_runner: str
    status: str
    exit_code: int
    output_dir: Path
    manifest_path: Path
    execution_plan_path: Path
    scorecard_path: Path
    sample_count: int
    successful_sample_count: int
    failed_sample_count: int
    artifact_count: int

    def to_dict(self) -> dict[str, Any]:
        """Converts the evaluation execution summary into a serializable dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the `EvaluateRunResult`
                            with `Path` objects converted to strings for serialization.
        """
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "delegate_runner": self.delegate_runner,
            "status": self.status,
            "exit_code": self.exit_code,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "execution_plan_path": str(self.execution_plan_path),
            "scorecard_path": str(self.scorecard_path),
            "sample_count": self.sample_count,
            "successful_sample_count": self.successful_sample_count,
            "failed_sample_count": self.failed_sample_count,
            "artifact_count": self.artifact_count,
        }


@dataclass(frozen=True)
class _ResolvedRunner:
    """Internal encapsulation bridging explicit runner objects and runtime metadata context.

    This dataclass stores an instantiated model runner along with its resolved model ID
    and other diagnostic information, used internally during 'model' evaluation mode.
    """
    runner: Any
    model_id: str
    source: str = "provided_runner"
    runner_target: str | None = None
    diagnostics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the resolved model runner information into a structured dictionary.

        Returns:
            dict[str, Any]: A dictionary containing metadata about the resolved runner.
        """
        return {
            "model_id": self.model_id,
            "source": self.source,
            "runner_target": self.runner_target,
            "runner_class": f"{self.runner.__class__.__module__}:{self.runner.__class__.__qualname__}",
            "diagnostics": dict(self.diagnostics or {}),
        }


def _load_optional_source(value: Any, path: str | Path | None) -> Any:
    """Safely retrieves Python objects or reads JSON payloads from disk.

    Args:
        value: An already loaded object, if available.
        path: A file path to load the object from if `value` is None.

    Returns:
        The loaded object or the provided value, or None if both are empty.
    """
    if value is not None:
        return value
    if path is not None:
        return read_json_or_jsonl(path)
    return None


def _looks_like_row(value: Mapping[str, Any]) -> bool:
    """Heuristic identifying whether an arbitrary dictionary structurally matches a WorldFoundry row item.

    This function checks for the presence of common keys found in WorldFoundry sample
    request or result rows to determine if a mapping represents a single row.

    Args:
        value: The dictionary to check.

    Returns:
        True if the dictionary contains keys typical of a WorldFoundry row, False otherwise.
    """
    row_keys = {
        "sample_id",
        "id",
        "request_id",
        "task_name",
        "task_id",
        "inputs",
        "outputs",
        "artifacts",
        "status",
        "error",
        "metrics",
        "scores",
    }
    return bool(row_keys.intersection(value.keys()))


def _rows_from_source(value: Any, *, preferred_keys: Sequence[str]) -> list[Any]:
    """Recursively drills down into nested data sources (JSON blocks or raw dicts) to extract iterable sample rows.

    This function attempts to parse various input types (file paths, JSON strings, mappings, iterables)
    into a flat list of sample rows. It prioritizes keys specified in `preferred_keys` for nested data.

    Args:
        value: The source data which can be a path, string, mapping, or iterable.
        preferred_keys: A sequence of keys to look for when `value` is a mapping to find nested rows.

    Returns:
        A list of extracted sample rows.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value)
        stripped = text.lstrip()
        # If it's a Path object or a string that doesn't look like JSON, try to read from file.
        if isinstance(value, Path) or not stripped.startswith(("{", "[")):
            path = Path(value)
            if path.exists():
                return _rows_from_source(read_json_or_jsonl(path), preferred_keys=preferred_keys)
        # If it's a string that looks like JSON, parse it.
        if stripped.startswith(("{", "[")):
            return _rows_from_source(json.loads(stripped), preferred_keys=preferred_keys)
    if isinstance(value, Mapping):
        # Look for preferred nested keys first.
        for key in preferred_keys:
            nested = value.get(key)
            if nested is not None:
                return _rows_from_source(nested, preferred_keys=preferred_keys)
        # If the mapping itself looks like a row, return it as a single element list.
        if _looks_like_row(value):
            return [value]
        # Otherwise, treat the mapping as a dict of sample_id -> item.
        rows = []
        for sample_id, item in value.items():
            if isinstance(item, Mapping):
                row = dict(item)
            else:
                row = {"value": item}
            row.setdefault("sample_id", str(sample_id))
            rows.append(row)
        return rows
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        # If it's an iterable of rows, return it directly.
        return list(value)
    # Default case: treat the value itself as a single row.
    return [value]


def _sample_id_from(value: Any, index: int) -> str:
    """Resolves or defaults an execution tracking `sample_id` from generic dictionaries/objects.

    Attempts to extract 'sample_id' or 'id' from the value or its nested 'request' field.
    If no ID is found, it generates a default ID based on the provided index.

    Args:
        value: The sample item (e.g., dict, object) from which to extract the ID.
        index: The positional index of the item, used for generating default IDs.

    Returns:
        A string representing the sample ID.
    """
    if isinstance(value, Mapping):
        for key in ("sample_id", "id"):
            item = value.get(key)
            if item is not None:
                return str(item)
        request = value.get("request")
        if isinstance(request, Mapping):
            for key in ("sample_id", "id"):
                item = request.get(key)
                if item is not None:
                    return str(item)
    for attr in ("sample_id", "id"):
        if hasattr(value, attr):
            item = getattr(value, attr)
            if item is not None:
                return str(item)
    return f"sample-{index:04d}"


def _task_name_from(value: Any) -> str:
    """Resolves the physical simulator or target dataset's task constraint grouping.

    Attempts to extract 'task_name' or 'task_id' from the value or its nested 'request' field.
    If no task name is found, it defaults to "materialized_results".

    Args:
        value: The sample item (e.g., dict, object) from which to extract the task name.

    Returns:
        A string representing the task name.
    """
    if isinstance(value, Mapping):
        for key in ("task_name", "task_id"):
            item = value.get(key)
            if item is not None:
                return str(item)
        request = value.get("request")
        if isinstance(request, Mapping):
            for key in ("task_name", "task_id"):
                item = request.get(key)
                if item is not None:
                    return str(item)
    return "materialized_results"


def _normalize_requests(source: Any) -> list[Any]:
    """Expands ambiguous objects into standard GenerationRequest-capable payloads.

    Takes a source of requests (e.g., list of dicts, GenerationRequest objects) and
    normalizes them into a consistent list of dictionaries, ensuring each has
    a `sample_id` and `task_name`.

    Args:
        source: The raw input source for generation requests.

    Returns:
        A list of dictionaries, each representing a normalized request.
    """
    rows = _rows_from_source(source, preferred_keys=("requests", "samples"))
    normalized = []
    for index, item in enumerate(rows):
        if isinstance(item, GenerationRequest):
            normalized.append(item)
            continue
        if isinstance(item, Mapping):
            row = dict(item)
        else:
            row = {"inputs": {"value": item}}
        row.setdefault("sample_id", _sample_id_from(item, index))
        row.setdefault("task_name", _task_name_from(item))
        normalized.append(row)
    return normalized


def _derive_requests_from_results(results_source: Any) -> list[GenerationRequest]:
    """Extracts original inputs and control boundaries when running directly from cached output JSONs.

    This is used in 'existing-results' mode when no explicit `requests` are provided,
    inferring them from the `results` by extracting 'request' sub-fields or creating
    minimal requests.

    Args:
        results_source: The source of existing `GenerationResult` objects or raw result data.

    Returns:
        A list of `GenerationRequest` objects derived from the results.
    """
    rows = _rows_from_source(results_source, preferred_keys=("results",))
    requests: list[GenerationRequest] = []
    for index, item in enumerate(rows):
        sample_id = _sample_id_from(item, index)
        if isinstance(item, Mapping) and isinstance(item.get("request"), Mapping):
            row = dict(item["request"])
        else:
            row = {}
        row.setdefault("sample_id", sample_id)
        row.setdefault("task_name", _task_name_from(item))
        requests.append(GenerationRequest.from_dict(row))
    return requests


def _coerce_generation_requests(source: Any) -> list[GenerationRequest]:
    """Force-casts payloads strictly to valid physical `GenerationRequest` instances.

    Normalizes a source of requests and then converts each item into a `GenerationRequest` object.

    Args:
        source: The raw input source for generation requests.

    Returns:
        A list of `GenerationRequest` objects.
    """
    requests: list[GenerationRequest] = []
    for index, item in enumerate(_normalize_requests(source)):
        if isinstance(item, GenerationRequest):
            requests.append(item)
            continue
        if isinstance(item, Mapping):
            row = dict(item)
        else:
            row = {"inputs": {"value": item}}
        row.setdefault("sample_id", _sample_id_from(item, index))
        row.setdefault("task_name", _task_name_from(item))
        requests.append(GenerationRequest.from_dict(row))
    return requests


def _coerce_model_result(item: Any, request: GenerationRequest, model_id: str) -> GenerationResult:
    """Coerces dictionaries or generic model responses securely into tracked `GenerationResult`s.

    Ensures that the `GenerationResult` is correctly associated with its originating `GenerationRequest`
    and contains necessary metadata like `model_id`.

    Args:
        item: The raw result object or dictionary from the model runner.
        request: The original `GenerationRequest` for context.
        model_id: The ID of the model that produced the result.

    Returns:
        A `GenerationResult` object.
    """
    if isinstance(item, GenerationResult):
        if item.sample_id == request.sample_id:
            return item
        # If result's sample_id doesn't match request's, prioritize request's but preserve result's metadata.
        return GenerationResult(
            sample_id=request.sample_id,
            request_id=item.request_id or request.request_id,
            model_id=item.model_id or model_id,
            artifacts=item.artifacts,
            status=item.status,
            error=item.error,
            timings=item.timings,
            metadata={**dict(item.metadata), "source_sample_id": item.sample_id},
        )
    if isinstance(item, Mapping):
        row = dict(item)
        row.setdefault("sample_id", request.sample_id)
        row.setdefault("request_id", request.request_id)
        row.setdefault("model_id", model_id)
        return GenerationResult.from_dict(row)
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id=model_id,
        status="failed",
        error=f"runner returned unsupported result type: {type(item).__name__}",
    )


def _align_model_results(
    requests: Sequence[GenerationRequest],
    raw_results: Sequence[Any],
    *,
    model_id: str,
) -> list[GenerationResult]:
    """Matches unordered runner outputs securely back to their originating requests.

    Attempts key-based lookups (`sample_id`) first to survive shuffling or asynchronous
    batch variations, falling back to positional zipping if no IDs were echoed.

    Args:
        requests: The sequence of original `GenerationRequest` objects.
        raw_results: The sequence of raw results returned by the model runner.
        model_id: The ID of the model that produced the results.

    Returns:
        A list of `GenerationResult` objects, aligned with the input requests.
    """
    keyed: dict[str, Any] = {}
    sequential: list[Any] = []
    # First pass: try to key results by sample_id
    for item in raw_results:
        sample_id = getattr(item, "sample_id", None)
        if sample_id is None and isinstance(item, Mapping):
            sample_id = item.get("sample_id")
        if sample_id is None:
            sequential.append(item)
        else:
            keyed[str(sample_id)] = item

    results: list[GenerationResult] = []
    sequential_index = 0
    # Second pass: align requests with results
    for request in requests:
        if request.sample_id in keyed:
            # Found a keyed result
            results.append(_coerce_model_result(keyed[request.sample_id], request, model_id))
        elif sequential_index < len(sequential):
            # Fallback to positional matching for unkeyed results
            results.append(_coerce_model_result(sequential[sequential_index], request, model_id))
            sequential_index += 1
        else:
            # If no result found (keyed or sequential), generate a failed result
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    model_id=model_id,
                    status="failed",
                    error="runner did not return a result for sample",
                )
            )
    return results


def _generate_with_resolved_model(
    resolved: Any,
    requests: Sequence[GenerationRequest],
    *,
    cleanup: bool,
) -> list[GenerationResult]:
    """Safe runner sandbox enforcing the Generation Protocol.

    Invokes `runner.generate()` inside a fault-isolated try/except block. Ensures that
    crashes or OOM exceptions return structured failed `GenerationResult`s rather than
    crashing the evaluation loop. Reaps runner resources automatically if cleanup is requested.

    Args:
        resolved: The resolved model runner object.
        requests: The sequence of `GenerationRequest` objects to process.
        cleanup: Whether to call the runner's `cleanup` method after generation.

    Returns:
        A list of `GenerationResult` objects, corresponding to the input requests.
    """
    runner = resolved.runner
    model_id = str(getattr(runner, "model_id", resolved.model_id))
    try:
        raw_results = list(runner.generate(requests))
    except Exception as exc:  # noqa: BLE001 - evaluate isolates runner failures.
        # If the runner itself fails, return a list of failed results for all requests.
        return [
            GenerationResult(
                sample_id=request.sample_id,
                request_id=request.request_id,
                model_id=model_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            for request in requests
        ]
    finally:
        # Ensure cleanup is performed if requested, even if generation fails.
        if cleanup:
            cleanup_fn = getattr(runner, "cleanup", None)
            if callable(cleanup_fn):
                cleanup_fn()
    return _align_model_results(requests, raw_results, model_id=model_id)


def _mode(value: str) -> str:
    """Standardizes string abbreviations into evaluated mode indicators ('existing-results' or 'model').

    Args:
        value: The raw mode string (e.g., "model", "existing", "generate").

    Returns:
        The normalized mode string ("existing-results" or "model").

    Raises:
        ValueError: If an unsupported mode value is provided.
    """
    normalized = value.replace("_", "-").lower()
    aliases = {
        "existing": "existing-results",
        "existing-result": "existing-results",
        "existing-results": "existing-results",
        "results": "existing-results",
        "model": "model",
        "generate": "model",
        "resolved-model": "model",
        "resolved": "model",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported evaluate mode: {value}")
    return aliases[normalized]


def _mapping_or_default(value: Mapping[str, Any] | Any | None, default: Mapping[str, Any]) -> dict[str, Any]:
    """Coerces objects or dictionaries safely into dictionary format, applying standard defaults.

    Args:
        value: The input object, which can be a mapping, an object with a `to_dict` method, or None.
        default: A mapping of default values to apply if `value` is None or cannot be coerced.

    Returns:
        A dictionary representation of the value, with defaults applied.
    """
    if value is None:
        return dict(default)
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return {"value": repr(value), **dict(default)}


def _benchmark_metadata(run_request: EvaluateRunRequest, mode: str) -> dict[str, Any]:
    """Compiles benchmark metadata context for the run manifest, applying defaults as needed.

    Args:
        run_request: The `EvaluateRunRequest` containing benchmark configuration.
        mode: The evaluation mode ("existing-results" or "model").

    Returns:
        A dictionary containing normalized benchmark metadata.
    """
    default = {
        "suite": "worldfoundry",
        "benchmark_name": run_request.benchmark_id or f"evaluate_{mode}",
        "task_type": f"evaluate_{mode}",
        "evaluation_protocol": mode,
    }
    payload = _mapping_or_default(run_request.benchmark, default)
    payload.setdefault("suite", "worldfoundry")
    payload.setdefault("benchmark_name", run_request.benchmark_id or f"evaluate_{mode}")
    payload.setdefault("task_type", f"evaluate_{mode}")
    payload.setdefault("evaluation_protocol", mode)
    return payload


def _model_metadata(run_request: EvaluateRunRequest) -> dict[str, Any]:
    """Constructs model evaluation metadata dictionary for materialized runs.

    This is used when a model is not actively run, but results are loaded from an
    existing source, hence the 'materialized' type.

    Args:
        run_request: The `EvaluateRunRequest` containing model configuration.

    Returns:
        A dictionary containing normalized model metadata.
    """
    payload = _mapping_or_default(
        run_request.model,
        {"model_type": "materialized", "model_name": run_request.model_id or "materialized-results"},
    )
    if run_request.model_id:
        payload.setdefault("model_id", run_request.model_id)
        payload.setdefault("model_name", run_request.model_id)
    return payload


def _resolved_model_metadata(run_request: EvaluateRunRequest, resolved: Any) -> dict[str, Any]:
    """Captures complex metadata and classes from active, instantiated model runners.

    This is used when an active model runner is instantiated, providing detailed
    information about the resolved model and its resolver.

    Args:
        run_request: The `EvaluateRunRequest` containing model configuration.
        resolved: The `_ResolvedRunner` object.

    Returns:
        A dictionary containing normalized model metadata, including resolver details.
    """
    payload = _mapping_or_default(
        run_request.model,
        {"model_type": "resolved", "model_name": str(resolved.model_id)},
    )
    payload["model_id"] = str(resolved.model_id)
    payload.setdefault("model_name", str(resolved.model_id))
    payload.setdefault("model_type", "resolved")
    to_dict = getattr(resolved, "to_dict", None)
    if callable(to_dict):
        payload.setdefault("resolver", to_dict())
    return payload


def _dataset_metadata(run_request: EvaluateRunRequest, sample_count: int) -> dict[str, Any]:
    """Constructs dataset context metadata (shards, splits, item counts) for execution records.

    Args:
        run_request: The `EvaluateRunRequest` containing dataset configuration.
        sample_count: The number of samples in the dataset.

    Returns:
        A dictionary containing normalized dataset metadata.
    """
    payload = _mapping_or_default(
        run_request.dataset,
        {"name": run_request.dataset_id or "materialized-results", "split": "materialized"},
    )
    if run_request.dataset_id:
        payload.setdefault("dataset_id", run_request.dataset_id)
        payload.setdefault("name", run_request.dataset_id)
    payload.setdefault("sample_count", sample_count)
    return payload


class _MetricObjectsCallable:
    """Wrapper encapsulating a suite of Metric objects into a single cohesive callable.

    This class allows a list of individual `Metric` objects to be treated as a single
    metric callable, suitable for contexts expecting a single metric computation function.
    """
    def __init__(self, metrics: Sequence[Metric]) -> None:
        """Initializes the wrapper with a sequence of Metric objects.

        Args:
            metrics: A sequence of `Metric` objects to be wrapped.
        """
        self.metrics = tuple(metrics)

    def __call__(self, request: GenerationRequest, result: GenerationResult) -> list[Any]:
        """Computes sample results for all wrapped metrics.

        Args:
            request: The `GenerationRequest` for the sample.
            result: The `GenerationResult` for the sample.

        Returns:
            A list of sample results, one for each wrapped metric.
        """
        return [metric.compute_sample(request, result) for metric in self.metrics]


def _is_metric_object(value: Any) -> bool:
    """Checks if an arbitrary object satisfies the standard programmatic Metric protocol interface.

    An object is considered a metric object if it's not a string and has callable
    `compute_sample` and `aggregate` methods.

    Args:
        value: The object to check.

    Returns:
        True if the object conforms to the `Metric` interface, False otherwise.
    """
    return (
        not isinstance(value, str)
        and callable(getattr(value, "compute_sample", None))
        and callable(getattr(value, "aggregate", None))
    )


def _partition_metrics(metrics: Sequence[Any]) -> tuple[tuple[str, ...], tuple[Metric, ...]]:
    """Splits an input sequence of metrics into flat string IDs and structural Metric objects.

    Args:
        metrics: A sequence of metric identifiers (strings) or `Metric` objects.

    Returns:
        A tuple containing two tuples:
        - The first tuple contains all string metric IDs.
        - The second tuple contains all `Metric` objects.

    Raises:
        TypeError: If an item in `metrics` is neither a string nor a valid `Metric` object.
    """
    metric_ids: list[str] = []
    metric_objects: list[Metric] = []
    for metric in metrics or ():
        if isinstance(metric, str):
            metric_ids.append(metric)
        elif _is_metric_object(metric):
            metric_objects.append(metric)
        else:
            raise TypeError(
                "evaluate metrics must be built-in metric ids or objects implementing "
                "compute_sample(request, result) and aggregate(results)"
            )
    return tuple(metric_ids), tuple(metric_objects)


def _ensure_single_metric_mode(
    metrics: Sequence[Any],
    required_artifacts: Sequence[str],
) -> tuple[tuple[str, ...], tuple[Metric, ...]]:
    """Enforces exclusive metric constraints, throwing if string IDs are mixed with raw Metric objects.

    This ensures that a run uses either string-based metric IDs (which can implicitly resolve)
    or explicitly provided `Metric` objects, but not both. Also, `required_artifacts` are
    only supported with string-based metrics.

    Args:
        metrics: A sequence of metric identifiers or objects.
        required_artifacts: A sequence of artifact names required by metrics.

    Returns:
        A tuple containing partitioned metric IDs and metric objects.

    Raises:
        TypeError: If metric IDs and objects are mixed, or if `required_artifacts` are
                   used with explicit `Metric` objects.
    """
    metric_ids, metric_objects = _partition_metrics(metrics)
    if metric_ids and metric_objects:
        raise TypeError("evaluate metrics cannot mix built-in metric ids and Metric objects in one run")
    if metric_objects and required_artifacts:
        raise TypeError("required_artifacts are only supported with built-in metric ids")
    return metric_ids, metric_objects


def _metric_callable(
    metrics: Sequence[Any],
    required_artifacts: Sequence[str],
    benchmark_id: str | None = None,
) -> BuiltinExistingResultsMetric | ExistingResultsMetric | None:
    """Resolves and returns the proper callable metric handler depending on config modes.

    If explicit `Metric` objects are provided, it returns a `_MetricObjectsCallable`.
    If string `metric_ids` are provided and match a known benchmark, it uses `BenchmarkZooInTreeEvaluator`.
    Otherwise, it uses `create_existing_results_metric` for general built-in metrics.

    Args:
        metrics: A sequence of metric identifiers or objects.
        required_artifacts: A sequence of artifact names required by metrics.
        benchmark_id: An optional ID for the benchmark being evaluated.

    Returns:
        A callable metric handler (`BuiltinExistingResultsMetric`, `ExistingResultsMetric`,
        or `_MetricObjectsCallable`), or None if no metrics are specified.
    """
    metric_ids, metric_objects = _ensure_single_metric_mode(metrics, required_artifacts)
    if metric_objects:
        return _MetricObjectsCallable(metric_objects)
    if benchmark_id is not None and benchmark_id in target_benchmark_metrics():
        target_metrics = target_benchmark_metrics()[benchmark_id]
        if metric_ids and all(metric_id in target_metrics for metric_id in metric_ids):
            # If benchmark ID is known and all requested metrics are in-tree for it, use the specialized evaluator.
            return BenchmarkZooInTreeEvaluator(
                benchmark_id,
                metric_ids=metric_ids,
                required_artifacts=required_artifacts or None,
            )
    # Default to generic existing results metric if no specific benchmark match or explicit objects.
    return create_existing_results_metric(metrics=metric_ids, required_artifacts=required_artifacts)


def _contract_metric_objects(metrics: Sequence[Any], required_artifacts: Sequence[str]) -> tuple[Metric, ...]:
    """Ensures safe Metric object validation for Live model execution runs.

    This function is specifically for 'model' mode when `ContractRunner` is used,
    which requires explicit `Metric` objects rather than string IDs.

    Args:
        metrics: A sequence of metric identifiers or objects.
        required_artifacts: A sequence of artifact names required by metrics.

    Returns:
        A tuple of `Metric` objects.

    Raises:
        TypeError: If any string metric IDs are present or `required_artifacts` are specified.
    """
    _, metric_objects = _ensure_single_metric_mode(metrics, required_artifacts)
    return metric_objects


def _benchmark_id_from_metadata(run_request: EvaluateRunRequest, benchmark: Mapping[str, Any]) -> str | None:
    """Discovers a clean, normalized benchmark_id string from various metadata properties.

    Prioritizes `run_request.benchmark_id`, then `benchmark["benchmark_id"]`, then `benchmark["benchmark_name"]`.

    Args:
        run_request: The `EvaluateRunRequest` object.
        benchmark: The compiled benchmark metadata dictionary.

    Returns:
        A normalized string benchmark ID, or None if not found.
    """
    item = run_request.benchmark_id or benchmark.get("benchmark_id") or benchmark.get("benchmark_name")
    return None if item is None else str(item).strip().lower()


def _resolved_from_runner(runner: Any, model_id: str | None) -> _ResolvedRunner:
    """Safely wraps an explicitly instantiated model runner into a structured _ResolvedRunner context.

    Args:
        runner: The instantiated model runner object.
        model_id: An optional explicit model ID to associate with the runner.

    Returns:
        A `_ResolvedRunner` object encapsulating the runner and its metadata.
    """
    resolved_model_id = str(model_id or getattr(runner, "model_id", "") or "provided-runner")
    return _ResolvedRunner(
        runner=runner,
        model_id=resolved_model_id,
        diagnostics={"explicit_runner": True},
    )


def _coerce_request(
    request: EvaluateRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> EvaluateRunRequest:
    """Merges and safely forces arbitrary parameters into a structured EvaluateRunRequest.

    If `request` is already an `EvaluateRunRequest`, it updates it with `kwargs`.
    Otherwise, it constructs a new `EvaluateRunRequest` from `request` (if a mapping) and `kwargs`.

    Args:
        request: An optional `EvaluateRunRequest` object or a mapping of parameters.
        kwargs: Additional keyword arguments to apply or override request parameters.

    Returns:
        A fully formed `EvaluateRunRequest` object.

    Raises:
        TypeError: If `output_dir` is not specified in the merged parameters.
    """
    if isinstance(request, EvaluateRunRequest):
        if kwargs:
            payload = asdict(request)
            payload.update(kwargs)
            return EvaluateRunRequest(**payload)
        return request

    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    if "output_dir" not in payload:
        raise TypeError("run_evaluate requires an output_dir")
    return EvaluateRunRequest(**payload)


def _result_from_delegate(mode: str, delegate_runner: str, result: Any) -> EvaluateRunResult:
    """Translates subsystem results (e.g. ExistingResultsRunner or ContractRunner) into the global schema.

    Converts the result from a specific delegated runner into the common `EvaluateRunResult` format.

    Args:
        mode: The evaluation mode ("existing-results" or "model").
        delegate_runner: A string identifying the specific runner that handled the execution.
        result: The result object returned by the delegated runner (e.g., `ExistingResultsRunResult`).

    Returns:
        An `EvaluateRunResult` object.
    """
    return EvaluateRunResult(
        schema_version=EVALUATE_RUN_RESULT_SCHEMA_VERSION,
        mode=mode,
        delegate_runner=delegate_runner,
        status=result.status,
        exit_code=result.exit_code,
        output_dir=result.output_dir,
        manifest_path=result.manifest_path,
        execution_plan_path=result.execution_plan_path,
        scorecard_path=result.scorecard_path,
        sample_count=result.sample_count,
        successful_sample_count=result.successful_sample_count,
        failed_sample_count=result.failed_sample_count,
        artifact_count=result.artifact_count,
    )


def run_evaluate(
    request: EvaluateRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> EvaluateRunResult:
    """Core evaluation pipeline entry point handling offline metrics and live model simulation.

    Dispatches to two main execution modes:
    1. "existing-results": Does not instantiate a model, but rather processes previously cached or
       externally generated results, applying the metrics suite and generating a scorecard.
    2. "model": Dynamically resolves the target `WorldModelRunner` (via registry or manifest),
       routes a batch of `GenerationRequest` instances through it (potentially utilizing the
       smart generation cache to resume/skip completed tasks), evaluates the generated outputs,
       and builds the final leaderboard scorecard.

    Args:
        request: An optional `EvaluateRunRequest` object or a mapping of parameters.
        **kwargs: Additional keyword arguments to construct or override `EvaluateRunRequest` parameters.

    Returns:
        An `EvaluateRunResult` object summarizing the evaluation execution and artifact paths.

    Raises:
        ValueError: If an unsupported `EvaluateRunRequest` schema version or evaluation mode is encountered.
        TypeError: If required parameters (like `output_dir`, `requests`, or `results`) are missing
                   for a given mode.
    """
    run_request = _coerce_request(request, kwargs)
    if run_request.schema_version != EVALUATE_RUN_REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported EvaluateRunRequest schema_version: {run_request.schema_version}")

    mode = _mode(run_request.mode)
    output_dir = Path(run_request.output_dir)

    if mode == "existing-results":
        # Handle 'existing-results' mode: process pre-generated results with metrics.
        results_source = _load_optional_source(run_request.results, run_request.results_path)
        if results_source is None:
            raise TypeError("existing-results mode requires results or results_path")
        request_source = _load_optional_source(run_request.requests, run_request.requests_path)
        # Derive requests either from explicit source or from the results themselves.
        requests = (
            _normalize_requests(request_source)
            if request_source is not None
            else _derive_requests_from_results(results_source)
        )
        benchmark = _benchmark_metadata(run_request, mode)
        benchmark_id = _benchmark_id_from_metadata(run_request, benchmark)
        
        # Route to the offline evaluation delegator
        delegate = execute_existing_results(
            ExistingResultsRunRequest(
                output_dir=output_dir,
                requests=requests,
                results=results_source,
                metric=_metric_callable(run_request.metrics, run_request.required_artifacts, benchmark_id),
                benchmark=benchmark,
                model=_model_metadata(run_request),
                dataset=_dataset_metadata(run_request, len(requests)),
                run_id=run_request.run_id,
                fail_on_sample_error=run_request.fail_on_sample_error,
                write_artifacts_index=run_request.write_artifacts_index,
                run_metadata=run_request.run_metadata,
            )
        )
        return _result_from_delegate(mode, "ExistingResultsRunner", delegate)

    if mode == "model":
        # Handle 'model' mode: instantiate and run a model, then evaluate outputs.
        from worldfoundry.evaluation.models import resolve_model_zoo_runner, resolve_world_model_runner

        request_source = _load_optional_source(run_request.requests, run_request.requests_path)
        if request_source is None:
            raise TypeError("model mode requires requests, requests_path, or materialized task requests")
        requests = _coerce_generation_requests(request_source)
        
        # Resolve dynamic runner class dependencies based on configuration (explicit runner, model zoo, or global registry).
        if run_request.runner is not None:
            resolved = _resolved_from_runner(run_request.runner, run_request.model_id)
        elif run_request.model_zoo_manifest_dir is not None:
            if not run_request.model_id:
                raise TypeError("model-zoo model mode requires model_id")
            resolved = resolve_model_zoo_runner(
                run_request.model_id,
                manifest_dir=run_request.model_zoo_manifest_dir,
                variant_id=run_request.model_variant_id,
                parameters=run_request.model_parameters,
                runtime=run_request.model_runtime,
            )
        else:
            resolved = resolve_world_model_runner(
                run_request.model_id,
                runner=run_request.model_runner,
                parameters=run_request.model_parameters,
                runtime=run_request.model_runtime,
                config=run_request.model_config,
            )

        contract_metrics = _contract_metric_objects(run_request.metrics, run_request.required_artifacts)
        benchmark = _benchmark_metadata(run_request, mode)
        dataset = _dataset_metadata(run_request, len(requests))
        model = _resolved_model_metadata(run_request, resolved)
        
        if contract_metrics:
            # If explicit Metric objects are provided, use the ContractRunner for live metric computation.
            delegate = execute_contract_run(
                ContractRunRequest(
                    output_dir=output_dir,
                    requests=requests,
                    runner=resolved.runner,
                    metrics=contract_metrics,
                    benchmark=benchmark,
                    model=model,
                    dataset=dataset,
                    run_id=run_request.run_id,
                    fail_on_sample_error=run_request.fail_on_sample_error,
                    write_artifacts_index=run_request.write_artifacts_index,
                    cleanup_runner=run_request.cleanup_runner,
                    generation_cache_dir=run_request.generation_cache_dir,
                    generation_cache_mode=run_request.generation_cache_mode,
                    generation_cache_namespace=run_request.generation_cache_namespace,
                )
            )
            return _result_from_delegate(mode, "ResolvedWorldModelRunner+ContractRunner", delegate)

        # If no explicit Metric objects, perform generation (with caching) and then evaluate offline.
        version_context = build_version_context(
            runner="evaluate_model_generation",
            benchmark=benchmark,
            model=model,
            dataset=dataset,
            model_runner=resolved.runner,
            extra={"mode": "model"},
        )
        try:
            # Run generation using the resolved model runner, incorporating smart caching.
            results, generation_cache_stats = run_generation_with_cache(
                requests,
                lambda rows: _generate_with_resolved_model(resolved, rows, cleanup=False),
                cache_dir=run_request.generation_cache_dir,
                cache_mode=run_request.generation_cache_mode,
                namespace=run_request.generation_cache_namespace,
                version_context=version_context,
                artifact_base_dir=output_dir,
                run_id=run_request.run_id,
            )
        finally:
            # Ensure model runner cleanup happens even if generation fails.
            if run_request.cleanup_runner:
                cleanup_fn = getattr(resolved.runner, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
        # After generation, delegate to the existing results runner for metric calculation and scorecard generation.
        delegate = execute_existing_results(
            ExistingResultsRunRequest(
                output_dir=output_dir,
                requests=requests,
                results=results,
                metric=_metric_callable(
                    run_request.metrics,
                    run_request.required_artifacts,
                    _benchmark_id_from_metadata(run_request, benchmark),
                ),
                benchmark=benchmark,
                model=model,
                dataset=dataset,
                run_id=run_request.run_id,
                fail_on_sample_error=run_request.fail_on_sample_error,
                write_artifacts_index=run_request.write_artifacts_index,
                run_metadata={
                    **dict(run_request.run_metadata or {}),
                    "generation_cache": generation_cache_stats.to_dict(),
                },
                cache_paths=cache_paths_from_stats(generation_cache_stats),
            )
        )
        return _result_from_delegate(mode, "ResolvedWorldModelRunner+ExistingResultsRunner", delegate)

    raise ValueError(f"unsupported evaluate mode after normalization: {mode!r}")


execute_evaluate_run = run_evaluate # Alias for backward compatibility or convenience.

__all__ = [
    "EVALUATE_RUN_REQUEST_SCHEMA_VERSION",
    "EVALUATE_RUN_RESULT_SCHEMA_VERSION",
    "BuiltinExistingResultsMetric",
    "EvaluateRunRequest",
    "EvaluateRunResult",
    "execute_evaluate_run",
    "run_evaluate",
]
