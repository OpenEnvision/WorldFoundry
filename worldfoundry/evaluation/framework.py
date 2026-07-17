"""Public entrypoint and dispatch logic for running WorldFoundry benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR

from .runner import (
    EvaluateRunRequest,
    ModelBenchmarkRunRequest,
    ModelBenchmarkSuiteRequest,
    execute_evaluate_run,
    run_model_benchmark,
    run_model_benchmark_suite,
)


WORLDFOUNDRY_RUN_RESULT_SCHEMA_VERSION = "worldfoundry-run-result"


@dataclass(frozen=True)
class WorldFoundryRunRequest:
    """One public run request for existing results, one benchmark, or a model x benchmark suite."""

    output_dir: str | Path
    model_ids: Sequence[str] = ()
    benchmark_ids: Sequence[str] = ()
    suite_ids: Sequence[str] = ()
    all_benchmarks: bool = False
    benchmark_id: str | None = None
    benchmark_manifest_dir: str | Path = BENCHMARK_ZOO_DIR
    model_manifest_dir: str | Path | None = MODEL_ZOO_DIR
    suite_preset_path: str | Path | None = None
    engine: str = "auto"
    benchmark_mode: str = "official-run"
    execute: bool = True
    resume: bool = False
    skip_incompatible: bool = True
    fail_on_skipped: bool = False
    model_runner: str | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    requests_path: str | Path | None = None
    results_path: str | Path | None = None
    task_name: str | None = None
    task_roots: Sequence[str | Path] | None = None
    task_benchmark: str | None = None
    task_recursive: bool = False
    task_root_dir: str | Path | None = None
    dataset_root: str | Path | None = None
    dataset_id: str | None = None
    split: str = "default"
    num_samples: int | None = None
    generated_artifact_dir: str | Path | None = None
    output_artifact: str | None = None
    required_artifacts: Sequence[str] | None = None
    metrics: Sequence[str] = ("artifact_count", "required_artifacts_present")
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "worldfoundry_run"
    benchmark_timeout_seconds: float | None = None
    benchmark_workdir: str | Path | None = None
    benchmark_env: Mapping[str, Any] | None = None
    benchmark_parameters: Mapping[str, Any] | None = None
    materialize_placeholders: bool | None = None
    contract_fixture: bool = False
    fail_on_generation_error: bool = False
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True


@dataclass(frozen=True)
class WorldFoundryRunResult:
    """The result of a completed or failed WorldFoundry run, encapsulating execution details and artifacts."""

    schema_version: str
    kind: str
    status: str
    exit_code: int
    output_dir: Path
    delegate: Any

    @property
    def ok(self) -> bool:
        """Return True if the run completed successfully with exit code 0."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the run result and its delegate details into a serializable dictionary."""
        delegate_payload = _delegate_payload(self.delegate)
        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "output_dir": str(self.output_dir),
            "delegate": delegate_payload,
        }
        payload.update(_delegate_artifact_summary(delegate_payload))
        return payload


def _coerce_request(
    request: WorldFoundryRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> WorldFoundryRunRequest:
    """Coerce input request or keyword arguments into a unified WorldFoundryRunRequest instance."""
    if isinstance(request, WorldFoundryRunRequest):
        if not kwargs:
            return request
        payload = asdict(request)
        payload.update(kwargs)
        return WorldFoundryRunRequest(**payload)
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    return WorldFoundryRunRequest(**payload)


def _delegate_payload(value: Any) -> Any:
    """Convert the delegate result object into a serializable representation."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": repr(value)}


def _first_text(*values: Any) -> str | None:
    """Return the string representation of the first non-empty value among the arguments."""
    for value in values:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str) and value:
            return value
    return None


def _delegate_artifact_summary(delegate_payload: Any) -> dict[str, Any]:
    """Extract and consolidate paths to critical run manifests, scorecards, and reports from delegates."""
    if not isinstance(delegate_payload, Mapping):
        return {}

    artifacts = delegate_payload.get("artifacts")
    artifact_map = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    generation = delegate_payload.get("generation_result")
    generation_map = generation if isinstance(generation, Mapping) else {}
    benchmark = delegate_payload.get("benchmark_result")
    benchmark_map = benchmark if isinstance(benchmark, Mapping) else {}

    summary: dict[str, Any] = {}
    run_manifest_path = _first_text(
        delegate_payload.get("run_manifest_path"),
        delegate_payload.get("manifest_path"),
        delegate_payload.get("suite_manifest_path"),
        artifact_map.get("run_manifest"),
        artifact_map.get("standard_run_manifest"),
    )
    scorecard_path = _first_text(
        delegate_payload.get("scorecard_path"),
        benchmark_map.get("scorecard_path"),
        artifact_map.get("benchmark_scorecard"),
        artifact_map.get("scorecard"),
        generation_map.get("scorecard_path"),
        artifact_map.get("generation_scorecard"),
    )
    generation_scorecard_path = _first_text(generation_map.get("scorecard_path"), artifact_map.get("generation_scorecard"))
    benchmark_scorecard_path = _first_text(benchmark_map.get("scorecard_path"), artifact_map.get("benchmark_scorecard"))
    artifact_manifest_path = _first_text(delegate_payload.get("artifact_manifest_path"), artifact_map.get("generated_artifact_manifest"))
    suite_manifest_path = _first_text(delegate_payload.get("suite_manifest_path"))
    suite_report_path = _first_text(delegate_payload.get("suite_report_path"))

    for key, value in (
        ("run_manifest_path", run_manifest_path),
        ("scorecard_path", scorecard_path),
        ("generation_scorecard_path", generation_scorecard_path),
        ("benchmark_scorecard_path", benchmark_scorecard_path),
        ("artifact_manifest_path", artifact_manifest_path),
        ("suite_manifest_path", suite_manifest_path),
        ("suite_report_path", suite_report_path),
    ):
        if value is not None:
            summary[key] = value
    if artifact_map:
        summary["artifacts"] = artifact_map
    return summary


def _result(kind: str, delegate: Any) -> WorldFoundryRunResult:
    """Construct a WorldFoundryRunResult from a delegate runner output."""
    return WorldFoundryRunResult(
        schema_version=WORLDFOUNDRY_RUN_RESULT_SCHEMA_VERSION,
        kind=kind,
        status=str(getattr(delegate, "status", "unknown")),
        exit_code=int(getattr(delegate, "exit_code", 1)),
        output_dir=Path(getattr(delegate, "output_dir", ".")),
        delegate=delegate,
    )


def _model_ids(request: WorldFoundryRunRequest, *, default: Sequence[str] | None = None) -> tuple[str, ...]:
    """Resolve and canonicalize model identifiers specified in the run request."""
    if default is None:
        default = ()
    values = tuple(str(item) for item in request.model_ids if str(item).strip())
    if values:
        return tuple(_canonical_model_id_or_self(item, request.model_manifest_dir) for item in values)
    return tuple(default)


def _benchmark_ids(request: WorldFoundryRunRequest) -> tuple[str, ...]:
    """Resolve and canonicalize benchmark identifiers specified in the run request."""
    values = tuple(str(item) for item in request.benchmark_ids if str(item).strip())
    if values:
        return tuple(_canonical_benchmark_id_or_self(item, request.benchmark_manifest_dir) for item in values)
    if request.all_benchmarks:
        from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids

        return tuple(formal_benchmark_ids(request.benchmark_manifest_dir))
    return ()


def _suite_ids(request: WorldFoundryRunRequest) -> tuple[str, ...]:
    """Retrieve explicitly requested legacy suite preset identifiers."""
    return tuple(str(item) for item in request.suite_ids if str(item).strip())


def _canonical_model_id_or_self(value: str, manifest_dir: str | Path | None) -> str:
    """Lookup and return the canonical model ID from the model-zoo registry, falling back to original value."""
    if manifest_dir is None:
        return value
    root = Path(manifest_dir)
    if not root.exists() or not root.is_dir():
        return value
    try:
        from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

        return load_model_zoo_registry(root).get(value).model_id
    except (KeyError, TypeError, ValueError):
        return value


def _canonical_benchmark_id_or_self(value: str, manifest_dir: str | Path | None) -> str:
    """Lookup and return the canonical benchmark ID from the benchmark-zoo registry, falling back to original value."""
    if manifest_dir is None:
        return value
    root = Path(manifest_dir)
    if not root.exists():
        return value
    try:
        from worldfoundry.evaluation.tasks.catalog.schema import load_entries
        from worldfoundry.evaluation.tasks.catalog.zoo_registry import (
            BenchmarkZooRegistry,
            load_benchmark_zoo_registry,
        )

        if root.is_file():
            return BenchmarkZooRegistry(load_entries(root)).get(value).benchmark_id
        return load_benchmark_zoo_registry(root).get(value).benchmark_id
    except (KeyError, TypeError, ValueError):
        return value


def _single_required_artifacts(request: WorldFoundryRunRequest) -> tuple[str, ...]:
    """Determine the sequence of required artifacts for a single benchmark run."""
    if request.required_artifacts is not None:
        return tuple(str(item) for item in request.required_artifacts)
    if request.output_artifact:
        return (str(request.output_artifact),)
    return ()


def _runner_kwargs(request: WorldFoundryRunRequest) -> dict[str, Any]:
    """Extract model-runner configuration arguments from the run request."""
    return {
        "model_runner": request.model_runner,
        "model_variant_id": request.model_variant_id,
        "model_parameters": request.model_parameters,
        "model_runtime": request.model_runtime,
        "model_config": request.model_config,
    }


def _generation_kwargs(request: WorldFoundryRunRequest) -> dict[str, Any]:
    """Extract sample generation and caching options from the run request."""
    return {
        "requests_path": request.requests_path,
        "task_name": request.task_name,
        "task_roots": request.task_roots,
        "task_benchmark": request.task_benchmark,
        "task_recursive": request.task_recursive,
        "task_root_dir": request.task_root_dir,
        "dataset_root": request.dataset_root,
        "dataset_id": request.dataset_id,
        "split": request.split,
        "num_samples": request.num_samples,
        "generated_artifact_dir": request.generated_artifact_dir,
        "generation_cache_dir": request.generation_cache_dir,
        "generation_cache_mode": request.generation_cache_mode,
        "generation_cache_namespace": request.generation_cache_namespace,
    }


def _benchmark_execution_kwargs(request: WorldFoundryRunRequest) -> dict[str, Any]:
    """Extract benchmark execution and timeout options from the run request."""
    return {
        "metrics": tuple(request.metrics),
        "benchmark_timeout_seconds": request.benchmark_timeout_seconds,
        "benchmark_workdir": request.benchmark_workdir,
        "benchmark_env": request.benchmark_env,
        "benchmark_parameters": request.benchmark_parameters,
        "materialize_placeholders": request.materialize_placeholders,
        "contract_fixture": request.contract_fixture,
        "fail_on_generation_error": request.fail_on_generation_error,
        "run_id": request.run_id,
    }


def _model_manifest_dir_for(request: WorldFoundryRunRequest, model_id: str | None) -> str | Path | None:
    """Determine the model manifest catalog directory, returning None for checkpointless or custom runs."""
    if request.model_runner or request.model_config:
        return None
    try:
        from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import CONTRACT_VALIDATION_ID

        if model_id == CONTRACT_VALIDATION_ID:
            return None
    except Exception:  # noqa: BLE001 - fallback to normal model-zoo resolution.
        pass
    return request.model_manifest_dir


def _run_existing_or_model(request: WorldFoundryRunRequest) -> WorldFoundryRunResult:
    """Execute an in-process model generation or existing-results evaluation run."""
    mode = "model" if request.engine in {"model", "generate", "in-process"} else "existing-results"
    if request.results_path is not None:
        mode = "existing-results"
    model_ids = _model_ids(request)
    delegate = execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=request.output_dir,
            mode=mode,
            requests_path=request.requests_path,
            results_path=request.results_path,
            metrics=tuple(request.metrics or ("artifact_count",)),
            required_artifacts=tuple(request.required_artifacts or ()),
            benchmark_id=request.benchmark_id,
            model_id=model_ids[0] if model_ids else None,
            model_zoo_manifest_dir=_model_manifest_dir_for(request, model_ids[0] if model_ids else None),
            **_runner_kwargs(request),
            dataset_id=request.dataset_id,
            run_id=request.run_id,
            fail_on_sample_error=request.fail_on_sample_error,
            write_artifacts_index=request.write_artifacts_index,
            generation_cache_dir=request.generation_cache_dir,
            generation_cache_mode=request.generation_cache_mode,
            generation_cache_namespace=request.generation_cache_namespace,
        )
    )
    return _result("evaluate", delegate)


def _run_single_benchmark(request: WorldFoundryRunRequest, model_id: str, benchmark_id: str) -> WorldFoundryRunResult:
    """Execute a standard model benchmark run for a single model and benchmark combination."""
    output_artifact = request.output_artifact or "generated_video"
    delegate = run_model_benchmark(
        ModelBenchmarkRunRequest(
            output_dir=request.output_dir,
            benchmark_id=benchmark_id,
            benchmark_manifest_path=request.benchmark_manifest_dir,
            benchmark_mode=request.benchmark_mode,
            model_id=model_id,
            model_zoo_manifest_dir=_model_manifest_dir_for(request, model_id),
            **_runner_kwargs(request),
            **_generation_kwargs(request),
            output_artifact=output_artifact,
            required_artifacts=_single_required_artifacts(request) or (output_artifact,),
            **_benchmark_execution_kwargs(request),
        )
    )
    return _result("model-benchmark", delegate)


def _run_suite(request: WorldFoundryRunRequest, model_ids: Sequence[str], benchmark_ids: Sequence[str]) -> WorldFoundryRunResult:
    """Execute a multi-model or multi-benchmark evaluation suite."""
    delegate = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=request.output_dir,
            benchmark_manifest_dir=request.benchmark_manifest_dir,
            model_manifest_dir=request.model_manifest_dir,
            suite_ids=_suite_ids(request),
            suite_preset_path=request.suite_preset_path,
            model_ids=tuple(model_ids),
            benchmark_ids=tuple(benchmark_ids),
            mode=request.benchmark_mode,
            execute=request.execute,
            resume=request.resume,
            skip_incompatible=request.skip_incompatible,
            fail_on_skipped=request.fail_on_skipped,
            **_runner_kwargs(request),
            **_generation_kwargs(request),
            output_artifact=request.output_artifact,
            required_artifacts=tuple(request.required_artifacts) if request.required_artifacts is not None else None,
            **_benchmark_execution_kwargs(request),
        )
    )
    return _result("model-benchmark-suite", delegate)


def _should_run_suite(
    request: WorldFoundryRunRequest,
    model_ids: Sequence[str],
    benchmark_ids: Sequence[str],
) -> bool:
    """Determine whether the requested parameters necessitate executing a multi-benchmark suite."""
    if _suite_ids(request) or len(model_ids) > 1 or len(benchmark_ids) > 1:
        return True
    return bool(benchmark_ids) and not request.execute


def _suite_model_ids(request: WorldFoundryRunRequest, model_ids: Sequence[str], benchmark_ids: Sequence[str]) -> tuple[str, ...]:
    """Resolve the model identifiers to evaluate within a suite run."""
    from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import CONTRACT_VALIDATION_ID

    if model_ids:
        return tuple(model_ids)
    if request.contract_fixture and request.all_benchmarks:
        return (CONTRACT_VALIDATION_ID,)
    if request.contract_fixture and benchmark_ids and not _suite_ids(request):
        return (CONTRACT_VALIDATION_ID,)
    return ()


def run_worldfoundry(
    request: WorldFoundryRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> WorldFoundryRunResult:
    """Dispatch the public WorldFoundry run surface to the narrow canonical runners."""

    run_request = _coerce_request(request, kwargs)
    explicit_model_ids = _model_ids(run_request, default=())
    benchmark_ids = _benchmark_ids(run_request)

    if _should_run_suite(run_request, explicit_model_ids, benchmark_ids):
        return _run_suite(run_request, _suite_model_ids(run_request, explicit_model_ids, benchmark_ids), benchmark_ids)
    if benchmark_ids:
        if explicit_model_ids:
            model_ids = explicit_model_ids
        elif run_request.contract_fixture:
            from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import CONTRACT_VALIDATION_ID

            model_ids = (CONTRACT_VALIDATION_ID,)
        else:
            raise ValueError(
                "model-benchmark runs require --model."
            )
        return _run_single_benchmark(run_request, model_ids[0], benchmark_ids[0])
    return _run_existing_or_model(run_request)


__all__ = [
    "WORLDFOUNDRY_RUN_RESULT_SCHEMA_VERSION",
    "WorldFoundryRunRequest",
    "WorldFoundryRunResult",
    "run_worldfoundry",
]
