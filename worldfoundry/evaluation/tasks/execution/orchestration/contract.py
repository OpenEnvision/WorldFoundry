"""Generate contract results, then score through the shared offline runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from worldfoundry.evaluation.api import (
    GenerationRequest,
    GenerationResult,
    Metric,
    WorldModelRunner,
    is_generation_result_successful,
)
from worldfoundry.evaluation.reporting.run_manifest import redact_secrets
from worldfoundry.evaluation.utils import build_version_context, jsonable, model_runner_fingerprint, write_jsonl

from .cache import cache_paths_from_stats, run_generation_with_cache
from .existing_results import run_existing_results
from .run_session import (
    coerce_generation_result,
    prepare_generation,
    prepare_run_output_dir,
    resume_generation,
    run_stage,
    session_paths,
    utcnow_iso,
)
from .scoring import metric_owners


@dataclass(frozen=True)
class ContractRunRequest:
    """Inputs for generation followed by shared metric scoring."""

    output_dir: str | Path
    requests: Sequence[GenerationRequest | Mapping[str, Any]]
    runner: WorldModelRunner
    metrics: Sequence[Metric] = ()
    benchmark: Mapping[str, Any] | Any | None = None
    model: Mapping[str, Any] | Any | None = None
    dataset: Mapping[str, Any] | Any | None = None
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    cleanup_runner: bool = True
    resume: bool = False
    metric_batch_size: int = 32
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "contract_runner"


@dataclass(frozen=True)
class ContractRunResult:
    """Result paths and sample counts from a contract run."""

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


class ContractRunner:
    """Object interface for run_contract."""

    def run(
        self,
        request: ContractRunRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ContractRunResult:
        """Entrypoint for executing the localized pipeline against a bounded model."""
        return run_contract(request, **kwargs)


def _normalize_requests(source: Sequence[GenerationRequest | Mapping[str, Any]]) -> list[GenerationRequest]:
    """Coerce requests and fill the contract defaults."""
    requests: list[GenerationRequest] = []
    for index, item in enumerate(source):
        if isinstance(item, GenerationRequest):
            requests.append(item)
            continue
        row = dict(item)
        # Ensure sample_id and task_name are present for proper tracking
        row.setdefault("sample_id", f"sample-{index:04d}")
        row.setdefault("task_name", "contract_runner")
        requests.append(GenerationRequest.from_dict(row))
    return requests


def _is_failed(result: GenerationResult) -> bool:
    """Return whether generation failed."""
    return not is_generation_result_successful(result)


def _align_generation_results(
    requests: Sequence[GenerationRequest], raw_results: Sequence[Any]
) -> list[GenerationResult]:
    """Align keyed or ordered outputs, retaining missing results as failures."""
    keyed: dict[str, Any] = {}
    sequential: list[Any] = []
    # Separate raw results into keyed (by sample_id) and sequential (no sample_id)
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
    # Iterate through requests to find or create corresponding results
    for request in requests:
        if request.sample_id in keyed:
            results.append(coerce_generation_result(keyed[request.sample_id], request.sample_id))
        elif sequential_index < len(sequential):
            # Assign sequential results if no keyed match found
            results.append(coerce_generation_result(sequential[sequential_index], request.sample_id))
            sequential_index += 1
        else:
            # If no result found for a request, create a failed placeholder
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    status="failed",
                    error="runner did not return a result for sample",
                )
            )
    return results


def _generate(runner: WorldModelRunner, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
    """Run generation and retain model failures as sample results."""
    try:
        raw_results = runner.generate(requests)
    except Exception as exc:  # noqa: BLE001 - runner isolates model failures.
        # If the runner's generate method throws an exception,
        # return failed results for all requests
        return [
            GenerationResult(
                sample_id=request.sample_id,
                request_id=request.request_id,
                model_id=getattr(runner, "model_id", ""),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            for request in requests
        ]
    return _align_generation_results(requests, list(raw_results))


def _coerce_request(
    request: ContractRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> ContractRunRequest:
    """Merge a request with keyword overrides."""
    if isinstance(request, ContractRunRequest):
        if kwargs:
            # If kwargs are provided, merge them with the existing request
            return replace(request, **kwargs)
        return request

    # Start with kwargs, then overlay the request mapping if it exists
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}

    # Validate required fields
    if "output_dir" not in payload:
        raise TypeError("run_contract requires an output_dir")
    if "requests" not in payload:
        raise TypeError("run_contract requires requests")
    if "runner" not in payload:
        raise TypeError("run_contract requires a runner")

    return ContractRunRequest(**payload)


def run_contract(request: ContractRunRequest | Mapping[str, Any] | None = None, **kwargs: Any) -> ContractRunResult:
    """Reuse successful generation independently of versioned metric checkpoints."""
    run_request = _coerce_request(request, kwargs)
    if run_request.metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    metric_owners(run_request.metrics)
    output_dir = prepare_run_output_dir(run_request.output_dir)
    requests = _normalize_requests(run_request.requests)
    runner = run_request.runner
    model = jsonable(
        run_request.model or {"model_type": "contract", "model_name": getattr(runner, "model_id", "contract-model")}
    )
    identity = redact_secrets(
        jsonable(
            {
                "model": model,
                "runner": model_runner_fingerprint(runner),
            }
        )
    )
    paths = session_paths(output_dir)
    cached = resume_generation(paths, requests, identity) if run_request.resume else {}
    run_id = run_request.run_id or f"contract-{uuid4().hex[:12]}"
    generation_metadata = {
        "runner": "contract_runner", "generation_identity": identity, "run_id": run_id, "started_at": utcnow_iso(),
    }
    version_context = build_version_context(runner="contract_generation", model=model, model_runner=runner)
    with run_stage(paths, "generate", manifest={
        **generation_metadata, "model": model, "benchmark": jsonable(run_request.benchmark or {}),
        "dataset": jsonable(run_request.dataset or {}), "sample_count": len(requests),
    }):
        prepare_generation(paths, requests, cached)
        try:
            generated, stats = run_generation_with_cache(
                [row for row in requests if row.sample_id not in cached],
                lambda rows: _generate(runner, rows),
                cache_dir=run_request.generation_cache_dir,
                cache_mode=run_request.generation_cache_mode,
                namespace=run_request.generation_cache_namespace,
                version_context=version_context,
                artifact_base_dir=output_dir,
                run_id=run_id,
            )
        finally:
            if run_request.cleanup_runner:
                cleanup = getattr(runner, "cleanup", None)
                if callable(cleanup):
                    cleanup()
        results = {**cached, **{row.sample_id: row for row in generated}}
        write_jsonl(paths["results"], (results[row.sample_id].to_dict() for row in requests))
    result = run_existing_results(
        output_dir=output_dir,
        requests=requests,
        results=[results[row.sample_id] for row in requests],
        metrics=run_request.metrics,
        benchmark=run_request.benchmark,
        model=model,
        dataset=run_request.dataset,
        run_id=run_id,
        fail_on_sample_error=run_request.fail_on_sample_error,
        write_artifacts_index=run_request.write_artifacts_index,
        resume=run_request.resume,
        metric_batch_size=run_request.metric_batch_size,
        cache_paths=cache_paths_from_stats(stats),
        run_metadata={**generation_metadata, "evaluation_kind": "contract_runner", "generation_cache": stats.to_dict()},
    )
    return ContractRunResult(**asdict(result))


execute_contract_run = run_contract

__all__ = ["ContractRunRequest", "ContractRunResult", "ContractRunner", "execute_contract_run", "run_contract"]
