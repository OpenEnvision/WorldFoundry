"""Embodied AI, VLA, VA, and WAM Task Evaluation Executor.

This module provides the high-level orchestration interface to compile, build, and trigger
an evaluation run for Vision-Language-Action, Video Action, and World Action Models.

It serves as the transition layer between:
1. High-level, user-friendly request formats (`VlaVaWamRunRequest`).
2. Materialized inputs and dataset layouts.
3. The underlying generic `worldfoundry.evaluation.tasks.execution.orchestration.evaluate` executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import (
    EvaluateRunRequest,
    EvaluateRunResult,
    execute_evaluate_run,
)

from .contracts import EmbodiedGenerationSpec
from .materialize import materialize_vla_va_wam_requests
from .metrics import metric_suite

VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION = "worldfoundry-vla-va-wam-run-request"


@dataclass(frozen=True)
class VlaVaWamRunRequest:
    """Configures the complete context and hyper-parameters for an Embodied AI evaluation run.

    Provides granular controls over dataset selection, runner manifest options,
    evaluation tracks, metric suites, and execution safety switches.
    """
    output_dir: str | Path
    spec: EmbodiedGenerationSpec | Mapping[str, Any]
    samples: Sequence[Mapping[str, Any]] = ()
    requests: Sequence[GenerationRequest | Mapping[str, Any]] = ()
    runner: Any = None
    model_id: str | None = None
    model_runner: str | None = None
    model_zoo_manifest_dir: str | Path | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    metric_ids: Sequence[str] = ()
    benchmark_id: str | None = None
    dataset_id: str | None = None
    split: str = "default"
    run_id: str | None = None
    fail_on_sample_error: bool = False
    cleanup_runner: bool = True
    schema_version: str = VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION


def _coerce_request(
    request: VlaVaWamRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> VlaVaWamRunRequest:
    """Safely coerces dictionary representations or parameter overrides into a clean VlaVaWamRunRequest.

    Args:
        request: A pre-formed VlaVaWamRunRequest, a raw mapping, or None.
        kwargs: Overrides to apply to the request configuration.

    Returns:
        A fully-coerced VlaVaWamRunRequest instance.
    """
    if isinstance(request, VlaVaWamRunRequest):
        if not kwargs:
            return request
        return replace(request, **kwargs)
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    if "output_dir" not in payload:
        raise TypeError("run_vla_va_wam requires output_dir")
    if "spec" not in payload:
        raise TypeError("run_vla_va_wam requires spec")
    return VlaVaWamRunRequest(**payload)


def _generation_requests(run_request: VlaVaWamRunRequest, spec: EmbodiedGenerationSpec) -> tuple[GenerationRequest, ...]:
    """Compiles and materializes physical GenerationRequest instances.

    If pre-formed requests are provided in the run request, validates and returns them.
    Otherwise, dynamically materializes raw sample dictionaries into target contracts.

    Args:
        run_request: The overarching evaluation request context.
        spec: The embodied generation specification.

    Returns:
        A tuple of materialized GenerationRequests.
    """
    if run_request.requests:
        requests: list[GenerationRequest] = []
        for item in run_request.requests:
            requests.append(item if isinstance(item, GenerationRequest) else GenerationRequest.from_dict(item))
        return tuple(requests)
    samples = tuple(run_request.samples or ({"sample_id": "sample-000000"},))
    return materialize_vla_va_wam_requests(samples, spec=spec, split=run_request.split).requests


def _model_parameters(run_request: VlaVaWamRunRequest, spec: EmbodiedGenerationSpec) -> dict[str, Any]:
    """Assembles model hyper-parameters and capabilities targeting the underlying runner.

    Args:
        run_request: The source evaluation request.
        spec: The embodied specification providing default capabilities and track.

    Returns:
        A dictionary containing parameters formatted for the model runner.
    """
    return {
        "track": spec.track.value,
        "metadata_namespace": "vla_va_wam",
        "capabilities": list(spec.required_capabilities),
        **dict(run_request.model_parameters or {}),
    }


def _benchmark_metadata(run_request: VlaVaWamRunRequest, spec: EmbodiedGenerationSpec) -> dict[str, Any]:
    """Compiles unified, immutable benchmark and task metadata for reporting and metric logging.

    Args:
        run_request: The overarching evaluation request.
        spec: The embodied task specification.

    Returns:
        A dictionary of benchmark metadata.
    """
    return {
        "suite": "vla_va_wam",
        "benchmark_name": run_request.benchmark_id or spec.task_name,
        "benchmark_id": run_request.benchmark_id or spec.task_name,
        "task_type": spec.task_name,
        "evaluation_protocol": "worldfoundry_evaluate_model",
        "track": spec.track.value,
        "request_kind": spec.kind.value,
        "action_space": spec.action_space.to_dict(),
        "required_capabilities": list(spec.required_capabilities),
    }


def _model_metadata(run_request: VlaVaWamRunRequest, spec: EmbodiedGenerationSpec) -> dict[str, Any]:
    """Extracts and formats static model metadata.

    Args:
        run_request: The current evaluation run request context.
        spec: The underlying generation specification.

    Returns:
        A formatted model metadata dictionary.
    """
    model_id = run_request.model_id
    metadata: dict[str, Any] = {
        "model_type": "world_model",
        "evaluation_track": spec.track.value,
        "required_capabilities": list(spec.required_capabilities),
    }
    if model_id is not None:
        metadata["model_id"] = model_id
        metadata["model_name"] = model_id
    return metadata


def build_vla_va_wam_evaluate_request(
    request: VlaVaWamRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> EvaluateRunRequest:
    """Builds a structured EvaluateRunRequest from the user's high-level run configuration.

    Resolves and links:
    1. Standardized input requests and schemas.
    2. Model configuration details (variants, parameters, Manifest paths).
    3. The metrics suite tailored to the chosen Evaluation Track.
    4. Dataset count and metadata attributes.
    """
    run_request = _coerce_request(request, kwargs)
    if run_request.schema_version != VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported VlaVaWamRunRequest schema_version: {run_request.schema_version}")
    spec = run_request.spec if isinstance(run_request.spec, EmbodiedGenerationSpec) else EmbodiedGenerationSpec.from_dict(run_request.spec)
    requests = _generation_requests(run_request, spec)
    return EvaluateRunRequest(
        output_dir=run_request.output_dir,
        mode="model",
        requests=requests,
        runner=run_request.runner,
        model_id=run_request.model_id,
        model_runner=run_request.model_runner,
        model_zoo_manifest_dir=run_request.model_zoo_manifest_dir,
        model_variant_id=run_request.model_variant_id,
        model_parameters=_model_parameters(run_request, spec),
        model_runtime=run_request.model_runtime,
        model_config=run_request.model_config,
        metrics=metric_suite(run_request.metric_ids, track=spec.track.value),
        benchmark=_benchmark_metadata(run_request, spec),
        model=_model_metadata(run_request, spec),
        dataset={
            "dataset_id": run_request.dataset_id or "vla_va_wam_materialized",
            "name": run_request.dataset_id or "vla_va_wam_materialized",
            "split": run_request.split,
            "sample_count": len(requests),
        },
        benchmark_id=run_request.benchmark_id or spec.task_name,
        dataset_id=run_request.dataset_id,
        run_id=run_request.run_id,
        fail_on_sample_error=run_request.fail_on_sample_error,
        cleanup_runner=run_request.cleanup_runner,
    )


def run_vla_va_wam(
    request: VlaVaWamRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> EvaluateRunResult:
    """Core execution entry-point.

    Builds an `EvaluateRunRequest` and runs the evaluation loop, accumulating outputs and
    aggregating metric results.
    """
    return execute_evaluate_run(build_vla_va_wam_evaluate_request(request, **kwargs))


execute_vla_va_wam_run = run_vla_va_wam


__all__ = [
    "VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION",
    "VlaVaWamRunRequest",
    "build_vla_va_wam_evaluate_request",
    "execute_vla_va_wam_run",
    "run_vla_va_wam",
]
