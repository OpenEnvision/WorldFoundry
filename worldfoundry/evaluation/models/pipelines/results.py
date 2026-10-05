"""Pipeline result normalization and error extraction.

Converts raw pipeline output mappings into structured
:class:`GenerationResult` objects, compiling metadata, resolving
artifact references, and formatting error messages for failed runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import (
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    is_generation_status_successful,
    normalize_generation_status,
)

from .invocation import PipelineInvocation


# ── Result context dataclass ────────────────────────────────────────

@dataclass(frozen=True)
class PipelineResultContext:
    """Context for identifying model ID, task family, and artifact kind for results.

    Attributes:
        model_id: Canonical model identifier used for the generation.
        artifact_kind: Expected output type (e.g. ``"video"``, ``"image"``, ``"3d"``).
        task_family: High-level task category the request belongs to.
        pipeline_target: ``module:Class`` string of the pipeline that produced the result.
    """

    model_id: str
    artifact_kind: str
    task_family: str
    pipeline_target: str


# ── Metadata & status helpers ───────────────────────────────────────

def pipeline_metadata(
    *,
    result: Mapping[str, Any],
    context: PipelineResultContext,
) -> dict[str, Any]:
    """Compile and normalize execution metadata from raw pipeline outputs.

    Merges pipeline-specific keys (``status``, ``runtime``, ``error``, etc.)
    with context-level identifiers (``model_id``, ``task_family``).
    """
    return {
        "runtime_status": result.get("status"),
        "runtime": result.get("runtime"),
        "backend_quality": result.get("backend_quality"),
        "artifact_path": result.get("artifact_path"),
        "plan_path": result.get("plan_path"),
        "run_dir": result.get("run_dir"),
        "metadata_path": result.get("metadata_path"),
        "artifact_sha256": result.get("artifact_sha256"),
        "blocked_reason": result.get("blocked_reason"),
        "error": result.get("error"),
        "pipeline_metadata": result.get("metadata"),
        "profile_id": context.model_id,
        "profile_task_family": context.task_family,
        "pipeline_target": context.pipeline_target,
    }


def pipeline_result_status(result: Mapping[str, Any]) -> str:
    """Extract and normalize status text from a pipeline result mapping.

    In-process pipelines signal failure by raising (see
    :func:`failed_generation_result`), so a mapping returned without an
    explicit ``status`` is a successful return and defaults to
    ``"succeeded"`` here. External/deserialized rows instead go through
    :func:`~worldfoundry.evaluation.api.normalize_generation_status`, which
    fail-closes missing statuses to ``"unknown"``.
    """
    raw = str(result.get("status") or "").strip()
    if not raw:
        return "succeeded"
    status = normalize_generation_status(raw)
    return "succeeded" if is_generation_status_successful(status) else status


def pipeline_result_error(result: Mapping[str, Any], status: str) -> str | None:
    """Extract or formulate error/block messages from a failed pipeline result.

    Checks ``error``, ``blocked_reason``, and ``unsupported_reason`` keys in
    priority order.  If none are present, joins any ``blocked_reasons`` list
    into a single string.  Returns a generic fallback message if no explicit
    reason is found.

    Returns:
        ``None`` when ``status`` is ``"succeeded"``; otherwise a descriptive
        error string.
    """
    if status == "succeeded":
        return None
    for key in ("error", "blocked_reason", "unsupported_reason"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # NOTE: blocked_reasons may be a list; join them into a single string.
    blocked_reasons = result.get("blocked_reasons")
    if isinstance(blocked_reasons, Sequence) and not isinstance(blocked_reasons, (str, bytes)):
        reasons = [str(item) for item in blocked_reasons if str(item).strip()]
        if reasons:
            return "; ".join(reasons)
    return f"pipeline returned non-generation status: {status}"


# ── Generation result construction ──────────────────────────────────

def generation_result_from_pipeline(
    *,
    invocation: PipelineInvocation,
    result: Mapping[str, Any],
    context: PipelineResultContext,
    timings: Mapping[str, Any] | None = None,
) -> GenerationResult:
    """Normalize raw pipeline output mapping into a structured public :class:`GenerationResult`.

    On success, builds an :class:`ArtifactRef` from the output path or the
    ``artifact_path`` reported by the pipeline.  On failure, the result
    carries the resolved error message and no artifacts.

    Args:
        invocation: The :class:`PipelineInvocation` that produced the result.
        result: Raw key-value mapping returned by the pipeline.
        context: :class:`PipelineResultContext` with model/task metadata.
        timings: Optional timing measurements (e.g. ``generate_seconds``)
            recorded by the caller around the pipeline invocation; forwarded
            to the contract ``timings`` field.

    Returns:
        A fully-populated :class:`GenerationResult`.
    """
    request = invocation.request
    status = pipeline_result_status(result)
    artifacts = {}
    if status == "succeeded":
        artifact_kind = str(result.get("artifact_kind") or context.artifact_kind)
        # NOTE: prefer the pipeline-reported path; fall back to invocation.output_path.
        artifact_path = str(result.get("artifact_path") or invocation.output_path)
        artifacts[artifact_kind] = ArtifactRef.from_uri(artifact_path, kind=artifact_kind)
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id=context.model_id,
        artifacts=artifacts,
        status=status,
        error=pipeline_result_error(result, status),
        timings=dict(timings) if timings else None,
        metadata=pipeline_metadata(result=result, context=context),
    )


def failed_generation_result(request: GenerationRequest, model_id: str, exc: Exception) -> GenerationResult:
    """Create a failed :class:`GenerationResult` capturing an unhandled execution exception.

    Args:
        request: The :class:`GenerationRequest` that triggered the pipeline run.
        model_id: Canonical model identifier.
        exc: The exception that caused the failure.

    Returns:
        A :class:`GenerationResult` with ``status="failed"`` and the exception
        formatted as ``"<ExceptionType>: <message>"``.
    """
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id=model_id,
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
    )


__all__ = [
    "PipelineResultContext",
    "failed_generation_result",
    "generation_result_from_pipeline",
    "pipeline_metadata",
    "pipeline_result_error",
    "pipeline_result_status",
]
