"""Pipeline lifecycle protocols and one-request orchestration.

Defines the :class:`WorldFoundryPipelineProtocol` and
:class:`WorldFoundryPipelineInvocationProtocol` protocol interfaces that
pipeline implementations may satisfy, plus the
:class:`WorldFoundryPipelineLifecycle` class that normalises a single
generation request into invocation → execution → result-mapping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult

from .invocation import PipelineInvocation, build_pipeline_invocation, invoke_pipeline, request_output_dir
from .results import PipelineResultContext, generation_result_from_pipeline

# ── Protocol interfaces ────────────────────────────────────────────────


class WorldFoundryPipelineProtocol(Protocol):
    """Callable pipeline surface supported by the standard invocation path."""

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        """Run one normalized generation request and return a result mapping."""


class WorldFoundryPipelineInvocationProtocol(Protocol):
    """Optional native lifecycle hook for pipelines that accept normalized invocations."""

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        """Run one request from a pre-normalized invocation."""


# ── Data-backed context types ─────────────────────────────────────────


@dataclass(frozen=True)
class PipelineRuntimeProfile:
    """Runtime profile capturing target artifact naming, kind, and task family.

    Attributes:
        artifact_filename: Filename template for generated output artifacts.
        artifact_kind: Category of artifact (e.g. ``"video"``).
        task_family: High-level task grouping this profile belongs to.
    """

    artifact_filename: str
    artifact_kind: str
    task_family: str

    @classmethod
    def from_profile(cls, profile: Any) -> "PipelineRuntimeProfile":
        """Instantiate a PipelineRuntimeProfile from a raw profile object."""
        return cls(
            artifact_filename=str(profile.artifact_filename),
            artifact_kind=str(profile.artifact_kind),
            task_family=str(profile.task_family),
        )


@dataclass(frozen=True)
class PipelineLifecycleContext:
    """Context describing model, output, target, and runtime profile for a lifecycle.

    Attributes:
        model_id: HuggingFace-style model identifier.
        output_dir: Directory path for generated output files, or ``None``.
        pipeline_target: ``module:Class`` dotted path of the pipeline implementation.
        profile: :class:`PipelineRuntimeProfile` carrying artifact metadata.
    """

    model_id: str
    output_dir: Path | None
    pipeline_target: str
    profile: PipelineRuntimeProfile

    @property
    def result_context(self) -> PipelineResultContext:
        """Derive the corresponding PipelineResultContext from this lifecycle context."""
        return PipelineResultContext(
            model_id=self.model_id,
            artifact_kind=self.profile.artifact_kind,
            task_family=self.profile.task_family,
            pipeline_target=self.pipeline_target,
        )


# ── Lifecycle orchestration ───────────────────────────────────────────


class WorldFoundryPipelineLifecycle:
    """One-request lifecycle for pipeline-backed generation.

    The runner owns batch orchestration and the error boundary. This lifecycle
    owns request normalization, pipeline invocation, and result normalization.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        context: PipelineLifecycleContext,
    ) -> None:
        """Initialize the lifecycle with a pipeline instance and lifecycle context."""
        self.pipeline = pipeline
        self.context = context

    def build_invocation(self, request: GenerationRequest) -> PipelineInvocation:
        """Build and normalize a PipelineInvocation from a GenerationRequest."""
        kwargs = dict(request.generation_kwargs or {})
        output_dir = request_output_dir(self.context.output_dir, kwargs)
        return build_pipeline_invocation(
            request=request,
            output_dir=output_dir,
            artifact_filename=self.context.profile.artifact_filename,
            generation_kwargs=kwargs,
        )

    def invoke(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        """Invoke the loaded pipeline on a given PipelineInvocation."""
        return invoke_pipeline(self.pipeline, invocation)

    def normalize(
        self,
        invocation: PipelineInvocation,
        result: Mapping[str, Any],
        *,
        timings: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Convert raw pipeline output mapping into a structured GenerationResult."""
        return generation_result_from_pipeline(
            invocation=invocation,
            result=result,
            context=self.context.result_context,
            timings=timings,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute the entire generation request lifecycle sequentially under the inference context."""
        from worldfoundry.core import worldfoundry_inference_context

        with worldfoundry_inference_context():
            return self.generate_in_context(request)

    def generate_in_context(self, request: GenerationRequest) -> GenerationResult:
        """Generate one request when the caller already owns the inference context."""
        invocation = self.build_invocation(request)
        started = time.perf_counter()
        result = self.invoke(invocation)
        elapsed = time.perf_counter() - started
        return self.normalize(invocation, result, timings={"generate_seconds": round(elapsed, 6)})


__all__ = [
    "PipelineLifecycleContext",
    "PipelineRuntimeProfile",
    "WorldFoundryPipelineInvocationProtocol",
    "WorldFoundryPipelineLifecycle",
    "WorldFoundryPipelineProtocol",
]
