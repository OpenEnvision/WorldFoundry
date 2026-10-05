"""Pipeline runner: orchestrates evaluation generation through a WorldFoundry pipeline.

The :class:`WorldFoundryPipelineRunner` is the primary built-in runner that
delegates generation requests to a category-native pipeline lifecycle,
handling per-sample failure isolation and runtime profile resolution.
"""

from __future__ import annotations

import gc
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, WorldModelConfig

from ..pipelines.lifecycle import (
    PipelineLifecycleContext,
    PipelineRuntimeProfile,
    WorldFoundryPipelineLifecycle,
)
from ..pipelines.loading import (
    build_pipeline_runner_spec,
    load_pipeline_from_config,
)
from ..pipelines.results import (
    PipelineResultContext,
    failed_generation_result,
)


def load_runtime_profile(model_id: str) -> Any:
    """Load synthesis runtime profile on demand.

    Delegates to :func:`worldfoundry.evaluation.models.runtime.profiles.load_runtime_profile`.

    Args:
        model_id: The model identifier used to select the runtime profile.

    Returns:
        The loaded runtime profile object.
    """
    from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile as load_profile

    return load_profile(model_id)


class WorldFoundryPipelineRunner:
    """Evaluation runner that invokes a category-native WorldFoundry pipeline.

    Each generation request is dispatched through a
    :class:`WorldFoundryPipelineLifecycle` that handles per-sample failure
    isolation — exceptions are caught and converted to
    :func:`failed_generation_result` rather than aborting the batch.

    Attributes:
        model_id: Canonical identifier of the model being evaluated.
        pipeline: Loaded pipeline object responsible for actual generation.
        pipeline_target: ``module:Class`` path identifying the pipeline.
        runtime_profile_id: Profile identifier used to resolve runtime settings.
        output_dir: Optional directory where generated artifacts are written.
        generation_defaults: Defaults merged below per-request generation kwargs.
        cleaned: Flag indicating whether :meth:`cleanup` has been called.
    """

    capabilities = {"worldfoundry.pipeline"}

    def __init__(
        self,
        model_id: str,
        pipeline: Any,
        *,
        pipeline_target: str,
        runtime_profile_id: str | None = None,
        output_dir: Path | None = None,
        generation_defaults: Mapping[str, Any] | None = None,
        config: WorldModelConfig | None = None,
    ) -> None:
        """Initialize the pipeline runner with a loaded pipeline and configuration.

        Args:
            model_id: Canonical model identifier.
            pipeline: Loaded pipeline object.
            pipeline_target: ``module:Class`` path identifying the pipeline.
            runtime_profile_id: Overrides the profile used for runtime settings;
                defaults to ``model_id`` when ``None``.
            output_dir: Optional directory for generated artifacts.
            generation_defaults: Defaults applied to each generation request.
            config: Resolved construction settings, including model weights.
        """
        self.model_id = model_id
        self.pipeline = pipeline
        self.pipeline_target = pipeline_target
        self.runtime_profile_id = runtime_profile_id or model_id
        self.output_dir = output_dir
        self.generation_defaults = dict(generation_defaults or {})
        self.config = config
        self.cleaned = False

    @property
    def parameters(self) -> dict[str, Any]:
        """Current generation settings used by the shared runner identity."""
        return {
            "pipeline_target": self.pipeline_target,
            "runtime_profile_id": self.runtime_profile_id,
            "generation_defaults": dict(self.generation_defaults),
        }

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> "WorldFoundryPipelineRunner":
        """Build a runner from a :class:`WorldModelConfig` using the pipeline loading subsystem.

        Args:
            config: Fully resolved configuration object containing model ID,
                runner target, and pipeline parameters.

        Returns:
            A fully initialised :class:`WorldFoundryPipelineRunner` instance.
        """
        spec, pipeline = load_pipeline_from_config(config)
        return cls(
            model_id=spec.model_id,
            pipeline=pipeline,
            pipeline_target=spec.pipeline_target,
            runtime_profile_id=spec.runtime_profile_id,
            output_dir=spec.output_dir,
            generation_defaults=spec.generation_defaults,
            config=config,
        )

    def _runtime_profile(self) -> PipelineRuntimeProfile:
        """Resolve the runtime profile for the current ``runtime_profile_id``."""
        return PipelineRuntimeProfile.from_profile(load_runtime_profile(self.runtime_profile_id))

    def _lifecycle(self, profile: PipelineRuntimeProfile) -> WorldFoundryPipelineLifecycle:
        """Build a pipeline lifecycle context from the resolved runtime profile.

        Args:
            profile: The resolved :class:`PipelineRuntimeProfile` to embed in
                the lifecycle context.

        Returns:
            A :class:`WorldFoundryPipelineLifecycle` ready for generation.
        """
        return WorldFoundryPipelineLifecycle(
            pipeline=self.pipeline,
            context=PipelineLifecycleContext(
                model_id=self.model_id,
                output_dir=self.output_dir,
                pipeline_target=self.pipeline_target,
                profile=profile,
            ),
        )

    def _result_context(self, profile: Any) -> PipelineResultContext:
        """Build a result context for recording generation outcomes.

        Coerces ``profile`` to a :class:`PipelineRuntimeProfile` if it is not
        already one, then extracts ``artifact_kind`` and ``task_family`` for
        the result context.

        Args:
            profile: A runtime profile object or raw profile data.

        Returns:
            A :class:`PipelineResultContext` scoped to this runner's model ID.
        """
        runtime_profile = (
            profile if isinstance(profile, PipelineRuntimeProfile) else PipelineRuntimeProfile.from_profile(profile)
        )
        return PipelineResultContext(
            model_id=self.model_id,
            artifact_kind=runtime_profile.artifact_kind,
            task_family=runtime_profile.task_family,
            pipeline_target=self.pipeline_target,
        )

    def _generate_one(
        self,
        request: GenerationRequest,
        lifecycle: WorldFoundryPipelineLifecycle,
    ) -> GenerationResult:
        """Generate a single sample, catching exceptions to avoid batch abort.

        Args:
            request: The generation request to dispatch.
            lifecycle: The active pipeline lifecycle to invoke.

        Returns:
            A :class:`GenerationResult` — either the successful output or a
            failed result wrapping the caught exception.
        """
        try:
            return lifecycle.generate_in_context(request)
        except Exception as exc:  # noqa: BLE001 - evaluation records per-sample failures.
            return failed_generation_result(request, self.model_id, exc)

    def _apply_generation_defaults(self, request: GenerationRequest) -> GenerationRequest:
        """Merge model defaults below request-local generation kwargs."""
        if not self.generation_defaults:
            return request
        payload = request.to_dict()
        payload["generation_kwargs"] = {
            **self.generation_defaults,
            **dict(request.generation_kwargs or {}),
        }
        return GenerationRequest.from_dict(payload)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate results for a batch of requests through the pipeline lifecycle.

        Each request is processed independently so that individual failures do
        not abort the remaining batch.

        Args:
            requests: Sequence of :class:`GenerationRequest` objects.

        Returns:
            A list of :class:`GenerationResult` objects, one per request.
        """
        if self.cleaned:
            raise RuntimeError("cannot generate with a cleaned WorldFoundryPipelineRunner")
        lifecycle = self._lifecycle(self._runtime_profile())
        from worldfoundry.core import worldfoundry_inference_context

        with worldfoundry_inference_context():
            return [
                self._generate_one(self._apply_generation_defaults(request), lifecycle)
                for request in requests
            ]

    def reset_for_evaluation(self) -> None:
        """Clear request/session state while keeping model weights resident."""
        if self.cleaned:
            raise RuntimeError("cannot reset a cleaned WorldFoundryPipelineRunner")
        reset = getattr(self.pipeline, "reset_for_evaluation", None)
        if callable(reset):
            reset()
            return

        reset_performed = False
        reset_memory = getattr(self.pipeline, "reset_memory", None)
        if callable(reset_memory):
            reset_memory()
            reset_performed = True
        memory = getattr(self.pipeline, "memory_module", None)
        reset_records = getattr(memory, "reset_records", None)
        if callable(reset_records):
            reset_records()
            reset_performed = True

        # Conventional reset methods are used by several policy runners. Only
        # call a zero-argument contract so simulator resets that require an
        # episode or environment cannot be invoked accidentally.
        reset = getattr(self.pipeline, "reset", None)
        if not reset_performed and callable(reset):
            try:
                inspect.signature(reset).bind()
            except (TypeError, ValueError):
                return
            reset()

    def cleanup(self) -> None:
        """Release the pipeline and return unreferenced CUDA blocks to PyTorch."""
        if self.cleaned:
            return
        self.cleaned = True
        pipeline = self.pipeline
        self.pipeline = None
        cleanup_fn = getattr(pipeline, "cleanup", None)
        try:
            if callable(cleanup_fn):
                try:
                    cleanup_fn()
                except Exception as exc:  # noqa: BLE001 - cleanup must not invalidate completed outputs.
                    warnings.warn(f"pipeline cleanup failed: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            del cleanup_fn
            del pipeline
            gc.collect()
            torch = sys.modules.get("torch")
            cuda = getattr(torch, "cuda", None)
            is_initialized = getattr(cuda, "is_initialized", None)
            if callable(is_initialized) and is_initialized():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass


__all__ = [
    "PipelineLifecycleContext",
    "PipelineRuntimeProfile",
    "WorldFoundryPipelineLifecycle",
    "WorldFoundryPipelineRunner",
    "build_pipeline_runner_spec",
]
