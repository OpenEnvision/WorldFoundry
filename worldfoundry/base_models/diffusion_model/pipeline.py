"""Small public construction surface around a fully built native runner.

This module does not implement the sampling loop.  It only:

1. Builds a runner through ``from_pretrained`` / ``from_registry``.
2. Forwards :class:`DiffusionRequest` to ``runner.run`` from ``__call__``.
3. Exposes ``model_id`` / ``components`` / ``device`` / ``dtype`` for
   higher-level orchestration.

Do not put family-specific logic here.  Variants belong in a recipe, an
execution strategy, or an instance-local extension.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .components import ComponentKey
from .contracts import DiffusionOutput, DiffusionRequest
from .extensions import DiffusionExtension
from .loaders import CheckpointSpec
from .optimizations import RuntimePolicy
from .registry import NativeDiffusionRegistry
from .runners import DiffusionExecutor


class NativeDiffusionPipeline:
    """Request boundary around one fully constructed :class:`DiffusionExecutor`.

    Callers should depend on this class plus :class:`DiffusionRequest` /
    :class:`DiffusionOutput`, not on the assembler or strategy registry.
    """

    def __init__(self, runner: DiffusionExecutor) -> None:
        self.runner = runner

    @property
    def model_id(self) -> str:
        """Canonical recipe id (not an alias)."""
        return self.runner.model_id

    @property
    def components(self):
        """Expose the assembled role bundle for training or visualization.

        Some execution strategies do not expose ``components``; fail loudly
        instead of returning ``None``.
        """

        try:
            return self.runner.components
        except AttributeError as error:
            raise AttributeError("this diffusion execution strategy does not expose a component bundle") from error

    @property
    def device(self):
        """Device that actually holds the runner weights."""
        return self.runner.device

    @property
    def dtype(self):
        """Inference compute dtype, usually taken from :class:`RuntimePolicy`."""
        return self.runner.dtype

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        registry: NativeDiffusionRegistry | None = None,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        """Construct a built-in model id through the canonical assembler.

        When ``registry`` is omitted, the frozen
        :func:`default_native_diffusion_registry` is used.
        ``checkpoint_overrides`` may replace a Hub source with a local
        directory or a single file.  ``extensions`` attach to this runner
        instance only.
        """

        if registry is None:
            from .registry import default_native_diffusion_registry

            registry = default_native_diffusion_registry()
        return cls.from_registry(
            registry,
            model_id,
            policy=policy,
            checkpoint_overrides=checkpoint_overrides,
            component_options=component_options,
            extensions=extensions,
        )

    @classmethod
    def from_registry(
        cls,
        registry: NativeDiffusionRegistry,
        model_id: str,
        *,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        """Resolve a recipe on ``registry`` and assemble its runner."""
        return cls(
            registry.build_runner(
                model_id,
                policy=policy,
                checkpoint_overrides=checkpoint_overrides,
                component_options=component_options,
                extensions=extensions,
            )
        )

    def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
        """Run one normalized request.  Bare dicts are rejected so contracts stay enforced."""
        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"native diffusion pipeline expects DiffusionRequest, got {type(request).__name__}")
        return self.runner.run(request)


__all__ = ["NativeDiffusionPipeline"]
