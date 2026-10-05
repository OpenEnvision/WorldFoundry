"""Canonical WorldFoundry diffusion inference infrastructure.

This package owns the sampling contracts and the execution loop.  Model
families contribute pure component factories plus a declarative recipe; they
must not ship a runner, a pipeline, or process-wide mutable configuration.

The package deliberately does not wrap DiffSynth, Diffusers pipelines, or
model-specific upstream runtimes.

Typical call chain
------------------
1. :class:`NativeDiffusionPipeline.from_pretrained` looks up ``model_id``.
2. :class:`NativeDiffusionRegistry` lazily materializes a
   :class:`NativeDiffusionRecipe`.
3. :class:`NativeDiffusionAssembler` builds components and hands them to an
   execution strategy.
4. A runner (for example :class:`NativeDiffusionRunner`) executes
   encode → initialize → schedule → denoise(+CFG) → decode.
5. The loop returns an immutable :class:`DiffusionOutput`.

Layer responsibilities
----------------------
- ``contracts``: request / conditioning / latent / scheduler / protocols
- ``components``: component kinds, factory context, train/infer purpose checks
- ``recipes``: declarative model recipes (bindings + checkpoints + strategy id)
- ``assembly``: turn a recipe into an executable runner
- ``runners``: framework-owned sampling loops and the strategy registry
- ``extensions``: instance-local research hooks (no class mutation, no globals)
- ``optimizations``: offload / quantization / attention-backend policy
- ``loaders``: checkpoint description and materialization
- ``models``: family-specific component implementations
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public names are imported on first access so
# ``import worldfoundry.base_models.diffusion_model`` does not pull optional
# stacks such as Wan or SANA.
_EXPORTS = {
    "AttentionBackend": ".optimizations",
    "BuildPurpose": ".components",
    "CheckpointSpec": ".loaders",
    "ComponentBuildContext": ".components",
    "ComponentKey": ".components",
    "ComponentKind": ".components",
    "ComponentSpec": ".components",
    "Conditioning": ".contracts",
    "DenoiserInput": ".contracts",
    "DenoiserOutput": ".contracts",
    "DiffusionOutput": ".contracts",
    "DiffusionRequest": ".contracts",
    "DiffusionRunContext": ".extensions",
    "DiffusionExtension": ".extensions",
    "DiffusionExecutor": ".runners",
    "ExecutionSpec": ".components",
    "ExecutionStrategyRegistry": ".runners",
    "LatentEncoder": ".contracts",
    "NativeDiffusionPipeline": ".pipeline",
    "NativeDiffusionRecipe": ".registry",
    "NativeDiffusionRegistry": ".registry",
    "NativeDiffusionRunner": ".runners",
    "OffloadMode": ".optimizations",
    "OffloadPolicy": ".optimizations",
    "QuantizationMode": ".optimizations",
    "QuantizationPolicy": ".optimizations",
    "RunnerComponents": ".runners",
    "RuntimePolicy": ".optimizations",
    "SamplingConfig": ".contracts",
    "SchedulerStep": ".contracts",
    "default_native_diffusion_registry": ".registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export: import the owning submodule on first access and cache it."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in ``dir(module)`` for IDE / REPL completion."""
    return sorted({*globals(), *__all__})
