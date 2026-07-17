"""WRBench generation backends for the WorldFoundry integration."""

from wrbench.backends.base import (
    DryRunBackend,
    GenerationBackend,
    GenerationRequest,
    GenerationResult,
    default_backend,
)
from wrbench.backends.registry import list_backends, resolve_backend
from wrbench.backends.worldfoundry import WorldFoundryPipelineBackend

__all__ = [
    "DryRunBackend",
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResult",
    "WorldFoundryPipelineBackend",
    "default_backend",
    "list_backends",
    "resolve_backend",
]
