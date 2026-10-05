"""Native :class:`~...contracts.LatentInitializer` factories by family.

Each initializer implements ``initialize(DiffusionRequest, generator,
device, dtype)`` and optionally
:class:`~...contracts.EncodedLatentInitializer.initialize_with_encoder`
when the recipe also binds a shared :class:`~...contracts.LatentEncoder`.

This package is a lazy export surface: recipes import
``build_*_latent_initializer`` factories (or the few public classes
listed in ``_EXPORTS``).  Family-specific geometry (Wan 4n+1 frames,
StepVideo 17-frame chunks, Cosmos Video2World masks) lives in the
sibling modules, not here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Cosmos3LatentInitializer": ".cosmos3",
    "ProvidedLatentInitializer": ".provided",
    "build_cosmos3_latent_initializer": ".cosmos3",
    "build_ltx_multistage_latent_initializer": ".ltx",
    "build_ltx_video_latent_initializer": ".ltx",
    "build_provided_latent_initializer": ".provided",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
