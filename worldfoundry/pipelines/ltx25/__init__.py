"""WorldFoundry pipeline and SOL-Engine controls for official LTX-2.5 inference."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.ltx25 import (
        LTX25AccelerationConfig,
        LTX25FirstBlockCacheConfig,
        LTX25ParallelConfig,
        ltx25_sol_profile,
    )

    from .pipeline_ltx25 import LTX25Pipeline

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LTX25AccelerationConfig": ("worldfoundry.synthesis.visual_generation.ltx25", "LTX25AccelerationConfig"),
    "LTX25FirstBlockCacheConfig": ("worldfoundry.synthesis.visual_generation.ltx25", "LTX25FirstBlockCacheConfig"),
    "LTX25ParallelConfig": ("worldfoundry.synthesis.visual_generation.ltx25", "LTX25ParallelConfig"),
    "LTX25Pipeline": (f"{__name__}.pipeline_ltx25", "LTX25Pipeline"),
    "ltx25_sol_profile": ("worldfoundry.synthesis.visual_generation.ltx25", "ltx25_sol_profile"),
}

__all__ = [
    "LTX25AccelerationConfig",
    "LTX25FirstBlockCacheConfig",
    "LTX25ParallelConfig",
    "LTX25Pipeline",
    "ltx25_sol_profile",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
