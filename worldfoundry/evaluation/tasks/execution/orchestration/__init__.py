"""Compatibility aliases for the legacy execution.orchestration import path."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

_ALIASES = (
    "benchmark_runner",
    "cache",
    "contract",
    "existing_results",
    "evaluate",
    "fidelity",
    "interfaces",
    "materialize",
    "model_benchmark",
    "model_benchmark_suite",
    "plan",
    "runtime_preflight",
    "run_mode",
    "service",
)

def __getattr__(name: str) -> ModuleType:
    if name not in _ALIASES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    sys.modules[f"{__name__}.{name}"] = module
    globals()[name] = module
    return module

__all__ = list(_ALIASES)
