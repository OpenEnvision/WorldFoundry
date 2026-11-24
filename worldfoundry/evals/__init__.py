"""WorldFoundry evaluation: benchmark catalogs, run orchestration, metrics, and reporting.

:mod:`worldfoundry.evals.public` is the programmatic entry point (``run_worldfoundry``,
``run_benchmark``, and the request/result types). Subpackages:

- :mod:`~worldfoundry.evals.api` — JSON-safe contracts shared by runners, CLI, and reports.
- :mod:`~worldfoundry.evals.tasks` — benchmark catalog, datasets, metrics, and execution.
- :mod:`~worldfoundry.evals.models` — model catalog, runtime profiles, and pipeline bindings.
- :mod:`~worldfoundry.evals.reporting` — scorecards, run indexes, and comparisons.

Subpackages load lazily so ``import worldfoundry.evals`` stays cheap.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["api", "models", "public", "reporting", "tasks", "utils"]


def __getattr__(name: str):
    """Import a public subpackage on first attribute access."""
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return the public subpackages alongside module globals."""
    return sorted({*globals(), *__all__})
