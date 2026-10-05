"""Lazy ``benchmark_id → module attribute`` lookup.

Per-benchmark hook modules often import the shared evaluator engines.  Registries
must therefore load those modules on demand.  One table is the source of truth;
supported ids, error text, and cached attributes are derived from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from typing import Any

from worldfoundry.evaluation.tasks.catalog.benchmark_id import normalize_benchmark_id


@lru_cache(maxsize=None)
def load_module_attr(module_name: str, attr: str) -> Any:
    """Import ``module_name`` once and return ``attr``."""
    return getattr(import_module(module_name), attr)


def normalize_registry_key(value: str) -> str:
    return normalize_benchmark_id(value)


def lookup_module_attr(
    table: Mapping[str, str] | Mapping[str, tuple[str, str]],
    key: str,
    *,
    kind: str,
    attr: str | None = None,
) -> Any:
    """Return a cached attribute for ``key``.

    ``table`` values are either a module path (then ``attr`` is required) or
    ``(module_path, attr_name)``.
    """
    normalized = normalize_registry_key(key)
    spec = table.get(normalized)
    if spec is None:
        known = ", ".join(table)
        raise KeyError(f"unsupported {kind} {key!r}; known: {known}")
    if isinstance(spec, tuple):
        module_name, attr_name = spec
    else:
        if attr is None:
            raise ValueError(f"{kind} table values are module paths; pass attr=")
        module_name, attr_name = spec, attr
    return load_module_attr(module_name, attr_name)


def supported_ids(table: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(table)
