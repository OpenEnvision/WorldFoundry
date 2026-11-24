"""Resolve and construct Python objects from declarative model configuration.

Hydra-style mappings name a ``target`` / ``class_path`` plus
``params`` / ``init_args``. :func:`resolve_symbol` imports
``package.module:Symbol`` (or dotted form). :func:`instantiate_from_config`
calls the constructor.

This does not load weights or download repositories.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Symbol resolve — colon form or last-dot; reload / invalidate_cache are opt-in
# ──────────────────────────────────────────────────────────────────────────


def resolve_symbol(
    target: str,
    *,
    reload: bool = False,
    invalidate_cache: bool = False,
) -> Any:
    """Resolve ``package.module:Symbol`` or ``package.module.Symbol``."""

    module_path, separator, symbol_name = str(target).partition(":")
    if not separator:
        module_path, symbol_name = str(target).rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    module = importlib.import_module(module_path)
    if reload:
        module = importlib.reload(module)
    return getattr(module, symbol_name)


# ──────────────────────────────────────────────────────────────────────────
# Instantiate — Hydra target/class_path; no weight load
# ──────────────────────────────────────────────────────────────────────────


def get_obj_from_str(
    target: str,
    reload: bool = False,
    invalidate_cache: bool = False,
) -> Any:
    """Compatibility name for :func:`resolve_symbol`."""

    return resolve_symbol(
        target,
        reload=reload,
        invalidate_cache=invalidate_cache,
    )


def instantiate_from_config(config: Mapping[str, Any] | str, **additional_kwargs: Any) -> Any:
    """Instantiate a target declared by ``target``/``class_path`` and its arguments."""

    if isinstance(config, str) and config in {"__is_first_stage__", "__is_unconditional__"}:
        return None
    if not isinstance(config, Mapping):
        raise KeyError("Expected a mapping with `target` or `class_path`.")

    target = config.get("target") or config.get("class_path")
    if not target:
        raise KeyError("Expected a mapping with `target` or `class_path`.")
    params = dict(config.get("params") or config.get("init_args") or {})
    params.update(additional_kwargs)
    return resolve_symbol(str(target))(**params)


__all__ = [
    "get_obj_from_str",
    "instantiate_from_config",
    "resolve_symbol",
]
