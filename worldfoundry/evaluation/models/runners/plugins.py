"""Plugin discovery for third-party model runners.

Scans Python entry-points under the ``worldfoundry.model_runners`` group and
environment-variable definitions in ``WORLDFOUNDRY_MODEL_RUNNERS`` to discover
runner entries without mutating the builtin registry.
"""

from __future__ import annotations

import importlib
import os
import traceback
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any, Mapping

from .registry import ModelRunnerRegistryEntry, ModelRunnerRegistryIssue

# ── Entry-point and environment-variable names ─────────────────────────
ENTRY_POINT_GROUP = "worldfoundry.model_runners"
ENV_VAR = "WORLDFOUNDRY_MODEL_RUNNERS"


@dataclass(frozen=True)
class ModelRunnerPluginDiscovery:
    """Discovered model-runner plugins and load warnings.

    Attributes:
        entries: Successfully coerced plugin runner entries.
        issues: Warnings or errors from the discovery process (e.g. failed
            entry-point loads or invalid ``NAME=module:Class`` definitions).
    """

    entries: tuple[ModelRunnerRegistryEntry, ...] = ()
    issues: tuple[ModelRunnerRegistryIssue, ...] = ()


def discover_model_runner_plugins(*, env: Mapping[str, str] | None = None) -> ModelRunnerPluginDiscovery:
    """Discover third-party model runner entries without mutating the builtin registry.

    Scans two sources:

    1. **Entry-points** registered under the
       :data:`ENTRY_POINT_GROUP` group (``worldfoundry.model_runners``).
    2. **Environment-variable** definitions from :data:`ENV_VAR`
       (``WORLDFOUNDRY_MODEL_RUNNERS``), formatted as ``NAME=module:Class``
       comma-separated values.

    Args:
        env: Environment mapping to read.  Defaults to ``os.environ`` when
            ``None``, useful for testing with isolated environment dicts.

    Returns:
        A :class:`ModelRunnerPluginDiscovery` containing discovered entries
        and any issues encountered during loading.
    """

    environment = os.environ if env is None else env
    # Entry-points cannot change within a process, so the scan (and plugin
    # module imports it triggers) is cached; env-var definitions are re-read
    # on every call because the environment can change between calls.
    ep_entries, ep_issues = _entry_point_discovery()
    entries: list[ModelRunnerRegistryEntry] = list(ep_entries)
    issues: list[ModelRunnerRegistryIssue] = list(ep_issues)

    # ── Parse WORLDFOUNDRY_MODEL_RUNNERS env-var definitions ────────────────
    raw = environment.get(ENV_VAR, "")
    for definition in (item.strip() for item in raw.split(",")):
        if not definition:
            continue
        origin = f"{ENV_VAR} entry {definition!r}"
        name, separator, runner_target = definition.partition("=")
        if not separator or not name.strip() or not runner_target.strip():
            issues.append(
                ModelRunnerRegistryIssue(
                    code="plugin_definition_invalid",
                    message=f"expected NAME=module:Class, got {definition!r}",
                    severity="warning",
                    origin=origin,
                )
            )
            continue
        entries.append(
            ModelRunnerRegistryEntry(
                name=name.strip(),
                runner_target=runner_target.strip(),
                source="plugin",
                origin=origin,
            )
        )

    return ModelRunnerPluginDiscovery(entries=tuple(entries), issues=tuple(issues))


@lru_cache(maxsize=1)
def _entry_point_discovery() -> tuple[tuple[ModelRunnerRegistryEntry, ...], tuple[ModelRunnerRegistryIssue, ...]]:
    """Scan ``worldfoundry.model_runners`` entry-points once per process.

    Use ``_entry_point_discovery.cache_clear()`` to force a re-scan (tests).
    """
    entries: list[ModelRunnerRegistryEntry] = []
    issues: list[ModelRunnerRegistryIssue] = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda item: item.name):
        origin = f"entry point {ep.name!r} -> {ep.value}"
        try:
            entries.append(_coerce_plugin_entry(ep.load(), fallback_name=ep.name, origin=origin))
        except Exception as exc:  # noqa: BLE001 - broken plugins must not break builtin runners.
            issues.append(
                ModelRunnerRegistryIssue(
                    code="plugin_load_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    severity="warning",
                    name=ep.name,
                    origin=origin,
                    details={"traceback": traceback.format_exc()},
                )
            )
    return tuple(entries), tuple(issues)


def _coerce_plugin_entry(value: Any, *, fallback_name: str, origin: str) -> ModelRunnerRegistryEntry:
    """Coerce any plugin entry value to a :class:`ModelRunnerRegistryEntry`.

    Accepts several input shapes:

    - A :class:`ModelRunnerRegistryEntry` instance (source/origin overwritten).
    - A dict-like mapping (missing keys filled with defaults).
    - A ``module:Class`` target string.
    - A Python class (``runner_class`` set directly).
    - A zero-argument factory callable (called first, then re-coerced).

    Args:
        value: The plugin-provided entry object to coerce.
        fallback_name: Name assigned when the value does not carry one.
        origin: Provenance string recording where this entry was discovered.

    Raises:
        TypeError: If ``value`` is none of the recognised shapes.
    """
    if isinstance(value, ModelRunnerRegistryEntry):
        return replace(value, source="plugin", origin=origin)
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("name", fallback_name)
        payload.setdefault("source", "plugin")
        payload.setdefault("origin", origin)
        return ModelRunnerRegistryEntry.from_mapping(payload)
    if isinstance(value, str):
        return ModelRunnerRegistryEntry(
            name=fallback_name,
            runner_target=value,
            source="plugin",
            origin=origin,
        )
    if isinstance(value, type):
        return ModelRunnerRegistryEntry(
            name=fallback_name,
            runner_target=f"{value.__module__}:{value.__qualname__}",
            source="plugin",
            origin=origin,
            runner_class=value,
        )
    if callable(value):
        return _coerce_plugin_entry(value(), fallback_name=fallback_name, origin=origin)
    module_name, separator, attr_name = str(value).partition(":")
    if separator and module_name and attr_name:
        loaded = getattr(importlib.import_module(module_name), attr_name)
        return _coerce_plugin_entry(loaded, fallback_name=fallback_name, origin=origin)
    raise TypeError(f"expected runner entry, mapping, class, target string, or factory; got {type(value).__name__}")


__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
    "ModelRunnerPluginDiscovery",
    "discover_model_runner_plugins",
]
