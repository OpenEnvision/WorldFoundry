"""Small transactional installer for reversible, model-owned accelerations.

Factories validate without mutation and return prepared changes. Model adapters
own factories; this module owns composition, conflicting seams and rollback.
Installation receipts describe configuration, never prove kernel execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakSet


@dataclass
class AccelerationHandle:
    """One installed component; remove while its model is idle, before capture."""

    name: str
    details: Mapping[str, Any]
    undo: Callable[[], None] = field(repr=False)


@dataclass(frozen=True)
class PreparedAcceleration:
    """Validated plan; activation must undo its own partial changes on failure."""

    name: str
    seams: frozenset[str]
    activate: Callable[[], AccelerationHandle]


class AccelerationSession:
    """A removable group installed on exactly one model."""

    def __init__(self, model: Any, handles: list[AccelerationHandle]) -> None:
        self.model = model
        self.handles = handles
        self._runtime_owners = WeakSet()

    def bind_runtime(self, owner: Any) -> None:
        self._runtime_owners.add(owner)

    def report(self) -> dict[str, Any]:
        return {
            "installed": [{"name": item.name, **deepcopy(dict(item.details))} for item in self.handles],
            "execution_verified": False,
        }

    def uninstall(self) -> None:
        """Restore original attributes; compiled/captured models must be rebuilt."""
        if getattr(self.model, "_worldfoundry_compile_runtime", None) is not None:
            raise RuntimeError("rebuild a compiled model before changing acceleration plugins")
        if any(owner.feature_cache_lifecycle_report()["live_requests"] for owner in tuple(self._runtime_owners)):
            raise RuntimeError("finish active requests before removing acceleration plugins")
        for handle in reversed(self.handles):
            handle.undo()
        self.handles.clear()
        if getattr(self.model, "_worldfoundry_accelerations", None) is self:
            delattr(self.model, "_worldfoundry_accelerations")


PluginFactory = Callable[[Any, Mapping[str, Any], Any], PreparedAcceleration]


class AccelerationRegistry:
    """Explicit registry; applications may add factories without global patches."""

    def __init__(self) -> None:
        self._factories: dict[str, PluginFactory] = {}

    def register(self, name: str, factory: PluginFactory) -> None:
        if not isinstance(name, str) or not name or name in self._factories:
            raise ValueError(f"invalid or duplicate acceleration plugin {name!r}")
        if not callable(factory):
            raise TypeError("plugin factory must be callable")
        self._factories[name] = factory

    def install(self, model: Any, options: Mapping[str, Any], context: Any = None) -> AccelerationSession:
        """Validate all plugins first, then activate; roll back on any failure."""
        if not isinstance(options, Mapping):
            raise TypeError("accelerations must be a mapping of plugin names to options")
        if getattr(model, "_worldfoundry_accelerations", None) is not None:
            raise ValueError("uninstall the existing acceleration session before installing another")
        if getattr(model, "_worldfoundry_compile_runtime", None) is not None:
            raise ValueError("install acceleration plugins before torch.compile")
        unknown = set(options) - self._factories.keys()
        if unknown:
            raise ValueError(f"unknown acceleration plugins: {sorted(unknown)}")
        prepared: list[PreparedAcceleration] = []
        occupied: set[str] = set()
        for name, raw in options.items():
            if raw is False or raw is None:
                continue
            if raw is True:
                raw = {}
            if not isinstance(raw, Mapping):
                raise TypeError(f"plugin {name!r} options must be a mapping")
            plan = self._factories[name](model, dict(raw), context)
            conflicts = occupied.intersection(plan.seams)
            if conflicts:
                raise ValueError(f"plugin {name!r} conflicts on seams {sorted(conflicts)}")
            occupied.update(plan.seams)
            prepared.append(plan)
        handles: list[AccelerationHandle] = []
        try:
            for plan in prepared:
                handles.append(plan.activate())
            session = AccelerationSession(model, handles)
            model._worldfoundry_accelerations = session
        except BaseException:
            for handle in reversed(handles):
                handle.undo()
            raise
        return session
