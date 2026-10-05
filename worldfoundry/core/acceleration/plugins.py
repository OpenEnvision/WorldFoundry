"""Small transactional installer for reversible, model-owned accelerations.

Factories validate without mutation and return prepared changes. Model adapters
own factories; this module owns composition, conflicting seams and rollback.
Installation receipts describe configuration, never prove kernel execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any
from weakref import WeakKeyDictionary, WeakSet


class _RuntimeOwnership:
    """Model-owned execution guard, independent of any installed plugin."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.owners = WeakSet()
        self.requests = WeakKeyDictionary()
        self.active_calls = 0

    def __deepcopy__(self, memo):
        # Execution ownership belongs to the original model's runtime.
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()


_ownership_creation_lock = Lock()


def _runtime_ownership(model: Any) -> _RuntimeOwnership:
    state = getattr(model, "_worldfoundry_acceleration_runtime_ownership", None)
    if state is not None:
        return state
    with _ownership_creation_lock:
        state = getattr(model, "_worldfoundry_acceleration_runtime_ownership", None)
        if state is None:
            state = _RuntimeOwnership()
            model._worldfoundry_acceleration_runtime_ownership = state
        return state


@contextmanager
def _locked_runtime_ownership(model: Any):
    """Retry if an idle child was rebound while this thread waited for its lock."""
    while True:
        state = _runtime_ownership(model)
        with state.lock:
            if getattr(model, "_worldfoundry_acceleration_runtime_ownership", None) is not state:
                continue
            yield state
            return


def _bind_runtime_tree(model: Any, state: _RuntimeOwnership) -> None:
    """Share the parent's guard; migrating another runtime's ownership is unsafe."""
    modules = getattr(model, "modules", None)
    with _ownership_creation_lock, ExitStack() as locks:
        targets = tuple(modules()) if callable(modules) else (model,)
        changed = [
            (target, getattr(target, "_worldfoundry_acceleration_runtime_ownership", None))
            for target in targets
            if getattr(target, "_worldfoundry_acceleration_runtime_ownership", None) is not state
        ]
        if not changed:
            return
        previous_states = {previous for _, previous in changed if previous is not None}
        for previous in sorted(previous_states, key=id):
            if not previous.lock.acquire(blocking=False):
                raise RuntimeError("cannot bind a model tree while another acceleration runtime is mutating it")
            locks.callback(previous.lock.release)
        if any(
            previous.owners or previous.active_calls or any(previous.requests.values()) for previous in previous_states
        ):
            raise RuntimeError("cannot bind a model tree with another registered acceleration runtime")
        published = []
        try:
            for target, previous in changed:
                target._worldfoundry_acceleration_runtime_ownership = state
                published.append((target, previous))
        except BaseException:
            for target, previous in reversed(published):
                if previous is None:
                    delattr(target, "_worldfoundry_acceleration_runtime_ownership")
                else:
                    target._worldfoundry_acceleration_runtime_ownership = previous
            raise


def register_acceleration_runtime(model: Any, owner: Any) -> None:
    """Register a denoiser before installation, without retaining its lifetime."""
    with _locked_runtime_ownership(model) as state:
        if owner in state.owners:
            return
        _bind_runtime_tree(model, state)
        state.owners.add(owner)


@contextmanager
def acceleration_runtime_scope(model: Any, owner: Any, request_id: str | None = None):
    """Keep model mutation outside forwards and explicit request lifecycles.

    A request stays live across CFG calls and sampling steps until finalization.
    Calls without a request id still guard their actual execution. Counters are
    released on exceptions; the runner owns request cleanup in its finally path.
    """
    with _locked_runtime_ownership(model) as state:
        state.owners.add(owner)
        if isinstance(request_id, str) and request_id.strip():
            state.requests.setdefault(owner, set()).add(request_id)
        state.active_calls += 1
    try:
        yield
    finally:
        with state.lock:
            state.active_calls -= 1


def end_acceleration_request(model: Any, owner: Any, request_id: str) -> None:
    """Release one request after all denoiser cleanup, never an in-flight call."""
    with _locked_runtime_ownership(model) as state:
        requests = state.requests.get(owner)
        if requests is not None:
            requests.discard(request_id)
            if not requests:
                state.requests.pop(owner, None)


def _validate_runtime_mutation(state: _RuntimeOwnership) -> None:
    if state.active_calls or any(state.requests.values()):
        raise RuntimeError("finish active requests before changing acceleration plugins")
    for owner in tuple(state.owners):
        if getattr(getattr(owner, "model", None), "_worldfoundry_compile_runtime", None) is not None:
            raise RuntimeError("rebuild a compiled model before changing acceleration plugins")
        if getattr(owner, "_graph_runner", None) is not None:
            raise RuntimeError("rebuild a CUDA Graph runtime before changing acceleration plugins")
        reporter = getattr(owner, "feature_cache_lifecycle_report", None)
        if callable(reporter) and reporter()["live_requests"]:
            raise RuntimeError("finish active requests before changing acceleration plugins")


@dataclass
class AccelerationHandle:
    """One installed component; remove while its model is idle, before capture."""

    name: str
    details: Mapping[str, Any]
    undo: Callable[[], None] = field(repr=False)
    runtime_report: Callable[[], Mapping[str, Any]] | None = field(default=None, repr=False)
    reset_request_window: Callable[[], None] | None = field(default=None, repr=False)


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

    def bind_runtime(self, owner: Any) -> None:
        register_acceleration_runtime(self.model, owner)

    def report(self) -> dict[str, Any]:
        return {
            "installed": [
                {
                    "name": item.name,
                    **deepcopy(dict(item.details)),
                    **({"runtime": deepcopy(dict(item.runtime_report()))} if item.runtime_report else {}),
                }
                for item in self.handles
            ],
            "execution_verified": False,
        }

    def reset_request_window(self) -> None:
        for handle in self.handles:
            if handle.reset_request_window is not None:
                handle.reset_request_window()

    def uninstall(self) -> None:
        """Restore original attributes; compiled/captured models must be rebuilt."""
        with _locked_runtime_ownership(self.model) as state:
            if getattr(self.model, "_worldfoundry_compile_runtime", None) is not None:
                raise RuntimeError("rebuild a compiled model before changing acceleration plugins")
            _validate_runtime_mutation(state)
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
        with _locked_runtime_ownership(model) as state:
            _validate_runtime_mutation(state)
            _bind_runtime_tree(model, state)
            return self._install(model, options, context)

    def _install(self, model: Any, options: Mapping[str, Any], context: Any) -> AccelerationSession:
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
            _bind_runtime_tree(model, _runtime_ownership(model))
            session = AccelerationSession(model, handles)
            model._worldfoundry_accelerations = session
        except BaseException:
            for handle in reversed(handles):
                handle.undo()
            raise
        return session
