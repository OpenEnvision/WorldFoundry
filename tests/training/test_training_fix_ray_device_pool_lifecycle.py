"""Regression tests for TR-12/TR-13: RayDevicePool setup failure rollback.

Second-round fix (plan/code_review/fixes/second_round_fixes.md).  Historically
``RayDevicePool.setup()`` created placement groups (and possibly started the
Ray session) before ``self._ray`` was assigned, so a failing
``ray.get([group.ready() ...])`` leaked both: ``shutdown()`` early-returns on
``self._ray is None`` and ``__exit__`` never runs when ``__enter__`` raises.

All tests run on CPU with a fake ``ray`` module injected into ``sys.modules``
(``ray_runtime`` imports Ray lazily via ``import_module``).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from worldfoundry.training.distributed.ray_runtime import (
    RayDevicePool,
    RayDevicePoolConfig,
)


class _FakePlacementGroup:
    def __init__(self, bundles, strategy):
        self.bundles = bundles
        self.strategy = strategy

    def ready(self):
        return ("ready", id(self))


def _install_fake_ray(
    monkeypatch,
    *,
    get_error=None,
    init_error=None,
    already_initialized=False,
):
    """Register fake ``ray`` / ``ray.util.placement_group`` modules."""

    state = SimpleNamespace(
        initialized=already_initialized,
        init_calls=[],
        shutdown_calls=0,
        created=[],
        removed=[],
        get_calls=[],
        table_calls=0,
    )

    ray_module = types.ModuleType("ray")

    def is_initialized():
        return state.initialized

    def init(address=None, ignore_reinit_error=False):
        state.init_calls.append(address)
        state.initialized = True
        if init_error is not None:
            raise init_error

    def shutdown():
        state.shutdown_calls += 1
        state.initialized = False

    def get(refs, timeout=None):
        state.get_calls.append((tuple(refs) if isinstance(refs, list) else refs, timeout))
        if get_error is not None:
            raise get_error
        return [None] * (len(refs) if isinstance(refs, list) else 1)

    ray_module.is_initialized = is_initialized
    ray_module.init = init
    ray_module.shutdown = shutdown
    ray_module.get = get
    ray_module.kill = lambda actor, no_restart=False: None

    pg_module = types.ModuleType("ray.util.placement_group")

    def placement_group(bundles, strategy=None):
        group = _FakePlacementGroup(bundles, strategy)
        state.created.append(group)
        return group

    def remove_placement_group(group):
        state.removed.append(group)

    def placement_group_table():
        state.table_calls += 1
        return {"placement_groups": len(state.created)}

    pg_module.placement_group = placement_group
    pg_module.remove_placement_group = remove_placement_group
    pg_module.placement_group_table = placement_group_table

    util_module = types.ModuleType("ray.util")
    util_module.placement_group = pg_module
    ray_module.util = util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", util_module)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", pg_module)
    return state


def test_setup_failure_removes_placement_groups_and_shuts_down_started_ray(monkeypatch):
    state = _install_fake_ray(monkeypatch, get_error=RuntimeError("cluster fault"))
    pool = RayDevicePool(RayDevicePoolConfig(num_devices=4, devices_per_node=2))

    with pytest.raises(RuntimeError, match="cluster fault"):
        pool.setup()

    assert len(state.created) == 2
    assert state.removed == state.created
    # setup started this Ray session, so the rollback must stop it again.
    assert state.shutdown_calls == 1
    assert pool._ray is None
    assert pool._started_ray is False

    # shutdown() after a failed setup stays a harmless no-op (no double free).
    pool.shutdown()
    assert state.shutdown_calls == 1
    assert state.removed == state.created


def test_setup_failure_keeps_externally_initialized_ray_running(monkeypatch):
    state = _install_fake_ray(
        monkeypatch,
        get_error=RuntimeError("cluster fault"),
        already_initialized=True,
    )
    pool = RayDevicePool(RayDevicePoolConfig(num_devices=2, devices_per_node=2))

    with pytest.raises(RuntimeError, match="cluster fault"):
        pool.setup()

    assert state.removed == state.created
    # The session belongs to the caller; rollback must not stop it.
    assert state.shutdown_calls == 0
    assert state.initialized is True
    assert pool._ray is None


def test_context_manager_enter_failure_does_not_leak(monkeypatch):
    state = _install_fake_ray(monkeypatch, get_error=RuntimeError("cluster fault"))

    with pytest.raises(RuntimeError, match="cluster fault"):
        with RayDevicePool(RayDevicePoolConfig(num_devices=2, devices_per_node=2)):
            pytest.fail("body must not run when __enter__ raises")

    assert state.removed == state.created
    assert state.shutdown_calls == 1


def test_setup_import_failure_shuts_down_started_ray(monkeypatch):
    state = _install_fake_ray(monkeypatch)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", None)
    pool = RayDevicePool(RayDevicePoolConfig(num_devices=2, devices_per_node=2))

    with pytest.raises(ModuleNotFoundError, match="ray.util.placement_group"):
        pool.setup()

    assert state.created == []
    assert state.shutdown_calls == 1
    assert state.initialized is False
    assert pool._ray is None


def test_partial_init_failure_shuts_down_started_ray(monkeypatch):
    state = _install_fake_ray(monkeypatch, init_error=RuntimeError("partial init"))
    pool = RayDevicePool(RayDevicePoolConfig(num_devices=2, devices_per_node=2))

    with pytest.raises(RuntimeError, match="partial init"):
        pool.setup()

    assert state.created == []
    assert state.shutdown_calls == 1
    assert state.initialized is False
    assert pool._ray is None


def test_setup_success_then_shutdown_releases_everything(monkeypatch):
    state = _install_fake_ray(monkeypatch)
    pool = RayDevicePool(RayDevicePoolConfig(num_devices=4, devices_per_node=2))

    pool.setup()
    assert pool._ray is sys.modules["ray"]
    assert len(pool._placement_groups) == 2
    assert pool._started_ray is True
    # Default config keeps the historical unbounded ready() wait.
    assert state.get_calls[-1][1] is None
    assert state.removed == []

    # Second setup is a no-op once initialized.
    pool.setup()
    assert len(state.created) == 2

    pool.shutdown()
    assert state.removed == state.created
    assert state.shutdown_calls == 1
    assert pool._ray is None


def test_placement_timeout_is_forwarded_to_ray_get(monkeypatch):
    state = _install_fake_ray(monkeypatch)
    pool = RayDevicePool(
        RayDevicePoolConfig(num_devices=2, devices_per_node=2, placement_timeout_seconds=7.5)
    )

    pool.setup()
    assert state.get_calls[-1][1] == 7.5
    pool.shutdown()


def test_setup_failure_consults_scheduler_diagnostics(monkeypatch):
    state = _install_fake_ray(monkeypatch, get_error=TimeoutError("placement wait timed out"))
    pool = RayDevicePool(
        RayDevicePoolConfig(num_devices=2, devices_per_node=2, placement_timeout_seconds=1.0)
    )

    with pytest.raises(TimeoutError):
        pool.setup()

    assert state.table_calls == 1
    assert state.removed == state.created


@pytest.mark.parametrize("timeout", [0, -1, -0.5])
def test_placement_timeout_must_be_positive_when_set(timeout):
    with pytest.raises(ValueError, match="placement_timeout_seconds"):
        RayDevicePoolConfig(num_devices=2, devices_per_node=2, placement_timeout_seconds=timeout)
