"""Composition is atomic and reversible; capability planning never mutates."""

from types import SimpleNamespace

import pytest

from worldfoundry.core.acceleration.plugins import AccelerationHandle, AccelerationRegistry, PreparedAcceleration


def _factory(name, seam, *, fail=False):
    def prepare(model, options, context):
        def activate():
            if fail:
                raise RuntimeError("test installation failure")
            model.values.append(name)
            return AccelerationHandle(name, {}, lambda: model.values.remove(name))

        return PreparedAcceleration(name, frozenset({seam}), activate)

    return prepare


def test_install_uninstall_and_reporting():
    registry = AccelerationRegistry()
    registry.register("a", _factory("a", "norm"))
    model = SimpleNamespace(values=[])
    session = registry.install(model, {"a": True})
    assert model.values == ["a"]
    assert session.report() == {"installed": [{"name": "a"}], "execution_verified": False}
    session.uninstall()
    session.uninstall()
    assert not model.values and not hasattr(model, "_worldfoundry_accelerations")


def test_conflict_is_detected_before_any_activation():
    registry = AccelerationRegistry()
    registry.register("a", _factory("a", "norm"))
    registry.register("b", _factory("b", "norm"))
    model = SimpleNamespace(values=[])
    with pytest.raises(ValueError, match="conflicts"):
        registry.install(model, {"a": {}, "b": {}})
    assert not model.values


def test_late_failure_rolls_back_earlier_installs():
    registry = AccelerationRegistry()
    registry.register("a", _factory("a", "norm"))
    registry.register("b", _factory("b", "attention", fail=True))
    model = SimpleNamespace(values=[])
    with pytest.raises(RuntimeError, match="installation"):
        registry.install(model, {"a": {}, "b": {}})
    assert not model.values and not hasattr(model, "_worldfoundry_accelerations")


def test_unknown_disabled_plugins_and_compiled_mutation_fail():
    registry = AccelerationRegistry()
    with pytest.raises(ValueError, match="unknown"):
        registry.install(SimpleNamespace(), {"typo": False})
    with pytest.raises(ValueError, match="compile"):
        registry.install(SimpleNamespace(_worldfoundry_compile_runtime={}), {})


def test_removal_checks_every_bound_runtime():
    class Owner:
        live = 0

        def feature_cache_lifecycle_report(self):
            return {"live_requests": self.live}

    registry = AccelerationRegistry()
    registry.register("a", _factory("a", "norm"))
    session = registry.install(SimpleNamespace(values=[]), {"a": True})
    busy, idle = Owner(), Owner()
    busy.live = 1
    session.bind_runtime(busy)
    session.bind_runtime(idle)
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    busy.live = 0
    session.uninstall()


def test_report_cannot_mutate_live_provider_options():
    from worldfoundry.core.acceleration.plugins import AccelerationSession

    options = {"tau": 1.0}
    session = AccelerationSession(SimpleNamespace(), [AccelerationHandle("test", {"options": options}, lambda: None)])
    session.report()["installed"][0]["options"]["tau"] = 100.0
    assert options == {"tau": 1.0}
