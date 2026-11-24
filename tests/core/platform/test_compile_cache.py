"""Shared compile wrapper identity, setup failures, and explicit Dynamo limits."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch.fx import GraphModule

from worldfoundry.core.platform import compile_cache

RECOMPILE_LIMIT = "WORLDFOUNDRY_COMPILE_RECOMPILE_LIMIT"
ACCUMULATED_RECOMPILE_LIMIT = "WORLDFOUNDRY_COMPILE_ACCUMULATED_RECOMPILE_LIMIT"
HELPERS = (compile_cache.compile_module_cached, compile_cache.compile_callable_cached)


@pytest.fixture(autouse=True)
def isolated_compile_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep compiler limits, caches, and warning state local to each test."""
    import torch._dynamo as dynamo
    from torch._dynamo import config

    monkeypatch.delenv(RECOMPILE_LIMIT, raising=False)
    monkeypatch.delenv(ACCUMULATED_RECOMPILE_LIMIT, raising=False)
    monkeypatch.setattr(compile_cache, "_FALLBACK_WARNED", set())
    monkeypatch.setattr(compile_cache, "configure_persistent_compile_cache", lambda **kwargs: None)
    attributes = (
        "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit",
        "accumulated_recompile_limit"
        if hasattr(config, "accumulated_recompile_limit")
        else "accumulated_cache_size_limit",
    )
    saved = {name: getattr(config, name) for name in attributes}
    dynamo.reset()
    try:
        yield
    finally:
        dynamo.reset()
        for name, value in saved.items():
            setattr(config, name, value)


@pytest.fixture
def compiler_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def compile_target(target: Any, **kwargs: Any) -> Any:
        calls.append((target, kwargs))
        if isinstance(target, torch.nn.Module):
            return SimpleNamespace(_orig_mod=target)
        return SimpleNamespace(_torchdynamo_orig_callable=target)

    monkeypatch.setattr(torch, "compile", compile_target)
    return calls


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("modern_names", (False, True))
@pytest.mark.parametrize(
    "settings",
    (
        {},
        {RECOMPILE_LIMIT: "12"},
        {ACCUMULATED_RECOMPILE_LIMIT: "64"},
        {RECOMPILE_LIMIT: "12", ACCUMULATED_RECOMPILE_LIMIT: "64"},
    ),
)
def test_compile_limits_unset_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[tuple[Any, dict[str, Any]]],
    helper: Callable[..., Any],
    modern_names: bool,
    settings: dict[str, str],
) -> None:
    names = (
        ("recompile_limit", "accumulated_recompile_limit")
        if modern_names
        else ("cache_size_limit", "accumulated_cache_size_limit")
    )
    config = SimpleNamespace(**dict(zip(names, (8, 256))))
    import_module = compile_cache.importlib.import_module
    monkeypatch.setattr(
        compile_cache.importlib,
        "import_module",
        lambda name: config if name == "torch._dynamo.config" else import_module(name),
    )
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    module = torch.nn.Identity()
    compiled = helper(module, strict=True)
    assert compiled is helper(module, strict=True)
    assert getattr(config, names[0]) == int(settings.get(RECOMPILE_LIMIT, "8"))
    assert getattr(config, names[1]) == int(settings.get(ACCUMULATED_RECOMPILE_LIMIT, "256"))
    assert compiler_calls == [(module, {"backend": "inductor", "mode": "default"})]


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("invalid_name", (RECOMPILE_LIMIT, ACCUMULATED_RECOMPILE_LIMIT))
@pytest.mark.parametrize("value", ("", "abc", "1.5", "0", "-1"))
def test_compile_limits_invalid_before_any_setup(
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[tuple[Any, dict[str, Any]]],
    helper: Callable[..., Any],
    invalid_name: str,
    value: str,
) -> None:
    from torch._dynamo import config

    names = (
        ("recompile_limit", "accumulated_recompile_limit")
        if hasattr(config, "recompile_limit")
        else ("cache_size_limit", "accumulated_cache_size_limit")
    )
    before = [getattr(config, name) for name in names]
    setup_calls: list[str] = []
    monkeypatch.setattr(
        compile_cache, "configure_persistent_compile_cache", lambda **kwargs: setup_calls.append("setup")
    )
    monkeypatch.setenv(RECOMPILE_LIMIT, "12")
    monkeypatch.setenv(ACCUMULATED_RECOMPILE_LIMIT, "64")
    monkeypatch.setenv(invalid_name, value)
    with pytest.raises(ValueError, match=invalid_name):
        helper(torch.nn.Identity())
    assert [getattr(config, name) for name in names] == before
    assert setup_calls == []
    assert compiler_calls == []


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_limits_unsupported_before_mutation(
    monkeypatch: pytest.MonkeyPatch, helper: Callable[..., Any]
) -> None:
    config = SimpleNamespace(recompile_limit=8)
    import_module = compile_cache.importlib.import_module
    monkeypatch.setattr(
        compile_cache.importlib,
        "import_module",
        lambda name: config if name == "torch._dynamo.config" else import_module(name),
    )
    setup_calls: list[str] = []
    monkeypatch.setattr(
        compile_cache, "configure_persistent_compile_cache", lambda **kwargs: setup_calls.append("setup")
    )
    monkeypatch.setenv(RECOMPILE_LIMIT, "12")
    monkeypatch.setenv(ACCUMULATED_RECOMPILE_LIMIT, "64")
    with pytest.raises(RuntimeError, match="accumulated_recompile_limit"):
        helper(torch.nn.Identity())
    assert config.recompile_limit == 8
    assert setup_calls == []


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_cached_wrapper_still_validates_limits(
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[tuple[Any, dict[str, Any]]],
    helper: Callable[..., Any],
) -> None:
    module = torch.nn.Identity()
    compiled = helper(module)
    monkeypatch.setenv(ACCUMULATED_RECOMPILE_LIMIT, "invalid")
    with pytest.raises(ValueError, match=ACCUMULATED_RECOMPILE_LIMIT):
        helper(compiled)
    assert len(compiler_calls) == 1


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_disabled_does_not_configure(
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[tuple[Any, dict[str, Any]]],
    helper: Callable[..., Any],
) -> None:
    monkeypatch.setenv(RECOMPILE_LIMIT, "invalid")
    module = torch.nn.Identity()
    assert helper(module, policy=compile_cache.CompilePolicy(enabled=False)) is module
    assert compiler_calls == []


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_cached_wrapper_respects_policy_and_options(
    compiler_calls: list[tuple[Any, dict[str, Any]]], helper: Callable[..., Any]
) -> None:
    module = torch.nn.Identity()
    first = helper(module, options={"max_autotune": False})
    assert helper(first, options={"max_autotune": False}) is first
    assert helper(module, policy=compile_cache.CompilePolicy(mode=None), options={"max_autotune": False}) is first
    assert compiler_calls[0][1] == {"backend": "inductor", "options": {"max_autotune": False}}
    second = helper(first, options={"max_autotune": True})
    assert second is not first
    policy = compile_cache.CompilePolicy(backend="eager", mode=None, fullgraph=True, dynamic=False)
    third = helper(second, policy=policy, options={"max_autotune": True})
    assert third is not second
    assert helper(module, policy=policy, options={"max_autotune": True}) is third
    assert len(compiler_calls) == 3
    assert all(target is module for target, _ in compiler_calls)
    assert compiler_calls[-1][1] == {
        "backend": "eager",
        "fullgraph": True,
        "dynamic": False,
        "options": {"max_autotune": True},
    }


def test_compile_bound_method_implementation_identity(
    compiler_calls: list[tuple[Any, dict[str, Any]]],
) -> None:
    def make_forward(offset: int) -> Callable[..., torch.Tensor]:
        def forward(self: Any, value: torch.Tensor) -> torch.Tensor:
            return value + offset

        return forward

    owner = SimpleNamespace()
    owner.forward = MethodType(make_forward(1), owner)
    first = compile_cache.compile_callable_cached(owner.forward)
    assert compile_cache.compile_callable_cached(owner.forward) is first
    assert compile_cache.compile_callable_cached(first) is first
    owner.forward = MethodType(make_forward(2), owner)
    second = compile_cache.compile_callable_cached(owner.forward)
    assert second is not first
    assert len(compiler_calls) == 2


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("failure", ("setup", "compile", "unavailable"))
def test_compile_setup_and_errors_obey_strict(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    helper: Callable[..., Any],
    failure: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("compile setup failed")

    if failure == "setup":
        monkeypatch.setattr(compile_cache, "configure_persistent_compile_cache", fail)
    else:
        monkeypatch.setattr(torch, "compile", fail if failure == "compile" else None)
    module = torch.nn.Identity()
    with pytest.raises(RuntimeError):
        helper(module, strict=True)
    assert helper(module) is module
    assert "falling back to eager execution" in caplog.text
    assert not getattr(module, "_worldfoundry_compiled_variants", {})


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_failed_attempt_can_retry(monkeypatch: pytest.MonkeyPatch, helper: Callable[..., Any]) -> None:
    calls = 0
    wrapper = object()

    def compile_target(target: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("setup failed")
        return wrapper

    monkeypatch.setattr(torch, "compile", compile_target)
    module = torch.nn.Identity()
    assert helper(module) is module
    assert helper(module) is wrapper
    assert helper(module) is wrapper
    assert calls == 2


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("limits, expected_graphs", ((None, 8), ((16, 64), 10), ((16, 4), 4)))
def test_compile_guarded_cpu_variants(
    monkeypatch: pytest.MonkeyPatch,
    helper: Callable[..., Any],
    limits: tuple[int, int] | None,
    expected_graphs: int,
) -> None:
    from torch._dynamo import config

    limit_name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    accumulated_name = (
        "accumulated_recompile_limit"
        if hasattr(config, "accumulated_recompile_limit")
        else "accumulated_cache_size_limit"
    )
    setattr(config, limit_name, 8)
    setattr(config, accumulated_name, 256)
    if limits is not None:
        monkeypatch.setenv(RECOMPILE_LIMIT, str(limits[0]))
        monkeypatch.setenv(ACCUMULATED_RECOMPILE_LIMIT, str(limits[1]))
    graphs = 0

    def counting_backend(graph: GraphModule, inputs: list[torch.Tensor]) -> Callable[..., Any]:
        nonlocal graphs
        graphs += 1
        return graph.forward

    original_compile = torch.compile

    def compile_target(target: Any, **kwargs: Any) -> Any:
        kwargs["backend"] = counting_backend
        return original_compile(target, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_target)

    class GuardedModule(torch.nn.Module):
        def forward(self, value: torch.Tensor, branch: int) -> torch.Tensor:
            return value + branch

    module = GuardedModule()
    target = module if helper is compile_cache.compile_module_cached else module.forward
    policy = compile_cache.CompilePolicy(mode=None, dynamic=False)
    compiled = helper(target, policy=policy, strict=True)
    value = torch.ones(2)
    for branch in range(10):
        torch.testing.assert_close(compiled(value, branch), module(value, branch), rtol=0, atol=0)
    assert graphs == expected_graphs
    assert (getattr(config, limit_name), getattr(config, accumulated_name)) == (limits or (8, 256))
    assert helper(target, policy=policy, strict=True) is compiled
    torch.testing.assert_close(compiled(value, 0), module(value, 0), rtol=0, atol=0)
    assert graphs == expected_graphs


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_nondefault_mode_and_options_fail_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    compiler_calls: list[tuple[Any, dict[str, Any]]],
    helper: Callable[..., Any],
) -> None:
    setup_calls: list[str] = []
    monkeypatch.setattr(compile_cache, "_configure_recompile_limits", lambda: setup_calls.append("limits"))
    monkeypatch.setattr(
        compile_cache, "configure_persistent_compile_cache", lambda **kwargs: setup_calls.append("cache")
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        helper(
            torch.nn.Identity(),
            policy=compile_cache.CompilePolicy(mode="reduce-overhead"),
            options={"max_autotune": True},
        )
    assert setup_calls == []
    assert compiler_calls == []


@pytest.mark.parametrize("helper", HELPERS)
def test_compile_options_with_default_mode_cpu(monkeypatch: pytest.MonkeyPatch, helper: Callable[..., Any]) -> None:
    backend_options: list[dict[str, Any]] = []

    def counting_backend(
        graph: GraphModule, inputs: list[torch.Tensor], *, options: dict[str, Any]
    ) -> Callable[..., Any]:
        backend_options.append(options)
        return graph.forward

    original_compile = torch.compile

    def compile_target(target: Any, **kwargs: Any) -> Any:
        kwargs["backend"] = counting_backend
        return original_compile(target, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_target)
    module = torch.nn.ReLU()
    target = module if helper is compile_cache.compile_module_cached else module.forward
    options = {"test_option": True}
    compiled = helper(target, options=options, strict=True)
    inputs = torch.tensor([-1.0, 2.0])
    torch.testing.assert_close(compiled(inputs), module(inputs), rtol=0, atol=0)
    assert helper(target, policy=compile_cache.CompilePolicy(mode=None), options=options, strict=True) is compiled
    assert backend_options == [options]
