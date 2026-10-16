# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Check argument compatibility without executing the inspected callable."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Literal

import pytest

from worldfoundry.core.utils.python.functional_utils import (
    ClassRegistry,
    call_once,
    is_signature_compatible,
    make_recursive_func,
)

pytestmark = pytest.mark.unit


def _takes_value(value: int, *, factor: int = 1) -> int:
    raise AssertionError("Signature checking must not execute the callable")


@pytest.mark.parametrize(
    ("args", "kwargs", "compatible"),
    [
        ((2,), {}, True),
        ((), {"value": 2}, True),
        ((2,), {"factor": 3}, True),
        ((), {}, False),
        ((2, 3), {}, False),
        ((2,), {"unexpected": 3}, False),
    ],
)
def test_signature_compatibility(args: tuple[int, ...], kwargs: dict[str, int], compatible: bool) -> None:
    assert is_signature_compatible(_takes_value, *args, **kwargs) is compatible


def test_signature_compatibility_supports_variadic_arguments() -> None:
    def accepts_all(*args: object, **kwargs: object) -> None:
        raise AssertionError("Signature checking must not execute the callable")

    assert is_signature_compatible(accepts_all, 1, 2, label="batch")


def test_signature_compatibility_honors_builtin_positional_only_arguments() -> None:
    assert is_signature_compatible(abs, -1)
    assert not is_signature_compatible(abs, value=-1)


@pytest.mark.parametrize("mode", ["noop", "warn", "raise"])
def test_call_once_preserves_second_call_policy(mode: Literal["noop", "warn", "raise"]) -> None:
    calls: list[int] = []

    @call_once(on_second_call=mode)
    def initialize(value: int) -> int:
        calls.append(value)
        return value + 1

    assert initialize(1) == 2
    if mode == "raise":
        with pytest.raises(RuntimeError, match="initialize has already been called"):
            initialize(2)
    elif mode == "warn":
        with pytest.warns(UserWarning, match="initialize has already been called"):
            assert initialize(2) is None
    else:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            assert initialize(2) is None
        assert not recorded
    assert calls == [1]
    assert initialize.__name__ == "initialize"


def test_call_once_has_independent_state_and_counts_failed_attempts() -> None:
    calls: list[str] = []

    @call_once
    def fail() -> None:
        calls.append("fail")
        raise ValueError("initialization failed")

    @call_once()
    def succeed() -> str:
        calls.append("succeed")
        return "ready"

    with pytest.raises(ValueError, match="initialization failed"):
        fail()
    assert fail() is None
    assert succeed() == "ready"
    assert succeed() is None
    assert calls == ["fail", "succeed"]


def test_recursive_function_walks_nested_values_without_dm_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tree", None)

    @make_recursive_func
    def increment(value: int) -> int:
        return value + 1

    assert increment({"items": [1, (2, 3)], "last": 4}) == {"items": [2, (3, 4)], "last": 5}


def test_recursive_paths_require_dm_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tree", None)

    @make_recursive_func(with_path=True)
    def record(path: tuple[str, ...], value: int) -> tuple[tuple[str, ...], int]:
        return path, value

    with pytest.raises(ImportError, match="dm_tree is required"):
        record({"value": 1})


@pytest.mark.parametrize("with_path", [False, True])
def test_recursive_function_delegates_to_available_dm_tree(monkeypatch: pytest.MonkeyPatch, with_path: bool) -> None:
    module = ModuleType("tree")
    calls: list[str] = []

    def map_structure(function: Callable[..., object], structure: object) -> object:
        calls.append("map")
        return function(structure)

    def map_structure_with_path(function: Callable[..., object], structure: object) -> object:
        calls.append("path")
        return function(("value",), structure)

    setattr(module, "map_structure", map_structure)
    setattr(module, "map_structure_with_path", map_structure_with_path)
    monkeypatch.setitem(sys.modules, "tree", module)
    if with_path:
        wrapped = make_recursive_func(with_path=True)(lambda path, value: (path, value + 1))
        assert wrapped(1) == (("value",), 2)
        assert calls == ["path"]
    else:
        wrapped = make_recursive_func(lambda value: value + 1)
        assert wrapped(1) == 2
        assert calls == ["map"]


@pytest.mark.parametrize("base_name", [None, "Base"])
def test_class_registry_accepts_optional_base_name(base_name: str | None) -> None:
    class Entry:
        def __init__(self, value: int) -> None:
            self.value = value

    registry = ClassRegistry(base_name)
    registry.add(Entry)
    assert registry["Entry"] is Entry
    assert registry.instantiate("Entry", value=7).value == 7
    with pytest.raises(KeyError) as error:
        registry.get("missing")
    prefix = "Base subclass" if base_name else "subclass"
    assert f"{prefix} 'missing' not found" in str(error.value)
