from __future__ import annotations

import pytest

from worldfoundry.core.kernels.registry import (
    KernelNotSupported,
    KernelRegistry,
    kernel_dispatch_receipt_scope,
)


def test_kernel_dispatch_receipt_records_concrete_candidate(monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")
    registry = KernelRegistry()
    registry.register(
        "toy",
        backend="test",
        name="toy_accelerated",
        implementation=lambda value: value * 2,
        predicate=lambda _value: True,
    )
    receipt: dict[str, object] = {}

    with kernel_dispatch_receipt_scope(receipt):
        result = registry.dispatch(
            "toy",
            lambda value: value + 1,
            3,
            signature=("shape",),
        )

    assert result == 6
    assert receipt["dispatches"] == [
        {
            "op": "toy",
            "implementation": "toy_accelerated",
            "backend": "test",
            "accelerated": True,
            "fallback": False,
            "cache_hit": False,
            "failures": [],
            "quarantined": [],
            "reason": None,
        }
    ]


def test_kernel_dispatch_receipt_records_explicit_torch_fallback(monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    registry = KernelRegistry()
    receipt: dict[str, object] = {}

    with kernel_dispatch_receipt_scope(receipt):
        result = registry.dispatch(
            "toy",
            lambda value: value + 1,
            3,
            signature=("shape",),
        )

    assert result == 4
    dispatch = receipt["dispatches"][0]
    assert dispatch["implementation"] == "torch"
    assert dispatch["accelerated"] is False
    assert dispatch["fallback"] is True
    assert dispatch["reason"] == "torch backend explicitly requested"


def test_kernel_dispatch_receipt_retains_candidate_failure(monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")
    registry = KernelRegistry()

    def unsupported(_value):
        raise KernelNotSupported("unsupported test shape")

    registry.register(
        "toy",
        backend="test",
        name="toy_broken",
        implementation=unsupported,
        predicate=lambda _value: True,
    )
    receipt: dict[str, object] = {}

    with pytest.warns(RuntimeWarning, match="toy_broken"):
        with kernel_dispatch_receipt_scope(receipt):
            result = registry.dispatch(
                "toy",
                lambda value: value + 1,
                3,
                signature=("shape",),
            )

    assert result == 4
    dispatch = receipt["dispatches"][0]
    assert dispatch["implementation"] == "torch"
    assert dispatch["fallback"] is True
    assert len(dispatch["failures"]) == 1
    assert "toy_broken" in dispatch["failures"][0]
