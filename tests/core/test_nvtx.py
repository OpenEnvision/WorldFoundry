from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from worldfoundry.core.observability.nvtx import nvtx_range


def test_disabled_range_does_not_import_torch(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_NVTX", raising=False)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name != "torch"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with nvtx_range("disabled"):
        pass


@pytest.mark.parametrize("available", [False, True])
def test_ranges_are_balanced_on_failure_without_synchronization(monkeypatch, available):
    import sys

    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: available,
        nvtx=SimpleNamespace(range_push=lambda name: calls.append(name), range_pop=lambda: calls.append("pop")),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("WORLDFOUNDRY_NVTX", "1")
    with pytest.raises(ValueError, match="original"):
        with nvtx_range("outer"), nvtx_range("inner"):
            raise ValueError("original")
    assert calls == (["outer", "inner", "pop", "pop"] if available else [])
