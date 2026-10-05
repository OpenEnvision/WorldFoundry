"""BaseSynthesis discovery and optional Torch handling need no tensor dependencies."""

from __future__ import annotations

import subprocess
import sys

import pytest

from worldfoundry.synthesis.base_synthesis import _lazy_no_grad


def test_import_does_not_load_torch_and_prediction_works_when_torch_is_absent():
    script = '''
import builtins, sys
from worldfoundry.synthesis.base_synthesis import BaseSynthesis, _lazy_no_grad
assert "torch" not in sys.modules
original_import = builtins.__import__
def unavailable(name, *args, **kwargs):
    if name == "torch":
        raise ModuleNotFoundError("Torch is not installed", name="torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = unavailable
@_lazy_no_grad
def operation(value): return value + 1
assert operation(2) == 3
try:
    BaseSynthesis().predict()
except NotImplementedError:
    pass
else:
    raise AssertionError("BaseSynthesis.predict lost its abstract contract")
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_broken_torch_dependency_is_not_silently_treated_as_an_absent_torch(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("A Torch dependency is broken", name="broken_torch_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    invoked = []

    @_lazy_no_grad
    def operation():
        invoked.append(True)

    with pytest.raises(ModuleNotFoundError, match="dependency is broken"):
        operation()
    assert not invoked
