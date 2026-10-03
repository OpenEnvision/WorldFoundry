"""Catalog discovery must not attempt to import inference tensor libraries."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("asset_gates", ["0", "1"])
def test_catalog_discovery_does_not_attempt_torch_or_diffusers_imports(asset_gates):
    script = '''
import builtins, importlib, json, sys
attempts = []
original_import, original_module = builtins.__import__, importlib.import_module
def check(name):
    if name.split(".", 1)[0] in {"torch", "diffusers"}:
        attempts.append(name)
        raise ModuleNotFoundError("Inference dependency unavailable", name=name)
def guarded_import(name, *args, **kwargs):
    check(name)
    return original_import(name, *args, **kwargs)
def guarded_module(name, *args, **kwargs):
    check(name)
    return original_module(name, *args, **kwargs)
builtins.__import__, importlib.import_module = guarded_import, guarded_module
from worldfoundry.runtime.inference_catalog import get_model_inference_spec, list_model_inference_specs
specs = list_model_inference_specs()
assert specs and get_model_inference_spec(specs[0].model_family_id) is specs[0]
assert not attempts, attempts
assert "torch" not in sys.modules and "diffusers" not in sys.modules
print(json.dumps({"model_count": len(specs), "blocked_import_attempts": attempts}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env={**os.environ, "WORLDFOUNDRY_INFERENCE_LOAD_RUNTIME_ASSET_GATES": asset_gates},
    )
    assert result.returncode == 0, result.stdout + result.stderr
