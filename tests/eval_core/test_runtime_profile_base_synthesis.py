from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_profile_synthesis_inherits_real_base_synthesis() -> None:
    code = """
import sys
import types

fake_torch = types.ModuleType("torch")

def no_grad():
    def decorator(function):
        return function
    return decorator

fake_torch.no_grad = no_grad
sys.modules["torch"] = fake_torch

from worldfoundry.synthesis import base_synthesis
from worldfoundry.evaluation.models.runtime import profiles as runtime_profiles

assert runtime_profiles.BaseSynthesis is base_synthesis.BaseSynthesis
assert runtime_profiles.RuntimeProfileSynthesis.__mro__[1] is base_synthesis.BaseSynthesis
assert runtime_profiles.BaseSynthesis.__module__ == "worldfoundry.synthesis.base_synthesis"
assert runtime_profiles.BaseSynthesis.__module__ != runtime_profiles.__name__
assert not hasattr(runtime_profiles, "BASE_SYNTHESIS_OPTIONAL_FALLBACK")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
