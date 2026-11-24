# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from worldfoundry.runtime.conda import load_runtime_conda_env_spec
from worldfoundry.synthesis.visual_generation.scope.worldfoundry_runtime import SCOPERuntime
from worldfoundry.zoo.runtime.environments import load_runtime_environment_profile_by_id

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pythonpath", [None, "/operator/packages"])
def test_subprocess_env_preserves_operator_pythonpath(monkeypatch: pytest.MonkeyPatch, pythonpath: str | None) -> None:
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    original = dict(os.environ)

    env = SCOPERuntime(model_dir="/weights/scope", device="cuda:1")._subprocess_env()

    assert env.get("PYTHONPATH") == pythonpath
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["SCOPE_DEVICE"] == "cuda"
    assert dict(os.environ) == original


def test_argv_selects_installed_module_and_preserves_sampler_flags(tmp_path: Path) -> None:
    runtime = SCOPERuntime(model_dir="/weights/scope", python_executable=sys.executable)
    command = runtime._argv(
        image_path=tmp_path / "image.png",
        action_path=tmp_path / "actions.parquet",
        output_dir=tmp_path / "output",
        prompt="a camera in a garden",
        height=480,
        width=832,
        max_frames=81,
        num_inference_steps=30,
        seed=7,
    )

    assert command[:3] == [
        sys.executable,
        "-m",
        "worldfoundry.synthesis.visual_generation.scope.scope_runtime.inference",
    ]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--model_dir": "/weights/scope",
        "--input_image": str(tmp_path / "image.png"),
        "--action_path": str(tmp_path / "actions.parquet"),
        "--prompt": "a camera in a garden",
        "--output_dir": str(tmp_path / "output"),
        "--height": "480",
        "--width": "832",
        "--max_frames": "81",
        "--num_inference_steps": "30",
        "--seed": "7",
    }


def test_scope_environment_readers_default_to_no_source_paths() -> None:
    environment = load_runtime_environment_profile_by_id("scope")
    conda = load_runtime_conda_env_spec("scope")

    assert environment.pythonpath_dirs == ()
    assert conda is not None
    assert conda.pythonpath_dirs == ()
    assert "worldfoundry.synthesis.visual_generation.scope.scope_runtime.native_pipeline" in conda.validation_imports
