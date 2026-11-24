# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.synthesis.visual_generation.animatediff.worldfoundry_runtime import AnimateDiffRuntime

pytestmark = pytest.mark.unit


def _runtime(tmp_path: Path) -> AnimateDiffRuntime:
    return AnimateDiffRuntime(
        profile=SimpleNamespace(),
        model_id="animatediff",
        device="cpu",
        motion_module_path=str(tmp_path / "motion.ckpt"),
        base_model_path=str(tmp_path / "sd15"),
        dreambooth_model_path=str(tmp_path / "dreambooth.safetensors"),
        official_python=sys.executable,
        hf_hub_cache=str(tmp_path / "hf" / "hub"),
        integrated_runtime_root=str(tmp_path / "assets"),
        inference_config=str(tmp_path / "inference.yaml"),
    )


@pytest.mark.parametrize("pythonpath", [None, "/operator/packages"])
def test_official_runtime_env_preserves_operator_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pythonpath: str | None
) -> None:
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("DIFFUSERS_OFFLINE", raising=False)
    before = dict(os.environ)

    env = _runtime(tmp_path).official_runtime_env()

    assert env.get("PYTHONPATH") == pythonpath
    assert env["HF_HOME"] == str(tmp_path / "hf")
    assert env["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "hf" / "hub")
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["TRANSFORMERS_OFFLINE"] == env["DIFFUSERS_OFFLINE"] == "1"
    assert dict(os.environ) == before


def test_ensure_pipe_imports_package_without_changing_search_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    from worldfoundry.core.io.paths import REPO_ROOT

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    search_path = [path for path in sys.path if path != str(REPO_ROOT)]
    monkeypatch.setattr(sys, "path", search_path.copy())

    with pytest.raises(RuntimeError, match="requires CUDA"):
        _runtime(tmp_path).ensure_pipe()

    assert sys.path == search_path
