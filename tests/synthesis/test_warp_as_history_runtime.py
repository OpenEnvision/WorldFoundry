# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.warp_as_history.worldfoundry_runtime import WarpAsHistoryRuntime

pytestmark = pytest.mark.unit


def test_warp_as_history_imports_without_native_helios_transformer() -> None:
    code = """
import sys

sys.modules['worldfoundry.base_models.diffusion_model.models.networks.helios.model'] = None
from worldfoundry.synthesis.visual_generation.warp_as_history.warp_as_history_runtime.warp_as_history import infer
assert callable(infer.infer_plan_kwargs)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("tokenizers_parallelism", [None, "true"])
def test_warp_as_history_environment_preserves_import_paths(
    monkeypatch: pytest.MonkeyPatch, tokenizers_parallelism: str | None
) -> None:
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("WAH_PI3X_CKPT_PATH", raising=False)
    if tokenizers_parallelism is not None:
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", tokenizers_parallelism)
    original_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")
    runtime = WarpAsHistoryRuntime(device="cpu", pi3x_ckpt_path="/checkpoints/pi3x.safetensors")

    runtime._configure_runtime_environment()

    assert sys.path == original_path
    assert os.environ.get("PYTHONPATH") == original_pythonpath
    assert os.environ["TOKENIZERS_PARALLELISM"] == (tokenizers_parallelism or "false")
    assert os.environ["WAH_PI3X_CKPT_PATH"] == "/checkpoints/pi3x.safetensors"


def test_warp_as_history_dispatches_the_packaged_inference_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.synthesis.visual_generation.warp_as_history.warp_as_history_runtime.warp_as_history import (
        infer,
    )

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    original_path = list(sys.path)
    runtime = WarpAsHistoryRuntime(device="cpu", model_path="/checkpoints/helios", dtype="fp32")
    target = tmp_path / "output.mp4"
    payload = runtime._infer_kwargs(
        csv_path=tmp_path / "input.csv",
        target=target,
        height=384,
        width=640,
        num_frames=33,
        fps=16,
        seed=7,
        dtype="fp32",
        lora_path=None,
        warp_debug_dir=tmp_path / "warps",
        enable_optional_attention=True,
    )
    captured: dict[str, Any] = {}

    def run_infer_from_csv(**kwargs: Any) -> dict[str, str]:
        captured.update(kwargs)
        return {"event": "infer_done"}

    monkeypatch.setattr(infer, "run_infer_from_csv", run_infer_from_csv)

    assert runtime._run_infer(payload, target) == {"event": "infer_done"}
    assert captured == {
        **payload,
        "csv_path": tmp_path / "input.csv",
        "output": target,
        "warp_debug_dir": tmp_path / "warps",
    }
    assert payload["model_path"] == "/checkpoints/helios"
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "fp32"
    assert payload["no_lora"] is True
    assert payload["enable_optional_attention"] is True
    assert sys.path == original_path
