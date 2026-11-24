# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the vendored original GR00T-N1 worker runtime."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worldfoundry.synthesis.action_generation.gr00t import legacy_n1

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "worldfoundry" / "synthesis" / "action_generation" / "gr00t" / "legacy_n1_runtime"


def _checkpoint(root: Path) -> Path:
    (root / "experiment_cfg").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "gr00t_n1", "action_horizon": 16}))
    (root / "model.safetensors").write_bytes(b"")
    (root / "experiment_cfg" / "metadata.json").write_text(json.dumps({"gr1": {}}))
    return root


def test_worker_is_a_vendored_package_module() -> None:
    spec = importlib.util.find_spec(legacy_n1.WORKER_MODULE)

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).is_relative_to(REPO_ROOT / "worldfoundry")
    assert (RUNTIME / "gr00t" / "model" / "policy.py").is_file()


def test_eagle2_backbone_files_are_package_data() -> None:
    """The worker builds Eagle2 from these files; its classes replace the checkpoint's remote code."""
    backbone = RUNTIME / "gr00t" / "model" / "backbone" / "eagle2_hg_model"
    data = REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "configs" / "gr00t-n1" / "eagle2_hg_model"
    config = json.loads((data / "config.json").read_text())

    assert config["vision_config"]["model_type"] == "siglip_vision_model"
    assert config["llm_config"]["architectures"] == ["LlamaForCausalLM"]
    for name in ("tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        assert (data / name).is_file(), name
    for target in config["auto_map"].values():
        module, name = target.split(".")
        tree = ast.parse((backbone / f"{module}.py").read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body), target


def test_predict_starts_the_worker_as_a_package_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        request = json.loads(Path(command[-1]).read_text())
        Path(request["output_path"]).write_text(json.dumps({"status": "success"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(legacy_n1.subprocess, "run", fake_run)
    result = legacy_n1.predict_n1_action(
        checkpoint_dir=_checkpoint(tmp_path / "GR00T-N1-2B"),
        python_executable=sys.executable,
        data_config_name="gr1_arms_only",
        embodiment_tag="gr1",
        device="cuda",
        seed=0,
        instruction="pick up the apple",
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        observation={"state": {"left_arm": np.zeros(7)}},
        output_path=tmp_path / "out" / "trace.json",
        run_dir=tmp_path / "run",
    )

    (call,) = calls
    command = call["command"]
    assert isinstance(command, list)
    assert isinstance(call["env"], dict)
    assert command[:3] == [sys.executable, "-m", legacy_n1.WORKER_MODULE]
    assert Path(command[3]).name == "request.json"
    assert call["env"].get("PYTHONPATH") == os.environ.get("PYTHONPATH")
    assert call["env"]["HF_HUB_OFFLINE"] == "1"
    assert result["status"] == "success"
