# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the in-tree GR00T-N1.7 runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from worldfoundry.synthesis.action_generation.gr00t import policy as gr00t_policy
from worldfoundry.synthesis.action_generation.gr00t import runtime as gr00t_runtime

pytestmark = pytest.mark.unit


def _write_minimal_gr00t_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "model_name": "nvidia/Cosmos-Reason2-2B"}),
        encoding="utf-8",
    )
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "modality_configs": {
                        "libero": {
                            "video": {"modality_keys": ["image"]},
                            "language": {"modality_keys": ["annotation.human.task_description"]},
                            "state": {"modality_keys": ["x", "gripper"]},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "statistics.json").write_text(
        json.dumps({"libero": {"state": {"x": {"mean": [0.0]}, "gripper": {"mean": [0.0]}}}}),
        encoding="utf-8",
    )
    return root


def test_resolve_observation_coerces_json_state_lists_to_float32_arrays(tmp_path: Path) -> None:
    ckpt_dir = _write_minimal_gr00t_checkpoint(tmp_path / "gr00t")
    image = Image.new("RGB", (16, 16), color=(12, 34, 56))
    obs, _ = gr00t_runtime._resolve_observation(
        instruction="pick up the cup",
        image=image,
        gr00t_observation={
            "state": {
                "x": [[[-0.05]]],
                "gripper": [[[0.25]]],
            }
        },
        checkpoint_dir=ckpt_dir,
        embodiment_value="libero",
    )
    assert isinstance(obs["state"]["x"], np.ndarray)
    assert obs["state"]["x"].dtype == np.float32
    assert obs["state"]["x"].shape == (1, 1, 1)
    assert np.isclose(obs["state"]["x"][0, 0, 0], -0.05)
    assert isinstance(obs["state"]["gripper"], np.ndarray)
    assert obs["state"]["gripper"].dtype == np.float32
    assert np.isclose(obs["state"]["gripper"][0, 0, 0], 0.25)


def test_local_hfd_repo_finds_sibling_and_ckpt_dir_backbone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ckpt_root = tmp_path / "ckpts"
    model_dir = ckpt_root / "nvidia--GR00T-N1.7-LIBERO" / "libero_10"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}),
        encoding="utf-8",
    )
    backbone = ckpt_root / "nvidia--Cosmos-Reason2-2B"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text("{}", encoding="utf-8")
    (backbone / "model.safetensors").write_bytes(b"")

    empty_hfd = tmp_path / "empty_hfd"
    empty_hfd.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(empty_hfd))
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(ckpt_root))

    kwargs = gr00t_policy._local_backbone_kwargs(model_dir)
    assert Path(kwargs["model_name"]) == backbone.resolve()

    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "other_ckpt_dir"))
    kwargs_from_parent = gr00t_policy._local_backbone_kwargs(model_dir)
    assert Path(kwargs_from_parent["model_name"]) == backbone.resolve()
