# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the vendored AlayaWorld v1.1 inference runtime."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from PIL import Image

from worldfoundry.synthesis.visual_generation.alayaworld_v11 import runtime as alaya_runtime


def _staged_runtime(tmp_path, monkeypatch):
    stage2b = tmp_path / "stage2b"
    stage3 = tmp_path / "stage3"
    for path in (stage2b / "transformer.pt", stage2b / "history_encoder.pt", stage3 / "lora.safetensors"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assets = {name: tmp_path / name for name in ("ltx.safetensors", "gemma", "vigeo.pt")}
    for path in assets.values():
        path.write_bytes(b"fixture")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    return alaya_runtime.AlayaWorldV11Runtime(
        checkpoint_dir=stage2b,
        student_dir=stage3,
        base_checkpoint=assets["ltx.safetensors"],
        gemma_path=assets["gemma"],
        vigeo_checkpoint=assets["vigeo.pt"],
        python_executable=sys.executable,
    )


def test_plan_launches_the_package_entry_module(tmp_path, monkeypatch):
    runtime = _staged_runtime(tmp_path, monkeypatch)
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(image)

    plan = runtime.build_plan(request={"prompt": "walk forward", "images": str(image)}, output_dir=tmp_path / "run")

    assert runtime.preflight()["status"] == "ready"
    assert plan.command[1:3] == ("-m", alaya_runtime.ENTRYPOINT_MODULE)
    assert plan.workdir == str(tmp_path / "run")
    assert plan.env.get("PYTHONPATH", "") == ""
    request = json.loads((tmp_path / "run" / "request.json").read_text())
    assert not {"source_root", "vigeo_source_root"} & request.keys()
    assert request["config"].endswith("models/runtime/configs/alayaworld_v11/dmd.yaml")


def test_released_config_builds_without_external_source_fields(tmp_path):
    from worldfoundry.core.io import resolve_data_path
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.inference_entry import build_config

    request = {
        "config": str(resolve_data_path("models/runtime/configs/alayaworld_v11/dmd.yaml")),
        "output_dir": str(tmp_path),
        "seed": 7,
        "base_checkpoint": "ltx.safetensors",
        "gemma_path": "gemma",
        "vigeo_checkpoint": "vigeo.pt",
        "rounds": 2,
    }

    cfg = build_config(request)

    assert cfg.spatial_memory.vigeo_checkpoint == "vigeo.pt"
    assert cfg.spatial_memory.depth_backend == "vigeo"
    assert not hasattr(cfg.spatial_memory, "vigeo_repo_path")
    assert cfg.validation.modes["custom_i2v"].rollout_rounds == 2


def test_input_preparation_runs_in_process(tmp_path):
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.prepare_i2v_inputs import main

    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(image)

    main(["--image", str(image), "--prompt", "walk", "--out", str(tmp_path / "inputs"), "--rounds", "1"])

    poses = np.load(tmp_path / "inputs" / "cam_c2w.npz")["cam_c2w"]
    assert poses.shape == (512, 4, 4)
    assert json.loads((tmp_path / "inputs" / "captions.json").read_text()) == {"frame.png": "walk"}


def test_vigeo_geometry_loads_the_shared_vigeo_model(monkeypatch):
    from worldfoundry.base_models.three_dimensions.depth.vigeo import vigeo as shared_vigeo
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.alaya.memory.vigeo_geometry import (
        ViGeoGeometryEstimator,
    )

    torch = pytest.importorskip("torch")
    loaded = []

    def from_pretrained(path):
        loaded.append(path)
        return torch.nn.Identity()

    monkeypatch.setattr(shared_vigeo.ViGeo, "from_pretrained", staticmethod(from_pretrained))
    estimator = ViGeoGeometryEstimator(checkpoint="missing/vigeo.pt", device="cpu")

    assert isinstance(estimator._load_model(), torch.nn.Identity)
    assert loaded == ["missing/vigeo.pt"]
