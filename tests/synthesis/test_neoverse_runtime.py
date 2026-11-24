# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.neoverse import runtime_env
from worldfoundry.synthesis.visual_generation.neoverse.worldfoundry_runtime import NeoVerseOfficialRuntime

pytestmark = pytest.mark.unit


def test_camera_trajectory_dispatch_does_not_change_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class CameraTrajectory:
        @staticmethod
        def from_predefined(name: str, **kwargs: Any) -> dict[str, Any]:
            return {"name": name, **kwargs}

    auxiliary = ModuleType("worldfoundry.synthesis.visual_generation.neoverse.native.auxiliary")
    monkeypatch.setattr(auxiliary, "CameraTrajectory", CameraTrajectory, raising=False)
    monkeypatch.setitem(sys.modules, auxiliary.__name__, auxiliary)
    runtime = NeoVerseOfficialRuntime(
        None, model_dir="/weights/neoverse", reconstructor_path="/weights/reconstructor.ckpt", lora_path=None
    )
    search_path = sys.path.copy()

    trajectory = runtime._build_camera_trajectory(predefined_trajectory="tilt_up", num_frames=9, zoom_ratio=1.2)

    assert trajectory == {
        "name": "tilt_up",
        "num_frames": 9,
        "mode": "relative",
        "angle": None,
        "distance": None,
        "orbit_radius": None,
        "zoom_ratio": 1.2,
    }
    assert sys.path == search_path


@pytest.mark.parametrize("nested", [False, True])
def test_local_model_and_auxiliary_asset_paths_are_preserved(tmp_path: Path, nested: bool) -> None:
    model = tmp_path / "NeoVerse" if nested else tmp_path
    model.mkdir(exist_ok=True)
    for name in (
        "reconstructor.ckpt",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
    ):
        (model / name).touch()
    (model / "google").mkdir()
    lora = model / "loras" / runtime_env.DEFAULT_NEOVERSE_LORA_NAME
    lora.parent.mkdir()
    lora.touch()

    assert runtime_env.resolve_neoverse_model_dir(tmp_path) == model
    assert runtime_env.resolve_neoverse_reconstructor_path(model) == model / "reconstructor.ckpt"
    assert runtime_env.resolve_neoverse_lora_path(model) == lora
    assert runtime_env.resolve_neoverse_lora_path(model, use_lora=False) is None
