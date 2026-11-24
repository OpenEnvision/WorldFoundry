# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from worldfoundry.synthesis.visual_generation.vmem import runtime_env
from worldfoundry.synthesis.visual_generation.vmem.worldfoundry_runtime import VMemRuntime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config_source", ["default", "runtime_root", "env_root", "config_path"])
def test_from_pretrained_uses_data_config_without_mutating_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_source: str
) -> None:
    from worldfoundry.synthesis.visual_generation.vmem.vmem_runtime.utils import util

    class Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def reset(self) -> None:
            pass

    pipeline_module = ModuleType("worldfoundry.synthesis.visual_generation.vmem.vmem_runtime.modeling.pipeline")
    monkeypatch.setattr(pipeline_module, "VMemPipeline", Pipeline, raising=False)
    monkeypatch.setitem(sys.modules, pipeline_module.__name__, pipeline_module)
    foreign_modules = {}
    for name in ("models", "modeling", "utils", "add_ckpt_path"):
        module = ModuleType(name)
        module.__file__ = f"/operator/{name}.py"
        foreign_modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("VMEM_RUNTIME_ROOT", raising=False)
    data_root = tmp_path / "data"
    config_path = data_root / "configs" / "inference" / "inference.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model: {height: 8, width: 12}\nsurfel: {}\n", encoding="utf-8")
    options: dict[str, Any] = {}
    if config_source == "runtime_root":
        options["runtime_root"] = str(data_root)
    elif config_source == "env_root":
        monkeypatch.setenv("VMEM_RUNTIME_ROOT", str(data_root))
    elif config_source == "config_path":
        options["config_path"] = str(config_path)
    search_path = sys.path.copy()

    runtime = VMemRuntime.from_pretrained(
        "/weights/vmem", surfel_model_path="/weights/cut3r", device="cpu", weight_dtype=torch.float32, **options
    )

    assert runtime.height == (576 if config_source == "default" else 8)
    assert runtime.config.model.model_path == "/weights/vmem"
    assert runtime.config.surfel.model_path == "/weights/cut3r"
    assert runtime.transform_img_and_K is util.transform_img_and_K
    assert runtime.get_default_intrinsics is util.get_default_intrinsics
    assert runtime.runtime_pipeline.kwargs["device"] == "cpu"
    assert runtime.runtime_pipeline.kwargs["dtype"] is torch.float32
    assert sys.path == search_path
    assert all(sys.modules[name] is module for name, module in foreign_modules.items())


def test_default_config_path_keeps_fallback_when_data_config_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VMEM_RUNTIME_ROOT", raising=False)
    default = runtime_env.default_config_path()
    assert default.is_file()
    assert runtime_env.default_config_path(tmp_path) == default
    monkeypatch.setenv("VMEM_RUNTIME_ROOT", str(tmp_path))
    assert runtime_env.default_config_path() == default


def test_camera_outlier_filtered_center_deterministic_mode() -> None:
    from worldfoundry.synthesis.visual_generation.vmem.vmem_runtime.utils.util import (
        camera_outlier_filtered_center,
    )

    c2ws = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(5, 1, 1)
    c2ws[:, :3, 3] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
        dtype=torch.float32,
    )
    prev = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        cpu_center = camera_outlier_filtered_center(c2ws)
        torch.testing.assert_close(cpu_center, torch.tensor([[2.0, 4.0, 6.0]]))
        if torch.cuda.is_available():
            cuda_center = camera_outlier_filtered_center(c2ws.cuda())
            torch.testing.assert_close(cuda_center.cpu(), cpu_center)
    finally:
        torch.use_deterministic_algorithms(prev)
