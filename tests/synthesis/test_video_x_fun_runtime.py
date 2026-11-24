# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.visual_generation.video_x_fun import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.video_x_fun.runtime_request import (
    CameraInferenceRequest,
    parse_camera_request,
)

pytestmark = pytest.mark.unit

RUNTIME_CLASSES = (
    runtime.Wan21Fun1P3BCameraRuntime,
    runtime.Wan21Fun14BCameraRuntime,
    runtime.Wan22Fun5BCameraRuntime,
    runtime.Wan22FunA14BCameraRuntime,
)
MEMORY_MODES = (
    "model_full_load",
    "model_full_load_and_qfloat8",
    "model_cpu_offload",
    "model_cpu_offload_and_qfloat8",
    "model_group_offload",
    "sequential_cpu_offload",
)


def _request(tmp_path: Path, config: str) -> CameraInferenceRequest:
    return {
        "GPU_memory_mode": "sequential_cpu_offload",
        "config_path": str(resolve_data_path("models", "runtime", "configs", "video_x_fun", config)),
        "model_name": str(tmp_path / "checkpoint"),
        "sample_size": [480, 832],
        "video_length": 81,
        "fps": 16,
        "control_video": None,
        "control_camera_txt": str(tmp_path / "pose.txt"),
        "start_image": str(tmp_path / "image.png"),
        "ref_image": None,
        "prompt": "a camera orbit",
        "seed": 7,
        "num_inference_steps": 5,
        "enable_teacache": False,
        "save_path": str(tmp_path / "output"),
    }


@pytest.mark.parametrize("runtime_class", RUNTIME_CLASSES)
@pytest.mark.parametrize("memory_mode", MEMORY_MODES)
def test_video_x_fun_launches_a_data_only_request_for_every_camera_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_class: type[runtime._WanFunCameraRuntime],
    memory_mode: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    for relative in runtime_class.REQUIRED_COMPONENT_FILES:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    pose = tmp_path / "pose.txt"
    pose.touch()
    captured: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        captured.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    model = runtime_class(
        checkpoint_path=checkpoint, python_executable="python", device="cuda:2", gpu_memory_mode=memory_mode
    )
    model.predict(
        prompt="  a camera orbit  ",
        image_path=image,
        pose_txt=pose,
        output_path=tmp_path / "video.mp4",
        num_inference_steps=5,
        seed=7,
        enable_teacache=False,
    )

    command = captured["command"]
    module = (
        "worldfoundry.synthesis.visual_generation.video_x_fun.video_x_fun_runtime."
        + model.ENTRYPOINT.removesuffix(".py").replace("/", ".")
    )
    assert command[:4] == ["python", "-m", module, "--request"]
    request_path = Path(command[4])
    expected = _request(tmp_path, model.CONFIG)
    expected["GPU_memory_mode"] = memory_mode
    expected["save_path"] = str(request_path.parent / "outputs")
    assert json.loads(request_path.read_text()) == expected
    monkeypatch.setattr(sys, "argv", [module, "--request", str(request_path)])
    assert parse_camera_request() == expected
    assert captured["env"] == {"CUDA_VISIBLE_DEVICES": "2"}
    assert "python_paths" not in captured
    assert request_path.name == "request.json"
    assert not (request_path.parent / "predict.py").exists()
    assert (model.repo_root / model.plan()["entrypoint"]).is_file()


@pytest.mark.parametrize(
    "runtime_class",
    [runtime.Wan21Fun1P3BCameraRuntime, runtime.Wan22Fun5BCameraRuntime, runtime.Wan22FunA14BCameraRuntime],
)
@pytest.mark.parametrize("supports_bf16", [False, True])
def test_video_x_fun_selects_dtype_in_the_child_before_loading_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_class: type[runtime._WanFunCameraRuntime],
    supports_bf16: bool,
) -> None:
    module = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.video_x_fun.video_x_fun_runtime."
        + runtime_class.ENTRYPOINT.removesuffix(".py").replace("/", ".")
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supports_bf16)
    monkeypatch.setattr(module, "set_multi_gpus_devices", lambda *_args: "cpu")
    captured: dict[str, Any] = {}

    class StopBeforeWeights(Exception):
        pass

    def load_transformer(*args: Any, **kwargs: Any) -> None:
        captured.update(args=args, **kwargs)
        raise StopBeforeWeights

    monkeypatch.setattr(module, "load_wan_transformer", load_transformer)
    original_path = list(sys.path)

    with pytest.raises(StopBeforeWeights):
        module.run_camera(_request(tmp_path, runtime_class.CONFIG))

    assert captured["torch_dtype"] == (torch.bfloat16 if supports_bf16 else torch.float16)
    assert ("low_cpu_mem_usage" in captured) is (runtime_class is not runtime.Wan22Fun5BCameraRuntime)
    assert sys.path == original_path
