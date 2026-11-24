# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""minWM package commands preserve both Action2V input and launch contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.minwm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


def _assets(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    checkpoint, base = tmp_path / "checkpoint", tmp_path / "base"
    checkpoint_files = (
        ["HY15/Action2V/dmd/config.json", "HY15/Action2V/dmd/diffusion_pytorch_model.safetensors"]
        if mode == "hy"
        else ["Wan21/Action2V/dmd/model.pt"]
    )
    base_files = (
        ["config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
        if mode == "hy"
        else ["config.json", "diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"]
    )
    for root, files in ((checkpoint, checkpoint_files), (base, base_files)):
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return checkpoint, base


def test_minwm_hy_preserves_recipe_and_json_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint, base = _assets(tmp_path, "hy")
    image = tmp_path / "image.png"
    image.touch()
    captured: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        captured.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    model = runtime.MinWMHYAction2VRuntime(
        checkpoint_path=checkpoint, base_model_path=base, python_executable="/env/python", env={"CUSTOM": "value"}
    )
    output = tmp_path / "result.mp4"
    result = model.predict(
        prompt="An orchard",
        image_path=image,
        trajectory="w*4, d*2",
        output_path=output,
        token_mapping_details={"runtime_yaw_deg_per_token": 2.0},
        num_frames=41,
        seed=19,
        return_dict=False,
    )
    command = captured["command"]
    assert command[:7] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "-m",
        "worldfoundry.synthesis.visual_generation.minwm.hy15_inference",
    ]
    assert "--standalone" not in command
    assert command[command.index("--video_length") + 1] == "41"
    assert command[command.index("--seed") + 1] == "19"
    assert command[command.index("--shift") + 1] == "5.0"
    assert command[command.index("--guidance_scale") + 1] == "1.0"
    assert "--use_camera" in command
    example = Path(command[command.index("--example_json") + 1])
    assert json.loads(example.read_text()) == [
        {"id": 0, "image": str(image), "caption": "An orchard", "trajectory": "w*4, d*2"}
    ]
    assert captured["env"] == {"CUSTOM": "value", "WORLDFOUNDRY_MINWM_ROT_STEP_DEG": "2.0"}
    assert "python_paths" not in captured
    assert captured["cwd"] == runtime._PROJECT_ROOT
    assert result == str(output)


@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("master_port", [None, 29431])
def test_minwm_wan_preserves_rendezvous_and_sequence_parallelism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sp_size: int, master_port: int | None
) -> None:
    checkpoint, base = _assets(tmp_path, "wan")
    captured: dict[str, Any] = {}
    port_calls: list[bool] = []

    def free_port() -> int:
        port_calls.append(True)
        return 30123

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        captured.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    monkeypatch.setattr(runtime, "find_free_port", free_port)
    model = runtime.MinWMWanAction2VRuntime(
        checkpoint_path=checkpoint, base_model_path=base, python_executable="/env/python"
    )
    model.predict(
        prompt=" Orchard \n",
        trajectory=" w*19 \n",
        output_path=tmp_path / "wan.mp4",
        sp_size=sp_size,
        master_port=master_port,
        num_output_frames=12,
        seed=19,
    )
    command = captured["command"]
    assert command[:9] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--master_port",
        str(master_port or 30123),
        "--nproc_per_node",
        str(sp_size),
        "-m",
        "worldfoundry.synthesis.visual_generation.minwm.minwm_runtime.Wan21.wan_inference",
    ]
    assert port_calls == ([True] if master_port is None else [])
    assert command[command.index("--sp_size") + 1] == str(sp_size)
    assert command[command.index("--num_output_frames") + 1] == "12"
    assert Path(command[command.index("--data_path") + 1]).read_text() == "Orchard\n"
    assert Path(command[command.index("--trajectory_path") + 1]).read_text() == "w*19\n"
    assert captured["env"] == {
        "SP_SIZE": str(sp_size),
        "WORLDFOUNDRY_MINWM_ROT_STEP_DEG": "3.0",
        "WORLDFOUNDRY_MINWM_WAN_BASE_MODEL": str(base),
    }
    assert "python_paths" not in captured


def test_minwm_wan_rejects_invalid_sp_size_before_loading() -> None:
    model = runtime.MinWMWanAction2VRuntime()
    with pytest.raises(ValueError, match="sp_size must be >= 1"):
        model.predict(prompt="orchard", trajectory="w", output_path="out.mp4", sp_size=0)
