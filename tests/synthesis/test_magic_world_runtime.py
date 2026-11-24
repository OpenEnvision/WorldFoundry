# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldfoundry.synthesis.visual_generation.magic_world import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("frames", [9, 21, 81])
def test_magic_world_launches_its_package_with_the_staged_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int
) -> None:
    checkpoint = tmp_path / "magic"
    (checkpoint / "MagicWorld-Fast").mkdir(parents=True)
    (checkpoint / "MagicWorld-Fast" / "model.pt").touch()
    (checkpoint / "MagicWorld-Base" / "transformer").mkdir(parents=True)
    wan = tmp_path / "wan"
    wan.mkdir()
    depth = tmp_path / "depth_pro.pt"
    depth.touch()
    monkeypatch.setenv("DEPTH_PRO_CHECKPOINT", str(depth))
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    captured: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        captured.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    model = runtime.MagicWorldRuntime(checkpoint_path=checkpoint, base_model_path=wan, python_executable="python")
    rows = [[float(index), *([0.25] * 18)] for index in range(frames)]

    result = model.predict(
        prompt="a camera moving left",
        image_path=image,
        native_rows=rows,
        output_path=tmp_path / "video.mp4",
        num_frames=frames,
        seed=7,
    )

    command = captured["command"]
    assert command[:3] == [
        "python",
        "-m",
        "worldfoundry.synthesis.visual_generation.magic_world.magic_world_runtime.inference.inference_magicworld_fast",
    ]
    assert command[command.index("--num_output_frames") + 1] == str((frames - 1) // 4 + 1)
    assert command[command.index("--seed") + 1] == "7"
    assert "--i2v" in command
    assert "python_paths" not in captured
    assert "PYTHONPATH" not in captured["env"]
    assert captured["env"]["DEPTH_PRO_CHECKPOINT"] == str(depth)
    assert captured["env"]["WORLDFOUNDRY_MAGICWORLD_WAN_ROOT"] == str(wan)
    assert captured["env"]["WORLDFOUNDRY_MAGICWORLD_BASE_ROOT"] == str(checkpoint / "MagicWorld-Base")
    prompts = Path(command[command.index("--extended_prompt_path") + 1])
    assert json.loads(prompts.read_text()) == [{"name": "input.png", "describe": "a camera moving left"}]
    camera = Path(command[command.index("--control_camera_txt") + 1])
    assert len(camera.read_text().splitlines()) == frames + 1
    assert result == {"video": str(tmp_path / "video.mp4")}
