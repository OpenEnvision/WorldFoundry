# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free LiveWorld package-launch and asset configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from worldfoundry.synthesis.visual_generation.liveworld import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


def _asset(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


@pytest.mark.parametrize("nested_checkpoint", [False, True])
def test_liveworld_predict_launches_package_with_local_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested_checkpoint: bool
) -> None:
    checkpoint = tmp_path / "liveworld"
    weights = checkpoint / "ckpts" if nested_checkpoint else checkpoint
    for relative in ("lora/model.pt", "state_adapter/model.pt"):
        _asset(weights, relative)
    base = tmp_path / "wan"
    for relative in (
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ):
        _asset(base, relative)
    qwen = tmp_path / "qwen"
    for relative in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        _asset(qwen, relative)
    sam3 = _asset(tmp_path / "sam3", "sam3.pt")
    stream3r = tmp_path / "stream3r"
    dinov3 = tmp_path / "dinov3"
    for root in (stream3r, dinov3):
        for relative in ("config.json", "model.safetensors"):
            _asset(root, relative)
    geometry = _asset(tmp_path, "geometry.npz")
    image = _asset(tmp_path, "first.png")
    output = tmp_path / "output.mp4"
    captured: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, Any]:
        captured.update(command=command, **kwargs)
        return {"status": "succeeded", "artifact_path": str(output)}

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    model = runtime.LiveWorldRuntime(
        checkpoint_path=checkpoint,
        base_model_path=base,
        qwen_model_path=qwen,
        sam3_model_path=sam3,
        stream3r_model_path=stream3r,
        dinov3_model_path=dinov3,
        python_executable="/env/bin/python",
        device="cuda:1",
        env={"TOKENIZERS_PARALLELISM": "false"},
    )

    result = model.predict(
        prompt="camera move",
        geometry_npz=geometry,
        image_path=image,
        output_path=output,
        width=640,
        height=384,
        num_frames=97,
        fps=12,
        seed=17,
    )

    command = captured["command"]
    assert command[:3] == [
        "/env/bin/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.scripts.infer",
    ]
    assert command[command.index("--device") + 1] == "cuda:1"
    assert "python_paths" not in captured
    assert captured["env"] == {"TOKENIZERS_PARALLELISM": "false"}
    sample = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert sample["geometry_file_name"] == str(geometry)
    assert sample["first_frame_image"] == str(image)
    assert sample["iter_input"] == {"0": {"scene_text": "camera move", "fg_text": ""}}
    system = yaml.safe_load(Path(command[command.index("--system-config") + 1]).read_text())
    assert system["runtime"]["seed"] == 17
    assert system["observer"]["lora_path"] == str(weights / "lora/model.pt")
    assert system["observer"]["sp_path"] == str(weights / "state_adapter/model.pt")
    assert system["observer"]["frames_per_iter"] == 65
    assert (system["observer"]["width"], system["observer"]["height"], system["observer"]["fps"]) == (640, 384, 12)
    assert system["auxiliary"]["qwen_model_path"] == str(qwen)
    assert system["auxiliary"]["sam3_model_path"] == str(sam3)
    assert system["auxiliary"]["stream3r_model_path"] == str(stream3r)
    assert system["event"]["deduplication"]["model_path"] == str(dinov3)
    observer = yaml.safe_load(Path(system["observer"]["config"]).read_text())
    assert observer["wan_model_name"] == str(base)
    assert result == {"status": "succeeded", "artifact_path": str(output)}
