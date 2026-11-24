# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Package subprocess and asset contracts for VerseCrafter's three stages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from worldfoundry.synthesis.visual_generation.versecrafter import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit
_MODULES = "worldfoundry.synthesis.visual_generation.versecrafter.versecrafter_runtime.inference."


def _runtime(tmp_path: Path, memory_mode: str) -> runtime.VerseCrafterRuntime:
    checkpoint, base, moge = (tmp_path / name for name in ("checkpoint", "base", "moge"))
    for root, files in (
        (
            checkpoint,
            [
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                *(f"diffusion_pytorch_model-{i:05d}-of-00004.safetensors" for i in range(1, 5)),
            ],
        ),
        (
            base,
            [
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                "Wan2.1_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth",
            ],
        ),
        (moge, ["model.pt"]),
    ):
        root.mkdir()
        for file in files:
            (root / file).touch()
    return runtime.VerseCrafterRuntime(
        checkpoint_path=checkpoint,
        base_model_path=base,
        moge_model_path=moge,
        python_executable="/env/python",
        device="cuda:3",
        gpu_memory_mode=memory_mode,
        env={"CUSTOM": "preserved"},
    )


@pytest.mark.parametrize("memory_mode", runtime.VerseCrafterRuntime.GPU_MEMORY_MODES)
@pytest.mark.parametrize("default_trajectory", [True, False])
def test_versecrafter_package_stages_preserve_inputs_and_memory_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory_mode: str, default_trajectory: bool
) -> None:
    model = _runtime(tmp_path, memory_mode)
    image, trajectory = tmp_path / "image.png", tmp_path / "poses.npz"
    image.touch()
    trajectory.touch()
    stages: list[tuple[list[str], dict[str, Any]]] = []
    diffusion: dict[str, Any] = {}
    validated: list[Path] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stages.append((command, kwargs))
        if command[2] == _MODULES + "moge_v2_infer":
            depth_dir = Path(command[command.index("--output") + 1])
            np.savez(depth_dir / "depth_intrinsics.npz", depth=np.ones((2, 3)), intrinsics=np.eye(3))
        else:
            assert command[2] == _MODULES + "rendering_4D_control_maps"
            render_dir = Path(command[command.index("--output_dir") + 1])
            for name in (
                "background_RGB.mp4",
                "background_depth.mp4",
                "3D_gaussian_RGB.mp4",
                "3D_gaussian_depth.mp4",
                "merged_mask.mp4",
            ):
                (render_dir / name).touch()
        return subprocess.CompletedProcess(command, 0, "stage output", "")

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        diffusion.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    monkeypatch.setattr(model, "_validate_default_control_maps", validated.append)
    monkeypatch.setenv("PYTHONPATH", "/caller-path")
    result = model.predict(
        prompt="An orchard",
        image_path=image,
        trajectory_npz=None if default_trajectory else trajectory,
        output_path=tmp_path / "result.mp4",
        num_frames=5,
        width=768,
        height=448,
        seed=17,
        fps=12,
    )
    assert len(stages) == 2
    for command, kwargs in stages:
        assert command[:2] == ["/env/python", "-m"]
        assert kwargs["cwd"] == model.repo_root
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert kwargs["env"]["PYTHONPATH"] == "/caller-path"
        assert command[command.index("--device") + 1] == "cuda"
    render_command = stages[1][0]
    camera_path = Path(render_command[render_command.index("--trajectory_npz") + 1])
    if default_trajectory:
        assert np.load(camera_path)["extrinsics"].shape == (5, 4, 4)
        assert len(validated) == 1
    else:
        assert camera_path == trajectory
        assert validated == []
    ellipsoid = json.loads(Path(render_command[render_command.index("--ellipsoid_json") + 1]).read_text())
    assert ellipsoid["metadata"]["num_frames"] == 5
    assert all(frame["objects"] == [] for frame in ellipsoid["frames"])
    command = diffusion["command"]
    assert command[:8] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        "-m",
        _MODULES + "versecrafter_inference",
    ]
    assert command[command.index("--gpu_memory_mode") + 1] == memory_mode
    assert command[command.index("--sample_size") + 1] == "448,768"
    assert command[command.index("--seed") + 1] == "17"
    assert diffusion["env"] == {"CUSTOM": "preserved", "CUDA_VISIBLE_DEVICES": "3"}
    assert "python_paths" not in diffusion
    assert result["video"] == str(tmp_path / "result.mp4")


def test_versecrafter_stage_errors_keep_log_and_inherited_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = runtime.VerseCrafterRuntime(env={"CUDA_VISIBLE_DEVICES": "7"}, device="cuda:0")
    env, device = model._process_env()
    assert env == {"CUDA_VISIBLE_DEVICES": "7"}
    assert device == "cuda:0"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
        return subprocess.CompletedProcess(command, 9, "output", "depth failed")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="stage 'depth' exited with code 9"):
        model._run_stage("depth", ["/env/python", "-m", _MODULES + "moge_v2_infer"], log_dir=tmp_path, env=env)
    assert (tmp_path / "depth.log").read_text() == "output\ndepth failed"
