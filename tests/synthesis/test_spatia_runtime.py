# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Spatia package workers retain camera inputs, masks and cached-stage behavior."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.spatia import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit
_MODULE = "worldfoundry.synthesis.visual_generation.spatia.spatia_runtime."


def test_spatia_mapanything_worker_imports_canonical_components(monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = {
        "MapAnything": object(),
        "closed_form_pose_inverse": object(),
        "IMAGE_NORMALIZATION_DICT": {},
    }
    components = {
        "worldfoundry.base_models.three_dimensions.general_3d.mapanything.mapanything.models": "MapAnything",
        "worldfoundry.base_models.three_dimensions.general_3d.mapanything.mapanything.utils.geometry": "closed_form_pose_inverse",
        "worldfoundry.base_models.perception_core.general_perception.uniception.uniception.models.encoders.image_normalizations": "IMAGE_NORMALIZATION_DICT",
    }
    for name, symbol in components.items():
        module = ModuleType(name)
        setattr(module, symbol, bindings[symbol])
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "mapanything", None)
    monkeypatch.setitem(sys.modules, "uniception", None)
    monkeypatch.setitem(sys.modules, "open3d", ModuleType("open3d"))
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker = runtime.SpatiaRuntime.bundled_repo_root() / "utils" / "map_anything_inference.py"
    namespace = runpy.run_path(str(worker))
    for name, value in bindings.items():
        assert namespace[name] is value


def _runtime(tmp_path: Path) -> runtime.SpatiaRuntime:
    checkpoint, base, geometry = (tmp_path / name for name in ("checkpoint", "base", "map"))
    for path in (checkpoint, base, geometry):
        path.mkdir()
    for path in (
        checkpoint / "step-8500.safetensors",
        checkpoint / "lora_weights_10000.safetensors",
        geometry / "config.json",
        geometry / "model.safetensors",
    ):
        path.touch()
    return runtime.SpatiaRuntime(
        checkpoint_path=checkpoint,
        base_model_path=base,
        map_model_path=geometry,
        python_executable="/env/python",
        device="cuda:2",
        env={"CUSTOM": "preserved"},
    )


@pytest.mark.parametrize("input_kind", ["image", "video"])
@pytest.mark.parametrize("mask_kind", [None, "image", "array"])
def test_spatia_predict_preserves_camera_mask_and_package_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_kind: str, mask_kind: str | None
) -> None:
    model = _runtime(tmp_path)
    image, video, trajectory = (tmp_path / name for name in ("image.png", "source.mp4", "w2c.txt"))
    Image.new("RGB", (6, 4), (20, 30, 40)).save(image)
    video.touch()
    trajectory.touch()
    mask = None
    if mask_kind == "image":
        mask = tmp_path / "mask.png"
        Image.new("L", (6, 4), 255).save(mask)
    elif mask_kind == "array":
        mask = tmp_path / "mask.npy"
        np.save(mask, np.ones((1, 4, 6), dtype=np.bool_))
    captured: dict[str, Any] = {}
    first_frames: list[tuple[Path, Path]] = []

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        captured.update(command=command, **kwargs)
        return {"video": str(kwargs["output_path"])}

    def first_frame(path: Path, output: Path) -> Path:
        first_frames.append((path, output))
        return image

    monkeypatch.setattr(runtime, "execute_in_tree", execute)
    monkeypatch.setattr(runtime, "_first_frame", first_frame)
    output = tmp_path / "result.mp4"
    result = model.predict(
        prompt="An orchard",
        image_path=image if input_kind == "image" else None,
        video_path=video if input_kind == "video" else None,
        w2c_trajectory_file=trajectory,
        intrinsics=[[[416, 0, 416], [0, 240, 240], [0, 0, 1]]],
        mask_path=mask,
        output_path=output,
        width=832,
        height=480,
        num_frames=17,
        seed=19,
        force_rebuild_intermediate=mask is not None,
    )
    command = captured["command"]
    assert command[:3] == ["/env/python", "-m", _MODULE + "inference"]
    assert command[command.index("--img_path") + 1] == str(image)
    assert command[command.index("--camera_w2c_path") + 1] == str(trajectory)
    intrinsic_path = Path(command[command.index("--camera_intrinsics_path") + 1])
    assert intrinsic_path.read_text() == "[0.5 0.5 0.5 0.5]\n"
    assert command[command.index("--first_round_frames") + 1] == "17"
    assert command[command.index("--map_device") + 1] == "cuda:2"
    assert command[command.index("--render_device") + 1] == "cuda:2"
    assert command[command.index("--lora_path") + 1] == str(tmp_path / "checkpoint/lora_weights_10000.safetensors")
    assert ("--force_rebuild_intermediate" in command) == (mask is not None)
    assert ("--mask_path" in command) == (mask is not None)
    if mask is not None:
        assert command[command.index("--mask_path") + 1] == str(mask)
    assert len(first_frames) == (1 if input_kind == "video" else 0)
    assert captured["cwd"] == model.repo_root
    assert "python_paths" not in captured
    assert captured["env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUSTOM": "preserved",
    }
    assert result["video"] == str(output)


def test_spatia_mapanything_package_worker_preserves_masks_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime import inference

    calls: list[tuple[list[str], dict[str, Any]]] = []

    def save_video(frames: Any, path: str, **kwargs: Any) -> None:
        Path(path).touch()

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))
        output = Path(command[command.index("--output_path") + 1])
        (output / "points.ply").touch()
        (output / "intrinsics.txt").touch()

    monkeypatch.setattr(inference, "save_video", save_video)
    monkeypatch.setattr(inference, "run_command", run)
    args = SimpleNamespace(
        fps=12,
        force_rebuild_intermediate=False,
        map_python="/map/python",
        map_model_path="/map/weights",
        map_conf_percentile=12,
        map_voxel_size=0.003,
        map_device="cuda:1",
        mask_path=[tmp_path / "mask.npy"],
        per_frame_mask=True,
        verbose_subprocess=False,
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    intrinsic = tmp_path / "intrinsic.txt"
    for _ in range(2):
        result = inference.run_mapanything(args, 0, [], poses, intrinsic, tmp_path)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ["/map/python", "-m", _MODULE + "utils.map_anything_inference"]
    assert command[command.index("--intrinsic_path") + 1] == str(intrinsic)
    assert command[command.index("--mask_path") + 1] == str(args.mask_path[0])
    assert "--per_frame_mask" in command
    assert kwargs == {"cwd": inference.SCRIPT_DIR, "quiet": True}
    records = (result["round_dir"] / "map_hist_extrinsic.txt").read_text().splitlines()
    assert [json.loads(line) for line in records] == poses.tolist()
    args.force_rebuild_intermediate = True
    inference.run_mapanything(args, 0, [], poses, intrinsic, tmp_path)
    assert len(calls) == 2


def test_spatia_render_package_worker_preserves_geometry_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime import inference

    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))
        output = Path(command[command.index("--output_path") + 1])
        for name in ("render.mp4", "render-score.mp4"):
            (output / name).touch()

    monkeypatch.setattr(inference, "run_command", run)
    args = SimpleNamespace(
        force_rebuild_intermediate=False,
        render_python="/render/python",
        width=832,
        height=480,
        render_voxel_size=0.002,
        render_voxel_size_step=0.001,
        render_batchsize=7,
        render_device="cuda:2",
        verbose_subprocess=True,
    )
    poses, intrinsic = torch.eye(4)[None], torch.tensor([0.5, 0.5, 0.5, 0.5])
    for _ in range(2):
        result = inference.run_render(
            args, 2, tmp_path / "points.ply", tmp_path / "source.mp4", poses, intrinsic, tmp_path
        )
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ["/render/python", "-m", _MODULE + "utils.render_point"]
    assert command[command.index("--voxel_size") + 1] == "0.004"
    assert command[command.index("--point_coordinate") + 1] == "opencv"
    assert "--dummy_score" in command and "--score_render" in command and "--downsample" in command
    assert kwargs == {"cwd": inference.SCRIPT_DIR, "quiet": False}
    assert all(path.is_file() for path in result)
    assert Path(command[command.index("--normed_intrinsics") + 1]).read_text() == "[0.5 0.5 0.5 0.5]"
    args.force_rebuild_intermediate = True
    inference.run_render(args, 2, tmp_path / "points.ply", tmp_path / "source.mp4", poses, intrinsic, tmp_path)
    assert len(calls) == 2
