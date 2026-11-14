# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.unianimate_dit import worldfoundry_runner as runner
from worldfoundry.synthesis.visual_generation.unianimate_dit.unianimate_dit_runtime.examples.unianimate_wan import (
    inference_unianimate_wan_480p as worker,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_usp", [False, True])
def test_runner_launches_package_worker_with_workspace_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, use_usp: bool
) -> None:
    checkpoint, wan, demo = (tmp_path / name for name in ("checkpoint", "wan", "demo"))
    for path in (checkpoint, wan, demo):
        path.mkdir()
    reference = tmp_path / "reference.png"
    reference.touch()
    output = tmp_path / "result.mp4"
    monkeypatch.setenv("WORLDFOUNDRY_UNIANIMATE_DIT_DEMO_ROOT", str(demo))
    monkeypatch.setenv("PYTHONPATH", "/operator/package-path")
    launch_time = 1_700_000_000.0
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(time=lambda: launch_time))
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            checkpoint_dir=str(checkpoint),
            wan_root=str(wan),
            output_path=str(output),
            prompt="turn left",
            image_path=str(reference),
            max_frames=9,
            num_inference_steps=4,
            height=256,
            width=448,
            seed=17,
            use_usp=use_usp,
            nproc_per_node=4,
        ),
    )
    poses: list[tuple[Path, str]] = []
    monkeypatch.setattr(runner, "_ensure_demo_pose", lambda root, image: poses.append((root, image)))
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        generated = Path(kwargs["cwd"]) / "outputs" / "generated.mp4"
        generated.parent.mkdir()
        generated.write_bytes(b"composite")
        os.utime(generated, (launch_time + 1, launch_time + 1))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "_extract_generated_panel", lambda *_: False)
    runner.main()
    command, kwargs = calls[0]
    prefix = [sys.executable]
    if use_usp:
        prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4"]
    assert command == [
        *prefix,
        "-m",
        worker.__name__,
        "--prompt",
        "turn left",
        "--image-path",
        str(reference),
        "--max-frames",
        "9",
        "--num-inference-steps",
        "4",
        "--height",
        "256",
        "--width",
        "448",
        "--seed",
        "17",
    ]
    workspace = tmp_path / "unianimate_dit_workspace"
    assert kwargs["cwd"] == str(workspace)
    assert kwargs["env"]["PYTHONPATH"] == "/operator/package-path"
    assert kwargs["env"]["WORLDFOUNDRY_UNIANIMATE_USE_USP"] == str(int(use_usp))
    assert poses == [(workspace, str(reference))]
    assert (workspace / "Wan2.1-I2V-14B-720P").resolve() == wan
    assert (workspace / "checkpoints").resolve() == checkpoint
    assert (workspace / "data").resolve() == demo
    assert output.read_bytes() == b"composite"
    assert (tmp_path / "result_official_composite.mp4").read_bytes() == b"composite"
    assert not list(tmp_path.rglob("*.py"))


def test_newest_video_rejects_stale_artifacts_and_includes_launch_boundary(tmp_path: Path) -> None:
    cutoff = 1_700_000_000.0
    stale = tmp_path / "stale.mp4"
    stale.write_bytes(b"stale")
    os.utime(stale, (cutoff - 1, cutoff - 1))
    assert runner._newest_mp4(tmp_path, since=cutoff) is None
    boundary = tmp_path / "boundary.mp4"
    boundary.write_bytes(b"boundary")
    os.utime(boundary, (cutoff, cutoff))
    assert runner._newest_mp4(tmp_path, since=cutoff) == boundary
    newest = tmp_path / "newest.mp4"
    newest.write_bytes(b"newest")
    os.utime(newest, (cutoff + 1, cutoff + 1))
    assert runner._newest_mp4(tmp_path, since=cutoff) == newest


def test_pose_alignment_launches_package_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._ensure_demo_pose(tmp_path, "reference.png")
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "worldfoundry.synthesis.visual_generation.unianimate_dit.unianimate_dit_runtime.run_align_pose",
        "--ref_name",
        "reference.png",
        "--source_video_paths",
        runner.DEFAULT_SOURCE_VIDEO,
        "--saved_pose_dir",
        runner.DEFAULT_POSE_DIR,
    ]
    assert kwargs["cwd"] == str(tmp_path)
    assert "env" not in kwargs
    pose_dir = tmp_path / runner.DEFAULT_POSE_DIR
    (pose_dir / "pose.jpg").touch()
    (pose_dir / "0000.png").touch()
    runner._ensure_demo_pose(tmp_path, "reference.png")
    assert len(calls) == 1


@pytest.mark.parametrize("pose_count", [3, 12])
def test_worker_preserves_pose_tensors_and_random_draw_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pose_count: int
) -> None:
    import worldfoundry.core.io as media_io

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_UNIANIMATE_USE_USP", "0")
    pose_dir = tmp_path / runner.DEFAULT_POSE_DIR
    pose_dir.mkdir(parents=True)
    Image.new("RGB", (8, 12), color=(17, 31, 47)).save("reference.png")
    Image.new("RGB", (8, 12), color=(5, 7, 11)).save(pose_dir / "pose.jpg")
    for index in range(pose_count):
        Image.new("RGB", (8, 12), color=index).save(pose_dir / f"{index:04d}.png")
    calls: dict[str, Any] = {}
    events: list[str] = []

    def randint(low: int, high: int) -> int:
        assert (low, high) == (0, pose_count - 5 - 1)
        events.append("sample_start")
        return 2

    def pipe(**kwargs: Any) -> list[Image.Image]:
        events.append("inference")
        calls["inference"] = kwargs
        return [Image.new("RGB", (8, 12), color=73) for _ in range(kwargs["num_frames"])]

    def load(*args: Any, **kwargs: Any) -> Any:
        events.append("load")
        calls["load"] = (args, kwargs)
        return pipe

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.synthesis.visual_generation.unianimate_dit.native_pipeline",
        types.SimpleNamespace(load_unianimate_pipeline=load),
    )
    monkeypatch.setattr(worker.random, "randint", randint)
    monkeypatch.setattr(
        media_io, "save_video", lambda frames, path, **kwargs: calls.update(output=(frames, path, kwargs))
    )
    worker.run_inference(
        prompt="turn left", reference_image="reference.png", max_frames=5, steps=7, height=12, width=8, seed=19
    )
    assert events == (["load", "sample_start", "inference"] if pose_count == 12 else ["load", "inference"])
    assert calls["load"] == (
        ("./Wan2.1-I2V-14B-720P", "./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt"),
        {"device": "cuda", "use_usp": False},
    )
    inference = calls["inference"]
    assert inference["prompt"] == "turn left"
    assert (inference["num_frames"], inference["num_inference_steps"], inference["seed"]) == (5, 7, 19)
    assert inference["cfg_scale"] == 1.5
    assert inference["tiled"] is True
    assert inference["tea_cache_l1_thresh"] is None
    indices = [2, 3, 4, 5, 6] if pose_count == 12 else [0, 1, 2, 2, 2]
    expected = torch.zeros(3, 5, 12, 8)
    expected[0] = torch.tensor(indices)[:, None, None]
    torch.testing.assert_close(inference["dwpose_data"], expected, rtol=0, atol=0)
    reference_pose = torch.from_numpy(np.array(Image.open(pose_dir / "pose.jpg")))
    torch.testing.assert_close(inference["random_ref_dwpose"], reference_pose, rtol=0, atol=0)
    frames, path, options = calls["output"]
    assert len(frames) == 5 and all(frame.size == (24, 12) for frame in frames)
    assert path.endswith("video_480P_reference.png_WOMEN-Blouses_Shirts-id_00004955-01_4_full.mp4")
    assert options == {"fps": 15, "quality": 5}
