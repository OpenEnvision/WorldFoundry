# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.runtime import in_tree_cli
from worldfoundry.synthesis.visual_generation.moverse import moverse_synthesis as synthesis
from worldfoundry.synthesis.visual_generation.moverse.moverse_runtime.scripts import infer
from worldfoundry.synthesis.visual_generation.moverse.moverse_synthesis import MoVerseSynthesis

pytestmark = pytest.mark.unit
RUNTIME_MODULE = "worldfoundry.synthesis.visual_generation.moverse.moverse_runtime"


def test_moverse_public_model_path_selects_the_checkpoint(tmp_path: Path) -> None:
    from worldfoundry.pipelines.moverse.pipeline_moverse import MoVersePipeline

    checkpoint = tmp_path / "MoVerse"
    pipeline = MoVersePipeline.from_pretrained(model_path=str(checkpoint), device="cpu")
    assert pipeline.synthesis_model.options["moverse_path"] == str(checkpoint)
    assert "repo_root" not in pipeline.synthesis_model.options
    assert pipeline.synthesis_model.device == "cpu"


def test_moverse_plan_uses_the_current_shared_wan_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    is_file = Path.is_file

    def asset_is_file(path: Path) -> bool:
        assert not (path.suffix == ".py" and {"base_models", "moverse_runtime"}.intersection(path.parts)), (
            f"canonical source is not a staged asset: {path}"
        )
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", asset_is_file)
    result = MoVerseSynthesis(options={"run_dir": str(tmp_path)}).predict(plan_only=True)
    assert result["status"] == "prepared"
    assert result["metadata"]["missing_requirements"] == []
    assert result["command"][:3] == [sys.executable, "-m", f"{RUNTIME_MODULE}.scripts.infer"]
    assert result["command"][-2:] == ["--python_executable", sys.executable]


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_moverse_executes_package_command_without_import_path_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed: int | None
) -> None:
    assets = {}
    for role, required in synthesis.REQUIRED_ASSET_FILES.items():
        root = tmp_path / role
        for relative in required:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        assets[role] = str(root)
    image = tmp_path / "reference.png"
    image.touch()
    trajectory = tmp_path / "camera.txt"
    trajectory.write_text("0 0\n15 0\n")
    monkeypatch.setenv("PYTHONPATH", "/operator/packages")
    monkeypatch.delenv(in_tree_cli.IN_TREE_CLI_TIMEOUT_ENV, raising=False)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        output_dir = Path(command[command.index("--output_dir") + 1])
        (output_dir / "video.mp4").write_bytes(b"generated video")
        (output_dir / "video_side_by_side.mp4").write_bytes(b"comparison")
        (output_dir / "reference_pano.ply").write_text("gaussian scaffold")
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(in_tree_cli.subprocess, "run", run)
    model = MoVerseSynthesis(
        device="cuda:2", options={**assets, "run_dir": str(tmp_path / "run"), "python_executable": "/opt/python"}
    )
    result = model.predict(
        prompt="a garden",
        images=image,
        interactions={"trajectory": trajectory},
        fps=24,
        num_frames=17,
        back_prompt="a pond",
        output_path=tmp_path / "world.mp4",
        seed=seed,
    )
    assert result["status"] == "succeeded"
    command, options = calls[0]
    assert command[:3] == ["/opt/python", "-m", f"{RUNTIME_MODULE}.scripts.infer"]
    assert command[command.index("--num_frames") + 1] == "17"
    assert command[command.index("--fps") + 1] == "24"
    if seed is None:
        assert "--seed" not in command
    else:
        assert command[-2:] == ["--seed", str(seed)]
        command = command[:-2]
    assert command[-4:] == ["--back_prompt", "a pond", "--traj", str(trajectory)]
    assert options["cwd"] == synthesis.runtime_root()
    assert options["env"]["PYTHONPATH"] == "/operator/packages"
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert options["env"]["WAN_MODEL_PATH"] == assets["wan_path"]
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(result["artifact_path"]).read_bytes() == b"generated video"
    assert Path(result["comparison_video_path"]).read_bytes() == b"comparison"
    assert Path(result["world_artifact_path"]).read_text() == "gaussian scaffold"
    assert json.loads(Path(result["plan_path"]).read_text())["resolved_assets"] == assets


@pytest.mark.parametrize("seed", [None, 0, 42])
@pytest.mark.parametrize("custom_trajectory", [False, True])
def test_moverse_orchestrator_launches_all_four_package_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, custom_trajectory: bool, seed: int | None
) -> None:
    image = tmp_path / "reference.png"
    image.touch()
    output = tmp_path / "output"
    args = [
        "infer",
        "--image",
        str(image),
        "--prompt",
        "a garden",
        "--back_prompt",
        "a pond",
        "--output_dir",
        str(output),
        "--gimbal360_dir",
        "/weights/gimbal",
        "--flux_model_path",
        "/weights/flux",
        "--da3_model_dir",
        "/weights/da3",
        "--gaussian_checkpoint",
        "/weights/gaussian.pth",
        "--video_checkpoint",
        "/weights/video.pt",
        "--taehv_checkpoint",
        "/weights/taehv.pth",
        "--config_path",
        "/config/inference.yaml",
        "--default_config_path",
        "/config/default.yaml",
        "--num_frames",
        "17",
        "--fps",
        "24",
        "--python_executable",
        "/opt/python",
    ]
    if custom_trajectory:
        args.extend(["--traj", str(tmp_path / "custom.txt")])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    monkeypatch.setattr(sys, "argv", args)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(infer.subprocess, "run", run)
    infer.main()
    assert [command[:3] for command, _ in calls] == [
        ["/opt/python", "-m", f"{RUNTIME_MODULE}.{module}"]
        for module in (
            "pano_gen.inference",
            "gaussian_gen.infer_da3",
            "gaussian_gen.infer_pano",
            "scripts.run_pipeline",
        )
    ]
    assert all(options == {"cwd": infer.RUNTIME_ROOT, "check": True} for _, options in calls)
    panorama_command = calls[0][0]
    command = calls[-1][0]
    if seed is None:
        assert all("--seed" not in stage_command for stage_command, _ in calls)
    else:
        assert panorama_command[-2:] == command[-2:] == ["--seed", str(seed)]
        panorama_command = panorama_command[:-2]
        command = command[:-2]
    assert panorama_command[-2:] == ["--back_prompt", "a pond"]
    assert command[command.index("--scene") + 1] == str(output / "reference_pano.ply")
    assert command[command.index("--prompt") + 1] == "A video of a garden"
    assert command[command.index("--taehv_checkpoint") + 1] == "/weights/taehv.pth"
    assert command[-4:] == ["--num_frames", "17", "--fps", "24"]
    trajectory = tmp_path / "custom.txt" if custom_trajectory else output / "trajectory.txt"
    assert command[command.index("--traj") + 1] == str(trajectory)
    if not custom_trajectory:
        assert trajectory.read_text() == "0 0\n0 0\n0 0\n"


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_moverse_optional_seed_reaches_headless_worker_rngs(seed: int | None) -> None:
    source = synthesis.runtime_root() / "scripts/run_pipeline.py"
    tree = ast.parse(source.read_text())
    seed_argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_argument"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "--seed"
    )
    seed_guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.seed is not None"
    )
    code = f"""\
import random
from argparse import ArgumentParser
import numpy as np
import torch
p = ArgumentParser()
{ast.unparse(seed_argument)}
args = p.parse_args()
def draw() -> list[float]:
    return [random.random(), float(np.random.random()), torch.randn(()).item()]
expected_seed = 91 if args.seed is None else args.seed
random.seed(expected_seed)
np.random.seed(expected_seed)
torch.manual_seed(expected_seed)
expected = draw()
random.seed(91)
np.random.seed(91)
torch.manual_seed(91)
flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
{ast.unparse(seed_guard)}
assert draw() == expected
if args.seed is None:
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
"""
    flags = [] if seed is None else ["--seed", str(seed)]
    result = subprocess.run([sys.executable, "-c", code, *flags], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
