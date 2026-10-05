from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "imageio" dependency at import time; skip when it is unavailable.
pytest.importorskip("imageio")

import subprocess
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.models.catalog.registry import discover_model_registry
from worldfoundry.operators.ac3d_operator import AC3DOperator
from worldfoundry.pipelines.ac3d.pipeline_ac3d import AC3DPipeline
from worldfoundry.synthesis.visual_generation.ac3d import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.ac3d.runtime import AC3DRuntime


def test_ac3d_operator_parses_camera_range() -> None:
    operator = AC3DOperator()
    operator.get_interaction("2:5")

    result = operator.process_interaction()

    assert result["actions"] == [2, 3, 4]
    assert result["start_camera_idx"] == 2
    assert result["end_camera_idx"] == 5


def test_ac3d_pipeline_forwards_runtime_options() -> None:
    pipe = AC3DPipeline.from_pretrained(
        model_path={
            "runtime_root": "/tmp/ac3d",
            "base_model_path": "/tmp/cogvideo",
            "controlnet_model_path": "/tmp/ac3d.pt",
            "variant": "5b",
        },
        device="cuda:3",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.runtime_root == "/tmp/ac3d"
    assert runtime.base_model_path == "/tmp/cogvideo"
    assert runtime.controlnet_model_path == "/tmp/ac3d.pt"
    assert runtime.variant == "5b"
    assert runtime.device == "cuda:3"
    assert runtime.defaults["controlnet_transformer_attention_head_dim"] == 64


def test_ac3d_default_runtime_root_is_in_tree(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_AC3D_REPO_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_ROOT", str(tmp_path / "github_repos"))

    runtime_text = Path(runtime_module.__file__).read_text(encoding="utf-8")
    assert "official_runtime_repo_path" not in runtime_text
    assert "github_repos" not in runtime_text
    assert "WORLDFOUNDRY_MODEL_SOURCE_DIR" not in runtime_text

    runtime = AC3DRuntime(base_model_path="/tmp/base", controlnet_model_path="/tmp/controlnet.pt", device="cpu")
    assert Path(runtime.runtime_root) == (
        Path(runtime_module.__file__).resolve().parent / "ac3d_runtime"
    )

    override = tmp_path / "ac3d-override"
    monkeypatch.setenv("WORLDFOUNDRY_AC3D_REPO_ROOT", str(override))
    runtime = AC3DRuntime(base_model_path="/tmp/base", controlnet_model_path="/tmp/controlnet.pt", device="cpu")
    assert Path(runtime.runtime_root) == override


def test_ac3d_studio_dataset_default_ignores_official_source_repo(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.studio.inference import catalog as studio_catalog

    text = Path(studio_catalog.__file__).read_text(encoding="utf-8")
    section = text[
        text.index("def _ac3d_default_video_root_dir") : text.index("def _ac3d_default_call_kwargs")
    ]
    assert "official_runtime_repo_path" not in section

    dataset_root = tmp_path / "RealEstate10K"
    (dataset_root / "annotations").mkdir(parents=True)
    (dataset_root / "pose_files").mkdir()
    (dataset_root / "video_clips").mkdir()
    (dataset_root / "annotations" / "test.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_AC3D_VIDEO_ROOT_DIR", str(dataset_root))

    assert studio_catalog._ac3d_default_video_root_dir() == str(dataset_root)


def test_ac3d_runtime_builds_official_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "repo"
    video_root = tmp_path / "re10k"
    controlnet = tmp_path / "checkpoint.pt"
    base = tmp_path / "CogVideoX-2b"
    runtime_root.mkdir()
    video_root.mkdir()
    base.mkdir()
    controlnet.write_bytes(b"stub")

    captured = {}

    def fake_run(command, check, cwd, env, stdout, stderr):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        output_dir = Path(command[command.index("--output_path") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "00003_out.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runtime_module,
        "_load_video_frames",
        lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )
    model = AC3DRuntime(
        runtime_root=runtime_root,
        base_model_path=base,
        controlnet_model_path=controlnet,
        python_executable="/tmp/python",
    )

    result = model.predict(
        prompt="demo",
        video_root_dir=str(video_root),
        output_path=tmp_path / "result.mp4",
        start_camera_idx=3,
        end_camera_idx=4,
        num_inference_steps=1,
        num_frames=2,
        height=64,
        width=64,
        show_progress=False,
    )

    command = captured["command"]
    assert command[:3] == [
        "/tmp/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.ac3d.runtime",
    ]
    assert "--run-official" in command
    assert command[command.index("--runtime_root") + 1] == str(runtime_root)
    assert command[command.index("--base_model_path") + 1] == str(base)
    assert command[command.index("--controlnet_model_path") + 1] == str(controlnet)
    assert command[command.index("--start_camera_idx") + 1] == "3"
    assert command[command.index("--controlnet_transformer_attention_head_dim") + 1] == "32"
    assert captured["cwd"] == str(runtime_root)
    assert str(runtime_root) in captured["env"]["PYTHONPATH"]
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_model_registry_contains_ac3d() -> None:
    registry = discover_model_registry()
    model = registry.get("ac3d")

    assert model.family == "camera-control-video"
    assert model.has_loader is True
    assert model.has_infer is True
    assert model.pipeline_target == "worldfoundry.pipelines.ac3d.pipeline_ac3d:AC3DPipeline"
