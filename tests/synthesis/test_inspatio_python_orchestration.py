from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime/run_inference_pipeline.py"
spec = importlib.util.spec_from_file_location("inspatio_python_orchestration", SOURCE)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_args(tmp_path, *flags):
    trajectory = tmp_path / "camera '# path.txt"
    trajectory.touch()
    return pipeline.parser().parse_args([
        "--input_dir", str(tmp_path), "--traj_txt_path", str(trajectory), *flags,
    ])


def test_cached_depth_still_renders_new_trajectory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    args = make_args(tmp_path, "--skip_step1", "--skip_step2", "--skip_step3", "--rotation_only")
    pipeline.run_pipeline(args)
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == sys.executable
    assert Path(command[1]).name == "run_render_parallel.py"
    assert command[command.index("--traj_txt_path") + 1] == args.traj_txt_path
    assert "--rotation_only" in command
    assert options["check"] is True


def test_full_pipeline_preserves_gpu_flags_and_yaml_paths(monkeypatch, tmp_path):
    (tmp_path / "models_t5_umt5-xxl-enc-bf16.safetensors").touch()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"camera": {"traj_txt_path": "old", "relative_to_source": False,
                                              "rotation_only": False, "adaptive_frame": True,
                                              "freeze_repeat": 0, "freeze_frame": 5}}))
    args = make_args(tmp_path, "--config_path", str(config), "--wan_model_path", str(tmp_path),
                     "--step1_gpus", "2,3", "--step2_gpus", "4", "--step3_gpus", "5,6",
                     "--step3_nproc", "2", "--relative_to_source", "--disable_adaptive_frame",
                     "--freeze_repeat", "3", "--freeze_frame", "0", "--use_tae", "--compile_dit")
    calls = []
    saved_config = {}

    def run(argv, **options):
        calls.append((argv, options))
        if "torch.distributed.run" in argv:
            path = Path(argv[argv.index("--config_path") + 1])
            saved_config.update(path=path, value=yaml.safe_load(path.read_text()))

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.run_pipeline(args)
    assert len(calls) == 6  # two caption workers, merge, depth, render, inference
    assert all(command[0] == sys.executable for command, _ in calls)
    assert {options["env"]["CUDA_VISIBLE_DEVICES"] for _, options in calls[:2]} == {"2", "3"}
    depth = next(command for command, _ in calls if Path(command[1]).name == "run_da3_parallel.py")
    assert json.loads(depth[depth.index("--da3_config") + 1])["fix_resize_width"] == 832
    infer, options = calls[-1]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "5,6"
    assert "--nproc_per_node=2" in infer
    assert "--use_tae" in infer and "--compile_dit" in infer
    camera = saved_config["value"]["camera"]
    assert camera["traj_txt_path"] == args.traj_txt_path
    assert camera["relative_to_source"] is True and camera["adaptive_frame"] is False
    assert camera["freeze_repeat"] == 3 and camera["freeze_frame"] == 0
    assert not saved_config["path"].exists()
    assert yaml.safe_load(config.read_text())["camera"]["traj_txt_path"] == "old"


def test_failed_caption_does_not_start_geometry(monkeypatch, tmp_path):
    calls = []

    def fail(argv, **options):
        calls.append(argv)
        raise subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(make_args(tmp_path))
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "gen_json.py"
