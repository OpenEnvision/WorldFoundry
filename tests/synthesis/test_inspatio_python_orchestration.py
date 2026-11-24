from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime import (
    run_inference_pipeline as pipeline,
)

pytestmark = pytest.mark.unit
RUNTIME_MODULE = "worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime"


def make_args(tmp_path, *flags):
    trajectory = tmp_path / "camera '# path.txt"
    trajectory.touch()
    return pipeline.parser().parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--traj_txt_path",
            str(trajectory),
            *flags,
        ]
    )


def test_cached_depth_still_renders_new_trajectory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    args = make_args(tmp_path, "--skip_step1", "--skip_step2", "--skip_step3", "--rotation_only")
    pipeline.run_pipeline(args)
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == sys.executable
    assert command[:3] == [sys.executable, "-m", f"{RUNTIME_MODULE}.scripts.run_render_parallel"]
    assert command[command.index("--traj_txt_path") + 1] == args.traj_txt_path
    assert "--rotation_only" in command
    assert options["check"] is True


def test_full_pipeline_preserves_gpu_flags_and_yaml_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/operator/packages")
    (tmp_path / "models_t5_umt5-xxl-enc-bf16.safetensors").touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "traj_txt_path": "old",
                    "relative_to_source": False,
                    "rotation_only": False,
                    "adaptive_frame": True,
                    "freeze_repeat": 0,
                    "freeze_frame": 5,
                }
            }
        )
    )
    args = make_args(
        tmp_path,
        "--config_path",
        str(config),
        "--wan_model_path",
        str(tmp_path),
        "--step1_gpus",
        "2,3",
        "--step2_gpus",
        "4",
        "--step3_gpus",
        "5,6",
        "--step3_nproc",
        "2",
        "--master_port",
        "29587",
        "--relative_to_source",
        "--disable_adaptive_frame",
        "--freeze_repeat",
        "3",
        "--freeze_frame",
        "0",
        "--use_tae",
        "--compile_dit",
        "--seed",
        "1730",
        "--deterministic",
    )
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
    assert all(options["env"]["WORLDFOUNDRY_INSPATIO_WORLD_SEED"] == "1730" for _, options in calls)
    assert all(options["env"]["PYTHONHASHSEED"] == "1730" for _, options in calls)
    assert all(options["env"]["WORLDFOUNDRY_DETERMINISTIC"] == "1" for _, options in calls)
    assert all(options["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" for _, options in calls)
    assert {options["env"]["CUDA_VISIBLE_DEVICES"] for _, options in calls[:2]} == {"2", "3"}
    assert all(options["env"]["PYTHONPATH"] == "/operator/packages" for _, options in calls)
    assert all(options["cwd"] == pipeline.ROOT for _, options in calls)
    assert [command[2] for command, _ in calls[:-1]] == [
        f"{RUNTIME_MODULE}.scripts.{name}"
        for name in ("gen_json", "gen_json", "merge_partial_jsons", "run_da3_parallel", "run_render_parallel")
    ]
    depth = next(command for command, _ in calls if command[2] == f"{RUNTIME_MODULE}.scripts.run_da3_parallel")
    assert json.loads(depth[depth.index("--da3_config") + 1])["fix_resize_width"] == 832
    infer, options = calls[-1]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "5,6"
    assert infer[:9] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_port",
        "29587",
        "--nproc_per_node",
        "2",
        "-m",
        f"{RUNTIME_MODULE}.inference_causal",
    ]
    assert "--use_tae" in infer and "--compile_dit" in infer
    assert infer[infer.index("--seed") + 1] == "1730"
    assert "--deterministic" in infer
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
    assert calls[0][:3] == [sys.executable, "-m", f"{RUNTIME_MODULE}.scripts.gen_json"]


@pytest.mark.parametrize(
    "flags, inherited, expected",
    [
        ((), None, None),
        (("--seed", "42"), None, "42"),
        (("--deterministic",), None, "0"),
        ((), "7", "7"),
    ],
)
def test_video_stage_receives_effective_seed(monkeypatch, tmp_path, flags, inherited, expected):
    monkeypatch.delenv("WORLDFOUNDRY_DETERMINISTIC", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_INSPATIO_WORLD_SEED", raising=False)
    if inherited is not None:
        monkeypatch.setenv("WORLDFOUNDRY_INSPATIO_WORLD_SEED", inherited)
    (tmp_path / "models_t5_umt5-xxl-enc-bf16.safetensors").touch()
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = make_args(
        tmp_path,
        "--config_path",
        str(config),
        "--wan_model_path",
        str(tmp_path),
        "--skip_step1",
        "--skip_step2",
        *flags,
    )
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    pipeline.run_pipeline(args)
    infer, options = calls[-1]
    if expected is None:
        assert "--seed" not in infer
        assert "WORLDFOUNDRY_INSPATIO_WORLD_SEED" not in options["env"]
    else:
        assert infer[infer.index("--seed") + 1] == expected
        assert options["env"]["WORLDFOUNDRY_INSPATIO_WORLD_SEED"] == expected


def test_t5_conversion_uses_package_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights = tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"
    weights.touch()
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = make_args(
        tmp_path, "--config_path", str(config), "--wan_model_path", str(tmp_path), "--skip_step1", "--skip_step2"
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append(argv))
    pipeline.run_pipeline(args)
    assert calls[1] == [
        sys.executable,
        "-m",
        f"{RUNTIME_MODULE}.utils.convert_pth_to_safetensors",
        "--input",
        str(weights),
        "--output",
        str(weights.with_suffix(".safetensors")),
    ]
